#!/usr/bin/env python3
"""
survivor_detector.py v2 - NIDAR AirMouse vision node (Jetson Orin Nano + OAK-D Lite)

OAK-D RGB + aligned depth -> YOLO (one class: human OR dummy) -> tracker ->
confirmation (N frames AND mean confidence) -> map position -> de-duplicate by
POSITION -> one JSON "detection signal" per NEW survivor, with grid cell.

Pose input (from the navigation/SLAM node), UDP JSON to --pose-port on localhost:
    {"x": 1.25, "y": 3.10, "yaw": 1.57}     metres, radians, map frame, yaw CCW
Without pose/depth the node still counts, per tracker ID only (a warning is printed).

Reliability: per-frame errors are caught and counted, the camera is re-opened if it
drops, and every new survivor is written to --state-file (use --resume after a restart).

  python3 survivor_detector.py --source oak --model best.engine --classes 0 --headless
  python3 survivor_detector.py --source webcam --model yolo11n.pt          (desk test)
  python3 survivor_detector.py --selftest                                  (logic test)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

import numpy as np


# --------------------------------------------------------------------------- geometry
def pixel_to_body(u, v, z_m, K, pitch_deg):
    """Pixel + depth along optical axis -> (forward, left) metres in the drone body frame.
    Camera axes: X right, Y down, Z forward. pitch_deg > 0 means camera tilted DOWN."""
    fx, fy, cx, cy = K[0][0], K[1][1], K[0][2], K[1][2]
    X = (u - cx) * z_m / fx
    Y = (v - cy) * z_m / fy
    p = math.radians(pitch_deg)
    forward = -Y * math.sin(p) + z_m * math.cos(p)
    left = -X
    return forward, left


def body_to_world(forward, left, pose):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + forward * c - left * s, y + forward * s + left * c


def track_ready(hits, confs, confirm_frames, confirm_conf):
    """A track becomes a survivor only if seen long enough AND confidently."""
    if hits < confirm_frames:
        return False
    recent = confs[-confirm_frames:]
    return len(recent) >= confirm_frames and sum(recent) / len(recent) >= confirm_conf


# --------------------------------------------------------------------------- registry
class SurvivorRegistry:
    """Unique survivors, de-duplicated by map position."""

    def __init__(self, merge_radius_m: float = 0.6, cell_size_m: float = 1.0):
        self.radius = merge_radius_m
        self.cell = cell_size_m
        self.survivors: list[dict] = []

    def grid_cell(self, xy):
        if xy is None:
            return None
        return (int(math.floor(xy[0] / self.cell)), int(math.floor(xy[1] / self.cell)))

    def confirm(self, xy):
        """A track was confirmed at map position xy (or None). Returns (survivor, is_new)."""
        if xy is not None:
            best, best_d = None, 1e9
            for s in self.survivors:
                if s["xy"] is None:
                    continue
                d = math.hypot(s["xy"][0] - xy[0], s["xy"][1] - xy[1])
                if d < best_d:
                    best, best_d = s, d
            if best is not None and best_d <= self.radius:
                n = best["tracks"]
                best["xy"] = ((best["xy"][0] * n + xy[0]) / (n + 1),
                              (best["xy"][1] * n + xy[1]) / (n + 1))
                best["tracks"] = n + 1
                best["cell"] = self.grid_cell(best["xy"])
                return best, False
        s = {"id": len(self.survivors) + 1, "xy": xy, "tracks": 1, "cell": self.grid_cell(xy)}
        self.survivors.append(s)
        return s, True

    def save(self, path):
        data = {"survivors": [{"id": s["id"], "tracks": s["tracks"],
                               "xy": None if s["xy"] is None else list(s["xy"])}
                              for s in self.survivors]}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)  # atomic: never leaves a half-written file

    def load(self, path):
        with open(path) as f:
            data = json.load(f)
        self.survivors = []
        for s in data["survivors"]:
            xy = None if s["xy"] is None else (float(s["xy"][0]), float(s["xy"][1]))
            self.survivors.append({"id": int(s["id"]), "tracks": int(s["tracks"]),
                                   "xy": xy, "cell": self.grid_cell(xy)})


# --------------------------------------------------------------------------- pose (UDP)
class PoseProvider:
    def __init__(self, port: int | None):
        self.pose = None
        self.sock = None
        if port:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.bind(("127.0.0.1", port))
            self.sock.setblocking(False)

    def latest(self):
        if self.sock is not None:
            while True:  # drain, keep newest
                try:
                    data, _ = self.sock.recvfrom(1024)
                except (BlockingIOError, OSError):
                    break
                try:
                    d = json.loads(data.decode())
                    self.pose = (float(d["x"]), float(d["y"]), float(d["yaw"]))
                except (ValueError, KeyError, TypeError):
                    pass
        return self.pose


# --------------------------------------------------------------------------- sources
class WebcamSource:
    def __init__(self, index=0):
        import cv2
        self.cap = cv2.VideoCapture(index)  # no CAP_DSHOW: must work on Linux/Jetson
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera {index}")
        self.K = None

    def frames(self):
        while True:
            ok, frame = self.cap.read()
            if not ok:
                return
            yield frame, None


class OakSource:
    """OAK-D Lite via DepthAI v2 API (pip install "depthai<3"). NOT tested on hardware."""

    def __init__(self, w=640, h=360, fps=15, with_depth=True):
        import depthai as dai
        self.dai, self.w, self.h = dai, w, h
        p = dai.Pipeline()
        cam = p.create(dai.node.ColorCamera)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setPreviewSize(w, h)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(fps)
        x_rgb = p.create(dai.node.XLinkOut)
        x_rgb.setStreamName("rgb")
        cam.preview.link(x_rgb.input)
        if with_depth:
            left = p.create(dai.node.MonoCamera)
            right = p.create(dai.node.MonoCamera)
            for m, sock in ((left, dai.CameraBoardSocket.LEFT),
                            (right, dai.CameraBoardSocket.RIGHT)):
                m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
                m.setBoardSocket(sock)
                m.setFps(fps)
            stereo = p.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            stereo.setLeftRightCheck(True)
            stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
            stereo.setOutputSize(w, h)
            left.out.link(stereo.left)
            right.out.link(stereo.right)
            x_dep = p.create(dai.node.XLinkOut)
            x_dep.setStreamName("depth")
            stereo.depth.link(x_dep.input)
        self.dev = dai.Device(p)
        self.q_rgb = self.dev.getOutputQueue("rgb", maxSize=1, blocking=False)
        self.q_dep = (self.dev.getOutputQueue("depth", maxSize=1, blocking=False)
                      if with_depth else None)
        self.K = self.dev.readCalibration().getCameraIntrinsics(
            dai.CameraBoardSocket.RGB, w, h)

    def frames(self):
        depth = None
        while True:
            rgb = self.q_rgb.get()
            if self.q_dep is not None:
                d = self.q_dep.tryGet()
                if d is not None:
                    depth = d.getFrame()  # uint16, millimetres
            yield rgb.getCvFrame(), depth


def box_depth_m(depth_mm, box):
    """Median valid depth (metres) in the central half of the box."""
    if depth_mm is None:
        return None
    h, w = depth_mm.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    hw, hh = (x2 - x1) / 4, (y2 - y1) / 4
    xa, xb = int(max(0, cx - hw)), int(min(w, cx + hw))
    ya, yb = int(max(0, cy - hh)), int(min(h, cy + hh))
    if xb <= xa or yb <= ya:
        return None
    roi = depth_mm[ya:yb, xa:xb]
    valid = roi[(roi > 300) & (roi < 12000)]
    if valid.size < 10:
        return None
    return float(np.median(valid)) / 1000.0


# --------------------------------------------------------------------------- node
class SurvivorNode:
    """All per-frame logic; no camera or display code, so it can be unit-tested."""

    def __init__(self, args, model, K, pose_src, registry, emit):
        self.args, self.model, self.K = args, model, K
        self.pose_src, self.reg, self.emit = pose_src, registry, emit
        self.classes = [int(c) for c in args.classes.split(",")]
        self.tracks: dict[int, dict] = {}
        self.n = 0
        self.warned = False
        self.warned_cap = False

    def step(self, frame, depth):
        """Process one frame. Returns [(x1, y1, x2, y2, label, confirmed)]."""
        a = self.args
        self.n += 1
        H, W = frame.shape[:2]
        res = self.model.track(frame, persist=True, tracker=a.tracker, classes=self.classes,
                               conf=a.conf, imgsz=a.imgsz, device=a.device, verbose=False)[0]
        pose = self.pose_src.latest() if self.pose_src else None
        out = []
        if res.boxes is None or res.boxes.id is None:
            return out
        ids = res.boxes.id.int().cpu().tolist()
        boxes = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().tolist()
        for tid, box, cf in zip(ids, boxes, confs):
            x1, y1, x2, y2 = [float(v) for v in box]
            if (x2 - x1) * (y2 - y1) < a.min_area * W * H:
                continue  # too small / far; deliberately NO aspect-ratio filter
            st = self.tracks.setdefault(tid, {"hits": 0, "last": -999, "pts": [],
                                              "confs": [], "sid": None})
            if self.n - st["last"] <= a.max_gap:
                st["hits"] += 1
            else:
                st["hits"], st["confs"] = 1, []
            st["last"] = self.n
            st["confs"] = (st["confs"] + [cf])[-30:]
            z = box_depth_m(depth, (x1, y1, x2, y2))
            if z is not None and pose is not None and self.K is not None:
                f, l = pixel_to_body((x1 + x2) / 2, (y1 + y2) / 2, z, self.K, a.cam_pitch_deg)
                st["pts"] = (st["pts"] + [body_to_world(f, l, pose)])[-30:]
            if st["sid"] is None and track_ready(st["hits"], st["confs"],
                                                 a.confirm_frames, a.confirm_conf):
                xy = None
                if len(st["pts"]) >= 3:
                    arr = np.array(st["pts"])
                    xy = (float(np.median(arr[:, 0])), float(np.median(arr[:, 1])))
                elif not self.warned:
                    print("WARNING: no depth/pose -> counting per tracker ID only")
                    self.warned = True
                s, is_new = self.reg.confirm(xy)
                st["sid"] = s["id"]
                if is_new:
                    mean_c = sum(st["confs"][-a.confirm_frames:]) / a.confirm_frames
                    self.emit({"event": "survivor", "id": s["id"], "grid": s["cell"],
                               "xy_m": None if s["xy"] is None else
                               [round(s["xy"][0], 2), round(s["xy"][1], 2)],
                               "conf": round(mean_c, 2), "total": len(self.reg.survivors),
                               "t": round(time.time(), 2)})
                    if len(self.reg.survivors) > a.max_survivors and not self.warned_cap:
                        print(f"WARNING: more than {a.max_survivors} survivors counted - "
                              "check for false positives / duplicates")
                        self.warned_cap = True
            label = f"S{st['sid']}" if st["sid"] else f"? {st['hits']}"
            out.append((x1, y1, x2, y2, label, st["sid"] is not None))
        return out


# --------------------------------------------------------------------------- selftest
class _T:  # minimal tensor stand-in for tests
    def __init__(self, a):
        self.a = np.array(a)

    def int(self):
        return _T(self.a.astype(int))

    def cpu(self):
        return self

    def tolist(self):
        return self.a.tolist()

    def numpy(self):
        return self.a


class _FakeModel:
    def __init__(self, script):
        self.script = list(script)

    def track(self, frame, **kw):
        dets = self.script.pop(0)  # list of (id, x1, y1, x2, y2, conf)
        if not dets:
            boxes = type("B", (), {"id": None})()
        else:
            boxes = type("B", (), {
                "id": _T([d[0] for d in dets]),
                "xyxy": _T([d[1:5] for d in dets]),
                "conf": _T([d[5] for d in dets])})()
        return [type("R", (), {"boxes": boxes})()]


def selftest():
    K = [[500, 0, 320], [0, 500, 180], [0, 0, 1]]
    f, l = pixel_to_body(320, 180, 2.0, K, 0)
    assert abs(f - 2.0) < 1e-9 and abs(l) < 1e-9
    assert all(abs(a - b) < 1e-9 for a, b in zip(body_to_world(f, l, (1, 1, math.pi / 2)), (1, 3)))
    f, l = pixel_to_body(320, 180, 2.0, K, 90)
    assert abs(f) < 1e-9
    f, l = pixel_to_body(320, 180 + 100, 2.0, K, 90)
    assert f < 0
    f, l = pixel_to_body(420, 180, 2.0, K, 0)
    assert l < 0

    assert track_ready(6, [0.8] * 6, 6, 0.5)
    assert not track_ready(5, [0.9] * 5, 6, 0.5)          # too few frames
    assert not track_ready(8, [0.3] * 8, 6, 0.5)          # confident enough? no
    assert not track_ready(8, [0.9, 0.9] + [0.2] * 6, 6, 0.5)

    reg = SurvivorRegistry(0.6, 1.0)
    s1, n1 = reg.confirm((3.0, 3.0))
    s2, n2 = reg.confirm((3.2, 3.1))
    s3, n3 = reg.confirm((6.0, 6.0))
    s4, n4 = reg.confirm(None)
    assert (n1, n2, n3, n4) == (True, False, True, True) and len(reg.survivors) == 3
    assert s1["cell"] == (3, 3)
    reg.save("/tmp/_state.json")
    reg2 = SurvivorRegistry(0.6, 1.0)
    reg2.load("/tmp/_state.json")
    assert len(reg2.survivors) == 3 and reg2.survivors[0]["cell"] == (3, 3)
    assert reg2.survivors[2]["xy"] is None

    pp = PoseProvider(45123)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    tx.sendto(b"garbage", ("127.0.0.1", 45123))
    tx.sendto(json.dumps({"x": 1, "y": 2, "yaw": 0.5}).encode(), ("127.0.0.1", 45123))
    time.sleep(0.05)
    assert pp.latest() == (1.0, 2.0, 0.5)

    d = np.zeros((360, 640), np.uint16)
    d[100:200, 100:200] = 2500
    assert abs(box_depth_m(d, (100, 100, 200, 200)) - 2.5) < 1e-6
    assert box_depth_m(d, (400, 100, 500, 200)) is None

    # ---- end-to-end logic test with a scripted fake model
    args = parse_args([])
    person = lambda tid, depth_u, c: (tid, depth_u - 50, 80, depth_u + 50, 280, c)
    script = []
    script += [[person(1, 320, 0.8)]] * 8      # person A, track 1  -> (3,0)
    script += [[]] * 3                          # lost
    script += [[person(2, 320, 0.8)]] * 8      # same place, tracker ID switched -> merged
    script += [[person(3, 320, 0.8)]] * 8      # (will be at depth 6 m below)
    script += [[person(4, 320, 0.3)]] * 10     # flickery low-confidence junk
    frame = np.zeros((360, 640, 3), np.uint8)
    depth3 = np.full((360, 640), 3000, np.uint16)
    depth6 = np.full((360, 640), 6000, np.uint16)
    signals = []
    node = SurvivorNode(args, _FakeModel(script), K, type("P", (), {"latest": lambda s: (0, 0, 0)})(),
                        SurvivorRegistry(0.6, 1.0), signals.append)
    for i in range(len(script)):
        dep = depth6 if 19 <= i < 27 else depth3
        node.step(frame, dep)
    assert len(signals) == 2, signals
    assert signals[0]["grid"] == [3, 0] or tuple(signals[0]["grid"]) == (3, 0), signals[0]
    assert tuple(signals[1]["grid"]) == (6, 0), signals[1]
    assert len(node.reg.survivors) == 2
    print("selftest OK")


# --------------------------------------------------------------------------- main
def parse_args(argv=None):
    a = argparse.ArgumentParser()
    a.add_argument("--source", choices=["oak", "webcam"], default="oak")
    a.add_argument("--camera", type=int, default=0)
    a.add_argument("--oak-w", type=int, default=640)
    a.add_argument("--oak-h", type=int, default=360)
    a.add_argument("--model", default="yolo11n.pt")
    a.add_argument("--classes", default="0", help="comma list of class ids")
    a.add_argument("--device", default="0")
    a.add_argument("--conf", type=float, default=0.30, help="detector threshold per frame")
    a.add_argument("--confirm-conf", type=float, default=0.50,
                   help="mean confidence over the confirmation window")
    a.add_argument("--imgsz", type=int, default=640)
    a.add_argument("--tracker", default="bytetrack.yaml")
    a.add_argument("--confirm-frames", type=int, default=6)
    a.add_argument("--max-gap", type=int, default=5)
    a.add_argument("--min-area", type=float, default=0.004, help="min box area / frame area")
    a.add_argument("--merge-radius", type=float, default=0.6, help="metres")
    a.add_argument("--cell-size", type=float, default=1.0, help="grid cell size, metres")
    a.add_argument("--max-survivors", type=int, default=6)
    a.add_argument("--cam-pitch-deg", type=float, default=0.0, help=">0 = tilted down")
    a.add_argument("--pose-port", type=int, default=None)
    a.add_argument("--signal-udp", default=None, help="host:port to send JSON signals")
    a.add_argument("--state-file", default="survivors_state.json")
    a.add_argument("--resume", action="store_true", help="load --state-file at start")
    a.add_argument("--max-errors", type=int, default=50, help="consecutive frame errors")
    a.add_argument("--max-restarts", type=int, default=5, help="camera re-open attempts")
    a.add_argument("--headless", action="store_true")
    a.add_argument("--selftest", action="store_true")
    return a.parse_args(argv)


def open_source(args):
    if args.source == "oak":
        return OakSource(args.oak_w, args.oak_h)
    return WebcamSource(args.camera)


def main():
    args = parse_args()
    if args.selftest:
        selftest()
        return
    import cv2
    from ultralytics import YOLO

    model = YOLO(args.model)
    reg = SurvivorRegistry(args.merge_radius, args.cell_size)
    if args.resume and os.path.exists(args.state_file):
        reg.load(args.state_file)
        print(f"resumed {len(reg.survivors)} survivors from {args.state_file}")
    out_sock, out_addr = None, None
    if args.signal_udp:
        host, port = args.signal_udp.split(":")
        out_sock, out_addr = socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port))

    def emit(sig):
        print(json.dumps(sig), flush=True)
        try:
            reg.save(args.state_file)
        except OSError as e:
            print(f"WARNING: cannot write state file: {e}")
        if out_sock:
            try:
                out_sock.sendto(json.dumps(sig).encode(), out_addr)
            except OSError as e:
                print(f"WARNING: signal send failed: {e}")

    pose_src = PoseProvider(args.pose_port)
    node = None
    restarts = errors = 0
    print("running; press q (window) or Ctrl+C to stop", flush=True)
    try:
        while True:
            try:
                src = open_source(args)
                if node is None:
                    node = SurvivorNode(args, model, src.K, pose_src, reg, emit)
                else:
                    node.K = src.K
                for frame, depth in src.frames():
                    try:
                        items = node.step(frame, depth)
                        errors = 0
                    except Exception as e:  # keep flying: one bad frame must not kill the node
                        errors += 1
                        print(f"ERROR frame {node.n}: {type(e).__name__}: {e}", flush=True)
                        if errors >= args.max_errors:
                            print("FATAL: too many consecutive errors", flush=True)
                            sys.exit(2)
                        continue
                    if not args.headless:
                        for x1, y1, x2, y2, label, ok in items:
                            col = (0, 220, 0) if ok else (0, 200, 255)
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
                            cv2.putText(frame, label, (int(x1), max(int(y1) - 6, 15)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
                        cv2.putText(frame, f"Survivors: {len(reg.survivors)}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        cv2.imshow("survivors", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            raise KeyboardInterrupt
                break  # source ended normally
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception as e:  # camera dropped / failed to open
                restarts += 1
                print(f"ERROR source: {type(e).__name__}: {e} (restart {restarts})", flush=True)
                if restarts > args.max_restarts:
                    print("FATAL: camera unavailable", flush=True)
                    sys.exit(3)
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"total survivors: {len(reg.survivors)}")


if __name__ == "__main__":
    main()