#!/usr/bin/env python3
"""
eval_pr.py - object-level precision / recall on a held-out image folder.

  python eval_pr.py --weights best.pt --images D:\\NIDAR\\test\\images
  (labels are read from the sibling 'labels' folder, YOLO txt format; images with no
   label file count as "no people", so keep empty arena shots in the test set)

Prints a table over confidence thresholds and the worst images at --conf-pick,
so you know exactly which photos to look at / add to training.
"""
from __future__ import annotations

import argparse
import glob
import os


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def match(preds, gts, thr=0.5):
    """preds: [(conf, box)], gts: [box]. Greedy by confidence. Returns (tp, fp, fn)."""
    used = set()
    tp = 0
    for conf, pb in sorted(preds, key=lambda t: -t[0]):
        best, best_i = 0.0, -1
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = iou(pb, g)
            if v > best:
                best, best_i = v, i
        if best >= thr:
            used.add(best_i)
            tp += 1
    return tp, len(preds) - tp, len(gts) - tp


def read_labels(path, w, h):
    boxes = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) < 5:
                continue
            cx, cy, bw, bh = (float(v) for v in p[1:5])
            boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                          (cx + bw / 2) * w, (cy + bh / 2) * h))
    return boxes


def selftest():
    g = [(0, 0, 10, 10), (20, 20, 30, 30)]
    assert match([(0.9, (0, 0, 10, 10)), (0.8, (20, 20, 30, 30))], g) == (2, 0, 0)
    assert match([(0.9, (0, 0, 10, 10)), (0.8, (1, 1, 10, 10))], g) == (1, 1, 1)  # duplicate = FP
    assert match([], g) == (0, 0, 2)
    assert match([(0.5, (100, 100, 110, 110))], []) == (0, 1, 0)
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9
    import tempfile
    d = tempfile.mkdtemp()
    p = os.path.join(d, "a.txt")
    open(p, "w").write("0 0.5 0.5 0.2 0.4\n")
    b = read_labels(p, 100, 200)[0]
    assert all(abs(x - y) < 1e-9 for x, y in zip(b, (40, 60, 60, 140)))
    assert read_labels(os.path.join(d, "none.txt"), 100, 100) == []
    print("eval selftest OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights")
    ap.add_argument("--images")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--classes", default="0")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf-pick", type=float, default=0.4)
    ap.add_argument("--worst", type=int, default=10)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    from ultralytics import YOLO

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    imgs = sorted(p for e in exts for p in glob.glob(os.path.join(a.images, e)))
    if not imgs:
        raise SystemExit(f"no images found in {a.images}")
    label_dir = os.path.join(os.path.dirname(a.images.rstrip("/\\")), "labels")
    classes = [int(c) for c in a.classes.split(",")]
    model = YOLO(a.weights)
    data = []  # per image: (path, gts, [(conf, box)])
    for r in model.predict(imgs, imgsz=a.imgsz, conf=0.05, classes=classes,
                           device=a.device, stream=True, verbose=False):
        h, w = r.orig_shape
        stem = os.path.splitext(os.path.basename(r.path))[0]
        gts = read_labels(os.path.join(label_dir, stem + ".txt"), w, h)
        preds = [(float(c), tuple(float(v) for v in b))
                 for c, b in zip(r.boxes.conf.cpu().tolist(), r.boxes.xyxy.cpu().numpy())]
        data.append((r.path, gts, preds))
    total_gt = sum(len(d[1]) for d in data)
    print(f"{a.weights}: {len(data)} images, {total_gt} ground-truth objects, IoU>={a.iou}")
    print(f"{'conf':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'prec':>7} {'recall':>7} {'F1':>6}")
    for conf in (0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
        tp = fp = fn = 0
        for _, gts, preds in data:
            t, f, n = match([p for p in preds if p[0] >= conf], gts, a.iou)
            tp, fp, fn = tp + t, fp + f, fn + n
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print(f"{conf:5.2f} {tp:5d} {fp:5d} {fn:5d} {prec:7.3f} {rec:7.3f} {f1:6.3f}")
    rows = []
    for path, gts, preds in data:
        t, f, n = match([p for p in preds if p[0] >= a.conf_pick], gts, a.iou)
        rows.append((f + n, f, n, path))
    rows.sort(reverse=True)
    print(f"\nworst images at conf={a.conf_pick} (errors, FP, FN):")
    for err, f, n, path in rows[:a.worst]:
        if err:
            print(f"  {err:2d} FP={f} FN={n}  {path}")


if __name__ == "__main__":
    main()