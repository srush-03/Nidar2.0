#!/usr/bin/env python3
"""
capture_dataset.py - save frames from the OAK-D Lite with the SAME pipeline/size the
detector uses, so training pixels match flight pixels.

  python3 capture_dataset.py --session day1_lab_a --interval 0.5 --seconds 300
Frames go to dataset_raw/<session>/ . Walk/fly the camera through the mock arena at
the real height/tilt; move dummies and people between takes (standing, sitting,
lying, partly hidden), and also record EMPTY shots (no people) as negatives.
Keep sessions separate: you will split train/val/test BY SESSION, never by frame.
"""
import argparse
import os
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--out", default="dataset_raw")
    ap.add_argument("--interval", type=float, default=0.5, help="seconds between saves")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--oak-w", type=int, default=640)
    ap.add_argument("--oak-h", type=int, default=360)
    a = ap.parse_args()
    import cv2
    from survivor_detector import OakSource

    folder = os.path.join(a.out, a.session)
    os.makedirs(folder, exist_ok=True)
    src = OakSource(a.oak_w, a.oak_h, with_depth=False)
    t0 = last = time.time()
    n = 0
    for frame, _ in src.frames():
        now = time.time()
        if now - last >= a.interval:
            cv2.imwrite(os.path.join(folder, f"{a.session}_{n:05d}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            n += 1
            last = now
        if now - t0 >= a.seconds:
            break
    print(f"saved {n} frames to {folder}")


if __name__ == "__main__":
    main()