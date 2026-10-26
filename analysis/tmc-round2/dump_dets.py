#!/usr/bin/env python3
"""Dump EVERY detection at a low confidence floor, with no tracking at all.

The cached streams hold only boxes the online tracker confirmed, so they cannot answer
the question the local experiment raised: when a vehicle vanishes mid-intersection, is
the detector still seeing it weakly, or has it genuinely gone blind? This runs the same
4-channel background-conditioned model over the same frames at conf=0.05 and records
every box, so the answer comes from detections rather than from tracker output.

One clip per process: a long single process drifts into a stall after a couple of
videos (atr-iter-2), and a crash must not take the other clips' work with it.
"""
import csv
import gzip
import os
import sys

import cv2

sys.path.insert(0, os.environ.get("PROCESSOR_SRC", "/workspace/src"))
from ultralytics import YOLO
from utils.background_model import BackgroundProvider, model_wants_background

CONF = float(os.environ.get("DUMP_CONF", "0.05"))
MAX_FRAMES = int(os.environ.get("DUMP_MAX_FRAMES", "0"))  # 0 = whole clip; smoke tests cap it
IOU = 0.2


def img_size(w, h, cap=1920):
    """Identical rule to tmc_processor._compute_img_size, so boxes are comparable."""
    return (min(max(w, h), cap) // 32) * 32 or 32


def main(video, model_path, out):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    model = YOLO(model_path)
    names = model.names
    bg = BackgroundProvider(video, fps, total) if model_wants_background(model) else None
    sz = img_size(w, h)
    print(f"{os.path.basename(video)} {w}x{h} fps={fps} frames={total} imgsz={sz} "
          f"bg={'4ch' if bg else '3ch'} conf={CONF}", flush=True)

    n = f = 0
    with gzip.open(out, "wt", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "x1", "y1", "x2", "y2", "conf", "cls"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            inp = bg.stack(frame, f) if bg else frame
            b = model.predict(inp, conf=CONF, iou=IOU, imgsz=sz, verbose=False)[0].boxes
            if b is not None and len(b):
                xy = b.xyxy.cpu().numpy()
                cf = b.conf.cpu().numpy()
                cl = b.cls.cpu().numpy()
                for k in range(len(cf)):
                    wr.writerow([f, round(float(xy[k][0]), 1), round(float(xy[k][1]), 1),
                                 round(float(xy[k][2]), 1), round(float(xy[k][3]), 1),
                                 round(float(cf[k]), 3), names[int(cl[k])]])
                n += len(cf)
            f += 1
            if MAX_FRAMES and f >= MAX_FRAMES:
                break
            if f % 5000 == 0:
                print(f"  {f}/{total} dets={n}", flush=True)
    cap.release()
    print(f"DONE {out} frames={f} dets={n}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
