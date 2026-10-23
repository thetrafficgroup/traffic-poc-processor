#!/usr/bin/env python3
"""Produce an observation stream in the harness's own format, at a chosen imgsz.

Round 2 left one large open lever: `imgsz = min(max(w,h),1920)//32*32` caps inference
at the video's native resolution, so a 640x480 camera gets NO upscaling while its
vehicles are 15.6 px tall. Upscaling small objects is the standard remedy, but it has
to be measured end to end -- more raw detections prove nothing if the tracker cannot
hold them together.

So this runs the REAL tracking path (same BoT-SORT config, same 4-channel background
model, same overlap post-processing on busy frames) and writes exactly the columns the
replay harness consumes, so an imgsz variant is scored by the shipped counting code
against the manual counts rather than by a detection count.

Control runs at the native imgsz must reproduce the existing cached stream closely; if
they do not, the comparison is measuring this script rather than imgsz.
"""
import csv
import gzip
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.environ.get("PROCESSOR_SRC", "/workspace/src"))
sys.path.insert(0, os.path.join(os.environ.get("PROCESSOR_SRC", "/workspace/src"), "tmc"))
from ultralytics import YOLO
from utils.background_model import BackgroundProvider, model_wants_background
from utils.overlap_detection import post_process_detections
from tmc.tmc_processor import _build_tracker_config

COLS = ["frame", "id", "cx", "cy", "wx", "wy", "w", "h", "conf", "cls"]
CONF = float(os.environ.get("TMC_CONF", "0.15"))
IOU = float(os.environ.get("TMC_IOU", "0.2"))
MAX_FRAMES = int(os.environ.get("DUMP_MAX_FRAMES", "0"))
# Low-light enhancement, applied to the colour channels only. Unlike upscaling -- which
# interpolates and adds no information -- this redistributes the information already
# present into the brightness range the detector was trained on. The 4th (background
# difference) channel is left on the raw frame so the motion signal is unchanged.
ENHANCE = os.environ.get("TMC_ENHANCE", "none")
_CLAHE = None


def enhance(stack):
    global _CLAHE
    if ENHANCE == "none":
        return stack
    out = stack.copy()
    bgr = out[:, :, :3] if out.ndim == 3 and out.shape[2] >= 3 else out
    if ENHANCE == "clahe":
        if _CLAHE is None:
            _CLAHE = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = _CLAHE.apply(lab[:, :, 0])
        bgr2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    elif ENHANCE == "gamma":
        g = float(os.environ.get("TMC_GAMMA", "0.55"))
        lut = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
        bgr2 = cv2.LUT(bgr, lut)
    else:
        raise SystemExit(f"unknown TMC_ENHANCE={ENHANCE!r}")
    if out.ndim == 3 and out.shape[2] >= 3:
        out[:, :, :3] = bgr2
        return out
    return bgr2


def native_imgsz(w, h, cap=1920):
    return (min(max(w, h), cap) // 32) * 32 or 32


def main(video, model_path, out, imgsz_override=0):
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
    sz = int(imgsz_override) if imgsz_override else native_imgsz(w, h)
    tracker_cfg = _build_tracker_config(fps)
    # Tracker overrides, e.g. TRK_OVERRIDES="match_thresh=0.9,new_track_thresh=0.5".
    # The shipped config scales track_buffer by fps but leaves the IoU and appearance
    # gates fixed, and those are fps-sensitive in effect: at 10 fps a vehicle moves
    # 1.5x further between frames than at 15, so the same IoU threshold is a stricter
    # association test on a slower camera.
    ov = os.environ.get("TRK_OVERRIDES", "").strip()
    if ov:
        import yaml
        cfg = yaml.safe_load(open(tracker_cfg))
        for item in ov.split(","):
            k, _, v = item.partition("=")
            k = k.strip()
            if k not in cfg:
                raise SystemExit(f"unknown tracker key {k!r}; have {sorted(cfg)}")
            cfg[k] = yaml.safe_load(v)
        yaml.dump(cfg, open(tracker_cfg, "w"), default_flow_style=False, sort_keys=False)
        print(f"TRK_OVERRIDES applied: {ov}", flush=True)
    print(f"{os.path.basename(video)} {w}x{h} fps={fps:.3f} frames={total} "
          f"imgsz={sz} (native {native_imgsz(w,h)}) bg={'4ch' if bg else '3ch'} "
          f"conf={CONF} enhance={ENHANCE}", flush=True)

    n = f = 0
    with gzip.open(out, "wt", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(COLS)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            inp = enhance(bg.stack(frame, f) if bg else frame)
            res = model.track(inp, persist=True, conf=CONF, imgsz=sz, iou=IOU,
                              tracker=tracker_cfg, verbose=False)
            b = res[0].boxes
            if b is not None and b.id is not None:
                ids = b.id.cpu().numpy()
                boxes = b.xyxy.cpu().numpy()
                classes = b.cls.cpu().numpy()
                scores = b.conf.cpu().numpy()
                # Same rule as the shipped loop: overlap post-processing only on busy
                # frames, every third frame.
                if len(boxes) >= 10 and f % 3 == 0:
                    boxes, scores, classes, ids = post_process_detections(
                        boxes, scores, classes, ids)
                for k in range(len(ids)):
                    x1, y1, x2, y2 = boxes[k]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    wx, wy = int((x1 + x2) / 2), int(y2)
                    wr.writerow([f, int(ids[k]), cx, cy, wx, wy,
                                 round(float(x2 - x1), 1), round(float(y2 - y1), 1),
                                 round(float(scores[k]), 3), names[int(classes[k])]])
                    n += 1
            f += 1
            if MAX_FRAMES and f >= MAX_FRAMES:
                break
            if f % 5000 == 0:
                print(f"  {f}/{total} obs={n}", flush=True)
    cap.release()
    print(f"DONE {out} frames={f} obs={n} imgsz={sz}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3],
         int(sys.argv[4]) if len(sys.argv) > 4 else 0)
