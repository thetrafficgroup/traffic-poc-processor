#!/usr/bin/env python3
"""Measure darkness where the traffic actually is, not where a zone guesses it is.

The outward zone for york_night's NORTH leg lands on a crosswalk and near-field
pavement, so brightness measured there says nothing about the approach. The daytime
run of the SAME camera knows where north traffic really travels: take the detections
of every DAY track whose origin was that leg and use their footprints as the mask.
Then measure the night clip inside that mask.
"""
import collections, json, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import run_exp as E

CLIPS = {"york_day": "customer-comparison/clips2/1005_york_day.mp4",
         "york_night": "customer-comparison/clips2/1005_york_night.mp4"}


def leg_masks(site, shape):
    s = [x for x in json.load(open(f"{A.ROOT}/round1_out/sites.json")) if x["site"] == site][0]
    trk = f"{A.ROOT}/round1_out/{site}/{site}.tracks.csv.gz"
    lines = f"{A.ROOT}/gt/{s['lines']}"
    fps = float(s.get("fps") or 15.0)
    _, tl = A.load(trk)
    st = E._h1_state(trk, lines, fps, True)
    H, W = shape
    masks = collections.defaultdict(lambda: np.zeros((H, W), np.uint8))
    for oid in st["entry_counted_ids"]:
        ts = st["crossing_timestamps"].get(oid)
        rows = tl.get(oid)
        if not ts or not rows:
            continue
        leg = ts[0][0].upper()
        for r in rows:
            cv2.circle(masks[leg], (int(r[2]), int(r[3])), 6, 255, -1)
    return {k: (v > 0) for k, v in masks.items()}


day_cap = cv2.VideoCapture(f"{A.ROOT}/{CLIPS['york_day']}")
H, W = int(day_cap.get(4)), int(day_cap.get(3))
day_cap.release()
masks = leg_masks("york_day", (H, W))
print(f"mask pixels from DAY tracks: " + ", ".join(f"{k}={int(v.sum())}" for k, v in sorted(masks.items())))

print(f"\n{'leg':8}{'day brightness':>16}{'night brightness':>18}{'night bright blobs/f':>22}")
res = {}
for site in ("york_day", "york_night"):
    cap = cv2.VideoCapture(f"{A.ROOT}/{CLIPS[site]}")
    N = int(cap.get(7)); step = max(1, N // 160)
    acc = collections.defaultdict(list); blob = collections.defaultdict(list)
    f = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if f % step == 0:
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            _, br = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
            nb, _, stats, cent = cv2.connectedComponentsWithStats(br, 8)
            for leg, m in masks.items():
                acc[leg].append(float(g[m].mean()) if m.any() else float("nan"))
                c = 0
                for i in range(1, nb):
                    x, y, w, h, area = stats[i]
                    if 4 <= area <= 400:
                        cy, cx = int(cent[i][1]), int(cent[i][0])
                        if 0 <= cy < H and 0 <= cx < W and m[cy, cx]:
                            c += 1
                blob[leg].append(c)
        f += 1
    cap.release()
    res[site] = (acc, blob)
for leg in sorted(masks):
    d = np.nanmean(res["york_day"][0][leg]); n = np.nanmean(res["york_night"][0][leg])
    b = np.mean(res["york_night"][1][leg])
    print(f"{leg:8}{d:16.1f}{n:18.1f}{b:22.2f}")
