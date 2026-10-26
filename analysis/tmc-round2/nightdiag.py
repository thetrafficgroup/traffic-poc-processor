#!/usr/bin/env python3
"""Is york_night's NORTH approach dark, or just unmodelled?

Its SOUTH approach is nearly exact (127 manual / 120 filed) while NORTH collapses
(138 / 50), with 66 of 83 lost tracks dying midblock. Before building any night
detector it is worth knowing whether a vehicle on that approach leaves ANY signal:
if the region is black and headlights never enter it, no detector helps and the honest
answer is to report the approach as unmeasurable.

Measures, per leg region (the quad from the leg's line extruded outward, which is the
roadway that approach occupies):
  brightness   mean grey level
  motion       mean absolute frame difference, i.e. is anything moving there at all
  bright blobs count of small bright regions, i.e. headlight-like signal
"""
import collections, json, math, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A

site = sys.argv[1] if len(sys.argv) > 1 else "york_night"
clip = sys.argv[2]
s = [x for x in json.load(open(f"{A.ROOT}/round1_out/sites.json")) if x["site"] == site][0]
LINES = A.replay_mod.build_lines(f"{A.ROOT}/gt/{s['lines']}")
zones = A.build_zones(f"{A.ROOT}/gt/{s['lines']}", 0.6)

cap = cv2.VideoCapture(clip)
W = int(cap.get(3)); H = int(cap.get(4)); N = int(cap.get(7))
masks = {}
for name, poly in zones.items():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    masks[name] = m.astype(bool)
print(f"{site}: {W}x{H}, {N} frames; zone pixel counts: "
      + ", ".join(f"{k}={int(v.sum())}" for k, v in masks.items()))

acc = collections.defaultdict(lambda: dict(b=[], mo=[], blob=[]))
prev = None
step = max(1, N // 240)
f = 0
sampled = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    if f % step == 0:
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is not None:
            diff = cv2.absdiff(g, prev)
            # headlight-like: small, very bright, locally isolated
            _, bright = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
            nb, _, stats, _ = cv2.connectedComponentsWithStats(bright, 8)
            for name, m in masks.items():
                d = acc[name]
                d["b"].append(float(g[m].mean()))
                d["mo"].append(float(diff[m].mean()))
                cnt = 0
                for i in range(1, nb):
                    x, y, w, h, area = stats[i]
                    if 4 <= area <= 400 and m[min(H-1, y + h // 2), min(W-1, x + w // 2)]:
                        cnt += 1
                d["blob"].append(cnt)
            sampled += 1
        prev = g
    f += 1
cap.release()
print(f"\nsampled {sampled} frame pairs (every {step}th frame)")
print(f"  {'leg':8}{'brightness':>12}{'motion':>9}{'bright blobs/frame':>21}")
for name in sorted(acc):
    d = acc[name]
    print(f"  {name:8}{np.mean(d['b']):12.1f}{np.mean(d['mo']):9.2f}"
          f"{np.mean(d['blob']):21.2f}")
