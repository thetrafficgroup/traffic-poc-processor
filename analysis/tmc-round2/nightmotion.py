#!/usr/bin/env python3
"""Last test for york_night NORTH: can motion find what the detector and headlights cannot?

138 vehicles enter on that approach; the detector books 50. Brightness there is 17.4 and
there are 0.10 headlight-like blobs per frame, so appearance gives nothing. Motion is not
zero (2.58 mean abs diff), so this asks whether background subtraction alone can see
vehicles crossing the NORTH line -- the only remaining mechanism.

Counts distinct moving blobs that cross the leg's line, with a per-blob cooldown so one
vehicle is not counted on consecutive frames. Deliberately generous: this is measuring
whether the SIGNAL exists, not proposing a counter. If a permissive motion detector
cannot reach the manual count, the approach is unmeasurable and should be reported as
such rather than silently undercounted.
"""
import collections, json, math, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A

site, clip = sys.argv[1], sys.argv[2]
s = [x for x in json.load(open(f"{A.ROOT}/round1_out/sites.json")) if x["site"] == site][0]
cfg = f"{A.ROOT}/gt/{s['lines']}"
LINES = {l["name"].upper(): l for l in A.replay_mod.build_lines(cfg)}
zones = A.build_zones(cfg, 0.6)

cap = cv2.VideoCapture(clip)
W, H, N = int(cap.get(3)), int(cap.get(4)), int(cap.get(7))
fps = cap.get(5)
masks = {}
for name, poly in zones.items():
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.array(poly, np.int32)], 255)
    masks[name] = m

mog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
MIN_AREA = int(os.environ.get("MIN_AREA", "40"))
crossings = collections.Counter()
last = collections.defaultdict(dict)
f = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    fg = mog.apply(frame)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    nb, lab, stats, cent = cv2.connectedComponentsWithStats(fg, 8)
    for i in range(1, nb):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA:
            continue
        cx, cy = cent[i]
        for name, m in masks.items():
            if m[min(H - 1, int(cy)), min(W - 1, int(cx))] == 0:
                continue
            seg = LINES[name]["segments"]
            d = min(min(math.dist((cx, cy), a), math.dist((cx, cy), b)) for a, b in seg)
            if d > 2.5 * max(10, h):
                continue
            key = (int(cx) // 25, int(cy) // 25)
            if f - last[name].get(key, -10**9) > fps * 1.5:
                crossings[name] += 1
            last[name][key] = f
            break
    f += 1
cap.release()
print(f"{site}: {W}x{H} @ {fps:.0f}fps, {f} frames, min_area={MIN_AREA}")
print(f"  {'leg':8}{'motion crossings':>18}{'manual row':>13}{'detector entries':>18}")
MAN = {"NORTH": 138, "SOUTH": 127, "WEST": 14, "EAST": 0}
DET = {"NORTH": 50, "SOUTH": 197, "WEST": 10, "EAST": 0}
for name in sorted(masks):
    print(f"  {name:8}{crossings.get(name,0):18}{MAN.get(name,0):13}{DET.get(name,0):18}")
