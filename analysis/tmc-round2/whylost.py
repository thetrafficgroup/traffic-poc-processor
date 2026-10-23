#!/usr/bin/env python3
"""Why was each lost track lost? Separate fixable geometry from genuine dropout.

A crossing needs the track to pass the line SEGMENT. If a counting line is too short
to span the roadway, a vehicle on an outer lane passes the segment's EXTENSION and is
never credited -- entry-counted, never classified, with flawless detection throughout.
That is a config defect, is fixable, and is invisible in any aggregate number.

Causes, tested in order of how cheaply they can be fixed:
  short_line   passed close to a line's infinite extension but outside the segment,
               i.e. the segment does not reach far enough across the road
  edge         left the frame without a second crossing: the line may sit too far
               inside the field of view
  midblock     died well inside the frame, far from any line: real dropout
"""
import collections, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import run_exp as E


def foot(p, a, b):
    """(perpendicular distance, u) where u<0 or u>1 means beyond the segment."""
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    dd = dx * dx + dy * dy
    if dd < 1e-9:
        return math.dist(p, a), 0.0
    u = ((px - ax) * dx + (py - ay) * dy) / dd
    fx, fy = ax + u * dx, ay + u * dy
    return math.hypot(px - fx, py - fy), u


sites = json.load(open(f"{A.ROOT}/round1_out/sites.json"))
print(f"{'site':18}{'lost':>6}{'short_line':>12}{'edge':>7}{'midblock':>10}{'near_crossed':>14}")
tot = collections.Counter()
for s in sites:
    name = s["site"]
    trk = f"{A.ROOT}/round1_out/{name}/{name}.tracks.csv.gz"
    if name in E.SKIP or not os.path.isfile(trk):
        continue
    W, H = (int(v) for v in s["res"].split("x"))
    fps = float(s.get("fps") or 15.0)
    lines = f"{A.ROOT}/gt/{s['lines']}"
    LINES = A.replay_mod.build_lines(lines)
    _, tl = A.load(trk)
    st = E._h1_state(trk, lines, fps, True)
    c = collections.Counter()
    for oid in st["entry_counted_ids"]:
        if oid in st["turn_types_by_id"]:
            continue
        rows = tl.get(oid)
        if not rows:
            continue
        last = rows[-1]
        p = (last[4], last[5])
        h = max(20.0, last[7])
        crossed = {x[0].upper() for x in st["crossing_timestamps"].get(oid, [])}
        best = None
        for L in LINES:
            for a, b in L["segments"]:
                d, u = foot(p, a, b)
                if best is None or d < best[0]:
                    best = (d, u, L["name"].upper())
        d, u, lname = best
        at_edge = not (0.04 * W < p[0] < 0.96 * W and 0.04 * H < p[1] < 0.96 * H)
        if d <= 1.5 * h and not (0.0 <= u <= 1.0) and lname not in crossed:
            c["short_line"] += 1
        elif at_edge:
            c["edge"] += 1
        elif d <= 1.5 * h and lname in crossed:
            c["near_crossed"] += 1
        else:
            c["midblock"] += 1
    n = sum(c.values())
    if not n:
        continue
    tot.update(c)
    print(f"{name:18}{n:6}{c['short_line']:12}{c['edge']:7}{c['midblock']:10}"
          f"{c['near_crossed']:14}")
N = sum(tot.values())
print(f"\n{'POOLED':18}{N:6}{tot['short_line']:12}{tot['edge']:7}{tot['midblock']:10}"
      f"{tot['near_crossed']:14}")
print(f"{'share':18}{'':6}{100*tot['short_line']/N:11.0f}%{100*tot['edge']/N:6.0f}%"
      f"{100*tot['midblock']/N:9.0f}%{100*tot['near_crossed']/N:13.0f}%")
