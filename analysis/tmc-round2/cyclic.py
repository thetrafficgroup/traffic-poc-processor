#!/usr/bin/env python3
"""Detect mislabelled leg geometry with NO ground truth, from the config alone.

The compass names of an intersection's legs must run N -> E -> S -> W around the
centre in ONE consistent handedness. A camera is at an arbitrary rotation, so absolute
positions say nothing -- but the CYCLIC ORDER is invariant under rotation and is
reversed by a mirror. So a reflection is detectable from the config by itself, which
matters because only 18 of ~358 customer TMC videos have manual counts to catch it.

A pure 180-degree rotation preserves cyclic order and is NOT detectable this way; that
class needs absolute orientation from OSM or imagery.
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A

CYCLE = ["NORTH", "EAST", "SOUTH", "WEST"]


def rotations(seq):
    return [tuple(seq[i:] + seq[:i]) for i in range(len(seq))]


FWD = set(rotations(CYCLE))
REV = set(rotations(CYCLE[::-1]))


def sub_cycles(order, present):
    """Cyclic sequences of the compass cycle restricted to the legs this site has."""
    keep = [l for l in CYCLE if l in present]
    f = set(rotations(keep))
    r = set(rotations(keep[::-1]))
    return f, r


sites = json.load(open(f"{A.ROOT}/round1_out/sites.json"))
print(f"{'site':18}{'legs':6}{'geometric order (by bearing)':34}{'verdict':>12}")
bad = []
for s in sites:
    name = s["site"]
    lines = f"{A.ROOT}/gt/{s['lines']}"
    if not os.path.exists(lines):
        continue
    LINES = A.replay_mod.build_lines(lines)
    pts = [p for ln in LINES for p in (ln["points"][0], ln["points"][-1])]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    ang = []
    for ln in LINES:
        p1, p2 = ln["points"][0], ln["points"][-1]
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ang.append((math.atan2(mid[1] - cy, mid[0] - cx), ln["name"].upper()))
    ang.sort()
    order = [n for _, n in ang]
    present = set(order)
    f, r = sub_cycles(order, present)
    t = tuple(order)
    if t in f:
        v = "ok"
    elif t in r:
        v = "MIRRORED"
        bad.append((name, v))
    else:
        v = "INCONSISTENT"
        bad.append((name, v))
    print(f"{name:18}{len(order):<6}{','.join(order):34}{v:>12}")
print()
if bad:
    print("flagged without using any manual count:")
    for n, v in bad:
        print(f"  {n:20} {v}")
else:
    print("nothing flagged")
