#!/usr/bin/env python3
"""Check a TMC lane config's leg labels against its own geometry. No ground truth.

Why this exists: the movement matrix is (origin leg) x (turn), and the turn is a LOOKUP
on compass names rather than a measured angle, so a camera whose legs carry the wrong
labels reports perfectly tracked vehicles in the wrong cells. Two of the 18 cameras that
have manual counts were wrong this way, and together they accounted for 27% of the whole
corpus's counting error. Manual counts exist for 18 of ~358 customer videos, so the rest
of the fleet has no way to notice.

What it can prove from the config alone:

  An intersection's legs must run NORTH -> EAST -> SOUTH -> WEST around the centre in
  ONE consistent handedness. A camera sits at an arbitrary rotation, so absolute leg
  positions say nothing -- but that cyclic ORDER is invariant under rotation and is
  reversed by a mirror. So a mirrored config is detectable here, with certainty.

What it cannot: a pure 180-degree label rotation preserves the cyclic order, so it is
invisible to this check and needs absolute orientation from OSM or imagery. Sanford was
that case. Treat a clean result as "not mirrored", never as "labels are correct".

Usage:
    python tools/validate_lane_config.py config.json [more.json ...]
    python tools/validate_lane_config.py configs/*.json --quiet

Exits non-zero if any config is flagged, so it can gate a config change.
"""
import argparse
import glob
import json
import math
import sys

CYCLE = ["NORTH", "EAST", "SOUTH", "WEST"]


def _rotations(seq):
    return {tuple(seq[i:] + seq[:i]) for i in range(len(seq))}


def leg_points(entry, name):
    """Vertices of one leg's counting line, tolerating both config schemas."""
    if isinstance(entry, dict):
        pts = entry.get("points")
        if pts:
            return [tuple(p) for p in pts]
        if "pt1" in entry and "pt2" in entry:
            return [tuple(entry["pt1"]), tuple(entry["pt2"])]
    raise ValueError(f"leg {name!r}: no points/pt1-pt2")


def check(path):
    """Return (verdict, detail, order)."""
    data = json.load(open(path))
    raw = dict(data.get("lines", data))
    raw.pop("crosswalks", None)
    legs = {}
    for name, entry in raw.items():
        up = name.upper()
        if up not in CYCLE:
            continue
        try:
            legs[up] = leg_points(entry, name)
        except ValueError as e:
            return "UNREADABLE", str(e), []
    if len(legs) < 3:
        return "SKIPPED", f"only {len(legs)} compass leg(s); need 3+", list(legs)

    pts = [p for v in legs.values() for p in (v[0], v[-1])]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    bearings = []
    for name, v in legs.items():
        mid = ((v[0][0] + v[-1][0]) / 2, (v[0][1] + v[-1][1]) / 2)
        bearings.append((math.atan2(mid[1] - cy, mid[0] - cx), name))
    bearings.sort()
    order = [n for _, n in bearings]

    expected = [l for l in CYCLE if l in legs]
    forward = _rotations(expected)
    reverse = _rotations(expected[::-1])
    t = tuple(order)
    if t in forward:
        return "OK", "cyclic order consistent", order
    if t in reverse:
        return "MIRRORED", ("legs run the wrong way round the intersection: the labels "
                            "are a reflection, which swaps every left and right turn"), order
    return "INCONSISTENT", ("cyclic order matches neither handedness, so at least one "
                            "leg is labelled as the wrong compass point"), order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--quiet", action="store_true", help="print only flagged configs")
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files.extend(sorted(glob.glob(p)) if any(c in p for c in "*?[") else [p])

    flagged = []
    for f in files:
        try:
            verdict, detail, order = check(f)
        except Exception as e:
            verdict, detail, order = "UNREADABLE", repr(e), []
        bad = verdict in ("MIRRORED", "INCONSISTENT", "UNREADABLE")
        if bad:
            flagged.append((f, verdict, detail))
        if bad or not a.quiet:
            print(f"{verdict:13} {f}")
            if order:
                print(f"              order by bearing: {','.join(order)}")
            if bad:
                print(f"              {detail}")
    print(f"\n{len(files)} config(s) checked, {len(flagged)} flagged")
    for f, v, _ in flagged:
        print(f"  {v:13} {f}")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
