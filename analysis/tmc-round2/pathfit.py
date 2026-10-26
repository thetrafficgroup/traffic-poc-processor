#!/usr/bin/env python3
"""Decide a movement from the whole trajectory, not from a crossing event (H5).

A line crossing is an event: it needs two consecutive positions straddling the line, so
a vehicle whose track ends inside the intersection is unclassifiable no matter how
obvious its path was. H1 patches that by extrapolating one second and testing for a
real crossing, which still fails when the track dies far from any line.

Offline, the entire trajectory is in hand, so the question can be asked the other way
round: of the legal movements out of this vehicle's origin leg, which one does its path
best fit? The ray from its last position along its own final heading is intersected
with every other leg's line; the destination is the leg it would actually reach.

Two guards keep this from inventing movements:
  * a NULL option -- if no leg is reached within max_reach_bh box-heights, or the
    heading is slower than min_speed, the vehicle stays unclassified. Undercounting is
    the honest failure here; a fabricated turn is not.
  * the ray must hit the line SEGMENT, not its infinite extension, so a vehicle
    heading into the kerb between two legs is not credited to either.
"""
import math


def _ray_segment(p, d, a, b):
    """Distance along ray p+t*d to its hit with segment a-b, or None."""
    (px, py), (dx, dy) = p, d
    (ax, ay), (bx, by) = a, b
    ex, ey = bx - ax, by - ay
    den = dx * ey - dy * ex
    if abs(den) < 1e-9:
        return None
    t = ((ax - px) * ey - (ay - py) * ex) / den
    u = ((ax - px) * dy - (ay - py) * dx) / den
    if t <= 0 or not (0.0 <= u <= 1.0):
        return None
    return t


def assign(st, traj_by_id, LINES, classify, fps,
           max_reach_bh=12.0, min_speed_px=0.5, tail=6):
    """Fill in movements for entry-counted tracks the crossing test could not classify.

    Returns (added, rejected). Only ever adds where the shipped logic produced nothing,
    so the arms stay nested and a gain cannot come from re-filing a correct vehicle.
    """
    added = rejected = 0
    for oid in list(st["entry_counted_ids"]):
        if oid in st["turn_types_by_id"]:
            continue
        ts = st["crossing_timestamps"].get(oid)
        tr = traj_by_id.get(oid)
        if not ts or not tr or len(tr) < 2:
            continue
        origin = ts[0][0].upper()
        already = {c[0].upper() for c in ts}

        pts = tr[-tail:] if len(tr) >= tail else tr
        f0, f1 = pts[0][0], pts[-1][0]
        if f1 <= f0:
            rejected += 1
            continue
        x0, y0 = pts[0][4], pts[0][5]
        x1, y1 = pts[-1][4], pts[-1][5]
        df = float(f1 - f0)
        vx, vy = (x1 - x0) / df, (y1 - y0) / df
        speed = math.hypot(vx, vy)
        if speed < min_speed_px:
            rejected += 1
            continue
        h = max(20.0, float(pts[-1][7]))
        reach = max_reach_bh * h

        best = None
        for L in LINES:
            if L["name"].upper() in already:
                continue
            for a, b in L["segments"]:
                t = _ray_segment((x1, y1), (vx / speed, vy / speed), a, b)
                if t is None or t > reach:
                    continue
                if best is None or t < best[0]:
                    best = (t, L["name"].upper())
        if best is None:
            rejected += 1
            continue

        cand = list(ts) + [(best[1], float("inf"))]
        turn = classify(cand)
        if turn in ("left", "right", "straight"):
            st["crossing_timestamps"][oid] = cand
            st["turn_types_by_id"][oid] = turn
            added += 1
        else:
            rejected += 1
    return added, rejected


def traj_from_stream(path):
    """id -> frame-ordered rows, in the cached stream's own column order."""
    import collections
    import assoc as A
    _, tl = A.load(path)
    return {k: sorted(v, key=lambda r: r[0]) for k, v in tl.items()}
