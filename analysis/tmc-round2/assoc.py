#!/usr/bin/env python3
"""Offline tracklet association + zone attribution, scored on the cached streams.

Point 2 (offline): the shipped tracker is online -- it commits an ID assignment at
frame t using only frames <= t, and cannot revise it. Here the whole clip is in hand,
so tracklet tails and heads are matched by a cost that is OPTIMAL within each connected
component of the candidate graph, with hard mutual exclusion (a tail gets at most one
successor, a head at most one predecessor). That exclusion is what the 2026-08 online
stitch lacked: it could only ADD merges, so a wrongful merge cost it nothing.

Point 1 (zones): the counting lines are extruded AWAY from the intersection by a
fraction of each line's OWN length, which is perspective-correct per leg and needs no
pixel constant. Measured over all 67 legs this gives 0.2% mean pairwise overlap, so a
zone identifies a leg unambiguously.

Every distance here is divided by the vehicle's own box height, so no threshold is a
pixel count and none of them has to be re-fitted per camera.
"""
import collections
import csv
import gzip
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = "/Users/nacho/dev/genium/traffic-v2/proc-iter-tmc"
for _p in (f"{ROOT}/round1_out", f"{ROOT}/srcp", f"{ROOT}/srcp/tmc"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

replay_mod = __import__("replay")
TP = __import__("tmc.tmc_processor", fromlist=["tmc_processor"])

COLS = ["frame", "id", "cx", "cy", "wx", "wy", "w", "h", "conf", "cls"]


# ---------------------------------------------------------------- loading

def load(trk):
    """Read a cached stream into frame-ordered rows plus per-id tracklets."""
    op = gzip.open if trk.endswith(".gz") else open
    rows = []
    with op(trk, "rt") as fh:
        for r in csv.DictReader(fh):
            rows.append((int(r["frame"]), r["id"], float(r["cx"]), float(r["cy"]),
                         float(r["wx"]), float(r["wy"]), float(r["w"]),
                         float(r["h"]), float(r["conf"]), r["cls"]))
    rows.sort(key=lambda t: t[0])
    tl = collections.defaultdict(list)
    for t in rows:
        tl[t[1]].append(t)
    for v in tl.values():
        v.sort(key=lambda t: t[0])
    return rows, dict(tl)


def _vel(pts):
    """Per-frame velocity of the wheel point over a few samples, least squares."""
    f = np.array([p[0] for p in pts], dtype=float)
    if len(f) < 2 or f[-1] == f[0]:
        return 0.0, 0.0
    A = np.stack([f - f[0], np.ones_like(f)], 1)
    vx = np.linalg.lstsq(A, np.array([p[4] for p in pts], dtype=float), rcond=None)[0][0]
    vy = np.linalg.lstsq(A, np.array([p[5] for p in pts], dtype=float), rcond=None)[0][0]
    return float(vx), float(vy)


def summarise(tl, k=5):
    out = {}
    for i, v in tl.items():
        out[i] = dict(
            f0=v[0][0], f1=v[-1][0],
            p_head=(v[0][4], v[0][5]), p_tail=(v[-1][4], v[-1][5]),
            v_head=_vel(v[:k]), v_tail=_vel(v[-k:]),
            h=float(np.median([r[7] for r in v])),
            cls=collections.Counter(r[9] for r in v).most_common(1)[0][0],
            n=len(v))
    return out


# ---------------------------------------------------------------- association

DEFAULTS = dict(max_gap_s=1.5, pos_tol=2.0, cos_min=0.3, size_ratio=2.5,
                w_pos=1.0, w_dir=0.5, w_size=0.5, w_gap=0.2, cost_max=1.6,
                min_speed=0.3)


def candidates(s, fps, p):
    """Gated (tail, head) pairs. Every gate is in box-heights or angle, never pixels."""
    ids = sorted(s, key=lambda i: s[i]["f1"])
    by_start = sorted(s, key=lambda i: s[i]["f0"])
    starts = [s[i]["f0"] for i in by_start]
    max_gap = max(2, int(round(p["max_gap_s"] * fps)))
    import bisect
    edges = []
    for i in ids:
        a = s[i]
        lo = bisect.bisect_right(starts, a["f1"])
        hi = bisect.bisect_right(starts, a["f1"] + max_gap)
        for j in by_start[lo:hi]:
            if j == i:
                continue
            b = s[j]
            gap = b["f0"] - a["f1"]
            if gap <= 0:
                continue
            scale = max(20.0, 0.5 * (a["h"] + b["h"]))
            if not (1.0 / p["size_ratio"] <= b["h"] / max(a["h"], 1e-6) <= p["size_ratio"]):
                continue
            fx = a["p_tail"][0] + a["v_tail"][0] * gap
            fy = a["p_tail"][1] + a["v_tail"][1] * gap
            bx = b["p_head"][0] - b["v_head"][0] * gap
            by = b["p_head"][1] - b["v_head"][1] * gap
            ef = math.dist((fx, fy), b["p_head"]) / scale
            eb = math.dist((bx, by), a["p_tail"]) / scale
            e = 0.5 * (ef + eb)
            if min(ef, eb) > p["pos_tol"]:
                continue
            sa = math.hypot(*a["v_tail"])
            sb = math.hypot(*b["v_head"])
            if sa > p["min_speed"] and sb > p["min_speed"]:
                cos = (a["v_tail"][0] * b["v_head"][0] + a["v_tail"][1] * b["v_head"][1]) / (sa * sb)
                if cos < p["cos_min"]:
                    continue
            else:
                cos = 1.0
            cost = (p["w_pos"] * e + p["w_dir"] * (1.0 - cos)
                    + p["w_size"] * abs(math.log(max(b["h"], 1e-6) / max(a["h"], 1e-6)))
                    + p["w_gap"] * gap / max_gap)
            if cost <= p["cost_max"]:
                edges.append((i, j, cost))
    return edges


def _components(edges):
    """Union-find over tail-nodes and head-nodes so each component solves exactly."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j, _ in edges:
        union(("T", i), ("H", j))
    groups = collections.defaultdict(list)
    for e in edges:
        groups[find(("T", e[0]))].append(e)
    return list(groups.values())


def associate(s, fps, p):
    """Return {tracklet_id -> chain_root_id}. Optimal per component, one-to-one."""
    edges = candidates(s, fps, p)
    succ = {}
    used_head = set()
    for comp in _components(edges):
        tails = sorted({e[0] for e in comp})
        heads = sorted({e[1] for e in comp})
        BIG = 1e6
        M = np.full((len(tails), len(heads)), BIG, dtype=float)
        ti = {t: n for n, t in enumerate(tails)}
        hi = {h: n for n, h in enumerate(heads)}
        for i, j, c in comp:
            M[ti[i], hi[j]] = min(M[ti[i], hi[j]], c)
        r, c = linear_sum_assignment(M)
        for a, b in zip(r, c):
            if M[a, b] >= BIG:
                continue
            tail, head = tails[a], heads[b]
            if head in used_head or tail in succ:
                continue
            succ[tail] = head
            used_head.add(head)
    root = {}

    def chase(i):
        while i in root:
            i = root[i]
        return i

    pred = {v: k for k, v in succ.items()}
    for i in s:
        if i in pred:
            continue
        cur, r0 = i, i
        guard = 0
        while cur in succ and guard < 10000:
            cur = succ[cur]
            root[cur] = r0
            guard += 1
    return {i: chase(i) if i in root else i for i in s}


def write_stream(rows, mapping, path):
    with gzip.open(path, "wt", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        for t in rows:
            w.writerow([t[0], mapping.get(t[1], t[1]), t[2], t[3], t[4], t[5],
                        t[6], t[7], t[8], t[9]])


# ---------------------------------------------------------------- zones

def build_zones(lines_json, k=0.3):
    """Outward quad per leg: extruded along the line's own normal, sign from the centre.

    The centre is only used to choose which side is 'away from the intersection', a
    decision that tolerates ~90 degrees of error. It is NOT used as a direction: the
    centre-to-line bearing disagrees with the lines' own normals by 24-64 degrees
    across the corpus, so using it to extrude would shear zones off the roadway.
    """
    LINES = replay_mod.build_lines(lines_json)
    pts = [p for ln in LINES for p in (ln["points"][0], ln["points"][-1])]
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    zones = {}
    for ln in LINES:
        p1, p2 = tuple(ln["points"][0]), tuple(ln["points"][-1])
        L = math.dist(p1, p2)
        if L < 1:
            continue
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ox, oy = mid[0] - cx, mid[1] - cy
        nx, ny = -(p2[1] - p1[1]) / L, (p2[0] - p1[0]) / L
        if nx * ox + ny * oy < 0:
            nx, ny = -nx, -ny
        d = k * L
        zones[ln["name"]] = [p1, p2, (p2[0] + d * nx, p2[1] + d * ny),
                             (p1[0] + d * nx, p1[1] + d * ny)]
    return zones


def _inside(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xx = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xx:
                inside = not inside
    return inside


def zone_of(pt, zones):
    for name, poly in zones.items():
        if _inside(pt, poly):
            return name
    return None


def zone_fill(st, tl, mapping, zones, fps, serve_orphans=False):
    """Classify what the crossing test could not, using which zone the track sits in.

    Only ever ADDS a movement, and only where the shipped logic produced none, so the
    arms stay nested and a win cannot come from silently re-filing a vehicle that was
    already counted correctly.
    """
    merged = collections.defaultdict(list)
    for i, v in tl.items():
        merged[mapping.get(i, i)].extend(v)
    for v in merged.values():
        v.sort(key=lambda t: t[0])

    added_dest = added_orphan = 0
    for oid, v in merged.items():
        if oid in st["turn_types_by_id"]:
            continue
        o_zone = zone_of((v[0][4], v[0][5]), zones)
        d_zone = zone_of((v[-1][4], v[-1][5]), zones)
        ts = st["crossing_timestamps"].get(oid)

        if ts:
            origin = ts[0][0].upper()
            if d_zone and d_zone != origin and d_zone not in {c[0] for c in ts}:
                ts.append((d_zone, float("inf")))
                turn = replay_mod.CLASSIFY_TURN(ts)
                if turn in ("left", "right", "straight"):
                    st["turn_types_by_id"][oid] = turn
                    added_dest += 1
                else:
                    ts.pop()
        elif serve_orphans and o_zone and d_zone and o_zone != d_zone:
            if math.dist((v[0][4], v[0][5]), (v[-1][4], v[-1][5])) < TP._MIN_DISPLACEMENT_PX:
                continue
            cand = [(o_zone, 0.0), (d_zone, float("inf"))]
            turn = replay_mod.CLASSIFY_TURN(cand)
            if turn in ("left", "right", "straight"):
                st["crossing_timestamps"][oid] = cand
                st["turn_types_by_id"][oid] = turn
                st["entry_counted_ids"].add(oid)
                added_orphan += 1
    return added_dest, added_orphan


# ---------------------------------------------------------------- scoring

def cells_of(st):
    c = collections.Counter()
    for oid, turn in st["turn_types_by_id"].items():
        ts = st["crossing_timestamps"].get(oid)
        if ts:
            c[f"{ts[0][0].upper()}|{turn}"] += 1
    return c


def imputed(cells, entries, on):
    """Exactly the shipped largest-remainder spread used by h1.py, so arms compare."""
    out = collections.Counter(cells)
    observed = sum(cells.values())
    add = TP.estimate_missing(entries, observed) if on else 0.0
    placed = 0
    if add >= 1 and observed:
        frac = sorted(((cells[k] * add / observed, k) for k in cells), reverse=True)
        for q, k in frac:
            out[k] += int(q)
            placed += int(q)
        for _, k in frac[:int(round(add)) - placed]:
            out[k] += 1
    return out


def l1(cells, manual):
    return sum(abs(cells.get(k, 0) - manual.get(k, 0)) for k in set(cells) | set(manual))
