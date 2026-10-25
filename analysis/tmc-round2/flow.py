#!/usr/bin/env python3
"""Offline multi-object tracking over raw detections, for the TMC counting layer.

This is the real offline formulation, not post-hoc merging of an online tracker's
output. The previous experiment inherited BoT-SORT's tracklets, so it could only ever
JOIN what the online tracker had already split, and it had no appearance information --
which is precisely the signal needed in a platoon, where motion is uninformative
because every vehicle moves identically.

Five stages:

  1. fine linking      raw detections -> short, high-precision tracklets, built by
                       mutual nearest neighbour over 1-2 frame gaps. Deliberately
                       conservative: a wrong link here cannot be undone later, while a
                       missing one is repaired in stage 3.
  2. appearance        one decode pass over the clip, an HSV histogram per detection
                       aggregated to a per-tracklet median descriptor.
  3. global linking     coarse-to-fine agglomerative rounds. Each round gates candidate
                       (tail, head) pairs, costs them on motion + appearance + size +
                       gap, and solves an OPTIMAL one-to-one assignment within every
                       connected component of the candidate graph. Later rounds allow
                       longer gaps, so cheap evidence is used before expensive guesses.
  4. chain filtering   drop chains that look like detector noise rather than vehicles.
                       This is what makes a conf=0.05 floor safe: a low-confidence
                       detection is kept when it joined a long, moving, confident chain
                       and discarded when it stands alone.
  5. emit              rewrite as the processor's own observation stream so the SHIPPED
                       counting code scores it, with no second implementation to drift.

Every spatial threshold is divided by the vehicle's own box height, so nothing here is
a pixel constant that would have to be refitted per camera.
"""
import argparse
import bisect
import collections
import csv
import gzip
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A

DEFAULTS = dict(
    # stage 1: fine linking
    fine_gap=2, fine_tol=0.9, fine_iou=0.1, nms_iou=0.6,
    # stage 3: global rounds, each (max_gap_seconds, position tolerance in box heights)
    rounds=((0.4, 1.0), (1.0, 1.8), (2.0, 2.5)),
    w_pos=1.0, w_app=0.8, w_size=0.4, w_gap=0.3, w_dir=0.6,
    cos_min=0.0, size_ratio=2.5, cost_max=2.2, min_speed=0.3,
    # stage 4: chain filtering
    min_len=3, min_conf=0.18, min_disp_bh=0.75,
)


# ---------------------------------------------------------------- detections

def load_dets(path):
    """Raw dump -> arrays, with the processor's own centroid/wheel derivation."""
    f, x1, y1, x2, y2, cf = [], [], [], [], [], []
    cls = []
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        for r in csv.DictReader(fh):
            f.append(int(r["frame"]))
            x1.append(float(r["x1"])); y1.append(float(r["y1"]))
            x2.append(float(r["x2"])); y2.append(float(r["y2"]))
            cf.append(float(r["conf"])); cls.append(r["cls"])
    d = dict(frame=np.array(f, dtype=np.int32),
             x1=np.array(x1), y1=np.array(y1), x2=np.array(x2), y2=np.array(y2),
             conf=np.array(cf), cls=np.array(cls))
    order = np.argsort(d["frame"], kind="stable")
    for k in d:
        d[k] = d[k][order]
    # Exactly tmc_processor.get_centroid / get_wheels_position, int() included.
    d["cx"] = ((d["x1"] + d["x2"]) / 2).astype(int).astype(float)
    d["cy"] = ((d["y1"] + d["y2"]) / 2).astype(int).astype(float)
    d["wx"] = d["cx"]
    d["wy"] = d["y2"].astype(int).astype(float)
    d["w"] = d["x2"] - d["x1"]
    d["h"] = d["y2"] - d["y1"]
    d["scale"] = np.maximum(20.0, d["h"])
    return d


def nms(d, idx, iou_thr):
    """Class-agnostic NMS. The dump is raw model output, so the same vehicle appears
    as several boxes under different FHWA classes -- 30% of detections overlap another
    in the same frame at IoU>0.5. Those duplicates compete for links and shred
    tracklets, so they are collapsed to the most confident box first."""
    keep = np.ones(len(d["frame"]), dtype=bool)
    for ii in idx.values():
        if len(ii) < 2:
            continue
        order = sorted(ii, key=lambda i: -d["conf"][i])
        for a_pos, i in enumerate(order):
            if not keep[i]:
                continue
            for j in order[a_pos + 1:]:
                if not keep[j]:
                    continue
                ix = max(0.0, min(d["x2"][i], d["x2"][j]) - max(d["x1"][i], d["x1"][j]))
                iy = max(0.0, min(d["y2"][i], d["y2"][j]) - max(d["y1"][i], d["y1"][j]))
                inter = ix * iy
                if inter <= 0:
                    continue
                ai = (d["x2"][i]-d["x1"][i]) * (d["y2"][i]-d["y1"][i])
                aj = (d["x2"][j]-d["x1"][j]) * (d["y2"][j]-d["y1"][j])
                if inter / max(1e-9, ai + aj - inter) > iou_thr:
                    keep[j] = False
    out = {k: (v[keep] if hasattr(v, "__len__") and len(v) == len(keep) else v)
           for k, v in d.items()}
    return out


def frame_index(d):
    idx = collections.defaultdict(list)
    for i, fr in enumerate(d["frame"]):
        idx[int(fr)].append(i)
    return idx


# ---------------------------------------------------------------- stage 1

def fine_link(d, idx, p):
    """Mutual nearest neighbour over 1..fine_gap frames. Precision over recall."""
    n = len(d["frame"])
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    linked_fwd, linked_bwd = set(), set()
    frames = sorted(idx)
    for fr in frames:
        for gap in range(1, p["fine_gap"] + 1):
            nxt = idx.get(fr + gap)
            if not nxt:
                continue
            cur = [i for i in idx[fr] if i not in linked_fwd]
            nxt = [j for j in nxt if j not in linked_bwd]
            if not cur or not nxt:
                continue
            ci = np.array(cur); nj = np.array(nxt)
            dx = d["wx"][nj][None, :] - d["wx"][ci][:, None]
            dy = d["wy"][nj][None, :] - d["wy"][ci][:, None]
            sc = d["scale"][ci][:, None]
            dist = np.hypot(dx, dy) / (sc * gap)
            hr = np.abs(np.log(np.maximum(d["h"][nj][None, :], 1e-6)
                               / np.maximum(d["h"][ci][:, None], 1e-6)))
            cost = dist + 0.5 * hr
            BIG = 1e6
            cost = np.where(dist > p["fine_tol"], BIG, cost)
            # Optimal one-to-one assignment for this frame pair. Mutual nearest
            # neighbour drops a link whenever two vehicles each prefer the other's
            # detection, which is exactly what happens in a platoon.
            ra, rb = linear_sum_assignment(cost)
            for a, b in zip(ra, rb):
                if cost[a, b] >= BIG:
                    continue
                i, j = int(ci[a]), int(nj[b])
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[rj] = ri
                linked_fwd.add(i)
                linked_bwd.add(j)
    groups = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return [sorted(v, key=lambda i: d["frame"][i]) for v in groups.values()]


# ---------------------------------------------------------------- stage 2

def appearance(clip, d, tracklets, cache):
    """Per-tracklet HSV descriptor. One decode pass; the inner 60% of each box only,
    so road surface around a loose box does not dominate the histogram."""
    if cache and os.path.exists(cache):
        return np.load(cache)
    import cv2
    nb = (6, 4, 4)
    dim = nb[0] * nb[1] * nb[2]
    per_det = np.zeros((len(d["frame"]), dim), dtype=np.float32)
    idx = frame_index(d)
    cap = cv2.VideoCapture(clip)
    fr = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for i in idx.get(fr, ()):
            x1, y1 = d["x1"][i], d["y1"][i]
            x2, y2 = d["x2"][i], d["y2"][i]
            mx, my = (x2 - x1) * 0.2, (y2 - y1) * 0.2
            a = max(0, int(y1 + my)); b = min(img.shape[0], int(y2 - my))
            c = max(0, int(x1 + mx)); e = min(img.shape[1], int(x2 - mx))
            if b - a < 2 or e - c < 2:
                a, b, c, e = (max(0, int(y1)), min(img.shape[0], int(y2)),
                              max(0, int(x1)), min(img.shape[1], int(x2)))
            if b - a < 1 or e - c < 1:
                continue
            crop = cv2.cvtColor(img[a:b, c:e], cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([crop], [0, 1, 2], None, list(nb),
                                [0, 180, 0, 256, 0, 256]).ravel()
            s = hist.sum()
            if s > 0:
                per_det[i] = hist / s
        fr += 1
    cap.release()
    out = np.zeros((len(tracklets), dim), dtype=np.float32)
    for t, members in enumerate(tracklets):
        m = per_det[members]
        keep = m.sum(1) > 0
        if keep.any():
            v = np.median(m[keep], axis=0)
            nv = np.linalg.norm(v)
            out[t] = v / nv if nv > 0 else v
    if cache:
        np.save(cache, out)
    return out


# ---------------------------------------------------------------- stage 3

def summarise(d, chains):
    s = []
    for members in chains:
        fr = d["frame"][members]
        wx, wy = d["wx"][members], d["wy"][members]
        k = min(5, len(members))
        s.append(dict(
            f0=int(fr[0]), f1=int(fr[-1]),
            p_head=(wx[0], wy[0]), p_tail=(wx[-1], wy[-1]),
            v_head=_vel(fr[:k], wx[:k], wy[:k]),
            v_tail=_vel(fr[-k:], wx[-k:], wy[-k:]),
            h=float(np.median(d["h"][members])),
            conf=float(np.mean(d["conf"][members])),
            n=len(members)))
    return s


def _vel(fr, x, y):
    if len(fr) < 2 or fr[-1] == fr[0]:
        return (0.0, 0.0)
    A_ = np.stack([fr - fr[0], np.ones_like(fr, dtype=float)], 1)
    vx = np.linalg.lstsq(A_, x.astype(float), rcond=None)[0][0]
    vy = np.linalg.lstsq(A_, y.astype(float), rcond=None)[0][0]
    return (float(vx), float(vy))


def link_round(s, app, fps, max_gap_s, pos_tol, p):
    """Gate, cost and optimally assign tails to heads inside each component."""
    order = sorted(range(len(s)), key=lambda i: s[i]["f0"])
    starts = [s[i]["f0"] for i in order]
    max_gap = max(2, int(round(max_gap_s * fps)))
    edges = []
    for i in range(len(s)):
        a = s[i]
        lo = bisect.bisect_right(starts, a["f1"])
        hi = bisect.bisect_right(starts, a["f1"] + max_gap)
        for j in order[lo:hi]:
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
            if min(ef, eb) > pos_tol:
                continue
            sa = math.hypot(*a["v_tail"]); sb = math.hypot(*b["v_head"])
            if sa > p["min_speed"] and sb > p["min_speed"]:
                cos = ((a["v_tail"][0] * b["v_head"][0] + a["v_tail"][1] * b["v_head"][1])
                       / (sa * sb))
                if cos < p["cos_min"]:
                    continue
            else:
                cos = 1.0
            app_d = 1.0
            if app is not None:
                na, nb_ = app[i], app[j]
                if na.any() and nb_.any():
                    app_d = 1.0 - float(np.dot(na, nb_))
            cost = (p["w_pos"] * 0.5 * (ef + eb)
                    + p["w_app"] * app_d
                    + p["w_dir"] * (1.0 - cos)
                    + p["w_size"] * abs(math.log(max(b["h"], 1e-6) / max(a["h"], 1e-6)))
                    + p["w_gap"] * gap / max_gap)
            if cost <= p["cost_max"]:
                edges.append((i, j, cost))
    if not edges:
        return {}
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j, _ in edges:
        ri, rj = find(("T", i)), find(("H", j))
        if ri != rj:
            parent[ri] = rj
    comp = collections.defaultdict(list)
    for e in edges:
        comp[find(("T", e[0]))].append(e)

    succ, used_head = {}, set()
    for group in comp.values():
        tails = sorted({e[0] for e in group})
        heads = sorted({e[1] for e in group})
        BIG = 1e6
        M = np.full((len(tails), len(heads)), BIG)
        ti = {t: n for n, t in enumerate(tails)}
        hi = {h: n for n, h in enumerate(heads)}
        for i, j, c in group:
            M[ti[i], hi[j]] = min(M[ti[i], hi[j]], c)
        r, c = linear_sum_assignment(M)
        for a_, b_ in zip(r, c):
            if M[a_, b_] >= BIG:
                continue
            t, hd = tails[a_], heads[b_]
            if hd in used_head or t in succ:
                continue
            succ[t] = hd
            used_head.add(hd)
    return succ


def merge_chains(chains, succ, d):
    pred = {v: k for k, v in succ.items()}
    out = []
    for i in range(len(chains)):
        if i in pred:
            continue
        members, cur, guard = list(chains[i]), i, 0
        while cur in succ and guard < 100000:
            cur = succ[cur]
            members.extend(chains[cur])
            guard += 1
        out.append(sorted(members, key=lambda k: d["frame"][k]))
    return out


# ---------------------------------------------------------------- stages 4-5

def filter_chains(d, chains, p):
    """A conf=0.05 floor only pays if isolated noise is removed afterwards."""
    kept, dropped = [], 0
    for members in chains:
        if len(members) < p["min_len"]:
            dropped += 1
            continue
        if float(np.mean(d["conf"][members])) < p["min_conf"]:
            dropped += 1
            continue
        h = max(20.0, float(np.median(d["h"][members])))
        disp = math.dist((d["wx"][members[0]], d["wy"][members[0]]),
                         (d["wx"][members[-1]], d["wy"][members[-1]]))
        if disp / h < p["min_disp_bh"]:
            dropped += 1
            continue
        kept.append(members)
    return kept, dropped


def emit(d, chains, path):
    rows = []
    for t, members in enumerate(chains):
        cls = collections.Counter(d["cls"][i] for i in members).most_common(1)[0][0]
        for i in members:
            rows.append((int(d["frame"][i]), f"f{t}", d["cx"][i], d["cy"][i],
                         d["wx"][i], d["wy"][i], d["w"][i], d["h"][i],
                         d["conf"][i], cls))
    rows.sort(key=lambda r: r[0])
    with gzip.open(path, "wt", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(A.COLS)
        for r in rows:
            wr.writerow([r[0], r[1], r[2], r[3], r[4], r[5],
                         round(r[6], 1), round(r[7], 1), r[8], r[9]])


def track(dets_path, clip, fps, out_path, p, app_cache=None, verbose=True):
    d = load_dets(dets_path)
    idx = frame_index(d)
    stats = dict(dets=len(d["frame"]))
    d = nms(d, idx, p["nms_iou"])
    idx = frame_index(d)
    stats["after_nms"] = len(d["frame"])
    chains = fine_link(d, idx, p)
    stats["fine_tracklets"] = len(chains)
    app = None
    if clip and os.path.exists(clip):
        app = appearance(clip, d, chains, app_cache)
    stats["appearance"] = bool(app is not None)
    for gi, (gap_s, tol) in enumerate(p["rounds"]):
        s = summarise(d, chains)
        succ = link_round(s, app, fps, gap_s, tol, p)
        chains = merge_chains(chains, succ, d)
        if app is not None:
            app = _reapp(app, succ, len(s))
        stats[f"round{gi}_gap{gap_s}"] = len(chains)
    chains, dropped = filter_chains(d, chains, p)
    stats["dropped"] = dropped
    stats["final_chains"] = len(chains)
    emit(d, chains, out_path)
    if verbose:
        print("  " + "  ".join(f"{k}={v}" for k, v in stats.items()), flush=True)
    return stats


def _reapp(app, succ, n):
    """Chain descriptors after a merge: mean of the merged members, renormalised."""
    pred = {v: k for k, v in succ.items()}
    out = []
    for i in range(n):
        if i in pred:
            continue
        acc, cur, guard = [app[i]], i, 0
        while cur in succ and guard < 100000:
            cur = succ[cur]
            acc.append(app[cur])
            guard += 1
        v = np.mean(np.stack(acc), axis=0)
        nv = np.linalg.norm(v)
        out.append(v / nv if nv > 0 else v)
    return np.stack(out) if out else None
