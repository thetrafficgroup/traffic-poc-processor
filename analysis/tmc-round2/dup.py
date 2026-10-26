#!/usr/bin/env python3
"""Can appearance tell a MISSED vehicle from a DUPLICATE fragment?

Every attribution fix tested so far recovers ~51% of what it touches, and the reason is
measurable: of 1168 entry-counted-but-unclassified tracks across 18 sites, at most 53%
can be real missing vehicles -- seven sites OVERCOUNT while still holding lost tracks,
so their lost tracks are second fragments of vehicles already counted. Nothing in a
trajectory distinguishes the two cases, so recoveries and double-counts cancel.

Appearance across the whole clip does distinguish them, and only offline: for a lost
track, ask whether some vehicle ALREADY COUNTED on the same origin leg, within a
plausible window, looks like it. If one does, the lost track is that vehicle seen twice
and must not be counted again. Measured separation on this corpus is +0.403 (same
vehicle median 0.067, different 0.470).

Arms:
  H1       shipped
  H1+P     path-fit applied to every lost track          (51% precision, net zero)
  H1+P+D   path-fit applied only to lost tracks that do NOT match a counted vehicle
"""
import argparse
import collections
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import pathfit
import run_exp as E

CLIPS = {
    "gunselford": "customer-comparison/clips2/968_gunselford.mp4",
    "rolling": "customer-comparison/clips2/973_rolling.mp4",
    "sanford": "customer-comparison/clips2/969_sanford.mp4",
    "md108": "customer-comparison/clips2/983_md108.mp4",
    "1022_meadowbranch": "customer-comparison/clips3/1022_meadowbranch.mp4",
    "1016_kriders": "customer-comparison/clips3/1016_kriders.mp4",
    "tamarhayshed": "customer-comparison/clips2/994_tamarhayshed.mp4",
    "honeygo_pm": "customer-comparison/clips2/996_honeygo_pm.mp4",
    "i195": "customer-comparison/clips2/978_i195.mp4",
    "oella": "customer-comparison/clips2/972_oella.mp4",
    "phelpsluck": "customer-comparison/clips2/984_phelpsluck.mp4",
    "selfordsulphur": "customer-comparison/clips2/976_selfordsulphur.mp4",
    "edmonson": "customer-comparison/clips2/980_edmonson.mp4",
    "hilton": "customer-comparison/clips2/982_hilton.mp4",
    "tamarlambskin": "customer-comparison/clips2/995_tamarlambskin.mp4",
}
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = f"{HERE}/dupwork"
os.makedirs(WORK, exist_ok=True)
NB = (6, 4, 4)


def descriptors(clip, tl, want, cache, per_track=8):
    """HSV descriptor per wanted tracklet, from one decode pass over the clip.

    Only `want` tracklets are histogrammed and only `per_track` of each one's
    detections, so the cost is the decode rather than the crops.
    """
    if cache and os.path.exists(cache):
        z = np.load(cache, allow_pickle=True)
        return {k: v for k, v in zip(z["ids"], z["vec"])}
    import cv2
    picks = collections.defaultdict(list)
    for tid in want:
        rows = tl.get(tid) or []
        if not rows:
            continue
        step = max(1, len(rows) // per_track)
        for r in rows[::step][:per_track]:
            picks[int(r[0])].append((tid, r))
    acc = collections.defaultdict(list)
    cap = cv2.VideoCapture(clip)
    fr = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for tid, r in picks.get(fr, ()):
            cx, cy, w, h = r[2], r[3], r[6], r[7]
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            mx, my = w * 0.2, h * 0.2
            a = max(0, int(y1 + my)); b = min(img.shape[0], int(y2 - my))
            c = max(0, int(x1 + mx)); e = min(img.shape[1], int(x2 - mx))
            if b - a < 2 or e - c < 2:
                a, b = max(0, int(y1)), min(img.shape[0], int(y2))
                c, e = max(0, int(x1)), min(img.shape[1], int(x2))
            if b - a < 1 or e - c < 1:
                continue
            crop = cv2.cvtColor(img[a:b, c:e], cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([crop], [0, 1, 2], None, list(NB),
                                [0, 180, 0, 256, 0, 256]).ravel()
            s = hist.sum()
            if s > 0:
                acc[tid].append(hist / s)
        fr += 1
    cap.release()
    out = {}
    for tid, vs in acc.items():
        v = np.median(np.stack(vs), axis=0)
        n = np.linalg.norm(v)
        out[tid] = (v / n if n > 0 else v).astype(np.float32)
    if cache:
        np.savez_compressed(cache, ids=np.array(list(out), dtype=object),
                            vec=np.stack(list(out.values())) if out else np.zeros((0, 96)))
    return out


def duplicates(st, tl, desc, fps, window_s, thresh):
    """Lost tracks that look like a vehicle already counted on the same origin leg."""
    counted_by_leg = collections.defaultdict(list)
    for oid, _ in st["turn_types_by_id"].items():
        ts = st["crossing_timestamps"].get(oid)
        rows = tl.get(oid)
        if not ts or not rows:
            continue
        counted_by_leg[ts[0][0].upper()].append((int(rows[0][0]), oid))
    for v in counted_by_leg.values():
        v.sort()
    win = window_s * fps
    dups = set()
    for oid in st["entry_counted_ids"]:
        if oid in st["turn_types_by_id"]:
            continue
        ts = st["crossing_timestamps"].get(oid)
        rows = tl.get(oid)
        if not ts or not rows or oid not in desc:
            continue
        f0 = int(rows[0][0])
        best = 1e9
        for cf, cid in counted_by_leg.get(ts[0][0].upper(), ()):
            if abs(cf - f0) > win:
                continue
            if cid not in desc:
                continue
            d = 1.0 - float(np.dot(desc[oid], desc[cid]))
            best = min(best, d)
        if best < thresh:
            dups.add(oid)
    return dups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default="1016_kriders,gunselford,rolling,sanford")
    ap.add_argument("--window", type=float, default=6.0)
    ap.add_argument("--thresh", type=float, default=0.25)
    ap.add_argument("--reach", type=float, default=12.0)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    sites = json.load(open(f"{A.ROOT}/round1_out/sites.json"))
    want = a.sites.split(",")
    ARMS = ["H1", "H1+P", "H1+P+D"]
    res, man, info = {}, {}, {}
    for s in sites:
        name = s["site"]
        if name not in want or name not in CLIPS:
            continue
        clip = f"{A.ROOT}/{CLIPS[name]}"
        if not os.path.exists(clip):
            print(f"  .. {name}: clip missing", file=sys.stderr)
            continue
        fps = float(s.get("fps") or 15.0)
        lines = f"{A.ROOT}/gt/{s['lines']}"
        manual = E.SC.manual(name, s["gt"])
        trk = f"{A.ROOT}/round1_out/{name}/{name}.tracks.csv.gz"
        LINES = A.replay_mod.build_lines(lines)
        CT = A.replay_mod.CLASSIFY_TURN
        tj = pathfit.traj_from_stream(trk)
        print(f"  .. {name}", file=sys.stderr, flush=True)

        base = E._h1_state(trk, lines, fps, True)
        lost = [i for i in base["entry_counted_ids"] if i not in base["turn_types_by_id"]]
        need = set(lost) | set(base["turn_types_by_id"])
        desc = descriptors(clip, tj, need, f"{WORK}/{name}.desc.npz")

        out = {"H1": E.score_state(base, manual, False)}
        st1 = E._h1_state(trk, lines, fps, True)
        n1 = pathfit.assign(st1, tj, LINES, CT, fps, max_reach_bh=a.reach)[0]
        out["H1+P"] = E.score_state(st1, manual, False)

        st2 = E._h1_state(trk, lines, fps, True)
        dups = duplicates(st2, tj, desc, fps, a.window, a.thresh)
        for oid in dups:
            st2["entry_counted_ids"].discard(oid)
        n2 = pathfit.assign(st2, tj, LINES, CT, fps, max_reach_bh=a.reach)[0]
        out["H1+P+D"] = E.score_state(st2, manual, False)

        res[name], man[name] = out, sum(manual.values())
        info[name] = dict(lost=len(lost), dups=len(dups), add_P=n1, add_PD=n2,
                          desc=len(desc))

    if not res:
        print("nothing scored")
        return
    print(f"\nduplicate suppression  window={a.window}s  appearance_thresh={a.thresh}  "
          f"reach={a.reach}bh")
    print(f"{'site':18}{'manual':>7}" + "".join(f"{x:>11}" for x in ARMS)
          + f"{'lost':>6}{'flagged dup':>13}{'adds P/P+D':>12}")
    for n in sorted(res, key=lambda k: -res[k]['H1'][0] / man[k]):
        i = info[n]
        print(f"{n:18}{man[n]:7}"
              + "".join(f"{100*res[n][x][0]/man[n]:10.2f}%" for x in ARMS)
              + f"{i['lost']:6}{i['dups']:13}{i['add_P']:>6}/{i['add_PD']:<5}")
    print(f"{'POOLED':18}{sum(man.values()):7}", end="")
    pooled = {}
    for x in ARMS:
        pooled[x] = 100 * sum(res[n][x][0] for n in res) / sum(man.values())
        print(f"{pooled[x]:10.2f}%", end="")
    print()
    print(f"{'vs H1':18}{'':7}" + "".join(f"{pooled[x]-pooled['H1']:+10.2f} " for x in ARMS))
    print("\nworst-site regression vs H1")
    for x in ARMS:
        worst, ws = 0.0, "-"
        for n in res:
            r = 100 * (res[n][x][0] - res[n]["H1"][0]) / man[n]
            if r > worst:
                worst, ws = r, n
        print(f"  {x:10} {worst:+7.2f} pp  ({ws})")
    if a.json_out:
        json.dump(dict(res={k: {x: list(v) for x, v in o.items()} for k, o in res.items()},
                       manual=man, info=info,
                       params=dict(window=a.window, thresh=a.thresh, reach=a.reach)),
                  open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
