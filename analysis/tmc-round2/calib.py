#!/usr/bin/env python3
"""Calibrate the linking gates from data instead of guessing them.

Uses the cached BoT-SORT stream, whose IDs are mostly right within a tracklet, to
measure (a) how far a real vehicle moves between consecutive detections, in units of
its own box height, and (b) how far apart HSV appearance descriptors are for the SAME
vehicle versus DIFFERENT vehicles. (b) decides whether appearance deserves any weight
at all -- if the two distributions overlap, the term is noise and is currently
blocking correct merges rather than enabling them.
"""
import collections, json, math, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import flow

site = sys.argv[1]
ROOT = A.ROOT
trk = f"{ROOT}/round1_out/{site}/{site}.tracks.csv.gz"
rows, tl = A.load(trk)

step, gaps = [], []
for i, v in tl.items():
    for a, b in zip(v, v[1:]):
        g = b[0] - a[0]
        gaps.append(g)
        sc = max(20.0, a[7])
        step.append(math.dist((a[4], a[5]), (b[4], b[5])) / (sc * max(1, g)))
step = np.array(step); gaps = np.array(gaps)
print(f"{site}: {len(tl)} tracklets, {len(rows)} dets")
print("  per-frame step (box-heights): " + "  ".join(
    f"p{q}={np.percentile(step,q):.2f}" for q in (50, 75, 90, 95, 99)))
print("  frame gap inside a tracklet:  " + "  ".join(
    f"p{q}={np.percentile(gaps,q):.0f}" for q in (50, 90, 99)) +
    f"   gap>1: {100*np.mean(gaps>1):.1f}%")

dets = f"{os.path.dirname(os.path.abspath(__file__))}/flowwork/{site}.dets.csv.gz"
if os.path.exists(dets):
    d = flow.load_dets(dets)
    idx = flow.frame_index(d)
    chains = flow.fine_link(d, idx, flow.DEFAULTS)
    app = flow.appearance(None, d, chains,
                          f"{os.path.dirname(os.path.abspath(__file__))}/flowwork/{site}.app.npy")
    if app is not None and app.shape[0] == len(chains):
        s = flow.summarise(d, chains)
        by_end = sorted(range(len(s)), key=lambda i: s[i]["f1"])
        same, diff = [], []
        for n, i in enumerate(by_end):
            for j in by_end[n+1:n+12]:
                g = s[j]["f0"] - s[i]["f1"]
                if g <= 0 or g > 30:
                    continue
                if not (app[i].any() and app[j].any()):
                    continue
                dist = 1.0 - float(np.dot(app[i], app[j]))
                sc = max(20.0, 0.5*(s[i]["h"]+s[j]["h"]))
                fx = s[i]["p_tail"][0] + s[i]["v_tail"][0]*g
                fy = s[i]["p_tail"][1] + s[i]["v_tail"][1]*g
                e = math.dist((fx, fy), s[j]["p_head"]) / sc
                (same if e < 0.6 else diff).append(dist)
        if same and diff:
            print(f"  appearance distance: motion-consistent pairs "
                  f"median={np.median(same):.3f} p90={np.percentile(same,90):.3f} (n={len(same)})")
            print(f"                       motion-INconsistent pairs "
                  f"median={np.median(diff):.3f} p90={np.percentile(diff,90):.3f} (n={len(diff)})")
            print(f"  -> separation { (np.median(diff)-np.median(same)):+.3f}"
                  f"  (useless if ~0; useful if > ~0.15)")
