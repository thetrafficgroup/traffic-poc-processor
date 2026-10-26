#!/usr/bin/env python3
"""Are the counting lines in the wrong place? Perturb them and re-score.

Of 1168 lost tracks, 29% left the frame with no exit crossing (md108 alone: 63) and 9%
passed a line's EXTENSION outside its segment (honeygo_pm: 42 of 54). Both are geometry
defects rather than tracking failures, and both are cheap to test: the streams are
recorded, so the lines can be moved and the shipped counting code re-run.

Two perturbations, each expressed in units of the line's OWN length so that nothing
here is a pixel constant and a single setting means the same thing on every camera:

  extend  scale each line about its midpoint. extend=1.5 makes a line half again as
          long, which is the fix for a segment that does not span the roadway.
  shift   move each line along its own normal, signed so that negative is TOWARDS the
          intersection centre. Pulling exit lines inward lets a vehicle register its
          exit before the track dies or leaves the frame.

Both are global: one value for all 18 sites, judged on pooled gain AND worst-site
regression, because a per-camera value would just be refitting the corpus.
"""
import argparse
import collections
import json
import math
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import geoaudit as G
import run_exp as E

SC = E.SC
PERM = dict(G.perms())
FIX = {"1022_meadowbranch": "rot0+mirror", "sanford": "rot180"}
DEDUP_S = 2.0


def perturb(cfg_path, extend, shift, out_path):
    data = json.load(open(cfg_path))
    raw = dict(data.get("lines", data))
    cross = raw.pop("crosswalks", None)
    legs = {}
    for name, entry in raw.items():
        pts = entry.get("points") or [entry["pt1"], entry["pt2"]]
        legs[name] = [tuple(p) for p in pts]
    allp = [p for v in legs.values() for p in (v[0], v[-1])]
    cx = sum(p[0] for p in allp) / len(allp)
    cy = sum(p[1] for p in allp) / len(allp)

    out = {}
    for name, pts in legs.items():
        p1, p2 = pts[0], pts[-1]
        L = math.dist(p1, p2)
        if L < 1:
            out[name] = {"points": [list(p1), list(p2)]}
            continue
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ux, uy = (p2[0] - p1[0]) / L, (p2[1] - p1[1]) / L
        half = L * extend / 2.0
        nx, ny = -uy, ux
        ox, oy = mid[0] - cx, mid[1] - cy
        if nx * ox + ny * oy < 0:          # make the normal point AWAY from the centre
            nx, ny = -nx, -ny
        d = shift * L
        q1 = (mid[0] - ux * half + nx * d, mid[1] - uy * half + ny * d)
        q2 = (mid[0] + ux * half + nx * d, mid[1] + uy * half + ny * d)
        out[name] = {"points": [[round(q1[0], 1), round(q1[1], 1)],
                                [round(q2[0], 1), round(q2[1], 1)]]}
    if cross is not None:
        out["crosswalks"] = cross
    json.dump(out, open(out_path, "w"))
    return out_path


def score(site_rows, extend, shift, tmpdir):
    tot_l1 = tot_m = 0
    per = {}
    for name, s, trk, cfg, fps, manual in site_rows:
        M = sum(manual.values())
        p = perturb(cfg, extend, shift, f"{tmpdir}/{name}.json")
        keep = A.replay_mod._DEFAULTS["_DEDUP_FRAMES"]
        A.replay_mod._DEFAULTS["_DEDUP_FRAMES"] = min(30, round(DEDUP_S * fps))
        try:
            st = E._h1_state(trk, p, fps, True)
        finally:
            A.replay_mod._DEFAULTS["_DEDUP_FRAMES"] = keep
        cells = A.cells_of(st)
        if name in FIX:
            cells = G.apply_perm(cells, PERM[FIX[name]])
        l1 = A.l1(cells, manual)
        per[name] = (l1, M)
        tot_l1 += l1
        tot_m += M
    return tot_l1, tot_m, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    rows = []
    for s in json.load(open(f"{A.ROOT}/round1_out/sites.json")):
        name = s["site"]
        trk = f"{A.ROOT}/round1_out/{name}/{name}.tracks.csv.gz"
        if name in E.SKIP or not os.path.isfile(trk):
            continue
        if a.only and name not in a.only.split(","):
            continue
        rows.append((name, s, trk, f"{A.ROOT}/gt/{s['lines']}",
                     float(s.get("fps") or 15.0), SC.manual(name, s["gt"])))

    grid = [(1.0, 0.0)]
    grid += [(e, 0.0) for e in (1.25, 1.5, 2.0)]
    grid += [(1.0, sh) for sh in (-0.3, -0.15, 0.15)]
    grid += [(1.5, -0.15), (1.5, -0.3), (2.0, -0.15)]

    with tempfile.TemporaryDirectory() as td:
        base_l1, base_m, base_per = score(rows, 1.0, 0.0, td)
        print(f"baseline (package C, lines unchanged): {100*base_l1/base_m:.2f}%  "
              f"{base_l1} error vehicles, {base_m} manual\n")
        print(f"{'extend':>8}{'shift':>8}{'pooled':>9}{'vs base':>9}{'veh':>6}"
              f"{'worst site regression':>26}")
        for e, sh in grid:
            l1, m, per = score(rows, e, sh, td)
            worst, ws = 0.0, "-"
            for n, (v, M) in per.items():
                reg = 100 * (v - base_per[n][0]) / M
                if reg > worst:
                    worst, ws = reg, n
            flag = "  PASS" if (l1 < base_l1 and worst <= 0.5) else ""
            print(f"{e:8.2f}{sh:8.2f}{100*l1/m:8.2f}%{100*(l1-base_l1)/m:+9.2f}"
                  f"{l1:6}{worst:+10.2f} pp ({ws[:12]}){flag}")


if __name__ == "__main__":
    main()
