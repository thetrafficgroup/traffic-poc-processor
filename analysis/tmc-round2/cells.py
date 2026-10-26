#!/usr/bin/env python3
"""Per-cell error for the sites that hold the error, AFTER the geometry fix.

A residual concentrated in one or two movements points at geometry still being wrong
(a line in the wrong place, a leg that is not where it is assumed to be). A residual
spread evenly across every cell points at detection or tracking, which no config
change will repair. The two call for completely different work, so it is worth four
minutes to tell them apart.
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import run_exp as E
import geoaudit as G

SC = E.SC
FIX = {"1022_meadowbranch": "rot0+mirror", "sanford": "rot180"}
PERM = dict(G.perms())

for site in ("1022_meadowbranch", "sanford", "md108", "york_night"):
    s = [x for x in json.load(open(f"{A.ROOT}/round1_out/sites.json"))
         if x["site"] == site][0]
    trk = f"{A.ROOT}/round1_out/{site}/{site}.tracks.csv.gz"
    lines = f"{A.ROOT}/gt/{s['lines']}"
    fps = float(s.get("fps") or 15.0)
    manual = SC.manual(site, s["gt"]); M = sum(manual.values())
    st = E._h1_state(trk, lines, fps, True)
    cells = A.cells_of(st)
    tag = "as shipped"
    if site in FIX:
        cells = G.apply_perm(cells, PERM[FIX[site]])
        tag = f"after {FIX[site]}"
    l1 = A.l1(cells, manual)
    keys = sorted(set(cells) | set(manual), key=lambda k: -abs(cells.get(k,0)-manual.get(k,0)))
    print(f"\n=== {site}  manual={M}  ours={sum(cells.values())}  "
          f"L1={100*l1/M:.2f}%  ({tag})")
    print(f"  {'cell':22}{'manual':>8}{'ours':>7}{'diff':>7}{'share of site err':>19}")
    acc = 0
    for k in keys[:8]:
        d = cells.get(k,0) - manual.get(k,0)
        acc += abs(d)
        print(f"  {k:22}{manual.get(k,0):8}{cells.get(k,0):7}{d:+7}{100*abs(d)/l1:18.0f}%")
    print(f"  {'(top 8 of %d cells)'%len(keys):22}{'':8}{'':7}{'':7}{100*acc/l1:18.0f}%")
