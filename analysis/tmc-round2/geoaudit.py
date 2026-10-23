#!/usr/bin/env python3
"""How much of the corpus error is leg-label geometry rather than algorithm?

The movement matrix is (origin leg) x (turn), and the turn is a LOOKUP on compass
names, not a measured angle. So a camera whose legs are labelled with the wrong
rotation or a mirror produces perfectly tracked vehicles filed in the wrong cells --
and the two documented cases (meadowbranch EAST running due west per OSM, sanford's
180 degree compass) are exactly that.

For each site this searches the 8 rigid relabelings of a 4-cycle (4 rotations x 2
reflections) and reports the best. That is fitted to the manual counts, so the number
is an UPPER BOUND, not an achievable score -- but the search space is 8 discrete
options whose correct answer is fixed by physical geometry, so a site where one
specific relabeling halves the error is evidence of a real config bug, not of
overfitting. Production fixes must come from OSM/imagery, never from this search.
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import run_exp as E

SC = E.SC
CYCLE = ["NORTH", "EAST", "SOUTH", "WEST"]


def perms():
    out = []
    for k in range(4):
        for mirror in (False, True):
            p = {}
            for i, leg in enumerate(CYCLE):
                j = ((-i if mirror else i) + k) % 4
                p[leg] = CYCLE[j]
            name = f"rot{k*90}" + ("+mirror" if mirror else "")
            out.append((name, p))
    return out


def apply_perm(cells, p):
    out = collections.Counter()
    for k, n in cells.items():
        leg, turn = k.split("|")
        dest = SC.INV.get((leg, turn))
        if dest is None:
            out[k] += n
            continue
        l2, d2 = p.get(leg, leg), p.get(dest, dest)
        t2 = SC.TRANS.get((l2, d2))
        out[f"{l2}|{t2 if t2 else turn}"] += n
    return out


sites = json.load(open(f"{A.ROOT}/round1_out/sites.json"))
print(f"{'site':18}{'manual':>7}{'shipped':>9}{'best relabel':>14}{'gain pp':>9}"
      f"{'err veh saved':>15}")
tot_m = tot_base = tot_best = 0
rows = []
for s in sites:
    name = s["site"]
    trk = f"{A.ROOT}/round1_out/{name}/{name}.tracks.csv.gz"
    if name in E.SKIP or not os.path.isfile(trk):
        continue
    lines = f"{A.ROOT}/gt/{s['lines']}"
    fps = float(s.get("fps") or 15.0)
    manual = SC.manual(name, s["gt"]); M = sum(manual.values())
    st = E._h1_state(trk, lines, fps, True)
    cells = A.cells_of(st)
    base = A.l1(cells, manual)
    best, bname = base, "identity"
    for pname, p in perms():
        l1 = A.l1(apply_perm(cells, p), manual)
        if l1 < best:
            best, bname = l1, pname
    rows.append((name, M, base, best, bname))
    tot_m += M; tot_base += base; tot_best += best
for name, M, base, best, bname in sorted(rows, key=lambda r: -(r[2] - r[3])):
    g = 100 * (base - best) / M
    print(f"{name:18}{M:7}{100*base/M:8.2f}%{bname:>14}{g:8.2f}%{base-best:15}")
print(f"\nPOOLED  shipped {100*tot_base/tot_m:.2f}%   "
      f"all-sites-best-relabel {100*tot_best/tot_m:.2f}%   "
      f"gain {100*(tot_base-tot_best)/tot_m:.2f}pp   "
      f"({tot_base-tot_best} of {tot_base} error vehicles = "
      f"{100*(tot_base-tot_best)/tot_base:.0f}% of ALL error)")
