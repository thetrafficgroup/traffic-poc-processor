#!/usr/bin/env python3
"""Why an arm moved: net volume error and entry count, not just cell L1.

L1 over movement cells can improve for the wrong reason -- a phantom in one cell
cancelling a miss in another. Net volume error has a sign, so over- and under-count
are distinguishable, and `entries` says how many distinct vehicles the arm believes
crossed an entry line, which is the quantity association is supposed to fix.
"""
import collections, copy, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import run_exp as E

SITES = [s for s in json.load(open(f"{A.ROOT}/round1_out/sites.json"))
         if s["site"] in sys.argv[1].split(",")]
params = dict(A.DEFAULTS); params["zone_k"] = 0.3

print(f"{'site':18}{'manual':>7}{'arm':>12}{'ours':>7}{'net%':>8}{'L1%':>8}{'entries':>9}")
agg = collections.defaultdict(lambda: [0, 0, 0, 0])
for s in SITES:
    name = s["site"]
    manual = E.SC.manual(name, s["gt"]); M = sum(manual.values())
    out, _, info = E.run_site(s, params, impute=False, keep=True)
    trk = f"{A.ROOT}/round1_out/{name}/{name}.tracks.csv.gz"
    lines = f"{A.ROOT}/gt/{s['lines']}"
    fps = float(s.get("fps") or 15.0)
    states = {
        "H1":     E._h1_state(trk, lines, fps, True),
        "H1-dd":  E._h1_state(trk, lines, fps, False),
        "AD+H1":  E._h1_state(f"{E.OUT}/{name}.merged.csv.gz", lines, fps, False),
        "A+H1":   E._h1_state(f"{E.OUT}/{name}.merged.csv.gz", lines, fps, True),
    }
    for arm, st in states.items():
        cells = A.cells_of(st)
        o = sum(cells.values()); l1 = A.l1(cells, manual); en = len(st["entry_counted_ids"])
        print(f"{name:18}{M:7}{arm:>12}{o:7}{100*(o-M)/M:+7.1f}%{100*l1/M:7.2f}%{en:9}")
        a = agg[arm]; a[0] += M; a[1] += o; a[2] += l1; a[3] += en
    print()
print(f"{'POOLED':18}{'':7}{'arm':>12}{'ours':>7}{'net%':>8}{'L1%':>8}{'entries':>9}")
for arm, (M, o, l1, en) in agg.items():
    print(f"{'':18}{M:7}{arm:>12}{o:7}{100*(o-M)/M:+7.1f}%{100*l1/M:7.2f}%{en:9}")
