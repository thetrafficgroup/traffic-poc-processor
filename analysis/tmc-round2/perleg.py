#!/usr/bin/env python3
"""Split each leg's error into DETECTION vs ATTRIBUTION.

manual_row  vehicles the counters saw entering on this leg
entries     distinct tracks our pipeline counted crossing that leg inbound
classified  of those, how many got a movement

entries << manual_row  -> the vehicles were never seen: a detection problem, and no
                         amount of counting logic will recover them.
entries ~ manual_row but classified << entries -> they were seen and lost at
                         attribution: a geometry / crossing-logic problem.
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import run_exp as E
import geoaudit as G

SC = E.SC
FIX = {"1022_meadowbranch": "rot0+mirror", "sanford": "rot180"}
PERM = dict(G.perms())

for site in ("1022_meadowbranch", "york_night", "md108", "sanford"):
    s = [x for x in json.load(open(f"{A.ROOT}/round1_out/sites.json"))
         if x["site"] == site][0]
    trk = f"{A.ROOT}/round1_out/{site}/{site}.tracks.csv.gz"
    lines = f"{A.ROOT}/gt/{s['lines']}"
    fps = float(s.get("fps") or 15.0)
    manual = SC.manual(site, s["gt"])
    st = E._h1_state(trk, lines, fps, True)

    # Rows are keyed by the leg the vehicle ENTERED on, before any relabelling, so the
    # permutation has to be applied to the manual side to compare like with like.
    perm = PERM.get(FIX.get(site), {})
    inv = {v: k for k, v in perm.items()} if perm else {}
    man_row = collections.Counter()
    for k, n in manual.items():
        leg = k.split("|")[0]
        man_row[inv.get(leg, leg)] += n

    ent = collections.Counter()
    cls = collections.Counter()
    for oid in st["entry_counted_ids"]:
        ts = st["crossing_timestamps"].get(oid)
        if not ts:
            continue
        leg = ts[0][0].upper()
        ent[leg] += 1
        if oid in st["turn_types_by_id"]:
            cls[leg] += 1
    print(f"\n=== {site}" + (f"   (manual rows mapped through {FIX[site]})" if perm else ""))
    print(f"  {'leg':8}{'manual_row':>12}{'entries':>9}{'classified':>12}"
          f"{'det gap':>9}{'attr gap':>10}{'verdict':>14}")
    for leg in sorted(set(man_row) | set(ent)):
        mr, e, c = man_row.get(leg, 0), ent.get(leg, 0), cls.get(leg, 0)
        det = e - mr
        attr = c - e
        if mr and e < 0.85 * mr:
            v = "DETECTION"
        elif e and c < 0.9 * e:
            v = "attribution"
        else:
            v = "ok"
        print(f"  {leg:8}{mr:12}{e:9}{c:12}{det:+9}{attr:+10}{v:>14}")
