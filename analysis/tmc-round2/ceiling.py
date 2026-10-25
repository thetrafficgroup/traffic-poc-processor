#!/usr/bin/env python3
"""Is the evidence there at all? Ceiling test for offline association.

For every track that was entry-counted but never classified -- the population every
attribution fix targets -- ask whether ANY later tracklet could plausibly be its
continuation, under gates far looser than anything shippable (4 s gap, 6 box-heights
of prediction error, 90 degrees of heading change). If a permissive search finds no
candidate, the vehicle simply stopped being detected, and no association scheme --
online, offline, global or otherwise -- can recover it.
"""
import collections, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import run_exp as E


def _seg(px, py, a, b):
    ax, ay = a; bx, by = b
    dx, dy = bx - ax, by - ay
    t = 0.0 if dx == dy == 0 else max(0, min(1, ((px-ax)*dx + (py-ay)*dy)/(dx*dx+dy*dy)))
    return math.hypot(px - (ax + t*dx), py - (ay + t*dy))

LOOSE = dict(A.DEFAULTS)
LOOSE.update(max_gap_s=4.0, pos_tol=6.0, cos_min=-0.1, size_ratio=4.0, cost_max=1e9)

print(f"{'site':18}{'unclass':>8}{'has cand':>9}{'%':>6}{'died<1bh of line':>18}{'in-frame':>9}")
tot = collections.Counter()
for s in [x for x in json.load(open(f"{A.ROOT}/round1_out/sites.json"))
          if x["site"] in sys.argv[1].split(",")]:
    name = s["site"]
    trk = f"{A.ROOT}/round1_out/{name}/{name}.tracks.csv.gz"
    lines = f"{A.ROOT}/gt/{s['lines']}"
    fps = float(s.get("fps") or 15.0)
    W, H = (int(v) for v in s["res"].split("x"))
    rows, tl = A.load(trk)
    st = E._h1_state(trk, lines, fps, dedup=True)
    summ = A.summarise(tl)
    edges = A.candidates(summ, fps, LOOSE)
    has_succ = collections.defaultdict(list)
    for i, j, c in edges:
        has_succ[i].append((c, j))
    LINES = A.replay_mod.build_lines(lines)

    unclass = [i for i in st["entry_counted_ids"] if i not in st["turn_types_by_id"]]
    cand = near = inframe = 0
    for i in unclass:
        if i not in summ:
            continue
        a = summ[i]
        if has_succ.get(i):
            cand += 1
        px, py = a["p_tail"]
        h = max(20.0, a["h"])
        d = min(min(_seg(px, py, p, q) for p, q in L["segments"]) for L in LINES)
        if d <= h:
            near += 1
        if 0.05 * W < px < 0.95 * W and 0.05 * H < py < 0.95 * H:
            inframe += 1
    n = max(1, len(unclass))
    print(f"{name:18}{len(unclass):8}{cand:9}{100*cand/n:5.0f}%{near:18}{inframe:9}")
    tot["u"] += len(unclass); tot["c"] += cand; tot["n"] += near; tot["f"] += inframe
print(f"{'POOLED':18}{tot['u']:8}{tot['c']:9}{100*tot['c']/max(1,tot['u']):5.0f}%"
      f"{tot['n']:18}{tot['f']:9}")
