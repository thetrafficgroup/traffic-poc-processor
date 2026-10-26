#!/usr/bin/env python3
"""Score zones (point 1) and offline association (point 2) on every cached GT site.

Arms are NESTED so a delta is attributable to one change:

  B0          shipped: online IDs, line crossing          (pre-H1 reference)
  H1          + terminal-crossing recovery                (what dev runs as of today)
  H1+Z        + zone destination for what H1 still missed
  H1+Z+O      + zone origin AND destination for orphan tracklets
  A           offline association, line crossing, no H1    (does assoc subsume H1?)
  A+H1        offline association + H1
  A+H1+Z      offline association + H1 + zone destination
  A+H1+Z+O    everything

1021_corpcenter is excluded: its clip is 3 h against a 1 h manual count (174% L1), and
the reference LOSO run excluded it for the same reason. Meadow Branch is NOT relabelled,
matching the reference, because the EAST/WEST swap is a prod decision.
"""
import argparse
import collections
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import pathfit

ROOT = A.ROOT
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "merged")
os.makedirs(OUT, exist_ok=True)
h1 = __import__("h1")
SC = __import__("score")

SKIP = {"1021_corpcenter"}
ARMS = ["B0", "H1", "H1+P", "H1+P12", "H1+P6"]


def score_state(st, manual, impute):
    cells = A.imputed(A.cells_of(st), len(st["entry_counted_ids"]), impute)
    return A.l1(cells, manual), sum(cells.values())


def _h1_state(trk, lines, fps, dedup):
    """H1 over a stream. replay() re-seeds every tunable from its own _DEFAULTS at
    call time, and h1.recover() owns the replay, so dedup is changed there."""
    keep = (A.replay_mod._DEFAULTS["_DEDUP_PX"], A.replay_mod._DEFAULTS["_DEDUP_FRAMES"])
    if not dedup:
        A.replay_mod._DEFAULTS["_DEDUP_PX"] = 0
        A.replay_mod._DEFAULTS["_DEDUP_FRAMES"] = 0
    try:
        h1.recover(trk, lines, mode="extrapolate", secs=1.0, fps=fps, min_speed_px=1.0)
        return A.replay_mod.LAST_STATE
    finally:
        (A.replay_mod._DEFAULTS["_DEDUP_PX"],
         A.replay_mod._DEFAULTS["_DEDUP_FRAMES"]) = keep


def run_site(s, params, impute, keep):
    """B0 / H1 references, then path-fit attribution at three reach limits.

    P decides a destination from the trajectory's own final heading when the crossing
    test produced nothing, with a null option. The reach is swept because it is the one
    parameter that could buy a win merely by guessing further.
    """
    name = s["site"]
    trk = f"{ROOT}/round1_out/{name}/{name}.tracks.csv.gz"
    lines = f"{ROOT}/gt/{s['lines']}"
    fps = float(s.get("fps") or 15.0)
    manual = SC.manual(name, s["gt"])
    LINES = A.replay_mod.build_lines(lines)
    CT = A.replay_mod.CLASSIFY_TURN
    tj = pathfit.traj_from_stream(trk)
    out, info = {}, {}

    A.replay_mod.replay(trk, lines)
    out["B0"] = score_state(A.replay_mod.LAST_STATE, manual, impute)

    for arm, reach in (("H1+P", 12.0), ("H1+P12", 24.0), ("H1+P6", 6.0)):
        st = _h1_state(trk, lines, fps, dedup=True)
        if arm == "H1+P":
            out["H1"] = score_state(st, manual, impute)
            info["lost_before"] = len([i for i in st["entry_counted_ids"]
                                       if i not in st["turn_types_by_id"]])
        added, rejected = pathfit.assign(st, tj, LINES, CT, fps, max_reach_bh=reach)
        out[arm] = score_state(st, manual, impute)
        info[f"added_{arm}"] = added
        if arm == "H1+P":
            info["rejected"] = rejected
    return out, sum(manual.values()), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--impute", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="recompute merged streams")
    ap.add_argument("--only", default="")
    ap.add_argument("--max-gap-s", type=float, default=1.5)
    ap.add_argument("--pos-tol", type=float, default=2.0)
    ap.add_argument("--cost-max", type=float, default=1.6)
    ap.add_argument("--zone-k", type=float, default=0.3)
    ap.add_argument("--json-out", default="")
    a = ap.parse_args()

    params = dict(A.DEFAULTS)
    params.update(max_gap_s=a.max_gap_s, pos_tol=a.pos_tol, cost_max=a.cost_max,
                  zone_k=a.zone_k)

    sites = [s for s in json.load(open(f"{ROOT}/round1_out/sites.json"))
             if s["site"] not in SKIP
             and os.path.isfile(f"{ROOT}/round1_out/{s['site']}/{s['site']}.tracks.csv.gz")
             and (not a.only or s["site"] in a.only.split(","))]

    res, man, infos = {}, {}, {}
    t0 = time.time()
    for s in sites:
        o, M, info = run_site(s, params, a.impute, keep=not a.fresh)
        res[s["site"]], man[s["site"]], infos[s["site"]] = o, M, info
        print(f"  .. {s['site']:20} {time.time()-t0:6.1f}s", file=sys.stderr)

    print(f"\nestimate {'ON' if a.impute else 'OFF'}   "
          f"max_gap={params['max_gap_s']}s pos_tol={params['pos_tol']}bh "
          f"cost_max={params['cost_max']} zone_k={params['zone_k']}")
    print(f"{'site':18}{'manual':>7}" + "".join(f"{x:>11}" for x in ARMS))
    for name in sorted(res, key=lambda n: -res[n]["B0"][0] / man[n]):
        M = man[name]
        print(f"{name:18}{M:7}" + "".join(
            f"{100*res[name][x][0]/M:10.2f}%" for x in ARMS))
    print(f"{'POOLED':18}{sum(man.values()):7}", end="")
    pooled = {}
    for x in ARMS:
        p = 100 * sum(res[n][x][0] for n in res) / sum(man.values())
        pooled[x] = p
        print(f"{p:10.2f}%", end="")
    print()
    print(f"{'vs B0':18}{'':7}" + "".join(
        f"{pooled[x]-pooled['B0']:+10.2f} " for x in ARMS))
    print(f"{'vs H1 (shipped)':18}{'':7}" + "".join(
        f"{pooled[x]-pooled['H1']:+10.2f} " for x in ARMS))
    print(f"\n{'worst site regression vs H1':34}")
    for x in ARMS:
        worst, ws = 0.0, "-"
        for n in res:
            reg = 100 * (res[n][x][0] - res[n]["H1"][0]) / man[n]
            if reg > worst:
                worst, ws = reg, n
        print(f"  {x:12} {worst:+7.2f} pp  ({ws})")
    print("\npath-fit additions per site (reach 12 bh), out of the lost population")
    for n in sorted(infos):
        i = infos[n]
        lb = i.get("lost_before", 0)
        print(f"  {n:18} lost={lb:5}  added={i.get('added_H1+P',0):4}"
              f"  ({100*i.get('added_H1+P',0)/max(1,lb):4.0f}% of lost)"
              f"  rejected={i.get('rejected',0):4}")
    if a.json_out:
        json.dump(dict(res={k: {a2: list(v) for a2, v in o.items()} for k, o in res.items()},
                       manual=man, infos=infos, pooled=pooled, params=params),
                  open(a.json_out, "w"), indent=1)


if __name__ == "__main__":
    main()
