#!/usr/bin/env python3
"""Score the offline tracker against the manual counts, on the shipped counting code.

Arms, all scored by the SAME replay path that reproduces the reference baseline to the
decimal, so a difference here is the tracking and nothing else:

  H1        shipped: online BoT-SORT stream + terminal-crossing recovery  (reference)
  F         offline tracker, shipped crossing logic, no H1
  F+H1      offline tracker + H1
  F+H1-dd   offline tracker + H1, de-dup OFF -- de-dup exists to suppress duplicate
            entries from a fragmenting tracker, so correct association should make it
            unnecessary, and leaving it on would hide a real improvement.
"""
import argparse
import collections
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import flow
import pathfit
import run_exp as E

CLIPS = {
    "gunselford": "customer-comparison/clips2/968_gunselford.mp4",
    "rolling": "customer-comparison/clips2/973_rolling.mp4",
    "sanford": "customer-comparison/clips2/969_sanford.mp4",
    "md108": "customer-comparison/clips2/983_md108.mp4",
    "1022_meadowbranch": "customer-comparison/clips3/1022_meadowbranch.mp4",
}
S3RUN = "s3://trafficgroup-ne-data-sb/experiments/round2-dets/runs"
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.join(HERE, "flowwork")
os.makedirs(WORK, exist_ok=True)
ARMS = ["H1", "H1+P", "F", "F+P", "F+P-dd", "F+H1+P"]


def fetch(run, site):
    dest = f"{WORK}/{site}.dets.csv.gz"
    if not os.path.exists(dest):
        env = dict(os.environ)
        for line in open(f"{HERE}/awsenv.sh"):
            pass
        r = subprocess.run(
            ["bash", "-c",
             f"source {HERE}/awsenv.sh && aws s3 cp {S3RUN}/{run}/out/{site}.dets.csv.gz "
             f"{dest} --only-show-errors"],
            capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(dest):
            return None
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--sites", default=",".join(CLIPS))
    ap.add_argument("--cost-max", type=float, default=None)
    ap.add_argument("--w-app", type=float, default=None)
    ap.add_argument("--min-conf", type=float, default=None)
    ap.add_argument("--no-appearance", action="store_true")
    ap.add_argument("--tag", default="base")
    a = ap.parse_args()

    p = dict(flow.DEFAULTS)
    if a.cost_max is not None:
        p["cost_max"] = a.cost_max
    if a.w_app is not None:
        p["w_app"] = a.w_app
    if a.min_conf is not None:
        p["min_conf"] = a.min_conf

    sites = json.load(open(f"{A.ROOT}/round1_out/sites.json"))
    want = a.sites.split(",")
    res, man, st = {}, {}, {}
    for s in sites:
        name = s["site"]
        if name not in want:
            continue
        dets = fetch(a.run, name)
        if not dets:
            print(f"  .. {name}: dump not ready", file=sys.stderr)
            continue
        fps = float(s.get("fps") or 15.0)
        lines = f"{A.ROOT}/gt/{s['lines']}"
        manual = E.SC.manual(name, s["gt"])
        clip = None if a.no_appearance else f"{A.ROOT}/{CLIPS[name]}"
        cache = None if a.no_appearance else f"{WORK}/{name}.app.npy"
        stream = f"{WORK}/{name}.{a.tag}.flow.csv.gz"
        print(f"  .. {name}", file=sys.stderr)
        tstats = flow.track(dets, clip, fps, stream, p, app_cache=cache)

        trk = f"{A.ROOT}/round1_out/{name}/{name}.tracks.csv.gz"
        LINES = A.replay_mod.build_lines(lines)
        CT = A.replay_mod.CLASSIFY_TURN
        out = {}

        sh = E._h1_state(trk, lines, fps, True)
        out["H1"] = E.score_state(sh, manual, False)
        # Control: the same new attribution rule on the SHIPPED stream. Without it a
        # gain below cannot be attributed to offline tracking rather than to the rule.
        tj_on = pathfit.traj_from_stream(trk)
        st_p_added = pathfit.assign(sh, tj_on, LINES, CT, fps)[0]
        out["H1+P"] = E.score_state(sh, manual, False)

        A.replay_mod.replay(stream, lines)
        out["F"] = E.score_state(A.replay_mod.LAST_STATE, manual, False)
        tj_off = pathfit.traj_from_stream(stream)
        A.replay_mod.replay(stream, lines)
        stf = A.replay_mod.LAST_STATE
        f_added = pathfit.assign(stf, tj_off, LINES, CT, fps)[0]
        out["F+P"] = E.score_state(stf, manual, False)
        # De-dup off only matters where the tracker used to fragment under load, so it
        # is kept as its own arm rather than folded into F+P.
        A.replay_mod._DEFAULTS["_DEDUP_PX"] = 0
        A.replay_mod._DEFAULTS["_DEDUP_FRAMES"] = 0
        A.replay_mod.replay(stream, lines)
        stdd = A.replay_mod.LAST_STATE
        pathfit.assign(stdd, tj_off, LINES, CT, fps)
        out["F+P-dd"] = E.score_state(stdd, manual, False)
        A.replay_mod._DEFAULTS["_DEDUP_PX"] = 60
        A.replay_mod._DEFAULTS["_DEDUP_FRAMES"] = 30

        stf2 = E._h1_state(stream, lines, fps, True)
        fh_added = pathfit.assign(stf2, tj_off, LINES, CT, fps)[0]
        out["F+H1+P"] = E.score_state(stf2, manual, False)
        st_diag = {}
        for arm, sx in (("H1", sh), ("F+P", stf)):
            en = len(sx["entry_counted_ids"])
            un = len([i for i in sx["entry_counted_ids"] if i not in sx["turn_types_by_id"]])
            st_diag[arm] = (en, un)

        st_stats = dict(path_on_online=st_p_added, path_on_offline=f_added,
                        path_on_offline_h1=fh_added, diag=st_diag)
        res[name], man[name] = out, sum(manual.values())
        tstats.update(st_stats)
        st[name] = tstats

    if not res:
        print("no sites scored")
        return
    print(f"\noffline tracker [{a.tag}]  cost_max={p['cost_max']} w_app={p['w_app']} "
          f"min_conf={p['min_conf']} appearance={'off' if a.no_appearance else 'on'}")
    print(f"{'site':18}{'manual':>7}" + "".join(f"{x:>11}" for x in ARMS) + f"{'chains':>9}")
    for n in sorted(res, key=lambda k: -res[k]["H1"][0] / man[k]):
        print(f"{n:18}{man[n]:7}"
              + "".join(f"{100*res[n][x][0]/man[n]:10.2f}%" for x in ARMS)
              + f"{st[n]['final_chains']:9}")
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
            reg = 100 * (res[n][x][0] - res[n]["H1"][0]) / man[n]
            if reg > worst:
                worst, ws = reg, n
        print(f"  {x:10} {worst:+7.2f} pp  ({ws})")
    print("\npath-fit additions (origin known, destination from trajectory)")
    for n in sorted(st):
        print(f"  {n:18} on online={st[n].get('path_on_online',0):4} "
              f"on offline={st[n].get('path_on_offline',0):4}")
    print("\nentries / still-unclassified  (the population every attribution fix targets)")
    for n in sorted(st):
        dg = st[n].get("diag", {})
        if dg:
            print(f"  {n:18} H1: {dg['H1'][0]:5} entries {dg['H1'][1]:4} lost   "
                  f"|  F+P: {dg['F+P'][0]:5} entries {dg['F+P'][1]:4} lost")
    print("\nvehicles filed vs manual")
    for n in sorted(res):
        print(f"  {n:18} manual={man[n]:5} " + " ".join(
            f"{x}={res[n][x][1]}" for x in ARMS))
    json.dump(dict(res={k: {a2: list(v) for a2, v in o.items()} for k, o in res.items()},
                   manual=man, stats=st, params={k: v for k, v in p.items()
                                                 if k != "rounds"}),
              open(f"{WORK}/score.{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
