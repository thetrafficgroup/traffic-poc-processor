#!/usr/bin/env python3
"""Stack the fixes the diagnosis points at, and score each one on its own.

  A  shipped H1
  B  A + the two leg-label fixes (meadowbranch mirror, sanford 180)   [config only]
  C  B + de-dup window expressed in SECONDS rather than frames        [code, 1 line]
  D  C + H1's dead-track window in seconds too                        [code, 1 line]

C is the interesting one: `_DEDUP_FRAMES = 30` is a frame count applied to cameras
running 10, 15 and 25 fps, so the window it actually enforces is 3.0 s, 2.0 s or 1.2 s
depending on the camera. Expressing it in seconds leaves every 15 fps site bit-identical
and only changes the cameras where the current behaviour was never intended.
"""
import collections, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import assoc as A
import run_exp as E
import geoaudit as G

SC = E.SC
PERM = dict(G.perms())
FIX = {"1022_meadowbranch": "rot0+mirror", "sanford": "rot180"}
DEDUP_S = 2.0
H1_DEAD_S = 2.0
ARMS = ["A shipped", "B +geometry", "C +dedup_cap"]


def state(trk, lines, fps, dedup_frames):
    kp = A.replay_mod._DEFAULTS["_DEDUP_FRAMES"]
    A.replay_mod._DEFAULTS["_DEDUP_FRAMES"] = dedup_frames
    try:
        return E._h1_state(trk, lines, fps, True)
    finally:
        A.replay_mod._DEFAULTS["_DEDUP_FRAMES"] = kp


rows = []
for s in json.load(open(f"{A.ROOT}/round1_out/sites.json")):
    name = s["site"]
    trk = f"{A.ROOT}/round1_out/{name}/{name}.tracks.csv.gz"
    if name in E.SKIP or not os.path.isfile(trk):
        continue
    lines = f"{A.ROOT}/gt/{s['lines']}"
    fps = float(s.get("fps") or 15.0)
    manual = SC.manual(name, s["gt"]); M = sum(manual.values())
    perm = PERM.get(FIX.get(name))

    def l1_of(st, use_perm):
        c = A.cells_of(st)
        if use_perm and perm:
            c = G.apply_perm(c, perm)
        return A.l1(c, manual)

    base = state(trk, lines, fps, 30)
    a = l1_of(base, False)
    b = l1_of(base, True)
    # min(), never max(): the window may only SHORTEN relative to the shipped 30
    # frames, so 15 and 25 fps cameras are bit-identical and only the sub-15 fps ones
    # (where 30 frames silently meant 3 s) change at all.
    c = l1_of(state(trk, lines, fps, min(30, round(DEDUP_S * fps))), True)
    rows.append((name, M, fps, a, b, c))

print(f"{'site':18}{'manual':>7}{'fps':>5}" + "".join(f"{x:>13}" for x in ARMS))
for name, M, fps, *v in sorted(rows, key=lambda r: -r[3]):
    print(f"{name:18}{M:7}{fps:5.0f}" + "".join(f"{100*x/M:12.2f}%" for x in v))
M = sum(r[1] for r in rows)
tot = [sum(r[3 + i] for r in rows) for i in range(3)]
print(f"{'POOLED':18}{M:7}{'':5}" + "".join(f"{100*t/M:12.2f}%" for t in tot))
print(f"{'vs shipped':18}{'':7}{'':5}" + "".join(f"{100*(t-tot[0])/M:+12.2f} " for t in tot))
print(f"{'error vehicles':18}{'':7}{'':5}" + "".join(f"{t:12} " for t in tot))
print("\nworst-site regression vs shipped")
for i, nm in enumerate(ARMS):
    worst, ws = 0.0, "-"
    for r in rows:
        reg = 100 * (r[3 + i] - r[3]) / r[1]
        if reg > worst:
            worst, ws = reg, r[0]
    print(f"  {nm:14} {worst:+7.2f} pp  ({ws})")
