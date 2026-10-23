# TMC round 2 — where the counting error actually is

2026-10-22/23. Scored on 18 GT sites, 15,150 manually counted vehicles, using the
shipped counting code driven from recorded observation streams (`proc-iter-tmc`
`round1_out/*/\*.tracks.csv.gz` replayed through `process_single_detection`, which
reproduces the reference baseline to the decimal).

`1021_corpcenter` is excluded throughout: its clip is 3 h against a 1 h manual count
(174% L1). Metric is `L1 = sum|ours - manual|` over `(origin leg) x (turn)` cells,
with the attribution estimate OFF (the shipped default now that H1 is on).

## Result

| arm | pooled L1 | vs shipped | error vehicles | worst site |
|---|---|---|---|---|
| A — shipped H1 | 7.35% | +0.00 | 1113 | +0.00 |
| B — A + two leg-label fixes | 5.35% | −1.99 | 811 | +0.00 |
| C — B + de-dup window cap | **5.22%** | **−2.13** | 791 | +0.00 |

322 of 1113 error vehicles removed, 29% of all counting error, with no site regressing.

## The error is concentrated, and mostly not algorithmic

74% of the error sits in four sites:

| site | share of all error | cause |
|---|---|---|
| 1022_meadowbranch | 37.1% | leg labels **mirrored** |
| md108 | 13.7% | attribution loss (188 lost tracks) |
| sanford | 13.2% | leg labels rotated **180°** |
| york_night | 9.6% | detection blindness on **one** approach |

### Leg-label geometry — 302 vehicles, 27% of all error

The movement matrix is (origin leg) x (turn), and the turn is a *lookup on compass
names*, not a measured angle. So a camera with the wrong labels files perfectly tracked
vehicles into the wrong cells.

Searching the 8 rigid relabelings of a 4-cycle (4 rotations x 2 reflections) per site:

| site | shipped | best relabel | gain | vehicles |
|---|---|---|---|---|
| 1022_meadowbranch | 26.04% | `rot0+mirror` | −13.37pp | 212 |
| sanford | 18.89% | `rot180` | −11.57pp | 90 |
| the other 16 sites | — | `identity` | 0.00pp | 0 |

Sixteen sites returning `identity` is what makes this a bug hunt rather than a fit.
Both flagged sites were already documented independently: meadowbranch's "EAST" leg
runs 275.6° per OSM, and sanford's 180° compass was proven earlier via OSM plus a dev
`rot180` experiment (turn accuracy 79.6 → 90.4%).

**These are prod config changes, not code.** Verify against imagery/OSM before applying.

### De-dup window — 20 vehicles

`_DEDUP_FRAMES = 30` was a frame count applied to a fleet running 10, 15 and 25 fps,
so the window it enforced was 3.0 s, 2.0 s or 1.2 s depending on the camera.
meadowbranch is 10 fps with a 786 veh/h through movement — a 4.6 s headway against a
3.0 s suppression window.

| de-dup window | entries | filed | L1 |
|---|---|---|---|
| 30 frames (3.0 s, shipped) | 1479 | 1385 | 12.67% |
| 20 frames (2.0 s) | 1522 | 1405 | 11.41% |
| off | 1584 | 1423 | 10.28% |

(manual = 1586)

Fixed as `min(30, round(TMC_DEDUP_SECS * fps))`. `min()` not `max()`: 15 and 25 fps
cameras stay bit-identical, and only sub-15 fps cameras change. Turning de-dup off
globally was tested and **regresses** pooled (+0.16pp), so the cap is the shippable form.

## Fleet exposure

2 of 18 sampled cameras were mislabelled, worth 27% of the error. Manual counts exist
for 18 of ~358 customer TMC videos. If the rate holds, **~40 customer cameras are
producing confidently wrong turn matrices** with nothing in the pipeline to reveal it.

`tools/validate_lane_config.py` catches the mirrored class from the config alone, with
no ground truth: the legs' compass names must run N→E→S→W around the centre in one
handedness, which is rotation-invariant and reversed by a mirror. It flags meadowbranch
and clears the other 21 configs on disk. A 180° rotation preserves cyclic order and is
invisible to it — that class needs an OSM bearing cross-check.

## Why the "lost track" population cannot be mined

1168 tracks across 18 sites are entry-counted but never classified. Every recovery
scheme tried against it returns roughly nothing, and the reason is an information
ceiling, not weak implementation:

- only 618 vehicles of real undercount exist pooled, and **7 of 18 sites OVERCOUNT
  while still holding lost tracks** (kriders: +7 overcount, 216 lost). So at most 53%
  of lost tracks can be real missing vehicles.
- path-fit attribution recovers 160 of them at a **51% hit rate** (~82 right, ~78
  wrong), net −4 vehicles out of 15,150.
- at kriders, all 25 additions were wrong: at 0.35% error there are no missing
  vehicles left to find.

Nothing in a trajectory distinguishes "missed vehicle" from "second fragment of one
already counted", so recoveries and double-counts cancel ~1:1.

Why each one was lost:

| cause | share | meaning |
|---|---|---|
| near its entry line | 40% | fragmented immediately after being counted in |
| left the frame | 29% | no exit crossing before leaving the FOV |
| midblock | 22% | genuine detection dropout |
| short line | 9% | passed the line's *extension*, outside the segment |

The first group needs *preventing*, not recovering.

### Approaches tested and rejected

| approach | pooled vs shipped | worst site |
|---|---|---|
| Outward zones for attribution | +0.02 | +0.44 |
| Offline tracklet association (from cached IDs) | +0.30 | +2.34 |
| Offline detection-level tracker (NMS + optimal per-frame assignment + HSV appearance + coarse-to-fine global linking) | matches shipped on the sparsest site; **rolling 4.43% → 20.68%** | — |
| Path-fit attribution (H5) | −0.03 | +1.26 |
| Path-fit + appearance duplicate suppression | −0.08 (4 sites, = 3 vehicles) | +0.15 |

The from-scratch offline tracker made H1 and de-dup both no-ops at gunselford — a real
architectural point — then produced 1367 chains for 948 vehicles at rolling. Beating
BoT-SORT + ReID on dense traffic is a research project. **H3 and H5 are closed by
measurement; neither should be funded.**

## Residual, and the next levers

After the package, 791 error vehicles remain; meadowbranch (181), md108 (152),
york_night (107) and sanford (57) are 63% of it. Per-leg detection/attribution split:

- **meadowbranch** SOUTH entries 653 vs 801 manual, but detections near that line are
  large (median 40.6 px) and confident — the gap was de-dup, not blindness. Residual is
  NORTH attribution (−85) plus fragmentation at the line (86 of its 94 lost tracks).
  Vehicles are **15.6 px tall** and `imgsz = min(max(w,h),1920)//32*32` gives **640** on
  this 640x480 clip, i.e. no upscaling. Testing `imgsz=1280` is the largest open lever.
- **md108** no leg has a detection gap >15%; its whole error is attribution, consistent
  with the documented Aug-12 config edit. 63 of its lost tracks left the frame with no
  exit crossing — exit lines may sit too far inside.
- **york_night** SOUTH approach is nearly exact (127 manual / 120 filed) while NORTH
  collapses (138 / 50, 88 of them midblock). Not global night blindness — one approach.
- **honeygo_pm** 42 of 54 lost tracks are `short_line`; those lines do not span the road.

## Reproducing

Analysis code is in `analysis/tmc-round2/`. It depends on `proc-iter-tmc/`
(cached streams, GT, lane configs), which is **not** version controlled — see that
directory's README.

---

# Round 2b — four more levers, all rejected

2026-10-23. Same corpus and metric. Baseline is package C (5.22%, 791 error vehicles).

## The targeting rule, which should have come first

**A lost-track population is only worth mining where there is a measured undercount.**

honeygo_pm holds 42 lost tracks and 33 vehicles of total error; kriders holds 216 lost
tracks and 7. At a site that already counts correctly, the lost tracks are duplicate
fragments of vehicles already counted, so any scheme that "recovers" them can only add
double-counts. This one test, applied up front, would have killed zones, tracklet
association, path-fit and line extension before any of them were built.

Applying it to the whole corpus, three places still hold recoverable vehicles:

| | real undercount | nature |
|---|---|---|
| md108 (whole site) | 150 | attribution; 80% of its 188 lost tracks are real |
| york_night NORTH | 88 | detection; 66 of 83 lost are midblock |
| meadowbranch NORTH | 85 | attribution; fragmenting at the entry line |

md108 is the instructive one. It is the only site whose lost population is genuinely
mostly real, and path-fit attribution *still* only reached ~42% there. Knowing that a
vehicle is missing does not tell you where it went: it died before revealing its turn.
That is a harder and different problem from the duplicate-versus-real question, and it
is the honest reason md108 has resisted every round.

## Line geometry — rejected

`lineaudit.py` perturbs each line in units of its own length and re-scores.

Extension, pooled: 5.22% (x1.00) -> 5.28% (x1.15) -> 5.33% (x1.25) -> 5.91% (x1.50),
monotonically worse; sanford +9.38pp at x1.50. And it does not help the sites the
short-line test flagged: honeygo_pm 2.12% -> 2.50%, honeygo_am 1.80% -> 1.92%,
york_day 4.67% -> 6.07%. Longer lines reach into neighbouring legs' traffic.

The shift sweep (moving lines along their normal) produced 12,161 error vehicles
against 15,150 manual, which is a broken invariant rather than a finding: moving the
lines changes which side counts as "outside", so `is_entering_from_outside` misfires
and vehicles are re-counted. **That test is invalid and its numbers must not be
quoted.** Testing line placement properly needs the entry/exit polygon rebuilt with
the lines, not the lines moved underneath it.

The short-line taxonomy remains useful for config QA -- a camera whose counting line
does not span its roadway is worth flagging on its own merits -- but it does not
predict recoverable accuracy.

## imgsz upscaling — rejected

`imgsz = min(max(w,h),1920)//32*32` caps inference at native resolution, so
meadowbranch (640x480, vehicles 15.6 px tall) gets no upscaling. Tested end to end
with `dump_tracks.py`, which runs the real tracking path and writes the harness schema,
so the variant is scored by the shipped counting code rather than by a detection count.

| stream | obs | entries | filed | L1 (+mirror) |
|---|---|---|---|---|
| cached (reference) | 71422 | 1522 | 1405 | 11.41% |
| pod control, imgsz=640 | 72128 | 1507 | 1386 | 12.61% |
| pod, imgsz=1280 | 78097 | 1506 | 1334 | **15.89%** |

Per-leg entries (manual / @640 / @1280): SOUTH 801 / 673 / **637**, NORTH 740 / 790 /
**826**. So 8.3% more observations yielded *fewer* complete vehicles.

Upscaling a 640x480 source adds no information -- a 15.6 px vehicle becomes 31 px of
interpolated pixels. The detector fires more often on no better evidence, and the extra
detections fragment tracks and inflate the already over-counting NORTH leg.

**The control earns its keep here:** the pod's own imgsz=640 run scores 12.61% against
the cached stream's 11.41%, so roughly 0.8pp of this harness is tracker
nondeterminism plus model/ultralytics drift. Variants must be compared against that
control, never against the cached stream, and anything under ~0.8pp is noise.

## What is left

Only detection work on two approaches, plus md108's destination problem. The strongest
remaining candidate is a brightness-gated classical detector for york_night's NORTH
approach: its SOUTH approach is already near-exact (127 manual / 120 filed), so this is
one approach rather than a night model, and the I-695 precedent (light-blob tracker 374
vs 368 manual where prod scored 10) is the best evidence in the project that it works.

---

# Round 2c — fleet scan, and a result I cannot explain

2026-10-23. Night work was deprioritised part way through this round (IR cameras are
coming and the model is the binding constraint there), so everything below is daytime.

## Fleet scan: 6 of 371, not ~40

`tools/validate_lane_config.py` run over every TMC config in prod:

```
371 checked, 6 flagged MIRRORED (1.6%), 365 clean
3 distinct cameras: 405427 (id 1022 = meadowbranch), 405482 (id 999), SCU4QI (ids 65-68)
```

**The earlier claim that ~40 customer cameras were mislabelled was wrong** and should be
disregarded. It extrapolated 2-of-18 across the fleet, ignoring that the GT sample is
selection-biased: problem cameras are precisely the ones that get manually counted.
Video 1022's live prod metadata was confirmed through the API to be the mirrored config,
so the 6 are real -- just far fewer than predicted.

The 180-degree class stays undetectable. `locations` holds lat/lon for 55 rows but
**no video references a camera** (`cameraId` is NULL across all 1017), so there is no
video-to-coordinate path for an OSM bearing cross-check. That is a data-linkage gap.

Separately, `videoRate` reads 30 for meadowbranch, york, sanford and md108 whose true
rates are 10, 25, 15 and 15. Nothing should trust that column.

## No resolution headroom

The prod SOURCE videos, probed via presigned URLs, are natively 640x480 (meadowbranch,
york) and 960x540 (sanford, md108) -- exactly what we already process. Upscaling was
never viable and there is no better source to pull.

## Line placement: a real effect without an explanation

Moving a camera's counting lines inward along their normals, in units of each line's
own length, recovers a lot at two cameras:

| site | shipped | best | recovered |
|---|---|---|---|
| md108 | 8.65% | 4.27% (shift -0.050, extend 1.30) | 77 vehicles |
| 1022_meadowbranch | 11.41% | ~7.06% (shift -0.050) | ~69 vehicles |

Together ~146 vehicles, 21% of the daytime error. The symptoms move with it: md108's
leave-frame losses fall 63 -> 29, meadowbranch's near-entry-line losses 100 -> 27.

**But the mechanism is not understood, and that matters more than the number.**

* First hypothesis -- md108's vehicles leave frame before reaching an exit line --
  predicted the edge-loss drop, and that held. But it cannot explain meadowbranch,
  which has ZERO leave-frame losses.
* Second hypothesis -- shrinking the polygon shortens the distance a track must survive
  -- was falsified directly: meadowbranch has the HIGHEST survival ratio of all 18 sites
  (3.85) and benefits most.
* Neither candidate GT-free predictor correlates with the benefit: completion rate
  (classified/entries) gives **+0.04**, survival ratio gives **+0.00**. md108 (89.5%
  completion) and kriders (90.2%) respond in opposite directions.

A global shift is also rejected outright: pooled 5.22% -> 5.02% at -0.025, but kriders
regresses +2.02pp and honeygo_am +3.48pp, failing the worst-site gate.

**Recommendation: do NOT ship a shift rule.** Treat this as evidence that md108's and
meadowbranch's lines deserve human re-placement against imagery, with ~146 vehicles as
the prize and the measured optimum as a validation target. Rendering md108's lines on
its video shows nothing obviously wrong, which is itself a reason for caution.

## Tracker parameters: noise

Five BoT-SORT arms on meadowbranch (10 fps), testing whether the IoU and appearance
gates are fps-sensitive in effect:

| arm | L1 |
|---|---|
| new_track_thresh=0.5 | 12.23% |
| control | 12.67% |
| match_thresh=0.9 | 12.99% |

All inside the harness noise floor.

## The noise floor, which every future round needs

The pod's own native-resolution control scores 12.61% where the cached stream scores
11.41% on the same configuration. That ~0.8pp is tracker nondeterminism plus model and
ultralytics drift between the original harness run and the pod. **Any variant must be
compared against a control produced the same way, and anything under ~0.8pp is noise.**
Three of this round's candidate effects sit inside it.
