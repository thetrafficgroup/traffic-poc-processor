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
