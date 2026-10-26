# TMC round-2 analysis

Findings: [`docs/TMC_ROUND2_FINDINGS.md`](../../docs/TMC_ROUND2_FINDINGS.md).

**These scripts are not self-contained.** They read `proc-iter-tmc/` from the
traffic-v2 monorepo — cached observation streams, manual counts and lane configs —
which is not version controlled. `assoc.ROOT` points at it; change that constant if
the harness moves. They are committed so the round can be re-run and audited, not as
production code.

## Scoring (free, CPU, no GPU)

| script | what it answers |
|---|---|
| `plan.py` | the shippable package: shipped vs leg-label fixes vs de-dup cap, all 18 sites |
| `geoaudit.py` | searches the 8 rigid relabelings per site; upper bound on label error |
| `cyclic.py` | ground-truth-free mirror detection (shipped as `tools/validate_lane_config.py`) |
| `cells.py` | per-cell error for the sites holding the error |
| `perleg.py` | splits each leg's error into DETECTION vs ATTRIBUTION |
| `whylost.py` | why each lost track was lost: short line / frame edge / midblock / at entry line |
| `ceiling.py` | whether a continuation tracklet exists at all for a lost vehicle |
| `diag.py` | net volume error and entry counts, so sign-cancellation is visible |
| `run_exp.py` | arm driver: B0 / H1 / zones / path-fit, with imputation on or off |

## Offline tracking (needs the detection dumps)

`flow.py` is the detection-level offline tracker (class-agnostic NMS, optimal
per-frame-pair assignment, HSV appearance, coarse-to-fine global linking with mutual
exclusion, chain filtering). `run_flow.py` scores it, `calib.py` derives its gates from
measured data, `pathfit.py` is trajectory-based attribution, `dup.py` is appearance-based
duplicate suppression. Conclusion: loses to BoT-SORT on dense traffic — see the findings.

Dumps live at `s3://trafficgroup-ne-data-sb/experiments/round2-dets/runs/<run>/out/`
(raw detections at conf=0.05, no tracking). `dump_dets.py` produces them; `pod/` launches
an autonomous GPU pod that flushes to S3 after every clip and terminates itself.
