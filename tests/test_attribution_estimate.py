"""Invariants for the attribution estimate's apportionment.

Naive per-cell rounding silently LOSES vehicles: a few-percent uplift on many
small cells rounds down far more often than up. These pin that down.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tmc.tmc_processor import estimate_missing  # noqa: E402
from utils.minute_tracker import apportion, recompute_totals  # noqa: E402

DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]
TURNS = ["left", "right", "straight", "u-turn"]


def _video(seed, scale=8):
    random.seed(seed)
    v = {c: {d: {t: random.randint(0, scale) for t in TURNS} for d in DIRS}
         for c in ("car", "pickup_truck", "bus")}
    recompute_totals(v)
    return v


def _total(v):
    return sum(x for d in v["total"].values() for x in d.values())


def test_adds_exactly_the_requested_count():
    for add in (1.0, 3.7, 29.1, 229.2, 1000.0):
        v = _video(1)
        before = _total(v)
        placed = apportion([v], add)
        assert placed == int(round(add))
        assert _total(v) - before == placed


def test_no_rounding_loss_across_many_small_cells():
    """The trap: 60 minutes x 3 classes x 16 cells, most of them tiny."""
    minutes = [_video(100 + i, scale=3) for i in range(60)]
    observed = sum(_total(v) for v in minutes)
    add = observed * 0.055
    placed = apportion(minutes, add)
    assert placed == int(round(add))
    assert sum(_total(v) for v in minutes) - observed == placed


def test_distribution_is_proportional_and_total_stays_consistent():
    v = {"car": {d: {t: 0 for t in TURNS} for d in DIRS}}
    v["car"]["NORTH"]["straight"] = 100
    v["car"]["NORTH"]["left"] = 50
    recompute_totals(v)
    apportion([v], 15)
    assert v["car"]["NORTH"]["straight"] == 110
    assert v["car"]["NORTH"]["left"] == 55
    assert _total(v) == 165


def test_degenerate_inputs_add_nothing():
    assert apportion([{"total": {d: {t: 0 for t in TURNS} for d in DIRS}}], 50.0) == 0
    for add in (0.4, 0.0, -5.0):
        v = _video(3)
        before = _total(v)
        assert apportion([v], add) == 0
        assert _total(v) == before


def test_zero_count_cells_never_receive_vehicles():
    """A movement never observed must not be invented by the estimate."""
    v = {"car": {d: {t: 0 for t in TURNS} for d in DIRS}}
    v["car"]["NORTH"]["straight"] = 40
    recompute_totals(v)
    apportion([v], 12)
    assert v["car"]["SOUTH"]["left"] == 0
    assert v["car"]["NORTH"]["u-turn"] == 0
    assert v["car"]["NORTH"]["straight"] == 52


def test_estimate_is_zero_without_evidence():
    """No entries beyond what was attributed, or none at all -> nothing to place."""
    assert estimate_missing(entries=500, observed=500) == 0.0
    assert estimate_missing(entries=0, observed=0) == 0.0
    assert estimate_missing(entries=100, observed=0) == 0.0


def test_small_gap_is_absorbed_by_the_fragmentation_floor():
    """A gap within normal fragmentation must not move the counts."""
    assert estimate_missing(entries=1020, observed=1000) == 0.0


def test_large_gap_is_estimated_but_capped():
    est = estimate_missing(entries=1400, observed=1000)
    assert est > 0
    assert est <= 0.06 * 1000 + 1e-9
