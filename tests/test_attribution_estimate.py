"""Invariants for the attribution estimate's apportionment.

The estimate adds vehicles we detected but could not attribute. The arithmetic
that places them must not invent or lose any: naive per-cell rounding silently
LOSES vehicles, because a few-percent uplift on many small cells rounds down far
more often than up. These tests pin that down.
"""
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.minute_tracker import apportion, collect_class_cells, recompute_totals  # noqa: E402

DIRS = ["NORTH", "SOUTH", "EAST", "WEST"]
TURNS = ["left", "right", "straight", "u-turn"]


def _video(seed, scale=8, classes=("car", "pickup_truck", "bus")):
    random.seed(seed)
    v = {c: {d: {t: random.randint(0, scale) for t in TURNS} for d in DIRS} for c in classes}
    recompute_totals(v)
    return v


def _total(v):
    return sum(x for d in v["total"].values() for x in d.values())


def test_adds_exactly_the_requested_count():
    for add in (1.0, 3.7, 29.1, 229.2, 1000.0):
        v = _video(1)
        before = _total(v)
        placed = apportion(collect_class_cells(v), add)
        recompute_totals(v)
        assert placed == int(round(add))
        assert _total(v) - before == placed


def test_no_rounding_loss_across_many_small_cells():
    """The trap: 60 minutes x 3 classes x 16 cells, most of them tiny."""
    minutes = [_video(100 + i, scale=3) for i in range(60)]
    cells = [c for v in minutes for c in collect_class_cells(v)]
    observed = sum(c[2] for c in cells)
    add = observed * 0.055
    placed = apportion(cells, add)
    for v in minutes:
        recompute_totals(v)
    assert placed == int(round(add))
    assert sum(_total(v) for v in minutes) - observed == placed


def test_distribution_is_proportional():
    v = {"car": {d: {t: 0 for t in TURNS} for d in DIRS}}
    v["car"]["NORTH"]["straight"] = 100
    v["car"]["NORTH"]["left"] = 50
    recompute_totals(v)
    apportion(collect_class_cells(v), 15)
    assert v["car"]["NORTH"]["straight"] == 110
    assert v["car"]["NORTH"]["left"] == 55


def test_total_block_always_equals_sum_over_classes():
    v = _video(7)
    apportion(collect_class_cells(v), 33)
    recompute_totals(v)
    per_class = sum(x for c, dd in v.items() if c != "total"
                    for d in dd.values() for x in d.values())
    assert per_class == _total(v)


def test_degenerate_inputs_add_nothing():
    empty = {"total": {d: {t: 0 for t in TURNS} for d in DIRS}}
    assert apportion(collect_class_cells(empty), 50.0) == 0
    for add in (0.4, 0.0, -5.0):
        v = _video(3)
        before = _total(v)
        assert apportion(collect_class_cells(v), add) == 0
        recompute_totals(v)
        assert _total(v) == before


def test_zero_count_cells_never_receive_vehicles():
    """A movement never observed must not be invented by the estimate."""
    v = {"car": {d: {t: 0 for t in TURNS} for d in DIRS}}
    v["car"]["NORTH"]["straight"] = 40
    recompute_totals(v)
    apportion(collect_class_cells(v), 12)
    assert v["car"]["SOUTH"]["left"] == 0
    assert v["car"]["NORTH"]["u-turn"] == 0
    assert v["car"]["NORTH"]["straight"] == 52
