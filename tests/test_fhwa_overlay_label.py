"""
Tests for the debug-video FHWA overlay caption.

The caption is display-only: it is drawn with cv2.putText and never stored,
counted or reported. These tests cover the formatting rules and act as a
tripwire proving the classification data path was left untouched.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.axle_count_classifier import (  # noqa: E402
    DEFAULT_TRUCK_FHWA,
    DIRECT_FHWA_MAP,
    AxleCountClassifier,
    format_display_label,
)

_PROCESSORS = [
    os.path.join(os.path.dirname(__file__), "..", "src", "atr", "atr_processor.py"),
    os.path.join(os.path.dirname(__file__), "..", "src", "tmc", "tmc_processor.py"),
]

# Identifiers that belong to the count/report path. The overlay must never
# appear on a line that touches one of them.
_DATA_PATH_IDENTIFIERS = [
    "final_class_name",
    "detected_classes",
    "vehicle_counts_by_lane",
    "vehicle_class",
    "process_vehicle_detection",
]


def test_non_truck_classes_show_their_direct_bin():
    assert format_display_label("car") == "C2/car"
    assert format_display_label("bus") == "C4/bus"
    assert format_display_label("work_van") == "C3/van"
    assert format_display_label("pickup_truck") == "C3/pickup"
    assert format_display_label("motorcycle") == "C1/moto"
    assert format_display_label("motorized_vehicle") == "C2/motveh"


def test_truck_with_recorded_bin_shows_bin_and_axles():
    assert format_display_label("articulated_truck", 8, 3) == "C8a3/artic"
    assert format_display_label("articulated_truck", 9, 5) == "C9a5/artic"
    assert format_display_label("multi_articulated_truck", 12, 6) == "C12a6/multi"


def test_bin_without_axle_count_omits_the_axle_suffix():
    assert format_display_label("single_unit_truck", 5) == "C5/SUT"
    assert format_display_label("single_unit_truck", 5, 0) == "C5/SUT"


def test_truck_with_no_recorded_bin_renders_bare():
    """The honesty property: the overlay never invents a bin for a truck."""
    assert format_display_label("articulated_truck") == "artic"
    assert format_display_label("single_unit_truck") == "SUT"
    assert format_display_label("multi_articulated_truck") == "multi"


def test_unknown_and_missing_classes():
    assert format_display_label("heavy_pickup") == "heavy_pickup"
    assert format_display_label("heavy_van") == "heavy_van"
    assert format_display_label("pedestrian") == "pedestrian"
    assert format_display_label(None) == ""
    assert format_display_label("") == ""


def test_labels_are_ascii_and_never_leak_the_string_none():
    classes = [
        None,
        "",
        "car",
        "bus",
        "work_van",
        "pickup_truck",
        "motorcycle",
        "motorized_vehicle",
        "single_unit_truck",
        "articulated_truck",
        "multi_articulated_truck",
        "heavy_pickup",
        "heavy_van",
        "pedestrian",
    ]
    for class_name in classes:
        for fhwa_class in (None, 1, 2, 5, 8, 13):
            for axle_count in (None, 0, 2, 5, 9):
                label = format_display_label(class_name, fhwa_class, axle_count)
                assert isinstance(label, str)
                label.encode("ascii")  # raises if a glyph cv2 cannot draw crept in
                assert "None" not in label


def test_classification_data_path_is_unchanged():
    """Tripwire: production FHWA mapping must behave exactly as before."""
    assert DIRECT_FHWA_MAP == {
        "motorcycle": 1,
        "car": 2,
        "pickup_truck": 3,
        "work_van": 3,
        "motorized_vehicle": 2,
        "bus": 4,
    }
    assert DEFAULT_TRUCK_FHWA == {
        "single_unit_truck": 5,
        "articulated_truck": 8,
        "multi_articulated_truck": 11,
    }

    get_fhwa_class = AxleCountClassifier.get_fhwa_class
    assert get_fhwa_class("articulated_truck", None) == 8
    assert get_fhwa_class("articulated_truck", 5) == 9
    assert get_fhwa_class("articulated_truck", 7) == 10
    assert get_fhwa_class("single_unit_truck", 3) == 6
    assert get_fhwa_class("single_unit_truck", 4) == 7
    assert get_fhwa_class("single_unit_truck", None) == 5
    assert get_fhwa_class("multi_articulated_truck", None) == 11
    assert get_fhwa_class("car", None) == 2
    assert get_fhwa_class("pedestrian", None) is None


def test_overlay_helper_is_only_used_at_the_two_draw_sites():
    for path in _PROCESSORS:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        call_lines = [line for line in lines if "format_display_label(" in line]
        assert len(call_lines) == 2, f"{path}: {len(call_lines)} call sites"


def test_overlay_helper_never_touches_the_count_path():
    for path in _PROCESSORS:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        for line in lines:
            if "format_display_label" not in line:
                continue
            for identifier in _DATA_PATH_IDENTIFIERS:
                assert identifier not in line, f"{path}: {identifier} in {line.strip()}"
