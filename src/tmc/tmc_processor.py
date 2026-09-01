import cv2
import json
import math
import time
import yaml
import tempfile
from collections import Counter
from ultralytics import YOLO
import numpy as np
import sys
import os
import torch
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.overlap_detection import (
    soft_nms, detect_occlusions, adjust_confidence_for_occlusion,
    TrackInterpolator, post_process_detections, analyze_overlap_patterns
)
from utils.minute_tracker import MinuteTracker
from utils.frame_utils import calculate_frame_ranges_from_seconds, validate_trim_periods
from utils.ffmpeg_writer import FFmpegH264Writer
from utils.background_model import BackgroundProvider, model_wants_background
from crosswalk.crosswalk_processor import CrosswalkProcessor
from crosswalk.crosswalk_minute_tracker import CrosswalkMinuteTracker
from pedestrian.pedestrian_processor import PedestrianProcessor

CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.2
_BASE_TRACKER_CONFIG = os.path.join(os.path.dirname(__file__), "botsort.yaml")
_TRACK_BUFFER_SECONDS = 5  # How many seconds to keep lost tracks alive

# Classes to exclude from the vehicle model (handled by the pedestrian model instead)
_VEHICLE_MODEL_EXCLUDE_CLASSES = {"pedestrian", "bicycle", "non-motorized_vehicle"}

# Set per-video by process_video: mean of all counting-line endpoints. Used by the
# label-independent "center" entry gate.
_INTERSECTION_CENTER = None

# Counted-entry events (line, cx, cy, frame) for fragment de-dup. Reset per video.
_ENTRY_EVENTS = []

# Fragment de-dup window: skip a new entry already counted on the SAME line within
# this many px / frames (the tracker splits one vehicle into several IDs).
_DEDUP_PX = 60
_DEDUP_FRAMES = 30

# Minimum net displacement (px) a track must have travelled since it was first seen
# before a line crossing may count as an entry. Static false detections (e.g. a road
# feature detected as a "bus" near a line) jitter in place, respawn under new track
# IDs, and each fragment can otherwise pass the entry gate — one phantom was counted
# 83 times in an hour. Real vehicles translate hundreds of px before crossing, so
# this does not affect them. Overridable via TMC_MIN_DISPLACEMENT_PX (0 disables);
# reset per video / trim period alongside the other counting state.
_MIN_DISPLACEMENT_PX = 40
_FIRST_POS = {}  # obj_id -> (cx, cy) at first sighting

# Fallback policy for line crossings (see the crossing test below).
# "tallpoint" (default): tall bodies fall back on a point low in the box,
# short classes on the centroid. "shortclass": no fallback at all for tall
# bodies (over-broad — dropped real trucks). "both": legacy, centroid fallback
# for every class. "tallall": the low point for every class. "wheels": no
# fallback (diagnostic only).
_CROSS_POINT_MODE = os.environ.get("TMC_CROSS_POINT", "tallpoint")
_CENTROID_OK_CLASSES = {"car", "pickup_truck", "work_van", "motorcycle"}
# Geometric fallback point: a fraction of the way from centroid to wheels.
# 0.6 == 80% of box height. A tall body's CENTROID projects across lines its
# wheels never approach (phantom turns); a point this low does not, while a
# truck whose wheel point merely grazed the line is still recovered.
_TALL_FRAC = float(os.environ.get("TMC_TALL_FRAC", "0.6"))


def _low_point(centroid, wheels):
    return (centroid[0] + (wheels[0] - centroid[0]) * _TALL_FRAC,
            centroid[1] + (wheels[1] - centroid[1]) * _TALL_FRAC)


_LAST_POS = {}   # obj_id -> (cx, cy) at most recent sighting

# Heading inference: a track that entered (crossed one counting line) and then
# died mid-intersection still carries evidence of where it was going — its own
# direction of travel. Extrapolate that heading and, if the ray meets another
# counting line, credit the movement it was completing.
#
# Add-only: it never merges identities and never removes a counted vehicle, so
# it cannot destroy a real one (unlike track stitching, which did exactly that
# and was removed). The geometry gates itself — a fragment dying while pointed
# at nothing recovers nothing, which is why the fragment-heavy sites are barely
# touched. Offline replay over 12 GT hours: 95.0% -> 96.2%, 9/12 sites improved,
# worst case -0.5. Disable with TMC_HEADING_INFER=0.
_HEADING_INFER_ON = os.environ.get("TMC_HEADING_INFER", "1") == "1"
_HEADING_MIN_DISP_PX = float(os.environ.get("TMC_HEADING_MIN_DISP_PX", "60"))


def _ray_hits_segment(origin, direction, a, b, max_t=2000.0):
    """Parametric distance at which ray origin+t*direction meets segment a-b.

    Returns t > 0 if the ray crosses the segment within max_t, else None.
    """
    rx, ry = direction
    sx, sy = b[0] - a[0], b[1] - a[1]
    den = rx * sy - ry * sx
    if abs(den) < 1e-9:
        return None                      # parallel
    qx, qy = a[0] - origin[0], a[1] - origin[1]
    t = (qx * sy - qy * sx) / den        # along the ray
    u = (qx * ry - qy * rx) / den        # along the segment
    if t > 0 and t < max_t and 0.0 <= u <= 1.0:
        return t
    return None


def infer_exit_line_by_heading(entry_line, first_pos, last_pos, LINES):
    """The counting line this track was heading for when it died, or None."""
    dx = last_pos[0] - first_pos[0]
    dy = last_pos[1] - first_pos[1]
    if math.hypot(dx, dy) < _HEADING_MIN_DISP_PX:
        return None                      # too little travel to trust a heading
    best = None
    for line in LINES:
        if line["name"] == entry_line:
            continue
        pts = line["points"]
        for a, b in zip(pts, pts[1:]):
            t = _ray_hits_segment(last_pos, (dx, dy), a, b)
            if t is not None and (best is None or t < best[0]):
                best = (t, line["name"])
    return best[1] if best else None


def _segments_intersect(p1, p2, q1, q2):
    """True if segment p1→p2 properly intersects segment q1→q2 (orientation test)."""
    def orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = orient(q1, q2, p1)
    d2 = orient(q1, q2, p2)
    d3 = orient(p1, p2, q1)
    d4 = orient(p1, p2, q2)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def _ensure_int_point(point):
    """Convert point coordinates to integers, handling both dict and tuple formats"""
    if isinstance(point, dict):
        return (int(round(point["x"])), int(round(point["y"])))
    elif isinstance(point, (list, tuple)):
        return (int(round(point[0])), int(round(point[1])))
    return point


def _is_finite_number(value):
    """True for a real, finite int/float. Bools and strings are not coordinates."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _tmc_vertex(point, label, index):
    """Validate one stored vertex, then convert it to an int (x, y).

    Anything that is not a 2-element sequence of finite numbers (or the dict
    {"x": x, "y": y} form) raises ValueError naming the line HERE, at parse time.
    Without this check a malformed vertex passes straight through
    _ensure_int_point and only blows up a frame later inside the crossing or
    drawing loop as an IndexError/TypeError — and that exception class strands
    the video in PROCESSING forever with no SQS failure message.
    """
    if isinstance(point, dict):
        valid = _is_finite_number(point.get("x")) and _is_finite_number(point.get("y"))
    elif isinstance(point, (list, tuple)):
        valid = len(point) == 2 and all(_is_finite_number(c) for c in point)
    else:
        valid = False
    if not valid:
        raise ValueError(
            f"TMC line '{label}' vertex {index} is not a 2-element [x, y] of finite "
            f"numbers (or {{'x': x, 'y': y}}): {point!r}"
        )
    return _ensure_int_point(point)


def tmc_line_vertices(data, name=None):
    """THE python normaliser for a stored TMC counting line -> [(x, y), ...].

    Prefers the polyline shape {"points": [[x, y], ...]} and falls back to the
    legacy {"pt1": [x, y], "pt2": [x, y]}. This is the ONLY place in this module
    allowed to read data["pt1"] / data["pt2"] on a stored line entry; every
    downstream consumer reads the normalised "points" list.

    A present, non-empty "points" is a commitment: it must parse. It is never
    silently abandoned in favour of pt1/pt2.

    Raises ValueError naming the line, never a bare KeyError or IndexError.
    """
    label = name if name is not None else repr(data)
    vertices = None
    if isinstance(data, dict):
        points = data.get("points")
        if isinstance(points, (list, tuple)) and len(points) > 0:
            vertices = [_tmc_vertex(p, label, i) for i, p in enumerate(points)]
        elif "pt1" in data and "pt2" in data:
            vertices = [
                _tmc_vertex(data["pt1"], label, "pt1"),
                _tmc_vertex(data["pt2"], label, "pt2"),
            ]
    if vertices is None:
        raise ValueError(
            f"TMC line '{label}' has no usable geometry: expected a non-empty "
            f"'points' list or both 'pt1' and 'pt2', got {data!r}"
        )
    if len(vertices) < 2:
        # Checked here rather than trusted from the API DTO: hand-edited lane
        # configs (proc-iter-tmc/gt/lane_configs/*.json) never pass through it.
        raise ValueError(
            f"TMC line '{label}' needs at least 2 vertices, got {len(vertices)}: {data!r}"
        )
    return vertices


def polyline_segments(points):
    """Consecutive vertex pairs of a polyline -> [(a, b), ...]."""
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def polyline_crosses(prev_pos, curr_pos, segments):
    """True if the movement segment crosses ANY segment of the polyline.

    A single boolean per line: a movement that crosses two segments of the same
    polyline is still ONE crossing, so it can only ever be counted once.
    """
    return any(_segments_intersect(prev_pos, curr_pos, a, b) for a, b in segments)


def intersection_center(lines):
    """Mean of the counting lines' ENDPOINTS (first and last vertex of each line)."""
    _pts = [p for ln in lines for p in (ln["points"][0], ln["points"][-1])]
    return (
        sum(p[0] for p in _pts) / len(_pts),
        sum(p[1] for p in _pts) / len(_pts),
    ) if _pts else None


def line_label_anchor(points):
    """Midpoint of the polyline's MIDDLE segment, so the label lands on the line.

    For 2 vertices this is bit-identical to ((x1 + x2) // 2, (y1 + y2) // 2).
    """
    i = (len(points) - 1) // 2
    (x1, y1), (x2, y2) = points[i], points[i + 1]
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def _compute_img_size(width: int, height: int, cap: int = 1920) -> int:
    """Choose YOLO imgsz from video resolution, capped and rounded to 32."""
    longest = max(width, height)
    size = min(longest, cap)
    return (size // 32) * 32 or 32


def _build_tracker_config(fps: float) -> str:
    """Read base botsort.yaml, override track_buffer for this video's FPS,
    and return the path to a temporary YAML file.

    Raises FileNotFoundError if the base config is missing.
    """
    if not os.path.isfile(_BASE_TRACKER_CONFIG):
        raise FileNotFoundError(
            f"Tracker config not found: {_BASE_TRACKER_CONFIG}. "
            "Ensure botsort.yaml is packaged with the processor."
        )

    with open(_BASE_TRACKER_CONFIG) as f:
        config = yaml.safe_load(f)

    config["track_buffer"] = max(30, int(fps * _TRACK_BUFFER_SECONDS))

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", prefix="botsort_", delete=False,
    )
    yaml.dump(config, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()

    print(f"🔧 Tracker config: track_buffer={config['track_buffer']} "
          f"({_TRACK_BUFFER_SECONDS}s @ {fps:.1f}fps), "
          f"with_reid={config.get('with_reid')}, "
          f"appearance_thresh={config.get('appearance_thresh')}, "
          f"model={config.get('model')}")
    return tmp.name



def build_analysis_by_vehicle_class(detected_classes, turn_types_by_id, crossing_timestamps, crossed_lines_by_id):
    """
    Build new analysis structure grouped by vehicle class first.
    Structure: vehicle_class -> origin_direction -> turn_direction -> count
    """
    analysis = {}
    
    # Initialize totals structure
    totals = {
        "NORTH": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
        "SOUTH": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
        "EAST": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
        "WEST": {"straight": 0, "left": 0, "right": 0, "u-turn": 0}
    }
    
    # Group vehicles by class
    vehicles_by_class = {}
    for obj_id, vehicle_class in detected_classes.items():
        if vehicle_class not in vehicles_by_class:
            vehicles_by_class[vehicle_class] = []
        vehicles_by_class[vehicle_class].append(obj_id)
    
    # For each vehicle class, analyze movements
    for vehicle_class, vehicle_ids in vehicles_by_class.items():
        analysis[vehicle_class] = {
            "NORTH": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
            "SOUTH": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
            "EAST": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
            "WEST": {"straight": 0, "left": 0, "right": 0, "u-turn": 0}
        }
        
        for vehicle_id in vehicle_ids:
            # Determine origin direction (first line crossed)
            if vehicle_id in crossing_timestamps and crossing_timestamps[vehicle_id]:
                origin_direction = crossing_timestamps[vehicle_id][0][0]
                
                # Determine turn type. A vehicle whose movement was never
                # classified (single crossing) is NOT assumed straight anymore:
                # that padding made results.vehicles disagree with the minute
                # rollup the TMC report reads, and inflated 'straight' by the
                # whole attribution residual. Unclassified tracks are surfaced
                # separately (results.unclassified_tracks).
                if vehicle_id in turn_types_by_id:
                    turn_type = turn_types_by_id[vehicle_id]
                else:
                    continue

                # Increment counters
                if origin_direction in analysis[vehicle_class] and turn_type in analysis[vehicle_class][origin_direction]:
                    analysis[vehicle_class][origin_direction][turn_type] += 1
                    # Also increment totals
                    totals[origin_direction][turn_type] += 1
    
    # Add totals to the analysis
    analysis["total"] = totals
    
    return analysis

def is_entering_from_outside(line_name, prev_pos, curr_pos, line_coords):
    """Decide whether a line crossing counts as an entry (for the total count).

    Default 'any': count every crossing. Measured best on the customer cameras on
    BOTH totals (95.1% vs center 85.1%) and approach×turn movements (err 128 vs 204,
    8 intersections vs manual counts). 'center' (TMC_COUNT_MODE=center): only count
    crossings moving TOWARD the intersection centre — use on cameras where heavy
    tracker fragmentation makes 'any' overcount; it over-filters otherwise (worst on
    3-leg intersections, whose missing leg skews the centre point). (Both replace the
    old per-direction cross-product heuristic, which assumed canonical line
    orientations and silently dropped ~15% of real vehicles.)
    """
    if os.environ.get("TMC_COUNT_MODE", "any") != "center":
        return True
    if _INTERSECTION_CENTER is not None:
        mx, my = _INTERSECTION_CENTER
        motion = (curr_pos[0] - prev_pos[0], curr_pos[1] - prev_pos[1])
        toward = (mx - curr_pos[0], my - curr_pos[1])
        return (motion[0] * toward[0] + motion[1] * toward[1]) > 0
    return True


def process_single_detection(
    obj_id, class_name, cx, cy, wx, wy, current_frame,
    prev_wheels, prev_centroids, counted_ids_per_line,
    entry_counted_ids, crossed_lines_by_id, crossing_timestamps,
    turn_types_by_id, detected_classes, class_counts_by_id,
    LINES, is_entering_fn, classify_turn_fn,
    track_interpolator=None, counts=None,
):
    """
    Process a single detection through TMC turn logic.
    Extracted from inline code so it can be reused for both vehicle and bicycle detections.

    Args:
        obj_id: Track ID (namespaced for ped model)
        class_name: Detection class name
        cx, cy: Centroid position
        wx, wy: Wheels/bottom position (for line crossing)
        current_frame: Current frame number
        prev_wheels, prev_centroids: Previous position dicts (mutated)
        counted_ids_per_line, entry_counted_ids: Counting state (mutated)
        crossed_lines_by_id, crossing_timestamps: Crossing tracking (mutated)
        turn_types_by_id, detected_classes, class_counts_by_id: Classification state (mutated)
        LINES: List of line dicts with name, points, segments
        is_entering_fn: Function to check if entering from outside
        classify_turn_fn: Function to classify turn type
        track_interpolator: Optional track interpolator for centroid tracking
    """
    # Store class for this object ID (first detection wins)
    if obj_id not in class_counts_by_id:
        class_counts_by_id[obj_id] = class_name

    # Record where this track was first seen (for the static-phantom entry guard)
    if obj_id not in _FIRST_POS:
        _FIRST_POS[obj_id] = (cx, cy)

    # Update track interpolator if available
    if track_interpolator is not None:
        track_interpolator.update_track(obj_id, (cx, cy), current_frame)

    # Get previous positions
    prev_wheels_pos = prev_wheels.get(obj_id)
    prev_centroid_pos = prev_centroids.get(obj_id)

    if prev_wheels_pos and prev_centroid_pos:
        for line in LINES:
            name = line["name"]
            segments = line["segments"]

            # True crossing: the movement segment (wheels, with centroid as a second
            # chance) intersects ANY segment of the counting polyline. Speed-independent,
            # so fast vehicles can't step over the line between frames.
            # Wheel path is the ground-truth crossing test. When the box bottom is
            # unreliable (occluded queues, truncation) a second path is tried — but
            # WHICH point matters. A tall body's centroid projects over lines its
            # wheels never approach, inventing crossings (York Rd 2026-08: bus
            # bodies clipping a far line turned 13 through-movements/hr into
            # phantom right turns). Banning the fallback outright for tall classes
            # (the earlier "shortclass" rule) fixed York but dropped REAL trucks
            # whose wheel point merely grazed the line: over 16 GT hours it cost
            # Oella -1.2, Hayshed -0.6 and Sanford -0.5, shedding heavy classes on
            # straight movements (SUT -34, heavy_pickup -20, artic -10).
            # So keep the fallback for every class, but evaluate tall bodies at a
            # point 80% down the box instead of the centroid. Battery 4 (full GT
            # hours): York 92.5 -> 95.8 (beating shortclass's 95.4) while Oella,
            # Hayshed and Sanford all return to baseline. Insensitive to the exact
            # fraction — 0.6 and 0.8 scored identically on all four sites.
            # Kill switch: TMC_CROSS_POINT=both restores the old behaviour.
            if _CROSS_POINT_MODE in ("tallpoint", "tallall"):
                if (_CROSS_POINT_MODE == "tallall"
                        or class_name not in _CENTROID_OK_CLASSES):
                    fb_prev = _low_point(prev_centroid_pos, prev_wheels_pos)
                    fb_cur = _low_point((cx, cy), (wx, wy))
                else:
                    fb_prev, fb_cur = prev_centroid_pos, (cx, cy)
                fallback = polyline_crosses(fb_prev, fb_cur, segments)
            else:
                _fallback_ok = (
                    _CROSS_POINT_MODE == "both"
                    or (_CROSS_POINT_MODE == "shortclass"
                        and class_name in _CENTROID_OK_CLASSES)
                )
                fallback = (_fallback_ok
                            and polyline_crosses(prev_centroid_pos, (cx, cy),
                                                 segments))
            crossed = (
                polyline_crosses(prev_wheels_pos, (wx, wy), segments)
                or fallback
            )

            if crossed and obj_id not in counted_ids_per_line[name]:
                counted_ids_per_line[name].add(obj_id)
                if counts is not None:
                    counts[name] = counts.get(name, 0) + 1

                # Static-phantom guard: only tracks that have actually travelled since
                # first sighting may be counted. Stationary false detections jitter in
                # place with near-zero net displacement.
                _f = _FIRST_POS.get(obj_id, (cx, cy))
                _moved = (
                    _MIN_DISPLACEMENT_PX <= 0
                    or (cx - _f[0]) ** 2 + (cy - _f[1]) ** 2
                    >= _MIN_DISPLACEMENT_PX * _MIN_DISPLACEMENT_PX
                )

                # Entry (with fragment de-dup): the tracker splits one vehicle into
                # several IDs, so the same physical vehicle can cross a line multiple
                # times as different IDs. Skip a new entry already counted on the SAME
                # line, near the SAME spot, within a short window.
                if obj_id not in entry_counted_ids and _moved and is_entering_fn(name, prev_centroid_pos, (cx, cy), line):
                    _dup = any(
                        _ln == name and abs(current_frame - _ff) <= _DEDUP_FRAMES
                        and (cx - _fx) ** 2 + (cy - _fy) ** 2 <= _DEDUP_PX * _DEDUP_PX
                        for (_ln, _fx, _fy, _ff) in _ENTRY_EVENTS
                    )
                    if not _dup:
                        entry_counted_ids.add(obj_id)
                        _ENTRY_EVENTS.append((name, cx, cy, current_frame))
                        if obj_id not in detected_classes:
                            detected_classes[obj_id] = class_name

                # Register crossing with timestamp
                if obj_id not in crossed_lines_by_id:
                    crossed_lines_by_id[obj_id] = []
                    crossing_timestamps[obj_id] = []

                if name not in [crossing[0] for crossing in crossing_timestamps[obj_id]]:
                    import time as time_mod
                    current_time = time_mod.time()
                    crossed_lines_by_id[obj_id].append(name)
                    crossing_timestamps[obj_id].append((name, current_time))

                # De-dup exemption: crossing a SECOND distinct line proves this
                # is a real mover (static phantoms jitter at one line), so grant
                # the entry the platoon de-dup / displacement guard withheld.
                # Recovers 27-44 platoon vehicles/hr (dev A/B 2026-08-20:
                # +2.2 to +2.9 pts turn accuracy vs manual counts).
                # Deliberately does NOT append to _ENTRY_EVENTS — the grant must
                # not suppress anyone else.
                if (obj_id not in entry_counted_ids
                        and len(crossing_timestamps[obj_id]) >= 2):
                    entry_counted_ids.add(obj_id)
                    if obj_id not in detected_classes:
                        detected_classes[obj_id] = class_name

                # Turn detection when 2+ crossings
                if len(crossing_timestamps[obj_id]) >= 2 and obj_id not in turn_types_by_id:
                    turn_type = classify_turn_fn(crossing_timestamps[obj_id])
                    if turn_type != 'invalid' and turn_type != 'unknown':
                        turn_types_by_id[obj_id] = turn_type

    # Always update previous positions
    _LAST_POS[obj_id] = (cx, cy)
    prev_centroids[obj_id] = (cx, cy)
    prev_wheels[obj_id] = (wx, wy)


def process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH="best.pt", video_uuid=None, progress_callback=None, minute_batch_callback=None, generate_video_output=False, output_video_path=None, trim_periods=None, pedestrian_model_path=None, truck_classifier_model_path=None, axle_detector_model_path=None):
    """
    Process video for TMC (Turning Movement Count) analysis with optional trimming.

    Args:
        VIDEO_PATH: Path to video file
        LINES_DATA: Line configuration data (may include 'crosswalks' key)
        MODEL_PATH: Path to YOLO model
        video_uuid: UUID of the video being processed
        progress_callback: Optional callback for progress updates
        minute_batch_callback: Optional callback for minute-by-minute batch data
        generate_video_output: Whether to generate annotated output video
        output_video_path: Path for output video (if generate_video_output=True)
        trim_periods: Optional list of trim periods in seconds [{"start": 3600, "end": 10800}, ...]
        pedestrian_model_path: Optional path to pedestrian/bicycle YOLO model
        axle_detector_model_path: Optional path to wheel/axle detector model for FHWA classification

    Returns:
        Dictionary with processing results (includes crosswalk data if applicable)
    """

    # Validate trim periods if provided
    if trim_periods:
        is_valid, error_msg = validate_trim_periods(trim_periods)
        if not is_valid:
            print(f"⚠️ Invalid trim_periods: {error_msg}")
            print("⚠️ Falling back to processing entire video")
            trim_periods = None

    # Per-track class is finalized from a cumulative confidence-weighted vote over the
    # track's WHOLE life (see the vote in the detection loop + finalization near the
    # end), so early far-away/blurry misreads get corrected by later, closer views.
    class_vote_scores = {}    # obj_id -> {class_name: cumulative conf}
    artic_subtype_by_id = {}  # obj_id -> truck-classifier verdict (cached once)

    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Fail fast on corrupt video metadata
    if not fps or fps <= 0 or math.isnan(fps):
        cap.release()
        raise ValueError(f"Invalid FPS ({fps}) from video: {VIDEO_PATH}")
    if video_width <= 0 or video_height <= 0:
        cap.release()
        raise ValueError(f"Invalid video dimensions ({video_width}x{video_height}): {VIDEO_PATH}")

    # Compute resolution-aware inference size
    img_size = _compute_img_size(video_width, video_height)
    print(f"📐 Video {video_width}x{video_height} → YOLO imgsz={img_size}")

    # Build fps-aware tracker config (temp file — cleaned up in finally block)
    tracker_config = _build_tracker_config(fps)

    # Load YOLO model
    model = YOLO(MODEL_PATH)
    print(f"✅ YOLO model loaded: {MODEL_PATH}")

    # Background-conditioned (4-channel) models get a per-video background model
    # computed in a sampled pre-pass; 3-channel models are unaffected.
    bg_provider = None
    if model_wants_background(model):
        print("🌄 4-channel background-conditioned model detected — building background model")
        bg_provider = BackgroundProvider(VIDEO_PATH, fps, total_frames)

    # Load optional truck subtype classifier
    truck_classifier = None
    if truck_classifier_model_path:
        from utils.truck_classifier import TruckClassifier
        truck_classifier = TruckClassifier(truck_classifier_model_path)

    # Load optional axle detector for FHWA classification
    axle_classifier = None
    if axle_detector_model_path:
        from utils.axle_count_classifier import AxleCountClassifier
        axle_classifier = AxleCountClassifier(axle_detector_model_path)

    # CRITICAL: Strip crosswalks from LINES_DATA before line-parsing loop.
    # The crosswalks key contains an array, not a dict with pt1/pt2.
    # Leaving it would crash with TypeError in the loop below.
    raw_lines = dict(LINES_DATA)  # Copy to avoid mutating caller's data
    crosswalks_config = raw_lines.pop("crosswalks", [])

    LINES = []
    for name, data in raw_lines.items():
        _points = tmc_line_vertices(data, name)
        # Built once per video, not per line per detection per frame
        # (~10^7-10^8 rebuilds on a 24 h clip).
        LINES.append({
            "name": name.upper(),
            "points": _points,
            "segments": polyline_segments(_points),
        })

    # Intersection center = mean of all line endpoints. Used by the label-independent
    # "center" entry gate (TMC_COUNT_MODE=center): a crossing counts as an entry when
    # the vehicle is moving toward this center, regardless of how lines are labeled.
    #
    # Endpoints only, never the intermediate bend vertices: including them would drag
    # the centre toward whichever approach was drawn with more vertices and silently
    # change the gate for every already-processed study.
    global _INTERSECTION_CENTER, _ENTRY_EVENTS, _MIN_DISPLACEMENT_PX, _FIRST_POS
    _ENTRY_EVENTS = []
    _FIRST_POS = {}
    _LAST_POS.clear()
    _MIN_DISPLACEMENT_PX = float(os.environ.get("TMC_MIN_DISPLACEMENT_PX", "40"))
    _INTERSECTION_CENTER = intersection_center(LINES)

    counts = {line["name"]: 0 for line in LINES}
    counted_ids_per_line = {line["name"]: set() for line in LINES}
    entry_counted_ids = set()  # IDs que entraron desde afuera (para conteo total)
    prev_centroids = {}  # Stores (cx, cy) for tracking
    prev_wheels = {}  # Stores (wx, wy) for line crossing
    crossed_lines_by_id = {}
    turn_types_by_id = {}
    crossing_timestamps = {}
    detected_classes = {}
    class_counts_by_id = {}
    max_axle_count_by_id = {}  # Track maximum axle count per vehicle for FHWA classification

    # Axle detection statistics for debugging and analysis
    axle_detection_stats = {
        "trucks_detected": 0,           # Total trucks that crossed finish line
        "axle_detection_attempted": 0,  # Number of trucks where axle detection was tried
        "axle_detection_successful": 0, # Number of trucks with successful axle count
        "axle_counts_distribution": {},  # {axle_count: vehicle_count}
        "fhwa_class_distribution": {},   # {fhwa_class: count}
        "detection_by_truck_type": {     # Per truck type stats
            "single_unit_truck": {"attempted": 0, "successful": 0},
            "articulated_truck": {"attempted": 0, "successful": 0},
            "multi_articulated_truck": {"attempted": 0, "successful": 0},
        }
    }

    # Initialize track interpolator for handling occlusions
    track_interpolator = TrackInterpolator(max_missing_frames=15, min_track_length=3)
    overlap_stats = {"total_overlaps": 0, "frames_with_overlaps": 0, "frames_optimized": 0}
    
    # Track vehicles that have been processed by minute tracker
    minute_processed_vehicles = set()

    def get_centroid(box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def get_wheels_position(box):
        """Get the wheel position (bottom center) of the bounding box"""
        x1, y1, x2, y2 = box
        # Wheels are at the bottom center of the vehicle
        return int((x1 + x2) / 2), int(y2)

    def classify_turn_from_lines(crossing_data):
        if len(crossing_data) < 2:
            return 'invalid'
        
        # Ordenar por timestamp para obtener la secuencia correcta
        sorted_crossings = sorted(crossing_data, key=lambda x: x[1])  # (direction, timestamp)
        
        # Tomar la primera y última línea cruzada
        from_dir = sorted_crossings[0][0].upper()
        to_dir = sorted_crossings[-1][0].upper()

        if from_dir == to_dir:
            return 'u-turn'

        # Tabla corregida basada en perspectiva del conductor
        # Giro a la derecha = clockwise, Giro a la izquierda = counterclockwise  
        transitions = {
            ('NORTH', 'EAST'): 'left',   # Norte -> Este = giro izquierda
            ('NORTH', 'WEST'): 'right',  # Norte -> Oeste = giro derecha
            ('NORTH', 'SOUTH'): 'straight',
            ('EAST', 'SOUTH'): 'left',   # Este -> Sur = giro izquierda
            ('EAST', 'NORTH'): 'right',  # Este -> Norte = giro derecha
            ('EAST', 'WEST'): 'straight',
            ('SOUTH', 'WEST'): 'left',   # Sur -> Oeste = giro izquierda
            ('SOUTH', 'EAST'): 'right',  # Sur -> Este = giro derecha
            ('SOUTH', 'NORTH'): 'straight',
            ('WEST', 'NORTH'): 'left',   # Oeste -> Norte = giro izquierda
            ('WEST', 'SOUTH'): 'right',  # Oeste -> Sur = giro derecha
            ('WEST', 'EAST'): 'straight',
        }

        return transitions.get((from_dir, to_dir), 'unknown')

    def apply_heading_inference():
        """Credit a movement to entered-but-unclassified tracks using heading.

        Runs per trim period (tracker IDs restart, so positions must be
        consumed before they are cleared) and once at the end.
        Returns the number of movements recovered.
        """
        if not _HEADING_INFER_ON:
            return 0
        recovered = 0
        for _oid in list(entry_counted_ids):
            if _oid in turn_types_by_id:
                continue
            _cs = crossing_timestamps.get(_oid)
            if not _cs:
                continue
            _fp, _lp = _FIRST_POS.get(_oid), _LAST_POS.get(_oid)
            if not _fp or not _lp:
                continue
            _entry = _cs[0][0]
            _exit = infer_exit_line_by_heading(_entry, _fp, _lp, LINES)
            if not _exit:
                continue
            # reuse the existing turn table by appending the inferred crossing
            import time as _tm
            _cs.append((_exit, _tm.time()))
            _turn = classify_turn_from_lines(_cs)
            if _turn in ("invalid", "unknown"):
                _cs.pop()                 # leave the track untouched
                continue
            turn_types_by_id[_oid] = _turn
            crossed_lines_by_id.setdefault(_oid, []).append(_exit)
            recovered += 1
        return recovered

    _heading_recovered = 0

    # Video capture already initialized above for model manager
    current_frame = 0
    start_time = time.time()
    last_progress_sent = -1

    # Progress tracking: frames actually processed (not just video position)
    frames_processed_total = 0

    # Calculate frame ranges from trim periods
    frame_ranges = []
    if trim_periods:
        frame_ranges = calculate_frame_ranges_from_seconds(trim_periods, fps, total_frames)
        if frame_ranges:
            print(f"🎬 Trimming enabled: processing {len(frame_ranges)} period(s)")
            total_processing_frames = sum(r['end_frame'] - r['start_frame'] for r in frame_ranges)
            print(f"   Total frames to process: {total_processing_frames} / {total_frames} ({total_processing_frames/total_frames*100:.1f}%)")
        else:
            print("⚠️ No valid frame ranges, processing entire video")
    else:
        print("📊 No trimming specified, processing entire video")

    # Calculate total_processing_frames for normal mode too (for unified progress calculation)
    if not frame_ranges:
        total_processing_frames = total_frames

    # Initialize minute tracker if callback provided
    minute_tracker = None
    if minute_batch_callback:
        # Use provided video_uuid or generate one from filename
        import uuid
        if not video_uuid:
            video_filename = os.path.basename(VIDEO_PATH)
            video_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, video_filename))
            print(f"⚠️ No video_uuid provided, generated: {video_uuid}")
        else:
            print(f"✅ Using provided video_uuid: {video_uuid}")
            
        minute_tracker = MinuteTracker(fps, video_uuid, minute_batch_callback)
        print(f"📊 Enhanced minute tracking enabled for video {video_uuid}")

    # Initialize pedestrian/bicycle processor if crosswalks are configured
    ped_processor = None
    if crosswalks_config and pedestrian_model_path:
        crosswalk_proc = CrosswalkProcessor(crosswalks_config, fps)
        crosswalk_minute_tracker = CrosswalkMinuteTracker(fps)
        if minute_tracker:
            minute_tracker.set_crosswalk_tracker(crosswalk_minute_tracker)
        ped_processor = PedestrianProcessor(
            model_path=pedestrian_model_path,
            crosswalk_proc=crosswalk_proc,
            crosswalk_minute_tracker=crosswalk_minute_tracker,
            fps=fps,
            img_size=img_size,
        )
        print(f"🚶 Crosswalk processing enabled with {len(crosswalks_config)} crosswalk(s)")
    elif crosswalks_config and not pedestrian_model_path:
        print(f"⚠️ Crosswalks configured ({len(crosswalks_config)}) but no pedestrian model path provided")
    elif pedestrian_model_path and not crosswalks_config:
        print(f"⚠️ Pedestrian model path provided but no crosswalks configured")

    # Initialize video writer if output video is requested.
    # opencv-python-headless cannot open an H.264 encoder in this image (its bundled
    # ffmpeg lacks libx264), so cv2.VideoWriter('H264') silently falls back to the
    # unplayable MPEG-4 Part 2 (mp4v). We instead encode H.264 directly via the
    # system ffmpeg CLI (libx264) using FFmpegH264Writer -> a browser-playable file
    # with no giant mp4v intermediate and no post-transcode.
    video_writer = None
    if generate_video_output and output_video_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Reduce output resolution if too large, for smaller files.
        if width > 1920 or height > 1080:
            scale_factor = min(1920 / width, 1080 / height)
            width = int(width * scale_factor)
            height = int(height * scale_factor)
            print(f"📉 Scaling output resolution to {width}x{height} for compression")

        # Cap FPS for additional compression.
        output_fps = min(fps, 15)  # Cap at 15 FPS for traffic analysis
        if output_fps != fps:
            print(f"📉 Reducing output FPS from {fps} to {output_fps} for compression")

        video_writer = FFmpegH264Writer(output_video_path, output_fps, width, height,
                                        crf=26, preset="veryfast")
        if not video_writer.isOpened():
            print("❌ Could not initialize TMC H.264 video writer (ffmpeg/libx264 unavailable)")
            video_writer = None
            generate_video_output = False
    
    # Helper function to send seeking progress
    def send_seeking_progress():
        """
        Send progress during seeking phase.
        For trimming mode: Always send 0% to avoid oscillation when processing starts.
        For normal mode: Send actual position (backward compatible).
        """
        if progress_callback:
            elapsed_time = time.time() - start_time
            # For trimming: don't show seeking progress to avoid oscillation
            # Seeking is fast and doesn't count as actual work
            if frame_ranges:
                seek_progress = 0  # Always 0% during seeking in trimming mode
            else:
                seek_progress = int((current_frame / total_frames) * 100)

            progress_callback({
                "progress": seek_progress,
                "estimatedTimeRemaining": 0,
                "status": "seeking"
            })
            # Only log at major seeking milestones to reduce log noise
            if frame_ranges:
                # For trimming: calculate seeking progress for logging
                start_frame = frame_ranges[0]['start_frame']
                seek_pct = int((current_frame / start_frame) * 100) if start_frame > 0 else 0
                if seek_pct in [25, 50, 75] or current_frame >= start_frame - 1000:
                    print(f"⏩ SEEKING: Frame {current_frame}/{total_frames} ({seek_pct}% of seeking phase)")
            else:
                # For normal mode: log every 25%
                if seek_progress % 25 == 0 and seek_progress > 0:
                    print(f"⏩ SEEKING: Frame {current_frame}/{total_frames} (showing {seek_progress}% to user)")

    # Helper function to reset tracker state
    def reset_tracker():
        """Reset YOLO tracker to start fresh tracking for new period"""
        if model.predictor is not None:
            model.predictor.trackers = [None]
            print("🔄 YOLO tracker reset - previous tracking state cleared")
        else:
            print("🔄 YOLO tracker not initialized yet (first period)")

    # Helper function for progress calculation
    def calculate_and_send_progress():
        """
        Calculate progress based on actual frames processed (trimming-aware).

        For trimmed videos:
            progress = frames_processed_total / total_processing_frames
        For normal videos:
            progress = current_frame / total_frames (backward compatible)

        Ensures progress never decreases and respects 5% threshold.
        """
        nonlocal last_progress_sent

        if not progress_callback or total_frames == 0:
            return

        # Calculate progress based on mode
        if frame_ranges:
            # TRIMMING MODE: Use frames actually processed
            if total_processing_frames > 0:
                progress = int((frames_processed_total / total_processing_frames) * 100)
            else:
                progress = 0
        else:
            # NORMAL MODE: Use video position (backward compatible)
            progress = int((current_frame / total_frames) * 100)

        # Ensure progress never exceeds 100% or decreases
        progress = min(100, max(0, progress))

        # CRITICAL: Prevent progress from going backwards
        if progress < last_progress_sent:
            print(f"⚠️  PROGRESS BACKWARDS PREVENTED: {progress}% < {last_progress_sent}% (frames_processed={frames_processed_total}, current_frame={current_frame})")
            return  # Don't send backwards progress

        # Send progress every 1%
        if progress >= last_progress_sent + 1 and progress < 100:
            elapsed_time = time.time() - start_time

            # Calculate time estimate
            if progress > 0:
                if frame_ranges and total_processing_frames > 0:
                    # TRIMMING MODE: Estimate based on frames processed, not video position
                    estimated_total_time = elapsed_time / (frames_processed_total / total_processing_frames)
                else:
                    # NORMAL MODE: Use progress percentage
                    estimated_total_time = elapsed_time / (progress / 100)

                estimated_remaining_time = int(estimated_total_time - elapsed_time)
            else:
                estimated_remaining_time = 0

            # Debug logging
            mode = "TRIM" if frame_ranges else "NORMAL"
            print(f"📊 PROGRESS UPDATE [{mode}]: {progress}% | Elapsed: {elapsed_time:.1f}s | ETA: {estimated_remaining_time}s ({estimated_remaining_time/60:.1f} min)")
            if frame_ranges:
                print(f"   └─ Frames: {frames_processed_total}/{total_processing_frames} processed | Current position: {current_frame}/{total_frames}")
            else:
                print(f"   └─ Frames: {current_frame}/{total_frames}")

            progress_callback({
                "progress": progress,
                "estimatedTimeRemaining": max(0, estimated_remaining_time)
            })
            last_progress_sent = progress

    # Main processing logic with frame-skipping support
    if frame_ranges:
        # TRIMMING MODE: Process only specified periods with frame-skipping
        print("🎬 Starting trimmed video processing")
        print(f"📊 PROGRESS TRACKING CONFIG:")
        print(f"   └─ Total video frames: {total_frames}")
        print(f"   └─ Frames to process: {total_processing_frames}")
        print(f"   └─ Trim coverage: {total_processing_frames/total_frames*100:.1f}%")
        print(f"   └─ Progress calculation: frames_processed_total / {total_processing_frames}")

        for period_idx, period in enumerate(frame_ranges):
            start_frame = period["start_frame"]
            end_frame = period["end_frame"]
            period_duration = (period["end_seconds"] - period["start_seconds"]) / 60  # minutes

            print(f"\n📍 Period {period_idx + 1}/{len(frame_ranges)}")
            print(f"   Frames: {start_frame} - {end_frame} ({end_frame - start_frame} frames)")
            print(f"   Time: {period['start_seconds']:.1f}s - {period['end_seconds']:.1f}s ({period_duration:.1f} min)")

            # CRITICAL: Reset tracker at start of each period
            reset_tracker()
            if ped_processor is not None:
                ped_processor.reset_tracker()
                ped_processor.crosswalk_proc.reset_state()

            # Clear previous positions to prevent cross-period tracking
            _heading_recovered += apply_heading_inference()  # consume before reset
            prev_centroids.clear()
            prev_wheels.clear()
            _FIRST_POS.clear()  # tracker IDs restart per period; drop stale origins
            _LAST_POS.clear()
            print("🧹 Previous positions cleared for new period")

            # Skip frames until we reach the start of this period (frame-skipping)
            ret = True  # Initialize to True - if no seeking needed, we're ready to process
            while current_frame < start_frame:
                ret, _ = cap.read()  # Read but don't process
                if not ret:
                    print(f"⚠️ Video ended at frame {current_frame} while seeking to {start_frame}")
                    break

                current_frame += 1

                # Progress update every 1000 frames during seeking
                if current_frame % 1000 == 0:
                    send_seeking_progress()
                    print(f"⏩ Seeking: {current_frame}/{start_frame} frames ({current_frame/start_frame*100:.1f}%)")

            if not ret:
                print(f"⚠️ Could not reach period {period_idx + 1}, skipping")
                continue

            print(f"✅ Reached start of period {period_idx + 1} at frame {current_frame}")

            # Process frames in this period
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ Video ended at frame {current_frame} during period {period_idx + 1}")
                    break

                # YOLO tracking
                results = model.track(
                    bg_provider.stack(frame, current_frame) if bg_provider else frame,
                    persist=True, conf=CONF_THRESHOLD, imgsz=img_size,
                    iou=IOU_THRESHOLD, tracker=tracker_config, verbose=False,
                )

                # Process detections (rest of existing logic)
                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy()
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()

                    # OPTIMIZATION: Only run expensive overlap detection on busy frames every 3 frames
                    # This reduces processing time by ~40% with minimal accuracy impact
                    if len(boxes) >= 10 and current_frame % 3 == 0:
                        # Apply overlap detection improvements
                        processed_boxes, processed_scores, processed_classes, processed_ids = post_process_detections(
                            boxes, scores, classes, ids
                        )

                        # Update overlap statistics
                        if len(processed_boxes) > 1:
                            frame_stats = analyze_overlap_patterns(processed_boxes, processed_ids, {})
                            if frame_stats['overlapping_pairs'] > 0:
                                overlap_stats["frames_with_overlaps"] += 1
                                overlap_stats["total_overlaps"] += frame_stats['overlapping_pairs']

                        # Use processed detections for tracking
                        boxes = processed_boxes
                        ids = processed_ids if processed_ids is not None else ids
                        classes = processed_classes
                        scores = processed_scores
                    else:
                        # Skip overlap detection for low-traffic frames or non-sampled frames
                        # Use original detections to avoid unnecessary O(n²) operations
                        processed_boxes = boxes
                        processed_ids = ids
                        processed_classes = classes
                        overlap_stats["frames_optimized"] += 1

                    for i, box in enumerate(boxes):
                        obj_id = int(ids[i])
                        class_id = int(classes[i])
                        raw_class_name = model.names[class_id]
                        if raw_class_name in _VEHICLE_MODEL_EXCLUDE_CLASSES:
                            continue
                        # Confidence-weighted class vote over the track's whole life;
                        # the truck sub-classifier runs once per articulated track.
                        if raw_class_name == "articulated_truck" and truck_classifier:
                            if obj_id not in artic_subtype_by_id:
                                artic_subtype_by_id[obj_id] = truck_classifier.classify(frame, box)
                            raw_class_name = artic_subtype_by_id[obj_id]
                        _vs = class_vote_scores.setdefault(obj_id, {})
                        _conf = float(scores[i]) if i < len(scores) else 0.3
                        _vs[raw_class_name] = _vs.get(raw_class_name, 0.0) + _conf
                        class_name = max(_vs, key=_vs.get)
                        class_counts_by_id[obj_id] = class_name
                        if class_name in _VEHICLE_MODEL_EXCLUDE_CLASSES:
                            continue
                        cx, cy = get_centroid(box)
                        wx, wy = get_wheels_position(box)

                        # Detect axles for trucks (accumulate max across frames)
                        # Sample every 5 frames to reduce computation while maintaining accuracy
                        # Note: We do NOT store FHWA suffix in class_counts_by_id here to avoid breaking
                        # subsequent axle detection checks. FHWA is computed on-demand for visualization/output.
                        if axle_classifier and current_frame % 5 == 0 and class_name in ("single_unit_truck", "articulated_truck", "multi_articulated_truck"):
                            axle_count = axle_classifier.detect_axles(frame, box)
                            if axle_count is not None:
                                current_max = max_axle_count_by_id.get(obj_id, 0)
                                max_axle_count_by_id[obj_id] = max(current_max, axle_count)

                        process_single_detection(
                            obj_id, class_name, cx, cy, wx, wy, current_frame,
                            prev_wheels, prev_centroids, counted_ids_per_line,
                            entry_counted_ids, crossed_lines_by_id, crossing_timestamps,
                            turn_types_by_id, detected_classes, class_counts_by_id,
                            LINES, is_entering_from_outside, classify_turn_from_lines,
                            track_interpolator, counts=counts,
                        )

                # Pedestrian/bicycle model inference (delegated to PedestrianProcessor)
                if ped_processor is not None:
                    ped_frame_result = ped_processor.process_frame(frame, current_frame)
                    for bike in ped_frame_result.bicycle_detections:
                        process_single_detection(
                            bike.namespaced_id, bike.class_name, bike.cx, bike.cy,
                            bike.cx, bike.cy, current_frame,
                            prev_wheels, prev_centroids, counted_ids_per_line,
                            entry_counted_ids, crossed_lines_by_id, crossing_timestamps,
                            turn_types_by_id, detected_classes, class_counts_by_id,
                            LINES, is_entering_from_outside, classify_turn_from_lines,
                            counts=counts,
                        )

                # Handle missing detections with track interpolation
                # Clean up old tracks to prevent memory buildup
                if current_frame % 30 == 0:  # Every 30 frames
                    track_interpolator.cleanup_old_tracks(current_frame, max_age=150)

                # Add visualizations if generating output video
                if generate_video_output and video_writer:
                    # Draw detection boxes and tracking
                    if results[0].boxes.id is not None:
                        ids = results[0].boxes.id.cpu().numpy()
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        vis_classes = results[0].boxes.cls.cpu().numpy()

                        for i, box in enumerate(boxes):
                            obj_id = int(ids[i])
                            raw_cls = model.names[int(vis_classes[i])]
                            if raw_cls in _VEHICLE_MODEL_EXCLUDE_CLASSES:
                                continue
                            cls_name = class_counts_by_id.get(obj_id, raw_cls)
                            if cls_name in _VEHICLE_MODEL_EXCLUDE_CLASSES:
                                continue
                            # Compute FHWA suffix on-demand for visualization
                            if axle_classifier and obj_id in max_axle_count_by_id:
                                fhwa_viz = axle_classifier.get_fhwa_class(cls_name, max_axle_count_by_id[obj_id])
                                if fhwa_viz is not None:
                                    cls_name = f"{cls_name}_fhwa{fhwa_viz}"
                            x1, y1, x2, y2 = box
                            cx, cy = get_centroid(box)

                            # Draw bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                            # Draw centroid
                            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                            # Draw ID, class, and turn type if available
                            label = f'{cls_name} ID {obj_id}'
                            if obj_id in turn_types_by_id:
                                label += f' | {turn_types_by_id[obj_id]}'
                            cv2.putText(frame, label, (cx, cy - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Draw lines
                    for line in LINES:
                        name = line["name"]
                        pts = np.array(line["points"], dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], False, (0, 255, 255), 3)

                        # Draw line label and count
                        mid_x, mid_y = line_label_anchor(line["points"])
                        cv2.putText(frame, f'{name}: {counts[name]}', (mid_x, mid_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Draw summary stats
                    total_current = sum(counts.values())
                    turn_summary = dict(Counter(turn_types_by_id.values()))
                    y_pos = 30
                    cv2.putText(frame, f'Total Crossings: {total_current}', (20, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    for turn_type, count in turn_summary.items():
                        y_pos += 25
                        cv2.putText(frame, f'{turn_type}: {count}', (20, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Draw pedestrian/bicycle detections and crosswalk overlays
                    if ped_processor is not None:
                        y_pos = ped_processor.draw_visualizations(frame, y_pos)

                    # Resize frame if needed for compression
                    if width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or height != int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                    # Write frame to output video
                    video_writer.write(frame)

                # Update minute tracker with vehicle detections that have complete movement data
                if minute_tracker:
                    # Process vehicles that have completed their movement (have both origin and turn data)
                    for vehicle_id in turn_types_by_id:
                        # Only process if vehicle has crossed lines and we know its movement
                        if vehicle_id in crossing_timestamps and len(crossing_timestamps[vehicle_id]) >= 2:
                            # Only process vehicles that entered from outside (to match final results)
                            if vehicle_id in entry_counted_ids:
                                # Only process each vehicle once for minute tracking
                                if vehicle_id not in minute_processed_vehicles:
                                    minute_processed_vehicles.add(vehicle_id)

                                    # Get vehicle class (use detected_classes for consistency with final results)
                                    vehicle_class = detected_classes.get(vehicle_id, 'unknown')

                                    # Refine truck class with FHWA-specific label if axle data available
                                    if axle_classifier and vehicle_id in max_axle_count_by_id:
                                        if vehicle_class in ("single_unit_truck", "articulated_truck", "multi_articulated_truck"):
                                            axle_count = max_axle_count_by_id[vehicle_id]
                                            fhwa_class = axle_classifier.get_fhwa_class(vehicle_class, axle_count)
                                            if fhwa_class is not None:
                                                vehicle_class = f"{vehicle_class}_fhwa{fhwa_class}"

                                    # Get origin direction (first line crossed)
                                    origin_direction = crossing_timestamps[vehicle_id][0][0].upper()

                                    # Get turn type
                                    turn_type = turn_types_by_id[vehicle_id]

                                    # Process this vehicle detection
                                    minute_tracker.process_vehicle_detection(
                                        current_frame,
                                        vehicle_id,
                                        vehicle_class,
                                        origin_direction,
                                        turn_type
                                    )

                                    # Clean up axle data for this vehicle to free memory
                                    max_axle_count_by_id.pop(vehicle_id, None)

                # Progress tracking
                current_frame += 1
                frames_processed_total += 1  # Track actual frames processed
                calculate_and_send_progress()

            print(f"✅ Completed period {period_idx + 1}/{len(frame_ranges)}")

        print("\n✅ All trim periods processed")

    else:
        # NORMAL MODE: Process entire video (existing logic)
        print("📊 Processing entire video (no trimming)")
        print(f"📊 PROGRESS TRACKING CONFIG:")
        print(f"   └─ Total frames: {total_frames}")
        print(f"   └─ Progress calculation: current_frame / {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO tracking
            results = model.track(
                bg_provider.stack(frame, current_frame) if bg_provider else frame,
                persist=True, conf=CONF_THRESHOLD, imgsz=img_size,
                iou=IOU_THRESHOLD, tracker=tracker_config, verbose=False,
            )

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()

                # OPTIMIZATION: Only run expensive overlap detection on busy frames every 3 frames
                # This reduces processing time by ~40% with minimal accuracy impact
                if len(boxes) >= 10 and current_frame % 3 == 0:
                    # Apply overlap detection improvements
                    processed_boxes, processed_scores, processed_classes, processed_ids = post_process_detections(
                        boxes, scores, classes, ids
                    )

                    # Update overlap statistics
                    if len(processed_boxes) > 1:
                        frame_stats = analyze_overlap_patterns(processed_boxes, processed_ids, {})
                        if frame_stats['overlapping_pairs'] > 0:
                            overlap_stats["frames_with_overlaps"] += 1
                            overlap_stats["total_overlaps"] += frame_stats['overlapping_pairs']

                    # Use processed detections for tracking
                    boxes = processed_boxes
                    ids = processed_ids if processed_ids is not None else ids
                    classes = processed_classes
                    scores = processed_scores
                else:
                    # Skip overlap detection for low-traffic frames or non-sampled frames
                    # Use original detections to avoid unnecessary O(n²) operations
                    processed_boxes = boxes
                    processed_ids = ids
                    processed_classes = classes
                    overlap_stats["frames_optimized"] += 1

                for i, box in enumerate(boxes):
                    obj_id = int(ids[i])
                    class_id = int(classes[i])
                    raw_class_name = model.names[class_id]
                    if raw_class_name in _VEHICLE_MODEL_EXCLUDE_CLASSES:
                        continue
                    # Confidence-weighted class vote over the track's whole life;
                    # the truck sub-classifier runs once per articulated track.
                    if raw_class_name == "articulated_truck" and truck_classifier:
                        if obj_id not in artic_subtype_by_id:
                            artic_subtype_by_id[obj_id] = truck_classifier.classify(frame, box)
                        raw_class_name = artic_subtype_by_id[obj_id]
                    _vs = class_vote_scores.setdefault(obj_id, {})
                    _conf = float(scores[i]) if i < len(scores) else 0.3
                    _vs[raw_class_name] = _vs.get(raw_class_name, 0.0) + _conf
                    class_name = max(_vs, key=_vs.get)
                    class_counts_by_id[obj_id] = class_name
                    if class_name in _VEHICLE_MODEL_EXCLUDE_CLASSES:
                        continue
                    cx, cy = get_centroid(box)
                    wx, wy = get_wheels_position(box)

                    # Detect axles for trucks (accumulate max across frames) - normal mode
                    # Sample every 5 frames to reduce computation while maintaining accuracy
                    # Note: We do NOT store FHWA suffix in class_counts_by_id here to avoid breaking
                    # subsequent axle detection checks. FHWA is computed on-demand for visualization/output.
                    if axle_classifier and current_frame % 5 == 0 and class_name in ("single_unit_truck", "articulated_truck", "multi_articulated_truck"):
                        axle_count = axle_classifier.detect_axles(frame, box)
                        if axle_count is not None:
                            current_max = max_axle_count_by_id.get(obj_id, 0)
                            max_axle_count_by_id[obj_id] = max(current_max, axle_count)

                    process_single_detection(
                        obj_id, class_name, cx, cy, wx, wy, current_frame,
                        prev_wheels, prev_centroids, counted_ids_per_line,
                        entry_counted_ids, crossed_lines_by_id, crossing_timestamps,
                        turn_types_by_id, detected_classes, class_counts_by_id,
                        LINES, is_entering_from_outside, classify_turn_from_lines,
                        track_interpolator, counts=counts,
                    )

            # Pedestrian/bicycle model inference (delegated to PedestrianProcessor)
            if ped_processor is not None:
                ped_frame_result = ped_processor.process_frame(frame, current_frame)
                for bike in ped_frame_result.bicycle_detections:
                    process_single_detection(
                        bike.namespaced_id, bike.class_name, bike.cx, bike.cy,
                        bike.cx, bike.cy, current_frame,
                        prev_wheels, prev_centroids, counted_ids_per_line,
                        entry_counted_ids, crossed_lines_by_id, crossing_timestamps,
                        turn_types_by_id, detected_classes, class_counts_by_id,
                        LINES, is_entering_from_outside, classify_turn_from_lines,
                        counts=counts,
                    )

            # Handle missing detections with track interpolation
            # Clean up old tracks to prevent memory buildup
            if current_frame % 30 == 0:  # Every 30 frames
                track_interpolator.cleanup_old_tracks(current_frame, max_age=150)

            # Add visualizations if generating output video
            if generate_video_output and video_writer:
                # Draw detection boxes and tracking
                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy()
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    vis_classes = results[0].boxes.cls.cpu().numpy()

                    for i, box in enumerate(boxes):
                        obj_id = int(ids[i])
                        raw_cls = model.names[int(vis_classes[i])]
                        if raw_cls in _VEHICLE_MODEL_EXCLUDE_CLASSES:
                            continue
                        cls_name = class_counts_by_id.get(obj_id, raw_cls)
                        if cls_name in _VEHICLE_MODEL_EXCLUDE_CLASSES:
                            continue
                        # Compute FHWA suffix on-demand for visualization
                        if axle_classifier and obj_id in max_axle_count_by_id:
                            fhwa_viz = axle_classifier.get_fhwa_class(cls_name, max_axle_count_by_id[obj_id])
                            if fhwa_viz is not None:
                                cls_name = f"{cls_name}_fhwa{fhwa_viz}"
                        x1, y1, x2, y2 = box
                        cx, cy = get_centroid(box)

                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Draw centroid
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                        # Draw ID, class, and turn type if available
                        label = f'{cls_name} ID {obj_id}'
                        if obj_id in turn_types_by_id:
                            label += f' | {turn_types_by_id[obj_id]}'
                        cv2.putText(frame, label, (cx, cy - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw lines
                for line in LINES:
                    name = line["name"]
                    pts = np.array(line["points"], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], False, (0, 255, 255), 3)

                    # Draw line label and count
                    mid_x, mid_y = line_label_anchor(line["points"])
                    cv2.putText(frame, f'{name}: {counts[name]}', (mid_x, mid_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw summary stats
                total_current = sum(counts.values())
                turn_summary = dict(Counter(turn_types_by_id.values()))
                y_pos = 30
                cv2.putText(frame, f'Total Crossings: {total_current}', (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                for turn_type, count in turn_summary.items():
                    y_pos += 25
                    cv2.putText(frame, f'{turn_type}: {count}', (20, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw pedestrian/bicycle detections and crosswalk overlays
                if ped_processor is not None:
                    y_pos = ped_processor.draw_visualizations(frame, y_pos)

                # Resize frame if needed for compression
                if width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or height != int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                # Write frame to output video
                video_writer.write(frame)

            # Update minute tracker with vehicle detections that have complete movement data
            if minute_tracker:
                # Process vehicles that have completed their movement (have both origin and turn data)
                for vehicle_id in turn_types_by_id:
                    # Only process if vehicle has crossed lines and we know its movement
                    if vehicle_id in crossing_timestamps and len(crossing_timestamps[vehicle_id]) >= 2:
                        # Only process vehicles that entered from outside (to match final results)
                        if vehicle_id in entry_counted_ids:
                            # Only process each vehicle once for minute tracking
                            if vehicle_id not in minute_processed_vehicles:
                                minute_processed_vehicles.add(vehicle_id)

                                # Get vehicle class (use detected_classes for consistency with final results)
                                vehicle_class = detected_classes.get(vehicle_id, 'unknown')

                                # Refine truck class with FHWA-specific label if axle data available - normal mode
                                if axle_classifier and vehicle_id in max_axle_count_by_id:
                                    if vehicle_class in ("single_unit_truck", "articulated_truck", "multi_articulated_truck"):
                                        axle_count = max_axle_count_by_id[vehicle_id]
                                        fhwa_class = axle_classifier.get_fhwa_class(vehicle_class, axle_count)
                                        if fhwa_class is not None:
                                            vehicle_class = f"{vehicle_class}_fhwa{fhwa_class}"

                                # Get origin direction (first line crossed)
                                origin_direction = crossing_timestamps[vehicle_id][0][0].upper()

                                # Get turn type
                                turn_type = turn_types_by_id[vehicle_id]

                                # Process this vehicle detection
                                minute_tracker.process_vehicle_detection(
                                    current_frame,
                                    vehicle_id,
                                    vehicle_class,
                                    origin_direction,
                                    turn_type
                                )

                                # Clean up axle data for this vehicle to free memory
                                max_axle_count_by_id.pop(vehicle_id, None)

            # Progress tracking
            current_frame += 1
            frames_processed_total += 1  # Track actual frames processed (same as current_frame in normal mode)
            calculate_and_send_progress()

    # Send final 100% progress
    if progress_callback:
        progress_callback({
            "progress": 100,
            "estimatedTimeRemaining": 0
        })
        print(f"✅ TMC Processing complete: {frames_processed_total} frames processed")

    # Log optimization statistics
    optimization_ratio = overlap_stats["frames_optimized"] / max(1, frames_processed_total) * 100
    print(f"⚡ Performance optimization: {overlap_stats['frames_optimized']}/{frames_processed_total} frames optimized ({optimization_ratio:.1f}%)")
    print(f"   └─ Overlap detection ran on {frames_processed_total - overlap_stats['frames_optimized']} frames")
    print(f"   └─ Strategy: Skip low-traffic (<10 vehicles) + sample every 3 frames")

    cap.release()
    if video_writer:
        video_writer.release()

    # Clean up temporary tracker config
    try:
        os.unlink(tracker_config)
    except FileNotFoundError:
        pass

    # CRITICAL: Release YOLO model(s) and GPU memory to prevent accumulation
    # RunPod workers are reused — on exception, the worker process is recycled
    # and the OS reclaims all resources including the temp file in /tmp.
    print("🧹 Releasing YOLO model(s) and GPU memory...")
    del model
    if ped_processor is not None:
        ped_processor.release()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU memory cache cleared")

    # Post procesamiento con lógica corregida
    # Usar entry_counted_ids para el conteo total (solo vehículos que entraron desde afuera)
    total_count = len(entry_counted_ids)

    _heading_recovered += apply_heading_inference()
    if _heading_recovered:
        print(f"🧭 Heading inference: {_heading_recovered} movement(s) recovered "
              f"from entered-but-unclassified tracks")

    # Class vote finalization: replace the entry-time label with the track-life
    # confidence-weighted winner (must run BEFORE the FHWA refinement below,
    # which reads the base class names).
    for _oid in list(detected_classes.keys()):
        _vs = class_vote_scores.get(_oid)
        if _vs:
            detected_classes[_oid] = max(_vs, key=_vs.get)

    # Refine detected_classes with FHWA-specific labels for trucks when axle data is available
    # Also collect axle detection statistics for analysis
    if axle_classifier:
        for obj_id in list(detected_classes.keys()):
            class_name = detected_classes[obj_id]
            if class_name in ("single_unit_truck", "articulated_truck", "multi_articulated_truck"):
                # Count this truck
                axle_detection_stats["trucks_detected"] += 1
                axle_detection_stats["detection_by_truck_type"][class_name]["attempted"] += 1
                axle_detection_stats["axle_detection_attempted"] += 1

                if obj_id in max_axle_count_by_id:
                    axle_count = max_axle_count_by_id[obj_id]
                    axle_detection_stats["axle_detection_successful"] += 1
                    axle_detection_stats["detection_by_truck_type"][class_name]["successful"] += 1

                    # Track axle count distribution
                    axle_detection_stats["axle_counts_distribution"][axle_count] = \
                        axle_detection_stats["axle_counts_distribution"].get(axle_count, 0) + 1

                    fhwa_class = axle_classifier.get_fhwa_class(class_name, axle_count)
                    if fhwa_class is not None:
                        detected_classes[obj_id] = f"{class_name}_fhwa{fhwa_class}"
                        # Track FHWA class distribution
                        axle_detection_stats["fhwa_class_distribution"][fhwa_class] = \
                            axle_detection_stats["fhwa_class_distribution"].get(fhwa_class, 0) + 1
        # Clear remaining axle data to free memory
        max_axle_count_by_id.clear()

    # Calculate success rate
    if axle_detection_stats["axle_detection_attempted"] > 0:
        axle_detection_stats["success_rate"] = round(
            axle_detection_stats["axle_detection_successful"] /
            axle_detection_stats["axle_detection_attempted"] * 100, 1
        )
    else:
        axle_detection_stats["success_rate"] = None

    # Convert detected_classes from {obj_id: class_name} to {class_name: count}
    class_summary = Counter(detected_classes.values())
    
    # Calcular turns incluyendo straight
    turn_counts = Counter(turn_types_by_id.values())
    turns_dict = dict(turn_counts)
    
    # Ensure all turn categories exist
    if 'left' not in turns_dict:
        turns_dict['left'] = 0
    if 'right' not in turns_dict:
        turns_dict['right'] = 0
    if 'straight' not in turns_dict:
        turns_dict['straight'] = 0
    if 'u-turn' not in turns_dict:
        turns_dict['u-turn'] = 0
    
    # Build new vehicle-class-first structure first
    vehicles = build_analysis_by_vehicle_class(
        detected_classes, turn_types_by_id, crossing_timestamps, crossed_lines_by_id
    )
    
    # Calculate vehicles with complete movement data (from vehicles analysis)
    vehicles_with_movement = 0
    for vehicle_class, origins in vehicles.items():
        if vehicle_class == 'total':
            continue
        for origin_data in origins.values():
            vehicles_with_movement += sum(origin_data.values())
    
    # Si no hay straight explícitos, calcularlos como vehicles_with_movement - left - right - u-turn
    if turns_dict['straight'] == 0:
        left_count = turns_dict.get('left', 0)
        right_count = turns_dict.get('right', 0)
        uturn_count = turns_dict.get('u-turn', 0)
        turns_dict['straight'] = max(0, vehicles_with_movement - left_count - right_count - uturn_count)
    
    # Finalize minute tracking if enabled
    video_duration_seconds = None
    if minute_tracker:
        video_duration_seconds = minute_tracker.finalize_processing()
        print(f"📊 Video duration calculated: {video_duration_seconds} seconds")

    # Finalize crosswalk tracking
    crosswalk_results = None
    crosswalk_totals = None
    if ped_processor is not None:
        crosswalk_results = ped_processor.get_crosswalk_results()
        crosswalk_totals = ped_processor.get_crosswalk_totals()
        ped_processor.finalize_crosswalk_minute_tracker()

    # Attribution residual, surfaced instead of padded into 'straight':
    # entry-counted tracks that never completed a classified movement.
    unclassified_tracks = sum(1 for _oid in entry_counted_ids
                              if _oid not in turn_types_by_id)

    return {
        # Original fields (backward compatibility)
        "counts": counts,
        "turns": turns_dict,
        "total": total_count,
        "totalcount": total_count,  # Added for clarity
        "detected_classes": dict(class_summary),

        # NEW: Analysis grouped by vehicle class first
        "vehicles": vehicles,

        "unclassified_tracks": unclassified_tracks,

        "validation": {
            "total_vehicles": total_count,
            "vehicles_with_movement": vehicles_with_movement,
            "unclassified_tracks": unclassified_tracks,
            "total_turns": sum(turns_dict.values()),
            "validation_passed": vehicles_with_movement == sum(turns_dict.values()),
            "entry_vehicles": len(entry_counted_ids),
            "total_crossings": sum(counts.values())
        },

        # NEW: Overlap detection statistics
        "overlap_analysis": {
            "frames_with_overlaps": overlap_stats["frames_with_overlaps"],
            "total_overlaps_detected": overlap_stats["total_overlaps"],
            "overlap_frame_ratio": overlap_stats["frames_with_overlaps"] / max(1, current_frame),
            "frames_optimized": overlap_stats["frames_optimized"],
            "optimization_ratio": overlap_stats["frames_optimized"] / max(1, current_frame),
            "processing_enhancements": {
                "soft_nms_applied": "conditional (busy frames only)",
                "track_interpolation": True,
                "confidence_adjustment": "conditional (busy frames only)",
                "optimization_strategy": "skip_low_traffic_and_sample_every_3_frames"
            }
        },

        # NEW: Axle detection statistics for FHWA classification debugging
        "axle_detection_stats": axle_detection_stats if axle_classifier else None,

        # Video metadata
        "video_metadata": {
            "duration_seconds": video_duration_seconds,
            "total_frames": current_frame,
            "fps": fps
        },

        # Crosswalk pedestrian/bicycle results (None if no crosswalks configured)
        "crosswalks": crosswalk_results,
        "crosswalk_totals": crosswalk_totals,
    }