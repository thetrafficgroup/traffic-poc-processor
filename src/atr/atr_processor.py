import cv2
import numpy as np
import json
import time
from shapely.geometry import Point, Polygon, LineString, box as shapely_box
from ultralytics import YOLO
from collections import OrderedDict, Counter, defaultdict
from .atr_minute_tracker import ATRMinuteTracker
import sys
import os
import torch
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.frame_utils import calculate_frame_ranges_from_seconds, validate_trim_periods

# === Centroid Tracker ===
class CentroidTracker:
    def __init__(self, max_disappeared=15, max_distance=150):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, input_centroids):
        if len(input_centroids) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.objects

        if len(self.objects) == 0:
            for centroid in input_centroids:
                self.register(centroid)
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = input_centroids[col]
                self.disappeared[objectID] = 0
                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)

            for row in unused_rows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)

            for col in unused_cols:
                self.register(input_centroids[col])

        return self.objects

def get_centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def get_wheels_position(box):
    """Get the wheel position (bottom center) of the bounding box"""
    x1, y1, x2, y2 = box
    # Wheels are at the bottom center of the vehicle
    return int((x1 + x2) / 2), int(y2)

def find_vehicle_lane(cx, cy, wx, wy, lane_polygons_buffered, bbox=None):
    """
    Find vehicle lane using wheels-priority approach.
    Falls back to centroid, then bottom-band intersection for large bboxes.
    """
    # Priority 1: wheels position
    if wx is not None and wy is not None:
        wheels_point = Point(wx, wy)
        for lane_id, poly in lane_polygons_buffered:
            if poly.covers(wheels_point):
                return lane_id

    # Priority 2: centroid
    centroid_point = Point(cx, cy)
    for lane_id, poly in lane_polygons_buffered:
        if poly.covers(centroid_point):
            return lane_id

    if bbox is not None:
        x1, y1, x2, y2 = bbox

        # Priority 3: bottom-band intersection (bottom 25% of bbox)
        band_top = y2 - (y2 - y1) * 0.25
        bottom_band = shapely_box(x1, band_top, x2, y2)
        best_lane, best_area = None, 0
        for lane_id, poly in lane_polygons_buffered:
            if poly.intersects(bottom_band):
                area = poly.intersection(bottom_band).area
                if area > best_area:
                    best_area = area
                    best_lane = lane_id
        if best_lane is not None:
            return best_lane

        # Priority 4: full bbox intersection (largest overlap wins)
        full_bbox = shapely_box(x1, y1, x2, y2)
        best_lane, best_area = None, 0
        for lane_id, poly in lane_polygons_buffered:
            if poly.intersects(full_bbox):
                area = poly.intersection(full_bbox).area
                if area > best_area:
                    best_area = area
                    best_lane = lane_id
        return best_lane

    return None

def dict_points_to_tuples(points):
    """Convert points from dict format to tuples, ensuring they are integers for OpenCV"""
    result = []
    for pt in points:
        if isinstance(pt, dict):
            x = int(round(pt["x"]))  # Convert float to int and round
            y = int(round(pt["y"]))
            result.append((x, y))
        else:
            x = int(round(pt[0])) if isinstance(pt[0], float) else pt[0]
            y = int(round(pt[1])) if isinstance(pt[1], float) else pt[1]
            result.append((x, y))
    return result


def point_side_of_line(p, a, b):
    """
    Determine which side of line ab point p is on.
    Returns >0 if p is left of ab, <0 if right, 0 if on line
    """
    # Ensure all coordinates are numeric (handle potential float/int mix)
    return (float(b[0]) - float(a[0])) * (float(p[1]) - float(a[1])) - (float(b[1]) - float(a[1])) * (float(p[0]) - float(a[0]))


def _match_track_to_detection(tracker_centroid, detections_map):
    """Pick the geometrically nearest detection within ±20 px (axis-aligned box)
    of the tracker centroid. Returns (detection_data, det_index, distance_px) or
    (None, None, None) when no detection is within the box.

    Replaces the legacy first-hit lookup (which iterated detections_map in
    confidence-descending insertion order and was systematically biased toward
    the highest-confidence detection — the engine of the H9 stale-centroid
    hijack and a cross-binding hazard for two tracks with nearby centroids).
    """
    cx, cy = tracker_centroid
    best = best_idx = best_d2 = None
    for det_idx, ((det_cx, det_cy), detection_data) in enumerate(detections_map.items()):
        if abs(det_cx - cx) >= 20 or abs(det_cy - cy) >= 20:
            continue
        d2 = (det_cx - cx) ** 2 + (det_cy - cy) ** 2
        if best_d2 is None or d2 < best_d2:
            best, best_idx, best_d2 = detection_data, det_idx, d2
    return best, best_idx, (best_d2 ** 0.5 if best_d2 is not None else None)


def find_vehicle_lane_with_source(cx, cy, wx, wy, lane_polygons_buffered, bbox=None):
    """Same priority order as find_vehicle_lane, but returns (lane_id, source).

    source ∈ {'wheels','centroid','bottom_band','full_bbox', None}. Sticky-lane is
    applied by the caller (it lives outside this function), so 'sticky' is never
    returned here.
    """
    if wx is not None and wy is not None:
        wheels_point = Point(wx, wy)
        for lane_id, poly in lane_polygons_buffered:
            if poly.covers(wheels_point):
                return lane_id, 'wheels'

    centroid_point = Point(cx, cy)
    for lane_id, poly in lane_polygons_buffered:
        if poly.covers(centroid_point):
            return lane_id, 'centroid'

    if bbox is not None:
        x1, y1, x2, y2 = bbox
        band_top = y2 - (y2 - y1) * 0.25
        bottom_band = shapely_box(x1, band_top, x2, y2)
        best_lane, best_area = None, 0
        for lane_id, poly in lane_polygons_buffered:
            if poly.intersects(bottom_band):
                area = poly.intersection(bottom_band).area
                if area > best_area:
                    best_area = area
                    best_lane = lane_id
        if best_lane is not None:
            return best_lane, 'bottom_band'

        full_bbox = shapely_box(x1, y1, x2, y2)
        best_lane, best_area = None, 0
        for lane_id, poly in lane_polygons_buffered:
            if poly.intersects(full_bbox):
                area = poly.intersection(full_bbox).area
                if area > best_area:
                    best_area = area
                    best_lane = lane_id
        if best_lane is not None:
            return best_lane, 'full_bbox'

    return None, None


def _atr_json_default(obj):
    """JSON serializer for numpy/shapely types that crop up in our payloads."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Type {type(obj).__name__} not JSON serializable")


class _ATRDebugEmitter:
    """Structured JSONL emitter for overcount diagnosis.

    Off by default. Enable with ATR_COUNT_DEBUG=1. Output is one JSON object per
    line, prefixed with [ATR_COUNT_DEBUG], printed to stdout (which log_capture.py
    tees to S3 — so no new artifact plumbing is needed).

    The five event families (see overcount.md §9):
      D0 run_config       — once at startup
      D1 frame_detections — diagnostic frames only
      D2 track_match      — per track, diagnostic frames only
      D3 count_decision   — every count-eligible decision (always)
      D4 debug_summary    — once at the end of the run
    """

    LOG_PREFIX = "[ATR_COUNT_DEBUG]"

    def __init__(self, video_uuid, fps, finish_linestring):
        self.enabled = os.environ.get("ATR_COUNT_DEBUG", "0") == "1"
        self.video_uuid = video_uuid
        self.fps = float(fps) if fps and fps > 0 else 30.0
        self.finish_linestring = finish_linestring
        try:
            self.band_px = float(os.environ.get("ATR_DEBUG_BAND_PX", "250"))
        except (TypeError, ValueError):
            self.band_px = 250.0

        # D4 aggregates
        self.accepted_counts = 0
        self.accepted_by_lane = defaultdict(int)
        self.accepted_by_class = defaultdict(int)
        self.suppressed_by_layer = defaultdict(int)
        self.cross_reason_counts = defaultdict(int)
        self.counts_from_disappeared_tracks = 0
        self.counts_with_bbox_none = 0
        self.max_simultaneous_tracks_near_finish_line = 0
        self.frames_with_multiple_finish_line_detections = 0
        self.object_ids_counted = []  # list of dicts

    def _ts(self, frame_count):
        return round(frame_count / self.fps, 3) if self.fps else None

    def _emit(self, payload):
        if not self.enabled:
            return
        try:
            print(f"{self.LOG_PREFIX} {json.dumps(payload, default=_atr_json_default)}")
        except Exception as e:  # pragma: no cover — diagnostic logging must not break processing
            print(f"{self.LOG_PREFIX} {{\"event\":\"emit_failed\",\"error\":\"{e!r}\"}}")

    # ---------------- D0 ----------------
    def emit_run_config(self, **fields):
        if not self.enabled:
            return
        payload = {"event": "run_config", "video_uuid": self.video_uuid}
        payload.update(fields)
        self._emit(payload)

    # ---------------- diagnostic-frame trigger ----------------
    def is_diagnostic_frame(self, detections_map, objects_dict):
        """True if any detection bbox intersects the finish line OR any tracked
        centroid is within band_px of it. The 'count decision occurs' trigger is
        implicitly covered: counting requires proximity to the finish line.
        """
        if not self.enabled or self.finish_linestring is None:
            return False
        for (_cx, _cy), data in detections_map.items():
            x1, y1, x2, y2 = data[:4]
            if shapely_box(x1, y1, x2, y2).intersects(self.finish_linestring):
                return True
        for centroid in objects_dict.values():
            cx, cy = centroid
            if Point(float(cx), float(cy)).distance(self.finish_linestring) <= self.band_px:
                return True
        return False

    # ---------------- D1 ----------------
    def emit_frame_detections(self, frame_count, detections_map, lane_polygons_buffered, tracker, objects_dict):
        if not self.enabled:
            return
        # tally derived counters that depend on per-frame state
        finish_line_dets = []
        if self.finish_linestring is not None:
            for (_cx, _cy), data in detections_map.items():
                x1, y1, x2, y2 = data[:4]
                if shapely_box(x1, y1, x2, y2).intersects(self.finish_linestring):
                    finish_line_dets.append(data)
        if len(finish_line_dets) >= 2:
            self.frames_with_multiple_finish_line_detections += 1

        tracks_near_line = 0
        if self.finish_linestring is not None:
            for centroid in objects_dict.values():
                cx, cy = centroid
                if Point(float(cx), float(cy)).distance(self.finish_linestring) <= self.band_px:
                    tracks_near_line += 1
        if tracks_near_line > self.max_simultaneous_tracks_near_finish_line:
            self.max_simultaneous_tracks_near_finish_line = tracks_near_line

        det_payloads = []
        for idx, ((det_cx, det_cy), data) in enumerate(detections_map.items()):
            x1, y1, x2, y2 = data[:4]
            class_name = data[4] if len(data) > 4 else None
            wx = data[5] if len(data) > 6 else None
            wy = data[6] if len(data) > 6 else None
            bbox_tuple = (float(x1), float(y1), float(x2), float(y2))
            intersects_line = (
                self.finish_linestring is not None
                and shapely_box(x1, y1, x2, y2).intersects(self.finish_linestring)
            )

            # All four lane fallbacks, independently — so we can see which path
            # would have selected a different lane (H7 split diagnosis).
            lane_by_wheels = None
            if wx is not None and wy is not None:
                wp = Point(wx, wy)
                for lid, poly in lane_polygons_buffered:
                    if poly.covers(wp):
                        lane_by_wheels = lid
                        break
            lane_by_centroid = None
            cp = Point(float(det_cx), float(det_cy))
            for lid, poly in lane_polygons_buffered:
                if poly.covers(cp):
                    lane_by_centroid = lid
                    break
            lane_by_bottom_band = None
            band_top = y2 - (y2 - y1) * 0.25
            bb_poly = shapely_box(x1, band_top, x2, y2)
            best_lane, best_area = None, 0
            for lid, poly in lane_polygons_buffered:
                if poly.intersects(bb_poly):
                    area = poly.intersection(bb_poly).area
                    if area > best_area:
                        best_area = area
                        best_lane = lid
            lane_by_bottom_band = best_lane
            lane_by_full_bbox = None
            full_poly = shapely_box(x1, y1, x2, y2)
            best_lane, best_area = None, 0
            for lid, poly in lane_polygons_buffered:
                if poly.intersects(full_poly):
                    area = poly.intersection(full_poly).area
                    if area > best_area:
                        best_area = area
                        best_lane = lid
            lane_by_full_bbox = best_lane

            det_payloads.append({
                "det_index": idx,
                "class_name": class_name,
                "bbox": bbox_tuple,
                "centroid": [int(det_cx), int(det_cy)],
                "wheels": [int(wx), int(wy)] if wx is not None and wy is not None else None,
                "bbox_intersects_finish_line": bool(intersects_line),
                "lane_by_wheels": lane_by_wheels,
                "lane_by_centroid": lane_by_centroid,
                "lane_by_bottom_band": lane_by_bottom_band,
                "lane_by_full_bbox": lane_by_full_bbox,
            })

        # Tracker state snapshot — closes the gap between matching-loop binding
        # and what tracker.update() actually decided this frame.
        tracker_state = {
            "active_ids": list(objects_dict.keys()),
            "disappeared_counts": {oid: int(tracker.disappeared.get(oid, 0))
                                    for oid in objects_dict.keys()},
            "next_object_id": int(tracker.nextObjectID),
            "tracks_within_band": tracks_near_line,
        }

        self._emit({
            "event": "frame_detections",
            "video_uuid": self.video_uuid,
            "frame": int(frame_count),
            "timestamp_seconds": self._ts(frame_count),
            "detection_count": len(detections_map),
            "detections": det_payloads,
            "tracker_state": tracker_state,
        })

    # ---------------- D2 ----------------
    def emit_track_match(self, frame_count, object_id, tracker_centroid, disappeared_count,
                         matched_detection_index, match_distance_px, matched_bbox,
                         matched_class, matched_wheels, cached_class, lane_id,
                         lane_source, previous_positions_tail, was_skipped_for_disappeared):
        if not self.enabled:
            return
        self._emit({
            "event": "track_match",
            "video_uuid": self.video_uuid,
            "frame": int(frame_count),
            "timestamp_seconds": self._ts(frame_count),
            "object_id": int(object_id),
            "tracker_centroid": [int(tracker_centroid[0]), int(tracker_centroid[1])],
            "disappeared_count": int(disappeared_count),
            "was_skipped_for_disappeared": bool(was_skipped_for_disappeared),
            "matched_detection_index": matched_detection_index,
            "match_distance_px": (
                round(float(match_distance_px), 2) if match_distance_px is not None else None
            ),
            "match_mode": "first_within_20",
            "match_failed_no_detection_within_20": matched_detection_index is None,
            "matched_bbox": (
                [float(matched_bbox[0]), float(matched_bbox[1]),
                 float(matched_bbox[2]), float(matched_bbox[3])]
                if matched_bbox is not None else None
            ),
            "matched_class": matched_class,
            "matched_wheels": (
                [int(matched_wheels[0]), int(matched_wheels[1])]
                if matched_wheels is not None and matched_wheels[0] is not None else None
            ),
            "cached_class": cached_class,
            "lane_id": lane_id,
            "lane_source": lane_source,
            "previous_positions_tail": [
                [int(p[0]), int(p[1])] for p in (previous_positions_tail or [])[-3:]
            ],
        })

    # ---------------- D3 helpers ----------------
    def compute_nearest_recent_bbox(self, current_bbox, recently_counted_bboxes, frame_count, fps):
        """Return the best dedup candidate (lowest age and highest overlap) for
        diagnosis even when it doesn't dominate."""
        if not self.enabled or current_bbox is None:
            return None
        cur_poly = shapely_box(current_bbox[0], current_bbox[1], current_bbox[2], current_bbox[3])
        cur_area = cur_poly.area
        best = None
        for stored_bbox, counted_at, *rest in recently_counted_bboxes:
            age = frame_count - counted_at
            stored_poly = shapely_box(*stored_bbox)
            if not cur_poly.intersects(stored_poly):
                continue
            inter = cur_poly.intersection(stored_poly).area
            min_area = min(cur_area, stored_poly.area)
            ratio = (inter / min_area) if min_area > 0 else 0.0
            union = cur_area + stored_poly.area - inter
            iou = (inter / union) if union > 0 else 0.0
            if best is None or ratio > best["overlap_min_area_ratio"]:
                best = {
                    "frame_age": int(age),
                    "overlap_min_area_ratio": round(float(ratio), 3),
                    "iou": round(float(iou), 3),
                }
        return best

    def compute_nearest_recent_position(self, lane_id, cx, cy, recently_counted_positions,
                                        frame_count, dist_threshold):
        if not self.enabled:
            return None
        best = None
        for prev_lane, prev_cx, prev_cy, counted_at in recently_counted_positions:
            if prev_lane != lane_id:
                continue
            d = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            entry = {
                "frame_age": int(frame_count - counted_at),
                "distance_px": round(float(d), 2),
                "threshold_px": round(float(dist_threshold), 2),
            }
            if best is None or d < best["distance_px"]:
                best = entry
        return best

    # ---------------- D3 ----------------
    def emit_count_decision(self, frame_count, object_id, lane_id, class_name, final_class_name,
                            bbox, centroid, wheels, crossing_point, history_len,
                            already_counted, crossed, cross_reason, dedup_dominated,
                            dedup_layer, accepted_count, lane_count_after, total_count_after,
                            disappeared_count, nearest_recent_bbox, nearest_recent_position):
        if not self.enabled:
            return
        # Maintain D4 counters.
        if accepted_count:
            self.accepted_counts += 1
            self.accepted_by_lane[lane_id] += 1
            self.accepted_by_class[final_class_name or class_name or "unknown"] += 1
            if cross_reason:
                self.cross_reason_counts[cross_reason] += 1
            if disappeared_count and disappeared_count > 0:
                self.counts_from_disappeared_tracks += 1
            if bbox is None:
                self.counts_with_bbox_none += 1
            self.object_ids_counted.append({
                "object_id": int(object_id),
                "frame": int(frame_count),
                "timestamp_seconds": self._ts(frame_count),
                "lane_id": lane_id,
                "class_name": final_class_name or class_name,
                "bbox": [float(b) for b in bbox] if bbox is not None else None,
                "disappeared_count": int(disappeared_count or 0),
            })
        elif dedup_dominated and dedup_layer:
            self.suppressed_by_layer[dedup_layer] += 1

        self._emit({
            "event": "count_decision",
            "video_uuid": self.video_uuid,
            "frame": int(frame_count),
            "timestamp_seconds": self._ts(frame_count),
            "object_id": int(object_id),
            "lane_id": lane_id,
            "class_name": class_name,
            "final_class_name": final_class_name,
            "bbox": [float(b) for b in bbox] if bbox is not None else None,
            "centroid": [int(centroid[0]), int(centroid[1])] if centroid is not None else None,
            "wheels": (
                [int(wheels[0]), int(wheels[1])]
                if wheels is not None and wheels[0] is not None else None
            ),
            "crossing_point": (
                [int(crossing_point[0]), int(crossing_point[1])]
                if crossing_point is not None and crossing_point[0] is not None else None
            ),
            "history_len": int(history_len),
            "already_counted": bool(already_counted),
            "crossed": bool(crossed),
            "cross_reason": cross_reason,
            "dedup_dominated": bool(dedup_dominated),
            "dedup_layer": dedup_layer,
            "accepted_count": bool(accepted_count),
            "lane_count_after": lane_count_after,
            "total_count_after": total_count_after,
            "disappeared_count": int(disappeared_count or 0),
            "nearest_recent_bbox": nearest_recent_bbox,
            "nearest_recent_position_same_lane": nearest_recent_position,
        })

    def emit_period_reset(self, frame_count, period_idx, next_object_id):
        if not self.enabled:
            return
        self._emit({
            "event": "period_reset",
            "video_uuid": self.video_uuid,
            "frame": int(frame_count),
            "timestamp_seconds": self._ts(frame_count),
            "period_index": int(period_idx),
            "next_object_id": int(next_object_id),
        })

    # ---------------- D4 ----------------
    def emit_summary(self, total_frames_processed):
        if not self.enabled:
            return
        self._emit({
            "event": "debug_summary",
            "video_uuid": self.video_uuid,
            "total_frames_processed": int(total_frames_processed),
            "accepted_counts": int(self.accepted_counts),
            "accepted_by_lane": dict(self.accepted_by_lane),
            "accepted_by_class": dict(self.accepted_by_class),
            "suppressed_by_layer": dict(self.suppressed_by_layer),
            "count_decisions_by_cross_reason": dict(self.cross_reason_counts),
            "counts_from_disappeared_tracks": int(self.counts_from_disappeared_tracks),
            "counts_with_bbox_none": int(self.counts_with_bbox_none),
            "max_simultaneous_tracks_near_finish_line": int(self.max_simultaneous_tracks_near_finish_line),
            "frames_with_multiple_finish_line_detections": int(self.frames_with_multiple_finish_line_detections),
            "object_ids_counted": self.object_ids_counted,
        })


def process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH="best.pt", progress_callback=None, generate_video_output=False, output_video_path=None, video_uuid=None, minute_batch_callback=None, trim_periods=None, truck_classifier_model_path=None, rear_model_path=None, axle_detector_model_path=None):
    """
    Process video for ATR (Automatic Traffic Recording) analysis with optional trimming.

    Args:
        VIDEO_PATH: Path to video file
        LINES_DATA: Lane configuration data with lanes and finish_line
        MODEL_PATH: Path to YOLO model
        progress_callback: Optional callback for progress updates
        generate_video_output: Whether to generate annotated output video
        output_video_path: Path for output video (if generate_video_output=True)
        video_uuid: UUID of the video being processed (optional, for minute tracking)
        minute_batch_callback: Optional callback for minute-by-minute batch data
        trim_periods: Optional list of trim periods in seconds [{"start": 3600, "end": 10800}, ...]
        axle_detector_model_path: Optional path to wheel/axle detector model for FHWA classification

    Returns:
        Dictionary with lane counts and total count
    """

    # Validate trim periods if provided
    if trim_periods:
        is_valid, error_msg = validate_trim_periods(trim_periods)
        if not is_valid:
            print(f"⚠️ Invalid trim_periods: {error_msg}")
            print("⚠️ Falling back to processing entire video")
            trim_periods = None

    # Constants
    CONF_THRESHOLD = 0.1

    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

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

    # Process lanes configuration
    lanes = LINES_DATA["lanes"]
    for lane in lanes:
        lane["points"] = dict_points_to_tuples(lane["points"])
    lane_polygons = [(lane["id"], Polygon(lane["points"])) for lane in lanes]
    lane_counts = {lane_id: 0 for lane_id, _ in lane_polygons}
    # Cache buffered polygons for performance optimization
    lane_polygons_buffered = [(lane_id, polygon.buffer(8)) for lane_id, polygon in lane_polygons]
    
    # Process finish line
    finish_line = LINES_DATA.get("finish_line")
    if finish_line and isinstance(finish_line[0], dict):
        finish_line = dict_points_to_tuples(finish_line)
    finish_linestring = LineString(finish_line) if finish_line and len(finish_line) == 2 else None

    # Initialize tracking variables
    counted_ids = set()
    previous_positions = {}
    last_known_lane = {}
    recently_counted_bboxes = []  # [(bbox_tuple, frame_count)] for overlap dedup
    recently_counted_positions = []  # [(lane_id, cx, cy, frame_count)] for position dedup
    tracker = CentroidTracker(max_disappeared=15)

    # Overcount-diagnosis emitter — gated on ATR_COUNT_DEBUG=1.
    debug_emitter = _ATRDebugEmitter(video_uuid, fps, finish_linestring)
    debug_emitter.emit_run_config(
        fps=float(fps) if fps else None,
        total_frames=int(total_frames),
        trim_periods=trim_periods,
        model_path=MODEL_PATH,
        rear_model_path=rear_model_path,
        truck_classifier_model_path=truck_classifier_model_path,
        axle_detector_model_path=axle_detector_model_path,
        conf_threshold=CONF_THRESHOLD,
        agnostic_nms=True,
        iou_threshold_default_ultralytics=0.7,
        tracker={"max_disappeared": 15, "max_distance": 150},
        dedup={
            "bbox_window_seconds": 1.0,
            "bbox_overlap_min_area_ratio": 0.5,
            "position_window_seconds": 0.5,
            "position_distance_rule": "max(30, bbox_height*0.25)",
        },
        finish_line=[list(map(int, p)) for p in finish_line] if finish_line else None,
        lane_ids=[lane_id for lane_id, _ in lane_polygons],
        debug_band_px=debug_emitter.band_px,
    )

    # Debug counters
    debug_total_detections = 0
    debug_tracked_objects = set()
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

    # Track raw detection labels by lane (use defaultdict pattern)
    from collections import defaultdict
    vehicle_counts_by_lane = defaultdict(lambda: defaultdict(int))

    # Video capture already initialized above for model manager
    frame_count = 0
    last_progress_sent = -1
    start_time = time.time()

    # Progress tracking: frames actually processed (not just video position)
    frames_processed_total = 0

    # Calculate frame ranges from trim periods
    frame_ranges = []
    if trim_periods:
        frame_ranges = calculate_frame_ranges_from_seconds(trim_periods, fps, total_frames)
        if frame_ranges:
            print(f"🎬 ATR Trimming enabled: processing {len(frame_ranges)} period(s)")
            total_processing_frames = sum(r['end_frame'] - r['start_frame'] for r in frame_ranges)
            print(f"   Total frames to process: {total_processing_frames} / {total_frames} ({total_processing_frames/total_frames*100:.1f}%)")
        else:
            print("⚠️ No valid frame ranges, processing entire video")
    else:
        print("📊 ATR: No trimming specified, processing entire video")

    # Calculate total_processing_frames for normal mode too (for unified progress calculation)
    if not frame_ranges:
        total_processing_frames = total_frames

    # ATR uses the rear-view detection model as the primary; if it isn't
    # available (e.g. S3 download failed or the key is missing) fall back to
    # the general-purpose vehicle model so processing still works.
    orientation_result = None
    if rear_model_path:
        model = YOLO(rear_model_path)
        print(f"✅ Rear-view YOLO model loaded (primary): {rear_model_path}")
    else:
        model = YOLO(MODEL_PATH)
        print(f"⚠️ Rear-view model not found, falling back to general model: {MODEL_PATH}")

    # Initialize minute tracker if callback provided
    minute_tracker = None
    if video_uuid and minute_batch_callback:
        # Set verbose=False in production for better performance
        minute_tracker = ATRMinuteTracker(fps, video_uuid, minute_batch_callback, verbose=False)
        print(f"🔄 ATR MinuteTracker enabled for video {video_uuid}")
    
    # Initialize video writer if output video is requested
    video_writer = None
    if generate_video_output and output_video_path:
        orig_fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Aggressive compression settings for ATR processor
        # Use H.264 with high compression for minimal file size
        width, height = orig_width, orig_height
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            # Reduce output resolution if too large for better compression
            if width > 1920 or height > 1080:
                scale_factor = min(1920/width, 1080/height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
                print(f"📉 Scaling ATR output resolution to {width}x{height} for compression")
            
            # Use lower FPS for additional compression if original is high
            output_fps = min(orig_fps, 15)  # Cap at 15 FPS for traffic analysis
            if output_fps != orig_fps:
                print(f"📉 Reducing ATR output FPS from {orig_fps} to {output_fps} for compression")
            
            video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))
            
            if video_writer.isOpened():
                print(f"✅ ATR video writer initialized: H264 codec, {width}x{height}@{output_fps}fps")
            else:
                video_writer.release()
                video_writer = None
                
        except Exception as e:
            print(f"⚠️ H264 codec failed: {e}")
            video_writer = None
        
        # Fallback to other codecs if H264 fails
        if not video_writer:
            codecs_to_try = ['X264', 'XVID', 'mp4v']
            
            for codec in codecs_to_try:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*codec)
                    temp_writer = cv2.VideoWriter(output_video_path, fourcc, orig_fps, (width, height))
                    if temp_writer.isOpened():
                        video_writer = temp_writer
                        print(f"✅ Fallback to ATR video codec: {codec}")
                        break
                    else:
                        temp_writer.release()
                except Exception as e:
                    print(f"⚠️ Codec {codec} failed: {e}")
                    continue
        
        if not video_writer:
            print("❌ Could not initialize ATR video writer with any codec")
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
                seek_progress = int((frame_count / total_frames) * 100)

            progress_callback({
                "progress": seek_progress,
                "estimatedTimeRemaining": 0,
                "status": "seeking"
            })
            # Only log at major seeking milestones to reduce log noise
            if frame_ranges:
                # For trimming: calculate seeking progress for logging
                start_frame = frame_ranges[0]["start_frame"]
                seek_pct = int((frame_count / start_frame) * 100) if start_frame > 0 else 0
                if seek_pct in [25, 50, 75] or frame_count >= start_frame - 1000:
                    print(f"⏩ SEEKING: Frame {frame_count}/{total_frames} ({seek_pct}% of seeking phase)")
            else:
                # For normal mode: log every 25%
                if seek_progress % 25 == 0 and seek_progress > 0:
                    print(f"⏩ SEEKING: Frame {frame_count}/{total_frames} (showing {seek_progress}% to user)")

    # Helper function to reset tracker state
    def reset_centroid_tracker():
        """Reset CentroidTracker to start fresh tracking for new period"""
        nonlocal tracker
        next_id = tracker.nextObjectID
        tracker = CentroidTracker(max_disappeared=15)
        tracker.nextObjectID = next_id
        print(f"🔄 ATR CentroidTracker reset (nextObjectID={next_id}) - previous tracking state cleared")

    # Helper function for progress calculation
    def calculate_and_send_progress():
        """
        Calculate progress based on actual frames processed (trimming-aware).

        For trimmed videos:
            progress = frames_processed_total / total_processing_frames
        For normal videos:
            progress = frame_count / total_frames (backward compatible)

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
            progress = int((frame_count / total_frames) * 100)

        # Ensure progress never exceeds 100% or decreases
        progress = min(100, max(0, progress))

        # CRITICAL: Prevent progress from going backwards
        if progress < last_progress_sent:
            print(f"⚠️  PROGRESS BACKWARDS PREVENTED: {progress}% < {last_progress_sent}% (frames_processed={frames_processed_total}, frame_count={frame_count})")
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
                print(f"   └─ Frames: {frames_processed_total}/{total_processing_frames} processed | Current position: {frame_count}/{total_frames}")
            else:
                print(f"   └─ Frames: {frame_count}/{total_frames}")

            progress_callback({
                "progress": progress,
                "estimatedTimeRemaining": max(0, estimated_remaining_time)
            })
            last_progress_sent = progress

    # Main processing logic with frame-skipping support
    if frame_ranges:
        # TRIMMING MODE: Process only specified periods with frame-skipping
        print("🎬 Starting trimmed ATR video processing")
        print(f"📊 PROGRESS TRACKING CONFIG:")
        print(f"   └─ Total video frames: {total_frames}")
        print(f"   └─ Frames to process: {total_processing_frames}")
        print(f"   └─ Trim coverage: {total_processing_frames/total_frames*100:.1f}%")
        print(f"   └─ Progress calculation: frames_processed_total / {total_processing_frames}")

        for period_idx, period in enumerate(frame_ranges):
            start_frame = period["start_frame"]
            end_frame = period["end_frame"]
            period_duration = (period["end_seconds"] - period["start_seconds"]) / 60  # minutes

            print(f"\n📍 ATR Period {period_idx + 1}/{len(frame_ranges)}")
            print(f"   Frames: {start_frame} - {end_frame} ({end_frame - start_frame} frames)")
            print(f"   Time: {period['start_seconds']:.1f}s - {period['end_seconds']:.1f}s ({period_duration:.1f} min)")

            # CRITICAL: Reset tracker at start of each period
            reset_centroid_tracker()

            # Clear per-track state to prevent cross-period tracking
            previous_positions.clear()
            last_known_lane.clear()
            print("🧹 Per-track state cleared for new period")
            debug_emitter.emit_period_reset(frame_count, period_idx, tracker.nextObjectID)

            # Skip frames until we reach the start of this period (frame-skipping)
            ret = True  # Initialize to True - if no seeking needed, we're already at the right position
            while frame_count < start_frame:
                ret, _ = cap.read()  # Read but don't process
                if not ret:
                    print(f"⚠️ Video ended at frame {frame_count} while seeking to {start_frame}")
                    break

                frame_count += 1

                # Progress update every 1000 frames during seeking
                if frame_count % 1000 == 0:
                    send_seeking_progress()
                    print(f"⏩ Seeking: {frame_count}/{start_frame} frames ({frame_count/start_frame*100:.1f}%)")

            if not ret:
                print(f"⚠️ Could not reach period {period_idx + 1}, skipping")
                continue

            print(f"✅ Reached start of period {period_idx + 1} at frame {frame_count}")

            # Process frames in this period
            while frame_count < end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ Video ended at frame {frame_count} during period {period_idx + 1}")
                    break

                # YOLO detection
                results = model.predict(frame, conf=CONF_THRESHOLD, agnostic_nms=True, verbose=False)
                boxes = results[0].boxes

                input_centroids = []
                detections_map = {}

                # Only process if there are detections
                if boxes is not None and len(boxes) > 0:
                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = model.names[class_id]
                        cx, cy = get_centroid((x1, y1, x2, y2))
                        wx, wy = get_wheels_position((x1, y1, x2, y2))
                        input_centroids.append(np.array([cx, cy]))
                        detections_map[(cx, cy)] = (x1, y1, x2, y2, class_name, wx, wy)
                        debug_total_detections += 1

                # Update tracker
                objects = tracker.update(np.array(input_centroids))

                # Diagnostic-frame trigger for D1/D2 emission
                is_diag_frame = debug_emitter.is_diagnostic_frame(detections_map, objects)
                if is_diag_frame:
                    debug_emitter.emit_frame_detections(
                        frame_count, detections_map, lane_polygons_buffered, tracker, objects,
                    )

                # Process tracked objects
                for objectID, centroid in objects.items():
                    # F7: skip tracks that didn't match this frame. Their stale centroid
                    # would otherwise hijack a foreign detection in the matching loop
                    # (overcount.md H9). The track is preserved in tracker.objects until
                    # it either re-acquires a detection or hits max_disappeared.
                    if tracker.disappeared.get(objectID, 0) > 0:
                        if is_diag_frame:
                            debug_emitter.emit_track_match(
                                frame_count=frame_count,
                                object_id=objectID,
                                tracker_centroid=centroid,
                                disappeared_count=tracker.disappeared.get(objectID, 0),
                                matched_detection_index=None,
                                match_distance_px=None,
                                matched_bbox=None,
                                matched_class=None,
                                matched_wheels=None,
                                cached_class=class_counts_by_id.get(objectID),
                                lane_id=last_known_lane.get(objectID),
                                lane_source=None,
                                previous_positions_tail=previous_positions.get(objectID),
                                was_skipped_for_disappeared=True,
                            )
                        continue

                    debug_tracked_objects.add(objectID)
                    cx, cy = centroid

                    # Match track to nearest detection within ±20 px (F8 — order-independent).
                    matched_detection, matched_detection_index, match_distance_px = (
                        _match_track_to_detection(centroid, detections_map)
                    )
                    class_name = (
                        matched_detection[4]
                        if matched_detection and len(matched_detection) > 4
                        else "unknown"
                    )

                    # Refine articulated_truck class on first detection of this track
                    if class_name == "articulated_truck" and truck_classifier and objectID not in class_counts_by_id and matched_detection:
                        class_name = truck_classifier.classify(frame, matched_detection[:4])

                    # Use persistent class: first detection wins (prevents class flip-flopping)
                    cached_class = class_counts_by_id.get(objectID)
                    class_name = class_counts_by_id.get(objectID, class_name)
                    if objectID not in class_counts_by_id:
                        class_counts_by_id[objectID] = class_name

                    # Extract wheels and bbox from matched detection
                    wheels_x, wheels_y = None, None
                    bbox = None
                    if matched_detection:
                        bbox = matched_detection[:4]
                        if len(matched_detection) > 6:
                            wheels_x, wheels_y = matched_detection[5], matched_detection[6]

                    # Detect axles for trucks (accumulate max across frames)
                    # Sample every 5 frames to reduce computation while maintaining accuracy
                    # Note: We do NOT store FHWA suffix in class_counts_by_id here to avoid breaking
                    # subsequent axle detection checks. FHWA is computed on-demand for visualization/output.
                    if axle_classifier and bbox is not None and frame_count % 5 == 0 and class_name in ("single_unit_truck", "articulated_truck", "multi_articulated_truck"):
                        axle_count = axle_classifier.detect_axles(frame, bbox)
                        if axle_count is not None:
                            current_max = max_axle_count_by_id.get(objectID, 0)
                            max_axle_count_by_id[objectID] = max(current_max, axle_count)

                    # Find lane (wheels → centroid → bottom-band → full-bbox fallback)
                    lane_id, lane_source = find_vehicle_lane_with_source(
                        cx, cy, wheels_x, wheels_y, lane_polygons_buffered, bbox=bbox,
                    )

                    # Sticky lane: remember last known lane, use as fallback
                    if lane_id is not None:
                        last_known_lane[objectID] = lane_id
                    elif objectID in last_known_lane:
                        lane_id = last_known_lane[objectID]
                        lane_source = 'sticky'

                    # Initialize position history with bounded size to prevent memory accumulation
                    # CRITICAL: Limit to last 10 positions - we only need recent history for line crossing
                    if objectID not in previous_positions:
                        previous_positions[objectID] = []

                    # Use wheels (bottom-center) for finish line crossing, consistent with lane detection
                    crossing_point = (wheels_x, wheels_y) if wheels_x is not None else (cx, cy)
                    previous_positions[objectID].append(crossing_point)
                    # Keep only last 10 positions to prevent unbounded memory growth
                    if len(previous_positions[objectID]) > 10:
                        previous_positions[objectID] = previous_positions[objectID][-10:]

                    disappeared_count = tracker.disappeared.get(objectID, 0)

                    if is_diag_frame:
                        debug_emitter.emit_track_match(
                            frame_count=frame_count,
                            object_id=objectID,
                            tracker_centroid=(cx, cy),
                            disappeared_count=disappeared_count,
                            matched_detection_index=matched_detection_index,
                            match_distance_px=match_distance_px,
                            matched_bbox=bbox,
                            matched_class=(matched_detection[4] if matched_detection and len(matched_detection) > 4 else None),
                            matched_wheels=((wheels_x, wheels_y) if wheels_x is not None else None),
                            cached_class=cached_class,
                            lane_id=lane_id,
                            lane_source=lane_source,
                            previous_positions_tail=previous_positions.get(objectID),
                            was_skipped_for_disappeared=False,
                        )

                    # Check finish line crossing
                    if objectID not in counted_ids and lane_id is not None and finish_line is not None:
                        crossed = False
                        cross_reason = None

                        # Primary: bbox intersects finish line (require min 2 frames tracked)
                        if bbox is not None and finish_linestring is not None and len(previous_positions[objectID]) >= 2:
                            bbox_poly = shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])
                            if bbox_poly.intersects(finish_linestring):
                                crossed = True
                                cross_reason = "bbox_intersects_finish_line"

                        # Fallback: point sign-change (when no bbox available)
                        if not crossed and len(previous_positions[objectID]) >= 2:
                            a, b = finish_line
                            prev = previous_positions[objectID][-2]
                            curr = previous_positions[objectID][-1]
                            side_prev = point_side_of_line(prev, a, b)
                            side_curr = point_side_of_line(curr, a, b)
                            if side_prev * side_curr < 0:
                                crossed = True
                                cross_reason = "point_sign_change"

                        # Pre-compute nearest dedup state for D3 (independent of dominance).
                        nearest_bbox_diag = debug_emitter.compute_nearest_recent_bbox(
                            bbox, recently_counted_bboxes, frame_count, fps,
                        )
                        if bbox is not None:
                            _bbox_h_for_diag = bbox[3] - bbox[1]
                            _dist_threshold_for_diag = max(30, _bbox_h_for_diag * 0.25)
                        else:
                            _dist_threshold_for_diag = 50
                        nearest_pos_diag = debug_emitter.compute_nearest_recent_position(
                            lane_id, cx, cy, recently_counted_positions, frame_count, _dist_threshold_for_diag,
                        )

                        accepted_count = False
                        dedup_dominated = False
                        dedup_layer = None
                        final_class_name = class_name

                        if crossed:
                            # Dedup layer 1: bbox overlap (when bbox available)
                            if bbox is not None:
                                current_poly = shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])
                                current_area = current_poly.area
                                for counted_bbox, counted_at in recently_counted_bboxes:
                                    if frame_count - counted_at > fps:
                                        continue
                                    counted_poly = shapely_box(*counted_bbox)
                                    if current_poly.intersects(counted_poly):
                                        overlap = current_poly.intersection(counted_poly).area
                                        min_area = min(current_area, counted_poly.area)
                                        if min_area > 0 and overlap / min_area > 0.5:
                                            dedup_dominated = True
                                            dedup_layer = "bbox_overlap"
                                            break

                            # Dedup layer 2: position-based (same lane only, short window).
                            # Compares the wheels-based crossing_point — same reference
                            # point used by the cross test — instead of the mid-bbox
                            # tracker centroid (overcount.md H4(c)).
                            if not dedup_dominated:
                                time_window = int(fps * 0.5)  # 0.5 second window
                                if bbox is not None:
                                    bbox_height = bbox[3] - bbox[1]
                                    dist_threshold = max(30, bbox_height * 0.25)
                                else:
                                    dist_threshold = 50
                                ref_x, ref_y = crossing_point
                                for prev_lane, prev_cx, prev_cy, counted_at in recently_counted_positions:
                                    if frame_count - counted_at > time_window:
                                        continue
                                    if prev_lane != lane_id:
                                        continue
                                    dist = ((ref_x - prev_cx)**2 + (ref_y - prev_cy)**2) ** 0.5
                                    if dist < dist_threshold:
                                        dedup_dominated = True
                                        dedup_layer = "position"
                                        break

                            if not dedup_dominated:
                                accepted_count = True
                                counted_ids.add(objectID)
                                lane_counts[lane_id] += 1
                                if bbox is not None:
                                    recently_counted_bboxes.append((tuple(bbox), frame_count))
                                recently_counted_positions.append(
                                    (lane_id, int(crossing_point[0]), int(crossing_point[1]), frame_count)
                                )

                                # Determine final class name with FHWA-specific label for trucks
                                # Track axle detection stats for trucks
                                if class_name in ("single_unit_truck", "articulated_truck", "multi_articulated_truck"):
                                    axle_detection_stats["trucks_detected"] += 1
                                    axle_detection_stats["detection_by_truck_type"][class_name]["attempted"] += 1
                                    axle_detection_stats["axle_detection_attempted"] += 1

                                    if axle_classifier and objectID in max_axle_count_by_id:
                                        axle_count = max_axle_count_by_id[objectID]
                                        axle_detection_stats["axle_detection_successful"] += 1
                                        axle_detection_stats["detection_by_truck_type"][class_name]["successful"] += 1
                                        axle_detection_stats["axle_counts_distribution"][axle_count] = \
                                            axle_detection_stats["axle_counts_distribution"].get(axle_count, 0) + 1

                                        fhwa_class = axle_classifier.get_fhwa_class(class_name, axle_count)
                                        if fhwa_class is not None:
                                            final_class_name = f"{class_name}_fhwa{fhwa_class}"
                                            axle_detection_stats["fhwa_class_distribution"][fhwa_class] = \
                                                axle_detection_stats["fhwa_class_distribution"].get(fhwa_class, 0) + 1

                                # Count detected class only ONCE per unique object ID
                                if objectID not in detected_classes:
                                    detected_classes[objectID] = final_class_name

                                # Use detection label (with FHWA suffix when available)
                                vehicle_counts_by_lane[final_class_name][lane_id] += 1

                                # Track in minute tracker if enabled
                                if minute_tracker:
                                    minute_tracker.process_vehicle_detection(frame_count, objectID, final_class_name, lane_id)

                                # Clean up per-track state for counted vehicle to free memory
                                del previous_positions[objectID]
                                last_known_lane.pop(objectID, None)
                                max_axle_count_by_id.pop(objectID, None)  # Clean up axle data too

                        debug_emitter.emit_count_decision(
                            frame_count=frame_count,
                            object_id=objectID,
                            lane_id=lane_id,
                            class_name=class_name,
                            final_class_name=final_class_name if accepted_count else class_name,
                            bbox=bbox,
                            centroid=(cx, cy),
                            wheels=((wheels_x, wheels_y) if wheels_x is not None else None),
                            crossing_point=crossing_point,
                            history_len=len(previous_positions.get(objectID, [])),
                            already_counted=False,
                            crossed=crossed,
                            cross_reason=cross_reason,
                            dedup_dominated=dedup_dominated,
                            dedup_layer=dedup_layer,
                            accepted_count=accepted_count,
                            lane_count_after=lane_counts.get(lane_id),
                            total_count_after=sum(lane_counts.values()),
                            disappeared_count=disappeared_count,
                            nearest_recent_bbox=nearest_bbox_diag,
                            nearest_recent_position=nearest_pos_diag,
                        )

                # Prune dedup lists
                recently_counted_bboxes = [(b, f) for b, f in recently_counted_bboxes if frame_count - f <= fps]
                recently_counted_positions = [(l, x, y, f) for l, x, y, f in recently_counted_positions if frame_count - f <= int(fps * 0.5)]

                # Add visualizations if generating output video
                if generate_video_output and video_writer:
                    # Draw detections and tracking
                    for objectID, centroid in objects.items():
                        cx, cy = int(centroid[0]), int(centroid[1])

                        # Find lane for visualization (nearest detection within ±20 px).
                        matched_detection_viz, _, _ = _match_track_to_detection(
                            (cx, cy), detections_map,
                        )
                        bbox = matched_detection_viz[:4] if matched_detection_viz else None
                        wheels_x = wheels_y = None
                        if matched_detection_viz and len(matched_detection_viz) > 6:
                            wheels_x, wheels_y = matched_detection_viz[5], matched_detection_viz[6]
                        lane_id = find_vehicle_lane(cx, cy, wheels_x, wheels_y, lane_polygons_buffered, bbox=bbox)

                        # Draw bounding box and label only when detection is available
                        if (cx, cy) in detections_map:
                            detection_data = detections_map[(cx, cy)]
                            x1, y1, x2, y2 = detection_data[:4]
                            class_name_viz = class_counts_by_id.get(objectID, detection_data[4] if len(detection_data) > 4 else None)
                            # Compute FHWA suffix on-demand for visualization
                            if axle_classifier and class_name_viz and objectID in max_axle_count_by_id:
                                fhwa_viz = axle_classifier.get_fhwa_class(class_name_viz, max_axle_count_by_id[objectID])
                                if fhwa_viz is not None:
                                    class_name_viz = f"{class_name_viz}_fhwa{fhwa_viz}"
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                            # Draw wheels position if available (for debugging)
                            if len(detection_data) > 6:
                                wx, wy = detection_data[5], detection_data[6]
                                cv2.circle(frame, (int(wx), int(wy)), 3, (255, 0, 0), -1)  # Blue for wheels

                            label = f'ID {objectID} | L{lane_id}'
                            if class_name_viz:
                                label = f'{class_name_viz} {label}'
                            cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Draw finish line
                    if finish_line is not None and len(finish_line) == 2:
                        pt1 = tuple(map(int, finish_line[0]))
                        pt2 = tuple(map(int, finish_line[1]))
                        cv2.line(frame, pt1, pt2, (255, 0, 255), 3)

                    # Draw lanes and counts
                    for lane in lanes:
                        pts = np.array(lane["points"], np.int32).reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                        cv2.putText(frame, f'L{lane["id"]}: {lane_counts[lane["id"]]}',
                                   (pts[0][0][0], pts[0][0][1] - 8),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Draw total count
                    total_count_current = sum(lane_counts.values())
                    cv2.putText(frame, f'Total: {total_count_current}', (20, 40),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

                    # Resize frame if needed for compression
                    if width != orig_width or height != orig_height:
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                    # Write frame to output video
                    video_writer.write(frame)

                # Progress tracking (after successful processing)
                frame_count += 1
                frames_processed_total += 1  # Track actual frames processed
                calculate_and_send_progress()

                # Small delay to stabilize tracking (similar to cv2.waitKey in example.py)
                time.sleep(0.001)  # 1ms delay to prevent too rapid processing

            print(f"✅ Completed period {period_idx + 1}/{len(frame_ranges)}")

        print("\n✅ All ATR trim periods processed")

    else:
        # NORMAL MODE: Process entire video (existing logic)
        print("📊 Processing entire ATR video (no trimming)")
        print(f"📊 PROGRESS TRACKING CONFIG:")
        print(f"   └─ Total frames: {total_frames}")
        print(f"   └─ Progress calculation: frame_count / {total_frames}")

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO detection
            results = model.predict(frame, conf=CONF_THRESHOLD, agnostic_nms=True, verbose=False)
            boxes = results[0].boxes

            input_centroids = []
            detections_map = {}

            # Only process if there are detections
            if boxes is not None and len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = model.names[class_id]
                    cx, cy = get_centroid((x1, y1, x2, y2))
                    wx, wy = get_wheels_position((x1, y1, x2, y2))
                    input_centroids.append(np.array([cx, cy]))
                    detections_map[(cx, cy)] = (x1, y1, x2, y2, class_name, wx, wy)
                    debug_total_detections += 1

            # Update tracker
            objects = tracker.update(np.array(input_centroids))

            # Diagnostic-frame trigger for D1/D2 emission
            is_diag_frame = debug_emitter.is_diagnostic_frame(detections_map, objects)
            if is_diag_frame:
                debug_emitter.emit_frame_detections(
                    frame_count, detections_map, lane_polygons_buffered, tracker, objects,
                )

            # Process tracked objects
            for objectID, centroid in objects.items():
                # F7: skip tracks that didn't match this frame. See trim-mode for full rationale.
                if tracker.disappeared.get(objectID, 0) > 0:
                    if is_diag_frame:
                        debug_emitter.emit_track_match(
                            frame_count=frame_count,
                            object_id=objectID,
                            tracker_centroid=centroid,
                            disappeared_count=tracker.disappeared.get(objectID, 0),
                            matched_detection_index=None,
                            match_distance_px=None,
                            matched_bbox=None,
                            matched_class=None,
                            matched_wheels=None,
                            cached_class=class_counts_by_id.get(objectID),
                            lane_id=last_known_lane.get(objectID),
                            lane_source=None,
                            previous_positions_tail=previous_positions.get(objectID),
                            was_skipped_for_disappeared=True,
                        )
                    continue

                debug_tracked_objects.add(objectID)
                cx, cy = centroid

                # Match track to nearest detection within ±20 px (F8 — order-independent).
                matched_detection, matched_detection_index, match_distance_px = (
                    _match_track_to_detection(centroid, detections_map)
                )
                class_name = (
                    matched_detection[4]
                    if matched_detection and len(matched_detection) > 4
                    else "unknown"
                )

                # Refine articulated_truck class on first detection of this track
                if class_name == "articulated_truck" and truck_classifier and objectID not in class_counts_by_id and matched_detection:
                    class_name = truck_classifier.classify(frame, matched_detection[:4])

                # Use persistent class: first detection wins (prevents class flip-flopping)
                cached_class = class_counts_by_id.get(objectID)
                class_name = class_counts_by_id.get(objectID, class_name)
                if objectID not in class_counts_by_id:
                    class_counts_by_id[objectID] = class_name

                # Extract wheels and bbox from matched detection
                wheels_x, wheels_y = None, None
                bbox = None
                if matched_detection:
                    bbox = matched_detection[:4]
                    if len(matched_detection) > 6:
                        wheels_x, wheels_y = matched_detection[5], matched_detection[6]

                # Detect axles for trucks (accumulate max across frames) - normal mode
                # Sample every 5 frames to reduce computation while maintaining accuracy
                # Note: We do NOT store FHWA suffix in class_counts_by_id here to avoid breaking
                # subsequent axle detection checks. FHWA is computed on-demand for visualization/output.
                if axle_classifier and bbox is not None and frame_count % 5 == 0 and class_name in ("single_unit_truck", "articulated_truck", "multi_articulated_truck"):
                    axle_count = axle_classifier.detect_axles(frame, bbox)
                    if axle_count is not None:
                        current_max = max_axle_count_by_id.get(objectID, 0)
                        max_axle_count_by_id[objectID] = max(current_max, axle_count)

                # Find lane (wheels → centroid → bottom-band → full-bbox fallback)
                lane_id, lane_source = find_vehicle_lane_with_source(
                    cx, cy, wheels_x, wheels_y, lane_polygons_buffered, bbox=bbox,
                )

                # Sticky lane: remember last known lane, use as fallback
                if lane_id is not None:
                    last_known_lane[objectID] = lane_id
                elif objectID in last_known_lane:
                    lane_id = last_known_lane[objectID]
                    lane_source = 'sticky'

                # Initialize position history with bounded size to prevent memory accumulation
                # CRITICAL: Limit to last 10 positions - we only need recent history for line crossing
                if objectID not in previous_positions:
                    previous_positions[objectID] = []

                # Use wheels (bottom-center) for finish line crossing, consistent with lane detection
                crossing_point = (wheels_x, wheels_y) if wheels_x is not None else (cx, cy)
                previous_positions[objectID].append(crossing_point)
                # Keep only last 10 positions to prevent unbounded memory growth
                if len(previous_positions[objectID]) > 10:
                    previous_positions[objectID] = previous_positions[objectID][-10:]

                disappeared_count = tracker.disappeared.get(objectID, 0)

                if is_diag_frame:
                    debug_emitter.emit_track_match(
                        frame_count=frame_count,
                        object_id=objectID,
                        tracker_centroid=(cx, cy),
                        disappeared_count=disappeared_count,
                        matched_detection_index=matched_detection_index,
                        match_distance_px=match_distance_px,
                        matched_bbox=bbox,
                        matched_class=(matched_detection[4] if matched_detection and len(matched_detection) > 4 else None),
                        matched_wheels=((wheels_x, wheels_y) if wheels_x is not None else None),
                        cached_class=cached_class,
                        lane_id=lane_id,
                        lane_source=lane_source,
                        previous_positions_tail=previous_positions.get(objectID),
                        was_skipped_for_disappeared=False,
                    )

                # Check finish line crossing
                if objectID not in counted_ids and lane_id is not None and finish_line is not None:
                    crossed = False
                    cross_reason = None

                    # Primary: bbox intersects finish line (require min 2 frames tracked)
                    if bbox is not None and finish_linestring is not None and len(previous_positions[objectID]) >= 2:
                        bbox_poly = shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])
                        if bbox_poly.intersects(finish_linestring):
                            crossed = True
                            cross_reason = "bbox_intersects_finish_line"

                    # Fallback: point sign-change (when no bbox available)
                    if not crossed and len(previous_positions[objectID]) >= 2:
                        a, b = finish_line
                        prev = previous_positions[objectID][-2]
                        curr = previous_positions[objectID][-1]
                        side_prev = point_side_of_line(prev, a, b)
                        side_curr = point_side_of_line(curr, a, b)
                        if side_prev * side_curr < 0:
                            crossed = True
                            cross_reason = "point_sign_change"

                    # Pre-compute nearest dedup state for D3 (independent of dominance).
                    nearest_bbox_diag = debug_emitter.compute_nearest_recent_bbox(
                        bbox, recently_counted_bboxes, frame_count, fps,
                    )
                    if bbox is not None:
                        _bbox_h_for_diag = bbox[3] - bbox[1]
                        _dist_threshold_for_diag = max(30, _bbox_h_for_diag * 0.25)
                    else:
                        _dist_threshold_for_diag = 50
                    nearest_pos_diag = debug_emitter.compute_nearest_recent_position(
                        lane_id, cx, cy, recently_counted_positions, frame_count, _dist_threshold_for_diag,
                    )

                    accepted_count = False
                    dedup_dominated = False
                    dedup_layer = None
                    final_class_name = class_name

                    if crossed:
                        # Dedup layer 1: bbox overlap (when bbox available)
                        if bbox is not None:
                            current_poly = shapely_box(bbox[0], bbox[1], bbox[2], bbox[3])
                            current_area = current_poly.area
                            for counted_bbox, counted_at in recently_counted_bboxes:
                                if frame_count - counted_at > fps:
                                    continue
                                counted_poly = shapely_box(*counted_bbox)
                                if current_poly.intersects(counted_poly):
                                    overlap = current_poly.intersection(counted_poly).area
                                    min_area = min(current_area, counted_poly.area)
                                    if min_area > 0 and overlap / min_area > 0.5:
                                        dedup_dominated = True
                                        dedup_layer = "bbox_overlap"
                                        break

                        # Dedup layer 2: position-based (same lane only, short window).
                        # Compares the wheels-based crossing_point — same reference
                        # point used by the cross test — instead of the mid-bbox
                        # tracker centroid (overcount.md H4(c)).
                        if not dedup_dominated:
                            time_window = int(fps * 0.5)  # 0.5 second window
                            if bbox is not None:
                                bbox_height = bbox[3] - bbox[1]
                                dist_threshold = max(30, bbox_height * 0.25)
                            else:
                                dist_threshold = 50
                            ref_x, ref_y = crossing_point
                            for prev_lane, prev_cx, prev_cy, counted_at in recently_counted_positions:
                                if frame_count - counted_at > time_window:
                                    continue
                                if prev_lane != lane_id:
                                    continue
                                dist = ((ref_x - prev_cx)**2 + (ref_y - prev_cy)**2) ** 0.5
                                if dist < dist_threshold:
                                    dedup_dominated = True
                                    dedup_layer = "position"
                                    break

                        if not dedup_dominated:
                            accepted_count = True
                            counted_ids.add(objectID)
                            lane_counts[lane_id] += 1
                            if bbox is not None:
                                recently_counted_bboxes.append((tuple(bbox), frame_count))
                            recently_counted_positions.append(
                                (lane_id, int(crossing_point[0]), int(crossing_point[1]), frame_count)
                            )

                            # Determine final class name with FHWA-specific label for trucks - normal mode
                            # Track axle detection stats for trucks - normal mode
                            if class_name in ("single_unit_truck", "articulated_truck", "multi_articulated_truck"):
                                axle_detection_stats["trucks_detected"] += 1
                                axle_detection_stats["detection_by_truck_type"][class_name]["attempted"] += 1
                                axle_detection_stats["axle_detection_attempted"] += 1

                                if axle_classifier and objectID in max_axle_count_by_id:
                                    axle_count = max_axle_count_by_id[objectID]
                                    axle_detection_stats["axle_detection_successful"] += 1
                                    axle_detection_stats["detection_by_truck_type"][class_name]["successful"] += 1
                                    axle_detection_stats["axle_counts_distribution"][axle_count] = \
                                        axle_detection_stats["axle_counts_distribution"].get(axle_count, 0) + 1

                                    fhwa_class = axle_classifier.get_fhwa_class(class_name, axle_count)
                                    if fhwa_class is not None:
                                        final_class_name = f"{class_name}_fhwa{fhwa_class}"
                                        axle_detection_stats["fhwa_class_distribution"][fhwa_class] = \
                                            axle_detection_stats["fhwa_class_distribution"].get(fhwa_class, 0) + 1

                            # Count detected class only ONCE per unique object ID
                            if objectID not in detected_classes:
                                detected_classes[objectID] = final_class_name

                            # Use detection label (with FHWA suffix when available)
                            vehicle_counts_by_lane[final_class_name][lane_id] += 1

                            # Track in minute tracker if enabled
                            if minute_tracker:
                                minute_tracker.process_vehicle_detection(frame_count, objectID, final_class_name, lane_id)

                            # Clean up per-track state for counted vehicle to free memory
                            del previous_positions[objectID]
                            last_known_lane.pop(objectID, None)
                            max_axle_count_by_id.pop(objectID, None)  # Clean up axle data too

                    debug_emitter.emit_count_decision(
                        frame_count=frame_count,
                        object_id=objectID,
                        lane_id=lane_id,
                        class_name=class_name,
                        final_class_name=final_class_name if accepted_count else class_name,
                        bbox=bbox,
                        centroid=(cx, cy),
                        wheels=((wheels_x, wheels_y) if wheels_x is not None else None),
                        crossing_point=crossing_point,
                        history_len=len(previous_positions.get(objectID, [])),
                        already_counted=False,
                        crossed=crossed,
                        cross_reason=cross_reason,
                        dedup_dominated=dedup_dominated,
                        dedup_layer=dedup_layer,
                        accepted_count=accepted_count,
                        lane_count_after=lane_counts.get(lane_id),
                        total_count_after=sum(lane_counts.values()),
                        disappeared_count=disappeared_count,
                        nearest_recent_bbox=nearest_bbox_diag,
                        nearest_recent_position=nearest_pos_diag,
                    )

            # Prune dedup lists
            recently_counted_bboxes = [(b, f) for b, f in recently_counted_bboxes if frame_count - f <= fps]
            recently_counted_positions = [(l, x, y, f) for l, x, y, f in recently_counted_positions if frame_count - f <= int(fps * 0.5)]

            # Add visualizations if generating output video
            if generate_video_output and video_writer:
                # Draw detections and tracking
                for objectID, centroid in objects.items():
                    cx, cy = int(centroid[0]), int(centroid[1])

                    # Find lane for visualization (nearest detection within ±20 px).
                    matched_detection_viz, _, _ = _match_track_to_detection(
                        (cx, cy), detections_map,
                    )
                    bbox = matched_detection_viz[:4] if matched_detection_viz else None
                    wheels_x = wheels_y = None
                    if matched_detection_viz and len(matched_detection_viz) > 6:
                        wheels_x, wheels_y = matched_detection_viz[5], matched_detection_viz[6]
                    lane_id = find_vehicle_lane(cx, cy, wheels_x, wheels_y, lane_polygons_buffered, bbox=bbox)

                    # Draw bounding box and label only when detection is available
                    if (cx, cy) in detections_map:
                        detection_data = detections_map[(cx, cy)]
                        x1, y1, x2, y2 = detection_data[:4]
                        class_name_viz = class_counts_by_id.get(objectID, detection_data[4] if len(detection_data) > 4 else None)
                        # Compute FHWA suffix on-demand for visualization
                        if axle_classifier and class_name_viz and objectID in max_axle_count_by_id:
                            fhwa_viz = axle_classifier.get_fhwa_class(class_name_viz, max_axle_count_by_id[objectID])
                            if fhwa_viz is not None:
                                class_name_viz = f"{class_name_viz}_fhwa{fhwa_viz}"
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Draw wheels position if available (for debugging)
                        if len(detection_data) > 6:
                            wx, wy = detection_data[5], detection_data[6]
                            cv2.circle(frame, (int(wx), int(wy)), 3, (255, 0, 0), -1)  # Blue for wheels

                        label = f'ID {objectID} | L{lane_id}'
                        if class_name_viz:
                            label = f'{class_name_viz} {label}'
                        cv2.putText(frame, label, (int(x1), int(y1) - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw finish line
                if finish_line is not None and len(finish_line) == 2:
                    pt1 = tuple(map(int, finish_line[0]))
                    pt2 = tuple(map(int, finish_line[1]))
                    cv2.line(frame, pt1, pt2, (255, 0, 255), 3)

                # Draw lanes and counts
                for lane in lanes:
                    pts = np.array(lane["points"], np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                    cv2.putText(frame, f'L{lane["id"]}: {lane_counts[lane["id"]]}',
                               (pts[0][0][0], pts[0][0][1] - 8),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw total count
                total_count_current = sum(lane_counts.values())
                cv2.putText(frame, f'Total: {total_count_current}', (20, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

                # Resize frame if needed for compression
                if width != orig_width or height != orig_height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                # Write frame to output video
                video_writer.write(frame)

            # Progress tracking (after successful processing)
            frame_count += 1
            frames_processed_total += 1  # Track actual frames processed (same as frame_count in normal mode)
            calculate_and_send_progress()

            # Small delay to stabilize tracking (similar to cv2.waitKey in example.py)
            time.sleep(0.001)  # 1ms delay to prevent too rapid processing

    # Send final 100% progress
    if progress_callback:
        progress_callback({
            "progress": 100,
            "estimatedTimeRemaining": 0
        })
        print(f"✅ ATR Processing complete: {frames_processed_total} frames processed")

    cap.release()
    if video_writer:
        video_writer.release()

    # CRITICAL: Release YOLO model and GPU memory to prevent accumulation
    # RunPod workers are reused, so memory accumulates if not released
    print("🧹 Releasing YOLO model and GPU memory...")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("✅ GPU memory cache cleared")

    # Finalize minute tracking if enabled
    total_duration = None
    if minute_tracker:
        total_duration = minute_tracker.finalize_processing()
    
    # Return results
    total_count = sum(lane_counts.values())
    
    # Debug output
    print(f"[ATR DEBUG] Total detections: {debug_total_detections}")
    print(f"[ATR DEBUG] Unique tracked objects: {len(debug_tracked_objects)}")
    print(f"[ATR DEBUG] Objects counted: {len(counted_ids)}")
    print(f"[ATR DEBUG] Final lane counts: {lane_counts}")
    print(f"[ATR DEBUG] Total count: {total_count}")

    # Overcount-diagnosis summary (gated on ATR_COUNT_DEBUG=1)
    debug_emitter.emit_summary(frames_processed_total)

    # Convert detected_classes from {obj_id: class_name} to {class_name: count}
    class_summary = Counter(detected_classes.values())

    # Convert vehicle_counts_by_lane from defaultdict to regular dict for JSON serialization
    # Structure: {detection_label: {lane_id: count}}
    vehicles_by_class = {class_name: dict(lane_data) for class_name, lane_data in vehicle_counts_by_lane.items()}

    # Create detected_classes summary from raw detection labels
    detected_classes_summary = {}
    for class_name, lane_data in vehicle_counts_by_lane.items():
        detected_classes_summary[class_name] = sum(lane_data.values())

    # Calculate axle detection success rate
    if axle_detection_stats["axle_detection_attempted"] > 0:
        axle_detection_stats["success_rate"] = round(
            axle_detection_stats["axle_detection_successful"] /
            axle_detection_stats["axle_detection_attempted"] * 100, 1
        )
    else:
        axle_detection_stats["success_rate"] = None

    result = {
        "lane_counts": lane_counts,
        "total_count": total_count,
        "study_type": "ATR",
        "detected_classes": detected_classes_summary,  # Raw detection labels with counts
        "vehicles": vehicles_by_class,  # Raw detection labels by lane: {class_name: {lane_id: count}}
        "axle_detection_stats": axle_detection_stats if axle_classifier else None
    }

    if orientation_result:
        result["orientation"] = orientation_result["orientation"]
        result["orientation_confidence"] = orientation_result["confidence"]
        result["orientation_model_used"] = "rear" if orientation_result["orientation"] == "rear" else "front"

    return result