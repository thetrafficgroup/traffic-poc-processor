"""
orientation_detector.py - Auto-detect vehicle orientation (approaching vs departing)
for ATR processing by analyzing bounding box area trends during a calibration phase.

Vehicles approaching the camera grow in apparent size (perspective effect).
Vehicles departing shrink. This signal is used to select the appropriate YOLO model.
"""

import cv2
import numpy as np
from collections import OrderedDict, defaultdict
from statistics import median
from ultralytics import YOLO


class _CalibrationTracker:
    """Lightweight centroid tracker for calibration phase only."""

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
            D = np.linalg.norm(
                np.array(objectCentroids)[:, np.newaxis] - input_centroids, axis=2
            )
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
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


CONF_THRESHOLD = 0.1
MIN_AREA_SAMPLES = 5
APPROACHING_RATIO = 1.3   # Vehicle grew by 30%+
DEPARTING_RATIO = 0.77     # Vehicle shrank by 23%+ (symmetric on log scale)


def detect_vehicle_orientation(video_path, model_path, fps,
                               calibration_seconds=30, min_vehicles=3,
                               start_frame=0):
    """
    Analyze the first N seconds of video to determine vehicle orientation.

    Uses bounding box area trends: approaching vehicles grow in apparent size
    due to perspective, departing vehicles shrink.

    Args:
        video_path: Path to video file
        model_path: Path to YOLO model (same model used for main processing)
        fps: Video frames per second
        calibration_seconds: How many seconds to analyze (default 30)
        min_vehicles: Minimum vehicles needed for a conclusive result (default 3)
        start_frame: Frame to start calibration from (for trim_periods support)

    Returns:
        dict with keys:
            - orientation: "front" | "rear" | "inconclusive"
            - confidence: float 0.0-1.0
            - vehicles_analyzed: int
            - approaching_count: int
            - departing_count: int
    """
    inconclusive = {
        "orientation": "inconclusive",
        "confidence": 0.0,
        "vehicles_analyzed": 0,
        "approaching_count": 0,
        "departing_count": 0,
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("⚠️ Orientation calibration: could not open video")
        return inconclusive

    # Seek to start frame if needed
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        print(f"🔍 Orientation calibration: seeking to frame {start_frame}")

    calibration_frames = int(calibration_seconds * fps)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Don't try to calibrate more frames than the video has
    available_frames = total_video_frames - start_frame
    if available_frames < fps * 2:  # Less than 2 seconds
        print("⚠️ Orientation calibration: video too short for calibration")
        cap.release()
        return inconclusive
    calibration_frames = min(calibration_frames, available_frames)

    # Load model for calibration
    model = YOLO(model_path)
    tracker = _CalibrationTracker(max_disappeared=15)
    area_history = defaultdict(list)  # objectID -> [(frame, area)]

    print(f"🔍 Orientation calibration: analyzing {calibration_frames} frames "
          f"({calibration_seconds}s) starting at frame {start_frame}")

    frames_read = 0
    for frame_idx in range(calibration_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1

        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
        boxes = results[0].boxes

        input_centroids = []
        detection_areas = {}  # (cx, cy) -> area

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                area = float((x2 - x1) * (y2 - y1))
                input_centroids.append(np.array([cx, cy]))
                detection_areas[(cx, cy)] = area

        objects = tracker.update(np.array(input_centroids) if input_centroids else np.array([]))

        for objectID, centroid in objects.items():
            cx, cy = int(centroid[0]), int(centroid[1])
            # Match to closest detection
            best_dist = float("inf")
            matched_area = None
            for (det_cx, det_cy), area in detection_areas.items():
                dist = abs(det_cx - cx) + abs(det_cy - cy)
                if dist < 40 and dist < best_dist:
                    best_dist = dist
                    matched_area = area
            if matched_area is not None:
                area_history[objectID].append((frame_idx, matched_area))

    cap.release()

    # Clean up calibration model
    del model
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass

    # Analyze area trends
    approaching = 0
    departing = 0

    for objectID, history in area_history.items():
        if len(history) < MIN_AREA_SAMPLES:
            continue

        areas = [a for _, a in history]
        first_median = median(areas[:3])
        last_median = median(areas[-3:])

        if first_median <= 0:
            continue

        ratio = last_median / first_median

        if ratio > APPROACHING_RATIO:
            approaching += 1
        elif ratio < DEPARTING_RATIO:
            departing += 1

    total = approaching + departing
    vehicles_analyzed = total

    print(f"🔍 Calibration results: {approaching} approaching, {departing} departing, "
          f"{len(area_history)} total tracked, {frames_read} frames analyzed")

    if total < min_vehicles:
        print(f"⚠️ Orientation calibration: only {total} vehicles with clear signal "
              f"(need {min_vehicles}), result inconclusive")
        return {
            "orientation": "inconclusive",
            "confidence": 0.0,
            "vehicles_analyzed": vehicles_analyzed,
            "approaching_count": approaching,
            "departing_count": departing,
        }

    confidence = abs(approaching - departing) / total

    if approaching > departing:
        orientation = "front"
    elif departing > approaching:
        orientation = "rear"
    else:
        orientation = "inconclusive"

    print(f"🔍 Orientation detected: {orientation} (confidence={confidence:.2f})")

    return {
        "orientation": orientation,
        "confidence": round(confidence, 3),
        "vehicles_analyzed": vehicles_analyzed,
        "approaching_count": approaching,
        "departing_count": departing,
    }
