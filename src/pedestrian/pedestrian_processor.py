"""
Pedestrian/bicycle detection processor with enhanced tracking.

Encapsulates all pedestrian model inference, tracking improvements,
crosswalk processing, and visualization that was previously inlined
in tmc_processor.py.

Key tracking improvements over the original inline code:
  1. Dedicated TrackInterpolator (max_missing_frames=15)
  2. Lower confidence threshold (0.12 vs 0.25)
  3. IOU=0.2 for tracker association
  4. Soft-NMS on detections when >=2 detections
  5. EMA position smoothing (alpha=0.3) before crosswalk feeding
  6. Crossing timeout raised to 60s (in crosswalk_processor.py)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.overlap_detection import TrackInterpolator, post_process_detections
from crosswalk.crosswalk_processor import CrosswalkProcessor
from crosswalk.crosswalk_minute_tracker import CrosswalkMinuteTracker


@dataclass
class BicycleDetection:
    """A bicycle detection to be fed into TMC turn logic by the caller."""
    namespaced_id: int
    class_name: str  # always "bicycle"
    cx: int
    cy: int


@dataclass
class PedestrianFrameResult:
    """Result of processing a single frame through the pedestrian model."""
    bicycle_detections: List[BicycleDetection] = field(default_factory=list)
    total_detections: int = 0
    pedestrian_count: int = 0
    bicycle_count: int = 0


class PedestrianProcessor:
    """
    Encapsulates pedestrian/bicycle YOLO model inference, tracking
    enhancements, crosswalk processing, and debug visualisation.
    """

    def __init__(
        self,
        model_path: str,
        crosswalk_proc: Optional[CrosswalkProcessor],
        crosswalk_minute_tracker: Optional[CrosswalkMinuteTracker],
        fps: float,
        conf_threshold: float = 0.12,
        img_size: int = 640,
        iou_threshold: float = 0.2,
        ped_track_id_offset: int = 1_000_000,
        ema_alpha: float = 0.3,
        max_missing_frames: int = 15,
        min_track_length: int = 3,
    ):
        self.model: YOLO = YOLO(model_path)
        print(f"✅ Pedestrian model loaded: {model_path}")

        self.crosswalk_proc = crosswalk_proc
        self.crosswalk_minute_tracker = crosswalk_minute_tracker
        self.fps = fps

        # Tracking parameters
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.iou_threshold = iou_threshold
        self.ped_track_id_offset = ped_track_id_offset
        self.ema_alpha = ema_alpha

        # Dedicated track interpolator for pedestrians
        self.track_interpolator = TrackInterpolator(
            max_missing_frames=max_missing_frames,
            min_track_length=min_track_length,
        )

        # EMA smoothed positions: namespaced_id -> (smoothed_cx, smoothed_cy)
        self._ema_positions: Dict[int, Tuple[float, float]] = {}

        # Cache latest results for visualisation
        self._last_ped_results = None
        self._last_ped_ids = None
        self._last_ped_boxes = None
        self._last_ped_classes = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_frame(self, frame, current_frame: int) -> PedestrianFrameResult:
        """Run pedestrian model on *frame* and return structured results.

        Bicycle detections are returned so the caller can feed them into
        TMC turn logic.  Pedestrian + bicycle detections are fed into the
        crosswalk processor internally.
        """
        result = PedestrianFrameResult()

        ped_results = self.model.track(
            frame,
            persist=True,
            conf=self.conf_threshold,
            imgsz=self.img_size,
            iou=self.iou_threshold,
            verbose=False,
        )

        self._last_ped_results = ped_results

        if (
            not ped_results
            or ped_results[0].boxes is None
            or ped_results[0].boxes.id is None
        ):
            self._last_ped_ids = None
            return result

        ped_ids = ped_results[0].boxes.id.cpu().numpy()
        ped_boxes = ped_results[0].boxes.xyxy.cpu().numpy()
        ped_classes = ped_results[0].boxes.cls.cpu().numpy()
        ped_scores = ped_results[0].boxes.conf.cpu().numpy()

        # -- Fix 4: Soft-NMS when >=2 detections --
        if len(ped_boxes) >= 2:
            ped_boxes, ped_scores, ped_classes, ped_ids = post_process_detections(
                ped_boxes, ped_scores, ped_classes, ped_ids
            )

        # Cache for visualisation
        self._last_ped_ids = ped_ids
        self._last_ped_boxes = ped_boxes
        self._last_ped_classes = ped_classes

        result.total_detections = len(ped_boxes)

        for i, box in enumerate(ped_boxes):
            raw_id = int(ped_ids[i])
            namespaced_id = self.ped_track_id_offset + raw_id
            class_name = self.model.names[int(ped_classes[i])]
            if class_name == "person":
                class_name = "pedestrian"

            cx = int((box[0] + box[2]) / 2)
            cy = int(box[3])  # Bottom = feet/wheels

            # -- Fix 1: Track interpolator for peds --
            self.track_interpolator.update_track(namespaced_id, (cx, cy), current_frame)

            # -- Fix 6: EMA position smoothing --
            cx_smooth, cy_smooth = self._smooth_position(namespaced_id, cx, cy)

            if class_name == "pedestrian":
                result.pedestrian_count += 1
            elif class_name == "bicycle":
                result.bicycle_count += 1
                result.bicycle_detections.append(
                    BicycleDetection(
                        namespaced_id=namespaced_id,
                        class_name=class_name,
                        cx=cx,
                        cy=cy,
                    )
                )

            # ALL ped/bike detections -> crosswalk processor (using smoothed positions)
            if self.crosswalk_proc is not None:
                crossing_result = self.crosswalk_proc.process_detection(
                    namespaced_id, class_name, cx_smooth, cy_smooth, current_frame
                )
                if crossing_result and self.crosswalk_minute_tracker is not None:
                    self.crosswalk_minute_tracker.record_crossing(
                        current_frame,
                        crossing_result["entity_id"],
                        crossing_result["crosswalk"],
                        crossing_result["class"],
                        crossing_result["direction"],
                    )

        # Periodic cleanup
        if current_frame % 30 == 0:
            self.track_interpolator.cleanup_old_tracks(current_frame, max_age=150)

        return result

    def draw_visualizations(self, frame, y_pos: int) -> int:
        """Draw pedestrian/bicycle boxes and crosswalk overlays on *frame*.

        Returns the updated y_pos for the next overlay section.
        """
        # Draw ped/bike bounding boxes
        if (
            self._last_ped_ids is not None
            and self._last_ped_boxes is not None
            and self._last_ped_classes is not None
        ):
            for i, box in enumerate(self._last_ped_boxes):
                x1, y1, x2, y2 = box
                cls_name = self.model.names[int(self._last_ped_classes[i])]
                if cls_name == "person":
                    cls_name = "pedestrian"
                color = (255, 0, 255) if cls_name == "pedestrian" else (0, 165, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                ped_label = f'{cls_name} #{int(self._last_ped_ids[i])}'
                cv2.putText(
                    frame, ped_label, (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

        # Draw crosswalk boundary lines
        if self.crosswalk_proc is not None:
            for cw in self.crosswalk_proc.crosswalks:
                for cw_line in cw.lines:
                    cv2.line(frame, cw_line.pt1, cw_line.pt2, (255, 0, 255), 2)
                    mid_x = (cw_line.pt1[0] + cw_line.pt2[0]) // 2
                    mid_y = (cw_line.pt1[1] + cw_line.pt2[1]) // 2
                    cv2.putText(
                        frame, f'{cw.name} - {cw_line.name}', (mid_x, mid_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2,
                    )

            # Draw crosswalk totals
            cw_totals = self.crosswalk_proc.get_totals()
            if cw_totals:
                y_pos += 30
                cv2.putText(
                    frame, 'Crosswalk Counts:', (20, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2,
                )
                for cls_name, total in cw_totals.items():
                    y_pos += 25
                    cv2.putText(
                        frame, f'  {cls_name}: {total}', (20, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2,
                    )

        return y_pos

    def reset_tracker(self):
        """Reset YOLO tracker state (call at trim-period boundaries)."""
        if self.model.predictor is not None:
            self.model.predictor.trackers = [None]
            print("🔄 Pedestrian model tracker reset")
        self._ema_positions.clear()

    def get_crosswalk_results(self) -> Optional[Dict[str, Any]]:
        if self.crosswalk_proc is not None:
            return self.crosswalk_proc.get_results()
        return None

    def get_crosswalk_totals(self) -> Optional[Dict[str, int]]:
        if self.crosswalk_proc is not None:
            totals = self.crosswalk_proc.get_totals()
            print(f"🚶 Crosswalk results: {totals}")
            return totals
        return None

    def finalize_crosswalk_minute_tracker(self):
        if self.crosswalk_minute_tracker is not None:
            self.crosswalk_minute_tracker.finalize()

    def release(self):
        """Release model and GPU memory."""
        del self.model
        self.model = None
        self._ema_positions.clear()
        self.track_interpolator = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _smooth_position(self, entity_id: int, cx: int, cy: int) -> Tuple[int, int]:
        """Apply EMA smoothing to reduce position jitter."""
        if entity_id in self._ema_positions:
            prev_cx, prev_cy = self._ema_positions[entity_id]
            smooth_cx = self.ema_alpha * cx + (1 - self.ema_alpha) * prev_cx
            smooth_cy = self.ema_alpha * cy + (1 - self.ema_alpha) * prev_cy
        else:
            smooth_cx, smooth_cy = float(cx), float(cy)

        self._ema_positions[entity_id] = (smooth_cx, smooth_cy)
        return int(round(smooth_cx)), int(round(smooth_cy))
