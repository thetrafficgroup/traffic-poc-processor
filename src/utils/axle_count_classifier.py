"""
axle_count_classifier.py - Detect wheels and count axles for FHWA classification.

Uses a YOLO detection model to find wheels, then clusters them into axles
based on horizontal (X) proximity. Combined with coarse vehicle class,
determines specific FHWA class (5-13).

Usage:
    classifier = AxleCountClassifier("best_axle_detector.pt")
    axle_count = classifier.detect_axles(frame, bbox)
    fhwa_class = AxleCountClassifier.get_fhwa_class("articulated_truck", axle_count)
"""

import numpy as np
from typing import Optional, List
from ultralytics import YOLO

# Minimum crop size for reliable detection
_MIN_CROP_SIZE = 80

# Padding ratio added to crop (captures wheels at edges)
_CROP_PADDING_RATIO = 0.15

# X-distance threshold for clustering wheels into axles (relative to crop width)
_AXLE_MERGE_THRESHOLD_RATIO = 0.08  # 8% of crop width

# Direct FHWA mappings for non-truck classes
DIRECT_FHWA_MAP = {
    "motorcycle": 1,
    "car": 2,
    "pickup_truck": 3,
    "work_van": 3,
    "motorized_vehicle": 2,
    "bus": 4,
}

# Default FHWA when axle detection is unavailable (conservative - lowest in range)
DEFAULT_TRUCK_FHWA = {
    "single_unit_truck": 5,
    "articulated_truck": 8,
    "multi_articulated_truck": 11,
}


class AxleCountClassifier:
    """
    Detects wheels in truck crops and counts axles for precise FHWA classification.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.25):
        """
        Args:
            model_path: Path to YOLOv8 wheel detection weights.
            confidence_threshold: Minimum confidence for wheel detections.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"[AxleCountClassifier] Loaded: {model_path} (conf={confidence_threshold})")

    def detect_axles(self, frame: np.ndarray, box: tuple) -> Optional[int]:
        """
        Detect wheels in truck crop and count axles.

        Args:
            frame: Full video frame (BGR numpy array)
            box: (x1, y1, x2, y2) bounding box of the truck

        Returns:
            Number of axles detected, or None if detection failed
        """
        try:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

            # Add padding to capture wheels at edges
            pad_x = (x2 - x1) * _CROP_PADDING_RATIO
            pad_y = (y2 - y1) * _CROP_PADDING_RATIO
            x1 = max(0, int(x1 - pad_x))
            y1 = max(0, int(y1 - pad_y))
            x2 = min(w, int(x2 + pad_x))
            y2 = min(h, int(y2 + pad_y))

            # Skip tiny crops
            crop_width = x2 - x1
            crop_height = y2 - y1
            if crop_width < _MIN_CROP_SIZE or crop_height < _MIN_CROP_SIZE:
                return None

            crop = frame[y1:y2, x1:x2]

            # Detect wheels
            results = self.model.predict(crop, conf=self.confidence_threshold, verbose=False)
            boxes = results[0].boxes

            if boxes is None or len(boxes) == 0:
                return None

            # Extract wheel X-centers (relative to crop)
            wheel_x_centers = []
            for det_box in boxes:
                wx1, wy1, wx2, wy2 = det_box.xyxy[0].cpu().numpy()
                x_center = (wx1 + wx2) / 2
                wheel_x_centers.append(x_center)

            # Cluster into axles
            merge_threshold = crop_width * _AXLE_MERGE_THRESHOLD_RATIO
            axle_count = self._cluster_wheels_to_axles(wheel_x_centers, merge_threshold)

            return axle_count

        except Exception as e:
            print(f"[AxleCountClassifier] Error detecting axles: {e}")
            return None

    def _cluster_wheels_to_axles(self, x_centers: List[float], threshold: float) -> int:
        """Cluster wheel X-positions into axles."""
        if not x_centers:
            return 0

        sorted_x = sorted(x_centers)
        axle_count = 1
        prev_x = sorted_x[0]

        for x in sorted_x[1:]:
            if x - prev_x > threshold:
                axle_count += 1
            prev_x = x

        return axle_count

    @staticmethod
    def get_fhwa_class(coarse_class: str, axle_count: Optional[int]) -> Optional[int]:
        """
        Map coarse class + axle count to FHWA class.

        Args:
            coarse_class: Detection label (e.g., 'articulated_truck')
            axle_count: Number of axles, or None if unknown

        Returns:
            FHWA class number (1-13), or None for unknown classes
        """
        # Direct mappings (non-trucks)
        if coarse_class in DIRECT_FHWA_MAP:
            return DIRECT_FHWA_MAP[coarse_class]

        # Truck mappings (axle-based)
        if coarse_class == "single_unit_truck":
            # Range: 5-7
            if axle_count is not None and axle_count >= 4:
                return 7  # 4+ axle
            elif axle_count == 3:
                return 6  # 3-axle
            return 5  # Default: 2-axle (conservative)

        elif coarse_class == "articulated_truck":
            # Range: 8-10
            if axle_count is None or axle_count <= 4:
                return 8  # Default (conservative)
            elif axle_count == 5:
                return 9  # Classic 18-wheeler
            else:
                return 10  # 6+ axle

        elif coarse_class == "multi_articulated_truck":
            # Range: 11-13
            if axle_count is None or axle_count <= 5:
                return 11  # Default (conservative)
            elif axle_count == 6:
                return 12
            else:
                return 13  # 7+ axle

        return None
