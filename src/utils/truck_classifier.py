"""
truck_classifier.py - Second-stage classifier for articulated truck subtyping.

After the main detection model identifies an 'articulated_truck', this classifier
examines the cropped region to distinguish:
  0: articulated_truck       (single trailer)
  1: multi_articulated_truck  (double/triple trailer)

Usage:
    classifier = TruckClassifier("best_truck_classifier.pt")
    refined_class = classifier.classify(frame, (x1, y1, x2, y2))
"""

import numpy as np
from ultralytics import YOLO

# Minimum crop dimension in pixels — below this the crop is too small for reliable classification
_MIN_CROP_SIZE = 20

# Padding ratio added to each side of the crop (0.1 = 10%)
_CROP_PADDING_RATIO = 0.10


class TruckClassifier:
    """
    Second-stage YOLOv8-cls model that refines 'articulated_truck' detections
    into 'articulated_truck' or 'multi_articulated_truck'.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.7):
        """
        Args:
            model_path: Path to the YOLOv8-cls truck classifier weights.
            confidence_threshold: Minimum confidence to accept 'multi_articulated_truck'.
                Below this threshold, the detection stays as 'articulated_truck'.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        print(f"[TruckClassifier] Loaded: {model_path} (threshold={confidence_threshold})")

    def classify(self, frame: np.ndarray, box) -> str:
        """
        Classify a detected truck region as articulated or multi-articulated.

        Args:
            frame: Full video frame (BGR, HxWxC numpy array).
            box: Bounding box as (x1, y1, x2, y2) — accepts numpy arrays, tuples, or lists.

        Returns:
            'articulated_truck' or 'multi_articulated_truck'
        """
        try:
            h, w = frame.shape[:2]

            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

            # Add padding to capture trailer hitch / second trailer that may be outside tight bbox
            pad_x = (x2 - x1) * _CROP_PADDING_RATIO
            pad_y = (y2 - y1) * _CROP_PADDING_RATIO
            x1 -= pad_x
            y1 -= pad_y
            x2 += pad_x
            y2 += pad_y

            # Clamp to frame bounds
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(w, int(x2))
            y2 = min(h, int(y2))

            # Skip tiny crops
            if (x2 - x1) < _MIN_CROP_SIZE or (y2 - y1) < _MIN_CROP_SIZE:
                return "articulated_truck"

            crop = frame[y1:y2, x1:x2]

            results = self.model.predict(crop, verbose=False)
            probs = results[0].probs
            top_class = int(probs.top1)
            confidence = float(probs.top1conf)
            class_name = self.model.names[top_class]

            if class_name == "multi_articulated_truck" and confidence >= self.confidence_threshold:
                return "multi_articulated_truck"

            return "articulated_truck"

        except Exception as e:
            print(f"[TruckClassifier] Error classifying crop: {e}")
            return "articulated_truck"
