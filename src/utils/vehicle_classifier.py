"""
Vehicle classification utilities for multi-trailer articulated truck detection.

Reclassifies articulated trucks based on aspect ratio to distinguish between
single-trailer and multi-trailer configurations.
"""

MULTI_TRAILER_ASPECT_RATIO_THRESHOLD = 2.5


def classify_articulated_truck(class_name: str, x1: float, y1: float, x2: float, y2: float) -> str:
    """
    Reclassify articulated trucks based on aspect ratio.
    Multi-trailer trucks have longer bounding boxes (higher width/height ratio).

    Args:
        class_name: Original class name from YOLO model
        x1, y1: Top-left corner of bounding box
        x2, y2: Bottom-right corner of bounding box

    Returns:
        Original class_name, or 'multi_trailer_articulated_truck' if aspect ratio > threshold
    """
    if class_name != "articulated_truck":
        return class_name

    width = x2 - x1
    height = y2 - y1

    if height <= 0:
        return class_name

    aspect_ratio = width / height

    if aspect_ratio > MULTI_TRAILER_ASPECT_RATIO_THRESHOLD:
        return "multi_trailer_articulated_truck"

    return class_name
