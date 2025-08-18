"""
Overlap detection utilities for improving vehicle tracking accuracy.
Implements Soft-NMS, occlusion handling, and confidence adjustment.
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from collections import defaultdict, deque


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def soft_nms(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, 
             sigma: float = 0.5, iou_threshold: float = 0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Soft-NMS to reduce confidence of overlapping detections instead of eliminating them.
    
    Args:
        boxes: Detection boxes in [x1, y1, x2, y2] format
        scores: Confidence scores
        classes: Class IDs
        sigma: Soft-NMS parameter (higher = gentler suppression)
        iou_threshold: IoU threshold for applying soft suppression
    
    Returns:
        Filtered boxes, scores, and classes
    """
    if len(boxes) == 0:
        return boxes, scores, classes
    
    # Convert to numpy arrays if needed
    boxes = np.array(boxes)
    scores = np.array(scores, dtype=np.float32)
    classes = np.array(classes)
    
    # Apply soft-NMS
    for i in range(len(boxes)):
        if scores[i] == 0:  # Already suppressed
            continue
            
        for j in range(i + 1, len(boxes)):
            if scores[j] == 0:  # Already suppressed
                continue
                
            iou = compute_iou(boxes[i], boxes[j])
            
            if iou > iou_threshold:
                # Apply soft suppression instead of hard removal
                scores[j] *= np.exp(-(iou ** 2) / sigma)
    
    # Filter out very low confidence detections
    keep = scores > 0.01
    return boxes[keep], scores[keep], classes[keep]


def detect_occlusions(boxes: np.ndarray, ids: np.ndarray, threshold: float = 0.3) -> Dict[int, List[int]]:
    """
    Detect occlusion relationships between vehicles.
    
    Args:
        boxes: Detection boxes
        ids: Object IDs
        threshold: IoU threshold for considering occlusion
    
    Returns:
        Dictionary mapping object_id -> list of occluding object_ids
    """
    occlusions = defaultdict(list)
    
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = compute_iou(boxes[i], boxes[j])
            
            if iou > threshold:
                # Determine which object is likely in front based on size
                area_i = (boxes[i][2] - boxes[i][0]) * (boxes[i][3] - boxes[i][1])
                area_j = (boxes[j][2] - boxes[j][0]) * (boxes[j][3] - boxes[j][1])
                
                if area_i > area_j:
                    # Object i is likely in front (larger)
                    occlusions[ids[j]].append(ids[i])
                else:
                    # Object j is likely in front (larger)
                    occlusions[ids[i]].append(ids[j])
    
    return dict(occlusions)


def adjust_confidence_for_occlusion(scores: np.ndarray, occlusion_levels: np.ndarray, 
                                   boost_factor: float = 1.2) -> np.ndarray:
    """
    Adjust detection confidence based on occlusion level.
    Boost confidence for partially occluded vehicles to prevent track loss.
    
    Args:
        scores: Original confidence scores
        occlusion_levels: Occlusion level (0-1) for each detection
        boost_factor: Multiplier for occluded detections
    
    Returns:
        Adjusted confidence scores
    """
    adjusted_scores = scores.copy()
    
    for i, occlusion_level in enumerate(occlusion_levels):
        if 0.3 < occlusion_level <= 0.7:  # Medium occlusion
            adjusted_scores[i] *= boost_factor
        elif occlusion_level > 0.7:  # Heavy occlusion
            adjusted_scores[i] *= (boost_factor * 1.2)
    
    # Ensure scores don't exceed 1.0
    adjusted_scores = np.clip(adjusted_scores, 0.0, 1.0)
    
    return adjusted_scores


class TrackInterpolator:
    """
    Handles track interpolation for temporarily lost detections due to occlusions.
    """
    
    def __init__(self, max_missing_frames: int = 10, min_track_length: int = 3):
        """
        Args:
            max_missing_frames: Maximum frames to interpolate
            min_track_length: Minimum track length before interpolation
        """
        self.max_missing_frames = max_missing_frames
        self.min_track_length = min_track_length
        self.track_history = defaultdict(lambda: deque(maxlen=20))
        self.missing_frames = defaultdict(int)
    
    def update_track(self, obj_id: int, centroid: Tuple[int, int], frame_num: int):
        """Update track history with new detection."""
        self.track_history[obj_id].append((centroid, frame_num))
        self.missing_frames[obj_id] = 0
    
    def handle_missing_detection(self, obj_id: int, frame_num: int) -> Optional[Tuple[int, int]]:
        """
        Handle missing detection by interpolating position.
        
        Returns:
            Interpolated centroid if successful, None if track should be terminated
        """
        self.missing_frames[obj_id] += 1
        
        if (self.missing_frames[obj_id] > self.max_missing_frames or 
            len(self.track_history[obj_id]) < self.min_track_length):
            return None
        
        # Linear interpolation based on recent motion
        recent_positions = list(self.track_history[obj_id])
        if len(recent_positions) < 2:
            return None
        
        # Calculate velocity from recent positions
        pos1, frame1 = recent_positions[-2]
        pos2, frame2 = recent_positions[-1]
        
        if frame2 == frame1:
            return pos2  # No motion detected
        
        # Predict next position
        velocity_x = (pos2[0] - pos1[0]) / (frame2 - frame1)
        velocity_y = (pos2[1] - pos1[1]) / (frame2 - frame1)
        
        frames_ahead = frame_num - frame2
        predicted_x = int(pos2[0] + velocity_x * frames_ahead)
        predicted_y = int(pos2[1] + velocity_y * frames_ahead)
        
        return (predicted_x, predicted_y)
    
    def cleanup_old_tracks(self, current_frame: int, max_age: int = 100):
        """Remove old tracks to prevent memory buildup."""
        to_remove = []
        for obj_id, history in self.track_history.items():
            if history and current_frame - history[-1][1] > max_age:
                to_remove.append(obj_id)
        
        for obj_id in to_remove:
            del self.track_history[obj_id]
            if obj_id in self.missing_frames:
                del self.missing_frames[obj_id]


def analyze_overlap_patterns(boxes: np.ndarray, ids: np.ndarray, 
                           frame_history: Dict[int, List]) -> Dict[str, float]:
    """
    Analyze overlap patterns to provide insights on detection quality.
    
    Returns:
        Dictionary with overlap statistics
    """
    stats = {
        'total_detections': len(boxes),
        'overlapping_pairs': 0,
        'avg_overlap_iou': 0.0,
        'max_overlap_iou': 0.0,
        'overlap_ratio': 0.0
    }
    
    if len(boxes) < 2:
        return stats
    
    overlap_ious = []
    overlapping_pairs = 0
    
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            iou = compute_iou(boxes[i], boxes[j])
            if iou > 0.1:  # Consider as overlap
                overlap_ious.append(iou)
                overlapping_pairs += 1
    
    if overlap_ious:
        stats['overlapping_pairs'] = overlapping_pairs
        stats['avg_overlap_iou'] = np.mean(overlap_ious)
        stats['max_overlap_iou'] = np.max(overlap_ious)
        stats['overlap_ratio'] = overlapping_pairs / len(boxes)
    
    return stats


def post_process_detections(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray, 
                          ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Apply complete overlap detection post-processing pipeline.
    
    Args:
        boxes: Detection boxes
        scores: Confidence scores
        classes: Class IDs
        ids: Object IDs (optional)
    
    Returns:
        Post-processed boxes, scores, classes, and ids
    """
    if len(boxes) == 0:
        return boxes, scores, classes, ids
    
    # 1. Apply Soft-NMS
    processed_boxes, processed_scores, processed_classes = soft_nms(
        boxes, scores, classes, sigma=0.5, iou_threshold=0.3
    )
    
    # 2. Detect occlusions if IDs are available
    if ids is not None and len(processed_boxes) > 0:
        # Filter IDs to match processed detections
        keep_indices = np.isin(np.arange(len(boxes)), 
                              np.where(scores > 0.01)[0][:len(processed_boxes)])
        processed_ids = ids[keep_indices] if np.any(keep_indices) else ids[:len(processed_boxes)]
        
        # Detect occlusions
        occlusions = detect_occlusions(processed_boxes, processed_ids)
        
        # Calculate occlusion levels (simplified)
        occlusion_levels = np.zeros(len(processed_boxes))
        for i, obj_id in enumerate(processed_ids):
            if obj_id in occlusions:
                # Simple occlusion level based on number of occluding objects
                occlusion_levels[i] = min(0.8, len(occlusions[obj_id]) * 0.3)
        
        # 3. Adjust confidence for occluded vehicles
        processed_scores = adjust_confidence_for_occlusion(
            processed_scores, occlusion_levels, boost_factor=1.15
        )
        
        return processed_boxes, processed_scores, processed_classes, processed_ids
    
    return processed_boxes, processed_scores, processed_classes, ids