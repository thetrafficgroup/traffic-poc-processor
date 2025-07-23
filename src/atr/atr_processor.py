import cv2
import numpy as np
import json
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
from collections import OrderedDict

# === Centroid Tracker ===
class CentroidTracker:
    def __init__(self, max_disappeared=15):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.max_disappeared = max_disappeared

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

def dict_points_to_tuples(points):
    return [(pt["x"], pt["y"]) if isinstance(pt, dict) else tuple(pt) for pt in points]

def point_side_of_line(p, a, b):
    # Devuelve >0 si p está a la izquierda de ab, <0 derecha, 0 sobre la línea
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

def process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH="best.pt", progress_callback=None):
    """
    Process video for ATR (Automatic Traffic Recording) analysis.
    
    Args:
        VIDEO_PATH: Path to video file
        LINES_DATA: Lane configuration data with lanes and finish_line
        MODEL_PATH: Path to YOLO model
        progress_callback: Optional callback for progress updates
        
    Returns:
        Dictionary with lane counts and total count
    """
    
    # Constants
    CONF_THRESHOLD = 0.1
    
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    
    # Process lanes configuration
    lanes = LINES_DATA["lanes"]
    for lane in lanes:
        lane["points"] = dict_points_to_tuples(lane["points"])
    lane_polygons = [(lane["id"], Polygon(lane["points"])) for lane in lanes]
    lane_counts = {lane_id: 0 for lane_id, _ in lane_polygons}
    
    # Process finish line
    finish_line = LINES_DATA.get("finish_line")
    if finish_line and isinstance(finish_line[0], dict):
        finish_line = dict_points_to_tuples(finish_line)
    
    # Initialize tracking variables
    counted_ids = set()
    previous_positions = {}
    tracker = CentroidTracker(max_disappeared=15)
    
    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    
    # Process video frames
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Update progress
        if progress_callback:
            progress = (frame_count / total_frames) * 100
            progress_callback(progress)
        
        # YOLO detection
        results = model.predict(frame, conf=CONF_THRESHOLD)
        boxes = results[0].boxes
        
        input_centroids = []
        detections_map = {}
        
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx, cy = get_centroid((x1, y1, x2, y2))
            input_centroids.append(np.array([cx, cy]))
            detections_map[(cx, cy)] = (x1, y1, x2, y2)
        
        # Update tracker
        objects = tracker.update(np.array(input_centroids))
        
        # Process tracked objects
        for objectID, centroid in objects.items():
            cx, cy = centroid
            pt = Point(cx, cy)
            lane_id = None
            
            # Find which lane the object is in
            for lid, polygon in lane_polygons:
                if polygon.buffer(8).contains(pt):
                    lane_id = lid
                    break
            
            # Initialize position history
            if objectID not in previous_positions:
                previous_positions[objectID] = []
            
            previous_positions[objectID].append((cx, cy))
            
            # Check finish line crossing
            if (
                objectID not in counted_ids and
                lane_id is not None and
                finish_line is not None and
                len(previous_positions[objectID]) >= 2
            ):
                a, b = finish_line
                prev = previous_positions[objectID][-2]
                curr = previous_positions[objectID][-1]
                side_prev = point_side_of_line(prev, a, b)
                side_curr = point_side_of_line(curr, a, b)
                
                if side_prev * side_curr < 0:  # Changed sides
                    counted_ids.add(objectID)
                    lane_counts[lane_id] += 1
    
    cap.release()
    
    # Return results
    total_count = sum(lane_counts.values())
    
    return {
        "lane_counts": lane_counts,
        "total_count": total_count,
        "study_type": "ATR"
    }