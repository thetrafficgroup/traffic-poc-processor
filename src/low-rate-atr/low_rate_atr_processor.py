import cv2
import math
import numpy as np
import json
import time
from collections import Counter
from shapely.geometry import Point, Polygon
from ultralytics import YOLO


def generate_parallel_lines(finish_line, distance=30):
    """
    Generate two parallel lines from a single finish line.
    
    Args:
        finish_line: List of two points [(x1, y1), (x2, y2)] defining the original line
        distance: Distance in pixels between the original line and the parallel lines
        
    Returns:
        Tuple of (line_a, line_b) where each is a list of two points
    """
    if not finish_line or len(finish_line) < 2:
        return None, None
    
    # Extract points
    p1 = np.array(finish_line[0], dtype=np.float32)
    p2 = np.array(finish_line[1], dtype=np.float32)
    
    # Calculate line vector and normalize
    line_vec = p2 - p1
    line_length = np.linalg.norm(line_vec)
    if line_length == 0:
        return None, None
    
    line_vec_norm = line_vec / line_length
    
    # Calculate perpendicular vector (normal to the line)
    # Rotate 90 degrees: (x, y) -> (-y, x)
    perpendicular = np.array([-line_vec_norm[1], line_vec_norm[0]])
    
    # Create parallel lines by moving along the perpendicular
    # Line A: moved in one direction
    p1_a = p1 + perpendicular * distance
    p2_a = p2 + perpendicular * distance
    line_a = [(int(p1_a[0]), int(p1_a[1])), (int(p2_a[0]), int(p2_a[1]))]
    
    # Line B: moved in opposite direction
    p1_b = p1 - perpendicular * distance
    p2_b = p2 - perpendicular * distance
    line_b = [(int(p1_b[0]), int(p1_b[1])), (int(p2_b[0]), int(p2_b[1]))]
    
    return line_a, line_b

def process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH="best.pt", progress_callback=None, generate_video_output=False, output_video_path=None, detection_distance=30):
    """
    Process low frame rate ATR video using zone-based detection.
    
    Args:
        VIDEO_PATH: Path to video file
        LINES_DATA: Configuration with lanes and finish_line (ATR format) or line_a/line_b
        MODEL_PATH: Path to YOLO model
        progress_callback: Optional callback for progress updates
        generate_video_output: Whether to generate output video
        output_video_path: Path for output video
        detection_distance: Distance in pixels for parallel lines generation (default 30)
        
    Returns:
        Dictionary with lane counts and total count
    """
    
    # Constants
    CONF_THRESHOLD = 0.1
    MAX_DIST = 100  # Maximum distance for object matching
    MAX_FRAMES_MISSING = 5  # Max frames an object can be missing before removal
    ZONE_BUFFER = 10  # Buffer pixels for zone detection
    
    # Load YOLO model
    model = YOLO(MODEL_PATH)
    
    # Process configuration
    lanes = LINES_DATA.get("lanes", [])
    
    # Convert points to tuples if needed
    def ensure_tuple_coords(points):
        if isinstance(points, list) and len(points) > 0:
            if isinstance(points[0], dict):
                return [(int(round(p["x"])), int(round(p["y"]))) for p in points]
            else:
                return [(int(round(p[0])), int(round(p[1]))) for p in points]
        return points
    
    # Process lanes
    for lane in lanes:
        if "points" in lane:
            lane["points"] = ensure_tuple_coords(lane["points"])
    
    # Get finish line from configuration (ATR format)
    finish_line = LINES_DATA.get("finish_line")
    if finish_line:
        finish_line = ensure_tuple_coords(finish_line)
    
    # Check if line_a and line_b are explicitly provided (for backward compatibility)
    line_a = LINES_DATA.get("line_a")
    line_b = LINES_DATA.get("line_b")
    
    # If line_a and line_b not provided, generate them from finish_line
    if not line_a or not line_b:
        if finish_line and len(finish_line) >= 2:
            # Generate parallel lines from the finish line
            line_a, line_b = generate_parallel_lines(finish_line, distance=detection_distance)
            print(f"[LOW-RATE-ATR] Generated parallel lines from finish_line with {detection_distance}px separation")
        else:
            print(f"[LOW-RATE-ATR WARNING] No finish_line found, using fallback detection")
            line_a = None
            line_b = None
    else:
        # Use provided lines
        line_a = ensure_tuple_coords(line_a)
        line_b = ensure_tuple_coords(line_b)
    
    # Create detection zone from lanes and finish lines
    # The zone is the area between the two finish lines
    if line_a and line_b and len(line_a) >= 2 and len(line_b) >= 2:
        # Create a polygon zone from the two lines
        zone_polygon = [
            line_a[0], line_a[1],  # First line points
            line_b[1], line_b[0]   # Second line points (reversed for proper polygon)
        ]
    else:
        # Fallback: use all lane points to create zone
        zone_polygon = []
        for lane in lanes:
            if "points" in lane:
                zone_polygon.extend(lane["points"])
    
    # Extract X boundaries from lines for detection band
    if line_a and line_b and len(line_a) >= 2 and len(line_b) >= 2:
        line_x1 = min(line_a[0][0], line_a[1][0]) - ZONE_BUFFER
        line_x2 = max(line_b[0][0], line_b[1][0]) + ZONE_BUFFER
    else:
        # Fallback: use zone polygon bounds
        if zone_polygon:
            xs = [p[0] for p in zone_polygon]
            line_x1 = min(xs) - ZONE_BUFFER
            line_x2 = max(xs) + ZONE_BUFFER
        else:
            line_x1, line_x2 = 0, 9999  # Full width
    
    # Initialize lane polygons for lane assignment
    lane_polygons = []
    lane_counts = {}
    for lane in lanes:
        if "id" in lane and "points" in lane:
            lane_id = lane["id"]
            lane_poly = Polygon(lane["points"])
            lane_polygons.append((lane_id, lane_poly))
            lane_counts[lane_id] = 0
    
    # Tracking variables
    next_id = 0
    tracked_objects = []
    counted_ids = set()
    detected_classes = {}
    
    # Video processing
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0
    last_progress_sent = -1
    start_time = time.time()
    
    # Initialize video writer if requested
    video_writer = None
    if generate_video_output and output_video_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try multiple codecs
        codecs_to_try = ['H264', 'X264', 'XVID', 'mp4v']
        for codec in codecs_to_try:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                temp_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                if temp_writer.isOpened():
                    video_writer = temp_writer
                    print(f"✅ Using video codec: {codec}")
                    break
                else:
                    temp_writer.release()
            except Exception as e:
                print(f"⚠️ Codec {codec} failed: {e}")
                continue
        
        if not video_writer:
            print("❌ Could not initialize video writer with any codec")
            generate_video_output = False
    
    def point_in_polygon(x, y, polygon):
        """Check if point is inside polygon"""
        return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (x, y), False) >= 0
    
    def euclidean_distance(p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
    
    def find_lane_for_point(x, y):
        """Find which lane a point belongs to"""
        pt = Point(x, y)
        for lane_id, lane_poly in lane_polygons:
            if lane_poly.buffer(8).contains(pt):
                return lane_id
        return None
    
    # Main processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Progress tracking
        if progress_callback and total_frames > 0:
            progress = int((frame_count / total_frames) * 100)
            
            if progress >= last_progress_sent + 5 and progress < 100:
                elapsed_time = time.time() - start_time
                if progress > 0:
                    estimated_total_time = elapsed_time / (progress / 100)
                    estimated_remaining_time = int(estimated_total_time - elapsed_time)
                else:
                    estimated_remaining_time = 0
                
                progress_callback({
                    "progress": progress,
                    "estimatedTimeRemaining": max(0, estimated_remaining_time)
                })
                last_progress_sent = progress
        
        # YOLO detection
        results = model(frame, conf=CONF_THRESHOLD)[0]
        detections = []
        
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = model.names[cls_id]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append({
                    "cx": cx, "cy": cy,
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "label": label
                })
        
        # Match detections with tracked objects
        for det in detections:
            cx, cy = det["cx"], det["cy"]
            best_match = None
            min_dist = MAX_DIST
            
            for obj in tracked_objects:
                if "matched" not in obj:  # Not yet matched in this frame
                    dist = euclidean_distance((cx, cy), (obj["cx"], obj["cy"]))
                    if dist < min_dist:
                        min_dist = dist
                        best_match = obj
            
            if best_match:
                # Update existing object
                best_match["last_cx"] = best_match["cx"]
                best_match["cx"] = cx
                best_match["cy"] = cy
                best_match["frames_missing"] = 0
                best_match["matched"] = True
                best_match["label"] = det["label"]
                best_match["bbox"] = (det["x1"], det["y1"], det["x2"], det["y2"])
            else:
                # Create new tracked object
                tracked_objects.append({
                    "id": next_id,
                    "cx": cx,
                    "cy": cy,
                    "last_cx": cx,
                    "frames_missing": 0,
                    "counted": False,
                    "label": det["label"],
                    "bbox": (det["x1"], det["y1"], det["x2"], det["y2"])
                })
                next_id += 1
        
        # Update missing frames and clean up lost objects
        objects_to_keep = []
        for obj in tracked_objects:
            if "matched" not in obj:
                obj["frames_missing"] += 1
            else:
                del obj["matched"]
            
            if obj["frames_missing"] <= MAX_FRAMES_MISSING:
                objects_to_keep.append(obj)
        
        tracked_objects = objects_to_keep
        
        # Count objects that cross the detection zone
        for obj in tracked_objects:
            if not obj["counted"]:
                # Check if object is within the detection band (X range) and zone polygon
                in_x_range = line_x1 <= obj["cx"] <= line_x2
                in_zone = point_in_polygon(obj["cx"], obj["cy"], zone_polygon) if zone_polygon else True
                
                if in_x_range and in_zone:
                    # Find which lane the object is in
                    lane_id = find_lane_for_point(obj["cx"], obj["cy"])
                    
                    if lane_id is not None:
                        lane_counts[lane_id] += 1
                        obj["counted"] = True
                        obj["lane"] = lane_id
                        counted_ids.add(obj["id"])
                        
                        # Track vehicle class
                        if obj["id"] not in detected_classes:
                            detected_classes[obj["id"]] = obj["label"]
                        
                        print(f"[LOW-RATE-ATR COUNTED] Vehicle ID={obj['id']} ({obj['label']}) | Lane={lane_id} | Lane Total: {lane_counts[lane_id]}")
        
        # Visualization if generating output video
        if generate_video_output and video_writer:
            # Draw tracked objects
            for obj in tracked_objects:
                color = (0, 255, 0) if obj["counted"] else (255, 255, 0)
                cv2.circle(frame, (obj["cx"], obj["cy"]), 5, color, -1)
                
                # Draw bounding box if available
                if "bbox" in obj:
                    x1, y1, x2, y2 = obj["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw ID and lane info
                label_text = f"{obj['label']} ID:{obj['id']}"
                if "lane" in obj:
                    label_text += f" L{obj['lane']}"
                cv2.putText(frame, label_text, (obj["cx"] + 5, obj["cy"] - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw detection lines
            if line_a and len(line_a) >= 2:
                cv2.line(frame, line_a[0], line_a[1], (0, 255, 255), 3)
                cv2.putText(frame, "Line A", line_a[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            if line_b and len(line_b) >= 2:
                cv2.line(frame, line_b[0], line_b[1], (0, 255, 255), 3)
                cv2.putText(frame, "Line B", line_b[0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Draw detection zone
            if zone_polygon:
                cv2.polylines(frame, [np.array(zone_polygon, dtype=np.int32)], 
                             isClosed=True, color=(255, 0, 0), thickness=2)
            
            # Draw lanes
            for lane in lanes:
                if "points" in lane and "id" in lane:
                    pts = np.array(lane["points"], dtype=np.int32)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=1)
                    cv2.putText(frame, f"L{lane['id']}: {lane_counts.get(lane['id'], 0)}", 
                               lane["points"][0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            # Draw counts
            total_count = sum(lane_counts.values())
            y_pos = 30
            cv2.putText(frame, f"Total: {total_count}", (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            
            y_pos += 30
            for lane_id, count in lane_counts.items():
                cv2.putText(frame, f"Lane {lane_id}: {count}", (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
                y_pos += 25
            
            # Write frame
            video_writer.write(frame)
    
    # Cleanup
    cap.release()
    if video_writer:
        video_writer.release()
    
    # Calculate final results
    total_count = sum(lane_counts.values())
    
    # Convert detected_classes to summary
    class_summary = Counter(detected_classes.values())
    
    # Debug output
    print(f"[LOW-RATE-ATR DEBUG] Total objects tracked: {next_id}")
    print(f"[LOW-RATE-ATR DEBUG] Objects counted: {len(counted_ids)}")
    print(f"[LOW-RATE-ATR DEBUG] Lane counts: {lane_counts}")
    print(f"[LOW-RATE-ATR DEBUG] Total count: {total_count}")
    print(f"[LOW-RATE-ATR DEBUG] Vehicle classes: {dict(class_summary)}")
    
    return {
        "lane_counts": lane_counts,
        "total_count": total_count,
        "study_type": "LOW_RATE_ATR",
        "detected_classes": dict(class_summary),
        "metadata": {
            "detection_method": "zone_based",
            "line_a": line_a,
            "line_b": line_b,
            "total_tracked": next_id,
            "total_counted": len(counted_ids)
        }
    }