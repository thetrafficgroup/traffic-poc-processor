import cv2
import numpy as np
import json
import time
from shapely.geometry import Point, Polygon
from ultralytics import YOLO
from collections import OrderedDict, Counter
from .atr_minute_tracker import ATRMinuteTracker
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.frame_utils import calculate_frame_ranges_from_seconds, validate_trim_periods

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

def get_wheels_position(box):
    """Get the wheel position (bottom center) of the bounding box"""
    x1, y1, x2, y2 = box
    # Wheels are at the bottom center of the vehicle
    return int((x1 + x2) / 2), int(y2)

def find_vehicle_lane(centroid_x, centroid_y, wheels_x, wheels_y, lane_polygons_buffered):
    """
    Find vehicle lane using wheels-priority approach.
    First checks wheel position, falls back to centroid if no match.
    
    Returns:
        lane_id or None
    """
    # Priority 1: Check wheels position
    wheels_point = Point(wheels_x, wheels_y)
    for lane_id, buffered_polygon in lane_polygons_buffered:
        if buffered_polygon.contains(wheels_point):
            return lane_id
    
    # Priority 2: Fallback to centroid
    centroid_point = Point(centroid_x, centroid_y)
    for lane_id, buffered_polygon in lane_polygons_buffered:
        if buffered_polygon.contains(centroid_point):
            return lane_id
    
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

def process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH="best.pt", progress_callback=None, generate_video_output=False, output_video_path=None, video_uuid=None, minute_batch_callback=None, trim_periods=None):
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

    # Load YOLO model
    model = YOLO(MODEL_PATH)
    
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
    
    # Initialize tracking variables
    counted_ids = set()
    previous_positions = {}
    tracker = CentroidTracker(max_disappeared=15)
    
    # Debug counters
    debug_total_detections = 0
    debug_tracked_objects = set()
    detected_classes = {}
    class_counts_by_id = {}

    # Track raw detection labels by lane (use defaultdict pattern)
    from collections import defaultdict
    vehicle_counts_by_lane = defaultdict(lambda: defaultdict(int))
    
    # Initialize video capture
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
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
        if progress_callback:
            elapsed_time = time.time() - start_time
            # During seeking, we can't accurately estimate time remaining
            # Just show that we're seeking
            progress_callback({
                "progress": int((frame_count / total_frames) * 100),
                "estimatedTimeRemaining": 0,
                "status": "seeking"
            })

    # Helper function to reset tracker state
    def reset_centroid_tracker():
        """Reset CentroidTracker to start fresh tracking for new period"""
        nonlocal tracker
        tracker = CentroidTracker(max_disappeared=15)
        print("🔄 ATR CentroidTracker reset - previous tracking state cleared")

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

        # Send progress every 5%
        if progress >= last_progress_sent + 5 and progress < 100:
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

            progress_callback({
                "progress": progress,
                "estimatedTimeRemaining": max(0, estimated_remaining_time)
            })
            last_progress_sent = progress

    # Main processing logic with frame-skipping support
    if frame_ranges:
        # TRIMMING MODE: Process only specified periods with frame-skipping
        print("🎬 Starting trimmed ATR video processing")

        for period_idx, period in enumerate(frame_ranges):
            start_frame = period["start_frame"]
            end_frame = period["end_frame"]
            period_duration = (period["end_seconds"] - period["start_seconds"]) / 60  # minutes

            print(f"\n📍 ATR Period {period_idx + 1}/{len(frame_ranges)}")
            print(f"   Frames: {start_frame} - {end_frame} ({end_frame - start_frame} frames)")
            print(f"   Time: {period['start_seconds']:.1f}s - {period['end_seconds']:.1f}s ({period_duration:.1f} min)")

            # CRITICAL: Reset tracker at start of each period
            reset_centroid_tracker()

            # Clear previous positions to prevent cross-period tracking
            previous_positions.clear()
            print("🧹 Previous positions cleared for new period")

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
                results = model.predict(frame, conf=CONF_THRESHOLD)
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

                # Process tracked objects
                for objectID, centroid in objects.items():
                    debug_tracked_objects.add(objectID)
                    cx, cy = centroid
                    pt = Point(cx, cy)
                    lane_id = None

                    # Find class for this detection
                    class_name = "unknown"
                    for (det_cx, det_cy), detection_data in detections_map.items():
                        if abs(det_cx - cx) < 20 and abs(det_cy - cy) < 20:  # Match centroid
                            if len(detection_data) > 4:  # Has class_name
                                class_name = detection_data[4]
                            break
                    class_counts_by_id[objectID] = class_name

                    # Find which lane the object is in using wheels-priority approach
                    wheels_x, wheels_y = None, None
                    # Extract wheels position from detection data if available
                    for (det_cx, det_cy), detection_data in detections_map.items():
                        if abs(det_cx - cx) < 20 and abs(det_cy - cy) < 20:  # Match centroid
                            if len(detection_data) > 6:  # Has wheels coordinates
                                wheels_x, wheels_y = detection_data[5], detection_data[6]
                            break

                    # Use wheels-priority detection if wheels data available
                    if wheels_x is not None and wheels_y is not None:
                        lane_id = find_vehicle_lane(cx, cy, wheels_x, wheels_y, lane_polygons_buffered)
                    else:
                        # Fallback to original centroid-only method for compatibility
                        for lid, buffered_polygon in lane_polygons_buffered:
                            if buffered_polygon.contains(pt):
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

                            # Count detected class only ONCE per unique object ID
                            if objectID not in detected_classes:
                                detected_classes[objectID] = class_name

                            # Use raw detection label (snake_case) directly
                            vehicle_counts_by_lane[class_name][lane_id] += 1

                            # Track in minute tracker if enabled (pass raw class name)
                            if minute_tracker:
                                minute_tracker.process_vehicle_detection(frame_count, objectID, class_name, lane_id)

                            print(f"[ATR COUNTED] Vehicle ID={objectID} ({class_name}) | Lane={lane_id} | Lane Total: {lane_counts[lane_id]}")

                # Add visualizations if generating output video
                if generate_video_output and video_writer:
                    # Draw detections and tracking
                    for objectID, centroid in objects.items():
                        cx, cy = int(centroid[0]), int(centroid[1])

                        # Find lane for this object using wheels-priority approach
                        pt = Point(cx, cy)
                        lane_id = None
                        # Try to get wheels position from detections_map
                        wheels_x, wheels_y = None, None
                        for (det_cx, det_cy), detection_data in detections_map.items():
                            if abs(det_cx - cx) < 20 and abs(det_cy - cy) < 20:
                                if len(detection_data) > 6:
                                    wheels_x, wheels_y = detection_data[5], detection_data[6]
                                break

                        if wheels_x is not None and wheels_y is not None:
                            lane_id = find_vehicle_lane(cx, cy, wheels_x, wheels_y, lane_polygons_buffered)
                        else:
                            for lid, buffered_polygon in lane_polygons_buffered:
                                if buffered_polygon.contains(pt):
                                    lane_id = lid
                                    break

                        # Draw bounding box if available
                        if (cx, cy) in detections_map:
                            detection_data = detections_map[(cx, cy)]
                            x1, y1, x2, y2, class_name_viz = detection_data[:5]
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                            # Draw wheels position if available (for debugging)
                            if len(detection_data) > 6:
                                wx, wy = detection_data[5], detection_data[6]
                                cv2.circle(frame, (int(wx), int(wy)), 3, (255, 0, 0), -1)  # Blue for wheels

                        # Draw centroid
                        color = (0, 255, 0) if lane_id is not None else (0, 0, 255)
                        cv2.circle(frame, (cx, cy), 5, color, -1)
                        cv2.putText(frame, f'ID {objectID} | L{lane_id}', (cx, cy - 10),
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

        # Process video frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO detection
            results = model.predict(frame, conf=CONF_THRESHOLD)
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

            # Process tracked objects
            for objectID, centroid in objects.items():
                debug_tracked_objects.add(objectID)
                cx, cy = centroid
                pt = Point(cx, cy)
                lane_id = None

                # Find class for this detection
                class_name = "unknown"
                for (det_cx, det_cy), detection_data in detections_map.items():
                    if abs(det_cx - cx) < 20 and abs(det_cy - cy) < 20:  # Match centroid
                        if len(detection_data) > 4:  # Has class_name
                            class_name = detection_data[4]
                        break
                class_counts_by_id[objectID] = class_name

                # Find which lane the object is in using wheels-priority approach
                wheels_x, wheels_y = None, None
                # Extract wheels position from detection data if available
                for (det_cx, det_cy), detection_data in detections_map.items():
                    if abs(det_cx - cx) < 20 and abs(det_cy - cy) < 20:  # Match centroid
                        if len(detection_data) > 6:  # Has wheels coordinates
                            wheels_x, wheels_y = detection_data[5], detection_data[6]
                        break

                # Use wheels-priority detection if wheels data available
                if wheels_x is not None and wheels_y is not None:
                    lane_id = find_vehicle_lane(cx, cy, wheels_x, wheels_y, lane_polygons_buffered)
                else:
                    # Fallback to original centroid-only method for compatibility
                    for lid, buffered_polygon in lane_polygons_buffered:
                        if buffered_polygon.contains(pt):
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

                        # Count detected class only ONCE per unique object ID
                        if objectID not in detected_classes:
                            detected_classes[objectID] = class_name

                        # Use raw detection label (snake_case) directly
                        vehicle_counts_by_lane[class_name][lane_id] += 1

                        # Track in minute tracker if enabled (pass raw class name)
                        if minute_tracker:
                            minute_tracker.process_vehicle_detection(frame_count, objectID, class_name, lane_id)

                        print(f"[ATR COUNTED] Vehicle ID={objectID} ({class_name}) | Lane={lane_id} | Lane Total: {lane_counts[lane_id]}")

            # Add visualizations if generating output video
            if generate_video_output and video_writer:
                # Draw detections and tracking
                for objectID, centroid in objects.items():
                    cx, cy = int(centroid[0]), int(centroid[1])

                    # Find lane for this object using wheels-priority approach
                    pt = Point(cx, cy)
                    lane_id = None
                    # Try to get wheels position from detections_map
                    wheels_x, wheels_y = None, None
                    for (det_cx, det_cy), detection_data in detections_map.items():
                        if abs(det_cx - cx) < 20 and abs(det_cy - cy) < 20:
                            if len(detection_data) > 6:
                                wheels_x, wheels_y = detection_data[5], detection_data[6]
                            break

                    if wheels_x is not None and wheels_y is not None:
                        lane_id = find_vehicle_lane(cx, cy, wheels_x, wheels_y, lane_polygons_buffered)
                    else:
                        for lid, buffered_polygon in lane_polygons_buffered:
                            if buffered_polygon.contains(pt):
                                lane_id = lid
                                break

                    # Draw bounding box if available
                    if (cx, cy) in detections_map:
                        detection_data = detections_map[(cx, cy)]
                        x1, y1, x2, y2, class_name_viz = detection_data[:5]
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Draw wheels position if available (for debugging)
                        if len(detection_data) > 6:
                            wx, wy = detection_data[5], detection_data[6]
                            cv2.circle(frame, (int(wx), int(wy)), 3, (255, 0, 0), -1)  # Blue for wheels

                    # Draw centroid
                    color = (0, 255, 0) if lane_id is not None else (0, 0, 255)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    cv2.putText(frame, f'ID {objectID} | L{lane_id}', (cx, cy - 10),
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
    
    # Convert detected_classes from {obj_id: class_name} to {class_name: count}
    class_summary = Counter(detected_classes.values())

    # Convert vehicle_counts_by_lane from defaultdict to regular dict for JSON serialization
    # Structure: {detection_label: {lane_id: count}}
    vehicles_by_class = {class_name: dict(lane_data) for class_name, lane_data in vehicle_counts_by_lane.items()}

    # Create detected_classes summary from raw detection labels
    detected_classes_summary = {}
    for class_name, lane_data in vehicle_counts_by_lane.items():
        detected_classes_summary[class_name] = sum(lane_data.values())

    return {
        "lane_counts": lane_counts,
        "total_count": total_count,
        "study_type": "ATR",
        "detected_classes": detected_classes_summary,  # Raw detection labels with counts
        "vehicles": vehicles_by_class  # Raw detection labels by lane: {class_name: {lane_id: count}}
    }