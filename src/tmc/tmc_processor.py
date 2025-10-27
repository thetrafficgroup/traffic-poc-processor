import cv2
import json
import time
from collections import Counter
from ultralytics import YOLO
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.overlap_detection import (
    soft_nms, detect_occlusions, adjust_confidence_for_occlusion,
    TrackInterpolator, post_process_detections, analyze_overlap_patterns
)
from utils.minute_tracker import MinuteTracker
from utils.frame_utils import calculate_frame_ranges_from_seconds, validate_trim_periods

CONF_THRESHOLD = 0.01
IMG_SIZE = 640
IOU_THRESHOLD = 0.2
DIST_THRESHOLD = 10



def build_analysis_by_vehicle_class(detected_classes, turn_types_by_id, crossing_timestamps, crossed_lines_by_id):
    """
    Build new analysis structure grouped by vehicle class first.
    Structure: vehicle_class -> origin_direction -> turn_direction -> count
    """
    analysis = {}
    
    # Initialize totals structure
    totals = {
        "NORTH": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
        "SOUTH": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
        "EAST": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
        "WEST": {"straight": 0, "left": 0, "right": 0, "u-turn": 0}
    }
    
    # Group vehicles by class
    vehicles_by_class = {}
    for obj_id, vehicle_class in detected_classes.items():
        if vehicle_class not in vehicles_by_class:
            vehicles_by_class[vehicle_class] = []
        vehicles_by_class[vehicle_class].append(obj_id)
    
    # For each vehicle class, analyze movements
    for vehicle_class, vehicle_ids in vehicles_by_class.items():
        analysis[vehicle_class] = {
            "NORTH": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
            "SOUTH": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
            "EAST": {"straight": 0, "left": 0, "right": 0, "u-turn": 0},
            "WEST": {"straight": 0, "left": 0, "right": 0, "u-turn": 0}
        }
        
        for vehicle_id in vehicle_ids:
            # Determine origin direction (first line crossed)
            if vehicle_id in crossing_timestamps and crossing_timestamps[vehicle_id]:
                origin_direction = crossing_timestamps[vehicle_id][0][0]
                
                # Determine turn type
                if vehicle_id in turn_types_by_id:
                    turn_type = turn_types_by_id[vehicle_id]
                else:
                    # If no turn detected, assume straight
                    turn_type = "straight"
                
                # Increment counters
                if origin_direction in analysis[vehicle_class] and turn_type in analysis[vehicle_class][origin_direction]:
                    analysis[vehicle_class][origin_direction][turn_type] += 1
                    # Also increment totals
                    totals[origin_direction][turn_type] += 1
    
    # Add totals to the analysis
    analysis["total"] = totals
    
    return analysis

def is_entering_from_outside(line_name, prev_pos, curr_pos, line_coords):
    """
    Determina si un vehículo está entrando desde afuera al cruzar una línea.
    Usa el producto cruzado para determinar de qué lado de la línea viene el vehículo.
    Retorna True si el vehículo viene desde el lado "exterior" de la intersección.
    
    Esta función está optimizada para las coordenadas específicas de este proyecto.
    """
    x1, y1 = line_coords["pt1"]
    x2, y2 = line_coords["pt2"]
    prev_x, prev_y = prev_pos
    curr_x, curr_y = curr_pos
    
    # Calcular el producto cruzado para determinar el lado de la línea
    # Si cross_product > 0: punto está a la izquierda de la línea (mirando de pt1 a pt2)
    # Si cross_product < 0: punto está a la derecha de la línea
    def cross_product(px, py):
        return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    
    prev_cross = cross_product(prev_x, prev_y)
    
    # Definir qué lado es "exterior" para cada línea según la configuración específica
    # Esta lógica está basada en las coordenadas reales de las líneas
    if line_name == "NORTH":
        # Línea norte: exterior está hacia arriba/izquierda
        return prev_cross > 0  # Viene del lado izquierdo de la línea
    elif line_name == "SOUTH": 
        # Línea sur: exterior está hacia abajo/derecha
        return prev_cross < 0  # Viene del lado derecho de la línea
    elif line_name == "EAST":
        # Línea este: exterior está hacia la derecha
        return prev_cross < 0  # Viene del lado derecho de la línea
    elif line_name == "WEST":
        # Línea oeste: exterior está hacia la izquierda
        return prev_cross > 0  # Viene del lado izquierdo de la línea
    
    return False


def process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH="best.pt", video_uuid=None, progress_callback=None, minute_batch_callback=None, generate_video_output=False, output_video_path=None, trim_periods=None):
    """
    Process video for TMC (Turning Movement Count) analysis with optional trimming.

    Args:
        VIDEO_PATH: Path to video file
        LINES_DATA: Line configuration data
        MODEL_PATH: Path to YOLO model
        video_uuid: UUID of the video being processed
        progress_callback: Optional callback for progress updates
        minute_batch_callback: Optional callback for minute-by-minute batch data
        generate_video_output: Whether to generate annotated output video
        output_video_path: Path for output video (if generate_video_output=True)
        trim_periods: Optional list of trim periods in seconds [{"start": 3600, "end": 10800}, ...]

    Returns:
        Dictionary with processing results
    """

    # Validate trim periods if provided
    if trim_periods:
        is_valid, error_msg = validate_trim_periods(trim_periods)
        if not is_valid:
            print(f"⚠️ Invalid trim_periods: {error_msg}")
            print("⚠️ Falling back to processing entire video")
            trim_periods = None

    model = YOLO(MODEL_PATH)

    raw_lines = LINES_DATA

    def ensure_int_coords(point):
        """Convert point coordinates to integers, handling both dict and tuple formats"""
        if isinstance(point, dict):
            return (int(round(point["x"])), int(round(point["y"])))
        elif isinstance(point, (list, tuple)):
            return (int(round(point[0])), int(round(point[1])))
        return point

    LINES = []
    for name, data in raw_lines.items():
        pt1 = ensure_int_coords(data["pt1"])
        pt2 = ensure_int_coords(data["pt2"])
        LINES.append({"name": name.upper(), "pt1": pt1, "pt2": pt2})

    counts = {line["name"]: 0 for line in LINES}
    counted_ids_per_line = {line["name"]: set() for line in LINES}
    entry_counted_ids = set()  # IDs que entraron desde afuera (para conteo total)
    prev_centroids = {}
    crossed_lines_by_id = {}
    turn_types_by_id = {}
    crossing_timestamps = {}
    detected_classes = {}
    class_counts_by_id = {}
    
    # Initialize track interpolator for handling occlusions
    track_interpolator = TrackInterpolator(max_missing_frames=15, min_track_length=3)
    overlap_stats = {"total_overlaps": 0, "frames_with_overlaps": 0}
    
    # Track vehicles that have been processed by minute tracker
    minute_processed_vehicles = set()

    def get_centroid(box):
        x1, y1, x2, y2 = box
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def point_line_distance(px, py, x1, y1, x2, y2):
        # Ensure all coordinates are float for precise calculations
        px, py = float(px), float(py)
        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
        
        A = px - x1
        B = py - y1
        C = x2 - x1
        D = y2 - y1
        dot = A * C + B * D
        len_sq = C * C + D * D
        param = dot / len_sq if len_sq != 0 else -1
        if param < 0:
            xx, yy = x1, y1
        elif param > 1:
            xx, yy = x2, y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D
        dx = px - xx
        dy = py - yy
        return (dx**2 + dy**2) ** 0.5
    

    def classify_turn_from_lines(crossing_data):
        if len(crossing_data) < 2:
            return 'invalid'
        
        # Ordenar por timestamp para obtener la secuencia correcta
        sorted_crossings = sorted(crossing_data, key=lambda x: x[1])  # (direction, timestamp)
        
        # Tomar la primera y última línea cruzada
        from_dir = sorted_crossings[0][0].upper()
        to_dir = sorted_crossings[-1][0].upper()

        if from_dir == to_dir:
            return 'u-turn'

        # Tabla corregida basada en perspectiva del conductor
        # Giro a la derecha = clockwise, Giro a la izquierda = counterclockwise  
        transitions = {
            ('NORTH', 'EAST'): 'right',  # Norte -> Este = giro derecha
            ('NORTH', 'WEST'): 'left',   # Norte -> Oeste = giro izquierda
            ('NORTH', 'SOUTH'): 'straight',
            ('EAST', 'SOUTH'): 'left',   # Este -> Sur = giro izquierda
            ('EAST', 'NORTH'): 'right',  # Este -> Norte = giro derecha
            ('EAST', 'WEST'): 'straight',
            ('SOUTH', 'WEST'): 'left',   # Sur -> Oeste = giro izquierda
            ('SOUTH', 'EAST'): 'right',  # Sur -> Este = giro derecha
            ('SOUTH', 'NORTH'): 'straight',
            ('WEST', 'NORTH'): 'left',   # Oeste -> Norte = giro izquierda
            ('WEST', 'SOUTH'): 'right',  # Oeste -> Sur = giro derecha
            ('WEST', 'EAST'): 'straight',
        }

        return transitions.get((from_dir, to_dir), 'unknown')

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    current_frame = 0
    start_time = time.time()
    last_progress_sent = -1

    # Progress tracking: frames actually processed (not just video position)
    frames_processed_total = 0

    # Calculate frame ranges from trim periods
    frame_ranges = []
    if trim_periods:
        frame_ranges = calculate_frame_ranges_from_seconds(trim_periods, fps, total_frames)
        if frame_ranges:
            print(f"🎬 Trimming enabled: processing {len(frame_ranges)} period(s)")
            total_processing_frames = sum(r['end_frame'] - r['start_frame'] for r in frame_ranges)
            print(f"   Total frames to process: {total_processing_frames} / {total_frames} ({total_processing_frames/total_frames*100:.1f}%)")
        else:
            print("⚠️ No valid frame ranges, processing entire video")
    else:
        print("📊 No trimming specified, processing entire video")

    # Calculate total_processing_frames for normal mode too (for unified progress calculation)
    if not frame_ranges:
        total_processing_frames = total_frames

    # Initialize minute tracker if callback provided
    minute_tracker = None
    if minute_batch_callback:
        # Use provided video_uuid or generate one from filename
        import uuid
        if not video_uuid:
            video_filename = os.path.basename(VIDEO_PATH)
            video_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, video_filename))
            print(f"⚠️ No video_uuid provided, generated: {video_uuid}")
        else:
            print(f"✅ Using provided video_uuid: {video_uuid}")
            
        minute_tracker = MinuteTracker(fps, video_uuid, minute_batch_callback)
        print(f"📊 Enhanced minute tracking enabled for video {video_uuid}")
    
    # Initialize video writer if output video is requested  
    video_writer = None
    if generate_video_output and output_video_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Aggressive compression settings for TMC processor
        # Use H.264 with high compression for minimal file size
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            # Reduce output resolution if too large for better compression
            if width > 1920 or height > 1080:
                scale_factor = min(1920/width, 1080/height)
                width = int(width * scale_factor)
                height = int(height * scale_factor)
                print(f"📉 Scaling output resolution to {width}x{height} for compression")
            
            # Use lower FPS for additional compression if original is high
            output_fps = min(fps, 15)  # Cap at 15 FPS for traffic analysis
            if output_fps != fps:
                print(f"📉 Reducing output FPS from {fps} to {output_fps} for compression")
            
            video_writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (width, height))
            
            if video_writer.isOpened():
                print(f"✅ TMC video writer initialized: H264 codec, {width}x{height}@{output_fps}fps")
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
                    temp_writer = cv2.VideoWriter(output_video_path, fourcc, int(fps), (width, height))
                    if temp_writer.isOpened():
                        video_writer = temp_writer
                        print(f"✅ Fallback to video codec: {codec}")
                        break
                    else:
                        temp_writer.release()
                except Exception as e:
                    print(f"⚠️ Codec {codec} failed: {e}")
                    continue
        
        if not video_writer:
            print("❌ Could not initialize video writer with any codec")
            generate_video_output = False
    
    # Helper function to send seeking progress
    def send_seeking_progress():
        if progress_callback:
            elapsed_time = time.time() - start_time
            # During seeking, we can't accurately estimate time remaining
            # Just show that we're seeking
            progress_callback({
                "progress": int((current_frame / total_frames) * 100),
                "estimatedTimeRemaining": 0,
                "status": "seeking"
            })

    # Helper function to reset tracker state
    def reset_tracker():
        """Reset YOLO tracker to start fresh tracking for new period"""
        if model.predictor is not None:
            model.predictor.trackers = [None]
            print("🔄 YOLO tracker reset - previous tracking state cleared")
        else:
            print("🔄 YOLO tracker not initialized yet (first period)")

    # Helper function for progress calculation
    def calculate_and_send_progress():
        """
        Calculate progress based on actual frames processed (trimming-aware).

        For trimmed videos:
            progress = frames_processed_total / total_processing_frames
        For normal videos:
            progress = current_frame / total_frames (backward compatible)

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
            progress = int((current_frame / total_frames) * 100)

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
        print("🎬 Starting trimmed video processing")

        for period_idx, period in enumerate(frame_ranges):
            start_frame = period["start_frame"]
            end_frame = period["end_frame"]
            period_duration = (period["end_seconds"] - period["start_seconds"]) / 60  # minutes

            print(f"\n📍 Period {period_idx + 1}/{len(frame_ranges)}")
            print(f"   Frames: {start_frame} - {end_frame} ({end_frame - start_frame} frames)")
            print(f"   Time: {period['start_seconds']:.1f}s - {period['end_seconds']:.1f}s ({period_duration:.1f} min)")

            # CRITICAL: Reset tracker at start of each period
            reset_tracker()

            # Clear previous centroids to prevent cross-period tracking
            prev_centroids.clear()
            print("🧹 Previous centroids cleared for new period")

            # Skip frames until we reach the start of this period (frame-skipping)
            while current_frame < start_frame:
                ret, _ = cap.read()  # Read but don't process
                if not ret:
                    print(f"⚠️ Video ended at frame {current_frame} while seeking to {start_frame}")
                    break

                current_frame += 1

                # Progress update every 1000 frames during seeking
                if current_frame % 1000 == 0:
                    send_seeking_progress()
                    print(f"⏩ Seeking: {current_frame}/{start_frame} frames ({current_frame/start_frame*100:.1f}%)")

            if not ret:
                print(f"⚠️ Could not reach period {period_idx + 1}, skipping")
                continue

            print(f"✅ Reached start of period {period_idx + 1} at frame {current_frame}")

            # Process frames in this period
            while current_frame < end_frame:
                ret, frame = cap.read()
                if not ret:
                    print(f"⚠️ Video ended at frame {current_frame} during period {period_idx + 1}")
                    break

                # YOLO processing (existing logic)
                results = model.track(
                    frame, persist=True, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, iou=IOU_THRESHOLD
                )

                # Process detections (rest of existing logic)
                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy()
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy()
                    scores = results[0].boxes.conf.cpu().numpy()

                    # Apply overlap detection improvements
                    processed_boxes, processed_scores, processed_classes, processed_ids = post_process_detections(
                        boxes, scores, classes, ids
                    )

                    # Update overlap statistics
                    if len(processed_boxes) > 1:
                        frame_stats = analyze_overlap_patterns(processed_boxes, processed_ids, {})
                        if frame_stats['overlapping_pairs'] > 0:
                            overlap_stats["frames_with_overlaps"] += 1
                            overlap_stats["total_overlaps"] += frame_stats['overlapping_pairs']

                    # Use processed detections for tracking
                    boxes = processed_boxes
                    ids = processed_ids if processed_ids is not None else ids
                    classes = processed_classes

                    for i, box in enumerate(boxes):
                        obj_id = int(ids[i])
                        class_id = int(classes[i])
                        class_name = model.names[class_id]
                        cx, cy = get_centroid(box)

                        # Store class for this object ID
                        class_counts_by_id[obj_id] = class_name

                        # Update track interpolator
                        track_interpolator.update_track(obj_id, (cx, cy), current_frame)

                        prev_pos = prev_centroids.get(obj_id)
                        if prev_pos:
                            for line in LINES:
                                name = line["name"]
                                x1, y1 = line["pt1"]
                                x2, y2 = line["pt2"]

                                dist = point_line_distance(cx, cy, x1, y1, x2, y2)
                                prev_dist = point_line_distance(
                                    prev_pos[0], prev_pos[1], x1, y1, x2, y2
                                )

                                crossed = dist < DIST_THRESHOLD and prev_dist > DIST_THRESHOLD

                                if crossed and obj_id not in counted_ids_per_line[name]:
                                    counted_ids_per_line[name].add(obj_id)
                                    counts[name] += 1

                                    # Verificar si está entrando desde afuera (para conteo total)
                                    if is_entering_from_outside(name, prev_pos, (cx, cy), line):
                                        entry_counted_ids.add(obj_id)
                                        print(f'[✔] ID {obj_id} ({class_name}) cruzó {name} (ENTRADA desde afuera)')

                                        # Count detected class only for vehicles entering from outside
                                        if obj_id not in detected_classes:
                                            detected_classes[obj_id] = class_name
                                    else:
                                        print(f'[✔] ID {obj_id} ({class_name}) cruzó {name} (interno, no cuenta para total)')

                                    # Registrar el cruce con timestamp
                                    if obj_id not in crossed_lines_by_id:
                                        crossed_lines_by_id[obj_id] = []
                                        crossing_timestamps[obj_id] = []

                                    if name not in [crossing[0] for crossing in crossing_timestamps[obj_id]]:
                                        current_time = time.time()
                                        crossed_lines_by_id[obj_id].append(name)
                                        crossing_timestamps[obj_id].append((name, current_time))

                                    # Detectar giro cuando haya al menos 2 cruces y no se haya clasificado aún
                                    if len(crossing_timestamps[obj_id]) >= 2 and obj_id not in turn_types_by_id:
                                        turn_type = classify_turn_from_lines(crossing_timestamps[obj_id])
                                        if turn_type != 'invalid' and turn_type != 'unknown':
                                            turn_types_by_id[obj_id] = turn_type
                                            from_line = crossing_timestamps[obj_id][0][0]
                                            to_line = crossing_timestamps[obj_id][-1][0]
                                            print(f'↪ ID {obj_id} ({class_name}) hizo un giro {turn_type}: {from_line} -> {to_line}')

                        prev_centroids[obj_id] = (cx, cy)

                # Handle missing detections with track interpolation
                # Clean up old tracks to prevent memory buildup
                if current_frame % 30 == 0:  # Every 30 frames
                    track_interpolator.cleanup_old_tracks(current_frame, max_age=150)

                # Add visualizations if generating output video
                if generate_video_output and video_writer:
                    # Draw detection boxes and tracking
                    if results[0].boxes.id is not None:
                        ids = results[0].boxes.id.cpu().numpy()
                        boxes = results[0].boxes.xyxy.cpu().numpy()

                        for i, box in enumerate(boxes):
                            obj_id = int(ids[i])
                            x1, y1, x2, y2 = box
                            cx, cy = get_centroid(box)

                            # Draw bounding box
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                            # Draw centroid
                            cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                            # Draw ID and turn type if available
                            label = f'ID {obj_id}'
                            if obj_id in turn_types_by_id:
                                label += f' | {turn_types_by_id[obj_id]}'
                            cv2.putText(frame, label, (cx, cy - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                    # Draw lines
                    for line in LINES:
                        name = line["name"]
                        x1, y1 = line["pt1"]
                        x2, y2 = line["pt2"]
                        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                        # Draw line label and count
                        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                        cv2.putText(frame, f'{name}: {counts[name]}', (mid_x, mid_y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Draw summary stats
                    total_current = sum(counts.values())
                    turn_summary = dict(Counter(turn_types_by_id.values()))
                    y_pos = 30
                    cv2.putText(frame, f'Total Crossings: {total_current}', (20, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                    for turn_type, count in turn_summary.items():
                        y_pos += 25
                        cv2.putText(frame, f'{turn_type}: {count}', (20, y_pos),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Resize frame if needed for compression
                    if width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or height != int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                    # Write frame to output video
                    video_writer.write(frame)

                # Update minute tracker with vehicle detections that have complete movement data
                if minute_tracker:
                    # Process vehicles that have completed their movement (have both origin and turn data)
                    for vehicle_id in turn_types_by_id:
                        # Only process if vehicle has crossed lines and we know its movement
                        if vehicle_id in crossing_timestamps and len(crossing_timestamps[vehicle_id]) >= 2:
                            # Only process vehicles that entered from outside (to match final results)
                            if vehicle_id in entry_counted_ids:
                                # Only process each vehicle once for minute tracking
                                if vehicle_id not in minute_processed_vehicles:
                                    minute_processed_vehicles.add(vehicle_id)

                                    # Get vehicle class (use detected_classes for consistency with final results)
                                    vehicle_class = detected_classes.get(vehicle_id, 'unknown')

                                    # Get origin direction (first line crossed)
                                    origin_direction = crossing_timestamps[vehicle_id][0][0].upper()

                                    # Get turn type
                                    turn_type = turn_types_by_id[vehicle_id]

                                    # Process this vehicle detection
                                    minute_tracker.process_vehicle_detection(
                                        current_frame,
                                        vehicle_id,
                                        vehicle_class,
                                        origin_direction,
                                        turn_type
                                    )

                # Progress tracking
                current_frame += 1
                frames_processed_total += 1  # Track actual frames processed
                calculate_and_send_progress()

            print(f"✅ Completed period {period_idx + 1}/{len(frame_ranges)}")

        print("\n✅ All trim periods processed")

    else:
        # NORMAL MODE: Process entire video (existing logic)
        print("📊 Processing entire video (no trimming)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(
                frame, persist=True, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, iou=IOU_THRESHOLD
            )

            if results[0].boxes.id is not None:
                ids = results[0].boxes.id.cpu().numpy()
                boxes = results[0].boxes.xyxy.cpu().numpy()
                classes = results[0].boxes.cls.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()

                # Apply overlap detection improvements
                processed_boxes, processed_scores, processed_classes, processed_ids = post_process_detections(
                    boxes, scores, classes, ids
                )

                # Update overlap statistics
                if len(processed_boxes) > 1:
                    frame_stats = analyze_overlap_patterns(processed_boxes, processed_ids, {})
                    if frame_stats['overlapping_pairs'] > 0:
                        overlap_stats["frames_with_overlaps"] += 1
                        overlap_stats["total_overlaps"] += frame_stats['overlapping_pairs']

                # Use processed detections for tracking
                boxes = processed_boxes
                ids = processed_ids if processed_ids is not None else ids
                classes = processed_classes

                for i, box in enumerate(boxes):
                    obj_id = int(ids[i])
                    class_id = int(classes[i])
                    class_name = model.names[class_id]
                    cx, cy = get_centroid(box)

                    # Store class for this object ID
                    class_counts_by_id[obj_id] = class_name

                    # Update track interpolator
                    track_interpolator.update_track(obj_id, (cx, cy), current_frame)

                    prev_pos = prev_centroids.get(obj_id)
                    if prev_pos:
                        for line in LINES:
                            name = line["name"]
                            x1, y1 = line["pt1"]
                            x2, y2 = line["pt2"]

                            dist = point_line_distance(cx, cy, x1, y1, x2, y2)
                            prev_dist = point_line_distance(
                                prev_pos[0], prev_pos[1], x1, y1, x2, y2
                            )

                            crossed = dist < DIST_THRESHOLD and prev_dist > DIST_THRESHOLD

                            if crossed and obj_id not in counted_ids_per_line[name]:
                                counted_ids_per_line[name].add(obj_id)
                                counts[name] += 1

                                # Verificar si está entrando desde afuera (para conteo total)
                                if is_entering_from_outside(name, prev_pos, (cx, cy), line):
                                    entry_counted_ids.add(obj_id)
                                    print(f'[✔] ID {obj_id} ({class_name}) cruzó {name} (ENTRADA desde afuera)')

                                    # Count detected class only for vehicles entering from outside
                                    if obj_id not in detected_classes:
                                        detected_classes[obj_id] = class_name
                                else:
                                    print(f'[✔] ID {obj_id} ({class_name}) cruzó {name} (interno, no cuenta para total)')

                                # Registrar el cruce con timestamp
                                if obj_id not in crossed_lines_by_id:
                                    crossed_lines_by_id[obj_id] = []
                                    crossing_timestamps[obj_id] = []

                                if name not in [crossing[0] for crossing in crossing_timestamps[obj_id]]:
                                    current_time = time.time()
                                    crossed_lines_by_id[obj_id].append(name)
                                    crossing_timestamps[obj_id].append((name, current_time))

                                # Detectar giro cuando haya al menos 2 cruces y no se haya clasificado aún
                                if len(crossing_timestamps[obj_id]) >= 2 and obj_id not in turn_types_by_id:
                                    turn_type = classify_turn_from_lines(crossing_timestamps[obj_id])
                                    if turn_type != 'invalid' and turn_type != 'unknown':
                                        turn_types_by_id[obj_id] = turn_type
                                        from_line = crossing_timestamps[obj_id][0][0]
                                        to_line = crossing_timestamps[obj_id][-1][0]
                                        print(f'↪ ID {obj_id} ({class_name}) hizo un giro {turn_type}: {from_line} -> {to_line}')

                    prev_centroids[obj_id] = (cx, cy)

            # Handle missing detections with track interpolation
            # Clean up old tracks to prevent memory buildup
            if current_frame % 30 == 0:  # Every 30 frames
                track_interpolator.cleanup_old_tracks(current_frame, max_age=150)

            # Add visualizations if generating output video
            if generate_video_output and video_writer:
                # Draw detection boxes and tracking
                if results[0].boxes.id is not None:
                    ids = results[0].boxes.id.cpu().numpy()
                    boxes = results[0].boxes.xyxy.cpu().numpy()

                    for i, box in enumerate(boxes):
                        obj_id = int(ids[i])
                        x1, y1, x2, y2 = box
                        cx, cy = get_centroid(box)

                        # Draw bounding box
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                        # Draw centroid
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

                        # Draw ID and turn type if available
                        label = f'ID {obj_id}'
                        if obj_id in turn_types_by_id:
                            label += f' | {turn_types_by_id[obj_id]}'
                        cv2.putText(frame, label, (cx, cy - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Draw lines
                for line in LINES:
                    name = line["name"]
                    x1, y1 = line["pt1"]
                    x2, y2 = line["pt2"]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)

                    # Draw line label and count
                    mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.putText(frame, f'{name}: {counts[name]}', (mid_x, mid_y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Draw summary stats
                total_current = sum(counts.values())
                turn_summary = dict(Counter(turn_types_by_id.values()))
                y_pos = 30
                cv2.putText(frame, f'Total Crossings: {total_current}', (20, y_pos),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                for turn_type, count in turn_summary.items():
                    y_pos += 25
                    cv2.putText(frame, f'{turn_type}: {count}', (20, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # Resize frame if needed for compression
                if width != int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or height != int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

                # Write frame to output video
                video_writer.write(frame)

            # Update minute tracker with vehicle detections that have complete movement data
            if minute_tracker:
                # Process vehicles that have completed their movement (have both origin and turn data)
                for vehicle_id in turn_types_by_id:
                    # Only process if vehicle has crossed lines and we know its movement
                    if vehicle_id in crossing_timestamps and len(crossing_timestamps[vehicle_id]) >= 2:
                        # Only process vehicles that entered from outside (to match final results)
                        if vehicle_id in entry_counted_ids:
                            # Only process each vehicle once for minute tracking
                            if vehicle_id not in minute_processed_vehicles:
                                minute_processed_vehicles.add(vehicle_id)

                                # Get vehicle class (use detected_classes for consistency with final results)
                                vehicle_class = detected_classes.get(vehicle_id, 'unknown')

                                # Get origin direction (first line crossed)
                                origin_direction = crossing_timestamps[vehicle_id][0][0].upper()

                                # Get turn type
                                turn_type = turn_types_by_id[vehicle_id]

                                # Process this vehicle detection
                                minute_tracker.process_vehicle_detection(
                                    current_frame,
                                    vehicle_id,
                                    vehicle_class,
                                    origin_direction,
                                    turn_type
                                )

            # Progress tracking
            current_frame += 1
            frames_processed_total += 1  # Track actual frames processed (same as current_frame in normal mode)
            calculate_and_send_progress()

    # Send final 100% progress
    if progress_callback:
        progress_callback({
            "progress": 100,
            "estimatedTimeRemaining": 0
        })
        print(f"✅ TMC Processing complete: {frames_processed_total} frames processed")

    cap.release()
    if video_writer:
        video_writer.release()

    # Post procesamiento con lógica corregida
    # Usar entry_counted_ids para el conteo total (solo vehículos que entraron desde afuera)
    total_count = len(entry_counted_ids)

    # Convert detected_classes from {obj_id: class_name} to {class_name: count}
    class_summary = Counter(detected_classes.values())
    
    # Calcular turns incluyendo straight
    turn_counts = Counter(turn_types_by_id.values())
    turns_dict = dict(turn_counts)
    
    # Ensure all turn categories exist
    if 'left' not in turns_dict:
        turns_dict['left'] = 0
    if 'right' not in turns_dict:
        turns_dict['right'] = 0
    if 'straight' not in turns_dict:
        turns_dict['straight'] = 0
    if 'u-turn' not in turns_dict:
        turns_dict['u-turn'] = 0
    
    # Build new vehicle-class-first structure first
    vehicles = build_analysis_by_vehicle_class(
        detected_classes, turn_types_by_id, crossing_timestamps, crossed_lines_by_id
    )
    
    # Calculate vehicles with complete movement data (from vehicles analysis)
    vehicles_with_movement = 0
    for vehicle_class, origins in vehicles.items():
        if vehicle_class == 'total':
            continue
        for origin_data in origins.values():
            vehicles_with_movement += sum(origin_data.values())
    
    # Si no hay straight explícitos, calcularlos como vehicles_with_movement - left - right - u-turn
    if turns_dict['straight'] == 0:
        left_count = turns_dict.get('left', 0)
        right_count = turns_dict.get('right', 0)
        uturn_count = turns_dict.get('u-turn', 0)
        turns_dict['straight'] = max(0, vehicles_with_movement - left_count - right_count - uturn_count)
    
    # Finalize minute tracking if enabled
    video_duration_seconds = None
    if minute_tracker:
        video_duration_seconds = minute_tracker.finalize_processing()
        print(f"📊 Video duration calculated: {video_duration_seconds} seconds")
    
    return {
        # Original fields (backward compatibility)
        "counts": counts, 
        "turns": turns_dict, 
        "total": total_count,
        "totalcount": total_count,  # Added for clarity
        "detected_classes": dict(class_summary),
        
        # NEW: Analysis grouped by vehicle class first
        "vehicles": vehicles,
        
        "validation": {
            "total_vehicles": total_count,
            "vehicles_with_movement": vehicles_with_movement,
            "total_turns": sum(turns_dict.values()),
            "validation_passed": vehicles_with_movement == sum(turns_dict.values()),
            "entry_vehicles": len(entry_counted_ids),
            "total_crossings": sum(counts.values())
        },
        
        # NEW: Overlap detection statistics
        "overlap_analysis": {
            "frames_with_overlaps": overlap_stats["frames_with_overlaps"],
            "total_overlaps_detected": overlap_stats["total_overlaps"],
            "overlap_frame_ratio": overlap_stats["frames_with_overlaps"] / max(1, current_frame),
            "processing_enhancements": {
                "soft_nms_applied": True,
                "track_interpolation": True,
                "confidence_adjustment": True
            }
        },
        
        # Video metadata
        "video_metadata": {
            "duration_seconds": video_duration_seconds,
            "total_frames": current_frame,
            "fps": fps
        }
    }