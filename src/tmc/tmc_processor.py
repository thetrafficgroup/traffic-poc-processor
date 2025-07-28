import cv2
import json
import time
from collections import Counter
from ultralytics import YOLO

CONF_THRESHOLD = 0.01
IMG_SIZE = 640
IOU_THRESHOLD = 0.2
DIST_THRESHOLD = 10

# Robust tracking parameters (adaptable for video quality)
MIN_DETECTION_CONFIDENCE = 0.3  # Higher confidence for tracking
MAX_DISAPPEARED_FRAMES = 30     # Max frames before considering object lost
PROXIMITY_THRESHOLD = 50        # Max distance to consider same vehicle
MIN_TRACK_LENGTH = 5           # Minimum frames to consider valid track

def adapt_parameters_for_quality(video_path):
    """Adapt tracking parameters based on video quality indicators"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    
    # Sample a few frames to assess quality
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_frames = min(10, frame_count // 10)
    
    brightness_values = []
    blur_values = []
    
    for i in range(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * (frame_count // sample_frames))
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Measure brightness
            brightness = cv2.mean(gray)[0]
            brightness_values.append(brightness)
            
            # Measure blur (Laplacian variance)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            blur_values.append(blur)
    
    cap.release()
    
    if not brightness_values:
        return {}
    
    avg_brightness = sum(brightness_values) / len(brightness_values)
    avg_blur = sum(blur_values) / len(blur_values)
    
    # Adapt parameters based on quality
    adaptations = {}
    
    # Low light conditions (dark video)
    if avg_brightness < 80:
        adaptations['MIN_DETECTION_CONFIDENCE'] = 0.25  # Lower threshold
        adaptations['PROXIMITY_THRESHOLD'] = 70  # More lenient matching
        print(f"🌙 Low light detected (brightness: {avg_brightness:.1f}) - adapted parameters")
    
    # High blur (poor quality)
    if avg_blur < 100:
        adaptations['MIN_DETECTION_CONFIDENCE'] = 0.25  # Lower threshold
        adaptations['PROXIMITY_THRESHOLD'] = 80  # Much more lenient
        adaptations['MAX_DISAPPEARED_FRAMES'] = 45  # Longer timeout
        print(f"🌫️ High blur detected (blur: {avg_blur:.1f}) - adapted parameters")
    
    # Very clear video
    if avg_blur > 500 and avg_brightness > 120:
        adaptations['MIN_DETECTION_CONFIDENCE'] = 0.4  # Higher threshold
        adaptations['PROXIMITY_THRESHOLD'] = 30  # Stricter matching
        print(f"📹 High quality detected - adapted parameters")
    
    return adaptations


def process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH="best.pt", progress_callback=None, generate_video_output=False, output_video_path=None):
    # Adapt parameters based on video quality
    quality_adaptations = adapt_parameters_for_quality(VIDEO_PATH)
    
    # Apply adaptations to global parameters
    global MIN_DETECTION_CONFIDENCE, MAX_DISAPPEARED_FRAMES, PROXIMITY_THRESHOLD
    
    # Store original values for restoration
    original_min_conf = MIN_DETECTION_CONFIDENCE
    original_max_frames = MAX_DISAPPEARED_FRAMES  
    original_proximity = PROXIMITY_THRESHOLD
    
    # Apply adaptations
    MIN_DETECTION_CONFIDENCE = quality_adaptations.get('MIN_DETECTION_CONFIDENCE', MIN_DETECTION_CONFIDENCE)
    MAX_DISAPPEARED_FRAMES = quality_adaptations.get('MAX_DISAPPEARED_FRAMES', MAX_DISAPPEARED_FRAMES)
    PROXIMITY_THRESHOLD = quality_adaptations.get('PROXIMITY_THRESHOLD', PROXIMITY_THRESHOLD)
    
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
    prev_centroids = {}
    crossed_lines_by_id = {}
    turn_types_by_id = {}
    crossing_timestamps = {}
    detected_classes = {}
    class_counts_by_id = {}
    
    # Robust tracking variables
    unique_vehicles = set()  # Track unique vehicle IDs that have crossed ANY line
    vehicle_first_seen = {}  # Track when vehicle was first detected
    vehicle_last_activity = {}  # Track last activity for timeout detection
    vehicle_positions_history = {}  # Track position history for each ID
    disappeared_vehicles = {}  # Track vehicles that disappeared with their last position
    merged_ids = {}  # Track which IDs have been merged (new_id: original_id)
    vehicle_confidence_history = {}  # Track confidence scores
    frame_count = 0

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
        if len(crossing_data) < 1:
            return 'invalid'
        
        # Si solo cruzó una línea, es un movimiento straight
        if len(crossing_data) == 1:
            return 'straight'
        
        # Ordenar por timestamp para obtener la secuencia correcta
        sorted_crossings = sorted(crossing_data, key=lambda x: x[1])  # (direction, timestamp)
        
        # Tomar la primera y última línea cruzada
        from_dir = sorted_crossings[0][0].upper()
        to_dir = sorted_crossings[-1][0].upper()

        if from_dir == to_dir:
            return 'u-turn'

        # Tabla corregida basada en perspectiva del observador desde el cielo
        transitions = {
            ('NORTH', 'EAST'): 'left',   # North -> East = giro izquierda
            ('NORTH', 'WEST'): 'right',  # North -> West = giro derecha
            ('NORTH', 'SOUTH'): 'straight',
            ('EAST', 'SOUTH'): 'left',   # East -> South = giro izquierda
            ('EAST', 'NORTH'): 'right',  # East -> North = giro derecha
            ('EAST', 'WEST'): 'straight',
            ('SOUTH', 'WEST'): 'left',   # South -> West = giro izquierda
            ('SOUTH', 'EAST'): 'right',  # South -> East = giro derecha
            ('SOUTH', 'NORTH'): 'straight',
            ('WEST', 'NORTH'): 'left',   # West -> North = giro izquierda
            ('WEST', 'SOUTH'): 'right',  # West -> South = giro derecha
            ('WEST', 'EAST'): 'straight',
        }

        return transitions.get((from_dir, to_dir), 'unknown')

    def find_similar_vehicle(new_pos, new_class, disappeared_vehicles, proximity_threshold=PROXIMITY_THRESHOLD):
        """Find if a new detection might be a previously disappeared vehicle"""
        for disappeared_id, (last_pos, last_class, disappeared_frame) in disappeared_vehicles.items():
            if frame_count - disappeared_frame > MAX_DISAPPEARED_FRAMES:
                continue  # Too old, ignore
            
            # Check class similarity and proximity
            if last_class == new_class:
                distance = ((new_pos[0] - last_pos[0])**2 + (new_pos[1] - last_pos[1])**2)**0.5
                if distance <= proximity_threshold:
                    return disappeared_id
        return None

    def clean_old_disappeared_vehicles():
        """Remove vehicles that have been disappeared for too long"""
        to_remove = []
        for vehicle_id, (_, _, disappeared_frame) in disappeared_vehicles.items():
            if frame_count - disappeared_frame > MAX_DISAPPEARED_FRAMES:
                to_remove.append(vehicle_id)
        for vehicle_id in to_remove:
            del disappeared_vehicles[vehicle_id]

    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    current_frame = 0
    start_time = time.time()
    last_progress_sent = -1
    
    # Initialize video writer if output video is requested  
    video_writer = None
    if generate_video_output and output_video_path:
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Try multiple codecs for web compatibility
        codecs_to_try = ['H264', 'X264', 'XVID', 'mp4v']
        video_writer = None
        
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
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        current_time = time.time()
        
        # Clean old disappeared vehicles periodically
        if frame_count % 30 == 0:
            clean_old_disappeared_vehicles()

        results = model.track(
            frame, persist=True, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, iou=IOU_THRESHOLD
        )

        # Track which vehicles are present in current frame
        current_frame_vehicles = set()

        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy()
            boxes = results[0].boxes.xyxy.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                obj_id = int(ids[i])
                class_id = int(classes[i])
                class_name = model.names[class_id]
                confidence = float(confidences[i])
                cx, cy = get_centroid(box)
                
                # Quality filter: skip low confidence detections
                if confidence < MIN_DETECTION_CONFIDENCE:
                    continue
                
                current_frame_vehicles.add(obj_id)
                
                # Check if this might be a reappeared vehicle
                if obj_id not in vehicle_first_seen:
                    similar_id = find_similar_vehicle((cx, cy), class_name, disappeared_vehicles)
                    if similar_id is not None:
                        # Merge with previously disappeared vehicle
                        print(f"🔄 Merging ID {obj_id} with previously disappeared ID {similar_id}")
                        merged_ids[obj_id] = similar_id
                        obj_id = similar_id  # Use original ID
                        
                        # Remove from disappeared list
                        if similar_id in disappeared_vehicles:
                            del disappeared_vehicles[similar_id]
                
                # Store class for this object ID and track first detection
                class_counts_by_id[obj_id] = class_name
                
                # Track first time we see this vehicle
                if obj_id not in vehicle_first_seen:
                    vehicle_first_seen[obj_id] = current_time
                
                # Update last activity time and position history
                vehicle_last_activity[obj_id] = current_time
                
                # Track position history (keep last 10 positions)
                if obj_id not in vehicle_positions_history:
                    vehicle_positions_history[obj_id] = []
                vehicle_positions_history[obj_id].append((cx, cy, frame_count))
                if len(vehicle_positions_history[obj_id]) > 10:
                    vehicle_positions_history[obj_id].pop(0)
                
                # Track confidence history
                if obj_id not in vehicle_confidence_history:
                    vehicle_confidence_history[obj_id] = []
                vehicle_confidence_history[obj_id].append(confidence)
                if len(vehicle_confidence_history[obj_id]) > 5:
                    vehicle_confidence_history[obj_id].pop(0)

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
                            
                            # Add to unique vehicles set (ultra-precise tracking)
                            unique_vehicles.add(obj_id)
                            
                            # Count detected class only ONCE per unique object ID
                            if obj_id not in detected_classes:
                                detected_classes[obj_id] = class_name

                            # Registrar el cruce con timestamp
                            if obj_id not in crossed_lines_by_id:
                                crossed_lines_by_id[obj_id] = []
                                crossing_timestamps[obj_id] = []
                            
                            if name not in [crossing[0] for crossing in crossing_timestamps[obj_id]]:
                                crossed_lines_by_id[obj_id].append(name)
                                crossing_timestamps[obj_id].append((name, current_time))

                            print(f'[✔] ID {obj_id} ({class_name}) cruzó {name}')

                            # Clasificar movimiento inmediatamente después de cada cruce
                            if obj_id not in turn_types_by_id:
                                turn_type = classify_turn_from_lines(crossing_timestamps[obj_id])
                                if turn_type != 'invalid' and turn_type != 'unknown':
                                    turn_types_by_id[obj_id] = turn_type
                                    if len(crossing_timestamps[obj_id]) == 1:
                                        print(f'→ ID {obj_id} ({class_name}) movimiento {turn_type}: {name}')
                                    else:
                                        from_line = crossing_timestamps[obj_id][0][0]
                                        to_line = crossing_timestamps[obj_id][-1][0]
                                        print(f'↪ ID {obj_id} ({class_name}) hizo un giro {turn_type}: {from_line} -> {to_line}')

                prev_centroids[obj_id] = (cx, cy)
        
        # Track vehicles that disappeared this frame
        for prev_id in list(vehicle_last_activity.keys()):
            if prev_id not in current_frame_vehicles and prev_id in prev_centroids:
                # Vehicle disappeared, add to disappeared list
                last_pos = prev_centroids[prev_id]
                last_class = class_counts_by_id.get(prev_id, 'unknown')
                disappeared_vehicles[prev_id] = (last_pos, last_class, frame_count)
                print(f"📤 Vehicle ID {prev_id} ({last_class}) disappeared at {last_pos}")
        
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
            
            # Write frame to output video
            video_writer.write(frame)
        
        # Progress tracking
        current_frame += 1
        if progress_callback and total_frames > 0:
            progress = int((current_frame / total_frames) * 100)
            
            # Send progress every 5%
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

    cap.release()
    if video_writer:
        video_writer.release()

    # Post procesamiento - Robust tracking analysis
    total_unique_vehicles = len(unique_vehicles)
    
    # Filter out vehicles with insufficient tracking quality
    quality_filtered_vehicles = set()
    for vehicle_id in unique_vehicles:
        # Check if vehicle has sufficient confidence history
        avg_confidence = sum(vehicle_confidence_history.get(vehicle_id, [0])) / len(vehicle_confidence_history.get(vehicle_id, [1]))
        track_length = len(vehicle_positions_history.get(vehicle_id, []))
        
        if avg_confidence >= MIN_DETECTION_CONFIDENCE and track_length >= MIN_TRACK_LENGTH:
            quality_filtered_vehicles.add(vehicle_id)
        else:
            print(f"🚫 Filtered out vehicle ID {vehicle_id} - avg_conf: {avg_confidence:.2f}, track_length: {track_length}")
    
    # Use quality filtered count for final results
    final_vehicle_count = len(quality_filtered_vehicles)
    
    # Convert detected_classes only for quality vehicles
    quality_classes = {vid: detected_classes[vid] for vid in detected_classes if vid in quality_filtered_vehicles}
    class_summary = Counter(quality_classes.values())
    
    # Calculate turns only for quality vehicles
    quality_turns = {vid: turn_types_by_id[vid] for vid in turn_types_by_id if vid in quality_filtered_vehicles}
    turn_counts = Counter(quality_turns.values())
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
    
    # Validation: sum of turns should equal final vehicle count
    total_turns = sum(turns_dict.values())
    
    # If there's a mismatch, classify unclassified vehicles as straight
    unclassified_count = final_vehicle_count - total_turns
    if unclassified_count > 0:
        turns_dict['straight'] += unclassified_count
        print(f"⚠️ Added {unclassified_count} unclassified vehicles as 'straight'")
    
    result = {
        "counts": counts, 
        "turns": turns_dict, 
        "total": final_vehicle_count,
        "totalcount": final_vehicle_count,
        "detected_classes": dict(class_summary),
        "tracking_stats": {
            "raw_detections": total_unique_vehicles,
            "quality_filtered": final_vehicle_count,
            "filtered_out": total_unique_vehicles - final_vehicle_count,
            "merged_vehicles": len(merged_ids)
        },
        "validation": {
            "total_vehicles": final_vehicle_count,
            "total_turns": sum(turns_dict.values()),
            "validation_passed": final_vehicle_count == sum(turns_dict.values())
        }
    }
    
    # Restore original parameters
    MIN_DETECTION_CONFIDENCE = original_min_conf
    MAX_DISAPPEARED_FRAMES = original_max_frames
    PROXIMITY_THRESHOLD = original_proximity
    
    return result