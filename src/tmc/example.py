import cv2
import json
from collections import Counter
from ultralytics import YOLO

# === Configuración ===
VIDEO_PATH = 'tmc3.mp4'
LINES_PATH = 'lines.json'
MODEL_PATH = 'best.pt'

CONF_THRESHOLD = 0.01
IMG_SIZE = 640
IOU_THRESHOLD = 0.2
DIST_THRESHOLD = 10  # píxeles de tolerancia para contar cruce

# === Cargar modelo YOLOv8 ===
model = YOLO(MODEL_PATH)

# === Cargar líneas desde JSON ===
with open(LINES_PATH, 'r') as f:
    raw_lines = json.load(f)

LINES = []
for name, data in raw_lines.items():
    pt1 = tuple(data["pt1"])
    pt2 = tuple(data["pt2"])
    LINES.append({"name": name.upper(), "pt1": pt1, "pt2": pt2})

# === Inicialización ===
cap = cv2.VideoCapture(VIDEO_PATH)
counts = {line["name"]: 0 for line in LINES}
counted_ids_per_line = {line["name"]: set() for line in LINES}
prev_centroids = {}
crossed_lines_by_id = {}
turn_types_by_id = {}
crossing_timestamps = {}

# === Funciones ===
def get_centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def point_line_distance(px, py, x1, y1, x2, y2):
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
    return (dx ** 2 + dy ** 2) ** 0.5

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
        ('NORTE', 'ESTE'): 'right',  # Norte -> Este = giro derecha
        ('NORTE', 'OESTE'): 'left',  # Norte -> Oeste = giro izquierda
        ('NORTE', 'SUR'): 'straight',
        ('ESTE', 'SUR'): 'left',     # Este -> Sur = giro izquierda
        ('ESTE', 'NORTE'): 'right',  # Este -> Norte = giro derecha
        ('ESTE', 'OESTE'): 'straight',
        ('SUR', 'OESTE'): 'left',    # Sur -> Oeste = giro izquierda
        ('SUR', 'ESTE'): 'right',    # Sur -> Este = giro derecha
        ('SUR', 'NORTE'): 'straight',
        ('OESTE', 'NORTE'): 'left',  # Oeste -> Norte = giro izquierda
        ('OESTE', 'SUR'): 'right',   # Oeste -> Sur = giro derecha
        ('OESTE', 'ESTE'): 'straight',
    }

    return transitions.get((from_dir, to_dir), 'unknown')

# === Loop principal ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=CONF_THRESHOLD, imgsz=IMG_SIZE, iou=IOU_THRESHOLD)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy()
        boxes = results[0].boxes.xyxy.cpu().numpy()

        for i, box in enumerate(boxes):
            obj_id = int(ids[i])
            cx, cy = get_centroid(box)

            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
            cv2.putText(frame, str(obj_id), (cx, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            prev_pos = prev_centroids.get(obj_id)
            if prev_pos:
                for line in LINES:
                    name = line["name"]
                    x1, y1 = line["pt1"]
                    x2, y2 = line["pt2"]

                    dist = point_line_distance(cx, cy, x1, y1, x2, y2)
                    prev_dist = point_line_distance(prev_pos[0], prev_pos[1], x1, y1, x2, y2)

                    crossed = dist < DIST_THRESHOLD and prev_dist > DIST_THRESHOLD

                    if crossed and obj_id not in counted_ids_per_line[name]:
                        counted_ids_per_line[name].add(obj_id)
                        counts[name] += 1

                        # Registrar el cruce con timestamp
                        if obj_id not in crossed_lines_by_id:
                            crossed_lines_by_id[obj_id] = []
                            crossing_timestamps[obj_id] = []
                        
                        if name not in [crossing[0] for crossing in crossing_timestamps[obj_id]]:
                            import time
                            current_time = time.time()
                            crossed_lines_by_id[obj_id].append(name)
                            crossing_timestamps[obj_id].append((name, current_time))

                        print(f'[✔] ID {obj_id} cruzó {name}')

                        # Detectar giro cuando haya al menos 2 cruces y no se haya clasificado aún
                        if len(crossing_timestamps[obj_id]) >= 2 and obj_id not in turn_types_by_id:
                            turn_type = classify_turn_from_lines(crossing_timestamps[obj_id])
                            if turn_type != 'invalid' and turn_type != 'unknown':
                                turn_types_by_id[obj_id] = turn_type
                                from_line = crossing_timestamps[obj_id][0][0]
                                to_line = crossing_timestamps[obj_id][-1][0]
                                print(f'↪ ID {obj_id} hizo un giro {turn_type}: {from_line} -> {to_line}')

            prev_centroids[obj_id] = (cx, cy)

    # === Dibujar líneas y contadores ===
    for i, line in enumerate(LINES):
        cv2.line(frame, line["pt1"], line["pt2"], (255, 0, 0), 2)
        cv2.putText(frame, f'{line["name"]}: {counts[line["name"]]}',
                    (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # === Calcular total general ===
    all_ids = []
    for ids in counted_ids_per_line.values():
        all_ids.extend(ids)
    id_counts = Counter(all_ids)
    total_count = sum(1 for v in id_counts.values() if v >= 1)

    cv2.putText(frame, f'TOTAL: {total_count}', (10, 30 + len(LINES) * 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # === Mostrar giros contabilizados ===
    turn_summary = Counter(turn_types_by_id.values())
    turn_colors = {'left': (0, 255, 255), 'right': (255, 0, 255), 'straight': (255, 255, 0), 'u-turn': (0, 0, 255)}
    for i, (turn, count) in enumerate(turn_summary.items()):
        color = turn_colors.get(turn, (255, 255, 255))
        cv2.putText(frame, f'{turn.upper()}: {count}', (10, 30 + (len(LINES) + i + 1) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow('Vehicle Counter + Turns', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
