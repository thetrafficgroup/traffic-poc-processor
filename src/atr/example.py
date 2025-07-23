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

# === Config ===
VIDEO_PATH = 'sample.mp4'
LANES_FILE = 'lanes.json'
CONF_THRESHOLD = 0.1

# === Modelo YOLOv8 ===
model = YOLO('best.pt')

# === Cargar carriles ===
with open(LANES_FILE, 'r') as f:
    config = json.load(f)

# Adaptar lanes y finish_line a formato de listas de tuplas (x, y)
def dict_points_to_tuples(points):
    return [(pt["x"], pt["y"]) if isinstance(pt, dict) else tuple(pt) for pt in points]

lanes = config["lanes"]
for lane in lanes:
    lane["points"] = dict_points_to_tuples(lane["points"])
lane_polygons = [(lane["id"], Polygon(lane["points"])) for lane in lanes]
lane_counts = {lane_id: 0 for lane_id, _ in lane_polygons}

# === Línea de meta inclinada ===
finish_line = config.get("finish_line")
if finish_line and isinstance(finish_line[0], dict):
    finish_line = dict_points_to_tuples(finish_line)

def point_side_of_line(p, a, b):
    # Devuelve >0 si p está a la izquierda de ab, <0 derecha, 0 sobre la línea
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])

# === Inicializar variables
counted_ids = set()
previous_positions = {}
tracker = CentroidTracker(max_disappeared=15)

# === Inicializar video ===
cap = cv2.VideoCapture(VIDEO_PATH)

def get_centroid(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

# === Bucle de procesamiento ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=CONF_THRESHOLD)
    boxes = results[0].boxes

    input_centroids = []
    detections_map = {}

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx, cy = get_centroid((x1, y1, x2, y2))
        input_centroids.append(np.array([cx, cy]))
        detections_map[(cx, cy)] = (x1, y1, x2, y2)

    objects = tracker.update(np.array(input_centroids))

    for objectID, centroid in objects.items():
        cx, cy = centroid
        pt = Point(cx, cy)
        lane_id = None

        for lid, polygon in lane_polygons:
            if polygon.buffer(8).contains(pt):
                lane_id = lid
                break

        # Inicializar historial
        if objectID not in previous_positions:
            previous_positions[objectID] = []

        previous_positions[objectID].append((cx, cy))

        # Verificar cruce de línea de meta inclinada
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
            if side_prev * side_curr < 0:  # Cambió de lado
                counted_ids.add(objectID)
                lane_counts[lane_id] += 1
                print(f"[CONTADO] Vehículo ID={objectID} | Carril={lane_id} | Total carril: {lane_counts[lane_id]}")

        # Visualización
        if (cx, cy) in detections_map:
            x1, y1, x2, y2 = detections_map[(cx, cy)]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        color = (0, 255, 0) if lane_id is not None else (0, 0, 255)
        cv2.circle(frame, (cx, cy), 5, color, -1)
        cv2.putText(frame, f'ID {objectID} | L{lane_id}', (cx, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Dibujar línea de meta inclinada
    if finish_line is not None and len(finish_line) == 2:
        pt1 = tuple(map(int, finish_line[0]))
        pt2 = tuple(map(int, finish_line[1]))
        cv2.line(frame, pt1, pt2, (255, 0, 255), 3)

    # Dibujar carriles y conteo
    for lane in lanes:
        pts = np.array(lane["points"], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(frame, f'L{lane["id"]}: {lane_counts[lane["id"]]}',
                    (pts[0][0][0], pts[0][0][1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # Dibujar conteo total
    total_count = sum(lane_counts.values())
    cv2.putText(frame, f'Total: {total_count}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)

    cv2.imshow('Vehicle Count with Meta Line', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
