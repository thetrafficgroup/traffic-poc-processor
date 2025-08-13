import cv2
import math
import numpy as np
import json
from ultralytics import YOLO

# Modelo
model = YOLO("best.pt")
cap = cv2.VideoCapture("sample3.mp4")

# Tracker
next_id = 0
tracked_objects = []
MAX_DIST = 100
MAX_FRAMES_MISSING = 5

# Cargar zona y líneas desde JSON
with open("zone.json") as f:
    zone_data = json.load(f)

zone_polygon = zone_data["zone"]
line_a = zone_data["line_a"]
line_b = zone_data["line_b"]

# Extraer X mínimos y máximos de las líneas
line_x1 = int((line_a[0][0] + line_a[1][0]) / 2)
line_x2 = int((line_b[0][0] + line_b[1][0]) / 2)

# Conteo
conteo = 0
conteo_por_clase = {}
ids_contados = set()

def point_in_polygon(x, y, polygon):
    return cv2.pointPolygonTest(np.array(polygon, dtype=np.int32), (x, y), False) >= 0

def euclidean(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while cap.isOpened():
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []

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

    # Matching detecciones con objetos ya seguidos
    for det in detections:
        cx, cy = det["cx"], det["cy"]
        best_match = None
        min_dist = MAX_DIST

        for obj in tracked_objects:
            dist = euclidean((cx, cy), (obj["cx"], obj["cy"]))
            if dist < min_dist and "matched" not in obj:
                min_dist = dist
                best_match = obj

        if best_match:
            best_match["last_cx"] = best_match["cx"]
            best_match["cx"] = cx
            best_match["cy"] = cy
            best_match["frames_missing"] = 0
            best_match["matched"] = True
            best_match["label"] = det["label"]
        else:
            tracked_objects.append({
                "id": next_id,
                "cx": cx,
                "cy": cy,
                "last_cx": cx,
                "frames_missing": 0,
                "counted": False,
                "label": det["label"]
            })
            next_id += 1

    # Limpieza de objetos perdidos
    for obj in tracked_objects:
        if "matched" not in obj:
            obj["frames_missing"] += 1
        else:
            del obj["matched"]
    tracked_objects = [obj for obj in tracked_objects if obj["frames_missing"] <= MAX_FRAMES_MISSING]

    # Conteo si cruza zona + banda
    for obj in tracked_objects:
        if not obj["counted"]:
            if (line_x1 <= obj["cx"] <= line_x2 and point_in_polygon(obj["cx"], obj["cy"], zone_polygon)):
                conteo += 1
                obj["counted"] = True
                ids_contados.add((obj["id"], obj["label"]))
                label = obj["label"]
                conteo_por_clase[label] = conteo_por_clase.get(label, 0) + 1

    # Dibujo
    for obj in tracked_objects:
        cv2.circle(frame, (obj["cx"], obj["cy"]), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"{obj['label']} ID:{obj['id']}", (obj["cx"] + 5, obj["cy"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Banda vertical (desde líneas del JSON)
    cv2.line(frame, tuple(line_a[0]), tuple(line_a[1]), (0, 255, 255), 2)
    cv2.line(frame, tuple(line_b[0]), tuple(line_b[1]), (0, 255, 255), 2)

    # Zona
    cv2.polylines(frame, [np.array(zone_polygon, dtype=np.int32)], isClosed=True, color=(255, 0, 0), thickness=2)

    # Conteo
    cv2.putText(frame, f"Total: {conteo}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    y_offset = 60
    for cls, count in conteo_por_clase.items():
        cv2.putText(frame, f"{cls}: {count}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
        y_offset += 30

    cv2.imshow("Tracking con zona y banda", frame)

cap.release()
cv2.destroyAllWindows()

# Guardar JSON final
output = {
    "total": conteo,
    "por_clase": conteo_por_clase
}
with open("results.json", "w") as f:
    json.dump(output, f, indent=4)

print("✅ Resultados guardados en results.json")
