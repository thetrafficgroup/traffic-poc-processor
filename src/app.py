import cv2
import json
from collections import Counter
from ultralytics import YOLO

CONF_THRESHOLD = 0.01
IMG_SIZE = 640
IOU_THRESHOLD = 0.2
DIST_THRESHOLD = 10


def process_video(VIDEO_PATH, LINES_DATA, MODEL_PATH="best.pt"):
    model = YOLO(MODEL_PATH)

    raw_lines = LINES_DATA

    LINES = []
    for name, data in raw_lines.items():
        pt1 = tuple(data["pt1"])
        pt2 = tuple(data["pt2"])
        LINES.append({"name": name.upper(), "pt1": pt1, "pt2": pt2})

    counts = {line["name"]: 0 for line in LINES}
    counted_ids_per_line = {line["name"]: set() for line in LINES}
    prev_centroids = {}
    crossed_lines_by_id = {}
    turn_types_by_id = {}

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
        return (dx**2 + dy**2) ** 0.5

    def classify_turn_from_lines(directions):
        if len(directions) != 2:
            return "invalid"
        from_dir, to_dir = directions[0].upper(), directions[1].upper()
        if from_dir == to_dir:
            return "u-turn"
        transitions = {
            ("NORTE", "ESTE"): "left",
            ("NORTE", "OESTE"): "right",
            ("NORTE", "SUR"): "straight",
            ("ESTE", "SUR"): "left",
            ("ESTE", "NORTE"): "right",
            ("ESTE", "OESTE"): "straight",
            ("SUR", "OESTE"): "left",
            ("SUR", "ESTE"): "right",
            ("SUR", "NORTE"): "straight",
            ("OESTE", "NORTE"): "left",
            ("OESTE", "SUR"): "right",
            ("OESTE", "ESTE"): "straight",
        }
        return transitions.get((from_dir, to_dir), "unknown")

    cap = cv2.VideoCapture(VIDEO_PATH)
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

            for i, box in enumerate(boxes):
                obj_id = int(ids[i])
                cx, cy = get_centroid(box)

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

                            if obj_id not in crossed_lines_by_id:
                                crossed_lines_by_id[obj_id] = []
                            if name not in crossed_lines_by_id[obj_id]:
                                crossed_lines_by_id[obj_id].append(name)

                            if (
                                len(crossed_lines_by_id[obj_id]) == 2
                                and obj_id not in turn_types_by_id
                            ):
                                turn_type = classify_turn_from_lines(
                                    crossed_lines_by_id[obj_id]
                                )
                                turn_types_by_id[obj_id] = turn_type

                prev_centroids[obj_id] = (cx, cy)

    cap.release()

    # Post procesamiento
    all_ids = []
    for ids in counted_ids_per_line.values():
        all_ids.extend(ids)
    id_counts = Counter(all_ids)
    total_count = sum(1 for v in id_counts.values() if v >= 1)

    return {"counts": counts, "turns": Counter(turn_types_by_id.values()), "total": total_count}
