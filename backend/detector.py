"""
Parking Space Detector
Uses YOLOv8 for vehicle detection + polygon-based slot matching
Falls back to OpenCV background subtraction if YOLO not available
"""

import cv2
import numpy as np
import json
import os
from collections import deque

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def load_yolo_model(model_path="yolov8n.pt"):
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        print(f"[YOLO] Loaded: {model_path}")
        return ("ultralytics", model)
    except Exception as e:
        print(f"[INFO] Using OpenCV detection fallback ({e})")
        return ("opencv", None)


def detect_vehicles_yolo(frame, model):
    # conf=0.2 catches aerial/top-down views where YOLO has lower confidence
    results = model(frame, verbose=False, conf=0.2)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            detections.append((x1, y1, x2, y2, conf, VEHICLE_CLASSES[cls_id]))
    return detections


def detect_vehicles_opencv(frame, bg_subtractor):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (21, 21), 0)
    fg_mask = bg_subtractor.apply(blur)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.dilate(fg_mask, kernel, iterations=3)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 1500:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        if 0.4 < aspect < 4.5:
            detections.append((x, y, x + w, y + h, 0.7, "vehicle"))
    return detections


def point_in_polygon(point, polygon):
    poly = np.array(polygon, dtype=np.float32)
    result = cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False)
    return result >= 0


def slot_box_overlap_ratio(box, polygon):
    """Fraction of the slot polygon area covered by the detection bounding box.
    Works correctly for aerial/top-down views and diagonal slot polygons."""
    x1, y1, x2, y2 = box
    pts = np.array(polygon, dtype=np.int32)

    # Fast bbox rejection before creating masks
    px_min, py_min = int(pts[:, 0].min()), int(pts[:, 1].min())
    px_max, py_max = int(pts[:, 0].max()), int(pts[:, 1].max())
    if x2 < px_min or x1 > px_max or y2 < py_min or y1 > py_max:
        return 0.0

    # Local coordinate system covering both box and polygon
    rx, ry = min(x1, px_min), min(y1, py_min)
    rw = max(x2, px_max) - rx + 1
    rh = max(y2, py_max) - ry + 1

    slot_mask = np.zeros((rh, rw), dtype=np.uint8)
    box_mask  = np.zeros((rh, rw), dtype=np.uint8)

    cv2.fillPoly(slot_mask, [pts - np.array([rx, ry])], 1)
    cv2.rectangle(box_mask, (x1 - rx, y1 - ry), (x2 - rx, y2 - ry), 1, -1)

    slot_area = int(slot_mask.sum())
    if slot_area == 0:
        return 0.0
    return float(int((slot_mask & box_mask).sum())) / slot_area


def check_slot_occupied(box, polygon, threshold=0.15):
    """Slot is occupied if a detection box covers ≥15% of its area."""
    return slot_box_overlap_ratio(box, polygon) >= threshold


class StateManager:
    def __init__(self, slot_ids, buffer_size=5):
        self.buffers = {sid: deque([False] * buffer_size, maxlen=buffer_size) for sid in slot_ids}
        self.current = {sid: False for sid in slot_ids}

    def update(self, slot_id, is_occupied):
        self.buffers[slot_id].append(is_occupied)
        votes = sum(self.buffers[slot_id])
        self.current[slot_id] = votes > len(self.buffers[slot_id]) // 2

    def get(self, slot_id):
        return self.current.get(slot_id, False)


class ParkingDetector:
    def __init__(self, slots_config_path=None, model_path="yolov8n.pt"):
        self.slots = []
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50)
        self.state_manager = None
        self.frame_count = 0
        self.config_frame_size = None
        self.model_type, self.model = load_yolo_model(model_path)

        if slots_config_path and os.path.exists(slots_config_path):
            self.load_slots(slots_config_path)

    def load_slots(self, config_path):
        with open(config_path) as f:
            config = json.load(f)
        self.slots = config.get("slots", [])
        self.config_frame_size = config.get("frame_size")  # [w, h] saved at annotation time
        self.state_manager = StateManager([s["id"] for s in self.slots])
        print(f"[CONFIG] Loaded {len(self.slots)} parking slots")

    def auto_detect_slots(self, frame, rows=2, cols=5):
        h, w = frame.shape[:2]
        mx, my = int(w * 0.05), int(h * 0.1)
        sw = (w - 2 * mx) // cols
        sh = (h - 2 * my) // rows
        self.slots = []
        for r in range(rows):
            for c in range(cols):
                x1 = mx + c * sw + 3
                y1 = my + r * sh + 3
                x2 = x1 + sw - 6
                y2 = y1 + sh - 6
                self.slots.append({
                    "id": f"{chr(65+r)}{c+1}",
                    "polygon": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
                    "type": "standard"
                })
        self.state_manager = StateManager([s["id"] for s in self.slots])
        print(f"[AUTO] Generated {len(self.slots)} slots ({rows}x{cols})")

    def process_frame(self, frame):
        if not self.slots:
            self.auto_detect_slots(frame)
        elif self.config_frame_size:
            # Scale polygon coords from annotation resolution to actual frame resolution
            cfg_w, cfg_h = self.config_frame_size
            actual_h, actual_w = frame.shape[:2]
            if cfg_w != actual_w or cfg_h != actual_h:
                sx = actual_w / cfg_w
                sy = actual_h / cfg_h
                for slot in self.slots:
                    slot["polygon"] = [[round(x * sx), round(y * sy)] for x, y in slot["polygon"]]
                print(f"[CONFIG] Scaled slots {cfg_w}x{cfg_h} → {actual_w}x{actual_h}")
            self.config_frame_size = None  # only scale once
        self.frame_count += 1

        # Detect vehicles
        if self.model_type == "ultralytics":
            detections = detect_vehicles_yolo(frame, self.model)
        else:
            detections = detect_vehicles_opencv(frame, self.bg_subtractor)

        # Match to slots
        for slot in self.slots:
            occupied = any(
                check_slot_occupied((x1,y1,x2,y2), slot["polygon"])
                for (x1,y1,x2,y2,_,__) in detections
            )
            self.state_manager.update(slot["id"], occupied)

        statuses = {s["id"]: self.state_manager.get(s["id"]) for s in self.slots}
        annotated = self._draw(frame.copy(), detections, statuses)
        return annotated, statuses, detections

    def _draw(self, frame, detections, statuses):
        overlay = frame.copy()
        for slot in self.slots:
            sid = slot["id"]
            poly = np.array(slot["polygon"], dtype=np.int32)
            occupied = statuses.get(sid, False)
            fill_color = (0, 60, 220) if occupied else (0, 200, 80)
            cv2.fillPoly(overlay, [poly], fill_color)

        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        for slot in self.slots:
            sid = slot["id"]
            poly = np.array(slot["polygon"], dtype=np.int32)
            occupied = statuses.get(sid, False)
            border = (0, 30, 180) if occupied else (0, 160, 50)
            cv2.polylines(frame, [poly], True, border, 2)
            cx = int(np.mean([p[0] for p in slot["polygon"]]))
            cy = int(np.mean([p[1] for p in slot["polygon"]]))
            cv2.putText(frame, sid, (cx-14, cy-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            mark = "OCC" if occupied else "FREE"
            mk_color = (100, 120, 255) if occupied else (80, 255, 140)
            cv2.putText(frame, mark, (cx-16, cy+14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, mk_color, 1)

        for (x1,y1,x2,y2,conf,cls) in detections:
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,255), 2)

        # HUD bar
        total = len(self.slots)
        occupied_n = sum(1 for v in statuses.values() if v)
        free_n = total - occupied_n
        hud = np.zeros((52, frame.shape[1], 3), dtype=np.uint8)
        hud[:] = (18, 18, 28)
        mode = "YOLO" if self.model_type == "ultralytics" else "OpenCV"
        cv2.putText(hud, f"PARKING MONITOR  [{mode}]  Frame {self.frame_count}",
                    (10,16), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (130,130,160), 1)
        cv2.putText(hud, f"TOTAL: {total}", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 2)
        cv2.putText(hud, f"FREE: {free_n}", (140,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,230,110), 2)
        cv2.putText(hud, f"OCCUPIED: {occupied_n}", (290,40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,100,255), 2)
        if total > 0:
            bx, bw = 470, frame.shape[1] - 480
            pct = occupied_n / total
            cv2.rectangle(hud, (bx,30),(bx+bw,44),(50,50,65),-1)
            cv2.rectangle(hud, (bx,30),(bx+int(bw*pct),44),(80,100,255),-1)
            cv2.putText(hud, f"{pct:.0%}", (bx+bw+4,43), cv2.FONT_HERSHEY_SIMPLEX, 0.45,(180,180,200),1)

        return np.vstack([hud, frame])

    def get_summary(self, statuses):
        total = len(self.slots)
        occ = sum(1 for v in statuses.values() if v)
        return {
            "total": total,
            "free": total - occ,
            "occupied": occ,
            "occupancy_pct": round(occ/total*100 if total else 0, 1),
            "detection_mode": self.model_type,
            "slots": [
                {"id": s["id"], "type": s.get("type","standard"),
                 "status": "occupied" if statuses.get(s["id"],False) else "free"}
                for s in self.slots
            ]
        }
