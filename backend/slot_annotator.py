"""
Slot Annotator Tool
Run ONCE per camera to define parking slot polygons on a real frame.

Usage:
  python slot_annotator.py --video parking.mp4 --output slots_config.json

Controls:
  Left click    Add a corner point to the current slot polygon
  Right click   Finish the current slot (needs >= 3 points)
  T             Cycle slot type: standard → handicap → ev
  U             Undo last point
  R             Restart current polygon
  D             Delete last completed slot
  S             Save config and exit
  Q             Quit without saving
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path


class SlotAnnotator:
    SLOT_TYPES = ["standard", "handicap", "ev"]
    TYPE_COLORS = {"standard": (0, 200, 0), "handicap": (255, 140, 0), "ev": (0, 140, 255)}

    def __init__(self, video_path: str, output_path: str, lot_name: str = "Parking Lot"):
        self.output_path = output_path
        self.lot_name = lot_name
        self.current_polygon = []
        self.completed_slots = []
        self.slot_counter = 1
        self.current_type = "standard"

        # Read first usable frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise ValueError(f"Cannot read from: {video_path}")

        self.orig_h, self.orig_w = frame.shape[:2]
        self.display_scale = 1.0
        if self.orig_w > 1280:
            self.display_scale = 1280 / self.orig_w
            frame = cv2.resize(frame, (1280, int(self.orig_h * self.display_scale)))

        self.base_frame = frame
        self.display_frame = frame.copy()
        print(f"[Annotator] Original: {self.orig_w}x{self.orig_h}, Display: {frame.shape[1]}x{frame.shape[0]}")
        print("[Annotator] LClick=point | RClick=finish slot | T=type | S=save | Q=quit")

    # ── Helpers ────────────────────────────────────────────────────────────

    def _next_slot_id(self) -> str:
        row = chr(ord('A') + (self.slot_counter - 1) // 10)
        col = ((self.slot_counter - 1) % 10) + 1
        return f"{row}{col}"

    def _redraw(self, mouse_pos=None):
        frame = self.base_frame.copy()

        # Completed slots
        for slot in self.completed_slots:
            pts = np.array(slot["polygon"], dtype=np.int32)
            color = self.TYPE_COLORS[slot["type"]]
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.polylines(frame, [pts], True, color, 2)
            cx = int(np.mean([p[0] for p in slot["polygon"]]))
            cy = int(np.mean([p[1] for p in slot["polygon"]]))
            cv2.putText(frame, slot["id"], (cx - 12, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Current in-progress polygon
        for i, pt in enumerate(self.current_polygon):
            cv2.circle(frame, tuple(pt), 5, (0, 255, 255), -1)
            if i > 0:
                cv2.line(frame, tuple(self.current_polygon[i - 1]), tuple(pt), (0, 200, 255), 2)

        # Preview closing line to mouse
        if mouse_pos and self.current_polygon:
            cv2.line(frame, tuple(self.current_polygon[-1]), mouse_pos, (100, 200, 255), 1)
            if len(self.current_polygon) >= 2:
                cv2.line(frame, tuple(self.current_polygon[0]), mouse_pos, (80, 180, 255), 1)

        # HUD bottom bar
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 90), (w, h), (0, 0, 0), -1)
        lines = [
            f"Slots defined: {len(self.completed_slots)}   |   Current type: {self.current_type}   |   Points: {len(self.current_polygon)}",
            "LClick=add point  |  RClick=finish slot  |  T=type  |  U=undo  |  R=reset  |  D=del last  |  S=save  |  Q=quit",
        ]
        for i, txt in enumerate(lines):
            cv2.putText(frame, txt, (10, h - 60 + i * 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 220, 255), 1)

        self.display_frame = frame

    # ── Mouse callback ──────────────────────────────────────────────────────

    def _mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_polygon.append([x, y])
            self._redraw()

        elif event == cv2.EVENT_RBUTTONDOWN:
            if len(self.current_polygon) >= 3:
                sid = self._next_slot_id()
                self.completed_slots.append({
                    "id": sid,
                    "type": self.current_type,
                    "polygon": self.current_polygon.copy()
                })
                print(f"[Annotator] Saved slot {sid} ({self.current_type}), {len(self.current_polygon)} pts")
                self.slot_counter += 1
                self.current_polygon = []
            else:
                print("[Annotator] Need at least 3 points.")
            self._redraw()

        elif event == cv2.EVENT_MOUSEMOVE:
            self._redraw(mouse_pos=(x, y))

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self):
        cv2.namedWindow("Slot Annotator", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Slot Annotator", self._mouse)
        self._redraw()

        while True:
            cv2.imshow("Slot Annotator", self.display_frame)
            key = cv2.waitKey(16) & 0xFF

            if key == ord('s'):
                self._save()
                break
            elif key == ord('q'):
                print("[Annotator] Quit without saving.")
                break
            elif key == ord('u') and self.current_polygon:
                self.current_polygon.pop()
                self._redraw()
            elif key == ord('r'):
                self.current_polygon = []
                self._redraw()
            elif key == ord('d') and self.completed_slots:
                removed = self.completed_slots.pop()
                self.slot_counter -= 1
                print(f"[Annotator] Deleted slot {removed['id']}")
                self._redraw()
            elif key == ord('t'):
                idx = self.SLOT_TYPES.index(self.current_type)
                self.current_type = self.SLOT_TYPES[(idx + 1) % len(self.SLOT_TYPES)]
                print(f"[Annotator] Type → {self.current_type}")
                self._redraw()

        cv2.destroyAllWindows()

    def _save(self):
        inv = 1.0 / self.display_scale
        slots_out = []
        for slot in self.completed_slots:
            scaled_poly = [[round(x * inv), round(y * inv)] for x, y in slot["polygon"]]
            slots_out.append({**slot, "polygon": scaled_poly})

        config = {
            "lot_name": self.lot_name,
            "camera_id": "cam_01",
            "frame_size": [self.orig_w, self.orig_h],
            "slots": slots_out
        }
        with open(self.output_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[Annotator] Saved {len(self.completed_slots)} slots at {self.orig_w}x{self.orig_h} → {self.output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",    required=True,                  help="Input video path")
    ap.add_argument("--output",   default="slots_config.json",    help="Output JSON path")
    ap.add_argument("--lot-name", default="Parking Lot",          help="Lot display name")
    args = ap.parse_args()

    SlotAnnotator(args.video, args.output, args.lot_name).run()


if __name__ == "__main__":
    main()
