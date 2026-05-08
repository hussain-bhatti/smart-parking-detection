"""
test_detector.py
Quick test to verify YOLO + slot matching works on a single frame.
Run via VS Code: F5 → "Debug Detector (single frame test)"
Or terminal:    python backend/test_detector.py
"""

import cv2
import json
import numpy as np
import sys
import os
from pathlib import Path

# ── Allow running from project root OR backend/ ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

SAMPLE_VIDEO = Path(__file__).parent.parent / "sample_videos" / "parking.mp4"
SLOTS_CONFIG = Path(__file__).parent / "slots_config.json"
OUTPUT_FRAME = Path(__file__).parent.parent / "output_frames" / "test_frame.jpg"


def create_demo_config(frame_shape, output_path):
    """Auto-generate a 3x4 demo slot grid if no config exists."""
    h, w = frame_shape[:2]
    margin_x, margin_y = int(w * 0.05), int(h * 0.1)
    rows, cols, gap = 3, 4, 8
    slot_w = (w - 2 * margin_x) // cols
    slot_h = (h - 2 * margin_y) // rows
    slots = []
    for row in range(rows):
        for col in range(cols):
            x1 = margin_x + col * slot_w + gap
            y1 = margin_y + row * slot_h + gap
            x2 = x1 + slot_w - gap * 2
            y2 = y1 + slot_h - gap * 2
            slots.append({
                "id": f"{chr(ord('A') + row)}{col + 1}",
                "type": "standard",
                "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            })
    config = {
        "lot_name": "Test Lot",
        "camera_id": "cam_01",
        "frame_size": [w, h],
        "slots": slots
    }
    with open(output_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  ✅ Demo slots config created: {output_path} ({len(slots)} slots)")
    return config


def get_test_frame():
    """Get a test frame — from sample video or generate a blank one."""
    if SAMPLE_VIDEO.exists():
        cap = cv2.VideoCapture(str(SAMPLE_VIDEO))
        ret, frame = cap.read()
        cap.release()
        if ret:
            print(f"  ✅ Frame from video: {SAMPLE_VIDEO.name}  ({frame.shape[1]}x{frame.shape[0]})")
            return frame

    # No video — make a synthetic parking-lot-looking frame
    print("  ℹ️  No sample video found — generating synthetic test frame")
    print(f"     (Put a real video at: {SAMPLE_VIDEO})")
    frame = np.ones((720, 1280, 3), dtype=np.uint8) * 50   # dark asphalt
    # Draw lane markings
    for i in range(0, 1280, 120):
        cv2.line(frame, (i, 0), (i, 720), (80, 80, 80), 2)
    for i in range(0, 720, 160):
        cv2.line(frame, (0, i), (1280, i), (80, 80, 80), 1)
    # Draw some fake cars
    car_positions = [(100, 150), (340, 150), (580, 150), (100, 450), (580, 450)]
    for cx, cy in car_positions:
        cv2.rectangle(frame, (cx, cy), (cx + 90, cy + 55), (30, 80, 150), -1)   # body
        cv2.rectangle(frame, (cx + 10, cy + 8), (cx + 80, cy + 30), (80, 160, 220), -1)  # windows
    return frame


def main():
    print("\n" + "="*55)
    print("  ParkVision — Detector Test")
    print("="*55)

    # 1. Get frame
    print("\n[1] Loading test frame...")
    frame = get_test_frame()

    # 2. Slots config
    print("\n[2] Loading slots config...")
    if not SLOTS_CONFIG.exists():
        print(f"  ℹ️  No slots_config.json found at {SLOTS_CONFIG}")
        print("     Auto-generating demo grid...")
        create_demo_config(frame.shape, SLOTS_CONFIG)

    # 3. Run detector
    print("\n[3] Running detector...")
    try:
        from detector import ParkingDetector
        detector = ParkingDetector(str(SLOTS_CONFIG), model_size="n")
        result = detector.process_frame(frame, frame_number=0)
        annotated = detector.annotate_frame(frame, result)
        print(f"\n  📊 Results:")
        print(f"     Total slots  : {result.total_slots}")
        print(f"     Free         : {result.free_slots}  🟢")
        print(f"     Occupied     : {result.occupied_slots}  🔴")
        print(f"     Occupancy    : {result.occupancy_rate}%")
        print(f"\n  Per-slot breakdown:")
        for s in result.slots:
            icon = "🔴" if s["status"] == "occupied" else "🟢"
            print(f"     {s['id']:4s}  {icon} {s['status']:8s}  conf={s['confidence']:.2f}")

    except ImportError:
        print("  ⚠️  ultralytics not installed — using mock detector")
        print("     Run: pip install ultralytics")
        from detector import ParkingDetector
        detector = ParkingDetector(str(SLOTS_CONFIG), model_size="n")
        result = detector.process_frame(frame, 0)
        annotated = detector.annotate_frame(frame, result)
        print(f"  Mock result — Free: {result.free_slots}, Occupied: {result.occupied_slots}")

    # 4. Save output frame
    print(f"\n[4] Saving annotated frame...")
    OUTPUT_FRAME.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_FRAME), annotated)
    print(f"  ✅ Saved → {OUTPUT_FRAME}")
    print(f"     Open it in VS Code explorer to visually verify slot overlays.")

    print("\n" + "="*55)
    print("  ✅  Test complete! Detector is working correctly.")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
