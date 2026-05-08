"""
Video Processor — CLI entry point
Processes a parking lot video and produces:
  - Annotated output video  (green=free, red=occupied per slot)
  - JSON results file       (per-frame stats + summary)

Usage:
  python process_video.py --video parking.mp4 --slots slots_config.json
  python process_video.py --video parking.mp4 --slots slots_config.json \
      --output-video out.mp4 --output-json results.json --frame-skip 15 --model n
"""

import cv2
import json
import time
import argparse
import sys
from pathlib import Path
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent))
from detector import ParkingDetector


def process_video(
    video_path: str,
    slots_config_path: str,
    output_video_path: str = None,
    results_json_path: str = None,
    frame_skip: int = 15,
    model_size: str = "n",
    progress_callback=None,
) -> dict:
    """
    Process a parking lot video end-to-end.

    Returns a summary dict with stats and per-frame results.
    """
    print(f"\n[Processor] Video       : {video_path}")
    print(f"[Processor] Slots config: {slots_config_path}")
    print(f"[Processor] Frame skip  : every {frame_skip} frames")
    print(f"[Processor] YOLO model  : yolov8{model_size}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps

    print(f"[Processor] Resolution  : {width}x{height}")
    print(f"[Processor] Duration    : {duration_sec:.1f}s  ({total_frames} frames @ {fps:.0f}fps)\n")

    # ── Detector ────────────────────────────────────────────────────────────
    detector = ParkingDetector(slots_config_path, model_size)

    # ── Video writer ─────────────────────────────────────────────────────────
    writer = None
    if output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Output video runs at fps/frame_skip so it plays in real time
        out_fps = max(1.0, fps / frame_skip)
        writer = cv2.VideoWriter(output_video_path, fourcc, out_fps, (width, height))
        print(f"[Processor] Output video: {output_video_path}  ({out_fps:.1f}fps)")

    # ── Processing loop ───────────────────────────────────────────────────────
    frame_results = []
    occupancy_history = []   # [{free, occupied, rate}]
    processed = 0
    frame_number = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_skip != 0:
            frame_number += 1
            continue

        # Run detection
        result = detector.process_frame(frame, frame_number)

        # Annotate
        annotated = detector.annotate_frame(frame, result)
        if writer:
            writer.write(annotated)

        # Collect stats
        ts = round(frame_number / fps, 2)
        frame_results.append({
            "frame_number":   result.frame_number,
            "timestamp_sec":  ts,
            "total_slots":    result.total_slots,
            "free_slots":     result.free_slots,
            "occupied_slots": result.occupied_slots,
            "occupancy_rate": result.occupancy_rate,
            "slots":          result.slots,
        })
        occupancy_history.append({
            "free":     result.free_slots,
            "occupied": result.occupied_slots,
            "rate":     result.occupancy_rate,
        })

        processed += 1
        frame_number += 1

        # ── Progress ──────────────────────────────────────────────────────
        elapsed = time.time() - t0
        pct = round(frame_number / total_frames * 100, 1)
        eta = (elapsed / processed) * (total_frames / frame_skip - processed) if processed > 1 else 0

        if processed % 10 == 0 or pct >= 99.9:
            print(f"  {pct:5.1f}%  frame {frame_number:>6}/{total_frames}  "
                  f"free={result.free_slots}  occ={result.occupied_slots}  ETA={eta:.0f}s")

        if progress_callback:
            progress_callback(pct, {
                "percent":          pct,
                "processed_frames": processed,
                "free":             result.free_slots,
                "occupied":         result.occupied_slots,
                "total":            result.total_slots,
                "eta_sec":          round(eta, 1),
            })

    cap.release()
    if writer:
        writer.release()

    # ── Summary ───────────────────────────────────────────────────────────
    n = len(occupancy_history) or 1
    avg_free = round(sum(h["free"]     for h in occupancy_history) / n, 1)
    avg_occ  = round(sum(h["occupied"] for h in occupancy_history) / n, 1)
    avg_rate = round(sum(h["rate"]     for h in occupancy_history) / n, 1)
    peak_occ = max((h["occupied"] for h in occupancy_history), default=0)
    min_occ  = min((h["occupied"] for h in occupancy_history), default=0)

    summary = {
        "video_path":                  str(video_path),
        "slots_config":                str(slots_config_path),
        "video_duration_sec":          round(duration_sec, 1),
        "total_frames_processed":      processed,
        "processing_time_sec":         round(time.time() - t0, 1),
        "total_slots":                 len(detector.slots_config["slots"]),
        "average_free_slots":          avg_free,
        "average_occupied_slots":      avg_occ,
        "average_occupancy_rate_pct":  avg_rate,
        "peak_occupied_slots":         peak_occ,
        "min_occupied_slots":          min_occ,
        "final_status":                frame_results[-1] if frame_results else {},
        "frame_results":               frame_results,
        "output_video":                output_video_path,
    }

    if results_json_path:
        with open(results_json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[Processor] Results → {results_json_path}")

    # ── Print report ──────────────────────────────────────────────────────
    print(f"\n{'═'*48}")
    print(f"  ✅  PROCESSING COMPLETE")
    print(f"{'═'*48}")
    print(f"  Total Slots      : {summary['total_slots']}")
    print(f"  Avg Free         : {avg_free}")
    print(f"  Avg Occupied     : {avg_occ}")
    print(f"  Avg Occupancy    : {avg_rate}%")
    print(f"  Peak Occupied    : {peak_occ}")
    print(f"  Frames Processed : {processed}")
    print(f"  Processing Time  : {summary['processing_time_sec']}s")
    print(f"{'═'*48}\n")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Process a parking lot video")
    ap.add_argument("--video",        required=True,           help="Input video path")
    ap.add_argument("--slots",        required=True,           help="slots_config.json path")
    ap.add_argument("--output-video", default=None,            help="Annotated output video path")
    ap.add_argument("--output-json",  default="results.json",  help="Results JSON output path")
    ap.add_argument("--frame-skip",   type=int, default=15,    help="Process every Nth frame")
    ap.add_argument("--model",        default="n",             choices=["n", "s", "m"],
                    help="YOLO model size (n=fastest, m=most accurate)")
    args = ap.parse_args()

    process_video(
        video_path=args.video,
        slots_config_path=args.slots,
        output_video_path=args.output_video,
        results_json_path=args.output_json,
        frame_skip=args.frame_skip,
        model_size=args.model,
    )


if __name__ == "__main__":
    main()
