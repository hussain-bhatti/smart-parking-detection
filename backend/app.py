"""
Flask API Server — Real-Time Frame Streaming
Streams annotated frames as MJPEG so the dashboard shows live video while processing.
Also streams per-frame stats via Server-Sent Events (SSE).

Run:  python backend/app.py
Open: http://localhost:5000
"""

import cv2
import json
import sys
import threading
import time
import uuid
import queue
from pathlib import Path
from flask import Flask, Response, request, jsonify, send_file, send_from_directory
from flask_cors import CORS

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "output_frames"
FRONTEND   = BASE_DIR / "frontend"
SLOTS_CFG  = BASE_DIR / "backend" / "slots_config.json"

sys.path.insert(0, str(BASE_DIR / "backend"))
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

ALLOWED = {"mp4", "avi", "mov", "mkv", "webm"}

app = Flask(__name__, static_folder=str(FRONTEND))
CORS(app)

# ── In-memory state ──────────────────────────────────────────────────────────
jobs         = {}   # job_id → job dict
frame_queues = {}   # job_id → Queue of JPEG bytes (for MJPEG stream)


# ════════════════════════════════════════════════════════════════════════════
#  Background processing
# ════════════════════════════════════════════════════════════════════════════

def run_job(job_id: str, video_path: str, slots_path: str):
    from detector import ParkingDetector

    jobs[job_id]["status"]     = "processing"
    jobs[job_id]["started_at"] = time.time()
    q = frame_queues[job_id]

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = f"Cannot open: {video_path}"
        q.put(None)
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_px  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    jobs[job_id]["video_info"] = {"width": w, "height": h_px, "fps": fps,
                                   "total_frames": total,
                                   "duration_sec": round(total / fps, 1)}

    # Output video writer
    out_mp4 = str(OUTPUT_DIR / f"{job_id}_annotated.mp4")
    writer  = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"),
                               max(1.0, fps / 15), (w, h_px))

    # Init detector
    try:
        detector = ParkingDetector(slots_path)
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        cap.release()
        q.put(None)
        return

    SKIP = 15   # process 1 in every 15 frames  (~0.5 s at 30 fps)
    results, history = [], []
    frame_no = processed = 0
    t0 = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_no % SKIP == 0:
            try:
                annotated, statuses, detections = detector.process_frame(frame)
                summary_data = detector.get_summary(statuses)
            except Exception as e:
                print(f"[{job_id}] frame {frame_no} err: {e}")
                annotated    = frame
                summary_data = None

            writer.write(annotated)
            processed += 1

            # Push JPEG into stream queue (drop old frames if browser is slow)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 72])
            if ok:
                while q.qsize() > 8:
                    try: q.get_nowait()
                    except: pass
                q.put(buf.tobytes())

            # Update job stats
            if summary_data:
                ts  = round(frame_no / fps, 2)
                pct = round(frame_no / total * 100, 1)
                elapsed = time.time() - t0
                eta = (elapsed / processed) * (total / SKIP - processed) if processed > 1 else 0

                results.append({
                    "frame_number":   frame_no,
                    "timestamp_sec":  ts,
                    "total_slots":    summary_data["total"],
                    "free_slots":     summary_data["free"],
                    "occupied_slots": summary_data["occupied"],
                    "occupancy_rate": summary_data["occupancy_pct"],
                    "slots":          summary_data["slots"],
                })
                history.append({"free": summary_data["free"],
                                 "occupied": summary_data["occupied"],
                                 "rate": summary_data["occupancy_pct"]})

                jobs[job_id]["progress"]       = pct
                jobs[job_id]["current_status"] = {
                    "percent":          pct,
                    "processed_frames": processed,
                    "free":             summary_data["free"],
                    "occupied":         summary_data["occupied"],
                    "total":            summary_data["total"],
                    "eta_sec":          round(eta, 1),
                    "timestamp_sec":    ts,
                    "slots":            summary_data["slots"],
                }

        frame_no += 1

    cap.release()
    writer.release()
    q.put(None)   # sentinel → tells MJPEG generator to stop

    # Summary
    n        = len(history) or 1
    avg_free = round(sum(x["free"]     for x in history) / n, 1)
    avg_occ  = round(sum(x["occupied"] for x in history) / n, 1)
    avg_rate = round(sum(x["rate"]     for x in history) / n, 1)

    summary = {
        "video_duration_sec":         round(total / fps, 1),
        "total_frames_processed":     processed,
        "processing_time_sec":        round(time.time() - t0, 1),
        "total_slots":                len(detector.slots),
        "average_free_slots":         avg_free,
        "average_occupied_slots":     avg_occ,
        "average_occupancy_rate_pct": avg_rate,
        "peak_occupied_slots":        max((x["occupied"] for x in history), default=0),
        "min_occupied_slots":         min((x["occupied"] for x in history), default=0),
        "final_status":               results[-1] if results else {},
        "frame_results":              results,
    }

    with open(OUTPUT_DIR / f"{job_id}_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    jobs[job_id]["status"]       = "done"
    jobs[job_id]["progress"]     = 100
    jobs[job_id]["summary"]      = summary
    jobs[job_id]["completed_at"] = time.time()
    print(f"[{job_id}] done  free={avg_free} occ={avg_occ}  {summary['processing_time_sec']}s")


# ════════════════════════════════════════════════════════════════════════════
#  Routes
# ════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND), "index.html")

@app.route("/<path:p>")
def static_files(p):
    return send_from_directory(str(FRONTEND), p)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "jobs": len(jobs)})


# ── Upload & start job ───────────────────────────────────────────────────────
@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video"}), 400

    vf  = request.files["video"]
    ext = (vf.filename or "").rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"Unsupported type. Use: {ALLOWED}"}), 400

    job_id = uuid.uuid4().hex[:8]
    vpath  = UPLOAD_DIR / f"{job_id}.{ext}"
    vf.save(str(vpath))

    if "slots_config" in request.files:
        sf    = request.files["slots_config"]
        spath = UPLOAD_DIR / f"{job_id}_slots.json"
        sf.save(str(spath))
    elif SLOTS_CFG.exists():
        spath = SLOTS_CFG
    else:
        spath = UPLOAD_DIR / f"{job_id}_slots.json"
        _auto_slots(str(vpath), str(spath))

    jobs[job_id] = {
        "job_id":     job_id,
        "status":     "queued",
        "progress":   0,
        "filename":   vf.filename,
        "created_at": time.time(),
    }
    frame_queues[job_id] = queue.Queue(maxsize=30)

    threading.Thread(target=run_job, args=(job_id, str(vpath), str(spath)),
                     daemon=True).start()

    return jsonify({"job_id": job_id}), 202


# ── MJPEG live stream  →  <img src="/api/stream/JOB_ID"> ────────────────────
@app.route("/api/stream/<job_id>")
def stream(job_id):
    """
    Streams annotated frames as MJPEG.
    The browser treats it like a live video feed.
    Each frame arrives ~0.5 s after it is processed.
    """
    if job_id not in frame_queues:
        return "Job not found", 404

    q = frame_queues[job_id]

    def generate():
        while True:
            try:
                data = q.get(timeout=60)
            except queue.Empty:
                break
            if data is None:    # job finished
                break
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")

    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ── SSE live stats  →  new EventSource('/api/events/JOB_ID') ─────────────────
@app.route("/api/events/<job_id>")
def events(job_id):
    """
    Server-Sent Events: pushes a JSON payload every ~0.4 s with
    {status, progress, free, occupied, total, eta_sec, slots[]}
    Frontend can update the stats panel and slot map in real time.
    """
    if job_id not in jobs:
        return "Job not found", 404

    def generate():
        last = -1
        while True:
            job = jobs.get(job_id, {})
            pct = job.get("progress", 0)
            cs  = job.get("current_status", {})

            if pct != last or job.get("status") in ("done", "error"):
                payload = {
                    "status":        job.get("status"),
                    "progress":      pct,
                    "free":          cs.get("free",     0),
                    "occupied":      cs.get("occupied", 0),
                    "total":         cs.get("total",    0),
                    "eta_sec":       cs.get("eta_sec",  0),
                    "timestamp_sec": cs.get("timestamp_sec", 0),
                    "slots":         cs.get("slots",    []),
                }
                yield f"data: {json.dumps(payload)}\n\n"
                last = pct

            if job.get("status") in ("done", "error"):
                # Send final summary before closing
                if job.get("status") == "done":
                    s = job.get("summary", {})
                    final = {
                        "status":   "done",
                        "progress": 100,
                        "summary":  {k: v for k, v in s.items() if k != "frame_results"},
                        "slots":    s.get("final_status", {}).get("slots", []),
                        "free":     s.get("final_status", {}).get("free_slots",     0),
                        "occupied": s.get("final_status", {}).get("occupied_slots", 0),
                        "total":    s.get("total_slots", 0),
                    }
                    yield f"data: {json.dumps(final)}\n\n"
                break

            time.sleep(0.4)

    return Response(generate(),
                    mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no"})


# ── Job endpoints ────────────────────────────────────────────────────────────
@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "Not found"}), 404
    job = jobs[job_id].copy()
    if "summary" in job:
        job["summary"] = {k: v for k, v in job["summary"].items()
                          if k != "frame_results"}
    return jsonify(job)

@app.route("/api/jobs/<job_id>/frames")
def job_frames(job_id):
    p = OUTPUT_DIR / f"{job_id}_results.json"
    return send_file(str(p)) if p.exists() else (jsonify({"error": "Not ready"}), 400)

@app.route("/api/output/<job_id>/video")
def output_video(job_id):
    p = OUTPUT_DIR / f"{job_id}_annotated.mp4"
    return send_file(str(p), mimetype="video/mp4") if p.exists() else ("Not ready", 404)

@app.route("/api/output/<job_id>/json")
def output_json(job_id):
    p = OUTPUT_DIR / f"{job_id}_results.json"
    return send_file(str(p)) if p.exists() else ("Not ready", 404)


# ── Auto slot grid ────────────────────────────────────────────────────────────
def _auto_slots(video_path: str, out: str):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    h, w = frame.shape[:2] if ret else (720, 1280)
    mx, my = int(w*.05), int(h*.10)
    rows, cols, gap = 3, 4, 8
    sw, sh = (w-2*mx)//cols, (h-2*my)//rows
    slots = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = mx+c*sw+gap, my+r*sh+gap
            x2, y2 = x1+sw-gap*2, y1+sh-gap*2
            slots.append({"id": f"{chr(65+r)}{c+1}", "type": "standard",
                           "polygon": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]})
    with open(out, "w") as f:
        json.dump({"lot_name":"Auto","camera_id":"cam_01",
                   "frame_size":[w,h],"slots":slots}, f, indent=2)
    print(f"[API] Auto slots: {len(slots)} for {w}x{h}")


# ── Start ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*52)
    print("  ParkVision  —  Real-Time Parking Detection")
    print("  http://localhost:5000")
    print("="*52 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
