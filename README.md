# ParkVision — Run Guide for VS Code

Complete step-by-step process to get ParkVision running locally in VS Code.

---

## Step 1 — Wait for pip to finish

Your terminal was still installing. Wait until you see:

```
Successfully installed ultralytics flask flask-cors ...
```

If it errored out, run it again:

```cmd
pip install -r requirements.txt
```

---

## Step 2 — Verify the folder structure

Make sure your VS Code has this folder open (the **inner** `parkvision_vscode` folder):

```
parkvision_vscode/        ← this is your VS Code workspace root
├── backend/
│   ├── app.py
│   ├── detector.py
│   ├── process_video.py
│   ├── slot_annotator.py
│   └── test_detector.py
├── frontend/
│   └── index.html
├── uploads/
├── output_frames/
├── sample_videos/
└── requirements.txt
```

> If you opened the outer folder by mistake, go to `File → Open Folder` and select the inner `parkvision_vscode` folder.

---

## Step 3 — Select Python Interpreter

Press `Ctrl + Shift + P` and type:

```
Python: Select Interpreter
```

Choose the one that says `.venv` — it looks like:

```
Python 3.10.x ('.venv': venv)  .\venv\Scripts\python.exe
```

---

## Step 4 — Open a new terminal

Press `` Ctrl + ` `` (backtick) to open the VS Code terminal.

You should see `.venv` activated — the prompt looks like:

```
(.venv) PS C:\Users\hussa\...\parkvision_vscode>
```

If `.venv` is **not** activated, run:

```cmd
.venv\Scripts\activate
```

---

## Step 5 — Run the server

```cmd
python backend/app.py
```

You should see:

```
====================================================
  ParkVision  —  Real-Time Parking Detection
  → http://localhost:5000
====================================================
```

> **Do not close this terminal.** The server runs here.

---

## Step 6 — Open the dashboard

Open your browser (Chrome or Edge recommended) and go to:

```
http://localhost:5000
```

You will see the ParkVision dashboard with a video upload zone.

---

## Step 7 — Run detection on your video

1. Click **"Choose Video"** or drag-and-drop your parking CCTV video onto the upload zone
2. The video uploads and processing starts automatically
3. The **live annotated feed** appears — frames play in real time with green (free) and red (occupied) overlays
4. The **stats cards** (free/occupied counts) update live as each frame is processed
5. The **slot map** updates every ~0.5 seconds
6. When done, the full annotated video appears below for playback and download

---

## Step 8 — Define exact slot polygons (Optional but recommended)

By default, the system auto-generates a grid of slots. For accurate results on a real camera, run the annotator once.

Open a **second terminal** (click the `+` icon in the VS Code terminal panel):

```cmd
.venv\Scripts\activate
python backend/slot_annotator.py --video sample_videos/your_video.mp4 --output backend/slots_config.json
```

A window opens showing the first frame of your video. Use your mouse to draw slot polygons:

| Action | Effect |
|--------|--------|
| Left click | Add a corner point |
| Right click | Finish the current slot |
| `T` | Cycle type — standard / handicap / EV |
| `U` | Undo last point |
| `D` | Delete last slot |
| `S` | Save and exit |

After saving, every future upload will use your precise slot definitions.

---

## Troubleshooting

### "python is not recognized"

```cmd
py backend/app.py
```

Or use `python3` instead of `python`.

---

### Port 5000 already in use

```cmd
netstat -ano | findstr :5000
taskkill /PID <the_pid_number> /F

python backend/app.py
```

---

### "No module named ultralytics"

```cmd
.venv\Scripts\activate
pip install ultralytics
python backend/app.py
```

---

### YOLO model downloading on first run

The first time you run it, YOLO automatically downloads `yolov8n.pt` (~6 MB). You will see:

```
Downloading yolov8n.pt ...
```

Wait ~30 seconds — it only happens once.

---

### Video uploads but nothing appears in the stream

- Use **Chrome or Edge** — Firefox has issues with MJPEG streams
- Check the terminal for any error messages in red
- Try a shorter video first (under 2 minutes) to confirm it works

---

### Detection looks wrong (all slots occupied or all free)

Run the slot annotator from Step 8 to draw precise polygons for your specific camera angle. The auto-grid works for testing but is not accurate on real footage.
