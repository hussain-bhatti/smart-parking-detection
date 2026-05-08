Here's the complete step-by-step process to run ParkVision in VS Code.

Step 1 — Wait for pip to finish
Your terminal was still installing. Wait until you see:
Successfully installed ultralytics flask flask-cors ...
If it errored out, run it again:
cmdpip install -r requirements.txt

Step 2 — Verify the folder structure
Make sure your VS Code has this folder open (the inner parkvision_vscode folder):
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
If you opened the outer folder by mistake, do File → Open Folder and select the inner parkvision_vscode folder.

Step 3 — Select Python Interpreter
Press Ctrl + Shift + P → type:
Python: Select Interpreter
Choose the one that says .venv — it looks like:
Python 3.10.x ('.venv': venv)  .\venv\Scripts\python.exe

Step 4 — Open a new terminal
Press Ctrl + ` (backtick) to open the VS Code terminal.
You should see .venv activated — the prompt looks like:
(.venv) PS C:\Users\hussa\...\parkvision_vscode>
If .venv is not activated, run:
cmd.venv\Scripts\activate

Step 5 — Run the server
In the terminal, run:
cmdpython backend/app.py
You should see:
====================================================
  ParkVision  —  Real-Time Parking Detection
  → http://localhost:5000
====================================================
Do not close this terminal. The server runs here.

Step 6 — Open the dashboard
Open your browser (Chrome or Edge recommended) and go to:
http://localhost:5000
You'll see the ParkVision dashboard with a video upload zone.

Step 7 — Run detection on your video

Click "Choose Video" or drag-and-drop your parking CCTV video onto the upload zone
The video uploads and processing starts automatically
You'll see the live annotated feed appear — frames play in real time with green (free) and red (occupied) overlays
The stats cards (free/occupied counts) update live as each frame is processed
The slot map updates every ~0.5 seconds
When done, the full annotated video appears below for playback and download


Step 8 — (Optional but recommended) Define exact slot polygons
By default, the system auto-generates a grid of slots. For accurate results on a real camera, run the annotator once:
Open a second terminal (click the + icon in VS Code terminal panel):
cmd.venv\Scripts\activate
python backend/slot_annotator.py --video sample_videos/your_video.mp4 --output backend/slots_config.json
A window opens showing the first frame. Use your mouse:
ActionWhat it doesLeft clickAdd a corner pointRight clickFinish the current slotTChange type (standard / handicap / EV)UUndo last pointDDelete last slotSSave and exit
After saving, every future upload will use your precise slot definitions.

Troubleshooting
"python is not recognized"
cmdpy backend/app.py
or use python3 instead of python
Port 5000 already in use
cmd# Find and kill what's using it:
netstat -ano | findstr :5000
taskkill /PID <the_pid_number> /F
# Then run again:
python backend/app.py
"No module named ultralytics"
cmd.venv\Scripts\activate
pip install ultralytics
python backend/app.py
YOLO model downloading on first run
The first time you run it, YOLO automatically downloads yolov8n.pt (~6 MB). You'll see:
Downloading yolov8n.pt ...
Just wait ~30 seconds — it only happens once.
Video uploads but nothing appears in the stream

Make sure you're on Chrome or Edge (Firefox has issues with MJPEG streams)
Check the terminal for any error messages in red
Try a shorter video first (under 2 minutes) to confirm it works

Detection looks wrong (all slots occupied or all free)
Run the slot annotator (Step 8) to draw precise slot polygons for your specific camera angle. The auto-grid works for testing but isn't accurate on real footage.
