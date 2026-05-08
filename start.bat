@echo off
echo.
echo ========================================
echo      ParkVision Parking Detector
echo ========================================
echo.

python --version >nul 2>&1 || (
    echo [ERROR] Python not found. Install from https://python.org
    pause
    exit /b 1
)

echo Checking dependencies...
pip install ultralytics opencv-python flask flask-cors -q

if not exist uploads mkdir uploads
if not exist output_frames mkdir output_frames

echo.
echo Starting server at http://localhost:5000
echo Press Ctrl+C to stop
echo.

cd backend
python app.py
pause
