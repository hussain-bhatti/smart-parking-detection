#!/bin/bash
# ParkVision Startup Script
# Run: chmod +x start.sh && ./start.sh

set -e

echo ""
echo "╔══════════════════════════════════════╗"
echo "║     ParkVision Parking Detector      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌  Python 3 not found. Install from https://python.org"
    exit 1
fi

echo "✅  Python: $(python3 --version)"

# Install dependencies if needed
echo ""
echo "📦  Checking dependencies..."
python3 -c "import ultralytics" 2>/dev/null || {
    echo "   Installing ultralytics (YOLO)..."
    pip install ultralytics -q
}
python3 -c "import cv2" 2>/dev/null || {
    echo "   Installing opencv..."
    pip install opencv-python -q
}
python3 -c "import flask" 2>/dev/null || {
    echo "   Installing flask..."
    pip install flask flask-cors -q
}
echo "✅  All dependencies ready"

# Create directories
mkdir -p uploads output_frames

echo ""
echo "🚀  Starting server at http://localhost:5000"
echo "   Press Ctrl+C to stop"
echo ""

cd backend
python3 app.py
