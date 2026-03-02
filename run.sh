#!/bin/bash

# ============================================================
# EOU Detector — Start Script
# ============================================================

# Set model directory (change this to your model path)
export EOU_MODEL_DIR="./model"

echo "🚀 Starting EOU Detector..."
echo "📁 Model directory: $EOU_MODEL_DIR"
echo "🌐 Server: http://localhost:8000"
echo ""

# Run with uvicorn
./env/Scripts/python.exe -m uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level info \
    --workers 1