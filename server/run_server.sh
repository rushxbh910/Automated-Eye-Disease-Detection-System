#!/bin/bash
set -e

echo "[INFO] Booting model inference server..."

# Navigate to server app
cd /home/cc/server/
pip install -r requirements.txt

# Run FastAPI or Flask server
uvicorn app:app --host 0.0.0.0 --port 8080

# Optional: Log to file
# >> /var/log/inference_server.log 2>&1
