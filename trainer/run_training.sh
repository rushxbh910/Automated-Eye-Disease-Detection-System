#!/bin/bash
set -e

echo "[INFO] Starting model training..."

# Set MLflow URI (adjust IP if MLflow is remote)
export MLFLOW_TRACKING_URI=http://<MLFLOW_NODE_IP>:8000/

# Navigate to training code
cd /home/cc/trainer/
pip install -r requirements.txt

# Run training script (PyTorch/Sklearn etc.)
python train.py

echo "[INFO] Training complete!"
