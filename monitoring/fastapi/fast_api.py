from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import MobileNet_V3_Large_Weights
import time
import numpy as np
import mlflow
import requests
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

app = FastAPI()

# === Prometheus metrics ===
PREDICTIONS_TOTAL = Counter("predictions_total", "Total prediction requests")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total failed prediction attempts")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Time taken for a prediction")
MODEL_CONFIDENCE = Gauge("model_confidence_score", "Confidence of last prediction")
DRIFT_ALERT = Gauge("drift_alert", "1 if low confidence drift detected")

low_confidence_streak = 0
confidence_threshold = 0.5
streak_threshold = 5


# === Device ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load model ===
def get_model_mobilenetv3(num_classes, freeze_layers=True):
    model = models.mobilenet_v3_large(weights=None)
    if freeze_layers:
        total_blocks = len(model.features)
        for idx, module in enumerate(model.features):
            if idx < total_blocks - 2:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[0].in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(512, num_classes)
    )
    return model.to(device)

model_path = "MobileNetV3.pth"
num_classes = 15
model = get_model_mobilenetv3(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Class Map ===
class_map = {
    0: "Central Serous Chorioretinopathy",
    1: "Diabetic Retinopathy",
    2: "Disc Edema",
    3: "Glaucoma",
    4: "Healthy",
    5: "Macular Scar",
    6: "Myopia",
    7: "Pterygium",
    8: "Retinal Detachment",
    9: "Retinitis Pigmentosa",
    10: "Unused-Class-10",
    11: "Unused-Class-11",
    12: "Unused-Class-12",
    13: "Unused-Class-13",
    14: "Unused-Class-14"
}

# === Preprocessing ===
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global low_confidence_streak
    try:
        start = time.time()
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = torch.argmax(probs).item()
            confidence = probs[0][pred].item()

        PREDICTIONS_TOTAL.inc()
        INFERENCE_LATENCY.observe(time.time() - start)
        MODEL_CONFIDENCE.set(confidence)

        if confidence < confidence_threshold:
            low_confidence_streak += 1
            if low_confidence_streak >= streak_threshold:
                DRIFT_ALERT.set(1)
        else:
            low_confidence_streak = 0
            DRIFT_ALERT.set(0)



        return JSONResponse({
            "predicted_class": class_map[pred],
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        PREDICTION_ERRORS.inc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)