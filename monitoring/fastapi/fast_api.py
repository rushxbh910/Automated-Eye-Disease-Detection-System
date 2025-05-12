from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
import io
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from torchvision.models import DenseNet121_Weights
import time
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
def get_model_densenet121(num_classes, freeze_layers=True):
    model = models.densenet121(weights=None)
    if freeze_layers:
        features = list(model.features.children())
        total_blocks = len(features)
        for idx, module in enumerate(features):
            if idx < total_blocks - 2:
                for param in module.parameters():
                    param.requires_grad = False
            else:
                for param in module.parameters():
                    param.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(model.classifier.in_features, num_classes)
    )
    return model.to(device)

model_path = "densenet121_model.pth"
num_classes = 10
model = get_model_densenet121(num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# === Class Map ===
class_map = {i: f"Class-{i}" for i in range(num_classes)}

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
