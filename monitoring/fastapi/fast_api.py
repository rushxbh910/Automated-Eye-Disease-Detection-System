from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, PlainTextResponse
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import models, transforms
import time
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
import os

app = FastAPI()

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("predictions_total", "Total prediction requests")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total failed prediction attempts")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Time taken for a prediction")
MODEL_CONFIDENCE = Gauge("model_confidence_score", "Confidence of last prediction")
DRIFT_ALERT = Gauge("drift_alert", "1 if low confidence drift detected")

low_confidence_streak = 0

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = os.path.join(os.getcwd(), "MobileNetV3.pth")

model = models.mobilenet_v3_small(pretrained=False)
model.classifier[3] = nn.Linear(model.classifier[3].in_features, 10)  # 10 classes
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval().to(device)

# Class labels
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
    9: "Retinitis Pigmentosa"
}

# Transform
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
            probs = torch.nn.functional.softmax(outputs, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()

        PREDICTIONS_TOTAL.inc()
        INFERENCE_LATENCY.observe(time.time() - start)
        MODEL_CONFIDENCE.set(confidence)

        if confidence < 0.5:
            low_confidence_streak += 1
            if low_confidence_streak >= 5:
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
