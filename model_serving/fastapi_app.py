from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse, PlainTextResponse
from PIL import Image
import io
import numpy as np
import onnxruntime as ort
import time
import os
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from torchvision import transforms
from datetime import datetime
import boto3
import uuid
from mimetypes import guess_type

app = FastAPI()

# Prometheus metrics
PREDICTIONS_TOTAL = Counter("predictions_total", "Total prediction requests")
PREDICTION_ERRORS = Counter("prediction_errors_total", "Total failed prediction attempts")
INFERENCE_LATENCY = Histogram("inference_latency_seconds", "Time taken for a prediction")
MODEL_CONFIDENCE = Gauge("model_confidence_score", "Confidence of last prediction")
DRIFT_ALERT = Gauge("drift_alert", "1 if low confidence drift detected")

low_confidence_streak = 0

# MinIO setup
MINIO_URL = os.getenv("MINIO_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER", "your-access-key")
MINIO_SECRET_KEY = os.getenv("MINIO_PASSWORD", "your-secret-key")
s3 = boto3.client(
    's3',
    endpoint_url=MINIO_URL,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
    region_name='us-east-1'
)
BUCKET_NAME = "production"

# Load ONNX model
model_path = os.path.join(os.getcwd(), "resnet50_custom_model.onnx")
session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
input_name = session.get_inputs()[0].name

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
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).numpy()

        outputs = session.run(None, {input_name: input_tensor})
        probs = np.exp(outputs[0]) / np.sum(np.exp(outputs[0]), axis=1, keepdims=True)
        pred = int(np.argmax(probs))
        confidence = float(probs[0][pred])

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

        # Save to MinIO
        prediction_id = str(uuid.uuid4())
        ext = os.path.splitext(file.filename)[1] or ".jpg"
        timestamp = datetime.utcnow().isoformat() + "Z"
        key = f"class_{pred:02d}/{prediction_id}{ext}"

        s3.upload_fileobj(io.BytesIO(image_bytes), BUCKET_NAME, key, ExtraArgs={"ContentType": guess_type(file.filename)[0] or "application/octet-stream"})
        s3.put_object_tagging(Bucket=BUCKET_NAME, Key=key, Tagging={
            "TagSet": [
                {"Key": "predicted_class", "Value": class_map[pred]},
                {"Key": "confidence", "Value": f"{confidence:.3f}"},
                {"Key": "timestamp", "Value": timestamp}
            ]
        })

        return JSONResponse({
            "predicted_class": class_map[pred],
            "confidence": round(confidence, 4),
            "s3_key": key
        })

    except Exception as e:
        PREDICTION_ERRORS.inc()
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/flag/{key:path}")
def flag_prediction(key: str):
    tags = s3.get_object_tagging(Bucket=BUCKET_NAME, Key=key)['TagSet']
    tag_dict = {tag['Key']: tag['Value'] for tag in tags}
    tag_dict['flagged'] = "true"
    tag_set = [{"Key": k, "Value": v} for k, v in tag_dict.items()]
    s3.put_object_tagging(Bucket=BUCKET_NAME, Key=key, Tagging={'TagSet': tag_set})
    return RedirectResponse(url="/success")

@app.post("/correct-label/{key:path}")
def correct_label(key: str, request: Request):
    from starlette.requests import FormData
    form_data = request._form
    new_label = form_data.get("corrected_class")
    tags = s3.get_object_tagging(Bucket=BUCKET_NAME, Key=key)['TagSet']
    tag_dict = {tag['Key']: tag['Value'] for tag in tags}
    tag_dict['corrected_class'] = new_label
    tag_set = [{"Key": k, "Value": v} for k, v in tag_dict.items()]
    s3.put_object_tagging(Bucket=BUCKET_NAME, Key=key, Tagging={'TagSet': tag_set})
    return RedirectResponse(url="/success")

@app.get("/success")
def success():
    return HTMLResponse("<h3>Thank you for your feedback!</h3>")

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)
