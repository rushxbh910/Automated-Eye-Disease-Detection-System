import requests
import time

# === CONFIG ===
API_URL = "http://localhost:8000/predict"  # Use external IP if needed
IMAGE_PATH = "sample.jpeg"
TOTAL_REQUESTS = 100

# === LOOP ===
for i in range(TOTAL_REQUESTS):
    with open(IMAGE_PATH, "rb") as f:
        files = {"file": (IMAGE_PATH, f, "image/jpeg")}
        try:
            response = requests.post(API_URL, files=files)
            if response.status_code == 200:
                result = response.json()
                print(f"[{i+1}] ✅ {result['predicted_class']} (confidence: {result['confidence']})")
            else:
                print(f"[{i+1}] ❌ Failed: {response.status_code}")
        except Exception as e:
            print(f"[{i+1}] ❌ Error: {e}")

    time.sleep(0.2)  # slight delay to mimic real traffic
