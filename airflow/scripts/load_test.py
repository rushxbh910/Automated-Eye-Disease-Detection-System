import requests

def load_test():
    for i in range(10):
        resp = requests.post("http://localhost:8501/predict", json={"image_path": "test.jpg"})
        print(f"[{i}] Response: {resp.status_code}")

if __name__ == "__main__":
    load_test()
