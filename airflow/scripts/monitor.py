import requests

def check_prometheus():
    response = requests.get("http://localhost:9090/api/v1/query", params={"query": "up"})
    print("Prometheus Response:", response.json())

if __name__ == "__main__":
    check_prometheus()
