# Project Report: Automated Eye Disease Detection System

## Data Pipeline Setup and Dashboard

---

### 1. Object Store Setup

The object store was created on the **CHI\@TACC** site using the **Chameleon GUI**. A container named `object-persist-project24` was created to hold datasets.

---

### 2. Block Storage Setup

Block storage was provisioned on **KVM\@TACC** using the **Chameleon GUI**. A volume named `block-persist-project24` was created and attached to the persistent KVM\@TACC node. Once attached, it was formatted and mounted:

```bash
sudo mkfs.ext4 /dev/vdb
sudo mkdir -p /mnt/block
sudo mount /dev/vdb /mnt/block
```

---

### 3. Loading Raw Data into the Object Store

The raw dataset was first uploaded to the node using `scp` and then transferred to the object store using the following `docker-compose-etl-upload-raw.yaml` file (located in the `Data-Pipelining` folder of this repository):

```yaml
name: eye-raw-etl

services:
  load-raw-data:
    container_name: etl_load_raw_data
    image: rclone/rclone:latest
    volumes:
      - ~/.config/rclone/rclone.conf:/root/.config/rclone/rclone.conf:ro
    entrypoint: /bin/sh
    command:
      - -c
      - |
        echo "Uploading raw EYE dataset to object store..."

        if [ -z "$RCLONE_CONTAINER" ]; then
          echo "ERROR: RCLONE_CONTAINER is not set"
          exit 1
        fi

        rclone copy /data chi_tacc:$RCLONE_CONTAINER/raw_eye_dataset \
        --progress \
        --transfers=32 \
        --checkers=16 \
        --multi-thread-streams=4 \
        --fast-list

        echo "Upload complete. Contents:"
        rclone lsd chi_tacc:$RCLONE_CONTAINER
```

To run:

```bash
docker compose -f docker-compose-etl-upload-raw.yaml up load-raw-data
```

---

### 4. ETL Pipeline (Extract-Transform-Load)

The data transformation is handled by `transform.py` (located in `Data-Pipelining/streamlit-dashboard/`). It reads raw images from the object store, applies resizing/normalization, and outputs 5 stratified splits: train, test, holdout\_1, holdout\_2, holdout\_3.

This is orchestrated using `docker-compose-etl.yaml`:

```yaml
name: eye-etl

volumes:
  eye_data:

services:
  extract-data:
    container_name: etl_extract_eye_data
    image: python:3.11
    user: root
    volumes:
      - eye_data:/data
      - /mnt/object:/mnt/object:ro
    working_dir: /data
    command:
      - bash
      - -c
      - |
        rm -rf raw_eye_dataset
        mkdir -p raw_eye_dataset
        cp -r /mnt/object/raw_eye_dataset/* raw_eye_dataset/

  transform-data:
    container_name: etl_transform_data
    image: python:3.11
    user: root
    volumes:
      - eye_data:/data
      - ./scripts/transform.py:/data/transform.py:ro
    working_dir: /data
    command:
      - bash
      - -c
      - |
        pip install torch torchvision scikit-learn --break-system-packages
        python3 transform.py

  load-transformed-data:
    container_name: etl_load_transformed_data
    image: rclone/rclone:latest
    volumes:
      - eye_data:/data
      - ~/.config/rclone/rclone.conf:/root/.config/rclone/rclone.conf:ro
    entrypoint: /bin/sh
    command:
      - -c
      - |
        rclone copy /data/processed chi_tacc:$RCLONE_CONTAINER/transformed_eye_dataset \
          --progress \
          --transfers=32 \
          --checkers=16 \
          --multi-thread-streams=4 \
          --fast-list
```

Run with:

```bash
docker compose -f docker-compose-etl.yaml up
```

Transformed output is stored at `object-persist-project24/transformed_eye_dataset`.

---

### 5. Streamlit Data Dashboard

A Streamlit dashboard allows dataset exploration by class distribution and sample visualization.

**Dockerfile:**

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**requirements.txt:**

```
streamlit
Pillow
matplotlib
```

**Build and run:**

```bash
docker build -t streamlit-eye-app .
docker run -d -p 8501:8501 \
  -v /mnt/object/transformed_eye_dataset:/data \
  --name streamlit_app \
  streamlit-eye-app
```

Access: `http://<NODE_IP>:8501`

---

## Training Using MLflow and Ray

### A. Node Provisioning

A GPU node was provisioned using the `CC-Ubuntu24.04-CUDA` image. A floating IP was associated.

```python
from chi import server, context, lease
context.choose_project()
context.choose_site("CHI@TACC")
l = lease.get_lease("Project24001")
s = server.Server("node-sb9880-01", reservation_id=l.node_reservations[0]["id"], image_name="CC-Ubuntu24.04-CUDA")
s.submit(idempotent=True)
s.associate_floating_ip()
s.refresh()
s.check_connectivity()
```

### B. Security Group

Ports opened:

* SSH (22)
* Jupyter (8888)
* MLflow (8000)
* MinIO (9000, 9001)

### C. Docker GPU Driver Setup

Required NVIDIA/ROCm drivers installed.

### D. Docker Images

* `Dockerfile.jupyter-torch-mlflow-rocm`
* `Dockerfile.ray-rocm`

### E. Rclone Configuration

Configure and mount object store via `rclone config` using Application Credentials.

### F. Docker Volume Setup

```bash
docker volume create EYE
```

### G. Docker Compose Launch

Run `docker-compose-ray-mlflow-data.yaml` to launch services.

### H. Jupyter + Ray + MLflow

```bash
HOST_IP=$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4)
docker run -d --rm -p 8888:8888 \
    -v ~/workspace:/home/jovyan/work/ \
    -v EYE:/mnt/ \
    -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
    -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
    -e EYE_DATA_DIR=/mnt/data/transformed_eye_dataset \
    --name jupyter \
    jupyter-ray
```

### I. Submit Ray Jobs

```bash
ray job submit \
  --runtime-env runtime.json \
  --entrypoint-num-gpus 1 \
  --entrypoint-num-cpus 8 \
  --verbose \
  --working-dir code/ \
  -- python ray-code00.py
```

---

## Serving and Monitoring

### FastAPI App (`fast_api.py`)

* Serves DenseNet121 for 15 diseases
* Tracks Prometheus metrics: predictions, errors, latency, drift
* Logs predictions to MLflow
* Endpoints: `POST /predict`, `GET /metrics`

### Deployment via Docker Compose

| Service        | Description             | Port(s)    |
| -------------- | ----------------------- | ---------- |
| eye-fastapi-\* | FastAPI for inference   | 8000       |
| mlflow         | Experiment tracking     | 5000       |
| prometheus     | Metrics collection      | 9090       |
| grafana        | Visualization dashboard | 3000       |
| minio          | Object storage          | 9000, 9001 |
| labelstudio    | Annotation interface    | 8080       |

### Prometheus

* Scrapes metrics from `eye-fastapi-*`
* Handles drift alerts

### Grafana

* Dashboards:

  * Inference count
  * Latency trends
  * Drift alert status
  * Confidence scores

---

### Next Steps

* Add Traefik/Nginx load balancing
* Use MinIO+Label Studio to trigger Airflow pipelines
* Provision Grafana dashboards via JSON

---

## Automated Eye Disease Detection System (MLOps)

Complete cloud-native pipeline using:

* **PyTorch**, **Terraform**, **Docker**, **Apache Airflow**
* Infrastructure provisioning, model lifecycle automation

### Provisioned Infrastructure (Terraform)

| Component     | Cloud Site | Flavor    | Node Name                | Floating IP |
| ------------- | ---------- | --------- | ------------------------ | ----------- |
| Training Node | KVM\@TACC  | m1.medium | eye-train-train-1-rb5726 | yes         |
| Serving Node  | KVM\@TACC  | m1.large  | eye-serve-serve-1-rb5726 | yes         |

### Block Storage

* **Volume Name:** `block-persist-project24`
* **Mounted at:** `/mnt/block`
* **Usage:** Stores training data and `model.pth`

### DAG Pipeline (Apache Airflow)

* Daily runs
* Retraining every 3 days via `eye_disease_ml_pipeline.py`

| Stage               | Tool             |
| ------------------- | ---------------- |
| Train Model         | Docker + Airflow |
| Evaluate            | Docker Compose   |
| Monitor Performance | Docker Compose   |

### Services Deployed

| Service        | Port | Description                   |
| -------------- | ---- | ----------------------------- |
| FastAPI        | 8000 | API interface for predictions |
| Prometheus     | 9090 | Metrics collection            |
| Grafana        | 3000 | Visualization dashboard       |
| MLflow         | 5000 | Experiment tracking           |
| MinIO          | 9000 | Object storage backend        |
| Label Studio   | 8080 | Annotation interface          |
| Airflow Web UI | 3001 | DAG management interface      |

---

## Folder Structure

```
Automated-Eye-Disease-Detection-System/
├── airflow/
│   ├── dags/
│   │   └── eye_ml_pipeline.py
│   ├── scripts/
│   │   ├── offline_eval.py
│   │   ├── load_test.py
│   │   ├── monitor.py
│   │   ├── run_evaluation.sh
│   │   └── run_retrain.sh
│   └── docker-compose-airflow.yaml
├── tf/
│   ├── main.tf, provider.tf, variables.tf, etc.
├── trainer/
│   ├── train.py
│   └── run_training.sh
├── docker/
│   ├── Dockerfile.train, Dockerfile.serve
├── model_serving/
├── monitoring/
│   └── docker-compose.yml
├── server/
├── Data-Pipelining/
└── README.md
```

---

## Automation Summary

| Stage                    | Tool             | Trigger           |
| ------------------------ | ---------------- | ----------------- |
| Provision Resources      | Terraform        | `terraform apply` |
| Mount Block Volume       | Terraform        | `terraform apply` |
| Model Training           | Docker + Airflow | DAG task          |
| Evaluation & Monitoring  | Docker Compose   | DAG task          |
| Image Build & Deployment | Airflow          | DAG task          |
| Load Testing             | Airflow          | DAG task          |

---

## Future Enhancements

* Add Prometheus alerting
* Push evaluation results to Grafana
* Add feedback-driven retraining (active learning)
