# Project Report: Automated Eye Disease Detection System 

## Data Pipeline Setup and Dashboard


---

### 1. Object Store Setup

The object store was created on the CHI\@TACC site using the **Chameleon GUI**. A container named `object-persist-project24` was created to hold datasets.

---

### 2. Block Storage Setup

Block storage was provisioned on KVM\@TACC using the **Chameleon GUI**. A volume named `block-persist-project24` was created and attached to the persistent KVM\@TACC node. Once attached, it was formatted and mounted:

```bash
sudo mkfs.ext4 /dev/vdb
sudo mkdir -p /mnt/block
sudo mount /dev/vdb /mnt/block
```

---

### 3. Loading Raw Data into the Object Store

The raw dataset was first uploaded to the node using `scp` and then transferred to the object store via the following `docker-compose-etl-upload-raw.yaml` file which is present in the `Data-Pipelining` folder of this repository:

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

This was run with:

```bash
docker compose -f docker-compose-etl-upload-raw.yaml up load-raw-data
```

---

### 4. ETL Pipeline (Extract-Transform-Load)

The data transformation was handled by the following `transform.py` script which is present in the `streamlit-dashboard` folder in the 'Data-Pipelining` folder of this repository. It reads raw images from the object store, applies resizing, normalization, and saves them into 5 stratified splits: train, test, holdout\_1, holdout\_2, holdout\_3.

```python
import os
import shutil
from sklearn.model_selection import StratifiedShuffleSplit
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset
import torch
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define paths
RAW_DATA_DIR = "/data/raw_eye_dataset"
OUTPUT_DIR = "/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define image transformations
base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Load dataset
dataset = ImageFolder(RAW_DATA_DIR, transform=base_transform)
labels = [label for _, label in dataset.samples]

# Split into train/test (80/20 stratified)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))

train_set = Subset(dataset, train_idx)
test_set = Subset(dataset, test_idx)

# Further split test set into 3 equal holdout subsets (stratified)
holdout_labels = [labels[i] for i in test_idx]
sss_holdout = StratifiedShuffleSplit(n_splits=3, test_size=1/3, random_state=42)
holdout_splits = list(sss_holdout.split(np.zeros(len(holdout_labels)), holdout_labels))

holdouts = []
used_indices = set()
for i, (train_h, test_h) in enumerate(holdout_splits):
    # Avoid overlaps
    new_indices = [j for j in test_h if j not in used_indices]
    used_indices.update(new_indices)
    subset_indices = [test_idx[j] for j in new_indices]
    holdouts.append(Subset(dataset, subset_indices))

# Save function
def save_subset(subset, output_path):
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    for i, (img, label) in enumerate(loader):
        class_dir = os.path.join(output_path, dataset.classes[label.item()])
        os.makedirs(class_dir, exist_ok=True)
        save_path = os.path.join(class_dir, f"img_{i:05d}.png")
        save_image(img, save_path)

# Save all splits
save_subset(train_set, os.path.join(OUTPUT_DIR, "train"))
save_subset(test_set, os.path.join(OUTPUT_DIR, "test"))
for i, holdout in enumerate(holdouts):
    save_subset(holdout, os.path.join(OUTPUT_DIR, f"holdout_{i+1}"))

print("✅ Data transformed and saved into:")
print(f"- {OUTPUT_DIR}/train")
print(f"- {OUTPUT_DIR}/test")
print(f"- {OUTPUT_DIR}/holdout_1, holdout_2, holdout_3")
```

This was orchestrated using the Docker Compose file which is the `docker-compose-etl.yaml` file in the `Data-Pipelining` folder of this repository:

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
      - /mnt/object:/mnt/object:ro  # Mounted object store (read-only)
    working_dir: /data
    command:
      - bash
      - -c
      - |
        set -e

        echo "Resetting local dataset directory..."
        rm -rf raw_eye_dataset
        mkdir -p raw_eye_dataset

        echo "Copying raw data from mounted object store to container volume..."
        cp -r /mnt/object/raw_eye_dataset/* raw_eye_dataset/

        echo "Contents of /data after extract stage:"
        ls -l /data/raw_eye_dataset

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
        echo "Installing Python dependencies..."
        pip install torch torchvision scikit-learn --break-system-packages

        echo "Running transform.py..."
        python3 transform.py

        echo "Listing contents of /data after transform:"
        ls -l /data

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
        if [ -z "$RCLONE_CONTAINER" ]; then
          echo "ERROR: RCLONE_CONTAINER is not set"
          exit 1
        fi

        echo "Uploading transformed dataset to object store..."

        rclone copy /data/processed chi_tacc:$RCLONE_CONTAINER/transformed_eye_dataset \
          --progress \
          --transfers=32 \
          --checkers=16 \
          --multi-thread-streams=4 \
          --fast-list

        echo "Upload complete. Listing remote contents:"
        rclone lsd chi_tacc:$RCLONE_CONTAINER/transformed_eye_dataset
```

The transformed output is stored back in the object store under:
`object-persist-project24/transformed_eye_dataset`

This was run with:

```bash
docker compose -f docker-compose-etl.yaml up
```

---

### 5. Streamlit Data Dashboard

An interactive Streamlit dashboard was built to explore the transformed dataset. The app reads from `/data/train`, `/data/test`, etc., and visualizes:

* Image counts per class
* Sample images from each class

Dockerfile:

```Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

requirements.txt:

```
streamlit
Pillow
matplotlib
```

app.py:

```python
import streamlit as st
from PIL import Image
import os
import random
import matplotlib.pyplot as plt

# Set the base path where your transformed dataset is mounted
DATA_PATH = "/data"

# Sidebar controls
st.sidebar.title("Dataset Viewer")
split = st.sidebar.selectbox("Select dataset", sorted(os.listdir(DATA_PATH)))
max_images = 5
num_images = st.sidebar.slider("Images per class", 1, max_images, 3)

# Title
st.title("Eye Disease Dataset Dashboard")

# === Class Distribution Bar Chart ===
split_path = os.path.join(DATA_PATH, split)
classes = sorted(os.listdir(split_path))
class_counts = {cls: len(os.listdir(os.path.join(split_path, cls))) for cls in classes}

st.subheader("Image Count per Class")
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(class_counts.keys(), class_counts.values(), color='skyblue')
ax.set_xlabel("Class")
ax.set_ylabel("Number of Images")
ax.set_title(f"Class Distribution in '{split}' Split")
plt.xticks(rotation=45, ha="right")
st.pyplot(fig)

# === Sample Image Display ===
st.subheader("Sample Images")

for cls in classes:
    cls_path = os.path.join(split_path, cls)
    image_files = os.listdir(cls_path)
    if not image_files:
        continue

    st.markdown(f"### {cls}")

    selected_images = random.sample(image_files, min(num_images, len(image_files)))
    cols = st.columns(min(len(selected_images), 5))  # Show up to 5 images in a row

    for col, img_name in zip(cols, selected_images):
        img_path = os.path.join(cls_path, img_name)
        image = Image.open(img_path)
        image.thumbnail((224, 224))
        col.image(image, caption=img_name, use_container_width=True)
```

Build and run:

```bash
docker build -t streamlit-eye-app .
docker run -d -p 8501:8501 \
  -v /mnt/object/transformed_eye_dataset:/data \
  --name streamlit_app \
  streamlit-eye-app
```

Access the dashboard at `http://<NODE_IP>:8501`

---

This end-to-end setup allows for a modular, observable, and visual pipeline from data ingestion to transformation and inspection.


---


Training Using MLflow and Ray
=============================

To enable persistent experiment tracking and model logging, MLflow was set up on a GPU-enabled node on Chameleon Cloud.

A. Node Provisioning
--------------------

A GPU instance was launched using the `CC-Ubuntu24.04-CUDA` image from the Chameleon GUI (CHI@TACC site). A floating IP was associated for remote access.

Provisioning script using the Chameleon Python SDK:

    from chi import server, context, lease

    context.choose_project()
    context.choose_site(default="CHI@TACC")

    l = lease.get_lease("Project24001")
    s = server.Server("node-sb9880-01", reservation_id=l.node_reservations[0]["id"], image_name="CC-Ubuntu24.04-CUDA")

    s.submit(idempotent=True)
    s.associate_floating_ip()
    s.refresh()
    s.check_connectivity()

B. Security Group Configuration
-------------------------------

OpenStack security groups were configured to allow traffic to key services:

- SSH (22)
- Jupyter (8888)
- MLflow (8000)
- MinIO (9000, 9001)

Security rules were added using `os_conn.create_security_group()` and attached to the server.

C. Docker GPU Driver Setup
--------------------------

Required drivers and Docker setup were installed for GPU support.

D. Docker Images for MLflow + Jupyter
-------------------------------------

Custom Dockerfiles were used for setup:

- Dockerfile.jupyter-torch-mlflow-rocm
- Dockerfile.ray-rocm

E. Rclone Configuration for Object Storage
------------------------------------------

1. Start `rclone config`
2. Add a config file using Application Credentials
3. Mount the object store to the node

F. Docker Volume Setup
----------------------

Create a named Docker volume (e.g., `EYE`) to persist and share data between containers.

G. Running All Services with Docker Compose
-------------------------------------------

Use the following compose file to run all services:

- docker-compose-ray-mlflow-data.yaml

H. Launch Jupyter and Connect to Ray & MLflow
---------------------------------------------

    HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)

    docker run -d --rm -p 8888:8888 \
        -v ~/workspace:/home/jovyan/work/ \
        -v EYE:/mnt/ \
        -e RAY_ADDRESS=http://${HOST_IP}:8265/ \
        -e MLFLOW_TRACKING_URI=http://${HOST_IP}:8000/ \
        -e EYE_DATA_DIR=/mnt/data/transformed_eye_dataset \
        --name jupyter \
        jupyter-ray

Image ID used:

    c907ecd9d31828ffce7d3840739f999ca42de8cab5c29897ad63ae68a30fd766

I. Submit Jobs to Ray Cluster
-----------------------------

    ray job submit \
      --runtime-env runtime.json \
      --entrypoint-num-gpus 1 \
      --entrypoint-num-cpus 8 \
      --verbose \
      --working-dir code/ \
      -- python ray-code00.py

Important:
----------

Make sure your dataset is NOT inside the working directory sent with `--working-dir`, as Ray will package and send all files in that directory to workers.
Workers already have access to data via the Docker volume, so duplicating it is unnecessary and inefficient.

---

### 7. Serving and Monitoring

This system deploys a DenseNet121-based eye disease classifier via FastAPI in multiple environments, tracks predictions with MLflow, and monitors inference metrics using Prometheus + Grafana. Feedback loops are supported via MinIO and Label Studio.

#### FastAPI Application (fast_api.py)

* Loads densenet121_model.pth with 15 disease classes
* Partial layer freezing for optimized inference
* Prometheus metrics:

  * predictions_total, prediction_errors_total
  * inference_latency_seconds
  * model_confidence_score
  * drift_alert (based on confidence threshold)
* Logs confidence + predicted_class to MLflow
* Exposes:

  * POST /predict
  * GET /metrics

#### ML + Metrics

* Model: DenseNet121
* Classes: 15 (customized)
* Drift: confidence < 0.5 for 5 consecutive samples

---

#### Deployment: Docker Compose (Staging, Canary, Prod)

| Service         | Description                        | Port(s)        |
| --------------- | ---------------------------------- | -------------- |
| eye-fastapi-* | FastAPI serving environments       | 8000         |
| mlflow        | ML tracking and experiment logging | 5000         |
| prometheus    | Metric scraper and alerting        | 9090         |
| grafana       | Metric dashboard                   | 3000         |
| minio         | Object storage for feedback data   | 9000, 9001 |
| labelstudio   | Human-in-the-loop annotation       | 8080         |

---

#### Monitoring Stack

##### Prometheus

* Scraping targets:

  * eye-fastapi-prod
  * eye-fastapi-staging
  * eye-fastapi-canary
* Common Error: server misbehaving if DNS isn't resolved. Use correct container_name and shared network.

##### Grafana

* Dashboards for:

  * Request rate
  * Confidence trend
  * Drift alert status
  * Inference latency per environment

---



---

##### Next Steps

* Add Traefik or Nginx for load balancing between environments
* Trigger Airflow pipelines using MinIO + Label Studio annotations
* Provision Grafana dashboards via mounted JSONs

---

### 8. Continous X

# Automated Eye Disease Detection System (MLOps)

This project implements a complete cloud-native MLOps pipeline for eye disease detection using PyTorch, Terraform, Docker, and Apache Airflow. It automates infrastructure provisioning, model training, evaluation, containerization, deployment, and monitoring.

---

## Project Overview

- *Model Training:* Deep learning model (PyTorch) for disease classification  
- *Data Storage:* Mounted block storage for persistent access across steps  
- *Workflow Orchestration:* Apache Airflow DAG pipeline (scheduled retraining, staged deployment)   
- *Experiment Tracking:* MLflow  

---

## Infrastructure Provisioning (Terraform)

Terraform provisions compute resources and networking across *KVM@TACC* on Chameleon Cloud.

### Provisioned Resources

| Component       | Cloud Site  | Flavor     | Node Name                      | Floating IP |
|----------------|-------------|------------|--------------------------------|-------------|
| Training Node   | KVM@TACC    | m1.medium  | eye-train-train-1-rb5726     | yes         |
| Serving Node    | KVM@TACC    | m1.large   | eye-serve-serve-1-rb5726     | yes          |

### Block Storage

- *Volume Name:* block-persist-project24  
- *ID:* 7153c651-2757-45f9-99e0-9cbbace429f5  
- *Mounted at:* /mnt/block on the training node  
- *Usage:* model.pth and training data are persisted here  

Terraform automatically attached and mounted this volume using a cloud-init script inside user_data_mount.sh.

---

## MLOps DAG (Apache Airflow)

Airflow automates all stages of the ML lifecycle via eye_disease_ml_pipeline.py. Runs daily and retrains every 3 days.

### 📊 DAG Stages

1. *Train Model* (train_model.sh inside Docker)
2. *Evaluate Offline* (test_model.py, launched via Docker Compose)
3. *Monitor Performance* (monitor.py)

---

## Docker Services

Docker Compose launches the following services on the persistent node:

| Service        | Port   | Description                              |
|----------------|--------|------------------------------------------|
| FastAPI        | 8000   | API interface for predictions            |
| Prometheus     | 9090   | Metrics collection                       |
| Grafana        | 3000   | Metrics dashboard                        |
| MLflow         | 5000   | Experiment tracking                      |
| MinIO          | 9000   | Object storage backend                   |
| Label Studio   | 8080   | Human labeling interface                 |
| Airflow Web UI | 3001   | DAG management and execution interface   |

---

## Folder Structure

bash

Automated-Eye-Disease-Detection-System/

├── airflow/

│   ├── dags/

│   │   └── eye_ml_pipeline.py

│   ├── scripts/

│   │   ├── offline_eval.py

│   │   ├── load_test.py

│   │   ├── monitor.py

│   │   ├── run_evaluation.sh

│   │   └── run_retrain.sh

│   └── docker-compose-airflow.yaml

├── tf/

│   ├── main.tf, provider.tf, outputs.tf, variables.tf, terraform.tfvars, etc.

├── trainer/

│   ├── train.py

│   └── run_training.sh

├── docker/

│   ├── Dockerfile.train, Dockerfile.serve

├── model_serving/

├── monitoring/

│   ├── docker-compose.yml

├── server/

├── Data-Pipelining/

└── README.md


---

## What's Automated?

| Stage                        | Tool           | Triggered By            |
|-----------------------------|----------------|--------------------------|
| VM + IP Provisioning        | Terraform      | terraform apply        |
| Volume Mounting             | Terraform      | terraform apply        |
| Model Training              | Docker + Airflow | DAG task              |
| Evaluation & Monitoring     | Docker Compose | DAG task                 |
| Image Build & Deployment    | Airflow        | DAG task                 |
| Load Testing & Promotion    | Airflow        | DAG task                 |

---
 

---

## Final Setup Recap

- Training: KVM@TACC with volume mount /mnt/block
- Serving: KVM@TACC with floating IP
- DAG triggers retrain → evaluation 
- Monitoring 
- Persistent node used to host containers for reproducible results

---

## Future Improvements

- Add Prometheus alerting rules  
- Push evaluation metrics to Grafana automatically  
- Expand feedback loop to include active learning
