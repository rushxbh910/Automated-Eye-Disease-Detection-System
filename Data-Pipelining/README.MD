## Project Report: Automated Eye Disease Detection System - Data Pipeline Setup and Dashboard

This report details the steps taken to build the data processing pipeline, including object and block storage setup, data ingestion and transformation, and an interactive dashboard for data exploration.

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

### 6. MLFlow Persistence Setup on Chameleon Cloud (Teammate Contribution)

To enable persistent experiment tracking and model logging, MLFlow was set up on a GPU-enabled node on Chameleon Cloud.

#### a. Node Provisioning

A GPU instance was launched using the `CC-Ubuntu24.04-CUDA` image from the Chameleon GUI (CHI\@TACC site). A floating IP was associated for remote access.

```python
from chi import server, context, lease
context.choose_project()
context.choose_site(default="CHI@TACC")
l = lease.get_lease("Project24001")
s = server.Server("node-sb9880-01", reservation_id=l.node_reservations[0]["id"], image_name="CC-Ubuntu24.04-CUDA")
s.submit(idempotent=True)
s.associate_floating_ip()
s.refresh()
s.check_connectivity()
```

#### b. Security Group Configuration

OpenStack security groups were configured to allow traffic to key services:

* SSH (22)
* Jupyter (8888)
* MLFlow (8000)
* MinIO (9000, 9001)

Security rules were added via `os_conn.create_security_group()` and attached to the server.

#### c. Docker and GPU Setup

Docker was installed with:

```bash
curl -sSL https://get.docker.com/ | sudo sh
sudo usermod -aG docker $USER
```

For NVIDIA GPUs:

```bash
# NVIDIA container runtime setup
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

For AMD GPUs:

```bash
wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb
sudo apt install ./amdgpu-install_6.3.60303-1_all.deb
amdgpu-install -y --usecase=dkms
sudo usermod -aG video,render $USER
```

#### d. Docker Image for MLFlow + Jupyter

A custom image `jupyter-mlflow` was built from a Dockerfile with PyTorch, CUDA, Jupyter, and MLFlow preinstalled.

```bash
docker build -t jupyter-mlflow -f Dockerfile.jupyter-torch-mlflow-cuda .
```

#### e. Rclone Configuration for Object Storage

Rclone was used to mount the CHI\@TACC object store to `/mnt/object`:

```bash
rclone config  # (manual step to set chi_tacc remote)
rclone mount chi_tacc:object-persist-project24 /mnt/object --allow-other --daemon
```

Rclone config snippet:

```
[chi_tacc]
type = swift
user_id = <your-user-id>
application_credential_id = <cred-id>
application_credential_secret = <cred-secret>
auth = https://chi.tacc.chameleoncloud.org:5000/v3
region = CHI@TACC
```

---


This system deploys a DenseNet121-based eye disease classifier via FastAPI in multiple environments, tracks predictions with MLflow, and monitors inference metrics using Prometheus + Grafana. Feedback loops are supported via MinIO and Label Studio.

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

