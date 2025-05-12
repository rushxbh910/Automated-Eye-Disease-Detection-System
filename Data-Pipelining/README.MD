###### 🔧 Environment Setup (so far)

Transfer data to chameleon node:
   ```
   scp -i ~/.ssh/id_rsa_chameleon /Users/vaibhavrouduri/EYE/Artifacts/04_29_2025_18_35_39/data_ingestion/ingested/transformed_data.zip cc@129.114.27.186:/home/cc/
   ```

1. SSH into the VM:
   ```bash
   ssh -i ~/.ssh/id_rsa_chameleon cc@<floating-IP>

2. **Install rclone:**
   ```
   curl https://rclone.org/install.sh | sudo bash
   ```

3. **Modify FUSE permissions:**
   ```
   sudo sed -i '/^#user_allow_other/s/^#//' /etc/fuse.conf
   ```

4. **Create rclone.conf:**

   File path: `~/.config/rclone/rclone.conf`

   ```
   [chi_tacc]
   type = swift
   user_id = YOUR_USER_ID
   application_credential_id = YOUR_CRED_ID
   application_credential_secret = YOUR_CRED_SECRET
   auth = https://chi.tacc.chameleoncloud.org:5000/v3
   region = CHI@TACC
   ```

5. **Test object store access:**
   ```
   rclone lsd chi_tacc:
   ```

6. **Docker ETL Upload (Minimal Version)**
The file docker/docker-compose-etl-upload.yaml contains a minimal service that:

Mounts preprocessed EYE dataset (~/transformed_data)

Uploads it to Chameleon's object store under object-persist-project24

To run:

export RCLONE_CONTAINER=object-persist-project24
docker compose -f docker/docker-compose-etl-upload.yaml run \
  -v ~/transformed_data:/data:ro \
  load-data

7. **Docker ETL Upload: Raw EYE Dataset**
The file docker/docker-compose-etl-upload-raw.yaml contains a minimal service that:

Mounts the raw EYE dataset directory (~/original_dataset)

Uploads its contents to Chameleon's object store under the container object-persist-project24

To run:

export RCLONE_CONTAINER=object-persist-project24
docker compose -f docker/docker-compose-etl-upload-raw.yaml run \
  -v ~/original_dataset:/data:ro \
  load-raw-data


### For Teammates: How to Mount Object Store

To access shared data on any VM, do the following:

1. Copy your `rclone.conf` into `~/.config/rclone/rclone.conf`
2. Run:
    ```bash
    sudo mkdir -p /mnt/object
    sudo chown -R cc /mnt/object
    sudo chgrp -R cc /mnt/object
    rclone mount chi_tacc:object-persist-project24 /mnt/object --read-only --allow-other --daemon
    ```

Now your data will be available under `/mnt/object`, ready for training.


Block Storage Setup (Artifacts Persistence)

Step 1: Format and Mount the Volume

Only needed the first time (dont need to do it, already done)

sudo parted -s /dev/vdb mklabel gpt
sudo parted -s /dev/vdb mkpart primary ext4 0% 100%
sudo mkfs.ext4 /dev/vdb1

Mount and configure

sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block
sudo chown -R cc /mnt/block
sudo chgrp -R cc /mnt/block

Step 2: Use /mnt/block to Store:
Model checkpoints

MLflow tracking logs

Docker volumes (Postgres/MinIO)


To verify:

echo "Block storage test successful" > /mnt/block/test.txt
cat /mnt/block/test.txt

Handoff Summary for Teammates
Training data: Available at /mnt/object

Artifacts directory: Save anything persistent to /mnt/block

To re-attach /mnt/block on new VMs:

sudo mkdir -p /mnt/block
sudo mount /dev/vdb1 /mnt/block


MLflow Infrastructure for Artifact Tracking
We use MLflow for experiment tracking, running as a containerized service using Docker Compose. Artifacts (like trained models) are stored in MinIO (S3-compatible object storage), and metrics/params are logged to a PostgreSQL database — both backed by persistent block storage on Chameleon.

🚀 Running the MLflow Infrastructure
Make sure your block storage volume is already attached and mounted on /mnt/block.

Then run:


cd ~/eye-upload/mlflow-infra
HOST_IP=$(curl --silent http://169.254.169.254/latest/meta-data/public-ipv4)
docker compose -f docker-compose-block.yaml up -d
🔍 Accessing the Services
Once all services are up:

MLflow Tracking UI → http://<your-floating-ip>:8000

MinIO Console → http://<your-floating-ip>:9001

Jupyter Notebook → http://<your-floating-ip>:8888

Replace <your-floating-ip> with the actual public IP of your compute instance.

File Structure
Your docker-compose-block.yaml should already be located at:


Data-Pipelining/docker-compose-block.yaml
This sets up:

postgres container (metrics and params)

minio container (artifact storage)

mlflow container (the MLflow tracking server)

jupyter container (to run experiments using MLflow)





**Project Report: Automated Eye Disease Detection System - Data Pipeline Setup and Dashboard**

This report details the steps taken to build the data processing pipeline, including object and block storage setup, data ingestion and transformation, and an interactive dashboard for data exploration.

---

### 1. Object Store Setup

The object store was created on the CHI\@TACC site using the **Chameleon GUI**. No CLI commands were used to provision the object store. A container named `object-persist-project24` was created to hold datasets.

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

The raw dataset was first uploaded to the node using `scp` and then transferred to the object store via the following `docker-compose-etl-upload.yaml` file:

```yaml
services:
  upload-raw-data:
    image: rclone/rclone:latest
    volumes:
      - ~/.config/rclone/rclone.conf:/root/.config/rclone/rclone.conf:ro
      - ./raw_eye_dataset:/data
    command: copy /data chi_tacc:object-persist-project24/raw_eye_dataset
```

This was run with:

```bash
docker compose -f docker-compose-etl-upload.yaml up upload-raw-data
```

---

### 4. ETL Pipeline (Extract-Transform-Load)

The data transformation was handled by the following `transform.py` script. It reads raw images from the object store, applies resizing, normalization, and saves them into 5 stratified splits: train, test, holdout\_1, holdout\_2, holdout\_3.

```python
from torchvision.datasets import ImageFolder
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset, DataLoader
from torchvision.utils import save_image
import numpy as np, os, torch

RAW_DATA_DIR = "/data/raw_eye_dataset"
OUTPUT_DIR = "/data/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

base_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

dataset = ImageFolder(RAW_DATA_DIR, transform=base_transform)
labels = [label for _, label in dataset.samples]

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(sss.split(np.zeros(len(labels)), labels))
train_set = Subset(dataset, train_idx)
test_set = Subset(dataset, test_idx)

# Holdouts
holdout_labels = [labels[i] for i in test_idx]
sss_holdout = StratifiedShuffleSplit(n_splits=3, test_size=1/3, random_state=42)
holdouts, used = [], set()
for i, (_, test_h) in enumerate(sss_holdout.split(np.zeros(len(holdout_labels)), holdout_labels)):
    new_idx = [j for j in test_h if j not in used]
    used.update(new_idx)
    holdouts.append(Subset(dataset, [test_idx[j] for j in new_idx]))

def save_subset(subset, path):
    loader = DataLoader(subset, batch_size=1, shuffle=False)
    for i, (img, label) in enumerate(loader):
        cls_dir = os.path.join(path, dataset.classes[label.item()])
        os.makedirs(cls_dir, exist_ok=True)
        save_image(img, os.path.join(cls_dir, f"img_{i:05d}.png"))

save_subset(train_set, os.path.join(OUTPUT_DIR, "train"))
save_subset(test_set, os.path.join(OUTPUT_DIR, "test"))
for i, holdout in enumerate(holdouts):
    save_subset(holdout, os.path.join(OUTPUT_DIR, f"holdout_{i+1}"))
```

This was orchestrated using the Docker Compose file:

```yaml
services:
  transform-data:
    image: python:3.11
    container_name: etl_transform_data
    volumes:
      - eye_data:/data
      - ./scripts/transform.py:/data/transform.py:ro
    working_dir: /data
    command: bash -c "pip install torch torchvision scikit-learn --break-system-packages && python3 transform.py"

volumes:
  eye_data:
```

The transformed output is stored back in the object store under:
`object-persist-project24/transformed_eye_dataset`

---

### 5. Streamlit Data Dashboard

An interactive Streamlit dashboard was built to explore the transformed dataset. The app reads from `/data/train`, `/data/test`, etc., and visualizes:

* Image counts per class
* Sample images from each class

Dockerfile:

```Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

requirements.txt:

```
streamlit
matplotlib
Pillow
torch
torchvision
```

app.py (abbreviated):

```python
import streamlit as st
import os
import matplotlib.pyplot as plt
from PIL import Image

def show_class_distribution(data_split):
    folder = f"/data/{data_split}"
    class_counts = {cls: len(os.listdir(os.path.join(folder, cls))) for cls in os.listdir(folder)}
    st.bar_chart(class_counts)

st.title("Eye Dataset Viewer")
split = st.selectbox("Select Dataset Split", ["train", "test", "holdout_1"])
show_class_distribution(split)
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

