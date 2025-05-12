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

The raw dataset was first uploaded to the node using `scp` and then transferred to the object store via the following `docker-compose-etl-upload-raw.yaml` file:

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
docker compose -f docker-compose-etl-upload-raw.yaml up upload-raw-data
```

---

### 4. ETL Pipeline (Extract-Transform-Load)

The data transformation was handled by the following `transform.py` script. It reads raw images from the object store, applies resizing, normalization, and saves them into 5 stratified splits: train, test, holdout\_1, holdout\_2, holdout\_3.

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

This was orchestrated using the Docker Compose file:

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

