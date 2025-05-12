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


