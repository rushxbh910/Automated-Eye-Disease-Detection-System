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
