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



