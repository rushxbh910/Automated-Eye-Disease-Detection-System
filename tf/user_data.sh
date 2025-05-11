#!/bin/bash
sudo mkfs.ext4 /dev/vdb || true
sudo mkdir -p /mnt/training_data
sudo mount /dev/vdb /mnt/training_data
echo "/dev/vdb /mnt/training_data ext4 defaults 0 0" | sudo tee -a /etc/fstab