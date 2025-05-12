#!/bin/bash
cd ~/Automated-Eye-Disease-Detection-System
python3 train.py
cp ./models/model.pth /mnt/block-storage/model.pth  # Assuming block is mounted here