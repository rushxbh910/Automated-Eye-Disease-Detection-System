#!/bin/bash
cd $HOME/Automated-Eye-Disease-Detection-System
docker compose -f docker-compose-ray-mlflow-data.yaml up --build -d
