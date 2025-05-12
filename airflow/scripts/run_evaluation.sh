#!/bin/bash

set -e
cd "$(dirname "$0")/../../docker"
docker compose -f docker-compose.yml up -d

echo "Evaluation services started successfully."