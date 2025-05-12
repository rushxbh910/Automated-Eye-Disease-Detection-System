import subprocess

subprocess.run([
    "docker", "compose", "-f", "/mnt/block/airflow/docker-compose.yml", "up", "-d", "eye-fastapi"
], check=True)
