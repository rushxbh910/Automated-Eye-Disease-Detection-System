from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.trigger_rule import TriggerRule
from docker.types import Mount
import subprocess
import sys

def offline_eval_callable():
    subprocess.run(["docker", "compose", "-f", "/mnt/block/airflow/docker-compose.yml", "up", "-d", "eye-fastapi"], check=True)

def load_test_callable():
    subprocess.run([sys.executable, '/mnt/block/airflow/dags/scripts/load_test.py'], check=True)

def monitor_callable():
    subprocess.run([sys.executable, '/mnt/block/airflow/dags/scripts/monitor.py'], check=True)

def_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'eye_disease_pipeline',
    default_args=def_args,
    description='Cloud-native MLOps pipeline for Eye Disease Detection',
    schedule_interval=timedelta(days=3),
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    log_model = DockerOperator(
        task_id='log_model',
        image='eye-disease-trainer:latest',
        command='python train_model.py',
        working_dir='/app',
        mounts=[
            Mount(source='/mnt/block', target='/app', type='bind'),
        ],
        environment={'PYTHONPATH': '/app'}
    )

    offline_eval = PythonOperator(
        task_id='offline_eval',
        python_callable=offline_eval_callable
    )

    build_container = DockerOperator(
        task_id='build_container',
        image='docker:24.0.2',
        command='docker build -t eye-disease-serving:latest .',
        working_dir='/mnt/block/deployment',
        auto_remove=True,
        mounts=[
            Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind'),
            Mount(source='/mnt/block/deployment', target='/mnt/block/deployment', type='bind')
        ]
    )

    deploy_staging = DockerOperator(
        task_id='deploy_staging',
        image='docker:24.0.2',
        auto_remove=True,
        command='sh -c "docker rm -f eye-staging || true && docker run -d --rm --name eye-staging -p 8501:8500 eye-disease-serving:latest"',
        mounts=[Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind')]
    )

    load_test = PythonOperator(
        task_id='load_test',
        python_callable=load_test_callable
    )

    deploy_canary = DockerOperator(
        task_id='deploy_canary',
        image='docker:24.0.2',
        auto_remove=True,
        command='sh -c "docker rm -f eye-canary || true && docker run -d --rm --name eye-canary -p 8601:8500 eye-disease-serving:latest"',
        mounts=[Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind')]
    )

    monitor = PythonOperator(
        task_id='monitor',
        python_callable=monitor_callable
    )

    deploy_prod = DockerOperator(
        task_id='deploy_prod',
        image='docker:24.0.2',
        auto_remove=True,
        command='sh -c "docker rm -f eye-prod || true && docker run -d --rm --name eye-prod -p 8701:8500 eye-disease-serving:latest"',
        mounts=[Mount(source='/var/run/docker.sock', target='/var/run/docker.sock', type='bind')],
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    log_model >> offline_eval >> build_container >> deploy_staging >> load_test >> deploy_canary >> monitor >> deploy_prod