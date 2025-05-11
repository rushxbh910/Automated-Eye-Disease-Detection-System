from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

def trigger_label_tasks(**context):
    # your custom logic to move sample data to MinIO and Label Studio
    print("Triggering label studio task generation...")
    os.system("python3 ./tools/drift_check.py")  # your own sampling/check script

with DAG(
    dag_id='data_ingestion_trigger',
    default_args=default_args,
    start_date=datetime(2025, 5, 10),
    schedule_interval='@daily',
    catchup=False
) as dag:
    start_label_pipeline = PythonOperator(
        task_id='start_label_sampling',
        python_callable=trigger_label_tasks
    )
