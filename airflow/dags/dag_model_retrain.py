from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

with DAG(
    dag_id='model_retrain_pipeline',
    default_args=default_args,
    start_date=datetime(2025, 5, 10),
    schedule_interval='@daily',
    catchup=False
) as dag:
    retrain = BashOperator(
        task_id='run_training',
        bash_command='bash /home/cc/Automated-Eye-Disease-Detection-System/trainer/run_training.sh'
    )

    optimize = BashOperator(
        task_id='onnx_export',
        bash_command='python3 /home/cc/Automated-Eye-Disease-Detection-System/model_serving/export_to_onnx.py'
    )

    retrain >> optimize
