from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='train_model_pipeline',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    train = BashOperator(
        task_id='train_model',
        bash_command='bash /opt/airflow/scripts/run_training.sh'
    )

    train