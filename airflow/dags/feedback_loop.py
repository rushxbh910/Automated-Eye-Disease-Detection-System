from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="model_training_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    train = BashOperator(
        task_id="train_model",
        bash_command="bash /opt/airflow/scripts/train_model.sh",
    )
