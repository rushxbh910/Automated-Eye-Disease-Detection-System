from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="model_evaluation_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    evaluate = BashOperator(
        task_id="evaluate_model",
        bash_command="bash /opt/airflow/scripts/evaluate_model.sh",
    )
