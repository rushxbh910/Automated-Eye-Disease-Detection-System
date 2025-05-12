from airflow import DAG
from airflow.operators.bash import BashOperator, PythonOperator
from datetime import datetime
import subprocess

def drift_check():
    result = subprocess.run(["python3", "/opt/airflow/scripts/check_drift.py"], capture_output=True, text=True)
    if "DRIFT" in result.stdout:
        raise Exception("Drift detected - stopping deployment.")

with DAG(
    dag_id="model_deploy_dag",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    drift = PythonOperator(
        task_id="check_drift",
        python_callable=drift_check,
    )

    deploy = BashOperator(
        task_id="deploy_model",
        bash_command="bash /opt/airflow/scripts/deploy_model.sh",
    )

    drift >> deploy
