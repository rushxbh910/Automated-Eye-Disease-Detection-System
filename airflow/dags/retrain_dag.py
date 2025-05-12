from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

with DAG(
    dag_id='retrain_model',
    start_date=datetime.today(),
    schedule_interval='0 0 */3 * *',  # Every 3 days
    catchup=False,
) as dag:
    
    retrain = BashOperator(
        task_id='run_retrain',
        bash_command='bash /opt/airflow/scripts/run_retrain.sh'
    )
