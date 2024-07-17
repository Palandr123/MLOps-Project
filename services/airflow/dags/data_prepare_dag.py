from pendulum import datetime
from datetime import timedelta

from airflow import DAG
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.sensors.external_task import ExternalTaskSensor


with DAG(dag_id="prepare_data",
         start_date=datetime(2024, 6, 29, tz="UTC"),
         schedule="*/10 * * * *",
         catchup=False) as dag:
    
    sensor = ExternalTaskSensor(
        task_id='wait_for_data_extract',
        external_dag_id='extract_data',
        execution_delta=timedelta(hours=0),
        dag=dag,
    )

    data_prepare_command = "python ./pipelines/data_prepare.py "
    data_prepare = BashOperator(
        task_id= 'version_data',
        # TODO: use config
        bash_command=f"cd $PROJECT_HOME && {data_prepare_command}",
        dag=dag
    )

    sensor >> data_prepare
