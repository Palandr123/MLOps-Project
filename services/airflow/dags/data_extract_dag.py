from pendulum import datetime
import os

from airflow import DAG
from airflow import AirflowException
from airflow.decorators import dag
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from data import sample_data, validate_initial_data


PROJECT_HOME = os.environ['PROJECT_HOME']
os.chdir(PROJECT_HOME)

with DAG(dag_id="extract_data",
         start_date=datetime(2024, 6, 29, tz="UTC"),
         schedule="*/10 * * * *",
         catchup=False) as dag:
    
    def sample_step():
        return_val = sample_data()
        if return_val is None:
            raise AirflowException("Failed to sample data")
        return "success"

    called_sample = PythonOperator(task_id="sample", 
                         python_callable=sample_step)

    def validate_step():
        return_val = validate_initial_data()
        if not return_val:
            raise AirflowException("Failed to validate data sample")
        return "success"
    
    called_validate = PythonOperator(task_id="validate", 
                         python_callable=validate_step)

    version_data_command = "./scripts/test_data.sh "
    if os.path.exists(version_data_command.strip()):
        version_step = BashOperator(
                task_id= 'version_data',
                # TODO: use config
                bash_command=f"cd $PROJECT_HOME && {version_data_command}",
                dag=dag
        )
    else:
        raise Exception(f"Cannot locate {version_data_command}")

    called_sample >> called_validate >> version_step
