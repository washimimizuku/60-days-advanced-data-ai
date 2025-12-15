"""
Example Branching DAG - Conditional logic in Airflow
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule
import random

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'example_branching_dag',
    default_args=default_args,
    description='Example of conditional logic and branching',
    schedule_interval='@daily',
    catchup=False,
    tags=['example', 'branching', 'conditional'],
)

def check_data_volume(**context):
    """Decide which processing path to take based on data volume"""
    # Simulate checking data volume
    volume = random.randint(1, 1000)
    
    context['task_instance'].xcom_push(key='data_volume', value=volume)
    
    if volume > 500:
        return 'high_volume_processing'
    else:
        return 'normal_processing'

def normal_processing(**context):
    """Normal volume processing"""
    volume = context['task_instance'].xcom_pull(key='data_volume', task_ids='check_volume')
    print(f"Processing {volume} records with normal algorithm")
    return f"Processed {volume} records normally"

def high_volume_processing(**context):
    """High volume processing with optimization"""
    volume = context['task_instance'].xcom_pull(key='data_volume', task_ids='check_volume')
    print(f"Processing {volume} records with optimized algorithm")
    return f"Processed {volume} records with optimization"

def final_step(**context):
    """Final processing step that runs regardless of branch"""
    print("Final processing step completed")
    return "Pipeline completed"

# Tasks
start = DummyOperator(task_id='start', dag=dag)

check_volume = BranchPythonOperator(
    task_id='check_volume',
    python_callable=check_data_volume,
    dag=dag,
)

normal_proc = PythonOperator(
    task_id='normal_processing',
    python_callable=normal_processing,
    dag=dag,
)

high_volume_proc = PythonOperator(
    task_id='high_volume_processing',
    python_callable=high_volume_processing,
    dag=dag,
)

# Join task that runs after either branch
join = DummyOperator(
    task_id='join',
    trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED,
    dag=dag,
)

final = PythonOperator(
    task_id='final_step',
    python_callable=final_step,
    dag=dag,
)

# Dependencies
start >> check_volume
check_volume >> [normal_proc, high_volume_proc]
[normal_proc, high_volume_proc] >> join >> final