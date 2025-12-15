from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.task_group import TaskGroup
import pandas as pd
import os

default_args = {
    'owner': 'data_engineering_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'production_data_pipeline',
    default_args=default_args,
    description='Production data pipeline with quality and monitoring',
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=1,
    tags=['production', 'phase2-integration']
)

def ingest_data_source(**context):
    """Ingest data from source"""
    source = context['params']['source']
    
    # Generate sample data
    if source == 'customers':
        data = pd.DataFrame({
            'customer_id': range(1, 1001),
            'email': [f'user{i}@example.com' for i in range(1, 1001)],
            'first_name': [f'User{i}' for i in range(1, 1001)],
            'customer_segment': ['premium'] * 300 + ['standard'] * 400 + ['basic'] * 300
        })
    elif source == 'transactions':
        data = pd.DataFrame({
            'transaction_id': range(1, 5001),
            'customer_id': [i % 1000 + 1 for i in range(5000)],
            'amount': [100 + (i % 500) for i in range(5000)],
            'transaction_date': datetime.now()
        })
    
    print(f"Ingested {len(data)} records from {source}")
    return len(data)

def validate_data_quality(**context):
    """Validate data quality"""
    print("Running data quality validation...")
    # Simulate quality check
    quality_score = 0.98
    print(f"Data quality score: {quality_score}")
    return quality_score

def run_dbt_models(**context):
    """Run dbt models"""
    model_type = context['params']['model_type']
    print(f"Running dbt {model_type} models...")
    # Simulate dbt run
    return f"dbt {model_type} completed"

# Data Ingestion Task Group
with TaskGroup('data_ingestion', dag=dag) as ingestion_group:
    ingest_customers = PythonOperator(
        task_id='ingest_customers',
        python_callable=ingest_data_source,
        params={'source': 'customers'}
    )
    
    ingest_transactions = PythonOperator(
        task_id='ingest_transactions',
        python_callable=ingest_data_source,
        params={'source': 'transactions'}
    )

# Quality Validation Task Group
with TaskGroup('quality_validation', dag=dag) as quality_group:
    validate_quality = PythonOperator(
        task_id='validate_data_quality',
        python_callable=validate_data_quality
    )

# dbt Transformations Task Group
with TaskGroup('transformations', dag=dag) as transform_group:
    dbt_staging = PythonOperator(
        task_id='dbt_staging',
        python_callable=run_dbt_models,
        params={'model_type': 'staging'}
    )
    
    dbt_marts = PythonOperator(
        task_id='dbt_marts',
        python_callable=run_dbt_models,
        params={'model_type': 'marts'}
    )
    
    dbt_staging >> dbt_marts

# Pipeline dependencies
ingestion_group >> quality_group >> transform_group