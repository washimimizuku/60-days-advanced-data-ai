"""
Example ETL DAG for Day 12 - Basic Airflow patterns
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
import pandas as pd
import logging

# Default arguments
default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,  # Set to True in production
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'example_etl_dag',
    default_args=default_args,
    description='Simple ETL pipeline example',
    schedule_interval='@daily',
    catchup=False,
    tags=['example', 'etl', 'tutorial'],
)

def extract_data(**context):
    """Extract sample data"""
    logging.info("Starting data extraction...")
    
    # Mock data extraction
    data = {
        'customers': [
            {'id': 1, 'name': 'Alice', 'email': 'alice@example.com'},
            {'id': 2, 'name': 'Bob', 'email': 'bob@example.com'},
            {'id': 3, 'name': 'Charlie', 'email': 'charlie@example.com'},
        ]
    }
    
    # Push to XCom
    context['task_instance'].xcom_push(key='raw_data', value=data)
    logging.info(f"Extracted {len(data['customers'])} customer records")
    return data

def transform_data(**context):
    """Transform the extracted data"""
    logging.info("Starting data transformation...")
    
    # Pull from XCom
    ti = context['task_instance']
    raw_data = ti.xcom_pull(key='raw_data', task_ids='extract')
    
    # Transform data
    df = pd.DataFrame(raw_data['customers'])
    df['email_domain'] = df['email'].str.split('@').str[1]
    df['processed_at'] = datetime.now().isoformat()
    
    transformed_data = df.to_dict('records')
    
    # Push transformed data
    ti.xcom_push(key='transformed_data', value=transformed_data)
    logging.info(f"Transformed {len(transformed_data)} records")
    return transformed_data

def load_data(**context):
    """Load data to destination"""
    logging.info("Starting data load...")
    
    # Pull transformed data
    ti = context['task_instance']
    data = ti.xcom_pull(key='transformed_data', task_ids='transform')
    
    # Mock data loading
    logging.info(f"Loading {len(data)} records to database...")
    
    # In production, this would write to actual database
    for record in data:
        logging.info(f"Loaded: {record}")
    
    load_result = {
        'records_loaded': len(data),
        'load_time': datetime.now().isoformat(),
        'status': 'success'
    }
    
    ti.xcom_push(key='load_result', value=load_result)
    return load_result

# Define tasks
extract_task = PythonOperator(
    task_id='extract',
    python_callable=extract_data,
    dag=dag,
)

transform_task = PythonOperator(
    task_id='transform',
    python_callable=transform_data,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load',
    python_callable=load_data,
    dag=dag,
)

# Cleanup task
cleanup_task = BashOperator(
    task_id='cleanup',
    bash_command='echo "Cleaning up temporary files..." && echo "Cleanup complete"',
    dag=dag,
)

# Optional notification (disabled by default)
# notify_task = EmailOperator(
#     task_id='notify',
#     to=['team@example.com'],
#     subject='ETL Pipeline Complete - {{ ds }}',
#     html_content='<p>ETL pipeline completed successfully for {{ ds }}</p>',
#     dag=dag,
# )

# Set dependencies
extract_task >> transform_task >> load_task >> cleanup_task