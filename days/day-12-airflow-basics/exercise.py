"""
Day 12: Apache Airflow Basics - Exercise
Build comprehensive production ETL pipelines with Airflow
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule
import json
import logging
import pandas as pd
from typing import Dict, List, Any

# TODO: Exercise 1 - Configure Production DAG Settings
def create_production_dag():
    """
    TODO: Create a production-ready DAG with proper configuration
    
    Requirements:
    1. Set appropriate default_args with:
       - Owner information
       - Start date
       - Email notifications
       - Retry configuration
       - Timeout settings
    2. Configure DAG with:
       - Meaningful dag_id
       - Description
       - Schedule interval (daily at 2 AM)
       - Catchup disabled
       - Tags for organization
    3. Add proper documentation
    
    Test scenarios:
    - Daily scheduling
    - Error handling
    - Email notifications
    - Task timeouts
    """
    
    # TODO: Define comprehensive default_args
    default_args = {
        'owner': 'data_engineering_team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email': ['alerts@company.com'],
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'execution_timeout': timedelta(hours=2),
        # TODO: Add more production settings
        # TODO: Add failure callbacks
    }
    
    # TODO: Create DAG with production settings
    dag = DAG(
        dag_id='production_customer_etl',
        default_args=default_args,
        description='Production ETL pipeline for customer analytics',
        schedule_interval='0 2 * * *',  # Daily at 2 AM
        catchup=False,
        tags=['production', 'etl', 'customer'],
        # TODO: Add more production configurations
    )
    
    return dag

# TODO: Exercise 2 - Implement Data Extraction with Error Handling
def extract_customer_data(**context):
    """
    TODO: Extract customer data with comprehensive error handling
    
    Requirements:
    1. Simulate API data extraction
    2. Handle different data scenarios (empty, malformed, large)
    3. Implement proper logging
    4. Use XCom to pass data to next task
    5. Add data validation
    6. Handle API failures gracefully
    
    Test with:
    - Normal data extraction
    - Empty API response
    - Malformed JSON
    - Network timeouts
    - Large datasets
    """
    # TODO: Implement data extraction logic
    logging.info("Starting customer data extraction...")
    
    try:
        # TODO: Replace with actual API call
        # Mock customer data for now
        customers = [
            {'id': i, 'name': f'Customer_{i}', 'email': f'customer{i}@example.com'}
            for i in range(1, 101)
        ]
        
        # TODO: Add data validation
        if len(customers) == 0:
            raise ValueError("No customer data received")
        
        # TODO: Use XCom to pass data to next task
        context['task_instance'].xcom_push(key='customer_data', value=customers)
        
        logging.info(f"Extracted {len(customers)} customer records")
        return {'record_count': len(customers), 'status': 'success'}
        
    except Exception as e:
        logging.error(f"Data extraction failed: {str(e)}")
        # TODO: Add custom error handling
        raise

def extract_order_data(**context):
    """TODO: Extract order data in parallel with customer data"""
    # TODO: Implement parallel data extraction
    pass

# TODO: Exercise 3 - Build Data Transformation Pipeline
def validate_data_quality(**context):
    """
    TODO: Validate data quality before transformation
    
    Requirements:
    1. Check data completeness
    2. Validate data types
    3. Check for duplicates
    4. Verify business rules
    5. Log quality metrics
    6. Fail pipeline if quality is poor
    """
    # TODO: Implement data quality checks
    pass

def transform_customer_data(**context):
    """
    TODO: Transform customer data with business logic
    
    Requirements:
    1. Pull data from XCom
    2. Apply business transformations
    3. Handle missing values
    4. Calculate derived fields
    5. Standardize formats
    6. Push transformed data to XCom
    """
    # TODO: Implement transformation logic
    pass

def aggregate_metrics(**context):
    """TODO: Calculate business metrics from transformed data"""
    # TODO: Implement metric calculations
    pass

# TODO: Exercise 4 - Implement Conditional Logic
def check_data_volume(**context):
    """
    TODO: Implement branching logic based on data volume
    
    Requirements:
    1. Check volume of processed data
    2. Branch to different paths based on volume
    3. Use BranchPythonOperator
    4. Handle high-volume vs normal processing
    """
    # TODO: Implement branching logic
    pass

# TODO: Exercise 5 - Build Data Loading with Transactions
def load_to_staging(**context):
    """
    TODO: Load data to staging with transaction handling
    
    Requirements:
    1. Use database transactions
    2. Implement upsert logic
    3. Handle connection failures
    4. Log loading statistics
    5. Validate loaded data
    """
    # TODO: Implement staging load
    pass

def load_to_production(**context):
    """TODO: Load data to production with validation"""
    # TODO: Implement production load
    pass

def validate_load_results(**context):
    """TODO: Validate that data was loaded correctly"""
    # TODO: Implement load validation
    pass

# TODO: Exercise 6 - Implement Monitoring and Alerting
def generate_data_report(**context):
    """
    TODO: Generate data processing report
    
    Requirements:
    1. Collect metrics from all tasks
    2. Calculate processing statistics
    3. Generate summary report
    4. Include data quality metrics
    5. Format for email notification
    """
    # TODO: Implement report generation
    pass

def check_sla_compliance(**context):
    """TODO: Check if pipeline met SLA requirements"""
    # TODO: Implement SLA checking
    pass

# TODO: Exercise 7 - Create Task Failure Callbacks
def task_failure_callback(context):
    """
    TODO: Handle task failures with custom logic
    
    Requirements:
    1. Log failure details
    2. Send custom alerts
    3. Clean up resources
    4. Update monitoring systems
    """
    # TODO: Implement failure handling
    pass

def dag_success_callback(context):
    """TODO: Handle successful DAG completion"""
    # TODO: Implement success handling
    pass

# TODO: Exercise 8 - Build Complete DAG with All Components
def build_production_etl_dag():
    """
    TODO: Assemble complete production ETL DAG
    
    Requirements:
    1. Create all tasks using appropriate operators
    2. Set up complex dependencies
    3. Implement parallel processing where possible
    4. Add sensors for external dependencies
    5. Include monitoring and alerting
    6. Add cleanup tasks
    
    DAG Structure:
    start -> [extract_customers, extract_orders] -> validate_quality -> 
    transform -> aggregate -> check_volume -> [normal_load, high_volume_load] ->
    validate_results -> generate_report -> notify -> cleanup -> end
    """
    
    # Get DAG instance
    dag = create_production_dag()
    
    # TODO: Create start and end dummy tasks
    start_task = DummyOperator(
        task_id='start_pipeline',
        dag=dag
    )
    
    end_task = DummyOperator(
        task_id='end_pipeline',
        dag=dag,
        trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED
    )
    
    # TODO: Create extraction tasks (parallel)
    extract_customers_task = PythonOperator(
        # TODO: Configure task
    )
    
    extract_orders_task = PythonOperator(
        # TODO: Configure task
    )
    
    # TODO: Create data quality validation task
    validate_quality_task = PythonOperator(
        # TODO: Configure task
    )
    
    # TODO: Create transformation tasks
    transform_task = PythonOperator(
        # TODO: Configure task
    )
    
    aggregate_task = PythonOperator(
        # TODO: Configure task
    )
    
    # TODO: Create branching task
    branch_task = BranchPythonOperator(
        # TODO: Configure branching logic
    )
    
    # TODO: Create loading tasks (conditional)
    normal_load_task = PythonOperator(
        # TODO: Configure normal volume loading
    )
    
    high_volume_load_task = PythonOperator(
        # TODO: Configure high volume loading
    )
    
    # TODO: Create validation task
    validate_load_task = PythonOperator(
        # TODO: Configure load validation
    )
    
    # TODO: Create reporting task
    report_task = PythonOperator(
        # TODO: Configure report generation
    )
    
    # TODO: Create notification task
    notify_task = EmailOperator(
        # TODO: Configure email notification
    )
    
    # TODO: Create cleanup task
    cleanup_task = BashOperator(
        # TODO: Configure cleanup operations
    )
    
    # TODO: Create file sensor for external dependencies
    file_sensor = FileSensor(
        # TODO: Configure file sensor
    )
    
    # TODO: Define complex task dependencies
    # TODO: Implement parallel extraction
    # TODO: Add conditional loading
    # TODO: Include error handling paths
    
    return dag

# TODO: Exercise 9 - Add Advanced Features
def create_dynamic_dag():
    """
    TODO: Create DAG with dynamic task generation
    
    Requirements:
    1. Generate tasks dynamically based on configuration
    2. Handle variable number of data sources
    3. Create parallel processing branches
    4. Implement dynamic dependencies
    """
    # TODO: Implement dynamic DAG creation
    pass

def add_custom_operators():
    """
    TODO: Create custom operators for specific use cases
    
    Requirements:
    1. Create custom database operator
    2. Implement custom API operator
    3. Add custom validation operator
    4. Include proper error handling
    """
    # TODO: Implement custom operators
    pass

# TODO: Exercise 10 - Testing and Validation
def test_dag_structure():
    """
    TODO: Test DAG structure and configuration
    
    Requirements:
    1. Validate DAG has no cycles
    2. Check task dependencies
    3. Verify operator configurations
    4. Test XCom data flow
    5. Validate scheduling configuration
    """
    # TODO: Implement DAG testing
    pass

def test_task_functions():
    """TODO: Unit test individual task functions"""
    # TODO: Implement task testing
    pass

def main():
    """Test all Airflow components"""
    
    print("=== Day 12: Apache Airflow Basics - Exercise ===\n")
    
    # TODO: Test Exercise 1 - DAG Configuration
    print("=== Exercise 1: Production DAG Configuration ===")
    # TODO: Create and validate DAG configuration
    print("TODO: Implement production DAG configuration\n")
    
    # TODO: Test Exercise 2 - Data Extraction
    print("=== Exercise 2: Data Extraction with Error Handling ===")
    # TODO: Test data extraction functions
    print("TODO: Implement data extraction with error handling\n")
    
    # TODO: Test Exercise 3 - Data Transformation
    print("=== Exercise 3: Data Transformation Pipeline ===")
    # TODO: Test transformation functions
    print("TODO: Implement data transformation pipeline\n")
    
    # TODO: Test Exercise 4 - Conditional Logic
    print("=== Exercise 4: Conditional Logic and Branching ===")
    # TODO: Test branching logic
    print("TODO: Implement conditional logic and branching\n")
    
    # TODO: Test Exercise 5 - Data Loading
    print("=== Exercise 5: Data Loading with Transactions ===")
    # TODO: Test loading functions
    print("TODO: Implement data loading with transactions\n")
    
    # TODO: Test Exercise 6 - Monitoring and Alerting
    print("=== Exercise 6: Monitoring and Alerting ===")
    # TODO: Test monitoring functions
    print("TODO: Implement monitoring and alerting\n")
    
    # TODO: Test Exercise 7 - Failure Handling
    print("=== Exercise 7: Failure Handling and Callbacks ===")
    # TODO: Test failure handling
    print("TODO: Implement failure handling and callbacks\n")
    
    # TODO: Test Exercise 8 - Complete DAG Assembly
    print("=== Exercise 8: Complete Production DAG ===")
    # TODO: Test complete DAG
    print("TODO: Assemble complete production ETL DAG\n")
    
    # TODO: Test Exercise 9 - Advanced Features
    print("=== Exercise 9: Advanced Airflow Features ===")
    # TODO: Test advanced features
    print("TODO: Implement advanced Airflow features\n")
    
    # TODO: Test Exercise 10 - Testing and Validation
    print("=== Exercise 10: Testing and Validation ===")
    # TODO: Test validation functions
    print("TODO: Implement testing and validation\n")
    
    print("=== All Exercises Complete ===")
    print("Review your implementations and test with Airflow webserver")
    print("Consider deploying to production environment")

if __name__ == "__main__":
    main()
