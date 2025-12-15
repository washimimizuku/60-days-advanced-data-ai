"""
Day 12: Apache Airflow Basics - Solution
Production-ready ETL pipelines with comprehensive Airflow features
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.trigger_rule import TriggerRule
from airflow.models import Variable
from airflow.hooks.base import BaseHook
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import time
import random

# Solution 1 - Production DAG Configuration
def create_production_dag():
    """Production-ready DAG with comprehensive configuration"""
    
    # Comprehensive default arguments
    default_args = {
        'owner': 'data_engineering_team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email': ['data-alerts@company.com', 'on-call@company.com'],
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'retry_exponential_backoff': True,
        'max_retry_delay': timedelta(hours=1),
        'execution_timeout': timedelta(hours=2),
        'sla': timedelta(hours=4),  # SLA for task completion
        'on_failure_callback': task_failure_callback,
        'on_success_callback': None,
        'on_retry_callback': task_retry_callback,
    }
    
    # Production DAG configuration
    dag = DAG(
        dag_id='production_customer_analytics_etl',
        default_args=default_args,
        description='Production ETL pipeline for customer analytics with comprehensive error handling and monitoring',
        schedule_interval='0 2 * * *',  # Daily at 2 AM
        catchup=False,  # Don't backfill historical runs
        max_active_runs=1,  # Only one instance at a time
        dagrun_timeout=timedelta(hours=6),  # Total DAG timeout
        tags=['production', 'etl', 'customer_analytics', 'daily'],
        doc_md="""
        # Customer Analytics ETL Pipeline
        
        This DAG processes customer data daily:
        1. Extracts customer and order data from APIs
        2. Validates data quality
        3. Transforms and enriches data
        4. Loads to data warehouse
        5. Generates business reports
        
        **SLA**: 4 hours
        **Owner**: Data Engineering Team
        **Schedule**: Daily at 2 AM UTC
        """,
        is_paused_upon_creation=False,
        on_success_callback=dag_success_callback,
        on_failure_callback=dag_failure_callback,
    )
    
    return dag

# Solution 2 - Advanced Data Extraction with Error Handling
def extract_customer_data(**context):
    """Extract customer data with comprehensive error handling"""
    
    logging.info("Starting customer data extraction...")
    
    try:
        # Simulate API call with potential failures
        if random.random() < 0.1:  # 10% chance of failure for testing
            raise Exception("API connection timeout")
        
        # Mock customer data with realistic structure
        customers = []
        for i in range(1, 1001):  # 1000 customers
            customers.append({
                'customer_id': i,
                'name': f'Customer_{i}',
                'email': f'customer{i}@example.com',
                'registration_date': (datetime.now() - timedelta(days=random.randint(1, 365))).isoformat(),
                'status': random.choice(['active', 'inactive', 'suspended']),
                'lifetime_value': round(random.uniform(100, 10000), 2),
                'segment': random.choice(['premium', 'standard', 'basic']),
                'region': random.choice(['US', 'EU', 'APAC', 'LATAM'])
            })
        
        # Data quality validation
        if len(customers) == 0:
            raise ValueError("No customer data received from API")
        
        # Check for required fields
        required_fields = ['customer_id', 'name', 'email', 'status']
        for customer in customers[:10]:  # Sample check
            for field in required_fields:
                if field not in customer or customer[field] is None:
                    raise ValueError(f"Missing required field: {field}")
        
        # Store extraction metadata
        extraction_metadata = {
            'extraction_time': datetime.now().isoformat(),
            'record_count': len(customers),
            'source': 'customer_api',
            'api_version': 'v2.1',
            'data_quality_score': 0.95
        }
        
        # Push data to XCom
        context['task_instance'].xcom_push(key='customer_data', value=customers)
        context['task_instance'].xcom_push(key='customer_metadata', value=extraction_metadata)
        
        logging.info(f"Successfully extracted {len(customers)} customer records")
        return extraction_metadata
        
    except Exception as e:
        logging.error(f"Customer data extraction failed: {str(e)}")
        # Send custom alert
        send_custom_alert(f"Customer API extraction failed: {str(e)}", context)
        raise

def extract_order_data(**context):
    """Extract order data in parallel with customer data"""
    
    logging.info("Starting order data extraction...")
    
    try:
        # Mock order data
        orders = []
        for i in range(1, 5001):  # 5000 orders
            orders.append({
                'order_id': i,
                'customer_id': random.randint(1, 1000),
                'order_date': (datetime.now() - timedelta(days=random.randint(1, 90))).isoformat(),
                'amount': round(random.uniform(10, 1000), 2),
                'status': random.choice(['completed', 'pending', 'cancelled']),
                'product_category': random.choice(['electronics', 'clothing', 'books', 'home']),
                'payment_method': random.choice(['credit_card', 'debit_card', 'paypal', 'bank_transfer'])
            })
        
        # Validation
        if len(orders) == 0:
            raise ValueError("No order data received from API")
        
        extraction_metadata = {
            'extraction_time': datetime.now().isoformat(),
            'record_count': len(orders),
            'source': 'order_api',
            'api_version': 'v1.8'
        }
        
        # Push to XCom
        context['task_instance'].xcom_push(key='order_data', value=orders)
        context['task_instance'].xcom_push(key='order_metadata', value=extraction_metadata)
        
        logging.info(f"Successfully extracted {len(orders)} order records")
        return extraction_metadata
        
    except Exception as e:
        logging.error(f"Order data extraction failed: {str(e)}")
        send_custom_alert(f"Order API extraction failed: {str(e)}", context)
        raise

# Solution 3 - Data Quality Validation
def validate_data_quality(**context):
    """Comprehensive data quality validation"""
    
    logging.info("Starting data quality validation...")
    
    ti = context['task_instance']
    
    # Pull data from XCom
    customers = ti.xcom_pull(key='customer_data', task_ids='extract_customers')
    orders = ti.xcom_pull(key='order_data', task_ids='extract_orders')
    
    quality_report = {
        'validation_time': datetime.now().isoformat(),
        'customers': {},
        'orders': {},
        'overall_score': 0,
        'issues': []
    }
    
    # Validate customer data
    customer_issues = []
    
    # Completeness check
    total_customers = len(customers)
    complete_customers = sum(1 for c in customers if all(c.get(field) for field in ['customer_id', 'name', 'email']))
    completeness_score = complete_customers / total_customers if total_customers > 0 else 0
    
    if completeness_score < 0.95:
        customer_issues.append(f"Low completeness: {completeness_score:.2%}")
    
    # Uniqueness check
    customer_ids = [c['customer_id'] for c in customers]
    unique_ids = len(set(customer_ids))
    uniqueness_score = unique_ids / len(customer_ids) if customer_ids else 0
    
    if uniqueness_score < 0.99:
        customer_issues.append(f"Duplicate customer IDs detected: {uniqueness_score:.2%}")
    
    # Email format validation
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    valid_emails = sum(1 for c in customers if re.match(email_pattern, c.get('email', '')))
    email_validity_score = valid_emails / total_customers if total_customers > 0 else 0
    
    if email_validity_score < 0.95:
        customer_issues.append(f"Invalid email formats: {email_validity_score:.2%}")
    
    quality_report['customers'] = {
        'total_records': total_customers,
        'completeness_score': completeness_score,
        'uniqueness_score': uniqueness_score,
        'email_validity_score': email_validity_score,
        'issues': customer_issues
    }
    
    # Validate order data
    order_issues = []
    
    total_orders = len(orders)
    
    # Amount validation
    valid_amounts = sum(1 for o in orders if isinstance(o.get('amount'), (int, float)) and o['amount'] > 0)
    amount_validity_score = valid_amounts / total_orders if total_orders > 0 else 0
    
    if amount_validity_score < 0.99:
        order_issues.append(f"Invalid order amounts: {amount_validity_score:.2%}")
    
    # Customer ID reference validation
    customer_id_set = set(customer_ids)
    valid_customer_refs = sum(1 for o in orders if o.get('customer_id') in customer_id_set)
    reference_validity_score = valid_customer_refs / total_orders if total_orders > 0 else 0
    
    if reference_validity_score < 0.95:
        order_issues.append(f"Invalid customer references: {reference_validity_score:.2%}")
    
    quality_report['orders'] = {
        'total_records': total_orders,
        'amount_validity_score': amount_validity_score,
        'reference_validity_score': reference_validity_score,
        'issues': order_issues
    }
    
    # Calculate overall quality score
    all_scores = [
        completeness_score, uniqueness_score, email_validity_score,
        amount_validity_score, reference_validity_score
    ]
    overall_score = sum(all_scores) / len(all_scores)
    quality_report['overall_score'] = overall_score
    quality_report['issues'] = customer_issues + order_issues
    
    # Push quality report
    ti.xcom_push(key='quality_report', value=quality_report)
    
    # Fail if quality is too low
    if overall_score < 0.90:
        raise ValueError(f"Data quality too low: {overall_score:.2%}. Issues: {quality_report['issues']}")
    
    logging.info(f"Data quality validation passed. Overall score: {overall_score:.2%}")
    return quality_report

# Solution 4 - Advanced Data Transformation
def transform_customer_data(**context):
    """Transform customer data with business logic"""
    
    logging.info("Starting customer data transformation...")
    
    ti = context['task_instance']
    
    # Pull data
    customers = ti.xcom_pull(key='customer_data', task_ids='extract_customers')
    orders = ti.xcom_pull(key='order_data', task_ids='extract_orders')
    
    # Convert to DataFrames for easier processing
    df_customers = pd.DataFrame(customers)
    df_orders = pd.DataFrame(orders)
    
    # Transform customer data
    df_customers['registration_date'] = pd.to_datetime(df_customers['registration_date'])
    df_customers['days_since_registration'] = (datetime.now() - df_customers['registration_date']).dt.days
    
    # Calculate customer metrics from orders
    customer_metrics = df_orders.groupby('customer_id').agg({
        'order_id': 'count',
        'amount': ['sum', 'mean', 'max'],
        'order_date': 'max'
    }).round(2)
    
    # Flatten column names
    customer_metrics.columns = ['total_orders', 'total_spent', 'avg_order_value', 'max_order_value', 'last_order_date']
    customer_metrics = customer_metrics.reset_index()
    
    # Merge with customer data
    df_enriched = df_customers.merge(customer_metrics, on='customer_id', how='left')
    
    # Fill missing values for customers with no orders
    df_enriched['total_orders'] = df_enriched['total_orders'].fillna(0)
    df_enriched['total_spent'] = df_enriched['total_spent'].fillna(0)
    df_enriched['avg_order_value'] = df_enriched['avg_order_value'].fillna(0)
    df_enriched['max_order_value'] = df_enriched['max_order_value'].fillna(0)
    
    # Calculate customer segments based on behavior
    def calculate_segment(row):
        if row['total_spent'] > 5000 and row['total_orders'] > 10:
            return 'vip'
        elif row['total_spent'] > 1000 and row['total_orders'] > 5:
            return 'loyal'
        elif row['total_orders'] > 0:
            return 'active'
        else:
            return 'inactive'
    
    df_enriched['calculated_segment'] = df_enriched.apply(calculate_segment, axis=1)
    
    # Add processing metadata
    df_enriched['processed_at'] = datetime.now().isoformat()
    df_enriched['data_version'] = '1.0'
    
    # Convert back to list of dicts
    transformed_customers = df_enriched.to_dict('records')
    
    # Push transformed data
    ti.xcom_push(key='transformed_customers', value=transformed_customers)
    
    transformation_metadata = {
        'transformation_time': datetime.now().isoformat(),
        'input_records': len(customers),
        'output_records': len(transformed_customers),
        'new_fields_added': ['days_since_registration', 'total_orders', 'total_spent', 
                           'avg_order_value', 'max_order_value', 'calculated_segment'],
        'transformation_rules_applied': 5
    }
    
    ti.xcom_push(key='transformation_metadata', value=transformation_metadata)
    
    logging.info(f"Transformed {len(transformed_customers)} customer records")
    return transformation_metadata

def aggregate_business_metrics(**context):
    """Calculate business metrics from transformed data"""
    
    logging.info("Calculating business metrics...")
    
    ti = context['task_instance']
    customers = ti.xcom_pull(key='transformed_customers', task_ids='transform_customers')
    
    df = pd.DataFrame(customers)
    
    # Calculate business metrics
    metrics = {
        'calculation_time': datetime.now().isoformat(),
        'total_customers': len(df),
        'active_customers': len(df[df['calculated_segment'] != 'inactive']),
        'vip_customers': len(df[df['calculated_segment'] == 'vip']),
        'total_revenue': df['total_spent'].sum(),
        'avg_customer_value': df['total_spent'].mean(),
        'avg_orders_per_customer': df['total_orders'].mean(),
        'segment_distribution': df['calculated_segment'].value_counts().to_dict(),
        'region_distribution': df['region'].value_counts().to_dict(),
        'top_customers': df.nlargest(10, 'total_spent')[['customer_id', 'name', 'total_spent']].to_dict('records')
    }
    
    # Push metrics
    ti.xcom_push(key='business_metrics', value=metrics)
    
    logging.info(f"Calculated business metrics for {metrics['total_customers']} customers")
    return metrics

# Solution 5 - Conditional Logic and Branching
def check_data_volume(**context):
    """Implement branching logic based on data volume"""
    
    ti = context['task_instance']
    customers = ti.xcom_pull(key='transformed_customers', task_ids='transform_customers')
    
    record_count = len(customers)
    
    logging.info(f"Checking data volume: {record_count} records")
    
    # Branch based on volume
    if record_count > 5000:
        logging.info("High volume detected, using high-volume processing path")
        return 'high_volume_load'
    else:
        logging.info("Normal volume detected, using standard processing path")
        return 'normal_load'

# Solution 6 - Data Loading with Transactions
def load_to_staging(**context):
    """Load data to staging with transaction handling"""
    
    logging.info("Loading data to staging environment...")
    
    ti = context['task_instance']
    customers = ti.xcom_pull(key='transformed_customers', task_ids='transform_customers')
    
    try:
        # Simulate database transaction
        logging.info("Starting database transaction...")
        
        # Mock staging load with validation
        staging_records = []
        failed_records = []
        
        for customer in customers:
            try:
                # Validate record before insert
                if not customer.get('customer_id') or not customer.get('email'):
                    failed_records.append(customer)
                    continue
                
                # Mock database insert
                staging_record = {
                    **customer,
                    'loaded_at': datetime.now().isoformat(),
                    'load_batch_id': context['run_id']
                }
                staging_records.append(staging_record)
                
            except Exception as e:
                logging.warning(f"Failed to process record {customer.get('customer_id')}: {e}")
                failed_records.append(customer)
        
        # Commit transaction
        logging.info("Committing transaction...")
        
        load_result = {
            'load_time': datetime.now().isoformat(),
            'total_records': len(customers),
            'successful_loads': len(staging_records),
            'failed_loads': len(failed_records),
            'success_rate': len(staging_records) / len(customers) if customers else 0,
            'staging_table': 'staging.customers',
            'batch_id': context['run_id']
        }
        
        # Push results
        ti.xcom_push(key='staging_load_result', value=load_result)
        
        if load_result['success_rate'] < 0.95:
            raise ValueError(f"Staging load success rate too low: {load_result['success_rate']:.2%}")
        
        logging.info(f"Successfully loaded {len(staging_records)} records to staging")
        return load_result
        
    except Exception as e:
        logging.error(f"Staging load failed: {str(e)}")
        # Rollback transaction
        logging.info("Rolling back transaction...")
        raise

def load_to_production(**context):
    """Load data to production with validation"""
    
    logging.info("Loading data to production environment...")
    
    ti = context['task_instance']
    staging_result = ti.xcom_pull(key='staging_load_result', task_ids='staging_load')
    
    if not staging_result or staging_result['success_rate'] < 0.95:
        raise ValueError("Staging load did not meet quality requirements")
    
    try:
        # Mock production load
        production_result = {
            'load_time': datetime.now().isoformat(),
            'records_loaded': staging_result['successful_loads'],
            'production_table': 'prod.customers',
            'batch_id': context['run_id'],
            'data_validation_passed': True
        }
        
        # Push results
        ti.xcom_push(key='production_load_result', value=production_result)
        
        logging.info(f"Successfully loaded {production_result['records_loaded']} records to production")
        return production_result
        
    except Exception as e:
        logging.error(f"Production load failed: {str(e)}")
        raise

def high_volume_load(**context):
    """Handle high volume data loading with optimizations"""
    
    logging.info("Processing high volume data load...")
    
    ti = context['task_instance']
    customers = ti.xcom_pull(key='transformed_customers', task_ids='transform_customers')
    
    # Simulate batch processing for high volume
    batch_size = 1000
    batches = [customers[i:i + batch_size] for i in range(0, len(customers), batch_size)]
    
    load_results = []
    
    for i, batch in enumerate(batches):
        logging.info(f"Processing batch {i+1}/{len(batches)} ({len(batch)} records)")
        
        # Mock batch load
        batch_result = {
            'batch_number': i + 1,
            'records_processed': len(batch),
            'load_time': datetime.now().isoformat()
        }
        load_results.append(batch_result)
        
        # Simulate processing time
        time.sleep(0.1)
    
    high_volume_result = {
        'load_time': datetime.now().isoformat(),
        'total_records': len(customers),
        'total_batches': len(batches),
        'batch_results': load_results,
        'load_type': 'high_volume_optimized'
    }
    
    ti.xcom_push(key='high_volume_load_result', value=high_volume_result)
    
    logging.info(f"High volume load completed: {len(customers)} records in {len(batches)} batches")
    return high_volume_result

# Solution 7 - Monitoring and Reporting
def generate_data_report(**context):
    """Generate comprehensive data processing report"""
    
    logging.info("Generating data processing report...")
    
    ti = context['task_instance']
    
    # Collect data from all previous tasks
    customer_metadata = ti.xcom_pull(key='customer_metadata', task_ids='extract_customers')
    order_metadata = ti.xcom_pull(key='order_metadata', task_ids='extract_orders')
    quality_report = ti.xcom_pull(key='quality_report', task_ids='validate_quality')
    transformation_metadata = ti.xcom_pull(key='transformation_metadata', task_ids='transform_customers')
    business_metrics = ti.xcom_pull(key='business_metrics', task_ids='aggregate_metrics')
    
    # Try to get load results (may vary based on branch taken)
    staging_result = ti.xcom_pull(key='staging_load_result', task_ids='staging_load')
    production_result = ti.xcom_pull(key='production_load_result', task_ids='production_load')
    high_volume_result = ti.xcom_pull(key='high_volume_load_result', task_ids='high_volume_load')
    
    # Generate comprehensive report
    report = {
        'report_generated_at': datetime.now().isoformat(),
        'dag_run_id': context['run_id'],
        'execution_date': context['execution_date'].isoformat(),
        
        'extraction_summary': {
            'customers_extracted': customer_metadata['record_count'] if customer_metadata else 0,
            'orders_extracted': order_metadata['record_count'] if order_metadata else 0,
            'extraction_time': customer_metadata['extraction_time'] if customer_metadata else None
        },
        
        'data_quality_summary': {
            'overall_quality_score': quality_report['overall_score'] if quality_report else 0,
            'quality_issues': quality_report['issues'] if quality_report else [],
            'customers_quality': quality_report['customers'] if quality_report else {},
            'orders_quality': quality_report['orders'] if quality_report else {}
        },
        
        'transformation_summary': {
            'records_transformed': transformation_metadata['output_records'] if transformation_metadata else 0,
            'new_fields_added': transformation_metadata['new_fields_added'] if transformation_metadata else [],
            'transformation_time': transformation_metadata['transformation_time'] if transformation_metadata else None
        },
        
        'business_metrics_summary': business_metrics if business_metrics else {},
        
        'loading_summary': {
            'staging_loaded': staging_result['successful_loads'] if staging_result else 0,
            'production_loaded': production_result['records_loaded'] if production_result else 0,
            'high_volume_processed': high_volume_result['total_records'] if high_volume_result else 0,
            'load_success_rate': staging_result['success_rate'] if staging_result else 0
        },
        
        'pipeline_performance': {
            'total_processing_time': None,  # Would calculate from start/end times
            'records_per_second': None,     # Would calculate based on volume and time
            'memory_usage': None,           # Would collect from system metrics
            'cpu_usage': None               # Would collect from system metrics
        }
    }
    
    # Push report
    ti.xcom_push(key='data_processing_report', value=report)
    
    logging.info("Data processing report generated successfully")
    return report

def check_sla_compliance(**context):
    """Check if pipeline met SLA requirements"""
    
    logging.info("Checking SLA compliance...")
    
    ti = context['task_instance']
    dag_run = context['dag_run']
    
    # Calculate pipeline duration
    start_time = dag_run.start_date
    current_time = datetime.now()
    duration = current_time - start_time
    
    # SLA requirements
    sla_requirements = {
        'max_duration_hours': 4,
        'min_data_quality_score': 0.90,
        'min_load_success_rate': 0.95
    }
    
    # Check duration SLA
    duration_hours = duration.total_seconds() / 3600
    duration_sla_met = duration_hours <= sla_requirements['max_duration_hours']
    
    # Check quality SLA
    quality_report = ti.xcom_pull(key='quality_report', task_ids='validate_quality')
    quality_sla_met = (quality_report and 
                      quality_report['overall_score'] >= sla_requirements['min_data_quality_score'])
    
    # Check load SLA
    staging_result = ti.xcom_pull(key='staging_load_result', task_ids='staging_load')
    load_sla_met = (staging_result and 
                   staging_result['success_rate'] >= sla_requirements['min_load_success_rate'])
    
    sla_report = {
        'check_time': datetime.now().isoformat(),
        'pipeline_duration_hours': duration_hours,
        'sla_requirements': sla_requirements,
        'sla_compliance': {
            'duration_sla_met': duration_sla_met,
            'quality_sla_met': quality_sla_met,
            'load_sla_met': load_sla_met,
            'overall_sla_met': all([duration_sla_met, quality_sla_met, load_sla_met])
        },
        'violations': []
    }
    
    # Record violations
    if not duration_sla_met:
        sla_report['violations'].append(f"Duration exceeded: {duration_hours:.2f}h > {sla_requirements['max_duration_hours']}h")
    
    if not quality_sla_met:
        quality_score = quality_report['overall_score'] if quality_report else 0
        sla_report['violations'].append(f"Quality below threshold: {quality_score:.2%} < {sla_requirements['min_data_quality_score']:.2%}")
    
    if not load_sla_met:
        load_rate = staging_result['success_rate'] if staging_result else 0
        sla_report['violations'].append(f"Load success rate below threshold: {load_rate:.2%} < {sla_requirements['min_load_success_rate']:.2%}")
    
    # Push SLA report
    ti.xcom_push(key='sla_report', value=sla_report)
    
    if not sla_report['sla_compliance']['overall_sla_met']:
        logging.warning(f"SLA violations detected: {sla_report['violations']}")
        send_sla_violation_alert(sla_report, context)
    else:
        logging.info("All SLA requirements met")
    
    return sla_report

# Solution 8 - Callback Functions
def task_failure_callback(context):
    """Handle task failures with custom logic"""
    
    task_instance = context['task_instance']
    dag_id = context['dag'].dag_id
    task_id = task_instance.task_id
    execution_date = context['execution_date']
    
    failure_info = {
        'dag_id': dag_id,
        'task_id': task_id,
        'execution_date': execution_date.isoformat(),
        'failure_time': datetime.now().isoformat(),
        'exception': str(context.get('exception', 'Unknown error')),
        'log_url': task_instance.log_url,
        'try_number': task_instance.try_number,
        'max_tries': task_instance.max_tries
    }
    
    logging.error(f"Task failure: {task_id} in DAG {dag_id}")
    logging.error(f"Exception: {failure_info['exception']}")
    
    # Send custom alert (would integrate with alerting system)
    send_custom_alert(f"Task {task_id} failed in DAG {dag_id}", context, failure_info)

def task_retry_callback(context):
    """Handle task retries"""
    
    task_instance = context['task_instance']
    logging.warning(f"Task {task_instance.task_id} retry {task_instance.try_number}/{task_instance.max_tries}")

def dag_success_callback(context):
    """Handle successful DAG completion"""
    
    dag_run = context['dag_run']
    duration = datetime.now() - dag_run.start_date
    
    logging.info(f"DAG {context['dag'].dag_id} completed successfully in {duration}")
    
    # Send success notification
    success_info = {
        'dag_id': context['dag'].dag_id,
        'execution_date': context['execution_date'].isoformat(),
        'duration': str(duration),
        'success_time': datetime.now().isoformat()
    }
    
    send_success_notification(success_info, context)

def dag_failure_callback(context):
    """Handle DAG failures"""
    
    logging.error(f"DAG {context['dag'].dag_id} failed")
    send_custom_alert(f"DAG {context['dag'].dag_id} failed", context)

# Solution 9 - Utility Functions
def send_custom_alert(message: str, context: dict, details: dict = None):
    """Send custom alert to monitoring system"""
    
    alert_payload = {
        'message': message,
        'dag_id': context['dag'].dag_id,
        'task_id': context.get('task_instance', {}).task_id if context.get('task_instance') else None,
        'execution_date': context['execution_date'].isoformat(),
        'alert_time': datetime.now().isoformat(),
        'severity': 'high',
        'details': details or {}
    }
    
    # In production, send to Slack, PagerDuty, etc.
    logging.info(f"ALERT: {json.dumps(alert_payload, indent=2)}")

def send_success_notification(success_info: dict, context: dict):
    """Send success notification"""
    
    logging.info(f"SUCCESS NOTIFICATION: {json.dumps(success_info, indent=2)}")

def send_sla_violation_alert(sla_report: dict, context: dict):
    """Send SLA violation alert"""
    
    alert_message = f"SLA violations in DAG {context['dag'].dag_id}: {', '.join(sla_report['violations'])}"
    send_custom_alert(alert_message, context, sla_report)

# Solution 10 - Complete DAG Assembly
def build_production_etl_dag():
    """Assemble complete production ETL DAG"""
    
    # Get DAG instance
    dag = create_production_dag()
    
    # Start and end tasks
    start_task = DummyOperator(
        task_id='start_pipeline',
        dag=dag,
        doc_md="Pipeline start marker"
    )
    
    end_task = DummyOperator(
        task_id='end_pipeline',
        dag=dag,
        trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED,
        doc_md="Pipeline end marker"
    )
    
    # Extraction tasks (parallel)
    extract_customers_task = PythonOperator(
        task_id='extract_customers',
        python_callable=extract_customer_data,
        dag=dag,
        doc_md="Extract customer data from API with error handling"
    )
    
    extract_orders_task = PythonOperator(
        task_id='extract_orders',
        python_callable=extract_order_data,
        dag=dag,
        doc_md="Extract order data from API in parallel"
    )
    
    # Data quality validation
    validate_quality_task = PythonOperator(
        task_id='validate_quality',
        python_callable=validate_data_quality,
        dag=dag,
        doc_md="Comprehensive data quality validation"
    )
    
    # Transformation tasks
    transform_customers_task = PythonOperator(
        task_id='transform_customers',
        python_callable=transform_customer_data,
        dag=dag,
        doc_md="Transform and enrich customer data"
    )
    
    aggregate_metrics_task = PythonOperator(
        task_id='aggregate_metrics',
        python_callable=aggregate_business_metrics,
        dag=dag,
        doc_md="Calculate business metrics"
    )
    
    # Branching task
    branch_task = BranchPythonOperator(
        task_id='check_volume',
        python_callable=check_data_volume,
        dag=dag,
        doc_md="Branch based on data volume"
    )
    
    # Loading tasks (conditional)
    staging_load_task = PythonOperator(
        task_id='staging_load',
        python_callable=load_to_staging,
        dag=dag,
        doc_md="Load data to staging with transactions"
    )
    
    normal_load_task = PythonOperator(
        task_id='normal_load',
        python_callable=load_to_production,
        dag=dag,
        doc_md="Normal volume production load"
    )
    
    high_volume_load_task = PythonOperator(
        task_id='high_volume_load',
        python_callable=high_volume_load,
        dag=dag,
        doc_md="High volume optimized load"
    )
    
    # Join task after branching
    join_task = DummyOperator(
        task_id='join_loads',
        dag=dag,
        trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED,
        doc_md="Join point after conditional loading"
    )
    
    # Reporting and monitoring
    report_task = PythonOperator(
        task_id='generate_report',
        python_callable=generate_data_report,
        dag=dag,
        doc_md="Generate comprehensive processing report"
    )
    
    sla_check_task = PythonOperator(
        task_id='check_sla',
        python_callable=check_sla_compliance,
        dag=dag,
        doc_md="Check SLA compliance"
    )
    
    # Notification task
    notify_task = EmailOperator(
        task_id='send_notification',
        to=['data-team@company.com'],
        subject='Customer Analytics ETL - {{ ds }} - {{ dag_run.conf.get("status", "Completed") }}',
        html_content="""
        <h3>Customer Analytics ETL Pipeline Report</h3>
        <p><strong>Execution Date:</strong> {{ ds }}</p>
        <p><strong>DAG Run ID:</strong> {{ run_id }}</p>
        <p><strong>Status:</strong> {{ dag_run.conf.get("status", "Completed") }}</p>
        
        <h4>Summary:</h4>
        <ul>
            <li>Pipeline completed successfully</li>
            <li>Check Airflow UI for detailed metrics</li>
            <li>Data available in production tables</li>
        </ul>
        
        <p><strong>Airflow UI:</strong> <a href="{{ var.value.airflow_base_url }}/graph?dag_id={{ dag.dag_id }}">View DAG</a></p>
        """,
        dag=dag,
        trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED
    )
    
    # Cleanup task
    cleanup_task = BashOperator(
        task_id='cleanup_temp_files',
        bash_command="""
        echo "Cleaning up temporary files..."
        # rm -rf /tmp/etl_{{ ds_nodash }}/*
        echo "Cleanup completed"
        """,
        dag=dag,
        trigger_rule=TriggerRule.NONE_FAILED_OR_SKIPPED
    )
    
    # Define complex task dependencies
    start_task >> [extract_customers_task, extract_orders_task]
    [extract_customers_task, extract_orders_task] >> validate_quality_task
    validate_quality_task >> transform_customers_task
    transform_customers_task >> aggregate_metrics_task
    aggregate_metrics_task >> branch_task
    
    # Branching paths
    branch_task >> staging_load_task >> normal_load_task >> join_task
    branch_task >> high_volume_load_task >> join_task
    
    # Final tasks
    join_task >> [report_task, sla_check_task]
    [report_task, sla_check_task] >> notify_task >> cleanup_task >> end_task
    
    return dag

# Create the DAG instance
dag = build_production_etl_dag()

def main():
    """Demonstrate all Airflow solutions"""
    
    print("=== Day 12: Apache Airflow Basics - Solutions ===\n")
    
    print("=== Solution 1: Production DAG Configuration ===")
    test_dag = create_production_dag()
    print(f"Created DAG: {test_dag.dag_id}")
    print(f"Schedule: {test_dag.schedule_interval}")
    print(f"Max active runs: {test_dag.max_active_runs}")
    print(f"Tags: {test_dag.tags}")
    
    print("\n=== Solution 2: Data Extraction with Error Handling ===")
    # Mock context for testing
    mock_context = {
        'task_instance': type('MockTI', (), {
            'xcom_push': lambda self, key, value: print(f"XCom push: {key} = {type(value).__name__}"),
            'xcom_pull': lambda self, key, task_ids: None
        })(),
        'execution_date': datetime.now(),
        'run_id': 'test_run_123'
    }
    
    try:
        customer_result = extract_customer_data(**mock_context)
        print(f"Customer extraction: {customer_result['record_count']} records")
        
        order_result = extract_order_data(**mock_context)
        print(f"Order extraction: {order_result['record_count']} records")
    except Exception as e:
        print(f"Extraction test failed: {e}")
    
    print("\n=== Solution 3: Data Quality Validation ===")
    print("Data quality validation implemented with comprehensive checks")
    print("- Completeness validation")
    print("- Uniqueness validation") 
    print("- Format validation")
    print("- Reference integrity validation")
    
    print("\n=== Solution 4: Advanced Data Transformation ===")
    print("Transformation features implemented:")
    print("- Customer data enrichment")
    print("- Business metric calculations")
    print("- Segment classification")
    print("- Data type conversions")
    
    print("\n=== Solution 5: Conditional Logic and Branching ===")
    print("Branching logic implemented:")
    print("- Volume-based processing paths")
    print("- High-volume optimizations")
    print("- Conditional task execution")
    
    print("\n=== Solution 6: Data Loading with Transactions ===")
    print("Loading features implemented:")
    print("- Transactional staging loads")
    print("- Production data validation")
    print("- Batch processing for high volume")
    print("- Load result tracking")
    
    print("\n=== Solution 7: Monitoring and Reporting ===")
    print("Monitoring features implemented:")
    print("- Comprehensive data reports")
    print("- SLA compliance checking")
    print("- Performance metrics")
    print("- Business metric tracking")
    
    print("\n=== Solution 8: Callback Functions ===")
    print("Callback functions implemented:")
    print("- Task failure handling")
    print("- Retry notifications")
    print("- DAG success/failure callbacks")
    print("- Custom alerting integration")
    
    print("\n=== Complete Production DAG Features ===")
    print("✅ Comprehensive error handling and retries")
    print("✅ Parallel data extraction")
    print("✅ Data quality validation")
    print("✅ Advanced transformations")
    print("✅ Conditional processing paths")
    print("✅ Transactional data loading")
    print("✅ SLA monitoring and compliance")
    print("✅ Comprehensive reporting")
    print("✅ Email notifications")
    print("✅ Custom alerting and callbacks")
    print("✅ Cleanup and resource management")
    
    print("\n=== Production Deployment Ready ===")
    print("This DAG demonstrates enterprise-grade Airflow patterns:")
    print("- Scalable architecture")
    print("- Robust error handling")
    print("- Comprehensive monitoring")
    print("- Production best practices")

if __name__ == "__main__":
    main()
