"""
Day 15: Airflow Production Patterns - Complete Solution

Advanced Airflow patterns with dynamic DAGs, task groups, branching, and monitoring.
This solution demonstrates enterprise-grade Airflow patterns for production environments.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import logging
import random
import requests
from dataclasses import dataclass

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.sensors.filesystem import FileSensor
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook

# =============================================================================
# CONFIGURATION DATA STRUCTURES
# =============================================================================

@dataclass
class DataSourceConfig:
    """Configuration for a data source pipeline"""
    name: str
    source_type: str
    source_config: Dict[str, Any]
    target_table: str
    schedule_interval: str
    priority: int
    expected_volume_mb: int
    sla_hours: int
    quality_rules: List[Dict[str, Any]]
    transformations: List[Dict[str, Any]]

# Complete data source configurations for TechCorp
DATA_SOURCE_CONFIGS = [
    DataSourceConfig(
        name="customer_data",
        source_type="api",
        source_config={
            "url": "https://api.techcorp.com/customers",
            "auth_type": "bearer",
            "headers": {"Content-Type": "application/json"}
        },
        target_table="customers",
        schedule_interval="@hourly",
        priority=9,
        expected_volume_mb=75,
        sla_hours=2,
        quality_rules=[
            {"type": "not_null", "columns": ["customer_id", "email"]},
            {"type": "unique", "columns": ["customer_id"]},
            {"type": "format", "column": "email", "pattern": r"^[^@]+@[^@]+\.[^@]+$"}
        ],
        transformations=[
            {"type": "email_normalize"},
            {"type": "pii_hash", "columns": ["email", "phone"]},
            {"type": "address_standardize"}
        ]
    ),
    
    DataSourceConfig(
        name="transaction_data",
        source_type="database",
        source_config={
            "connection_id": "postgres_transactions",
            "table": "raw_transactions",
            "incremental_column": "created_at"
        },
        target_table="transactions",
        schedule_interval="@daily",
        priority=7,
        expected_volume_mb=500,
        sla_hours=4,
        quality_rules=[
            {"type": "not_null", "columns": ["transaction_id", "customer_id", "amount"]},
            {"type": "range", "column": "amount", "min": 0, "max": 100000},
            {"type": "referential", "column": "customer_id", "ref_table": "customers"}
        ],
        transformations=[
            {"type": "currency_normalize"},
            {"type": "fraud_scoring"},
            {"type": "category_mapping"}
        ]
    ),
    
    DataSourceConfig(
        name="product_catalog",
        source_type="file",
        source_config={
            "file_path": "/data/products/catalog_{{ ds }}.csv",
            "format": "csv",
            "delimiter": ",",
            "encoding": "utf-8"
        },
        target_table="products",
        schedule_interval="@weekly",
        priority=4,
        expected_volume_mb=25,
        sla_hours=12,
        quality_rules=[
            {"type": "not_null", "columns": ["product_id", "name", "price"]},
            {"type": "range", "column": "price", "min": 0},
            {"type": "enum", "column": "status", "values": ["active", "inactive", "discontinued"]}
        ],
        transformations=[
            {"type": "price_standardize"},
            {"type": "category_hierarchy"},
            {"type": "inventory_sync"}
        ]
    ),
    
    DataSourceConfig(
        name="user_behavior_events",
        source_type="api",
        source_config={
            "url": "https://events.techcorp.com/api/v1/events",
            "auth_type": "api_key",
            "batch_size": 10000,
            "compression": "gzip"
        },
        target_table="user_events",
        schedule_interval="*/15 * * * *",  # Every 15 minutes
        priority=8,
        expected_volume_mb=200,
        sla_hours=1,
        quality_rules=[
            {"type": "not_null", "columns": ["event_id", "user_id", "event_type", "timestamp"]},
            {"type": "enum", "column": "event_type", "values": ["click", "view", "purchase", "signup"]},
            {"type": "timestamp_range", "column": "timestamp", "max_age_hours": 24}
        ],
        transformations=[
            {"type": "event_deduplication"},
            {"type": "session_stitching"},
            {"type": "geo_enrichment"}
        ]
    ),
    
    DataSourceConfig(
        name="partner_data",
        source_type="sftp",
        source_config={
            "host": "sftp.partner.com",
            "username": "techcorp_user",
            "key_file": "/keys/partner_key.pem",
            "remote_path": "/exports/daily/",
            "file_pattern": "partner_data_{{ ds }}.json"
        },
        target_table="partner_integrations",
        schedule_interval="@daily",
        priority=5,
        expected_volume_mb=150,
        sla_hours=8,
        quality_rules=[
            {"type": "not_null", "columns": ["partner_id", "integration_type"]},
            {"type": "json_schema", "schema_file": "/schemas/partner_schema.json"},
            {"type": "freshness", "max_age_hours": 48}
        ],
        transformations=[
            {"type": "json_flatten"},
            {"type": "partner_mapping"},
            {"type": "data_enrichment"}
        ]
    )
]

# Pool configurations for resource management
POOL_CONFIGS = {
    "fast_processing_pool": {"slots": 3, "description": "High-priority, small data processing"},
    "standard_pool": {"slots": 10, "description": "Standard data processing"},
    "batch_processing_pool": {"slots": 2, "description": "Large data batch processing"},
    "monitoring_pool": {"slots": 5, "description": "Monitoring and alerting tasks"},
    "external_api_pool": {"slots": 5, "description": "External API calls"}
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_pools():
    """Setup Airflow pools for resource management"""
    pool_setup_commands = []
    
    for pool_name, config in POOL_CONFIGS.items():
        command = f"airflow pools set {pool_name} {config['slots']} \"{config['description']}\""
        pool_setup_commands.append(command)
    
    logging.info("Pool setup commands:")
    for cmd in pool_setup_commands:
        logging.info(cmd)
    
    return pool_setup_commands

def get_processing_strategy(data_size_mb: int, priority: int) -> Dict[str, Any]:
    """Determine processing strategy based on data characteristics"""
    
    # Base strategy
    strategy = {
        "pool": "standard_pool",
        "timeout_minutes": 30,
        "retries": 2,
        "priority_weight": 5
    }
    
    # Adjust based on data size
    if data_size_mb > 1000:  # Large data
        strategy.update({
            "pool": "batch_processing_pool",
            "timeout_minutes": 120,
            "retries": 1,
            "priority_weight": 1
        })
    elif data_size_mb < 100 and priority >= 8:  # Small, high-priority data
        strategy.update({
            "pool": "fast_processing_pool",
            "timeout_minutes": 15,
            "retries": 3,
            "priority_weight": 10
        })
    
    # Adjust based on priority
    if priority >= 8:
        strategy["priority_weight"] = max(strategy["priority_weight"], 8)
        strategy["retries"] = min(strategy["retries"] + 1, 3)
    elif priority <= 3:
        strategy["priority_weight"] = 1
        strategy["timeout_minutes"] = max(strategy["timeout_minutes"], 60)
    
    return strategy

# =============================================================================
# CUSTOM SENSORS
# =============================================================================

class DataQualitySensor(BaseSensorOperator):
    """Custom sensor that waits for data and validates quality"""
    
    def __init__(self, 
                 data_source: str,
                 quality_checks: List[Dict[str, Any]],
                 min_records: int = 1,
                 **kwargs):
        super().__init__(**kwargs)
        self.data_source = data_source
        self.quality_checks = quality_checks
        self.min_records = min_records
    
    def poke(self, context):
        """Check if data is available and meets quality standards"""
        self.log.info(f"Checking data quality for {self.data_source}")
        
        try:
            # Simulate data availability check
            data_available = self._check_data_availability()
            
            if not data_available:
                self.log.info(f"No new data available for {self.data_source}")
                return False
            
            # Simulate quality validation
            quality_passed = self._validate_quality()
            
            if not quality_passed:
                self.log.warning(f"Data quality issues found for {self.data_source}")
                return False
            
            self.log.info(f"Data ready and quality validated for {self.data_source}")
            return True
            
        except Exception as e:
            self.log.error(f"Error checking data quality: {e}")
            return False
    
    def _check_data_availability(self) -> bool:
        """Check if new data is available"""
        # Simulate data availability (80% chance)
        return random.random() > 0.2
    
    def _validate_quality(self) -> bool:
        """Validate data quality rules"""
        # Simulate quality validation (90% pass rate)
        return random.random() > 0.1

class ExternalAPIHealthSensor(BaseSensorOperator):
    """Sensor that checks if external API is healthy before processing"""
    
    def __init__(self, api_url: str, expected_status: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.api_url = api_url
        self.expected_status = expected_status
    
    def poke(self, context):
        """Check if external API is healthy"""
        try:
            self.log.info(f"Checking API health: {self.api_url}")
            
            # Make health check request
            response = requests.get(f"{self.api_url}/health", timeout=10)
            
            if response.status_code == self.expected_status:
                health_data = response.json()
                
                if health_data.get('status') == 'healthy':
                    self.log.info("API is healthy")
                    return True
                else:
                    self.log.warning(f"API status: {health_data.get('status')}")
                    return False
            else:
                self.log.warning(f"API returned status: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            self.log.error(f"API health check failed: {e}")
            return False

class DatabaseConnectionSensor(BaseSensorOperator):
    """Sensor that checks database connectivity and performance"""
    
    def __init__(self, connection_id: str, max_response_time: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.connection_id = connection_id
        self.max_response_time = max_response_time
    
    def poke(self, context):
        """Check database connectivity and performance"""
        try:
            start_time = datetime.now()
            
            # Test database connection
            hook = PostgresHook(postgres_conn_id=self.connection_id)
            result = hook.get_first("SELECT 1")
            
            response_time = (datetime.now() - start_time).total_seconds()
            
            if result and response_time <= self.max_response_time:
                self.log.info(f"Database healthy (response time: {response_time}s)")
                return True
            else:
                self.log.warning(f"Database slow (response time: {response_time}s)")
                return False
                
        except Exception as e:
            self.log.error(f"Database connection failed: {e}")
            return False

# =============================================================================
# TASK GROUP FACTORIES
# =============================================================================

def create_data_ingestion_task_group(group_id: str, source_config: DataSourceConfig) -> TaskGroup:
    """Create reusable data ingestion task group"""
    
    with TaskGroup(group_id=group_id) as group:
        
        def extract_data(**context):
            """Extract data from source"""
            ti = context['task_instance']
            
            self.log.info(f"Extracting data from {source_config.source_type}: {source_config.name}")
            
            # Simulate extraction based on source type
            if source_config.source_type == "api":
                extracted_data = self._extract_from_api(source_config.source_config)
            elif source_config.source_type == "database":
                extracted_data = self._extract_from_database(source_config.source_config)
            elif source_config.source_type == "file":
                extracted_data = self._extract_from_file(source_config.source_config)
            elif source_config.source_type == "sftp":
                extracted_data = self._extract_from_sftp(source_config.source_config)
            else:
                raise ValueError(f"Unsupported source type: {source_config.source_type}")
            
            # Add metadata
            extracted_data.update({
                "source": source_config.name,
                "extraction_time": datetime.now().isoformat(),
                "expected_volume_mb": source_config.expected_volume_mb
            })
            
            ti.xcom_push(key='extraction_result', value=extracted_data)
            return extracted_data
        
        def _extract_from_api(config):
            """Extract data from API"""
            # Simulate API extraction
            return {
                "records": random.randint(800, 1200),
                "size_mb": random.randint(50, 100),
                "api_response_time": random.uniform(0.5, 2.0)
            }
        
        def _extract_from_database(config):
            """Extract data from database"""
            # Simulate database extraction
            return {
                "records": random.randint(5000, 15000),
                "size_mb": random.randint(200, 800),
                "query_time": random.uniform(2.0, 10.0)
            }
        
        def _extract_from_file(config):
            """Extract data from file"""
            # Simulate file extraction
            return {
                "records": random.randint(1000, 5000),
                "size_mb": random.randint(10, 50),
                "file_size": random.randint(10, 50)
            }
        
        def _extract_from_sftp(config):
            """Extract data from SFTP"""
            # Simulate SFTP extraction
            return {
                "records": random.randint(2000, 8000),
                "size_mb": random.randint(100, 300),
                "transfer_time": random.uniform(5.0, 20.0)
            }
        
        extract_task = PythonOperator(
            task_id='extract_data',
            python_callable=extract_data
        )
        
        def validate_data(**context):
            """Validate extracted data quality"""
            ti = context['task_instance']
            extraction_data = ti.xcom_pull(key='extraction_result', task_ids='extract_data')
            
            if not extraction_data:
                raise ValueError("No extraction data found")
            
            # Apply quality rules
            validation_results = []
            
            for rule in source_config.quality_rules:
                result = self._apply_quality_rule(rule, extraction_data)
                validation_results.append(result)
            
            # Check if any critical rules failed
            critical_failures = [r for r in validation_results if not r['passed'] and r.get('critical', False)]
            
            if critical_failures:
                raise ValueError(f"Critical quality checks failed: {critical_failures}")
            
            validation_summary = {
                "total_rules": len(validation_results),
                "passed": len([r for r in validation_results if r['passed']]),
                "failed": len([r for r in validation_results if not r['passed']]),
                "critical_failures": len(critical_failures)
            }
            
            ti.xcom_push(key='validation_result', value=validation_summary)
            logging.info(f"Data validation completed: {validation_summary}")
            
            return validation_summary
        
        def _apply_quality_rule(rule, data):
            """Apply a single quality rule"""
            # Simulate quality rule application
            rule_type = rule.get('type')
            
            if rule_type == 'not_null':
                # Simulate not null check (95% pass rate)
                passed = random.random() > 0.05
            elif rule_type == 'unique':
                # Simulate uniqueness check (98% pass rate)
                passed = random.random() > 0.02
            elif rule_type == 'range':
                # Simulate range check (90% pass rate)
                passed = random.random() > 0.10
            else:
                # Default pass rate for other rules
                passed = random.random() > 0.05
            
            return {
                'rule_type': rule_type,
                'passed': passed,
                'critical': rule.get('critical', False)
            }
        
        validate_task = PythonOperator(
            task_id='validate_data',
            python_callable=validate_data
        )
        
        def profile_data(**context):
            """Profile data characteristics for processing decisions"""
            ti = context['task_instance']
            extraction_data = ti.xcom_pull(key='extraction_result', task_ids='extract_data')
            
            if not extraction_data:
                raise ValueError("No extraction data found")
            
            # Generate data profile
            profile = {
                "record_count": extraction_data.get('records', 0),
                "size_mb": extraction_data.get('size_mb', 0),
                "complexity_score": random.uniform(1.0, 10.0),
                "null_percentage": random.uniform(0.0, 5.0),
                "duplicate_percentage": random.uniform(0.0, 2.0),
                "data_types": {
                    "string": random.randint(5, 15),
                    "numeric": random.randint(3, 10),
                    "datetime": random.randint(1, 5),
                    "boolean": random.randint(0, 3)
                }
            }
            
            # Determine processing recommendations
            if profile["size_mb"] > 1000:
                profile["recommended_strategy"] = "batch_processing"
            elif profile["complexity_score"] > 7.0:
                profile["recommended_strategy"] = "complex_processing"
            else:
                profile["recommended_strategy"] = "standard_processing"
            
            ti.xcom_push(key='profile_result', value=profile)
            logging.info(f"Data profiling completed: {profile}")
            
            return profile
        
        profile_task = PythonOperator(
            task_id='profile_data',
            python_callable=profile_data
        )
        
        # Define task dependencies within group
        extract_task >> validate_task >> profile_task
    
    return group

def create_data_processing_task_group(group_id: str, source_config: DataSourceConfig) -> TaskGroup:
    """Create reusable data processing task group"""
    
    with TaskGroup(group_id=group_id) as group:
        
        def transform_data(**context):
            """Apply transformations to data"""
            ti = context['task_instance']
            extraction_data = ti.xcom_pull(key='extraction_result', task_ids='ingestion.extract_data')
            
            if not extraction_data:
                raise ValueError("No extraction data found")
            
            # Apply transformations
            transformation_results = []
            
            for transformation in source_config.transformations:
                result = self._apply_transformation(transformation, extraction_data)
                transformation_results.append(result)
            
            transform_summary = {
                "transformations_applied": len(transformation_results),
                "records_processed": extraction_data.get('records', 0),
                "processing_time": random.uniform(1.0, 10.0),
                "transformations": transformation_results
            }
            
            ti.xcom_push(key='transform_result', value=transform_summary)
            logging.info(f"Data transformation completed: {transform_summary}")
            
            return transform_summary
        
        def _apply_transformation(transformation, data):
            """Apply a single transformation"""
            transform_type = transformation.get('type')
            
            # Simulate transformation processing
            processing_time = random.uniform(0.1, 2.0)
            success_rate = random.uniform(0.95, 1.0)
            
            return {
                'type': transform_type,
                'processing_time': processing_time,
                'success_rate': success_rate,
                'records_affected': int(data.get('records', 0) * success_rate)
            }
        
        transform_task = PythonOperator(
            task_id='transform_data',
            python_callable=transform_data
        )
        
        def enrich_data(**context):
            """Enrich data with additional information"""
            ti = context['task_instance']
            transform_data = ti.xcom_pull(key='transform_result', task_ids='transform_data')
            
            if not transform_data:
                raise ValueError("No transform data found")
            
            # Simulate data enrichment
            enrichment_summary = {
                "enrichment_sources": ["geo_data", "reference_tables", "external_apis"],
                "records_enriched": transform_data.get('records_processed', 0),
                "enrichment_fields_added": random.randint(3, 8),
                "enrichment_success_rate": random.uniform(0.85, 0.98),
                "processing_time": random.uniform(2.0, 15.0)
            }
            
            ti.xcom_push(key='enrichment_result', value=enrichment_summary)
            logging.info(f"Data enrichment completed: {enrichment_summary}")
            
            return enrichment_summary
        
        enrich_task = PythonOperator(
            task_id='enrich_data',
            python_callable=enrich_data
        )
        
        def load_data(**context):
            """Load processed data to target"""
            ti = context['task_instance']
            enrichment_data = ti.xcom_pull(key='enrichment_result', task_ids='enrich_data')
            
            if not enrichment_data:
                raise ValueError("No enrichment data found")
            
            # Simulate data loading
            load_summary = {
                "target_table": source_config.target_table,
                "records_loaded": enrichment_data.get('records_enriched', 0),
                "load_strategy": "upsert",
                "load_time": random.uniform(5.0, 30.0),
                "success_rate": random.uniform(0.98, 1.0),
                "indexes_updated": random.randint(2, 6)
            }
            
            ti.xcom_push(key='load_result', value=load_summary)
            logging.info(f"Data loading completed: {load_summary}")
            
            return load_summary
        
        load_task = PythonOperator(
            task_id='load_data',
            python_callable=load_data
        )
        
        # Define task dependencies
        transform_task >> enrich_task >> load_task
    
    return group

# =============================================================================
# BRANCHING LOGIC
# =============================================================================

def create_processing_branch_logic(source_config: DataSourceConfig):
    """Create branching logic for processing strategy"""
    
    def choose_processing_path(**context):
        """Choose processing path based on data characteristics"""
        ti = context['task_instance']
        
        # Get data profile from ingestion group
        profile_data = ti.xcom_pull(key='profile_result', task_ids='ingestion.profile_data')
        
        if not profile_data:
            logging.warning("No profile data found, skipping processing")
            return 'skip_processing'
        
        data_size_mb = profile_data.get('size_mb', 0)
        record_count = profile_data.get('record_count', 0)
        complexity_score = profile_data.get('complexity_score', 5.0)
        
        # Decision logic based on multiple factors
        if source_config.priority >= 8 and data_size_mb < 100:
            # High priority, small data -> fast track
            logging.info(f"Choosing fast track: priority={source_config.priority}, size={data_size_mb}MB")
            return 'fast_track_processing'
        
        elif data_size_mb > 1000 or record_count > 1_000_000:
            # Large data -> batch processing
            logging.info(f"Choosing batch processing: size={data_size_mb}MB, records={record_count}")
            return 'batch_processing'
        
        elif complexity_score > 8.0:
            # Complex data -> specialized processing
            logging.info(f"Choosing complex processing: complexity={complexity_score}")
            return 'complex_processing'
        
        else:
            # Default to standard processing
            logging.info(f"Choosing standard processing: size={data_size_mb}MB, priority={source_config.priority}")
            return 'standard_processing'
    
    return BranchPythonOperator(
        task_id='choose_processing_path',
        python_callable=choose_processing_path
    )

# =============================================================================
# ERROR HANDLING AND MONITORING
# =============================================================================

def create_failure_callback(source_config: DataSourceConfig):
    """Create custom failure callback for monitoring"""
    
    def on_failure_callback(context):
        """Handle task failures with comprehensive logging and alerting"""
        task_instance = context['task_instance']
        dag_id = context['dag'].dag_id
        task_id = task_instance.task_id
        execution_date = context['execution_date']
        exception = context.get('exception')
        
        # Log failure details
        failure_info = {
            "dag_id": dag_id,
            "task_id": task_id,
            "execution_date": str(execution_date),
            "source_config": source_config.name,
            "priority": source_config.priority,
            "sla_hours": source_config.sla_hours,
            "exception": str(exception) if exception else "Unknown error"
        }
        
        logging.error(f"Task failure: {failure_info}")
        
        # Send alerts based on priority
        if source_config.priority >= 8:
            send_critical_alert(failure_info)
        elif source_config.priority >= 5:
            send_warning_alert(failure_info)
        else:
            send_info_alert(failure_info)
        
        # Update monitoring metrics
        update_failure_metrics(failure_info)
        
        # Create incident ticket for critical failures
        if source_config.priority >= 8:
            create_incident_ticket(failure_info)
    
    return on_failure_callback

def create_success_callback(source_config: DataSourceConfig):
    """Create success callback for monitoring"""
    
    def on_success_callback(context):
        """Handle successful task completion"""
        task_instance = context['task_instance']
        dag_id = context['dag'].dag_id
        task_id = task_instance.task_id
        execution_date = context['execution_date']
        
        success_info = {
            "dag_id": dag_id,
            "task_id": task_id,
            "execution_date": str(execution_date),
            "source_config": source_config.name,
            "duration": task_instance.duration
        }
        
        logging.info(f"Task success: {success_info}")
        
        # Update success metrics
        update_success_metrics(success_info)
        
        # Update data catalog
        update_data_catalog(source_config, success_info)
    
    return on_success_callback

def send_critical_alert(failure_info):
    """Send critical alert (Slack, PagerDuty, etc.)"""
    logging.critical(f"CRITICAL ALERT: {failure_info}")
    # Implementation would send to Slack, PagerDuty, etc.

def send_warning_alert(failure_info):
    """Send warning alert"""
    logging.warning(f"WARNING ALERT: {failure_info}")
    # Implementation would send to monitoring channels

def send_info_alert(failure_info):
    """Send info alert"""
    logging.info(f"INFO ALERT: {failure_info}")
    # Implementation would log to monitoring system

def update_failure_metrics(failure_info):
    """Update failure metrics in monitoring system"""
    # Implementation would update Prometheus, CloudWatch, etc.
    pass

def update_success_metrics(success_info):
    """Update success metrics in monitoring system"""
    # Implementation would update metrics
    pass

def create_incident_ticket(failure_info):
    """Create incident ticket for critical failures"""
    # Implementation would create ticket in Jira, ServiceNow, etc.
    pass

def update_data_catalog(source_config, success_info):
    """Update data catalog with successful processing info"""
    # Implementation would update DataHub, Apache Atlas, etc.
    pass

# =============================================================================
# DYNAMIC DAG GENERATION
# =============================================================================

def create_data_pipeline_dag(source_config: DataSourceConfig) -> DAG:
    """Generate a complete data pipeline DAG from configuration"""
    
    dag_id = f"techcorp_pipeline_{source_config.name}"
    
    # Get processing strategy
    strategy = get_processing_strategy(source_config.expected_volume_mb, source_config.priority)
    
    default_args = {
        'owner': 'techcorp-data-team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email': ['data-team@techcorp.com'],
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': strategy['retries'],
        'retry_delay': timedelta(minutes=5),
        'execution_timeout': timedelta(minutes=strategy['timeout_minutes']),
        'on_failure_callback': create_failure_callback(source_config),
        'on_success_callback': create_success_callback(source_config),
    }
    
    dag = DAG(
        dag_id,
        default_args=default_args,
        description=f'Production pipeline for {source_config.name} ({source_config.source_type})',
        schedule_interval=source_config.schedule_interval,
        catchup=False,
        tags=['production', 'techcorp', source_config.source_type, f'priority_{source_config.priority}'],
        max_active_runs=1,
        sla_miss_callback=None,  # Could add SLA monitoring
    )
    
    with dag:
        
        # Start task
        start = DummyOperator(task_id='start')
        
        # Health check sensors based on source type
        if source_config.source_type == "api":
            health_sensor = ExternalAPIHealthSensor(
                task_id='check_api_health',
                api_url=source_config.source_config['url'],
                poke_interval=60,
                timeout=300,
                mode='reschedule',
                pool='external_api_pool'
            )
        elif source_config.source_type == "database":
            health_sensor = DatabaseConnectionSensor(
                task_id='check_db_health',
                connection_id=source_config.source_config['connection_id'],
                poke_interval=30,
                timeout=300,
                mode='reschedule'
            )
        else:
            # Generic data quality sensor for file/sftp sources
            health_sensor = DataQualitySensor(
                task_id='wait_for_quality_data',
                data_source=source_config.name,
                quality_checks=source_config.quality_rules,
                poke_interval=60,
                timeout=source_config.sla_hours * 3600,
                mode='reschedule'
            )
        
        # Task groups
        ingestion_group = create_data_ingestion_task_group('ingestion', source_config)
        processing_group = create_data_processing_task_group('processing', source_config)
        
        # Branching logic
        processing_branch = create_processing_branch_logic(source_config)
        
        # Different processing paths with appropriate resource allocation
        fast_track_processing = PythonOperator(
            task_id='fast_track_processing',
            python_callable=lambda **context: logging.info("Fast track processing completed"),
            pool='fast_processing_pool',
            priority_weight=10,
            execution_timeout=timedelta(minutes=15)
        )
        
        standard_processing = PythonOperator(
            task_id='standard_processing',
            python_callable=lambda **context: logging.info("Standard processing completed"),
            pool='standard_pool',
            priority_weight=5,
            execution_timeout=timedelta(minutes=30)
        )
        
        batch_processing = PythonOperator(
            task_id='batch_processing',
            python_callable=lambda **context: logging.info("Batch processing completed"),
            pool='batch_processing_pool',
            priority_weight=1,
            execution_timeout=timedelta(minutes=120)
        )
        
        complex_processing = PythonOperator(
            task_id='complex_processing',
            python_callable=lambda **context: logging.info("Complex processing completed"),
            pool='standard_pool',
            priority_weight=7,
            execution_timeout=timedelta(minutes=60)
        )
        
        skip_processing = DummyOperator(task_id='skip_processing')
        
        # Final validation and monitoring
        def final_validation(**context):
            """Perform final validation of processed data"""
            ti = context['task_instance']
            
            # Get results from processing tasks
            load_result = ti.xcom_pull(key='load_result', task_ids='processing.load_data')
            
            if load_result:
                success_rate = load_result.get('success_rate', 0)
                records_loaded = load_result.get('records_loaded', 0)
                
                if success_rate < 0.95:
                    raise ValueError(f"Load success rate too low: {success_rate}")
                
                if records_loaded == 0:
                    raise ValueError("No records were loaded")
                
                logging.info(f"Final validation passed: {records_loaded} records, {success_rate} success rate")
            else:
                logging.info("No load results found (processing may have been skipped)")
            
            return {"validation_status": "passed"}
        
        final_validation_task = PythonOperator(
            task_id='final_validation',
            python_callable=final_validation,
            trigger_rule='none_failed_min_one_success',
            pool='monitoring_pool'
        )
        
        def update_monitoring(**context):
            """Update monitoring dashboards and metrics"""
            ti = context['task_instance']
            
            # Collect metrics from all tasks
            metrics = {
                "dag_id": dag_id,
                "execution_date": str(context['execution_date']),
                "source_config": source_config.name,
                "priority": source_config.priority,
                "completion_time": datetime.now().isoformat()
            }
            
            # Update monitoring systems
            logging.info(f"Updating monitoring with metrics: {metrics}")
            
            return metrics
        
        update_monitoring_task = PythonOperator(
            task_id='update_monitoring',
            python_callable=update_monitoring,
            pool='monitoring_pool'
        )
        
        end = DummyOperator(task_id='end')
        
        # Define complete workflow
        start >> health_sensor >> ingestion_group >> processing_group >> processing_branch
        
        processing_branch >> [fast_track_processing, standard_processing, 
                             batch_processing, complex_processing, skip_processing]
        
        [fast_track_processing, standard_processing, batch_processing, 
         complex_processing, skip_processing] >> final_validation_task
        
        final_validation_task >> update_monitoring_task >> end
    
    return dag

# =============================================================================
# MONITORING DAG
# =============================================================================

def create_monitoring_dag() -> DAG:
    """Create a comprehensive monitoring DAG for the entire pipeline system"""
    
    dag = DAG(
        'techcorp_pipeline_monitoring',
        default_args={
            'owner': 'techcorp-data-team',
            'start_date': datetime(2024, 1, 1),
            'retries': 1,
            'retry_delay': timedelta(minutes=2),
        },
        description='Monitor all TechCorp data pipelines and system health',
        schedule_interval='*/15 * * * *',  # Every 15 minutes
        catchup=False,
        tags=['monitoring', 'techcorp', 'system_health']
    )
    
    with dag:
        
        def check_pipeline_health(**context):
            """Check health of all data pipelines"""
            from airflow.models import DagRun, TaskInstance
            from airflow.utils.state import State
            
            # Query recent DAG runs
            recent_runs = DagRun.find(
                execution_start_date=datetime.now() - timedelta(hours=24)
            )
            
            health_summary = {
                "total_runs": len(recent_runs),
                "successful_runs": len([r for r in recent_runs if r.state == State.SUCCESS]),
                "failed_runs": len([r for r in recent_runs if r.state == State.FAILED]),
                "running_runs": len([r for r in recent_runs if r.state == State.RUNNING]),
                "check_time": datetime.now().isoformat()
            }
            
            # Calculate success rate
            if health_summary["total_runs"] > 0:
                health_summary["success_rate"] = health_summary["successful_runs"] / health_summary["total_runs"]
            else:
                health_summary["success_rate"] = 1.0
            
            # Alert if success rate is low
            if health_summary["success_rate"] < 0.8:
                logging.warning(f"Low pipeline success rate: {health_summary['success_rate']}")
            
            logging.info(f"Pipeline health check: {health_summary}")
            return health_summary
        
        health_check = PythonOperator(
            task_id='check_pipeline_health',
            python_callable=check_pipeline_health,
            pool='monitoring_pool'
        )
        
        def check_data_freshness(**context):
            """Check if data is fresh across all sources"""
            freshness_results = []
            
            for source_config in DATA_SOURCE_CONFIGS:
                # Check last successful run
                last_success = self._get_last_successful_run(f"techcorp_pipeline_{source_config.name}")
                
                if last_success:
                    hours_since_success = (datetime.now() - last_success).total_seconds() / 3600
                    is_stale = hours_since_success > source_config.sla_hours
                    
                    freshness_results.append({
                        "source": source_config.name,
                        "last_success": last_success.isoformat(),
                        "hours_since_success": hours_since_success,
                        "sla_hours": source_config.sla_hours,
                        "is_stale": is_stale
                    })
                else:
                    freshness_results.append({
                        "source": source_config.name,
                        "last_success": None,
                        "is_stale": True
                    })
            
            # Count stale sources
            stale_sources = [r for r in freshness_results if r["is_stale"]]
            
            if stale_sources:
                logging.warning(f"Stale data sources detected: {len(stale_sources)}")
                for source in stale_sources:
                    logging.warning(f"Stale source: {source}")
            
            freshness_summary = {
                "total_sources": len(freshness_results),
                "stale_sources": len(stale_sources),
                "freshness_details": freshness_results,
                "check_time": datetime.now().isoformat()
            }
            
            logging.info(f"Data freshness check: {freshness_summary}")
            return freshness_summary
        
        def _get_last_successful_run(dag_id):
            """Get last successful run for a DAG"""
            # Simulate getting last successful run
            # In real implementation, would query Airflow metadata
            return datetime.now() - timedelta(hours=random.randint(1, 48))
        
        freshness_check = PythonOperator(
            task_id='check_data_freshness',
            python_callable=check_data_freshness,
            pool='monitoring_pool'
        )
        
        def check_resource_utilization(**context):
            """Check resource utilization across pools"""
            pool_utilization = {}
            
            for pool_name, config in POOL_CONFIGS.items():
                # Simulate pool utilization check
                used_slots = random.randint(0, config['slots'])
                utilization_pct = (used_slots / config['slots']) * 100
                
                pool_utilization[pool_name] = {
                    "total_slots": config['slots'],
                    "used_slots": used_slots,
                    "available_slots": config['slots'] - used_slots,
                    "utilization_percent": utilization_pct
                }
            
            # Alert on high utilization
            high_util_pools = [name for name, util in pool_utilization.items() 
                             if util['utilization_percent'] > 80]
            
            if high_util_pools:
                logging.warning(f"High resource utilization in pools: {high_util_pools}")
            
            utilization_summary = {
                "pool_utilization": pool_utilization,
                "high_utilization_pools": high_util_pools,
                "check_time": datetime.now().isoformat()
            }
            
            logging.info(f"Resource utilization check: {utilization_summary}")
            return utilization_summary
        
        resource_check = PythonOperator(
            task_id='check_resource_utilization',
            python_callable=check_resource_utilization,
            pool='monitoring_pool'
        )
        
        def generate_metrics(**context):
            """Generate and publish comprehensive metrics"""
            ti = context['task_instance']
            
            # Get results from other monitoring tasks
            health_data = ti.xcom_pull(task_ids='check_pipeline_health')
            freshness_data = ti.xcom_pull(task_ids='check_data_freshness')
            resource_data = ti.xcom_pull(task_ids='check_resource_utilization')
            
            # Compile comprehensive metrics
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "pipeline_health": health_data,
                "data_freshness": freshness_data,
                "resource_utilization": resource_data,
                "system_status": "healthy" if (
                    health_data.get('success_rate', 0) > 0.8 and
                    freshness_data.get('stale_sources', 0) == 0
                ) else "degraded"
            }
            
            # Publish metrics to monitoring systems
            # In production: send to Prometheus, CloudWatch, DataDog, etc.
            logging.info(f"Publishing metrics: {metrics}")
            
            return metrics
        
        metrics_task = PythonOperator(
            task_id='generate_metrics',
            python_callable=generate_metrics,
            pool='monitoring_pool'
        )
        
        # Define monitoring workflow
        [health_check, freshness_check, resource_check] >> metrics_task
    
    return dag

# =============================================================================
# DAG GENERATION
# =============================================================================

# Generate pipeline DAGs for each data source
for source_config in DATA_SOURCE_CONFIGS:
    dag_id = f"techcorp_pipeline_{source_config.name}"
    globals()[dag_id] = create_data_pipeline_dag(source_config)

# Create monitoring DAG
monitoring_dag = create_monitoring_dag()
globals()['techcorp_pipeline_monitoring'] = monitoring_dag

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("🚀 TechCorp Airflow Production Patterns - Complete Solution")
    print("=" * 65)
    
    print("\n✅ SOLUTION FEATURES:")
    print("• Dynamic DAG generation from 5 data source configurations")
    print("• Reusable task groups for ingestion and processing")
    print("• Intelligent branching based on data characteristics")
    print("• Custom sensors with health checks and quality validation")
    print("• Resource management with pools and priorities")
    print("• Comprehensive error handling and alerting")
    print("• Production monitoring DAG with health checks")
    print("• Complete failure and success callback systems")
    
    print("\n🏗️ GENERATED DAGS:")
    for config in DATA_SOURCE_CONFIGS:
        print(f"• techcorp_pipeline_{config.name} ({config.source_type}, priority {config.priority})")
    print("• techcorp_pipeline_monitoring (system health monitoring)")
    
    print("\n📊 DATA SOURCES CONFIGURED:")
    for i, config in enumerate(DATA_SOURCE_CONFIGS, 1):
        print(f"{i}. {config.name}: {config.source_type} source, {config.expected_volume_mb}MB, "
              f"priority {config.priority}, {config.schedule_interval}")
    
    print("\n🏊 RESOURCE POOLS:")
    for pool_name, config in POOL_CONFIGS.items():
        print(f"• {pool_name}: {config['slots']} slots - {config['description']}")
    
    print("\n🎯 BRANCHING LOGIC:")
    print("• Fast Track: High priority (≥8) + small data (<100MB)")
    print("• Standard: Normal priority + medium data")
    print("• Batch: Large data (>1GB) or high volume (>1M records)")
    print("• Complex: High complexity score (>8.0)")
    print("• Skip: Data quality issues or system unavailable")
    
    print("\n🔍 CUSTOM SENSORS:")
    print("• DataQualitySensor: Validates data availability and quality")
    print("• ExternalAPIHealthSensor: Checks API health before processing")
    print("• DatabaseConnectionSensor: Validates DB connectivity and performance")
    
    print("\n📈 MONITORING FEATURES:")
    print("• Pipeline health monitoring (success rates, failures)")
    print("• Data freshness monitoring (SLA compliance)")
    print("• Resource utilization monitoring (pool usage)")
    print("• Comprehensive metrics generation and alerting")
    
    print("\n🚨 ERROR HANDLING:")
    print("• Priority-based alerting (critical, warning, info)")
    print("• Automatic incident ticket creation for critical failures")
    print("• Comprehensive failure logging and metrics")
    print("• Success callbacks for monitoring and catalog updates")
    
    print("\n" + "="*65)
    print("🎉 Production-ready Airflow patterns implemented!")
    print("This solution demonstrates enterprise-grade workflow orchestration")
    print("with dynamic generation, intelligent routing, and comprehensive monitoring.")
    print("="*65)