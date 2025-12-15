"""
Day 15: Airflow Production Patterns - Exercise

Build a production-grade data processing platform with advanced Airflow patterns.

Scenario:
You're the Lead Data Engineer at "TechCorp", a SaaS company processing customer data 
from multiple sources. You need to build a scalable, maintainable pipeline system 
that can handle varying data volumes and adapt to changing business requirements.

Business Context:
- Process data from 5+ different sources (APIs, databases, files)
- Data volumes vary from 1MB to 10GB per source
- Different sources have different SLAs and processing requirements
- Need to handle failures gracefully and provide comprehensive monitoring
- System must scale to handle 50+ similar pipelines

Your Task:
Build an advanced Airflow system demonstrating production patterns.

Requirements:
1. Dynamic DAG generation from configuration
2. Task groups for reusable ETL patterns
3. Intelligent branching based on data characteristics
4. Resource management with pools and priorities
5. Advanced sensors with validation
6. Comprehensive error handling and monitoring
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import logging
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
    source_type: str  # 'api', 'database', 'file', 'sftp'
    source_config: Dict[str, Any]
    target_table: str
    schedule_interval: str
    priority: int
    expected_volume_mb: int
    sla_hours: int
    quality_rules: List[Dict[str, Any]]
    transformations: List[Dict[str, Any]]

# TODO: Define data source configurations
# Create configurations for different data sources that will drive dynamic DAG generation
DATA_SOURCE_CONFIGS = [
    # Example structure - you need to complete this
    # DataSourceConfig(
    #     name="customer_data",
    #     source_type="api",
    #     source_config={"url": "https://api.techcorp.com/customers", "auth_type": "bearer"},
    #     target_table="customers",
    #     schedule_interval="@hourly",
    #     priority=8,
    #     expected_volume_mb=50,
    #     sla_hours=2,
    #     quality_rules=[{"type": "not_null", "columns": ["customer_id", "email"]}],
    #     transformations=[{"type": "email_normalize"}, {"type": "pii_hash"}]
    # ),
    # Add 4 more data source configurations here
]

# TODO: Define pool configurations
# Create pool configurations for resource management
POOL_CONFIGS = {
    # Example: "large_processing": {"slots": 2, "description": "Large data processing"},
    # Add pool configurations here
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def setup_pools():
    """Setup Airflow pools for resource management"""
    # TODO: Implement pool setup logic
    # This would typically be done via Airflow CLI or UI
    # For exercise purposes, document the pools that should be created
    pass

def get_processing_strategy(data_size_mb: int, priority: int) -> Dict[str, Any]:
    """Determine processing strategy based on data characteristics"""
    # TODO: Implement logic to determine:
    # - Which pool to use
    # - Processing timeout
    # - Retry strategy
    # - Priority weight
    
    strategy = {
        "pool": "default_pool",
        "timeout_minutes": 30,
        "retries": 2,
        "priority_weight": 5
    }
    
    # Add your logic here to modify strategy based on data_size_mb and priority
    
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
        # TODO: Implement data quality checking logic
        # 1. Check if data source has new data
        # 2. Validate data quality rules
        # 3. Return True if data is ready and quality is acceptable
        
        self.log.info(f"Checking data quality for {self.data_source}")
        
        # Placeholder implementation - replace with actual logic
        import random
        return random.choice([True, False])  # Simulate random availability

# TODO: Create additional custom sensors
class ExternalAPIHealthSensor(BaseSensorOperator):
    """Sensor that checks if external API is healthy before processing"""
    
    def __init__(self, api_url: str, expected_status: int = 200, **kwargs):
        super().__init__(**kwargs)
        self.api_url = api_url
        self.expected_status = expected_status
    
    def poke(self, context):
        """Check if external API is healthy"""
        # TODO: Implement API health check
        # 1. Make HTTP request to API health endpoint
        # 2. Check response status and content
        # 3. Return True if API is healthy
        pass

# =============================================================================
# TASK GROUP FACTORIES
# =============================================================================

def create_data_ingestion_task_group(group_id: str, source_config: DataSourceConfig) -> TaskGroup:
    """Create reusable data ingestion task group"""
    
    with TaskGroup(group_id=group_id) as group:
        
        # TODO: Create data ingestion tasks
        # 1. Data extraction task
        def extract_data(**context):
            """Extract data from source"""
            # TODO: Implement data extraction logic based on source_config.source_type
            # Handle different source types: api, database, file, sftp
            
            ti = context['task_instance']
            
            # Placeholder - replace with actual extraction logic
            extracted_data = {
                "source": source_config.name,
                "records": 1000,  # Simulate extracted record count
                "size_mb": source_config.expected_volume_mb,
                "extraction_time": datetime.now().isoformat()
            }
            
            ti.xcom_push(key='extraction_result', value=extracted_data)
            return extracted_data
        
        extract_task = PythonOperator(
            task_id='extract_data',
            python_callable=extract_data
        )
        
        # TODO: Create data validation task
        def validate_data(**context):
            """Validate extracted data quality"""
            # TODO: Implement data quality validation
            # 1. Get extraction results from XCom
            # 2. Apply quality rules from source_config.quality_rules
            # 3. Log validation results
            # 4. Fail if critical quality issues found
            pass
        
        validate_task = PythonOperator(
            task_id='validate_data',
            python_callable=validate_data
        )
        
        # TODO: Create data profiling task
        def profile_data(**context):
            """Profile data characteristics for processing decisions"""
            # TODO: Implement data profiling
            # 1. Analyze data size, complexity, patterns
            # 2. Determine optimal processing strategy
            # 3. Store profiling results in XCom
            pass
        
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
        
        # TODO: Create data transformation tasks
        def transform_data(**context):
            """Apply transformations to data"""
            # TODO: Implement data transformations
            # 1. Get data from previous task group
            # 2. Apply transformations from source_config.transformations
            # 3. Handle different transformation types
            pass
        
        transform_task = PythonOperator(
            task_id='transform_data',
            python_callable=transform_data
        )
        
        # TODO: Create data enrichment task
        def enrich_data(**context):
            """Enrich data with additional information"""
            # TODO: Implement data enrichment
            # 1. Add derived fields
            # 2. Lookup reference data
            # 3. Apply business rules
            pass
        
        enrich_task = PythonOperator(
            task_id='enrich_data',
            python_callable=enrich_data
        )
        
        # TODO: Create data loading task
        def load_data(**context):
            """Load processed data to target"""
            # TODO: Implement data loading
            # 1. Connect to target system
            # 2. Load data using appropriate strategy (insert, upsert, merge)
            # 3. Handle loading errors gracefully
            pass
        
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
        # TODO: Implement intelligent branching logic
        # 1. Get data profiling results from XCom
        # 2. Analyze data volume, complexity, priority
        # 3. Choose appropriate processing path:
        #    - 'fast_track_processing' for small, high-priority data
        #    - 'standard_processing' for normal data
        #    - 'batch_processing' for large, low-priority data
        #    - 'skip_processing' if data quality issues
        
        ti = context['task_instance']
        
        # Get extraction results (placeholder logic)
        extraction_data = ti.xcom_pull(key='extraction_result', task_ids='ingestion.extract_data')
        
        if not extraction_data:
            return 'skip_processing'
        
        data_size_mb = extraction_data.get('size_mb', 0)
        record_count = extraction_data.get('records', 0)
        
        # TODO: Implement your branching logic here
        # Example logic (replace with your implementation):
        if source_config.priority >= 8 and data_size_mb < 100:
            return 'fast_track_processing'
        elif data_size_mb > 1000 or record_count > 1_000_000:
            return 'batch_processing'
        else:
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
        # TODO: Implement failure handling
        # 1. Log failure details
        # 2. Send alerts (Slack, email, PagerDuty)
        # 3. Update monitoring metrics
        # 4. Create incident tickets for critical failures
        
        task_instance = context['task_instance']
        dag_id = context['dag'].dag_id
        task_id = task_instance.task_id
        execution_date = context['execution_date']
        
        logging.error(f"Task failed: {dag_id}.{task_id} on {execution_date}")
        
        # TODO: Add your alerting logic here
        # Example: send_slack_alert(), create_incident_ticket(), etc.
    
    return on_failure_callback

def create_success_callback(source_config: DataSourceConfig):
    """Create success callback for monitoring"""
    
    def on_success_callback(context):
        """Handle successful task completion"""
        # TODO: Implement success handling
        # 1. Update monitoring metrics
        # 2. Log success metrics
        # 3. Update data catalog
        # 4. Send success notifications if needed
        pass
    
    return on_success_callback

# =============================================================================
# DYNAMIC DAG GENERATION
# =============================================================================

def create_data_pipeline_dag(source_config: DataSourceConfig) -> DAG:
    """Generate a complete data pipeline DAG from configuration"""
    
    dag_id = f"techcorp_pipeline_{source_config.name}"
    
    # TODO: Create DAG with appropriate configuration
    default_args = {
        'owner': 'techcorp-data-team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
        'on_failure_callback': create_failure_callback(source_config),
        'on_success_callback': create_success_callback(source_config),
    }
    
    # Get processing strategy
    strategy = get_processing_strategy(source_config.expected_volume_mb, source_config.priority)
    
    dag = DAG(
        dag_id,
        default_args=default_args,
        description=f'Production pipeline for {source_config.name}',
        schedule_interval=source_config.schedule_interval,
        catchup=False,
        tags=['production', 'techcorp', source_config.source_type],
        max_active_runs=1,
        sla_miss_callback=None,  # TODO: Add SLA monitoring
    )
    
    with dag:
        
        # TODO: Create start task
        start = DummyOperator(task_id='start')
        
        # TODO: Create data quality sensor
        data_ready_sensor = DataQualitySensor(
            task_id='wait_for_quality_data',
            data_source=source_config.name,
            quality_checks=source_config.quality_rules,
            poke_interval=60,
            timeout=source_config.sla_hours * 3600,
            mode='reschedule'
        )
        
        # TODO: Create task groups
        ingestion_group = create_data_ingestion_task_group('ingestion', source_config)
        processing_group = create_data_processing_task_group('processing', source_config)
        
        # TODO: Create branching logic
        processing_branch = create_processing_branch_logic(source_config)
        
        # TODO: Create different processing paths
        fast_track_processing = PythonOperator(
            task_id='fast_track_processing',
            python_callable=lambda **context: logging.info("Fast track processing"),
            pool='fast_processing_pool',
            priority_weight=10
        )
        
        standard_processing = PythonOperator(
            task_id='standard_processing',
            python_callable=lambda **context: logging.info("Standard processing"),
            pool='standard_pool',
            priority_weight=5
        )
        
        batch_processing = PythonOperator(
            task_id='batch_processing',
            python_callable=lambda **context: logging.info("Batch processing"),
            pool='batch_processing_pool',
            priority_weight=1
        )
        
        skip_processing = DummyOperator(task_id='skip_processing')
        
        # TODO: Create final validation and monitoring tasks
        final_validation = PythonOperator(
            task_id='final_validation',
            python_callable=lambda **context: logging.info("Final validation complete"),
            trigger_rule='none_failed_min_one_success'
        )
        
        update_monitoring = PythonOperator(
            task_id='update_monitoring',
            python_callable=lambda **context: logging.info("Monitoring updated")
        )
        
        end = DummyOperator(task_id='end')
        
        # TODO: Define task dependencies
        # Create the complete workflow:
        # start -> sensor -> ingestion -> processing -> branch -> [processing paths] -> validation -> monitoring -> end
        
        start >> data_ready_sensor >> ingestion_group >> processing_group >> processing_branch
        
        processing_branch >> [fast_track_processing, standard_processing, batch_processing, skip_processing]
        
        [fast_track_processing, standard_processing, batch_processing, skip_processing] >> final_validation
        
        final_validation >> update_monitoring >> end
    
    return dag

# =============================================================================
# MONITORING AND ALERTING
# =============================================================================

def create_monitoring_dag() -> DAG:
    """Create a monitoring DAG for the entire pipeline system"""
    
    dag = DAG(
        'techcorp_pipeline_monitoring',
        default_args={
            'owner': 'techcorp-data-team',
            'start_date': datetime(2024, 1, 1),
        },
        description='Monitor all TechCorp data pipelines',
        schedule_interval='*/15 * * * *',  # Every 15 minutes
        catchup=False,
        tags=['monitoring', 'techcorp']
    )
    
    with dag:
        
        # TODO: Create monitoring tasks
        def check_pipeline_health(**context):
            """Check health of all data pipelines"""
            # TODO: Implement pipeline health checking
            # 1. Query Airflow metadata database
            # 2. Check for failed tasks, long-running tasks
            # 3. Validate SLA compliance
            # 4. Generate health report
            pass
        
        health_check = PythonOperator(
            task_id='check_pipeline_health',
            python_callable=check_pipeline_health
        )
        
        def check_data_freshness(**context):
            """Check if data is fresh across all sources"""
            # TODO: Implement data freshness checking
            # 1. Check last successful run for each pipeline
            # 2. Compare against expected schedules
            # 3. Alert on stale data
            pass
        
        freshness_check = PythonOperator(
            task_id='check_data_freshness',
            python_callable=check_data_freshness
        )
        
        def generate_metrics(**context):
            """Generate and publish metrics"""
            # TODO: Implement metrics generation
            # 1. Calculate pipeline success rates
            # 2. Measure data processing volumes
            # 3. Track SLA compliance
            # 4. Publish to monitoring system (Prometheus, CloudWatch, etc.)
            pass
        
        metrics_task = PythonOperator(
            task_id='generate_metrics',
            python_callable=generate_metrics
        )
        
        # Define monitoring workflow
        [health_check, freshness_check] >> metrics_task
    
    return dag

# =============================================================================
# EXERCISE INSTRUCTIONS
# =============================================================================

def print_exercise_instructions():
    """Print detailed exercise instructions"""
    
    print("🎯 Airflow Production Patterns Exercise - TechCorp Data Platform")
    print("=" * 70)
    
    print("\n📋 REQUIREMENTS:")
    print("1. Complete DATA_SOURCE_CONFIGS with 5 different data sources")
    print("2. Implement dynamic DAG generation from configurations")
    print("3. Create reusable task groups for ingestion and processing")
    print("4. Implement intelligent branching based on data characteristics")
    print("5. Configure resource management with pools and priorities")
    print("6. Create custom sensors with data quality validation")
    print("7. Implement comprehensive error handling and monitoring")
    print("8. Create a monitoring DAG for the entire system")
    
    print("\n🚀 GETTING STARTED:")
    print("1. Complete DATA_SOURCE_CONFIGS with 5 data sources")
    print("2. Implement get_processing_strategy() function")
    print("3. Complete custom sensor implementations")
    print("4. Finish task group factory functions")
    print("5. Implement branching logic")
    print("6. Add error handling and monitoring")
    print("7. Test with 'python exercise.py'")

if __name__ == "__main__":
    print_exercise_instructions()