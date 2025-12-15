"""
Day 16: Airflow at Scale - Exercise

Deploy a production-ready, scalable Airflow cluster with comprehensive monitoring.

Scenario:
You're the Platform Engineer at "ScaleCorp", a rapidly growing data company. 
The current Airflow deployment can't handle the increasing workload of 500+ DAGs 
and 50,000+ daily tasks. You need to design and implement a scalable, highly 
available Airflow architecture.

Business Context:
- Current system: Single-node Airflow with LocalExecutor
- Growth: 10x increase in data processing needs over 6 months
- Requirements: 99.9% uptime, auto-scaling, comprehensive monitoring
- Constraints: Cost optimization, security compliance, operational simplicity

Your Task:
Design and implement an enterprise-scale Airflow deployment.

Requirements:
1. Distributed execution with CeleryExecutor
2. High availability with multi-scheduler setup
3. Comprehensive monitoring with Prometheus and Grafana
4. Auto-scaling based on queue length and resource utilization
5. Production-grade error handling and alerting
6. Performance optimization and cost management
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import logging
import os
from dataclasses import dataclass

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.email import send_email
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook

# =============================================================================
# CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ExecutorConfig:
    """Configuration for Airflow executor"""
    executor_type: str  # 'LocalExecutor', 'CeleryExecutor', 'KubernetesExecutor'
    parallelism: int
    max_active_runs_per_dag: int
    max_active_tasks_per_dag: int
    worker_concurrency: int
    broker_url: Optional[str] = None
    result_backend: Optional[str] = None

@dataclass
class MonitoringConfig:
    """Configuration for monitoring and alerting"""
    prometheus_enabled: bool
    grafana_enabled: bool
    statsd_host: str
    statsd_port: int
    alert_email: List[str]
    slack_webhook: Optional[str] = None
    pagerduty_key: Optional[str] = None

@dataclass
class ScalingConfig:
    """Configuration for auto-scaling"""
    min_workers: int
    max_workers: int
    scale_up_threshold: int  # Queue length to trigger scale up
    scale_down_threshold: int  # Queue length to trigger scale down
    cpu_threshold: float  # CPU utilization threshold
    memory_threshold: float  # Memory utilization threshold

# TODO: Define production configurations
# Create configurations for ScaleCorp's production environment
EXECUTOR_CONFIG = ExecutorConfig(
    # TODO: Configure for CeleryExecutor with appropriate settings
    executor_type="CeleryExecutor",
    parallelism=128,  # Adjust based on requirements
    max_active_runs_per_dag=16,
    max_active_tasks_per_dag=16,
    worker_concurrency=16,
    broker_url=os.getenv("CELERY_BROKER_URL", "redis://redis-cluster:6379/0"),
    result_backend=os.getenv("CELERY_RESULT_BACKEND", "db+postgresql://airflow:${POSTGRES_PASSWORD}@postgres-ha:5432/airflow")
)

MONITORING_CONFIG = MonitoringConfig(
    # TODO: Configure monitoring and alerting
    prometheus_enabled=True,
    grafana_enabled=True,
    statsd_host="prometheus-statsd-exporter",
    statsd_port=9125,
    alert_email=os.getenv("ALERT_EMAIL", "platform-team@scalecorp.com").split(","),
    slack_webhook=os.getenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK"),
    pagerduty_key=os.getenv("PAGERDUTY_INTEGRATION_KEY", "YOUR_PAGERDUTY_INTEGRATION_KEY")
)

SCALING_CONFIG = ScalingConfig(
    # TODO: Configure auto-scaling parameters
    min_workers=2,
    max_workers=50,
    scale_up_threshold=20,  # Scale up when queue > 20 tasks
    scale_down_threshold=5,  # Scale down when queue < 5 tasks
    cpu_threshold=70.0,     # Scale up when CPU > 70%
    memory_threshold=80.0   # Scale up when memory > 80%
)

# =============================================================================
# INFRASTRUCTURE CONFIGURATION
# =============================================================================

def generate_docker_compose():
    """Generate Docker Compose configuration for scalable Airflow"""
    
    # TODO: Create comprehensive Docker Compose configuration
    # Include: Redis cluster, PostgreSQL HA, multiple schedulers, workers, monitoring
    
    docker_compose = """
version: '3.8'

services:
  # TODO: Add Redis cluster configuration
  redis-master:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
  
  # TODO: Add PostgreSQL HA configuration
  postgres-primary:
    image: postgres:13
    environment:
      POSTGRES_DB: airflow
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_REPLICATION_MODE: master
      POSTGRES_REPLICATION_USER: replicator
      POSTGRES_REPLICATION_PASSWORD: replicator_password
    volumes:
      - postgres_primary_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
  
  # TODO: Add Airflow scheduler configuration (HA)
  airflow-scheduler-1:
    image: apache/airflow:2.7.0
    command: scheduler
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres-primary/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://redis-master:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres-primary/airflow
      # TODO: Add more configuration
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
    depends_on:
      - postgres-primary
      - redis-master
  
  # TODO: Add Airflow worker configuration (scalable)
  airflow-worker:
    image: apache/airflow:2.7.0
    command: celery worker
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres-primary/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://redis-master:6379/0
      AIRFLOW__CELERY__RESULT_BACKEND: db+postgresql://airflow:airflow@postgres-primary/airflow
      AIRFLOW__CELERY__WORKER_CONCURRENCY: 16
      # TODO: Add monitoring and optimization settings
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
    depends_on:
      - postgres-primary
      - redis-master
    deploy:
      replicas: 3  # Initial worker count
  
  # TODO: Add monitoring stack (Prometheus, Grafana)
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
  
  # TODO: Add more services (Grafana, AlertManager, etc.)

volumes:
  redis_data:
  postgres_primary_data:
  prometheus_data:
  grafana_data:

networks:
  airflow_network:
    driver: bridge
"""
    
    return docker_compose

# =============================================================================
# MONITORING AND ALERTING
# =============================================================================

def setup_prometheus_config():
    """Generate Prometheus configuration for Airflow monitoring"""
    
    # TODO: Create comprehensive Prometheus configuration
    prometheus_config = """
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "airflow_alerts.yml"

scrape_configs:
  # TODO: Add Airflow metrics scraping
  - job_name: 'airflow-scheduler'
    static_configs:
      - targets: ['airflow-scheduler-1:8080', 'airflow-scheduler-2:8080']
    metrics_path: '/admin/metrics'
    scrape_interval: 30s
  
  - job_name: 'airflow-workers'
    static_configs:
      - targets: ['airflow-worker:8793']
    scrape_interval: 30s
  
  # TODO: Add infrastructure monitoring
  - job_name: 'redis'
    static_configs:
      - targets: ['redis-master:6379']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-primary:5432']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
"""
    
    return prometheus_config

def setup_grafana_dashboards():
    """Generate Grafana dashboard configurations"""
    
    # TODO: Create Grafana dashboards for Airflow monitoring
    dashboard_config = {
        "dashboard": {
            "title": "ScaleCorp Airflow Production Monitoring",
            "panels": [
                # TODO: Add panels for key metrics
                {
                    "title": "DAG Success Rate",
                    "type": "stat",
                    "targets": [
                        {
                            "expr": "rate(airflow_dag_run_success_total[1h]) / rate(airflow_dag_run_total[1h]) * 100",
                            "legendFormat": "Success Rate %"
                        }
                    ]
                },
                # TODO: Add more panels
            ]
        }
    }
    
    return dashboard_config

# =============================================================================
# ERROR HANDLING AND ALERTING
# =============================================================================

def create_failure_callback(severity: str = "warning"):
    """Create failure callback with severity-based alerting"""
    
    def failure_callback(context):
        """Handle task failures with comprehensive alerting"""
        # TODO: Implement comprehensive failure handling
        # 1. Extract failure information
        # 2. Determine severity based on task/DAG importance
        # 3. Send appropriate alerts (email, Slack, PagerDuty)
        # 4. Update monitoring metrics
        # 5. Create incident tickets for critical failures
        
        task_instance = context['task_instance']
        dag_id = context['dag'].dag_id
        task_id = task_instance.task_id
        execution_date = context['execution_date']
        exception = context.get('exception')
        
        failure_info = {
            "dag_id": dag_id,
            "task_id": task_id,
            "execution_date": str(execution_date),
            "severity": severity,
            "exception": str(exception) if exception else "Unknown error",
            "log_url": task_instance.log_url
        }
        
        logging.error(f"Task failure: {failure_info}")
        
        # TODO: Implement alerting logic based on severity
        if severity == "critical":
            send_pagerduty_alert(failure_info)
            send_slack_alert(failure_info, urgent=True)
        elif severity == "warning":
            send_slack_alert(failure_info, urgent=False)
        
        send_email_alert(failure_info)
        update_failure_metrics(failure_info)
    
    return failure_callback

def send_pagerduty_alert(failure_info: Dict[str, Any]):
    """Send PagerDuty alert for critical failures"""
    # TODO: Implement PagerDuty integration
    logging.critical(f"PAGERDUTY ALERT: {failure_info}")

def send_slack_alert(failure_info: Dict[str, Any], urgent: bool = False):
    """Send Slack alert"""
    # TODO: Implement Slack integration
    logging.warning(f"SLACK ALERT ({'URGENT' if urgent else 'INFO'}): {failure_info}")

def send_email_alert(failure_info: Dict[str, Any]):
    """Send email alert"""
    # TODO: Implement email alerting
    logging.info(f"EMAIL ALERT: {failure_info}")

def update_failure_metrics(failure_info: Dict[str, Any]):
    """Update failure metrics in monitoring system"""
    # TODO: Implement metrics updates
    pass

# =============================================================================
# SLA MONITORING
# =============================================================================

def sla_miss_callback(dag, task_list, blocking_task_list, slas, blocking_tis):
    """Handle SLA misses with escalation"""
    # TODO: Implement SLA miss handling
    # 1. Log SLA miss details
    # 2. Determine impact and severity
    # 3. Send appropriate alerts
    # 4. Update SLA metrics
    # 5. Trigger auto-scaling if needed
    
    for sla in slas:
        sla_info = {
            "dag_id": sla.dag_id,
            "task_id": sla.task_id,
            "execution_date": str(sla.execution_date),
            "sla_duration": str(sla.sla),
            "actual_duration": "TBD"  # Calculate actual duration
        }
        
        logging.warning(f"SLA MISS: {sla_info}")
        
        # TODO: Implement SLA-specific alerting
        send_sla_alert(sla_info)

def send_sla_alert(sla_info: Dict[str, Any]):
    """Send SLA miss alert"""
    # TODO: Implement SLA alerting
    logging.warning(f"SLA ALERT: {sla_info}")

# =============================================================================
# AUTO-SCALING LOGIC
# =============================================================================

def check_scaling_metrics():
    """Check metrics to determine if scaling is needed"""
    # TODO: Implement scaling decision logic
    # 1. Get current queue length
    # 2. Get current resource utilization
    # 3. Get current worker count
    # 4. Determine if scaling up/down is needed
    # 5. Return scaling decision
    
    metrics = {
        "queue_length": get_queue_length(),
        "cpu_utilization": get_cpu_utilization(),
        "memory_utilization": get_memory_utilization(),
        "active_workers": get_active_worker_count()
    }
    
    scaling_decision = determine_scaling_action(metrics)
    return scaling_decision

def get_queue_length() -> int:
    """Get current task queue length"""
    # TODO: Implement queue length monitoring
    # Query Redis/Celery for current queue size
    return 0  # Placeholder

def get_cpu_utilization() -> float:
    """Get current CPU utilization across workers"""
    # TODO: Implement CPU monitoring
    return 0.0  # Placeholder

def get_memory_utilization() -> float:
    """Get current memory utilization across workers"""
    # TODO: Implement memory monitoring
    return 0.0  # Placeholder

def get_active_worker_count() -> int:
    """Get current number of active workers"""
    # TODO: Implement worker count monitoring
    return 0  # Placeholder

def determine_scaling_action(metrics: Dict[str, Any]) -> str:
    """Determine if scaling action is needed"""
    # TODO: Implement scaling logic
    # Based on SCALING_CONFIG thresholds
    
    queue_length = metrics["queue_length"]
    cpu_util = metrics["cpu_utilization"]
    memory_util = metrics["memory_utilization"]
    active_workers = metrics["active_workers"]
    
    # Scale up conditions
    if (queue_length > SCALING_CONFIG.scale_up_threshold or
        cpu_util > SCALING_CONFIG.cpu_threshold or
        memory_util > SCALING_CONFIG.memory_threshold):
        
        if active_workers < SCALING_CONFIG.max_workers:
            return "scale_up"
    
    # Scale down conditions
    elif (queue_length < SCALING_CONFIG.scale_down_threshold and
          cpu_util < SCALING_CONFIG.cpu_threshold * 0.5 and
          memory_util < SCALING_CONFIG.memory_threshold * 0.5):
        
        if active_workers > SCALING_CONFIG.min_workers:
            return "scale_down"
    
    return "no_action"

def execute_scaling_action(action: str):
    """Execute the scaling action"""
    # TODO: Implement scaling execution
    # 1. For Kubernetes: Update deployment replicas
    # 2. For Docker Compose: Scale service
    # 3. For cloud services: Update auto-scaling groups
    
    if action == "scale_up":
        logging.info("Scaling up workers")
        # Implementation depends on deployment method
    elif action == "scale_down":
        logging.info("Scaling down workers")
        # Implementation depends on deployment method
    else:
        logging.info("No scaling action needed")

# =============================================================================
# PRODUCTION DAG EXAMPLES
# =============================================================================

def create_production_dag_with_monitoring():
    """Create a production DAG with comprehensive monitoring"""
    
    # TODO: Create production-ready DAG configuration
    default_args = {
        'owner': 'scalecorp-platform-team',
        'depends_on_past': False,
        'start_date': datetime(2024, 1, 1),
        'email': MONITORING_CONFIG.alert_email,
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 3,
        'retry_delay': timedelta(minutes=5),
        'retry_exponential_backoff': True,
        'max_retry_delay': timedelta(hours=1),
        'execution_timeout': timedelta(hours=2),
        'on_failure_callback': create_failure_callback("warning"),
        'sla': timedelta(hours=1),
        'pool': 'production_pool',
        'priority_weight': 10,
    }
    
    dag = DAG(
        'scalecorp_production_pipeline',
        default_args=default_args,
        description='Production pipeline with comprehensive monitoring',
        schedule_interval='@hourly',
        catchup=False,
        max_active_runs=3,
        concurrency=16,
        sla_miss_callback=sla_miss_callback,
        tags=['production', 'monitored', 'scalecorp'],
    )
    
    return dag

def create_scaling_monitor_dag():
    """Create DAG to monitor and trigger auto-scaling"""
    
    dag = DAG(
        'scalecorp_autoscaling_monitor',
        default_args={
            'owner': 'scalecorp-platform-team',
            'start_date': datetime(2024, 1, 1),
        },
        description='Monitor system metrics and trigger auto-scaling',
        schedule_interval='*/5 * * * *',  # Every 5 minutes
        catchup=False,
        tags=['monitoring', 'autoscaling', 'system']
    )
    
    with dag:
        
        def monitor_and_scale(**context):
            """Monitor metrics and trigger scaling if needed"""
            # TODO: Implement monitoring and scaling logic
            scaling_decision = check_scaling_metrics()
            
            if scaling_decision != "no_action":
                execute_scaling_action(scaling_decision)
                
                # Log scaling action
                logging.info(f"Auto-scaling action executed: {scaling_decision}")
            
            return scaling_decision
        
        scaling_task = PythonOperator(
            task_id='monitor_and_scale',
            python_callable=monitor_and_scale,
            pool='monitoring_pool'
        )
    
    return dag

# =============================================================================
# EXERCISE INSTRUCTIONS
# =============================================================================

def print_exercise_instructions():
    """Print detailed exercise instructions"""
    
    print("🎯 Airflow at Scale Exercise - ScaleCorp Production Deployment")
    print("=" * 70)
    
    print("\n📋 REQUIREMENTS:")
    print("1. Configure CeleryExecutor for distributed execution")
    print("2. Set up high availability with multi-scheduler deployment")
    print("3. Implement comprehensive monitoring with Prometheus/Grafana")
    print("4. Create auto-scaling logic based on queue length and resources")
    print("5. Implement production-grade error handling and alerting")
    print("6. Optimize performance for 500+ DAGs and 50,000+ daily tasks")
    
    print("\n🏗️ ARCHITECTURE OVERVIEW:")
    print("""
    ScaleCorp Production Airflow Architecture:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Load Balancer                                │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Airflow Web Servers (HA)                       │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │ Web Server  │  │ Web Server  │  │ Web Server  │             │
    │  │     1       │  │     2       │  │     3       │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Airflow Schedulers (HA)                        │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │ Scheduler   │  │ Scheduler   │  │ Scheduler   │             │
    │  │     1       │  │     2       │  │     3       │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Message Broker                              │
    │                  Redis Cluster (HA)                            │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Celery Workers (Auto-Scaling)                  │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
    │  │   Worker    │  │   Worker    │  │   Worker    │  │   ...   │ │
    │  │     1       │  │     2       │  │     N       │  │         │ │
    │  │ [16 procs]  │  │ [16 procs]  │  │ [16 procs]  │  │         │ │
    │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────┘ │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │              PostgreSQL Database (HA)                          │
    │         Primary + Read Replicas + Backup                       │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                 Monitoring Stack                               │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │ Prometheus  │  │   Grafana   │  │ AlertManager│             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n🎯 SUCCESS CRITERIA:")
    print("• CeleryExecutor configured with Redis cluster")
    print("• Multi-scheduler HA setup implemented")
    print("• Comprehensive monitoring with Prometheus/Grafana")
    print("• Auto-scaling logic based on queue and resource metrics")
    print("• Production error handling with severity-based alerting")
    print("• Performance optimizations for large-scale deployment")
    print("• Complete Docker Compose or Kubernetes manifests")
    print("• Monitoring dashboards and alert rules configured")
    
    print("\n🚀 GETTING STARTED:")
    print("1. Complete the configuration classes with production values")
    print("2. Implement the Docker Compose or Kubernetes configuration")
    print("3. Set up Prometheus configuration and alert rules")
    print("4. Implement comprehensive error handling and alerting")
    print("5. Create auto-scaling logic and monitoring")
    print("6. Configure production DAGs with proper monitoring")
    print("7. Test the complete deployment")

if __name__ == "__main__":
    print_exercise_instructions()
    
    print("\n" + "="*70)
    print("🎯 Ready to build enterprise-scale Airflow!")
    print("Complete the TODOs above to create a production-ready deployment.")
    print("="*70)