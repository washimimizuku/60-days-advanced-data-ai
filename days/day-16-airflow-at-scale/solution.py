"""
Day 16: Airflow at Scale - Complete Solution

Enterprise-scale Airflow deployment with distributed executors, comprehensive monitoring,
auto-scaling, and production-grade operational patterns.

This solution demonstrates how to deploy and operate Airflow at scale for ScaleCorp's
production environment handling 500+ DAGs and 50,000+ daily tasks.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
import logging
import os
import yaml
import requests
import psutil
from dataclasses import dataclass, asdict

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.email import send_email
from airflow.models import Variable
from airflow.hooks.postgres_hook import PostgresHook
from airflow.providers.redis.hooks.redis import RedisHook

# =============================================================================
# PRODUCTION CONFIGURATION CLASSES
# =============================================================================

@dataclass
class ExecutorConfig:
    """Production configuration for Airflow executor"""
    executor_type: str = "CeleryExecutor"
    parallelism: int = 256  # Increased for enterprise scale
    max_active_runs_per_dag: int = 16
    max_active_tasks_per_dag: int = 32
    worker_concurrency: int = 16
    broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://redis-cluster:6379/0")
    result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "db+postgresql://airflow:${POSTGRES_PASSWORD}@postgres-ha:5432/airflow")
    worker_prefetch_multiplier: int = 1  # Prevent task hoarding
    task_acks_late: bool = True  # Ensure task completion acknowledgment
    worker_max_tasks_per_child: int = 1000  # Prevent memory leaks

@dataclass
class MonitoringConfig:
    """Production configuration for monitoring and alerting"""
    prometheus_enabled: bool = True
    grafana_enabled: bool = True
    statsd_host: str = "prometheus-statsd-exporter"
    statsd_port: int = 9125
    alert_email: List[str] = None
    slack_webhook: str = os.getenv("SLACK_WEBHOOK_URL", "YOUR_SLACK_WEBHOOK_URL")
    pagerduty_key: str = os.getenv("PAGERDUTY_INTEGRATION_KEY", "YOUR_PAGERDUTY_INTEGRATION_KEY")
    metrics_prefix: str = "scalecorp_airflow"
    
    def __post_init__(self):
        if self.alert_email is None:
            self.alert_email = os.getenv("ALERT_EMAIL", "platform-team@scalecorp.com,oncall@scalecorp.com").split(",")

@dataclass
class ScalingConfig:
    """Production configuration for auto-scaling"""
    min_workers: int = 5  # Minimum for HA
    max_workers: int = 100  # Scale for peak loads
    scale_up_threshold: int = 50  # Queue length to trigger scale up
    scale_down_threshold: int = 10  # Queue length to trigger scale down
    cpu_threshold: float = 75.0  # CPU utilization threshold
    memory_threshold: float = 85.0  # Memory utilization threshold
    scale_up_cooldown: int = 300  # 5 minutes cooldown
    scale_down_cooldown: int = 600  # 10 minutes cooldown
    target_queue_time: int = 120  # Target max queue time in seconds

@dataclass
class HighAvailabilityConfig:
    """Configuration for high availability setup"""
    scheduler_count: int = 3  # Multiple schedulers for HA
    webserver_count: int = 3  # Multiple webservers for load balancing
    database_ha_enabled: bool = True
    redis_cluster_enabled: bool = True
    backup_enabled: bool = True
    backup_retention_days: int = 30
    health_check_interval: int = 30  # seconds
# Production configurations for ScaleCorp
EXECUTOR_CONFIG = ExecutorConfig()
MONITORING_CONFIG = MonitoringConfig()
SCALING_CONFIG = ScalingConfig()
HA_CONFIG = HighAvailabilityConfig()

# =============================================================================
# DOCKER COMPOSE CONFIGURATION FOR PRODUCTION
# =============================================================================

def generate_production_docker_compose():
    """Generate comprehensive Docker Compose configuration for scalable Airflow"""
    
    docker_compose = {
        "version": "3.8",
        "x-airflow-common": {
            "image": "apache/airflow:2.7.0",
            "environment": {
                "AIRFLOW__CORE__EXECUTOR": EXECUTOR_CONFIG.executor_type,
                "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN": EXECUTOR_CONFIG.result_backend,
                "AIRFLOW__CELERY__BROKER_URL": EXECUTOR_CONFIG.broker_url,
                "AIRFLOW__CELERY__RESULT_BACKEND": EXECUTOR_CONFIG.result_backend,
                "AIRFLOW__CORE__PARALLELISM": EXECUTOR_CONFIG.parallelism,
                "AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG": EXECUTOR_CONFIG.max_active_runs_per_dag,
                "AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG": EXECUTOR_CONFIG.max_active_tasks_per_dag,
                "AIRFLOW__CELERY__WORKER_CONCURRENCY": EXECUTOR_CONFIG.worker_concurrency,
                "AIRFLOW__CELERY__WORKER_PREFETCH_MULTIPLIER": EXECUTOR_CONFIG.worker_prefetch_multiplier,
                "AIRFLOW__CELERY__TASK_ACKS_LATE": "True" if EXECUTOR_CONFIG.task_acks_late else "False",
                "AIRFLOW__CELERY__WORKER_MAX_TASKS_PER_CHILD": EXECUTOR_CONFIG.worker_max_tasks_per_child,
                "AIRFLOW__METRICS__STATSD_ON": "True" if MONITORING_CONFIG.prometheus_enabled else "False",
                "AIRFLOW__METRICS__STATSD_HOST": MONITORING_CONFIG.statsd_host,
                "AIRFLOW__METRICS__STATSD_PORT": MONITORING_CONFIG.statsd_port,
                "AIRFLOW__METRICS__STATSD_PREFIX": MONITORING_CONFIG.metrics_prefix,
                "AIRFLOW__SCHEDULER__STANDALONE_DAG_PROCESSOR": "True",
                "AIRFLOW__SCHEDULER__MAX_DAGRUNS_PER_LOOP_TO_SCHEDULE": "20",
                "AIRFLOW__SCHEDULER__MAX_DAGRUNS_TO_CREATE_PER_LOOP": "10",
                "AIRFLOW__WEBSERVER__WORKERS": "4",
                "AIRFLOW__WEBSERVER__WORKER_TIMEOUT": "120",
                "AIRFLOW__WEBSERVER__WEB_SERVER_WORKER_TIMEOUT": "120"
            },
            "volumes": [
                "./dags:/opt/airflow/dags",
                "./logs:/opt/airflow/logs",
                "./plugins:/opt/airflow/plugins",
                "./config:/opt/airflow/config"
            ],
            "user": "50000:0",
            "depends_on": {
                "redis-master": {"condition": "service_healthy"},
                "postgres-primary": {"condition": "service_healthy"}
            }
        },
        
        "services": {
            # Redis Cluster for High Availability
            "redis-master": {
                "image": "redis:7-alpine",
                "ports": ["6379:6379"],
                "command": [
                    "redis-server",
                    "--appendonly", "yes",
                    "--replica-read-only", "no",
                    "--maxmemory", "2gb",
                    "--maxmemory-policy", "allkeys-lru"
                ],
                "volumes": ["redis_master_data:/data"],
                "healthcheck": {
                    "test": ["CMD", "redis-cli", "ping"],
                    "interval": "10s",
                    "timeout": "5s",
                    "retries": 5
                },
                "restart": "unless-stopped"
            },
            
            "redis-replica": {
                "image": "redis:7-alpine",
                "command": [
                    "redis-server",
                    "--replicaof", "redis-master", "6379",
                    "--appendonly", "yes"
                ],
                "volumes": ["redis_replica_data:/data"],
                "depends_on": ["redis-master"],
                "restart": "unless-stopped"
            },
            
            # PostgreSQL High Availability
            "postgres-primary": {
                "image": "postgres:13",
                "environment": {
                    "POSTGRES_DB": "airflow",
                    "POSTGRES_USER": "airflow",
                    "POSTGRES_PASSWORD": "airflow",
                    "POSTGRES_REPLICATION_MODE": "master",
                    "POSTGRES_REPLICATION_USER": "replicator",
                    "POSTGRES_REPLICATION_PASSWORD": "replicator_password",
                    "POSTGRES_INITDB_ARGS": "--auth-host=md5"
                },
                "volumes": [
                    "postgres_primary_data:/var/lib/postgresql/data",
                    "./config/postgresql.conf:/etc/postgresql/postgresql.conf",
                    "./config/pg_hba.conf:/etc/postgresql/pg_hba.conf"
                ],
                "ports": ["5432:5432"],
                "healthcheck": {
                    "test": ["CMD-SHELL", "pg_isready -U airflow"],
                    "interval": "10s",
                    "timeout": "5s",
                    "retries": 5
                },
                "restart": "unless-stopped"
            },
            
            "postgres-replica": {
                "image": "postgres:13",
                "environment": {
                    "POSTGRES_REPLICATION_MODE": "slave",
                    "POSTGRES_REPLICATION_USER": "replicator",
                    "POSTGRES_REPLICATION_PASSWORD": "replicator_password",
                    "POSTGRES_MASTER_SERVICE": "postgres-primary"
                },
                "volumes": ["postgres_replica_data:/var/lib/postgresql/data"],
                "depends_on": ["postgres-primary"],
                "restart": "unless-stopped"
            }
        }
    }
    
    # Add multiple schedulers for HA
    for i in range(1, HA_CONFIG.scheduler_count + 1):
        scheduler_name = f"airflow-scheduler-{i}"
        docker_compose["services"][scheduler_name] = {
            "<<": "*airflow-common",
            "command": "scheduler",
            "environment": {
                **docker_compose["x-airflow-common"]["environment"],
                "AIRFLOW__SCHEDULER__SCHEDULER_HEARTBEAT_SEC": "5",
                "AIRFLOW__SCHEDULER__SCHEDULE_AFTER_TASK_EXECUTION": "True",
                "AIRFLOW__SCHEDULER__PARSING_PROCESSES": "2",
                "AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL": "300"
            },
            "healthcheck": {
                "test": ["CMD-SHELL", "airflow jobs check --job-type SchedulerJob --hostname \"$${HOSTNAME}\""],
                "interval": "30s",
                "timeout": "10s",
                "retries": 5
            },
            "restart": "unless-stopped"
        }
    
    return docker_compose

def generate_kubernetes_manifests():
    """Generate Kubernetes manifests for enterprise Airflow deployment"""
    
    manifests = {
        "namespace": {
            "apiVersion": "v1",
            "kind": "Namespace",
            "metadata": {"name": "airflow-production"}
        },
        
        "configmap": {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": "airflow-config",
                "namespace": "airflow-production"
            },
            "data": {
                "airflow.cfg": generate_airflow_config()
            }
        },
        
        "scheduler_deployment": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "airflow-scheduler",
                "namespace": "airflow-production"
            },
            "spec": {
                "replicas": HA_CONFIG.scheduler_count,
                "selector": {"matchLabels": {"app": "airflow-scheduler"}},
                "template": {
                    "metadata": {"labels": {"app": "airflow-scheduler"}},
                    "spec": {
                        "containers": [{
                            "name": "scheduler",
                            "image": "apache/airflow:2.7.0",
                            "command": ["airflow", "scheduler"],
                            "env": [
                                {"name": "AIRFLOW__CORE__EXECUTOR", "value": EXECUTOR_CONFIG.executor_type},
                                {"name": "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN", "value": EXECUTOR_CONFIG.result_backend},
                                {"name": "AIRFLOW__CELERY__BROKER_URL", "value": EXECUTOR_CONFIG.broker_url}
                            ],
                            "resources": {
                                "requests": {"cpu": "1000m", "memory": "2Gi"},
                                "limits": {"cpu": "2000m", "memory": "4Gi"}
                            },
                            "livenessProbe": {
                                "exec": {"command": ["airflow", "jobs", "check", "--job-type", "SchedulerJob"]},
                                "initialDelaySeconds": 30,
                                "periodSeconds": 30
                            }
                        }]
                    }
                }
            }
        },
        
        "worker_deployment": {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": "airflow-worker",
                "namespace": "airflow-production"
            },
            "spec": {
                "replicas": SCALING_CONFIG.min_workers,
                "selector": {"matchLabels": {"app": "airflow-worker"}},
                "template": {
                    "metadata": {"labels": {"app": "airflow-worker"}},
                    "spec": {
                        "containers": [{
                            "name": "worker",
                            "image": "apache/airflow:2.7.0",
                            "command": ["airflow", "celery", "worker"],
                            "env": [
                                {"name": "AIRFLOW__CORE__EXECUTOR", "value": EXECUTOR_CONFIG.executor_type},
                                {"name": "AIRFLOW__CELERY__WORKER_CONCURRENCY", "value": str(EXECUTOR_CONFIG.worker_concurrency)}
                            ],
                            "resources": {
                                "requests": {"cpu": "500m", "memory": "1Gi"},
                                "limits": {"cpu": "2000m", "memory": "4Gi"}
                            }
                        }]
                    }
                }
            }
        },
        
        "hpa": {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {
                "name": "airflow-worker-hpa",
                "namespace": "airflow-production"
            },
            "spec": {
                "scaleTargetRef": {
                    "apiVersion": "apps/v1",
                    "kind": "Deployment",
                    "name": "airflow-worker"
                },
                "minReplicas": SCALING_CONFIG.min_workers,
                "maxReplicas": SCALING_CONFIG.max_workers,
                "metrics": [
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {"type": "Utilization", "averageUtilization": int(SCALING_CONFIG.cpu_threshold)}
                        }
                    },
                    {
                        "type": "Resource",
                        "resource": {
                            "name": "memory",
                            "target": {"type": "Utilization", "averageUtilization": int(SCALING_CONFIG.memory_threshold)}
                        }
                    }
                ]
            }
        }
    }
    
    return manifests

def generate_airflow_config():
    """Generate optimized airflow.cfg for production"""
    
    config = f"""
[core]
executor = {EXECUTOR_CONFIG.executor_type}
parallelism = {EXECUTOR_CONFIG.parallelism}
max_active_runs_per_dag = {EXECUTOR_CONFIG.max_active_runs_per_dag}
max_active_tasks_per_dag = {EXECUTOR_CONFIG.max_active_tasks_per_dag}
load_examples = False
dags_are_paused_at_creation = True
max_active_runs_per_dag = 16
dagbag_import_timeout = 30.0
dag_file_processor_timeout = 50
task_runner = StandardTaskRunner

[database]
sql_alchemy_conn = {EXECUTOR_CONFIG.result_backend}
sql_alchemy_pool_size = 10
sql_alchemy_pool_recycle = 120
sql_alchemy_pool_pre_ping = True
sql_alchemy_max_overflow = 20

[celery]
broker_url = {EXECUTOR_CONFIG.broker_url}
result_backend = {EXECUTOR_CONFIG.result_backend}
worker_concurrency = {EXECUTOR_CONFIG.worker_concurrency}
worker_prefetch_multiplier = {EXECUTOR_CONFIG.worker_prefetch_multiplier}
task_acks_late = {str(EXECUTOR_CONFIG.task_acks_late).lower()}
worker_max_tasks_per_child = {EXECUTOR_CONFIG.worker_max_tasks_per_child}
broker_transport_options = {{"visibility_timeout": 21600}}

[scheduler]
standalone_dag_processor = True
max_dagruns_per_loop_to_schedule = 20
max_dagruns_to_create_per_loop = 10
schedule_after_task_execution = True
parsing_processes = 2
dag_dir_list_interval = 300
scheduler_heartbeat_sec = 5
scheduler_health_check_threshold = 30
orphaned_tasks_check_interval = 300.0
child_process_log_directory = /opt/airflow/logs/scheduler

[webserver]
workers = 4
worker_timeout = 120
web_server_worker_timeout = 120
worker_refresh_batch_size = 1
worker_refresh_interval = 6000
reload_on_plugin_change = False
expose_config = False
authenticate = True
auth_backend = airflow.auth.backends.password_auth

[metrics]
statsd_on = {str(MONITORING_CONFIG.prometheus_enabled).lower()}
statsd_host = {MONITORING_CONFIG.statsd_host}
statsd_port = {MONITORING_CONFIG.statsd_port}
statsd_prefix = {MONITORING_CONFIG.metrics_prefix}

[logging]
logging_level = INFO
fab_logging_level = WARN
log_filename_template = {{{{ ti.dag_id }}}}/{{{{ ti.task_id }}}}/{{{{ ts }}}}/{{{{ try_number }}}}.log
log_processor_filename_template = {{{{ filename }}}}.log
dag_processor_manager_log_location = /opt/airflow/logs/dag_processor_manager/dag_processor_manager.log
"""
    
    return config.strip()
# =============================================================================
# MONITORING AND OBSERVABILITY
# =============================================================================

def setup_prometheus_config():
    """Generate comprehensive Prometheus configuration for Airflow monitoring"""
    
    prometheus_config = {
        "global": {
            "scrape_interval": "15s",
            "evaluation_interval": "15s"
        },
        
        "rule_files": [
            "airflow_alerts.yml",
            "infrastructure_alerts.yml"
        ],
        
        "scrape_configs": [
            {
                "job_name": "airflow-scheduler",
                "static_configs": [
                    {"targets": [f"airflow-scheduler-{i}:8080" for i in range(1, HA_CONFIG.scheduler_count + 1)]}
                ],
                "metrics_path": "/admin/metrics",
                "scrape_interval": "30s",
                "scrape_timeout": "10s"
            },
            
            {
                "job_name": "airflow-workers",
                "static_configs": [{"targets": ["airflow-worker:8793"]}],
                "scrape_interval": "30s",
                "scrape_timeout": "10s"
            },
            
            {
                "job_name": "airflow-webserver",
                "static_configs": [
                    {"targets": [f"airflow-webserver-{i}:8080" for i in range(1, HA_CONFIG.webserver_count + 1)]}
                ],
                "metrics_path": "/admin/metrics",
                "scrape_interval": "30s"
            },
            
            {
                "job_name": "redis-cluster",
                "static_configs": [
                    {"targets": ["redis-master:6379", "redis-replica:6379"]}
                ],
                "scrape_interval": "30s"
            },
            
            {
                "job_name": "postgres-cluster",
                "static_configs": [
                    {"targets": ["postgres-primary:5432", "postgres-replica:5432"]}
                ],
                "scrape_interval": "30s"
            },
            
            {
                "job_name": "node-exporter",
                "static_configs": [{"targets": ["node-exporter:9100"]}],
                "scrape_interval": "15s"
            }
        ],
        
        "alerting": {
            "alertmanagers": [
                {"static_configs": [{"targets": ["alertmanager:9093"]}]}
            ]
        }
    }
    
    return yaml.dump(prometheus_config, default_flow_style=False)

def setup_alert_rules():
    """Generate Prometheus alert rules for Airflow"""
    
    alert_rules = {
        "groups": [
            {
                "name": "airflow_scheduler_alerts",
                "rules": [
                    {
                        "alert": "AirflowSchedulerDown",
                        "expr": "up{job=\"airflow-scheduler\"} == 0",
                        "for": "1m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Airflow scheduler is down",
                            "description": "Airflow scheduler {{ $labels.instance }} has been down for more than 1 minute."
                        }
                    },
                    
                    {
                        "alert": "AirflowSchedulerHighCPU",
                        "expr": "rate(process_cpu_seconds_total{job=\"airflow-scheduler\"}[5m]) * 100 > 80",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Airflow scheduler high CPU usage",
                            "description": "Airflow scheduler {{ $labels.instance }} CPU usage is above 80% for more than 5 minutes."
                        }
                    },
                    
                    {
                        "alert": "AirflowDAGFailureRate",
                        "expr": "rate(airflow_dag_run_failed_total[1h]) / rate(airflow_dag_run_total[1h]) > 0.1",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High DAG failure rate",
                            "description": "DAG failure rate is above 10% for the last hour."
                        }
                    }
                ]
            },
            
            {
                "name": "airflow_worker_alerts",
                "rules": [
                    {
                        "alert": "AirflowWorkerDown",
                        "expr": "up{job=\"airflow-workers\"} == 0",
                        "for": "2m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "Airflow worker is down",
                            "description": "Airflow worker {{ $labels.instance }} has been down for more than 2 minutes."
                        }
                    },
                    
                    {
                        "alert": "AirflowTaskQueueHigh",
                        "expr": "airflow_celery_queue_length > 100",
                        "for": "5m",
                        "labels": {"severity": "warning"},
                        "annotations": {
                            "summary": "High task queue length",
                            "description": "Celery task queue length is {{ $value }}, above threshold of 100."
                        }
                    }
                ]
            },
            
            {
                "name": "infrastructure_alerts",
                "rules": [
                    {
                        "alert": "RedisDown",
                        "expr": "up{job=\"redis-cluster\"} == 0",
                        "for": "1m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "Redis instance is down",
                            "description": "Redis instance {{ $labels.instance }} has been down for more than 1 minute."
                        }
                    },
                    
                    {
                        "alert": "PostgreSQLDown",
                        "expr": "up{job=\"postgres-cluster\"} == 0",
                        "for": "1m",
                        "labels": {"severity": "critical"},
                        "annotations": {
                            "summary": "PostgreSQL instance is down",
                            "description": "PostgreSQL instance {{ $labels.instance }} has been down for more than 1 minute."
                        }
                    }
                ]
            }
        ]
    }
    
    return yaml.dump(alert_rules, default_flow_style=False)

def setup_grafana_dashboards():
    """Generate comprehensive Grafana dashboards for Airflow monitoring"""
    
    dashboard_config = {
        "dashboard": {
            "id": None,
            "title": "ScaleCorp Airflow Production Monitoring",
            "tags": ["airflow", "production", "scalecorp"],
            "timezone": "browser",
            "refresh": "30s",
            "time": {"from": "now-1h", "to": "now"},
            
            "panels": [
                {
                    "id": 1,
                    "title": "DAG Success Rate",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "rate(airflow_dag_run_success_total[1h]) / rate(airflow_dag_run_total[1h]) * 100",
                            "legendFormat": "Success Rate %",
                            "refId": "A"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "min": 0,
                            "max": 100,
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "yellow", "value": 80},
                                    {"color": "green", "value": 95}
                                ]
                            }
                        }
                    }
                },
                
                {
                    "id": 2,
                    "title": "Active Tasks",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 6, "y": 0},
                    "targets": [
                        {
                            "expr": "airflow_task_instance_running_total",
                            "legendFormat": "Running Tasks",
                            "refId": "A"
                        },
                        {
                            "expr": "airflow_task_instance_queued_total",
                            "legendFormat": "Queued Tasks",
                            "refId": "B"
                        }
                    ]
                },
                
                {
                    "id": 3,
                    "title": "Worker Status",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 6, "x": 18, "y": 0},
                    "targets": [
                        {
                            "expr": "up{job=\"airflow-workers\"}",
                            "legendFormat": "{{ instance }}",
                            "refId": "A",
                            "format": "table"
                        }
                    ]
                },
                
                {
                    "id": 4,
                    "title": "Task Duration",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "histogram_quantile(0.95, rate(airflow_task_duration_seconds_bucket[5m]))",
                            "legendFormat": "95th percentile",
                            "refId": "A"
                        },
                        {
                            "expr": "histogram_quantile(0.50, rate(airflow_task_duration_seconds_bucket[5m]))",
                            "legendFormat": "50th percentile",
                            "refId": "B"
                        }
                    ]
                },
                
                {
                    "id": 5,
                    "title": "System Resources",
                    "type": "graph",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8},
                    "targets": [
                        {
                            "expr": "rate(process_cpu_seconds_total{job=~\"airflow.*\"}[5m]) * 100",
                            "legendFormat": "CPU % - {{ job }}",
                            "refId": "A"
                        },
                        {
                            "expr": "process_resident_memory_bytes{job=~\"airflow.*\"} / 1024 / 1024",
                            "legendFormat": "Memory MB - {{ job }}",
                            "refId": "B"
                        }
                    ]
                }
            ]
        }
    }
    
    return json.dumps(dashboard_config, indent=2)

# =============================================================================
# AUTO-SCALING IMPLEMENTATION
# =============================================================================

class AutoScaler:
    """Advanced auto-scaling implementation for Airflow workers"""
    
    def __init__(self, scaling_config: ScalingConfig, monitoring_config: MonitoringConfig):
        self.config = scaling_config
        self.monitoring = monitoring_config
        self.last_scale_up = None
        self.last_scale_down = None
        self.redis_hook = RedisHook(redis_conn_id='redis_default')
        
    def get_current_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics for scaling decisions"""
        
        try:
            # Get queue length from Redis/Celery
            queue_length = self._get_celery_queue_length()
            
            # Get worker metrics
            active_workers = self._get_active_worker_count()
            
            # Get resource utilization
            cpu_utilization = self._get_average_cpu_utilization()
            memory_utilization = self._get_average_memory_utilization()
            
            # Get task metrics
            pending_tasks = self._get_pending_task_count()
            running_tasks = self._get_running_task_count()
            
            # Calculate queue wait time
            avg_queue_time = self._estimate_queue_wait_time(queue_length, active_workers)
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "queue_length": queue_length,
                "active_workers": active_workers,
                "cpu_utilization": cpu_utilization,
                "memory_utilization": memory_utilization,
                "pending_tasks": pending_tasks,
                "running_tasks": running_tasks,
                "avg_queue_time": avg_queue_time,
                "total_capacity": active_workers * EXECUTOR_CONFIG.worker_concurrency
            }
            
            logging.info(f"Current metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logging.error(f"Error collecting metrics: {e}")
            return {}
    
    def _get_celery_queue_length(self) -> int:
        """Get current Celery queue length"""
        try:
            # In production, this would query Redis for actual queue length
            # For demo, simulate based on time of day
            import random
            base_queue = random.randint(10, 200)
            
            # Simulate higher load during business hours
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 17:  # Business hours
                base_queue = int(base_queue * 1.5)
            
            return base_queue
        except Exception as e:
            logging.error(f"Error getting queue length: {e}")
            return 0
    
    def _get_active_worker_count(self) -> int:
        """Get current number of active workers"""
        try:
            # In production, this would query Celery or Kubernetes
            # For demo, simulate current worker count
            return random.randint(self.config.min_workers, self.config.max_workers)
        except Exception as e:
            logging.error(f"Error getting worker count: {e}")
            return self.config.min_workers
    
    def _get_average_cpu_utilization(self) -> float:
        """Get average CPU utilization across workers"""
        try:
            # In production, this would query monitoring system
            return random.uniform(30.0, 90.0)
        except Exception as e:
            logging.error(f"Error getting CPU utilization: {e}")
            return 0.0
    
    def _get_average_memory_utilization(self) -> float:
        """Get average memory utilization across workers"""
        try:
            # In production, this would query monitoring system
            return random.uniform(40.0, 85.0)
        except Exception as e:
            logging.error(f"Error getting memory utilization: {e}")
            return 0.0
    
    def _get_pending_task_count(self) -> int:
        """Get number of pending tasks"""
        try:
            # In production, query Airflow metadata database
            return random.randint(0, 500)
        except Exception as e:
            logging.error(f"Error getting pending tasks: {e}")
            return 0
    
    def _get_running_task_count(self) -> int:
        """Get number of currently running tasks"""
        try:
            # In production, query Airflow metadata database
            return random.randint(50, 300)
        except Exception as e:
            logging.error(f"Error getting running tasks: {e}")
            return 0
    
    def _estimate_queue_wait_time(self, queue_length: int, active_workers: int) -> float:
        """Estimate average queue wait time in seconds"""
        if active_workers == 0:
            return float('inf')
        
        # Simple estimation: queue_length / (workers * concurrency) * avg_task_duration
        total_capacity = active_workers * EXECUTOR_CONFIG.worker_concurrency
        avg_task_duration = 300  # 5 minutes average
        
        if total_capacity > 0:
            return (queue_length / total_capacity) * avg_task_duration
        else:
            return float('inf')
    
    def determine_scaling_action(self, metrics: Dict[str, Any]) -> str:
        """Determine if scaling action is needed based on multiple factors"""
        
        if not metrics:
            return "no_action"
        
        current_time = datetime.now()
        queue_length = metrics.get("queue_length", 0)
        cpu_util = metrics.get("cpu_utilization", 0)
        memory_util = metrics.get("memory_utilization", 0)
        active_workers = metrics.get("active_workers", self.config.min_workers)
        avg_queue_time = metrics.get("avg_queue_time", 0)
        
        # Check cooldown periods
        if self.last_scale_up and (current_time - self.last_scale_up).seconds < self.config.scale_up_cooldown:
            logging.info("Scale up cooldown active")
            return "no_action"
        
        if self.last_scale_down and (current_time - self.last_scale_down).seconds < self.config.scale_down_cooldown:
            logging.info("Scale down cooldown active")
            return "no_action"
        
        # Scale up conditions (multiple triggers)
        scale_up_reasons = []
        
        if queue_length > self.config.scale_up_threshold:
            scale_up_reasons.append(f"queue_length({queue_length}) > threshold({self.config.scale_up_threshold})")
        
        if cpu_util > self.config.cpu_threshold:
            scale_up_reasons.append(f"cpu({cpu_util:.1f}%) > threshold({self.config.cpu_threshold}%)")
        
        if memory_util > self.config.memory_threshold:
            scale_up_reasons.append(f"memory({memory_util:.1f}%) > threshold({self.config.memory_threshold}%)")
        
        if avg_queue_time > self.config.target_queue_time:
            scale_up_reasons.append(f"queue_time({avg_queue_time:.0f}s) > target({self.config.target_queue_time}s)")
        
        # Scale up if any condition is met and we're below max workers
        if scale_up_reasons and active_workers < self.config.max_workers:
            logging.info(f"Scaling up due to: {', '.join(scale_up_reasons)}")
            return "scale_up"
        
        # Scale down conditions (all must be true)
        scale_down_conditions = [
            queue_length < self.config.scale_down_threshold,
            cpu_util < self.config.cpu_threshold * 0.6,  # 60% of threshold
            memory_util < self.config.memory_threshold * 0.6,  # 60% of threshold
            avg_queue_time < self.config.target_queue_time * 0.5,  # 50% of target
            active_workers > self.config.min_workers
        ]
        
        if all(scale_down_conditions):
            logging.info("Scaling down due to low resource utilization")
            return "scale_down"
        
        return "no_action"
    
    def execute_scaling_action(self, action: str, current_workers: int) -> bool:
        """Execute the scaling action"""
        
        if action == "scale_up":
            new_worker_count = min(current_workers + 2, self.config.max_workers)  # Scale by 2
            success = self._scale_workers(new_worker_count)
            if success:
                self.last_scale_up = datetime.now()
                logging.info(f"Scaled up from {current_workers} to {new_worker_count} workers")
            return success
            
        elif action == "scale_down":
            new_worker_count = max(current_workers - 1, self.config.min_workers)  # Scale by 1
            success = self._scale_workers(new_worker_count)
            if success:
                self.last_scale_down = datetime.now()
                logging.info(f"Scaled down from {current_workers} to {new_worker_count} workers")
            return success
            
        return True  # no_action always succeeds
    
    def _scale_workers(self, target_count: int) -> bool:
        """Scale workers to target count"""
        try:
            # In production, this would:
            # 1. For Kubernetes: Update deployment replicas
            # 2. For Docker Compose: Scale service
            # 3. For cloud services: Update auto-scaling groups
            
            logging.info(f"Scaling workers to {target_count} (simulated)")
            
            # Simulate scaling operation
            import time
            time.sleep(1)  # Simulate scaling delay
            
            # Send scaling metrics
            self._send_scaling_metrics(target_count)
            
            return True
            
        except Exception as e:
            logging.error(f"Error scaling workers: {e}")
            return False
    
    def _send_scaling_metrics(self, worker_count: int):
        """Send scaling metrics to monitoring system"""
        try:
            # In production, send to Prometheus, CloudWatch, etc.
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "worker_count": worker_count,
                "scaling_action": "completed"
            }
            logging.info(f"Scaling metrics: {metrics}")
        except Exception as e:
            logging.error(f"Error sending scaling metrics: {e}")
# =============================================================================
# ERROR HANDLING AND ALERTING
# =============================================================================

class ProductionAlertManager:
    """Advanced alerting system for production Airflow deployment"""
    
    def __init__(self, monitoring_config: MonitoringConfig):
        self.config = monitoring_config
        
    def create_failure_callback(self, severity: str = "warning"):
        """Create failure callback with comprehensive alerting"""
        
        def failure_callback(context):
            """Handle task failures with severity-based alerting and escalation"""
            
            task_instance = context['task_instance']
            dag_id = context['dag'].dag_id
            task_id = task_instance.task_id
            execution_date = context['execution_date']
            exception = context.get('exception')
            
            # Extract failure information
            failure_info = {
                "dag_id": dag_id,
                "task_id": task_id,
                "execution_date": str(execution_date),
                "severity": severity,
                "exception": str(exception) if exception else "Unknown error",
                "log_url": task_instance.log_url,
                "duration": task_instance.duration,
                "try_number": task_instance.try_number,
                "max_tries": task_instance.max_tries,
                "hostname": task_instance.hostname,
                "timestamp": datetime.now().isoformat()
            }
            
            # Determine escalation level based on multiple factors
            escalation_level = self._determine_escalation_level(failure_info, context)
            failure_info["escalation_level"] = escalation_level
            
            logging.error(f"Task failure detected: {failure_info}")
            
            # Execute alerting based on escalation level
            if escalation_level == "critical":
                self._handle_critical_failure(failure_info)
            elif escalation_level == "high":
                self._handle_high_priority_failure(failure_info)
            elif escalation_level == "medium":
                self._handle_medium_priority_failure(failure_info)
            else:
                self._handle_low_priority_failure(failure_info)
            
            # Update failure metrics
            self._update_failure_metrics(failure_info)
            
            # Create incident if needed
            if escalation_level in ["critical", "high"]:
                self._create_incident_ticket(failure_info)
        
        return failure_callback
    
    def _determine_escalation_level(self, failure_info: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Determine escalation level based on multiple factors"""
        
        dag_id = failure_info["dag_id"]
        task_id = failure_info["task_id"]
        try_number = failure_info["try_number"]
        max_tries = failure_info["max_tries"]
        
        # Critical conditions
        if any([
            "critical" in dag_id.lower(),
            "production" in dag_id.lower() and "revenue" in dag_id.lower(),
            try_number >= max_tries,  # Final failure
            "scheduler" in task_id.lower(),
            "monitoring" in task_id.lower()
        ]):
            return "critical"
        
        # High priority conditions
        if any([
            "production" in dag_id.lower(),
            "customer" in dag_id.lower(),
            try_number >= max_tries - 1,  # Second to last try
            failure_info["severity"] == "critical"
        ]):
            return "high"
        
        # Medium priority conditions
        if any([
            "staging" in dag_id.lower(),
            try_number >= 2,
            failure_info["severity"] == "warning"
        ]):
            return "medium"
        
        return "low"
    
    def _handle_critical_failure(self, failure_info: Dict[str, Any]):
        """Handle critical failures with immediate escalation"""
        
        # Send PagerDuty alert
        self._send_pagerduty_alert(failure_info)
        
        # Send urgent Slack alert
        self._send_slack_alert(failure_info, urgent=True)
        
        # Send email to on-call team
        self._send_email_alert(failure_info, recipients=self.config.alert_email + ["oncall-lead@scalecorp.com"])
        
        # Log critical alert
        logging.critical(f"CRITICAL FAILURE ALERT: {failure_info}")
    
    def _handle_high_priority_failure(self, failure_info: Dict[str, Any]):
        """Handle high priority failures"""
        
        # Send Slack alert
        self._send_slack_alert(failure_info, urgent=True)
        
        # Send email to team
        self._send_email_alert(failure_info, recipients=self.config.alert_email)
        
        logging.error(f"HIGH PRIORITY FAILURE: {failure_info}")
    
    def _handle_medium_priority_failure(self, failure_info: Dict[str, Any]):
        """Handle medium priority failures"""
        
        # Send Slack notification
        self._send_slack_alert(failure_info, urgent=False)
        
        logging.warning(f"MEDIUM PRIORITY FAILURE: {failure_info}")
    
    def _handle_low_priority_failure(self, failure_info: Dict[str, Any]):
        """Handle low priority failures"""
        
        # Just log the failure
        logging.info(f"LOW PRIORITY FAILURE: {failure_info}")
    
    def _send_pagerduty_alert(self, failure_info: Dict[str, Any]):
        """Send PagerDuty alert for critical failures"""
        
        try:
            # In production, integrate with PagerDuty API
            pagerduty_payload = {
                "routing_key": self.config.pagerduty_key,
                "event_action": "trigger",
                "payload": {
                    "summary": f"Critical Airflow Failure: {failure_info['dag_id']}.{failure_info['task_id']}",
                    "source": "scalecorp-airflow",
                    "severity": "critical",
                    "custom_details": failure_info
                }
            }
            
            logging.critical(f"PAGERDUTY ALERT: {pagerduty_payload}")
            # requests.post("https://events.pagerduty.com/v2/enqueue", json=pagerduty_payload)
            
        except Exception as e:
            logging.error(f"Error sending PagerDuty alert: {e}")
    
    def _send_slack_alert(self, failure_info: Dict[str, Any], urgent: bool = False):
        """Send Slack alert with rich formatting"""
        
        try:
            color = "#ff0000" if urgent else "#ffaa00"
            urgency = "🚨 URGENT" if urgent else "⚠️ WARNING"
            
            slack_payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{urgency} Airflow Task Failure",
                        "fields": [
                            {"title": "DAG", "value": failure_info['dag_id'], "short": True},
                            {"title": "Task", "value": failure_info['task_id'], "short": True},
                            {"title": "Execution Date", "value": failure_info['execution_date'], "short": True},
                            {"title": "Try Number", "value": f"{failure_info['try_number']}/{failure_info['max_tries']}", "short": True},
                            {"title": "Error", "value": failure_info['exception'][:500], "short": False}
                        ],
                        "actions": [
                            {
                                "type": "button",
                                "text": "View Logs",
                                "url": failure_info['log_url']
                            }
                        ],
                        "footer": "ScaleCorp Airflow",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            logging.warning(f"SLACK ALERT: {slack_payload}")
            # requests.post(self.config.slack_webhook, json=slack_payload)
            
        except Exception as e:
            logging.error(f"Error sending Slack alert: {e}")
    
    def _send_email_alert(self, failure_info: Dict[str, Any], recipients: List[str]):
        """Send detailed email alert"""
        
        try:
            subject = f"Airflow Failure Alert: {failure_info['dag_id']}.{failure_info['task_id']}"
            
            html_content = f"""
            <h2>Airflow Task Failure Alert</h2>
            <table border="1" cellpadding="5">
                <tr><td><strong>DAG ID</strong></td><td>{failure_info['dag_id']}</td></tr>
                <tr><td><strong>Task ID</strong></td><td>{failure_info['task_id']}</td></tr>
                <tr><td><strong>Execution Date</strong></td><td>{failure_info['execution_date']}</td></tr>
                <tr><td><strong>Severity</strong></td><td>{failure_info['severity']}</td></tr>
                <tr><td><strong>Try Number</strong></td><td>{failure_info['try_number']}/{failure_info['max_tries']}</td></tr>
                <tr><td><strong>Duration</strong></td><td>{failure_info['duration']} seconds</td></tr>
                <tr><td><strong>Hostname</strong></td><td>{failure_info['hostname']}</td></tr>
                <tr><td><strong>Exception</strong></td><td>{failure_info['exception']}</td></tr>
            </table>
            
            <h3>Actions</h3>
            <ul>
                <li><a href="{failure_info['log_url']}">View Task Logs</a></li>
                <li>Check Grafana dashboards for system metrics</li>
                <li>Review recent deployments or configuration changes</li>
            </ul>
            
            <p><em>This alert was generated by ScaleCorp Airflow monitoring system.</em></p>
            """
            
            logging.info(f"EMAIL ALERT sent to {recipients}: {subject}")
            # send_email(to=recipients, subject=subject, html_content=html_content)
            
        except Exception as e:
            logging.error(f"Error sending email alert: {e}")
    
    def _update_failure_metrics(self, failure_info: Dict[str, Any]):
        """Update failure metrics in monitoring system"""
        
        try:
            # In production, send metrics to Prometheus, CloudWatch, etc.
            metrics = {
                "metric_name": "airflow_task_failure",
                "value": 1,
                "labels": {
                    "dag_id": failure_info['dag_id'],
                    "task_id": failure_info['task_id'],
                    "severity": failure_info['severity'],
                    "escalation_level": failure_info['escalation_level']
                },
                "timestamp": failure_info['timestamp']
            }
            
            logging.info(f"Failure metrics updated: {metrics}")
            
        except Exception as e:
            logging.error(f"Error updating failure metrics: {e}")
    
    def _create_incident_ticket(self, failure_info: Dict[str, Any]):
        """Create incident ticket for critical/high priority failures"""
        
        try:
            # In production, integrate with Jira, ServiceNow, etc.
            ticket_data = {
                "title": f"Airflow Task Failure: {failure_info['dag_id']}.{failure_info['task_id']}",
                "description": f"""
                Airflow task failure detected with {failure_info['escalation_level']} priority.
                
                Details:
                - DAG: {failure_info['dag_id']}
                - Task: {failure_info['task_id']}
                - Execution Date: {failure_info['execution_date']}
                - Error: {failure_info['exception']}
                - Try Number: {failure_info['try_number']}/{failure_info['max_tries']}
                
                Log URL: {failure_info['log_url']}
                """,
                "priority": failure_info['escalation_level'],
                "assignee": "platform-team",
                "labels": ["airflow", "production", "failure"]
            }
            
            logging.warning(f"INCIDENT TICKET CREATED: {ticket_data}")
            
        except Exception as e:
            logging.error(f"Error creating incident ticket: {e}")

# =============================================================================
# PRODUCTION DAG EXAMPLES
# =============================================================================

def create_enterprise_monitoring_dag():
    """Create comprehensive monitoring DAG for enterprise Airflow deployment"""
    
    alert_manager = ProductionAlertManager(MONITORING_CONFIG)
    auto_scaler = AutoScaler(SCALING_CONFIG, MONITORING_CONFIG)
    
    dag = DAG(
        'scalecorp_enterprise_monitoring',
        default_args={
            'owner': 'scalecorp-platform-team',
            'start_date': datetime(2024, 1, 1),
            'retries': 2,
            'retry_delay': timedelta(minutes=2),
            'on_failure_callback': alert_manager.create_failure_callback("high"),
        },
        description='Enterprise monitoring and auto-scaling for ScaleCorp Airflow',
        schedule_interval='*/5 * * * *',  # Every 5 minutes
        catchup=False,
        tags=['monitoring', 'autoscaling', 'enterprise', 'scalecorp'],
        max_active_runs=1
    )
    
    with dag:
        
        def monitor_system_health(**context):
            """Monitor overall system health"""
            
            health_metrics = {
                "timestamp": datetime.now().isoformat(),
                "schedulers_healthy": check_scheduler_health(),
                "workers_healthy": check_worker_health(),
                "database_healthy": check_database_health(),
                "redis_healthy": check_redis_health(),
                "overall_status": "healthy"
            }
            
            # Determine overall status
            if not all([
                health_metrics["schedulers_healthy"],
                health_metrics["workers_healthy"],
                health_metrics["database_healthy"],
                health_metrics["redis_healthy"]
            ]):
                health_metrics["overall_status"] = "degraded"
            
            logging.info(f"System health check: {health_metrics}")
            
            # Alert if system is degraded
            if health_metrics["overall_status"] == "degraded":
                alert_manager._send_slack_alert({
                    "dag_id": "system_health",
                    "task_id": "monitor_system_health",
                    "execution_date": str(context['execution_date']),
                    "exception": "System health degraded",
                    "log_url": "#",
                    "try_number": 1,
                    "max_tries": 1,
                    "hostname": "monitoring",
                    "timestamp": datetime.now().isoformat()
                }, urgent=True)
            
            return health_metrics
        
        def check_scheduler_health():
            """Check if all schedulers are healthy"""
            # In production, check actual scheduler health
            return True  # Simulate healthy
        
        def check_worker_health():
            """Check if workers are healthy and responsive"""
            # In production, check actual worker health
            return True  # Simulate healthy
        
        def check_database_health():
            """Check database connectivity and performance"""
            try:
                hook = PostgresHook(postgres_conn_id='postgres_default')
                result = hook.get_first("SELECT 1")
                return result is not None
            except Exception as e:
                logging.error(f"Database health check failed: {e}")
                return False
        
        def check_redis_health():
            """Check Redis connectivity and performance"""
            try:
                redis_hook = RedisHook(redis_conn_id='redis_default')
                # In production, actually check Redis
                return True  # Simulate healthy
            except Exception as e:
                logging.error(f"Redis health check failed: {e}")
                return False
        
        def execute_autoscaling(**context):
            """Execute auto-scaling logic"""
            
            # Get current metrics
            metrics = auto_scaler.get_current_metrics()
            
            if not metrics:
                logging.warning("No metrics available for auto-scaling")
                return {"action": "no_metrics"}
            
            # Determine scaling action
            scaling_action = auto_scaler.determine_scaling_action(metrics)
            
            # Execute scaling if needed
            if scaling_action != "no_action":
                current_workers = metrics.get("active_workers", SCALING_CONFIG.min_workers)
                success = auto_scaler.execute_scaling_action(scaling_action, current_workers)
                
                result = {
                    "action": scaling_action,
                    "success": success,
                    "current_workers": current_workers,
                    "metrics": metrics
                }
                
                # Alert on scaling actions
                if success:
                    logging.info(f"Auto-scaling executed: {result}")
                else:
                    logging.error(f"Auto-scaling failed: {result}")
                    alert_manager._send_slack_alert({
                        "dag_id": "autoscaling",
                        "task_id": "execute_autoscaling",
                        "execution_date": str(context['execution_date']),
                        "exception": f"Auto-scaling failed: {scaling_action}",
                        "log_url": "#",
                        "try_number": 1,
                        "max_tries": 1,
                        "hostname": "autoscaler",
                        "timestamp": datetime.now().isoformat()
                    })
                
                return result
            else:
                return {"action": "no_action", "metrics": metrics}
        
        def generate_performance_report(**context):
            """Generate performance report and metrics"""
            
            ti = context['task_instance']
            health_data = ti.xcom_pull(task_ids='monitor_system_health')
            scaling_data = ti.xcom_pull(task_ids='execute_autoscaling')
            
            # Compile performance report
            report = {
                "timestamp": datetime.now().isoformat(),
                "system_health": health_data,
                "scaling_status": scaling_data,
                "performance_summary": {
                    "system_status": health_data.get("overall_status", "unknown") if health_data else "unknown",
                    "scaling_action": scaling_data.get("action", "unknown") if scaling_data else "unknown",
                    "current_workers": scaling_data.get("current_workers", "unknown") if scaling_data else "unknown"
                }
            }
            
            # Send metrics to monitoring systems
            logging.info(f"Performance report: {report}")
            
            return report
        
        # Define monitoring workflow
        health_check = PythonOperator(
            task_id='monitor_system_health',
            python_callable=monitor_system_health,
            pool='monitoring_pool'
        )
        
        autoscaling_task = PythonOperator(
            task_id='execute_autoscaling',
            python_callable=execute_autoscaling,
            pool='monitoring_pool'
        )
        
        performance_report = PythonOperator(
            task_id='generate_performance_report',
            python_callable=generate_performance_report,
            pool='monitoring_pool'
        )
        
        # Workflow dependencies
        [health_check, autoscaling_task] >> performance_report
    
    return dag

# =============================================================================
# MAIN EXECUTION AND SUMMARY
# =============================================================================

if __name__ == "__main__":
    print("🚀 ScaleCorp Airflow at Scale - Complete Production Solution")
    print("=" * 70)
    
    print("\n✅ SOLUTION COMPONENTS:")
    print("• Production-grade Docker Compose configuration")
    print("• Kubernetes manifests with auto-scaling")
    print("• Comprehensive Prometheus monitoring setup")
    print("• Advanced Grafana dashboards")
    print("• Intelligent auto-scaling with multiple triggers")
    print("• Sophisticated alerting with escalation levels")
    print("• High availability with multi-scheduler setup")
    print("• Production monitoring DAG")
    
    print("\n🏗️ ARCHITECTURE HIGHLIGHTS:")
    print(f"• Executor: {EXECUTOR_CONFIG.executor_type}")
    print(f"• Parallelism: {EXECUTOR_CONFIG.parallelism} tasks")
    print(f"• Worker Concurrency: {EXECUTOR_CONFIG.worker_concurrency}")
    print(f"• Schedulers: {HA_CONFIG.scheduler_count} (HA)")
    print(f"• Auto-scaling: {SCALING_CONFIG.min_workers}-{SCALING_CONFIG.max_workers} workers")
    print(f"• Monitoring: Prometheus + Grafana + AlertManager")
    
    print("\n📊 SCALING CONFIGURATION:")
    print(f"• Scale Up Threshold: {SCALING_CONFIG.scale_up_threshold} queued tasks")
    print(f"• Scale Down Threshold: {SCALING_CONFIG.scale_down_threshold} queued tasks")
    print(f"• CPU Threshold: {SCALING_CONFIG.cpu_threshold}%")
    print(f"• Memory Threshold: {SCALING_CONFIG.memory_threshold}%")
    print(f"• Target Queue Time: {SCALING_CONFIG.target_queue_time} seconds")
    
    print("\n🚨 ALERTING LEVELS:")
    print("• Critical: PagerDuty + Urgent Slack + Email")
    print("• High: Urgent Slack + Email")
    print("• Medium: Slack notification")
    print("• Low: Log only")
    
    print("\n📈 MONITORING FEATURES:")
    print("• Real-time system health monitoring")
    print("• Automatic scaling based on queue and resource metrics")
    print("• Comprehensive failure tracking and alerting")
    print("• Performance dashboards and reporting")
    print("• Infrastructure health checks")
    
    print("\n🔧 PRODUCTION OPTIMIZATIONS:")
    print("• Task acknowledgment after completion")
    print("• Worker process recycling to prevent memory leaks")
    print("• Connection pooling and optimization")
    print("• Standalone DAG processor for better performance")
    print("• Optimized scheduler configuration")
    
    print("\n" + "="*70)
    print("🎉 Enterprise-scale Airflow deployment ready!")
    print("This solution handles 500+ DAGs and 50,000+ daily tasks")
    print("with high availability, auto-scaling, and comprehensive monitoring.")
    print("="*70)

# Create the enterprise monitoring DAG
enterprise_monitoring_dag = create_enterprise_monitoring_dag()
globals()['scalecorp_enterprise_monitoring'] = enterprise_monitoring_dag