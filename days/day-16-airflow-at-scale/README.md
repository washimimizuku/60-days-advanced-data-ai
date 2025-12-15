# Day 16: Airflow at Scale - Distributed Executors, Monitoring, Enterprise Deployment

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** distributed Airflow executors for enterprise-scale deployments
- **Implement** comprehensive monitoring and observability with Prometheus and Grafana
- **Configure** auto-scaling strategies and resource optimization
- **Deploy** high-availability Airflow clusters with proper error handling
- **Apply** enterprise operational patterns for production environments

---

## Theory

### Scaling Airflow: From Development to Enterprise

While Days 12 and 15 covered Airflow fundamentals and production patterns, enterprise environments require sophisticated deployment architectures, distributed execution, and comprehensive monitoring to handle thousands of DAGs processing petabytes of data.

**Enterprise Scale Challenges**:
- **Volume**: 1000+ DAGs, 100,000+ daily tasks
- **Concurrency**: Hundreds of simultaneous task executions
- **Reliability**: 99.9% uptime requirements
- **Performance**: Sub-second task scheduling latency
- **Compliance**: Complete audit trails and security controls
- **Cost**: Efficient resource utilization and auto-scaling

### Airflow Executors: Choosing the Right Architecture

Executors determine how and where tasks are executed, making them critical for scalability and performance.

#### 1. Sequential Executor (Development Only)

```python
# airflow.cfg
[core]
executor = SequentialExecutor

# Characteristics:
# - Single-threaded execution
# - SQLite database support
# - Development and testing only
# - No parallelism
```

**Use Cases**: Local development, testing, learning
**Limitations**: No parallelism, not suitable for production

#### 2. Local Executor (Small Production)

```python
# airflow.cfg
[core]
executor = LocalExecutor
parallelism = 32

[celery]
worker_concurrency = 16

# Characteristics:
# - Multi-process execution on single machine
# - PostgreSQL/MySQL required
# - Good for small to medium workloads
# - Limited by single machine resources
```

**Use Cases**: Small teams, moderate workloads (< 100 DAGs)
**Limitations**: Single point of failure, limited scalability
#### 3. Celery Executor (Large Production)

```python
# airflow.cfg
[core]
executor = CeleryExecutor
parallelism = 128

[celery]
broker_url = redis://redis-cluster:6379/0
result_backend = db+postgresql://airflow:password@postgres:5432/airflow
worker_concurrency = 16
```

**Advantages**:
- Horizontal scaling across multiple machines
- High availability with multiple workers
- Mature ecosystem with monitoring tools
- Fine-grained resource control

#### 4. Kubernetes Executor (Cloud-Native)

```python
# airflow.cfg
[core]
executor = KubernetesExecutor

[kubernetes]
namespace = airflow
worker_container_repository = apache/airflow
worker_container_tag = 2.7.0
delete_worker_pods = True
delete_worker_pods_on_failure = False
```

**Advantages**:
- Dynamic pod creation per task
- Automatic resource scaling
- Isolation between tasks
- Cloud-native deployment
- Cost-effective (pay per task)
### Enterprise Monitoring and Observability

Production Airflow requires comprehensive monitoring across multiple dimensions.

#### 1. Metrics Collection with Prometheus

```python
# airflow.cfg
[metrics]
statsd_on = True
statsd_host = prometheus-statsd-exporter
statsd_port = 9125
statsd_prefix = airflow

# Custom metrics in DAGs
from airflow.providers.prometheus.hooks.prometheus import PrometheusHook

def custom_metrics_task(**context):
    """Task that publishes custom metrics"""
    prometheus = PrometheusHook()
    
    # Counter metric
    prometheus.send_metric(
        'airflow_custom_task_counter',
        1,
        labels={'dag_id': context['dag'].dag_id, 'task_id': context['task'].task_id}
    )
```

#### 2. Auto-Scaling Strategies

```yaml
# Horizontal Pod Autoscaler for Celery workers
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: airflow-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: airflow-worker
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```
### High Availability and Performance Optimization

#### 1. Multi-Scheduler Setup

```python
# airflow.cfg for HA schedulers
[scheduler]
standalone_dag_processor = True
max_dagruns_per_loop_to_schedule = 20
max_dagruns_to_create_per_loop = 10

# Multiple scheduler instances
# scheduler-1: Primary scheduler
# scheduler-2: Secondary scheduler (standby)
# scheduler-3: Tertiary scheduler (standby)
```

#### 2. Database Optimization

```sql
-- PostgreSQL optimization for Airflow
-- Indexes for common queries
CREATE INDEX CONCURRENTLY idx_dag_run_dag_id_execution_date 
ON dag_run (dag_id, execution_date);

CREATE INDEX CONCURRENTLY idx_task_instance_dag_task_execution_date 
ON task_instance (dag_id, task_id, execution_date);

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
```

---

## 💻 Hands-On Exercise (40 minutes)

Deploy a production-ready, scalable Airflow cluster with comprehensive monitoring.

**Scenario**: You're the Platform Engineer at "ScaleCorp", a rapidly growing data company. The current Airflow deployment can't handle the increasing workload of 500+ DAGs and 50,000+ daily tasks. You need to design and implement a scalable, highly available Airflow architecture.

**Requirements**:
1. **Distributed Execution**: Configure CeleryExecutor with auto-scaling workers
2. **High Availability**: Multi-scheduler setup with database HA
3. **Comprehensive Monitoring**: Prometheus metrics and Grafana dashboards
4. **Auto-Scaling**: Dynamic worker scaling based on queue length
5. **Error Handling**: Production-grade error handling and alerting
6. **Performance Optimization**: Database and scheduler tuning

See `exercise.py` for starter code and detailed requirements.
---

## 📚 Resources

- **Airflow Scaling Guide**: [airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/scaling-airflow.html](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/scaling-airflow.html)
- **Kubernetes Executor**: [airflow.apache.org/docs/apache-airflow/stable/core-concepts/executor/kubernetes.html](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/executor/kubernetes.html)
- **Celery Executor**: [airflow.apache.org/docs/apache-airflow/stable/core-concepts/executor/celery.html](https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/executor/celery.html)
- **Monitoring**: [airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/monitoring.html](https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/monitoring.html)
- **KEDA Scaling**: [keda.sh/docs/2.12/scalers/redis/](https://keda.sh/docs/2.12/scalers/redis/)

---

## 🎯 Key Takeaways

- **Executor choice determines scalability** - CeleryExecutor for distributed, KubernetesExecutor for cloud-native
- **Monitoring is critical** - Comprehensive metrics, logging, and alerting prevent outages
- **Auto-scaling saves costs** - Dynamic resource allocation based on workload
- **High availability requires redundancy** - Multiple schedulers, database HA, backup strategies
- **Performance optimization is ongoing** - Database tuning, scheduler configuration, worker optimization
- **Security and compliance** - Proper authentication, authorization, and audit trails
- **Operational excellence** - Automated deployment, monitoring, and incident response

---

## 🚀 What's Next?

Tomorrow (Day 17), you'll learn **dbt Deep Dive** - advanced dbt patterns including incremental models, snapshots, macros, and analytics engineering best practices.

**Preview**: You'll explore advanced dbt features like incremental strategies, slowly changing dimensions, custom materializations, and complex macro development. This builds on the dbt basics from Day 13 to create enterprise-grade analytics engineering workflows!

---

## ✅ Before Moving On

- [ ] Understand different Airflow executors and their use cases
- [ ] Can configure distributed Airflow with CeleryExecutor
- [ ] Know how to implement comprehensive monitoring and alerting
- [ ] Understand auto-scaling strategies for dynamic workloads
- [ ] Can design high-availability Airflow architectures
- [ ] Complete the hands-on exercise
- [ ] Take the quiz

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐⭐ (Expert-Level Enterprise Deployment)

Ready to deploy enterprise-scale Airflow! 🚀