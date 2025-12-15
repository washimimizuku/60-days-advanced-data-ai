# Airflow at Scale - Infrastructure Guide

This guide provides comprehensive instructions for deploying and operating the enterprise-scale Airflow infrastructure for ScaleCorp.

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Load Balancer (Nginx)                       │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                 Airflow Web Servers (HA)                       │
│  ┌─────────────┐  ┌─────────────┐                             │
│  │ Web Server  │  │ Web Server  │                             │
│  │     1       │  │     2       │                             │
│  └─────────────┘  └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                 Airflow Schedulers (HA)                        │
│  ┌─────────────┐  ┌─────────────┐                             │
│  │ Scheduler   │  │ Scheduler   │                             │
│  │     1       │  │     2       │                             │
│  └─────────────┘  └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                    Message Broker                              │
│                  Redis Cluster (HA)                            │
│  ┌─────────────┐  ┌─────────────┐                             │
│  │   Master    │  │   Replica   │                             │
│  └─────────────┘  └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                 Celery Workers (Auto-Scaling)                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │   Worker    │  │   Worker    │  │   Worker    │             │
│  │     1       │  │     2       │  │     N       │             │
│  │ [16 procs]  │  │ [16 procs]  │  │ [16 procs]  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│              PostgreSQL Database (HA)                          │
│         Primary + Read Replica + Backup                        │
│  ┌─────────────┐  ┌─────────────┐                             │
│  │   Primary   │  │   Replica   │                             │
│  └─────────────┘  └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
                                │
┌─────────────────────────────────────────────────────────────────┐
│                 Monitoring Stack                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Prometheus  │  │   Grafana   │  │ AlertManager│             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+
- Docker Compose 2.0+
- 16GB+ RAM (32GB recommended)
- 4+ CPU cores (8+ recommended)
- 50GB+ disk space

### 1. Initial Setup

```bash
# Clone and navigate to directory
cd day-16-airflow-at-scale

# Run setup script
./scripts/setup.sh
```

### 2. Manual Setup (Alternative)

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env

# Initialize Airflow
docker-compose up airflow-init

# Start all services
docker-compose up -d
```

## 📊 Service Access

| Service | URL | Default Credentials |
|---------|-----|-------------------|
| Airflow UI | http://localhost:80 | admin / (see .env) |
| Prometheus | http://localhost:9090 | - |
| Grafana | http://localhost:3000 | admin / (see .env) |
| Flower | http://localhost:5555 | - |
| AlertManager | http://localhost:9093 | - |

## 🔧 Configuration

### Environment Variables

Key environment variables in `.env`:

```bash
# Core Configuration
AIRFLOW_UID=50000
POSTGRES_PASSWORD=your_secure_password
_AIRFLOW_WWW_USER_PASSWORD=your_admin_password

# Scaling Configuration
MIN_WORKERS=3
MAX_WORKERS=50
SCALE_UP_THRESHOLD=50
SCALE_DOWN_THRESHOLD=10

# Monitoring
GRAFANA_PASSWORD=your_grafana_password
SLACK_WEBHOOK_URL=your_slack_webhook
```

### Airflow Configuration

Key settings in `docker-compose.yml`:

```yaml
AIRFLOW__CORE__PARALLELISM: 256
AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG: 16
AIRFLOW__CELERY__WORKER_CONCURRENCY: 16
AIRFLOW__METRICS__STATSD_ON: 'True'
```

## 📈 Scaling Operations

### Manual Scaling

```bash
# Scale workers to 10
./scripts/scale-workers.sh 10

# Check current status
./scripts/scale-workers.sh --status

# Enable auto-scaling
./scripts/scale-workers.sh --auto
```

### Docker Compose Scaling

```bash
# Scale workers
docker-compose up -d --scale airflow-worker=10

# Scale schedulers (if needed)
docker-compose up -d --scale airflow-scheduler-1=2
```

### Auto-Scaling Configuration

Auto-scaling is controlled by these parameters:

- **Scale Up Triggers**:
  - Queue length > 50 tasks
  - CPU utilization > 75%
  - Memory utilization > 85%
  - Queue wait time > 120 seconds

- **Scale Down Triggers**:
  - Queue length < 10 tasks
  - CPU utilization < 45%
  - Memory utilization < 50%
  - Queue wait time < 60 seconds

## 📊 Monitoring and Alerting

### Prometheus Metrics

Key metrics collected:

- `airflow_dag_run_total` - DAG run counts
- `airflow_task_duration_seconds` - Task execution times
- `airflow_scheduler_heartbeat` - Scheduler health
- `airflow_celery_queue_length` - Task queue length
- `airflow_pool_running_slots` - Pool utilization

### Grafana Dashboards

Import these dashboards:

1. **Airflow Overview** - System health and performance
2. **DAG Performance** - DAG-specific metrics
3. **Worker Monitoring** - Celery worker status
4. **Infrastructure** - Database and Redis metrics

### Alert Rules

Critical alerts configured:

- Scheduler down (1 minute)
- High DAG failure rate (>10%)
- Database connectivity issues
- Redis cluster problems
- High task queue length (>100)

## 🔒 Security Configuration

### Database Security

```sql
-- Create monitoring user with limited permissions
CREATE USER airflow_monitor WITH ENCRYPTED PASSWORD 'monitor_password';
GRANT CONNECT ON DATABASE airflow TO airflow_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO airflow_monitor;
```

### Network Security

- All services run in isolated Docker network
- PostgreSQL configured with proper authentication
- Redis protected with authentication (optional)
- Nginx load balancer with security headers

### Secrets Management

Store sensitive data in `.env` file:

```bash
# Generate secure passwords
openssl rand -base64 32  # For database passwords
openssl rand -base64 16  # For web passwords
```

## 🔄 Backup and Recovery

### Database Backup

```bash
# Create backup
docker-compose exec postgres-primary pg_dump -U airflow airflow > backup_$(date +%Y%m%d).sql

# Restore backup
docker-compose exec -T postgres-primary psql -U airflow airflow < backup_20240101.sql
```

### Configuration Backup

```bash
# Backup configuration
tar -czf airflow_config_$(date +%Y%m%d).tar.gz config/ monitoring/ .env docker-compose.yml
```

## 🚨 Troubleshooting

### Common Issues

#### Services Won't Start

```bash
# Check logs
docker-compose logs [service-name]

# Check resource usage
docker stats

# Restart specific service
docker-compose restart [service-name]
```

#### Database Connection Issues

```bash
# Check PostgreSQL logs
docker-compose logs postgres-primary

# Test connection
docker-compose exec postgres-primary psql -U airflow -d airflow -c "SELECT 1;"
```

#### Worker Scaling Issues

```bash
# Check Celery workers
docker-compose exec airflow-flower celery -A airflow.executors.celery_executor.app inspect active

# Check Redis connection
docker-compose exec redis-master redis-cli ping
```

#### High Memory Usage

```bash
# Check memory usage by service
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Restart workers to clear memory
docker-compose restart airflow-worker
```

### Performance Tuning

#### Database Optimization

```sql
-- Analyze query performance
SELECT query, mean_time, calls FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;

-- Update statistics
ANALYZE;

-- Reindex if needed
REINDEX DATABASE airflow;
```

#### Worker Optimization

```yaml
# Adjust worker concurrency
AIRFLOW__CELERY__WORKER_CONCURRENCY: 8  # Reduce for memory-constrained environments

# Enable worker recycling
AIRFLOW__CELERY__WORKER_MAX_TASKS_PER_CHILD: 1000
```

## 📋 Maintenance Tasks

### Daily Tasks

- Monitor system health in Grafana
- Check alert notifications
- Review failed DAG runs
- Monitor resource utilization

### Weekly Tasks

- Review and rotate logs
- Check database performance
- Update security patches
- Review scaling patterns

### Monthly Tasks

- Database maintenance (VACUUM, ANALYZE)
- Backup verification
- Capacity planning review
- Security audit

## 🔄 Upgrade Procedures

### Airflow Upgrade

```bash
# 1. Backup current state
./scripts/backup.sh

# 2. Update image version in docker-compose.yml
# 3. Stop services
docker-compose down

# 4. Upgrade database
docker-compose run --rm airflow-init

# 5. Start services
docker-compose up -d
```

### Infrastructure Updates

```bash
# Update monitoring stack
docker-compose pull prometheus grafana alertmanager
docker-compose up -d prometheus grafana alertmanager

# Update database
docker-compose pull postgres:13
docker-compose up -d postgres-primary postgres-replica
```

## 📞 Support and Contacts

- **Platform Team**: platform-team@scalecorp.com
- **On-Call**: oncall@scalecorp.com
- **Documentation**: Internal wiki
- **Monitoring**: Grafana dashboards
- **Alerts**: Slack #airflow-alerts

## 📚 Additional Resources

- [Airflow Production Deployment Guide](https://airflow.apache.org/docs/apache-airflow/stable/production-deployment.html)
- [Celery Monitoring](https://docs.celeryproject.org/en/stable/userguide/monitoring.html)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboard Design](https://grafana.com/docs/grafana/latest/best-practices/)

---

**Last Updated**: January 2024  
**Version**: 1.0  
**Maintained by**: ScaleCorp Platform Team