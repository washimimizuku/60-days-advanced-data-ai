# Day 15: Airflow Production Patterns - Setup Guide

## 🚀 Quick Start (15 minutes)

### Prerequisites
- Docker and Docker Compose installed
- 4GB+ RAM available
- 10GB+ disk space

### 1. Environment Setup
```bash
# Clone and navigate to directory
cd days/day-15-airflow-production-patterns

# Copy environment template
cp .env.example .env

# Generate Airflow Fernet key
echo "AIRFLOW__CORE__FERNET_KEY=$(python3 -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())')" >> .env

# Set Airflow UID (Linux/macOS)
echo "AIRFLOW_UID=$(id -u)" >> .env
```

### 2. Initialize Airflow
```bash
# Create required directories
mkdir -p ./dags ./logs ./plugins ./config ./data/{input,output,archive,products}

# Initialize Airflow database and create admin user
docker-compose up airflow-init

# Start Airflow services
docker-compose up -d
```

### 3. Setup Pools and Connections
```bash
# Wait for services to be ready
sleep 30

# Create Airflow pools for resource management
docker-compose exec airflow-webserver airflow pools set fast_processing_pool 3 "High-priority, small data processing"
docker-compose exec airflow-webserver airflow pools set standard_pool 10 "Standard data processing"
docker-compose exec airflow-webserver airflow pools set batch_processing_pool 2 "Large data batch processing"
docker-compose exec airflow-webserver airflow pools set monitoring_pool 5 "Monitoring and alerting tasks"
docker-compose exec airflow-webserver airflow pools set external_api_pool 5 "External API calls"

# Create database connections
docker-compose exec airflow-webserver airflow connections add 'postgres_transactions' \
    --conn-type 'postgres' \
    --conn-host 'postgres' \
    --conn-login 'techcorp_user' \
    --conn-password 'techcorp_password' \
    --conn-schema 'techcorp_data' \
    --conn-port 5432
```

### 4. Load Sample Data
```bash
# Create sample configuration files
python3 scripts/generate_sample_configs.py

# Load test data
docker-compose exec postgres psql -U airflow -d airflow -f /docker-entrypoint-initdb.d/init.sql
```

## 🌐 Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow Webserver** | http://localhost:8080 | airflow/airflow |
| **PostgreSQL** | localhost:5432 | airflow/airflow |

## 📊 Testing the Setup

### Verify DAG Generation
```bash
# Check that dynamic DAGs are loaded
docker-compose exec airflow-webserver airflow dags list | grep techcorp_pipeline

# Expected output:
# techcorp_pipeline_customer_data
# techcorp_pipeline_transaction_data  
# techcorp_pipeline_product_catalog
# techcorp_pipeline_user_behavior_events
# techcorp_pipeline_partner_data
# techcorp_pipeline_monitoring
```

### Test Individual Components
```bash
# Test custom sensors
docker-compose exec airflow-webserver python -c "
from dags.solution import DataQualitySensor
sensor = DataQualitySensor(task_id='test', data_source='test', quality_checks=[])
print('DataQualitySensor imported successfully')
"

# Test task groups
docker-compose exec airflow-webserver python -c "
from dags.solution import create_data_ingestion_task_group, DATA_SOURCE_CONFIGS
group = create_data_ingestion_task_group('test_group', DATA_SOURCE_CONFIGS[0])
print('Task group created successfully')
"
```

### Run Sample Pipeline
```bash
# Trigger a test pipeline
docker-compose exec airflow-webserver airflow dags trigger techcorp_pipeline_customer_data

# Check pipeline status
docker-compose exec airflow-webserver airflow dags state techcorp_pipeline_customer_data $(date +%Y-%m-%d)

# View task logs
docker-compose exec airflow-webserver airflow tasks logs techcorp_pipeline_customer_data start $(date +%Y-%m-%d)
```

## 🔧 Configuration Files

### Sample Data Source Configuration
Create `config/data_sources.yml`:
```yaml
data_sources:
  - name: "customer_data"
    source_type: "api"
    source_config:
      url: "https://api.techcorp.com/customers"
      auth_type: "bearer"
    target_table: "customers"
    schedule_interval: "@hourly"
    priority: 9
    expected_volume_mb: 75
    sla_hours: 2
    quality_rules:
      - type: "not_null"
        columns: ["customer_id", "email"]
      - type: "unique"
        columns: ["customer_id"]
    transformations:
      - type: "email_normalize"
      - type: "pii_hash"
```

### Airflow Configuration
Create `config/airflow.cfg` (optional customizations):
```ini
[core]
dags_folder = /opt/airflow/dags
base_log_folder = /opt/airflow/logs
remote_logging = False
executor = LocalExecutor
parallelism = 32
dag_concurrency = 16
max_active_runs_per_dag = 16

[webserver]
expose_config = True
rbac = True

[scheduler]
catchup_by_default = False
dag_dir_list_interval = 300
```

## 🧪 Running Tests

### Unit Tests
```bash
# Install test dependencies
pip install pytest pytest-airflow

# Run unit tests
python -m pytest tests/ -v

# Test specific components
python -m pytest tests/test_sensors.py -v
python -m pytest tests/test_task_groups.py -v
```

### Integration Tests
```bash
# Test DAG generation
python tests/test_dag_generation.py

# Test branching logic
python tests/test_branching.py

# Test monitoring DAG
python tests/test_monitoring.py
```

## 📈 Monitoring and Observability

### Check Pool Utilization
```bash
# View pool status
docker-compose exec airflow-webserver airflow pools list

# Monitor pool usage
docker-compose exec airflow-webserver airflow pools get standard_pool
```

### View DAG Performance
```bash
# Check DAG run statistics
docker-compose exec airflow-webserver airflow dags report

# View task instance statistics
docker-compose exec airflow-webserver airflow tasks states-for-dag-run techcorp_pipeline_customer_data $(date +%Y-%m-%d)
```

### Monitor System Health
```bash
# Trigger monitoring DAG
docker-compose exec airflow-webserver airflow dags trigger techcorp_pipeline_monitoring

# Check monitoring results
docker-compose exec airflow-webserver airflow tasks logs techcorp_pipeline_monitoring check_pipeline_health $(date +%Y-%m-%d)
```

## 🔍 Troubleshooting

### Common Issues

**DAGs not appearing:**
```bash
# Check DAG folder permissions
docker-compose exec airflow-webserver ls -la /opt/airflow/dags/

# Refresh DAGs
docker-compose exec airflow-webserver airflow dags reserialize
```

**Pool creation fails:**
```bash
# Check if pools exist
docker-compose exec airflow-webserver airflow pools list

# Recreate pools
./scripts/setup_pools.sh
```

**Database connection issues:**
```bash
# Test database connectivity
docker-compose exec postgres pg_isready -U airflow

# Check connection configuration
docker-compose exec airflow-webserver airflow connections list
```

**Memory issues:**
```bash
# Check resource usage
docker stats

# Increase Docker memory limit to 4GB+
# Restart Docker Desktop
```

### Debug Mode
```bash
# Enable debug logging
export AIRFLOW__LOGGING__LOGGING_LEVEL=DEBUG

# Run specific task in debug mode
docker-compose exec airflow-webserver airflow tasks test techcorp_pipeline_customer_data start $(date +%Y-%m-%d)
```

## 📚 Next Steps

1. **Explore Dynamic DAGs**: Modify `config/data_sources.yml` to add new pipelines
2. **Test Branching Logic**: Trigger pipelines with different data characteristics
3. **Monitor Performance**: Use the monitoring DAG to track system health
4. **Customize Alerts**: Configure Slack/email notifications in `.env`
5. **Scale Resources**: Adjust pool sizes based on workload requirements

## 🆘 Getting Help

- **Logs**: Check `docker-compose logs airflow-webserver`
- **Documentation**: Review README.md and solution.py
- **Health Check**: Use `docker-compose ps` to check service status
- **Database Access**: `docker-compose exec postgres psql -U airflow -d airflow`

## 🔒 Production Considerations

- Change default passwords in production
- Use proper secrets management (AWS Secrets Manager, HashiCorp Vault)
- Enable SSL/TLS for external access
- Configure proper backup strategies
- Set up monitoring and alerting
- Implement proper logging and audit trails

---

**Setup Time**: ~15 minutes  
**Resource Requirements**: 4GB RAM, 10GB disk  
**Services**: Airflow webserver, scheduler, PostgreSQL, Redis  
**Ready for**: Advanced Airflow production patterns