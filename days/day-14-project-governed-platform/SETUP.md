# DataCorp Governed Platform - Setup Guide

## 🚀 Quick Start (10 minutes)

### Prerequisites
- Docker and Docker Compose installed
- 8GB+ RAM available
- 20GB+ disk space

### 1. Clone and Setup
```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd days/day-14-project-governed-platform

# Copy environment template
cp .env.example .env

# Generate Airflow Fernet key
python3 -c "from cryptography.fernet import Fernet; print('AIRFLOW_FERNET_KEY=' + Fernet.generate_key().decode())" >> .env
```

### 2. Initialize Infrastructure
```bash
# Create required directories
mkdir -p {airflow/{dags,logs,plugins},dbt/{models,tests,macros},governance/{policies,scripts,logs},monitoring/{prometheus,grafana,alerts},sql,sample_data}

# Start core services
docker-compose up -d postgres airflow-db redis

# Wait for databases to be ready
sleep 30

# Initialize Airflow database
docker-compose run --rm airflow-webserver airflow db init

# Create Airflow admin user
docker-compose run --rm airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@datacorp.com \
    --password admin
```

### 3. Start All Services
```bash
# Start complete platform
docker-compose up -d

# Check service health
docker-compose ps
```

### 4. Verify Installation
```bash
# Test Airflow
curl -f http://localhost:8080/health

# Test Grafana
curl -f http://localhost:3000/api/health

# Test DataHub
curl -f http://localhost:8090/health

# Test Prometheus
curl -f http://localhost:9090/-/healthy
```

## 🌐 Access URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| **Airflow** | http://localhost:8080 | admin/admin |
| **Grafana** | http://localhost:3000 | admin/admin |
| **DataHub** | http://localhost:8090 | - |
| **Prometheus** | http://localhost:9090 | - |

## 📊 Initial Data Setup

### Load Sample Data
```bash
# Generate sample customer data
docker-compose exec postgres psql -U platform_user -d datacorp_platform -c "
CREATE TABLE IF NOT EXISTS raw_customers (
    customer_id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50),
    consent_status VARCHAR(20) DEFAULT 'granted',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT INTO raw_customers (email, first_name, last_name, phone, city, state, country)
SELECT 
    'user' || generate_series || '@datacorp.com',
    'First' || generate_series,
    'Last' || generate_series,
    '555-' || LPAD(generate_series::text, 3, '0') || '-' || LPAD((generate_series * 7)::text, 4, '0'),
    'City' || (generate_series % 10),
    'State' || (generate_series % 5),
    'USA'
FROM generate_series(1, 1000);
"

# Create schemas
docker-compose exec postgres psql -U platform_user -d datacorp_platform -c "
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS marts;
CREATE SCHEMA IF NOT EXISTS compliance;
"
```

### Setup dbt Project
```bash
# Initialize dbt project
docker-compose exec airflow-webserver dbt init datacorp_governance --project-dir /opt/dbt

# Install dbt dependencies
docker-compose exec airflow-webserver dbt deps --project-dir /opt/dbt

# Run initial dbt models
docker-compose exec airflow-webserver dbt run --project-dir /opt/dbt
```

## 🔧 Configuration

### Airflow Configuration
```bash
# Enable DAGs
docker-compose exec airflow-webserver airflow dags unpause governance_daily_pipeline
docker-compose exec airflow-webserver airflow dags unpause compliance_monitoring

# Trigger initial run
docker-compose exec airflow-webserver airflow dags trigger governance_daily_pipeline
```

### Monitoring Setup
```bash
# Import Grafana dashboards
docker-compose exec grafana grafana-cli admin reset-admin-password admin

# The dashboards will be automatically provisioned from ./monitoring/grafana/dashboards/
```

## 🧪 Testing the Platform

### Run Governance Pipeline
```bash
# Trigger the main governance pipeline
docker-compose exec airflow-webserver airflow dags trigger governance_daily_pipeline

# Check pipeline status
docker-compose exec airflow-webserver airflow dags state governance_daily_pipeline $(date +%Y-%m-%d)
```

### Verify Data Quality
```bash
# Run dbt tests
docker-compose exec airflow-webserver dbt test --project-dir /opt/dbt

# Check compliance reports
docker-compose exec postgres psql -U platform_user -d datacorp_platform -c "
SELECT * FROM compliance.gdpr_compliance_report ORDER BY report_date DESC LIMIT 1;
"
```

### Test PII Detection
```bash
# Run PII detection script
docker-compose exec airflow-webserver python /opt/governance/scripts/test_pii_detection.py
```

## 🔍 Troubleshooting

### Common Issues

**Services not starting:**
```bash
# Check logs
docker-compose logs airflow-webserver
docker-compose logs postgres

# Restart services
docker-compose restart
```

**Database connection issues:**
```bash
# Test database connectivity
docker-compose exec postgres pg_isready -U platform_user

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

**Airflow DAGs not appearing:**
```bash
# Check DAG folder permissions
docker-compose exec airflow-webserver ls -la /opt/airflow/dags/

# Refresh DAGs
docker-compose exec airflow-webserver airflow dags reserialize
```

**Memory issues:**
```bash
# Check resource usage
docker stats

# Increase Docker memory limit to 8GB+
# Restart Docker Desktop
```

### Health Checks
```bash
# Run comprehensive health check
./scripts/health_check.sh

# Check all service endpoints
curl -f http://localhost:8080/health  # Airflow
curl -f http://localhost:3000/api/health  # Grafana
curl -f http://localhost:8090/health  # DataHub
curl -f http://localhost:9090/-/healthy  # Prometheus
```

## 📚 Next Steps

1. **Explore Dashboards**: Visit Grafana to see governance metrics
2. **Review Data Catalog**: Check DataHub for dataset documentation
3. **Run Compliance Reports**: Generate GDPR compliance reports
4. **Test Data Quality**: Run dbt tests and quality checks
5. **Monitor Pipelines**: Watch Airflow DAG executions
6. **Customize Policies**: Update governance policies in `./governance/policies/`

## 🆘 Getting Help

- **Logs**: Check `docker-compose logs <service-name>`
- **Documentation**: Review README.md and project.md
- **Health Status**: Use `docker-compose ps` to check service status
- **Database Access**: Use `docker-compose exec postgres psql -U platform_user -d datacorp_platform`

## 🔒 Security Notes

- Change default passwords in production
- Use proper secrets management
- Enable SSL/TLS for external access
- Review and update governance policies
- Monitor audit logs regularly

---

**Setup Time**: ~10-15 minutes  
**Resource Requirements**: 8GB RAM, 20GB disk  
**Services**: 8 containers running  
**Ready for**: Production governance workflows