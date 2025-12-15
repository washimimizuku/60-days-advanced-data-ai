# Day 24: Production Pipeline - Setup Guide

## Quick Start (10 minutes)

```bash
# 1. Navigate to day 24 directory
cd days/day-24-project-production-pipeline

# 2. Run automated setup
./setup.sh

# 3. Start interactive demo
python demo.py
```

## Manual Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- 8GB+ available RAM

### Step-by-Step Setup

1. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env if needed (defaults work for local development)
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize Airflow**
   ```bash
   export AIRFLOW_HOME=$(pwd)/airflow
   airflow db init
   airflow users create --username admin --password admin --firstname Admin --lastname User --role Admin --email admin@example.com
   ```

4. **Start Services**
   ```bash
   docker-compose up -d
   ```

### Verify Setup

```bash
# Check services are running
docker-compose ps

# Test Airflow connection
curl http://localhost:8080/health

# Test database connection
docker exec -it day-24-project-production-pipeline-postgres-1 psql -U airflow -d dataflow_warehouse -c "SELECT 1;"
```

## Services

- **Airflow**: http://localhost:8080 (admin/admin)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **PostgreSQL**: localhost:5432 (airflow/airflow)

## Architecture

```
┌─────────────────────────────────────────┐
│           Production Pipeline            │
├─────────────────────────────────────────┤
│                                         │
│  Airflow ──► dbt ──► Great Expectations │
│     │                                   │
│     ▼                                   │
│  PostgreSQL ◄──► Monitoring Stack      │
│                                         │
└─────────────────────────────────────────┘
```

## What's Included

- ✅ Complete Airflow orchestration setup
- ✅ dbt transformation models
- ✅ PostgreSQL data warehouse
- ✅ Monitoring with Prometheus/Grafana
- ✅ LocalStack for AWS simulation
- ✅ Interactive demo script
- ✅ Comprehensive test suite
- ✅ Automated setup and teardown

## Troubleshooting

### Services not starting
```bash
docker-compose down
docker-compose up -d
docker-compose logs
```

### Airflow connection issues
```bash
# Reset Airflow database
airflow db reset
airflow db init
```

### Port conflicts
Edit `docker-compose.yml` to change port mappings if needed.