# Day 12 Setup Guide: Apache Airflow Basics

## Quick Start (5 minutes)

```bash
# 1. Navigate to day 12
cd days/day-12-airflow-basics

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment
cp .env.example .env

# 4. Start Airflow with Docker
docker-compose up -d

# 5. Initialize Airflow
./scripts/init_airflow.sh

# 6. Access Airflow UI
open http://localhost:8080
```

## Detailed Setup

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- 4GB+ RAM available
- 10GB+ disk space

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv airflow-env
source airflow-env/bin/activate  # On Windows: airflow-env\Scripts\activate

# Install Airflow and dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Generate Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Update .env with generated key
nano .env
```

**Required Configuration Changes:**
- `AIRFLOW__CORE__FERNET_KEY`: Use generated Fernet key
- `AIRFLOW__WEBSERVER__SECRET_KEY`: Generate with `openssl rand -hex 32`
- Email settings if using notifications

### 3. Airflow Setup Options

#### Option A: Docker (Recommended)
```bash
# Start all services
docker-compose up -d

# Check services
docker-compose ps

# View logs
docker-compose logs airflow-webserver
```

#### Option B: Local Installation
```bash
# Set Airflow home
export AIRFLOW_HOME=$(pwd)/airflow_home

# Initialize database
airflow db init

# Create admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start webserver (terminal 1)
airflow webserver --port 8080

# Start scheduler (terminal 2)
airflow scheduler
```

### 4. Verify Installation

```bash
# Test Airflow CLI
airflow version

# List DAGs
airflow dags list

# Test connections
airflow connections list

# Run test DAG
airflow dags test example_etl_dag 2024-01-01
```

## Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Webserver     │    │   Scheduler     │    │   PostgreSQL    │
│   (Port 8080)   │    │                 │    │   (Port 5432)   │
│                 │    │ • DAG Parsing   │    │ • Metadata DB   │
│ • Web UI        │◄──►│ • Task Scheduling│◄──►│ • Task State    │
│ • REST API      │    │ • Executors     │    │ • Connections   │
│ • Authentication│    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │     Worker      │
                    │                 │
                    │ • Task Execution│
                    │ • Log Generation│
                    │ • XCom Storage  │
                    └─────────────────┘
```

## Testing the Setup

### 1. Access Airflow UI
- URL: http://localhost:8080
- Username: admin
- Password: admin (or from setup)

### 2. Verify Sample DAGs
```bash
# Copy sample DAGs
cp dags/*.py $AIRFLOW_HOME/dags/

# Refresh DAGs in UI or wait for auto-refresh
```

### 3. Test DAG Execution
```bash
# Trigger sample DAG
airflow dags trigger example_etl_dag

# Check task status
airflow tasks list example_etl_dag
airflow tasks state example_etl_dag extract_data 2024-01-01
```

### 4. Test Connections
```bash
# Test database connection
airflow connections test postgres_default

# Add custom connection
airflow connections add 'my_postgres' \
    --conn-type 'postgres' \
    --conn-host 'localhost' \
    --conn-login 'user' \
    --conn-password 'password' \
    --conn-schema 'mydb'
```

## Sample DAGs Overview

The setup includes several sample DAGs:

### 1. example_etl_dag.py
- Basic ETL pipeline
- Extract → Transform → Load pattern
- Error handling and retries

### 2. example_branching_dag.py
- Conditional logic with BranchPythonOperator
- Dynamic task selection
- Join patterns

### 3. example_sensor_dag.py
- File and time sensors
- External dependency handling
- Timeout configurations

### 4. example_parallel_dag.py
- Parallel task execution
- Fan-out and fan-in patterns
- Resource management

## Exercise Workflow

### 1. Start with Basic DAG
```bash
# Edit the exercise file
nano exercise.py

# Copy to DAGs folder
cp exercise.py $AIRFLOW_HOME/dags/

# Check in Airflow UI
```

### 2. Test Your Implementation
```bash
# Validate DAG
python $AIRFLOW_HOME/dags/exercise.py

# Test specific task
airflow tasks test my_dag my_task 2024-01-01
```

### 3. Debug Issues
```bash
# Check DAG import errors
airflow dags list-import-errors

# View task logs
airflow tasks log my_dag my_task 2024-01-01 1
```

### 4. Compare with Solution
```bash
# Review production implementation
cat solution.py

# Copy solution DAG for testing
cp solution.py $AIRFLOW_HOME/dags/production_dag.py
```

## Troubleshooting

### Common Issues

#### Airflow Won't Start
```bash
# Check port availability
lsof -i :8080

# Reset database
airflow db reset

# Check configuration
airflow config list
```

#### DAG Import Errors
```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Validate DAG syntax
python -m py_compile dags/my_dag.py

# Check Airflow logs
tail -f $AIRFLOW_HOME/logs/scheduler/latest/*.log
```

#### Task Execution Failures
```bash
# Check task instance details
airflow tasks state my_dag my_task 2024-01-01

# View detailed logs
airflow tasks log my_dag my_task 2024-01-01 1

# Test task in isolation
airflow tasks test my_dag my_task 2024-01-01
```

#### Database Connection Issues
```bash
# Test database connectivity
airflow db check

# Check connection configuration
airflow connections get postgres_default

# Reset connections
airflow connections delete postgres_default
airflow connections add postgres_default --conn-uri postgresql://user:pass@host:5432/db
```

### Performance Issues

#### Slow DAG Loading
- Reduce DAG complexity
- Optimize imports in DAG files
- Increase `AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL`

#### Task Queue Buildup
- Check executor configuration
- Monitor resource usage
- Adjust parallelism settings

#### Memory Usage
- Monitor worker memory consumption
- Adjust `AIRFLOW__CORE__PARALLELISM`
- Use appropriate executor for workload

## Production Considerations

### Security
```bash
# Enable authentication
AIRFLOW__WEBSERVER__AUTHENTICATE=True
AIRFLOW__WEBSERVER__AUTH_BACKEND=airflow.auth.backends.password_auth

# Use strong Fernet key
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# Secure connections
# Store sensitive data in Airflow Variables/Connections, not in DAG code
```

### Monitoring
```bash
# Enable metrics
AIRFLOW__METRICS__STATSD_ON=True
AIRFLOW__METRICS__STATSD_HOST=localhost
AIRFLOW__METRICS__STATSD_PORT=8125

# Configure logging
AIRFLOW__LOGGING__REMOTE_LOGGING=True
AIRFLOW__LOGGING__REMOTE_BASE_LOG_FOLDER=s3://my-bucket/airflow-logs
```

### Scaling
```bash
# Use CeleryExecutor for distributed execution
AIRFLOW__CORE__EXECUTOR=CeleryExecutor
AIRFLOW__CELERY__BROKER_URL=redis://redis:6379/0
AIRFLOW__CELERY__RESULT_BACKEND=db+postgresql://airflow:airflow@postgres/airflow

# Or KubernetesExecutor for cloud-native scaling
AIRFLOW__CORE__EXECUTOR=KubernetesExecutor
```

## Next Steps

1. **Complete the exercises** in order (basic DAG → advanced features)
2. **Experiment with operators** - try different operator types
3. **Build custom operators** for specific use cases
4. **Integrate with external systems** (databases, APIs, cloud services)
5. **Deploy to production** following security and scaling best practices

## Additional Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [Airflow Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Airflow Concepts](https://airflow.apache.org/docs/apache-airflow/stable/concepts/index.html)
- [Operator Guide](https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/index.html)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review Airflow logs in the UI or CLI
3. Test components in isolation
4. Refer to official Airflow documentation