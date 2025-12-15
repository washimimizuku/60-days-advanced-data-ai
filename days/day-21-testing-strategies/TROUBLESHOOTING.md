# Day 21: Testing Strategies - Troubleshooting Guide

## 🚨 Common Issues and Solutions

### Docker and Environment Issues

#### Issue: Docker services won't start
```bash
# Symptoms
docker-compose up -d
# Error: Cannot connect to the Docker daemon

# Solutions
# 1. Start Docker daemon
sudo systemctl start docker  # Linux
# or restart Docker Desktop    # Windows/Mac

# 2. Check Docker installation
docker --version
docker-compose --version

# 3. Reset Docker if needed
docker system prune -a
```

#### Issue: Port conflicts
```bash
# Symptoms
Error: Port 5432 is already in use

# Solutions
# 1. Check what's using the port
sudo lsof -i :5432  # Linux/Mac
netstat -ano | findstr :5432  # Windows

# 2. Stop conflicting service
sudo systemctl stop postgresql  # If PostgreSQL is running locally

# 3. Use different ports in docker-compose.yml
ports:
  - "5433:5432"  # Change external port
```

#### Issue: Permission denied errors
```bash
# Symptoms
Permission denied: './scripts/setup.sh'

# Solutions
# 1. Fix script permissions
chmod +x scripts/setup.sh
chmod +x scripts/run_tests.py
chmod +x demo.py

# 2. Fix directory ownership
sudo chown -R $USER:$USER .

# 3. Run with sudo if needed (not recommended)
sudo ./scripts/setup.sh
```

### Database Issues

#### Issue: Database connection failed
```bash
# Symptoms
psycopg2.OperationalError: could not connect to server

# Diagnosis
docker-compose exec postgres pg_isready -U testuser -d testing_db

# Solutions
# 1. Wait for database to be ready
sleep 30
docker-compose exec postgres pg_isready -U testuser -d testing_db

# 2. Check database logs
docker-compose logs postgres

# 3. Recreate database
docker-compose down -v
docker-compose up -d postgres
sleep 30
```

#### Issue: Database initialization failed
```bash
# Symptoms
ERROR: relation "testing.transactions" does not exist

# Solutions
# 1. Manually initialize database
docker-compose exec postgres psql -U testuser -d testing_db -f /docker-entrypoint-initdb.d/init.sql

# 2. Check if init.sql exists
docker-compose exec postgres ls -la /docker-entrypoint-initdb.d/

# 3. Recreate with volume reset
docker-compose down -v
docker-compose up -d
```

#### Issue: Database authentication failed
```bash
# Symptoms
FATAL: password authentication failed for user "testuser"

# Solutions
# 1. Check environment variables
docker-compose exec testing-env env | grep DATABASE

# 2. Reset database with correct credentials
docker-compose down -v
# Edit .env file with correct credentials
docker-compose up -d

# 3. Use trust authentication temporarily
# In docker-compose.yml:
environment:
  POSTGRES_HOST_AUTH_METHOD: trust
```

### Python and Testing Issues

#### Issue: Import errors
```bash
# Symptoms
ModuleNotFoundError: No module named 'pandas'

# Solutions
# 1. Install dependencies in container
docker-compose exec testing-env pip install -r requirements.txt

# 2. Check Python path
docker-compose exec testing-env python -c "import sys; print(sys.path)"

# 3. Rebuild container
docker-compose build testing-env
docker-compose up -d testing-env
```

#### Issue: Test failures due to missing data
```bash
# Symptoms
FileNotFoundError: [Errno 2] No such file or directory: 'test_data.csv'

# Solutions
# 1. Generate test data
docker-compose exec testing-env python -c "
from demo import TestingFrameworkDemo
demo = TestingFrameworkDemo()
demo.generate_demo_data(1000)
"

# 2. Check data directory
docker-compose exec testing-env ls -la data/

# 3. Initialize database with sample data
docker-compose exec postgres psql -U testuser -d testing_db -f /docker-entrypoint-initdb.d/init.sql
```

#### Issue: Coverage reports not generated
```bash
# Symptoms
No coverage report found

# Solutions
# 1. Install coverage package
docker-compose exec testing-env pip install coverage pytest-cov

# 2. Run tests with coverage
docker-compose exec testing-env pytest --cov=. --cov-report=html

# 3. Check coverage configuration in pytest.ini
[tool:pytest]
addopts = --cov=. --cov-report=html --cov-report=xml
```

### Performance Issues

#### Issue: Tests running slowly
```bash
# Symptoms
Tests take more than 5 minutes to complete

# Solutions
# 1. Run tests in parallel
docker-compose exec testing-env pytest -n auto

# 2. Skip slow tests
docker-compose exec testing-env pytest -m "not slow"

# 3. Reduce test data size
export TEST_DATA_SIZE=100
docker-compose exec testing-env pytest

# 4. Run only specific test suites
docker-compose exec testing-env pytest tests/unit/
```

#### Issue: Memory errors
```bash
# Symptoms
MemoryError: Unable to allocate array

# Solutions
# 1. Increase Docker memory limit
# In Docker Desktop: Settings > Resources > Memory > 8GB

# 2. Reduce test data size
# In .env file:
TEST_DATA_SIZE=500

# 3. Run tests sequentially
docker-compose exec testing-env pytest -n 1

# 4. Clear memory between tests
docker-compose exec testing-env python -c "
import gc
gc.collect()
"
```

#### Issue: Disk space errors
```bash
# Symptoms
OSError: [Errno 28] No space left on device

# Solutions
# 1. Clean Docker system
docker system prune -a

# 2. Remove old test results
rm -rf test_results/* htmlcov/*

# 3. Check disk usage
df -h
docker system df

# 4. Clean up volumes
docker volume prune
```

### Monitoring and Dashboard Issues

#### Issue: Grafana not accessible
```bash
# Symptoms
Cannot connect to http://localhost:3000

# Solutions
# 1. Check if Grafana is running
docker-compose ps grafana

# 2. Check Grafana logs
docker-compose logs grafana

# 3. Restart Grafana
docker-compose restart grafana

# 4. Check port mapping
docker-compose ps | grep grafana
```

#### Issue: Prometheus not collecting metrics
```bash
# Symptoms
No data in Prometheus dashboard

# Solutions
# 1. Check Prometheus configuration
docker-compose exec prometheus cat /etc/prometheus/prometheus.yml

# 2. Check if targets are up
# Visit http://localhost:9090/targets

# 3. Restart Prometheus
docker-compose restart prometheus

# 4. Check network connectivity
docker-compose exec prometheus ping testing-env
```

#### Issue: Dashboard shows no data
```bash
# Symptoms
Grafana dashboard is empty

# Solutions
# 1. Check datasource configuration
# Grafana > Configuration > Data Sources

# 2. Verify database connection
docker-compose exec grafana curl -f http://postgres:5432

# 3. Run some tests to generate data
docker-compose exec testing-env pytest tests/unit/

# 4. Check if data exists in database
docker-compose exec postgres psql -U testuser -d testing_db -c "SELECT COUNT(*) FROM testing.test_results;"
```

### CI/CD and Integration Issues

#### Issue: GitHub Actions failing
```bash
# Symptoms
CI pipeline fails on GitHub

# Solutions
# 1. Check workflow syntax
# Use GitHub's workflow validator

# 2. Test locally with act
act -j unit-tests

# 3. Check secrets and environment variables
# GitHub > Settings > Secrets and variables

# 4. Review action logs
# GitHub > Actions > Failed workflow > View logs
```

#### Issue: Test results not being saved
```bash
# Symptoms
No test results in test_results/ directory

# Solutions
# 1. Create results directory
mkdir -p test_results

# 2. Check permissions
chmod 755 test_results

# 3. Run tests with explicit output
docker-compose exec testing-env pytest --junit-xml=test_results/results.xml

# 4. Check if directory is mounted
docker-compose exec testing-env ls -la test_results/
```

## 🔧 Diagnostic Commands

### System Health Check
```bash
#!/bin/bash
echo "🔍 System Health Check"
echo "====================="

# Docker status
echo "Docker version:"
docker --version
docker-compose --version

# Services status
echo -e "\nServices status:"
docker-compose ps

# Database connectivity
echo -e "\nDatabase connectivity:"
docker-compose exec -T postgres pg_isready -U testuser -d testing_db

# Python environment
echo -e "\nPython environment:"
docker-compose exec -T testing-env python --version
docker-compose exec -T testing-env python -c "import pandas, numpy, pytest; print('✅ Core packages available')"

# Disk space
echo -e "\nDisk space:"
df -h | grep -E "(Filesystem|/dev/)"

# Memory usage
echo -e "\nMemory usage:"
free -h | grep -E "(total|Mem)"

echo -e "\n✅ Health check complete"
```

### Test Environment Verification
```bash
#!/bin/bash
echo "🧪 Test Environment Verification"
echo "==============================="

# Run setup tests
docker-compose exec testing-env pytest tests/unit/test_setup.py -v

# Check test data
echo -e "\nTest data availability:"
docker-compose exec testing-env python -c "
import pandas as pd
from pathlib import Path
if Path('data/init.sql').exists():
    print('✅ Database initialization script found')
else:
    print('❌ Database initialization script missing')
"

# Verify database schema
echo -e "\nDatabase schema:"
docker-compose exec postgres psql -U testuser -d testing_db -c "\dt testing.*"

# Check monitoring endpoints
echo -e "\nMonitoring endpoints:"
curl -f http://localhost:3000/api/health 2>/dev/null && echo "✅ Grafana healthy" || echo "❌ Grafana not accessible"
curl -f http://localhost:9090/-/healthy 2>/dev/null && echo "✅ Prometheus healthy" || echo "❌ Prometheus not accessible"

echo -e "\n✅ Verification complete"
```

## 🚀 Performance Optimization

### Speed Up Test Execution
```bash
# 1. Use pytest-xdist for parallel execution
pip install pytest-xdist
pytest -n auto

# 2. Use pytest-benchmark for performance tests
pip install pytest-benchmark
pytest --benchmark-only

# 3. Skip slow tests during development
pytest -m "not slow"

# 4. Use pytest-cache for incremental testing
pytest --cache-clear  # Clear cache
pytest --lf          # Run last failed
pytest --ff          # Run failed first
```

### Optimize Docker Performance
```bash
# 1. Use multi-stage builds
# In Dockerfile:
FROM python:3.11-slim as base
# ... base setup ...

FROM base as testing
# ... testing specific setup ...

# 2. Use .dockerignore
echo "*.pyc
__pycache__
.git
.pytest_cache
htmlcov" > .dockerignore

# 3. Optimize layer caching
# Copy requirements first, then code
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

### Database Performance Tuning
```sql
-- 1. Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_transactions_customer_date 
ON testing.transactions(customer_id, transaction_date);

-- 2. Analyze tables for better query plans
ANALYZE testing.transactions;
ANALYZE testing.customer_metrics;

-- 3. Optimize PostgreSQL settings
-- In postgresql.conf:
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
```

## 📞 Getting Additional Help

### Log Collection
```bash
# Collect all logs for support
mkdir -p debug_logs
docker-compose logs > debug_logs/docker-compose.log
docker-compose logs postgres > debug_logs/postgres.log
docker-compose logs testing-env > debug_logs/testing-env.log
docker-compose logs grafana > debug_logs/grafana.log

# System information
uname -a > debug_logs/system_info.txt
docker info > debug_logs/docker_info.txt
docker-compose version > debug_logs/compose_version.txt

# Create debug archive
tar -czf debug_logs_$(date +%Y%m%d_%H%M%S).tar.gz debug_logs/
```

### Support Checklist
Before seeking help, please:

1. ✅ Run the health check script
2. ✅ Check the logs for error messages
3. ✅ Verify all prerequisites are installed
4. ✅ Try the suggested solutions above
5. ✅ Collect debug information

### Community Resources
- **GitHub Issues**: Report bugs and feature requests
- **Documentation**: Check README.md and SETUP.md
- **Examples**: Review demo.py and solution.py
- **Stack Overflow**: Search for similar issues

---

## 🎯 Quick Fix Commands

```bash
# Nuclear option: Reset everything
docker-compose down -v
docker system prune -a
./scripts/setup.sh

# Fix permissions
chmod +x scripts/*.sh scripts/*.py *.py

# Rebuild and restart
docker-compose build --no-cache
docker-compose up -d

# Verify setup
docker-compose exec testing-env pytest tests/unit/test_setup.py -v
```

Remember: Most issues are resolved by ensuring Docker is running, services are healthy, and dependencies are properly installed. When in doubt, try the setup script again! 🚀