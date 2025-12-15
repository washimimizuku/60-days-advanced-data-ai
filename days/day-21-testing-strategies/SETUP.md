# Day 21: Testing Strategies - Setup Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ (if running locally)
- 8GB+ RAM recommended
- 5GB+ free disk space

### Automated Setup
```bash
# 1. Clone and navigate
cd day-21-testing-strategies

# 2. Run automated setup
./scripts/setup.sh

# 3. Verify installation
docker-compose exec testing-env python -m pytest tests/unit/test_setup.py -v
```

### Manual Setup
```bash
# 1. Create environment file
cp .env.example .env

# 2. Start services
docker-compose up -d

# 3. Install dependencies
docker-compose exec testing-env pip install -r requirements.txt

# 4. Initialize database
docker-compose exec postgres psql -U testuser -d testing_db -f /docker-entrypoint-initdb.d/init.sql
```

## 🧪 Running Tests

### All Tests
```bash
# Using test runner script
./scripts/run_tests.py --all

# Using pytest directly
docker-compose exec testing-env pytest
```

### Specific Test Suites
```bash
# Unit tests only
./scripts/run_tests.py --suite unit

# Integration tests
./scripts/run_tests.py --suite integration

# Performance tests
./scripts/run_tests.py --suite performance

# End-to-end tests
./scripts/run_tests.py --suite e2e
```

### With Coverage
```bash
docker-compose exec testing-env pytest --cov=. --cov-report=html
# View report: open htmlcov/index.html
```

## 📊 Interactive Demo

### Run Complete Demo
```bash
# Comprehensive framework demonstration
docker-compose exec testing-env python demo.py
```

### Step-by-Step Demo
```bash
# 1. Generate sample data
docker-compose exec testing-env python -c "
from demo import TestingFrameworkDemo
demo = TestingFrameworkDemo()
demo.generate_demo_data(1000)
"

# 2. Run unit tests demo
docker-compose exec testing-env python -c "
from demo import TestingFrameworkDemo
demo = TestingFrameworkDemo()
demo.demo_unit_testing()
"
```

## 🔧 Development Environment

### Local Development
```bash
# 1. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variables
export DATABASE_URL="postgresql://testuser:testpass123@localhost:5432/testing_db"
export TESTING_MODE=true

# 4. Run tests
pytest
```

### Jupyter Notebook
```bash
# Start Jupyter in container
docker-compose exec testing-env jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root

# Access at: http://localhost:8888
```

## 📈 Monitoring and Dashboards

### Grafana Dashboard
- **URL**: http://localhost:3000
- **Username**: admin
- **Password**: admin123
- **Features**: Test results visualization, performance metrics

### Prometheus Metrics
- **URL**: http://localhost:9090
- **Features**: Real-time metrics collection, alerting

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U testuser -d testing_db

# View test results
SELECT * FROM testing.test_results ORDER BY executed_at DESC LIMIT 10;
```

## 🛠️ Configuration

### Environment Variables (.env)
```bash
# Database
DATABASE_URL=postgresql://testuser:testpass123@localhost:5432/testing_db

# Testing
TESTING_MODE=true
COVERAGE_THRESHOLD=85
PERFORMANCE_THRESHOLD_SECONDS=5.0

# Monitoring
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
```

### Test Configuration (pytest.ini)
```ini
[tool:pytest]
addopts = --cov=. --cov-report=html --cov-fail-under=85
testpaths = tests
markers = 
    unit: Unit tests
    integration: Integration tests
    performance: Performance tests
```

## 📁 Project Structure

```
day-21-testing-strategies/
├── README.md                    # Main documentation
├── SETUP.md                     # This setup guide
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker services
├── Dockerfile                   # Testing environment
├── pytest.ini                  # Pytest configuration
├── .env.example                 # Environment template
│
├── data/                        # Sample data and database init
│   └── init.sql                 # Database initialization
│
├── tests/                       # Test suites
│   ├── conftest.py             # Shared fixtures
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   ├── e2e/                    # End-to-end tests
│   └── performance/            # Performance tests
│
├── scripts/                     # Automation scripts
│   ├── setup.sh               # Environment setup
│   └── run_tests.py           # Test runner
│
├── monitoring/                  # Monitoring configuration
│   ├── grafana/               # Grafana dashboards
│   └── prometheus/            # Prometheus config
│
├── .github/workflows/          # CI/CD pipelines
│   └── testing-pipeline.yml   # GitHub Actions
│
├── exercise.py                 # Hands-on exercises
├── solution.py                 # Complete solutions
├── demo.py                     # Interactive demo
└── quiz.md                     # Knowledge quiz
```

## 🔍 Troubleshooting

### Common Issues

#### Docker Services Not Starting
```bash
# Check Docker status
docker --version
docker-compose --version

# Restart services
docker-compose down
docker-compose up -d

# Check logs
docker-compose logs postgres
docker-compose logs testing-env
```

#### Database Connection Issues
```bash
# Test database connection
docker-compose exec postgres pg_isready -U testuser -d testing_db

# Recreate database
docker-compose down -v
docker-compose up -d
```

#### Permission Issues
```bash
# Fix script permissions
chmod +x scripts/setup.sh
chmod +x scripts/run_tests.py
chmod +x demo.py

# Fix directory permissions
sudo chown -R $USER:$USER .
```

#### Memory Issues
```bash
# Check available memory
free -h

# Reduce test data size
export TEST_DATA_SIZE=500

# Run tests sequentially
pytest -n 1
```

### Performance Optimization

#### Faster Test Execution
```bash
# Parallel execution
pytest -n auto

# Skip slow tests
pytest -m "not slow"

# Run only failed tests
pytest --lf
```

#### Reduce Resource Usage
```bash
# Limit Docker memory
docker-compose up -d --scale testing-env=1

# Use smaller datasets
export TEST_DATA_SIZE=100
```

## 📚 Additional Resources

### Documentation
- [pytest Documentation](https://docs.pytest.org/)
- [Docker Compose Guide](https://docs.docker.com/compose/)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)

### Testing Best Practices
- [Testing Pyramid](https://martinfowler.com/articles/practical-test-pyramid.html)
- [Data Testing Patterns](https://www.thoughtworks.com/insights/blog/data-testing-patterns)
- [CI/CD for Data Pipelines](https://www.databricks.com/blog/2019/09/20/ci-cd-for-machine-learning-data-pipelines.html)

### Monitoring and Observability
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Guide](https://prometheus.io/docs/)
- [Data Observability](https://www.montecarlodata.com/blog-what-is-data-observability/)

## 🆘 Getting Help

### Check Status
```bash
# Service health
docker-compose ps

# Test environment
docker-compose exec testing-env python -c "import pandas; print('✅ Environment ready')"

# Database connectivity
docker-compose exec testing-env python -c "
import os
import psycopg2
try:
    conn = psycopg2.connect(os.getenv('DATABASE_URL'))
    print('✅ Database connected')
    conn.close()
except Exception as e:
    print(f'❌ Database error: {e}')
"
```

### Debug Mode
```bash
# Verbose test output
pytest -v -s

# Debug specific test
pytest tests/unit/test_setup.py::TestEnvironmentSetup::test_python_version -v -s

# Interactive debugging
pytest --pdb
```

### Support
- Check [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) for common issues
- Review logs in `logs/` directory
- Run setup verification: `pytest tests/unit/test_setup.py -v`

---

## ✅ Verification Checklist

After setup, verify these components work:

- [ ] Docker services running (`docker-compose ps`)
- [ ] Database accessible (`docker-compose exec postgres pg_isready`)
- [ ] Python environment ready (`pytest tests/unit/test_setup.py`)
- [ ] Unit tests pass (`pytest tests/unit/ -v`)
- [ ] Demo runs successfully (`python demo.py`)
- [ ] Grafana accessible (http://localhost:3000)
- [ ] Coverage reports generated (`pytest --cov=.`)

🎉 **Setup Complete!** You're ready to explore comprehensive testing strategies for data pipelines.