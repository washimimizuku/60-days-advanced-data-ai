# Day 60: Capstone Production System - Setup Guide

## Overview

This guide provides step-by-step instructions for setting up the complete **Intelligent Customer Analytics Platform** - the capstone project that integrates all technologies from the 60-day curriculum.

## Prerequisites

### Required Software
- **Python 3.9+** with pip
- **Docker Desktop** with Kubernetes enabled
- **kubectl** CLI tool
- **Terraform** (latest version)
- **AWS CLI** configured with credentials
- **Git** for version control

### Required Accounts
- **AWS Account** with appropriate permissions
- **OpenAI API Key** (optional, system works in mock mode without it)
- **MLflow** tracking server (can be local)

### System Requirements
- **Memory**: 16GB+ RAM recommended
- **Storage**: 50GB+ free disk space
- **CPU**: 4+ cores recommended
- **Network**: Stable internet connection

## Quick Start (10 minutes)

### 1. Clone and Setup Environment

```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/60-days-advanced-data-ai.git
cd 60-days-advanced-data-ai/days/day-60-capstone-production-system

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your configuration
nano .env
```

### 3. Start Local Development Stack

```bash
# Start infrastructure services
docker-compose up -d

# Verify services are running
docker-compose ps

# Run the application
python solution.py
```

### 4. Verify Installation

```bash
# Test the setup
python test_setup.py

# Run basic tests
pytest test_capstone_production_system.py -v
```

## Detailed Setup Instructions

### Infrastructure Setup

#### 1. Docker Compose Services

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: customers
      POSTGRES_USER: analytics_user
      POSTGRES_PASSWORD: secure_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-db.sql:/docker-entrypoint-initdb.d/init-db.sql

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

  mongodb:
    image: mongo:7
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: password
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db

  kafka:
    image: confluentinc/cp-kafka:latest
    environment:
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"
    depends_on:
      - zookeeper

  zookeeper:
    image: confluentinc/cp-zookeeper:latest
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000
    ports:
      - "2181:2181"

  mlflow:
    image: python:3.9-slim
    command: >
      bash -c "pip install mlflow boto3 psycopg2-binary &&
               mlflow server --host 0.0.0.0 --port 5000 --default-artifact-root ./mlruns"
    ports:
      - "5000:5000"
    volumes:
      - mlflow_data:/mlflow

  chromadb:
    image: chromadb/chroma:latest
    ports:
      - "8000:8000"
    volumes:
      - chromadb_data:/chroma/chroma

volumes:
  postgres_data:
  redis_data:
  mongodb_data:
  mlflow_data:
  chromadb_data:
```

#### 2. Database Initialization

Create `init-db.sql`:

```sql
-- Initialize customer analytics database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Customers table
CREATE TABLE customers (
    customer_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    segment VARCHAR(50) DEFAULT 'Basic',
    lifetime_value DECIMAL(10,2) DEFAULT 0.00,
    total_purchases INTEGER DEFAULT 0,
    avg_order_value DECIMAL(10,2) DEFAULT 0.00,
    days_since_last_purchase INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transactions table
CREATE TABLE transactions (
    transaction_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID REFERENCES customers(customer_id),
    amount DECIMAL(10,2) NOT NULL,
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    product_category VARCHAR(100),
    payment_method VARCHAR(50)
);

-- Events table for real-time tracking
CREATE TABLE events (
    event_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    customer_id UUID REFERENCES customers(customer_id),
    event_type VARCHAR(100) NOT NULL,
    event_data JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_customers_segment ON customers(segment);
CREATE INDEX idx_customers_ltv ON customers(lifetime_value);
CREATE INDEX idx_customers_registration ON customers(registration_date);
CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_events_customer ON events(customer_id);
CREATE INDEX idx_events_type ON events(event_type);
CREATE INDEX idx_events_timestamp ON events(timestamp);

-- Sample data
INSERT INTO customers (email, first_name, last_name, segment, lifetime_value, total_purchases, avg_order_value, days_since_last_purchase) VALUES
('john.doe@example.com', 'John', 'Doe', 'Premium', 5000.00, 25, 200.00, 15),
('jane.smith@example.com', 'Jane', 'Smith', 'Standard', 2500.00, 15, 166.67, 30),
('bob.johnson@example.com', 'Bob', 'Johnson', 'Basic', 500.00, 5, 100.00, 60);
```

### Kubernetes Setup

#### 1. Enable Kubernetes in Docker Desktop

1. Open Docker Desktop
2. Go to Settings → Kubernetes
3. Check "Enable Kubernetes"
4. Click "Apply & Restart"

#### 2. Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace customer-analytics

# Apply configurations
kubectl apply -f kubernetes/ -n customer-analytics

# Verify deployment
kubectl get pods -n customer-analytics
kubectl get services -n customer-analytics
```

### AWS Infrastructure Setup

#### 1. Configure AWS CLI

```bash
# Configure AWS credentials
aws configure

# Verify configuration
aws sts get-caller-identity
```

#### 2. Deploy with Terraform

```bash
# Initialize Terraform
cd infrastructure/terraform
terraform init

# Plan deployment
terraform plan -var-file="production.tfvars"

# Apply infrastructure
terraform apply -var-file="production.tfvars"
```

### Feature Store Setup

#### 1. Initialize Feast

```bash
# Create feature repository
mkdir feature_repo
cd feature_repo

# Initialize Feast
feast init

# Apply feature definitions
feast apply
```

#### 2. Configure Feature Store

Create `feature_store.yaml`:

```yaml
project: customer_analytics
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: "redis://localhost:6379"
offline_store:
  type: file
```

### Monitoring Setup

#### 1. Prometheus Configuration

Create `prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'customer-analytics-api'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
```

#### 2. Grafana Dashboards

Import the provided dashboard JSON files:
- `dashboards/system-overview.json`
- `dashboards/ml-performance.json`
- `dashboards/genai-metrics.json`

## Configuration

### Environment Variables

Required environment variables in `.env`:

```bash
# Database Configuration
POSTGRES_URL=postgresql://analytics_user:secure_password@localhost:5432/customers
MONGODB_URL=mongodb://admin:password@localhost:27017/events
REDIS_URL=redis://localhost:6379

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092

# ML Platform
MLFLOW_TRACKING_URI=http://localhost:5000
FEAST_REPO_PATH=./feature_repo

# GenAI Configuration
OPENAI_API_KEY=your-openai-api-key-here
CHROMA_HOST=localhost:8000

# AWS Configuration
AWS_REGION=us-west-2
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key

# Application Settings
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Security Configuration

#### 1. API Authentication

For production, replace the simple authentication in `solution.py`:

```python
# Use proper JWT authentication
from jose import JWTError, jwt
from passlib.context import CryptContext

SECRET_KEY = "your-secret-key"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
```

#### 2. Database Security

- Use strong passwords
- Enable SSL/TLS connections
- Configure network security groups
- Implement row-level security

## Testing

### Unit Tests

```bash
# Run all tests
pytest test_capstone_production_system.py -v

# Run specific test class
pytest test_capstone_production_system.py::TestAPIEndpoints -v

# Run with coverage
pytest test_capstone_production_system.py --cov=solution --cov-report=html
```

### Integration Tests

```bash
# Test database connectivity
python -c "import asyncio; from solution import DatabaseManager; asyncio.run(DatabaseManager().initialize())"

# Test ML pipeline
python -c "import asyncio; from solution import MLModelManager; asyncio.run(MLModelManager().initialize())"

# Test GenAI system
python -c "import asyncio; from solution import GenAIInsightGenerator; asyncio.run(GenAIInsightGenerator().initialize())"
```

### Load Testing

```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f load_tests.py --host=http://localhost:8080
```

## Deployment

### Local Development

```bash
# Start all services
docker-compose up -d

# Run application
python solution.py

# Access API documentation
open http://localhost:8080/docs
```

### Production Deployment

#### 1. Build Docker Images

```bash
# Build application image
docker build -t customer-analytics:latest .

# Push to registry
docker tag customer-analytics:latest your-registry/customer-analytics:latest
docker push your-registry/customer-analytics:latest
```

#### 2. Deploy to Kubernetes

```bash
# Update image in deployment
kubectl set image deployment/ml-api ml-api=your-registry/customer-analytics:latest -n customer-analytics

# Verify deployment
kubectl rollout status deployment/ml-api -n customer-analytics
```

#### 3. Configure Auto-scaling

```bash
# Apply HPA configuration
kubectl apply -f kubernetes/hpa.yaml -n customer-analytics

# Verify auto-scaling
kubectl get hpa -n customer-analytics
```

## Monitoring and Observability

### Metrics Collection

The application exposes metrics at `/metrics` endpoint:

- `api_requests_total` - Total API requests
- `api_request_duration_seconds` - Request duration
- `model_predictions_total` - ML predictions count
- `genai_insights_generated_total` - GenAI insights count
- `feature_store_requests_total` - Feature store requests

### Logging

Structured logging is configured with:

- **Level**: INFO (configurable via LOG_LEVEL)
- **Format**: JSON structured logs
- **Fields**: timestamp, level, message, context

### Alerting

Configure alerts for:

- High error rates (>5%)
- High response times (>1s p95)
- Low model accuracy (<80%)
- System resource usage (>80%)

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors

```bash
# Check database status
docker-compose ps postgres

# View database logs
docker-compose logs postgres

# Test connection
psql -h localhost -U analytics_user -d customers
```

#### 2. Kafka Connection Issues

```bash
# Check Kafka status
docker-compose ps kafka zookeeper

# List topics
docker-compose exec kafka kafka-topics --list --bootstrap-server localhost:9092

# Test producer/consumer
docker-compose exec kafka kafka-console-producer --topic test --bootstrap-server localhost:9092
```

#### 3. ML Model Loading Errors

```bash
# Check MLflow server
curl http://localhost:5000/api/2.0/mlflow/experiments/list

# Verify model registry
mlflow models list

# Test model loading
python -c "import mlflow; print(mlflow.sklearn.load_model('models:/churn_prediction/latest'))"
```

#### 4. GenAI Service Issues

```bash
# Check OpenAI API key
python -c "import openai; openai.api_key='your-key'; print(openai.Model.list())"

# Test ChromaDB connection
curl http://localhost:8000/api/v1/heartbeat

# Verify vector store
python -c "from langchain.vectorstores import Chroma; print(Chroma(persist_directory='./customer_knowledge_base'))"
```

### Performance Optimization

#### 1. Database Optimization

- Add appropriate indexes
- Use connection pooling
- Optimize query patterns
- Enable query caching

#### 2. API Optimization

- Implement response caching
- Use async/await properly
- Optimize serialization
- Enable compression

#### 3. ML Pipeline Optimization

- Cache feature computations
- Batch predictions
- Use model quantization
- Implement model caching

## Security Considerations

### Production Security Checklist

- [ ] Use HTTPS/TLS for all communications
- [ ] Implement proper authentication and authorization
- [ ] Encrypt sensitive data at rest and in transit
- [ ] Use secrets management (AWS Secrets Manager, Kubernetes secrets)
- [ ] Enable audit logging
- [ ] Implement rate limiting
- [ ] Use network security groups and firewalls
- [ ] Regular security updates and patches
- [ ] Vulnerability scanning
- [ ] Data privacy compliance (GDPR, CCPA)

### Data Privacy

- Implement data anonymization
- Use data retention policies
- Enable right to be forgotten
- Audit data access
- Encrypt PII data

## Cost Optimization

### AWS Cost Management

- Use spot instances for batch workloads
- Implement auto-scaling policies
- Monitor resource utilization
- Use reserved instances for predictable workloads
- Enable cost alerts and budgets

### Resource Optimization

- Right-size compute resources
- Use efficient data formats (Parquet, ORC)
- Implement data lifecycle policies
- Optimize storage classes
- Monitor and optimize network costs

## Support and Resources

### Documentation

- [API Documentation](http://localhost:8080/docs) - Interactive API docs
- [Architecture Guide](./docs/architecture.md) - System architecture
- [Deployment Guide](./docs/deployment.md) - Production deployment
- [Troubleshooting Guide](./docs/troubleshooting.md) - Common issues

### Community

- GitHub Issues - Bug reports and feature requests
- Discord Channel - Real-time support
- Office Hours - Weekly Q&A sessions

### Additional Resources

- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction.html)

---

**Congratulations!** You now have a complete production-ready customer analytics platform running. This system demonstrates the integration of all 60 days of curriculum into a real-world application that can handle enterprise-scale workloads.