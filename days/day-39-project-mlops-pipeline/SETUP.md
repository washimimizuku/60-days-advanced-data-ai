# Day 39: Complete MLOps Pipeline - Setup Guide

## 🚀 Quick Start (10 minutes)

### Prerequisites
- Docker and Docker Compose installed
- Python 3.9+ with pip
- 16GB+ RAM recommended
- 50GB+ free disk space

### 1. Environment Setup
```bash
# Clone and navigate to project
cd day-39-project-mlops-pipeline

# Create virtual environment
python -m venv mlops_env
source mlops_env/bin/activate  # On Windows: mlops_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Infrastructure Deployment
```bash
# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check service health
docker-compose logs -f
```

### 3. Initialize Feature Store
```bash
# Create feature store directory
mkdir -p feature_store

# Initialize Feast
cd feature_store
feast init .
cd ..

# Apply feature definitions (after running solution.py once)
cd feature_store && feast apply && cd ..
```

### 4. Run Complete Pipeline
```bash
# Execute the complete MLOps pipeline
python solution.py
```

### 5. Access Services
- **Model API**: http://localhost:8000
- **MLflow UI**: http://localhost:5000
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Prometheus**: http://localhost:9090
- **Jupyter Lab**: http://localhost:8888 (token: mlops123)
- **Airflow**: http://localhost:8080 (admin/admin123)

---

## 🏗️ Detailed Setup

### Infrastructure Components

#### 1. PostgreSQL Database
```bash
# Connect to database
docker exec -it mlops_postgres psql -U mlops_user -d mlops_db

# Check tables
\dt

# Exit
\q
```

#### 2. Redis Feature Store
```bash
# Connect to Redis
docker exec -it mlops_redis redis-cli

# Check keys
KEYS *

# Exit
exit
```

#### 3. MinIO Object Storage
```bash
# Access MinIO console: http://localhost:9001
# Username: minioadmin
# Password: minioadmin123

# Create buckets for MLflow artifacts
# This is done automatically by the setup
```

#### 4. MLflow Tracking Server
```bash
# MLflow UI: http://localhost:5000
# Experiments and models are tracked automatically
# Check the "churn_prediction" experiment
```

### Feature Store Configuration

#### 1. Feast Setup
```bash
# Navigate to feature store
cd feature_store

# Check feature store status
feast status

# List feature views
feast feature-views list

# Get online features (after materialization)
feast get-online-features \
  --feature-service customer_features \
  --entity customer_id=CUST_000001
```

#### 2. Feature Materialization
```bash
# Materialize features to online store
feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)

# Check materialization status
feast status
```

### Model Training and Deployment

#### 1. AutoML Pipeline
```python
# The solution.py automatically:
# 1. Generates training data
# 2. Trains multiple models with hyperparameter optimization
# 3. Creates ensemble models
# 4. Registers models in MLflow
# 5. Deploys the best model for serving
```

#### 2. Model Serving API
```bash
# Test model API
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": "CUST_000001",
    "features": {
      "age": 35,
      "account_length_months": 24,
      "monthly_charges": 65.5,
      "total_charges": 1500.0,
      "support_calls_3m": 2,
      "login_frequency_30d": 15.5,
      "feature_usage_score": 75.0,
      "days_since_last_login": 5,
      "avg_session_duration_min": 25.0,
      "late_payments_12m": 0,
      "gender": 1,
      "contract_type": 0,
      "payment_method": 1
    }
  }'

# Check API health
curl http://localhost:8000/health

# Get model info
curl http://localhost:8000/model/info
```

### Monitoring and Observability

#### 1. Prometheus Metrics
```bash
# Access Prometheus: http://localhost:9090
# Check targets: Status > Targets
# Query metrics: predictions_total, prediction_latency_seconds
```

#### 2. Grafana Dashboards
```bash
# Access Grafana: http://localhost:3000
# Login: admin / admin123
# Import dashboard from monitoring/grafana/dashboards/
```

#### 3. Model Performance Monitoring
```python
# The monitoring system automatically:
# 1. Tracks prediction metrics
# 2. Detects data drift
# 3. Monitors model performance
# 4. Triggers alerts when thresholds are exceeded
```

---

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
pytest test_mlops_pipeline.py -v

# Run specific test categories
pytest test_mlops_pipeline.py::TestDataGenerator -v
pytest test_mlops_pipeline.py::TestAutoMLPipeline -v
pytest test_mlops_pipeline.py::TestModelServingAPI -v

# Run with coverage
pytest test_mlops_pipeline.py --cov=solution --cov-report=html
```

### Integration Tests
```bash
# Test complete pipeline
python test_mlops_pipeline.py

# Performance benchmarks
python -c "from test_mlops_pipeline import run_performance_tests; run_performance_tests()"
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load tests (create locustfile.py first)
locust -f locustfile.py --host=http://localhost:8000
```

---

## 🚀 Production Deployment

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -n mlops-pipeline
kubectl get services -n mlops-pipeline

# Scale model API
kubectl scale deployment model-api --replicas=5 -n mlops-pipeline
```

### Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration
nano .env

# Key settings:
# - Database connections
# - Model serving parameters
# - Monitoring thresholds
# - Security settings
```

### Security Setup
```bash
# Generate JWT secret key
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Set up SSL certificates (for production)
# Configure authentication and authorization
# Set up network security groups
```

---

## 📊 Monitoring and Maintenance

### Health Checks
```bash
# Check all service health
./scripts/health_check.sh

# Monitor resource usage
docker stats

# Check logs
docker-compose logs -f model_api
docker-compose logs -f mlflow
```

### Backup and Recovery
```bash
# Backup PostgreSQL
docker exec mlops_postgres pg_dump -U mlops_user mlops_db > backup.sql

# Backup Redis
docker exec mlops_redis redis-cli BGSAVE

# Backup MinIO
# Use MinIO client (mc) for bucket backup
```

### Performance Tuning
```bash
# Monitor API performance
curl http://localhost:8000/metrics

# Check database performance
docker exec -it mlops_postgres psql -U mlops_user -d mlops_db -c "
  SELECT query, mean_time, calls 
  FROM pg_stat_statements 
  ORDER BY mean_time DESC 
  LIMIT 10;"

# Optimize Redis memory
docker exec mlops_redis redis-cli INFO memory
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Services Not Starting
```bash
# Check Docker resources
docker system df
docker system prune

# Check port conflicts
netstat -tulpn | grep :5432
netstat -tulpn | grep :6379

# Restart services
docker-compose down
docker-compose up -d
```

#### 2. Feature Store Issues
```bash
# Reset feature store
rm -rf feature_store/.feast
cd feature_store && feast init . && cd ..

# Check Redis connection
docker exec mlops_redis redis-cli ping

# Verify PostgreSQL connection
docker exec mlops_postgres pg_isready -U mlops_user
```

#### 3. Model Training Failures
```bash
# Check MLflow logs
docker-compose logs mlflow

# Verify data quality
python -c "
import pandas as pd
data = pd.read_csv('data/customer_data.csv')
print(data.info())
print(data.isnull().sum())
"

# Check resource usage
docker stats mlops_model_api
```

#### 4. API Performance Issues
```bash
# Check API logs
docker-compose logs model_api

# Monitor response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/health

# Scale API instances
docker-compose up -d --scale model_api=3
```

### Performance Optimization

#### 1. Database Optimization
```sql
-- Connect to PostgreSQL
-- Create indexes for better performance
CREATE INDEX idx_customer_id ON customer_features(customer_id);
CREATE INDEX idx_created_at ON customer_features(created_at);

-- Analyze query performance
EXPLAIN ANALYZE SELECT * FROM customer_features WHERE customer_id = 'CUST_000001';
```

#### 2. Redis Optimization
```bash
# Configure Redis for better performance
docker exec mlops_redis redis-cli CONFIG SET maxmemory 2gb
docker exec mlops_redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

#### 3. Model Serving Optimization
```python
# Use model caching and batch prediction
# Implement connection pooling
# Use async endpoints for better concurrency
```

---

## 📚 Additional Resources

### Documentation
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Feast Documentation](https://docs.feast.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)

### Best Practices
- Model versioning and rollback strategies
- A/B testing implementation
- Monitoring and alerting setup
- Security and compliance considerations

### Scaling Considerations
- Horizontal scaling with Kubernetes
- Database sharding and replication
- Caching strategies
- Load balancing and failover

---

## 🎯 Success Criteria

### Technical Metrics
- [ ] All services healthy and accessible
- [ ] Model AUC > 0.85
- [ ] API latency < 100ms p95
- [ ] Feature serving < 10ms p95
- [ ] System uptime > 99.9%

### Functional Requirements
- [ ] Complete data pipeline operational
- [ ] Model training and deployment automated
- [ ] Real-time predictions working
- [ ] Monitoring and alerting active
- [ ] A/B testing framework ready

### Business Outcomes
- [ ] Churn prediction accuracy demonstrated
- [ ] Explainable predictions available
- [ ] Audit trail and compliance ready
- [ ] Scalable architecture deployed
- [ ] ROI metrics tracked

---

## 🎉 Congratulations!

You've successfully deployed a production-ready MLOps platform! This system demonstrates enterprise-level capabilities including:

- **Automated ML pipelines** with hyperparameter optimization
- **Production model serving** with A/B testing
- **Real-time monitoring** with drift detection
- **Comprehensive observability** and alerting
- **Scalable infrastructure** ready for enterprise deployment

**Next Steps:**
1. Customize for your specific use case
2. Integrate with existing systems
3. Scale to production workloads
4. Implement advanced features like online learning
5. Add business-specific monitoring and KPIs

**You're now ready for senior MLOps engineering roles!** 🚀