# Day 32: ML Platform Setup Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Docker and Docker Compose installed
- Python 3.9+ (for local development)
- 8GB+ RAM recommended
- 10GB+ free disk space

### 1. Clone and Setup
```bash
cd days/day-32-project-ml-model
cp .env.example .env
```

### 2. Start Infrastructure
```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps
```

### 3. Verify Setup
```bash
# Check API health
curl http://localhost:8000/health

# Check Grafana (admin/admin)
open http://localhost:3000

# Check MLflow
open http://localhost:5000
```

---

## 📋 Detailed Setup Instructions

### Environment Configuration

Create `.env` file:
```bash
# Database
DATABASE_URL=postgresql://ml_user:ml_pass@localhost:5432/ml_platform_db
POSTGRES_DB=ml_platform_db
POSTGRES_USER=ml_user
POSTGRES_PASSWORD=ml_pass

# Redis
REDIS_URL=redis://localhost:6379

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# Feast
FEAST_SERVER_URL=http://localhost:6566

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1
DEBUG=false
LOG_LEVEL=INFO

# Security (generate your own keys)
JWT_SECRET_KEY=your-secret-key-here
API_KEY=your-api-key-here
```

### Service Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Load Balancer │    │   ML Platform   │    │  Feature Store  │
│   (Nginx:80)    │───▶│   API (:8000)   │───▶│  (Feast :6566)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Model Store   │    │   Data Storage  │
│ (Grafana :3000) │    │ (MLflow :5000)  │    │ (Postgres:5432) │
│ (Prometheus     │    │                 │    │ (Redis :6379)   │
│  :9090)         │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Step-by-Step Deployment

#### 1. Infrastructure Services
```bash
# Start database and cache
docker-compose up -d postgres redis

# Wait for services to be ready
docker-compose logs postgres
docker-compose logs redis
```

#### 2. Feature Store
```bash
# Start Feast feature store
docker-compose up -d feast-server

# Verify Feast is running
curl http://localhost:6566/health
```

#### 3. Model Management
```bash
# Start MLflow
docker-compose up -d mlflow

# Access MLflow UI
open http://localhost:5000
```

#### 4. ML Platform API
```bash
# Start main API service
docker-compose up -d ml-api

# Check API health
curl http://localhost:8000/health
```

#### 5. Monitoring Stack
```bash
# Start monitoring services
docker-compose up -d prometheus grafana

# Access Grafana (admin/admin)
open http://localhost:3000
```

#### 6. Load Balancer
```bash
# Start Nginx load balancer
docker-compose up -d nginx

# Test through load balancer
curl http://localhost/health
```

---

## 🧪 Testing the Platform

### 1. Health Checks
```bash
# API Health
curl http://localhost:8000/health

# Feature Store Health
curl http://localhost:6566/health

# Database Connection
docker-compose exec postgres pg_isready -U ml_user -d ml_platform_db
```

### 2. API Testing
```bash
# Credit Risk Prediction
curl -X POST http://localhost:8000/predict/credit-risk \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "income": 75000,
    "credit_score": 720,
    "credit_history_length": 8,
    "loan_amount": 25000,
    "loan_purpose": "home",
    "employment_length": 5,
    "debt_to_income_ratio": 0.25,
    "explain": true
  }'

# Fraud Detection
curl -X POST http://localhost:8000/predict/fraud-detection \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1500,
    "timestamp": "2023-12-01T02:30:00",
    "merchant_category": "online",
    "location_type": "international",
    "user_age": 25,
    "user_income": 45000,
    "explain": true
  }'
```

### 3. Performance Testing
```bash
# Install testing tools
pip install locust

# Run load test
locust -f load_test.py --host=http://localhost:8000
```

---

## 🔧 Development Setup

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DATABASE_URL=postgresql://ml_user:ml_pass@localhost:5432/ml_platform_db
export REDIS_URL=redis://localhost:6379

# Run API locally
python api.py
```

### Model Training
```bash
# Train models locally
python solution.py

# Or run training service
docker-compose up model-trainer
```

### Data Generation
```bash
# Generate synthetic data
python data_generator.py

# Or run data generation service
docker-compose up data-generator
```

---

## 📊 Monitoring and Observability

### Grafana Dashboards
1. **ML Platform Overview**: System health and performance
2. **Model Performance**: Prediction metrics and accuracy
3. **API Metrics**: Request rates, latency, errors
4. **Infrastructure**: Database, Redis, system resources

### Key Metrics to Monitor
- **API Latency**: p50, p95, p99 response times
- **Throughput**: Requests per second
- **Error Rates**: 4xx and 5xx responses
- **Model Performance**: Accuracy, drift detection
- **Resource Usage**: CPU, memory, disk

### Alerts Configuration
```yaml
# Example alert rules
groups:
  - name: ml_platform_alerts
    rules:
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, ml_platform_prediction_duration_seconds) > 0.5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High API latency detected"
      
      - alert: HighErrorRate
        expr: rate(ml_platform_api_requests_total{status=~"5.."}[5m]) > 0.1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Services Not Starting
```bash
# Check logs
docker-compose logs [service-name]

# Check resource usage
docker stats

# Restart specific service
docker-compose restart [service-name]
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

#### 3. Memory Issues
```bash
# Check memory usage
docker stats

# Reduce worker processes
# Edit docker-compose.yml: API_WORKERS=1
```

#### 4. Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8000

# Change ports in docker-compose.yml
```

### Performance Optimization

#### 1. API Performance
- Increase worker processes: `API_WORKERS=4`
- Enable connection pooling
- Add Redis caching for predictions

#### 2. Database Performance
- Add database indexes
- Optimize queries
- Use connection pooling

#### 3. Model Performance
- Use model quantization
- Implement model caching
- Batch predictions when possible

---

## 🔒 Security Considerations

### Production Security Checklist
- [ ] Change default passwords
- [ ] Enable SSL/TLS certificates
- [ ] Implement API authentication
- [ ] Set up network security groups
- [ ] Enable audit logging
- [ ] Regular security updates
- [ ] Backup and disaster recovery

### Environment Variables
```bash
# Production environment
DEBUG=false
LOG_LEVEL=WARNING
JWT_SECRET_KEY=<strong-random-key>
API_KEY=<secure-api-key>
```

---

## 📚 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Feast Documentation](https://docs.feast.dev/)
- [MLflow Documentation](https://mlflow.org/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)

### Monitoring
- [Grafana Dashboards](http://localhost:3000)
- [Prometheus Metrics](http://localhost:9090)
- [MLflow Experiments](http://localhost:5000)

### Support
- Check logs: `docker-compose logs`
- Health endpoints: `/health`
- Metrics endpoint: `/metrics`

---

## ✅ Verification Checklist

After setup, verify:
- [ ] All services are running (`docker-compose ps`)
- [ ] API responds to health checks
- [ ] Database is accessible
- [ ] Feature store is operational
- [ ] Monitoring dashboards are available
- [ ] Sample predictions work
- [ ] Explanations are generated
- [ ] Performance metrics are collected

**Setup Complete! 🎉**

Your ML platform is ready for production use. Proceed to run the complete solution or start with the exercise for hands-on learning.