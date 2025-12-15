# Day 33: Model Serving at Scale - Setup Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (optional)
- Redis (optional - mock implementation provided)
- 4GB+ RAM recommended

### 1. Environment Setup
```bash
cd days/day-33-model-serving-scale

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Testing
```bash
# Run the exercise (guided implementation)
python exercise.py

# Run the complete solution
python solution.py

# Run load tests (requires running API)
locust -f load_test.py --host=http://localhost:8000
```

### 3. Docker Deployment (Optional)
```bash
# Build and run with Docker
docker build -t fraud-detection-api .
docker run -p 8000:8000 fraud-detection-api

# Or use Kubernetes
kubectl apply -f k8s-deployment.yaml
```

---

## 📋 Detailed Setup Instructions

### Local Development Setup

#### 1. Python Environment
```bash
# Check Python version
python --version  # Should be 3.9+

# Create isolated environment
python -m venv model-serving-env
source model-serving-env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt
```

#### 2. Redis Setup (Optional)
```bash
# Option 1: Docker Redis
docker run -d -p 6379:6379 redis:7-alpine

# Option 2: Local Redis installation
# macOS: brew install redis && brew services start redis
# Ubuntu: sudo apt install redis-server && sudo systemctl start redis

# Test Redis connection
redis-cli ping  # Should return PONG
```

#### 3. Verify Installation
```bash
# Test basic imports
python -c "
import fastapi
import redis
import numpy as np
import pandas as pd
print('✅ All dependencies installed successfully')
"
```

### Production Deployment

#### 1. Docker Deployment
```bash
# Build production image
docker build -t fraud-detection-api:v1.0 .

# Run with environment variables
docker run -d \
  -p 8000:8000 \
  -e REDIS_URL=redis://redis:6379 \
  -e LOG_LEVEL=INFO \
  --name fraud-api \
  fraud-detection-api:v1.0

# Check health
curl http://localhost:8000/health
```

#### 2. Kubernetes Deployment
```bash
# Apply all manifests
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -l app=fraud-detection-api
kubectl get services

# Port forward for testing
kubectl port-forward service/fraud-detection-service 8000:80

# Scale deployment
kubectl scale deployment fraud-detection-api --replicas=5
```

#### 3. Load Balancer Setup
```bash
# Install nginx (if not using Kubernetes ingress)
# Configure upstream servers in nginx.conf
# Point to multiple API instances for load balancing
```

---

## 🧪 Testing and Validation

### 1. Unit Testing
```bash
# Run basic functionality tests
python -c "
from exercise import FraudDetectionModel, PredictionCache
from solution import ModelServingAPI

# Test model
model = FraudDetectionModel()
features = {
    'amount': 1500,
    'merchant_category': 'online',
    'location': 'international',
    'user_age': 25,
    'timestamp': '2023-12-01T02:30:00'
}
prob = model.predict_proba(features)
pred = model.predict(features)
print(f'✅ Model test: probability={prob:.3f}, prediction={pred}')

# Test caching
cache = PredictionCache(None)
key = cache.get_cache_key(features)
cache.set(key, {'test': 'data'})
result = cache.get(key)
print(f'✅ Cache test: {result}')
"
```

### 2. API Testing
```bash
# Start the API server
python solution.py &
API_PID=$!

# Wait for startup
sleep 5

# Test health endpoint
curl -s http://localhost:8000/health | jq .

# Test prediction endpoint
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "test_001",
    "amount": 1500.0,
    "merchant_category": "online",
    "user_id": 12345,
    "location": "international",
    "user_age": 25
  }' | jq .

# Test metrics endpoint
curl -s http://localhost:8000/metrics | jq .

# Cleanup
kill $API_PID
```

### 3. Performance Testing
```bash
# Install locust if not already installed
pip install locust

# Run load test with web UI
locust -f load_test.py --host=http://localhost:8000

# Run headless load test
locust -f load_test.py --host=http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 60s --headless

# Run specific test scenarios
python -c "
from load_test import LoadTestScenarios
scenarios = LoadTestScenarios()
print('Available scenarios:')
print('Normal Load:', scenarios.normal_load())
print('Peak Load:', scenarios.peak_load())
print('Stress Test:', scenarios.stress_test())
"
```

### 4. Monitoring Setup
```bash
# Start Prometheus (if available)
# Download from https://prometheus.io/download/
# Configure prometheus.yml to scrape http://localhost:8000/metrics

# Start Grafana (if available)
# Import dashboard for ML model serving metrics
# Configure data source: http://localhost:9090

# View metrics in browser
open http://localhost:8000/metrics  # Raw Prometheus metrics
open http://localhost:9090          # Prometheus UI
open http://localhost:3000          # Grafana dashboards
```

---

## 🔧 Configuration Options

### Environment Variables
```bash
# API Configuration
export API_HOST=0.0.0.0
export API_PORT=8000
export LOG_LEVEL=INFO

# Redis Configuration
export REDIS_URL=redis://localhost:6379
export CACHE_TTL=300

# Model Configuration
export MODEL_VERSION=1.0.0
export PREDICTION_THRESHOLD=0.5

# Performance Configuration
export MAX_WORKERS=4
export REQUEST_TIMEOUT=30
export CACHE_SIZE=10000
```

### Configuration Files
```yaml
# config.yaml (optional)
api:
  host: "0.0.0.0"
  port: 8000
  workers: 4

redis:
  url: "redis://localhost:6379"
  ttl: 300

model:
  version: "1.0.0"
  threshold: 0.5

monitoring:
  enable_metrics: true
  metrics_port: 8001
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Problem: ModuleNotFoundError
# Solution: Check virtual environment and requirements
pip list | grep fastapi
pip install -r requirements.txt

# Problem: Redis connection error
# Solution: Check Redis is running
redis-cli ping
docker ps | grep redis
```

#### 2. Performance Issues
```bash
# Problem: High latency
# Check: Model complexity, caching, database connections
# Solution: Enable caching, optimize model, use connection pooling

# Problem: Low throughput
# Check: Number of workers, async configuration
# Solution: Increase workers, use async/await properly
```

#### 3. Memory Issues
```bash
# Problem: High memory usage
# Check: Cache size, model size, memory leaks
# Solution: Limit cache size, optimize model, monitor memory

# Monitor memory usage
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
print(f'Available memory: {psutil.virtual_memory().available / 1024**3:.1f} GB')
"
```

#### 4. Docker Issues
```bash
# Problem: Container won't start
# Check logs
docker logs fraud-detection-api

# Problem: Port conflicts
# Check what's using the port
lsof -i :8000
netstat -tulpn | grep :8000

# Problem: Permission errors
# Check user permissions in Dockerfile
docker run --user $(id -u):$(id -g) fraud-detection-api
```

### Performance Optimization

#### 1. API Optimization
```python
# Use async/await for I/O operations
# Enable connection pooling
# Implement proper caching
# Use compression for responses
# Optimize JSON serialization
```

#### 2. Model Optimization
```python
# Model quantization
# ONNX conversion
# Feature preprocessing optimization
# Batch prediction for multiple requests
```

#### 3. Infrastructure Optimization
```bash
# Use multiple workers
# Enable HTTP/2
# Use CDN for static content
# Implement proper load balancing
# Use SSD storage for cache
```

---

## 📊 Performance Benchmarks

### Target Performance Metrics
- **Latency**: <50ms p95 response time
- **Throughput**: >1000 requests per second
- **Availability**: >99.9% uptime
- **Cache Hit Rate**: >80%
- **Error Rate**: <1%

### Benchmark Results (Example)
```
Load Test Results:
├── Normal Load (100 users)
│   ├── Avg Response Time: 25ms
│   ├── P95 Response Time: 45ms
│   ├── Throughput: 850 RPS
│   └── Success Rate: 99.8%
├── Peak Load (500 users)
│   ├── Avg Response Time: 35ms
│   ├── P95 Response Time: 65ms
│   ├── Throughput: 1200 RPS
│   └── Success Rate: 99.5%
└── Stress Test (1000 users)
    ├── Avg Response Time: 55ms
    ├── P95 Response Time: 120ms
    ├── Throughput: 1500 RPS
    └── Success Rate: 98.2%
```

---

## 📚 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Redis Documentation](https://redis.io/documentation)
- [Locust Documentation](https://docs.locust.io/)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Monitoring Tools
- [Prometheus](https://prometheus.io/)
- [Grafana](https://grafana.com/)
- [Jaeger Tracing](https://www.jaegertracing.io/)

### Performance Tools
- [Locust Load Testing](https://locust.io/)
- [Apache Bench](https://httpd.apache.org/docs/2.4/programs/ab.html)
- [wrk HTTP Benchmarking](https://github.com/wg/wrk)

---

## ✅ Verification Checklist

After setup, verify:
- [ ] Python environment activated
- [ ] All dependencies installed
- [ ] Redis connection working (if used)
- [ ] API starts without errors
- [ ] Health endpoint responds
- [ ] Prediction endpoint works
- [ ] Metrics endpoint accessible
- [ ] Load test runs successfully
- [ ] Docker build completes (if used)
- [ ] Kubernetes deployment works (if used)

**Setup Complete! 🎉**

You're ready to explore model serving at scale. Start with the exercise for guided learning, then examine the complete solution for production patterns.