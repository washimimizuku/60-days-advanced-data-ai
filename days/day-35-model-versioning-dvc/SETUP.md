# Day 35: Model Versioning with DVC - Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Git installed
- 8GB+ RAM recommended
- 10GB+ free disk space

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installations
dvc version
mlflow --version
```

### 2. Project Initialization

```bash
# Initialize Git repository (if not already done)
git init

# Initialize DVC
dvc init

# Set up MLflow tracking
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 &
```

### 3. Run the Exercise

```bash
# Run the complete solution
python solution.py

# Or run individual components
python exercise.py
```

## 🔧 Configuration

### DVC Remote Storage (Optional)

```bash
# Configure S3 remote (replace with your bucket)
dvc remote add -d myremote s3://your-bucket/dvc-cache
dvc remote modify myremote access_key_id YOUR_ACCESS_KEY
dvc remote modify myremote secret_access_key YOUR_SECRET_KEY

# Or use local remote for testing
dvc remote add -d local /tmp/dvc-cache
```

### MLflow Configuration

```bash
# Environment variables
export MLFLOW_TRACKING_URI=sqlite:///mlflow.db
export MLFLOW_DEFAULT_ARTIFACT_ROOT=./mlruns

# For production, use PostgreSQL
export MLFLOW_TRACKING_URI=postgresql://user:password@localhost/mlflow
```

## 🐳 Docker Setup

### Build and Run

```bash
# Build Docker image
docker build -t healthtech-ml:latest .

# Run container
docker run -p 8000:8000 -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  healthtech-ml:latest
```

### Docker Compose (Recommended)

```yaml
# docker-compose.yml
version: '3.8'
services:
  ml-pipeline:
    build: .
    ports:
      - "8000:8000"
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - MLFLOW_TRACKING_URI=postgresql://mlflow:password@postgres:5432/mlflow
    depends_on:
      - postgres
  
  postgres:
    image: postgres:13
    environment:
      POSTGRES_DB: mlflow
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

volumes:
  postgres_data:
```

```bash
# Run with Docker Compose
docker-compose up -d
```

## ☸️ Kubernetes Deployment

### Prerequisites
- Kubernetes cluster (minikube, kind, or cloud)
- kubectl configured

### Deploy

```bash
# Apply Kubernetes manifests
kubectl apply -f k8s-deployment.yaml

# Check deployment status
kubectl get pods -l app=healthtech-ml

# Access services
kubectl port-forward service/healthtech-ml-service 8000:80
```

### Create Secrets

```bash
# Create AWS credentials secret
kubectl create secret generic aws-credentials \
  --from-literal=access-key-id=YOUR_ACCESS_KEY \
  --from-literal=secret-access-key=YOUR_SECRET_KEY
```

## 🧪 Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest test_model_versioning.py -v

# Run with coverage
pytest test_model_versioning.py --cov=solution --cov-report=html

# Run specific test categories
pytest test_model_versioning.py::TestDVCPipelineManager -v
pytest test_model_versioning.py::TestMLflowModelManager -v
```

### Performance Testing

```bash
# Run performance tests
pytest test_model_versioning.py::TestPerformance -v

# Load testing (if you have locust installed)
pip install locust
locust -f load_test.py --host=http://localhost:8000
```

## 📊 Monitoring and Observability

### MLflow UI

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000

# Access at http://localhost:5000
```

### DVC Pipeline Visualization

```bash
# Visualize pipeline
dvc dag

# Check pipeline status
dvc status

# Show metrics
dvc metrics show
```

### Health Checks

```bash
# API health check
curl http://localhost:8000/health

# MLflow health check
curl http://localhost:5000/health
```

## 🔍 Troubleshooting

### Common Issues

#### 1. DVC Remote Access Issues
```bash
# Check DVC configuration
dvc config -l

# Test remote access
dvc push --dry-run

# Fix permissions
dvc remote modify myremote access_key_id YOUR_NEW_KEY
```

#### 2. MLflow Database Issues
```bash
# Reset MLflow database
rm mlflow.db
mlflow db upgrade sqlite:///mlflow.db

# Check database connection
python -c "import mlflow; print(mlflow.get_tracking_uri())"
```

#### 3. Memory Issues
```bash
# Monitor memory usage
docker stats

# Increase Docker memory limit
# Docker Desktop -> Settings -> Resources -> Memory
```

#### 4. Port Conflicts
```bash
# Check port usage
lsof -i :5000
lsof -i :8000

# Kill processes using ports
kill -9 $(lsof -t -i:5000)
```

### Performance Optimization

#### 1. DVC Cache Optimization
```bash
# Set cache directory
dvc config cache.dir /path/to/fast/storage

# Enable cache compression
dvc config cache.type symlink
```

#### 2. MLflow Performance
```bash
# Use PostgreSQL for better performance
export MLFLOW_TRACKING_URI=postgresql://user:pass@host/db

# Enable artifact caching
export MLFLOW_ARTIFACT_CACHE_SIZE=1000
```

#### 3. Model Serving Optimization
```bash
# Use model caching
export MODEL_CACHE_SIZE=5

# Enable batch prediction
export BATCH_SIZE=32
```

## 📚 Additional Resources

### Documentation
- [DVC Documentation](https://dvc.org/doc)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Kubernetes Documentation](https://kubernetes.io/docs/)

### Best Practices
- Always version your data and models
- Use meaningful experiment names and tags
- Implement proper error handling and logging
- Monitor model performance continuously
- Maintain clean pipeline dependencies

### Production Considerations
- Use external databases for MLflow tracking
- Implement proper authentication and authorization
- Set up automated backups for models and data
- Use container orchestration for scalability
- Implement proper logging and monitoring

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs: `docker logs <container_id>`
3. Check DVC status: `dvc status`
4. Verify MLflow connection: `mlflow experiments list`
5. Test individual components with the test suite

For additional support, refer to the official documentation or community forums for DVC and MLflow.