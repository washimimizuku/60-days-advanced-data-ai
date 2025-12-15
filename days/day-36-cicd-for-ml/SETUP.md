# Day 36: CI/CD for ML - Setup Guide

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker and Docker Compose
- AWS CLI configured (optional)
- Terraform installed (optional)
- Git repository access

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installations
python -c "import sklearn, pandas, boto3; print('Dependencies installed successfully')"
```

### 2. Local Development Setup

```bash
# Start local services with Docker Compose
docker-compose up -d

# Verify services are running
docker-compose ps

# Check MLflow UI
open http://localhost:5000

# Check Grafana dashboard
open http://localhost:3000  # admin/admin
```

### 3. Run the Exercise

```bash
# Run the complete solution
python solution.py

# Or run individual components
python exercise.py
```

## 🔧 Configuration

### GitHub Actions Setup

1. **Create GitHub Repository**
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/your-username/ml-cicd-pipeline.git
git push -u origin main
```

2. **Setup GitHub Actions Workflow**
```bash
# Create workflow directory
mkdir -p .github/workflows

# Copy workflow file
cp github-workflow-ml-pipeline.yml .github/workflows/ml_pipeline.yml

# Commit and push
git add .github/workflows/ml_pipeline.yml
git commit -m "Add CI/CD workflow"
git push
```

3. **Configure GitHub Secrets**
```bash
# Add these secrets in GitHub repository settings:
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
MLFLOW_TRACKING_URI=your_mlflow_uri
```

### AWS Infrastructure Setup

1. **Configure AWS CLI**
```bash
aws configure
# Enter your AWS credentials and region
```

2. **Setup Terraform Backend**
```bash
# Create S3 bucket for Terraform state
aws s3 mb s3://ml-terraform-state-your-suffix

# Create DynamoDB table for state locking (optional)
aws dynamodb create-table \
    --table-name terraform-state-lock \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --provisioned-throughput ReadCapacityUnits=5,WriteCapacityUnits=5
```

3. **Deploy Infrastructure**
```bash
# Create infrastructure directory
mkdir infrastructure
cp terraform-main.tf infrastructure/main.tf

# Initialize and apply Terraform
cd infrastructure
terraform init
terraform plan -var="environment=staging"
terraform apply -var="environment=staging"
```

## 🐳 Docker Setup

### Build and Run Locally

```bash
# Build Docker image
docker build -t ml-cicd-pipeline:latest .

# Run container
docker run -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  ml-cicd-pipeline:latest
```

### Docker Compose Services

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f ml-pipeline

# Stop services
docker-compose down
```

## 🧪 Testing

### Run Tests Locally

```bash
# Install test dependencies
pip install pytest pytest-cov pytest-mock

# Run all tests
pytest test_ml_pipeline.py -v

# Run with coverage
pytest test_ml_pipeline.py --cov=solution --cov-report=html

# Run specific test categories
pytest test_ml_pipeline.py::TestDataValidation -v
pytest test_ml_pipeline.py::TestBlueGreenDeployment -v
```

### Performance Testing

```bash
# Run performance tests
pytest test_ml_pipeline.py::TestPerformance -v

# Load testing (if you have locust installed)
pip install locust
locust -f load_test.py --host=http://localhost:8000
```

## 📊 Monitoring Setup

### MLflow Tracking

```bash
# Start MLflow server locally
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 0.0.0.0 \
  --port 5000

# Access MLflow UI
open http://localhost:5000
```

### Prometheus and Grafana

```bash
# Prometheus configuration
mkdir -p monitoring
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'ml-pipeline'
    static_configs:
      - targets: ['ml-pipeline:8000']
EOF

# Start monitoring stack
docker-compose up prometheus grafana -d

# Access Grafana
open http://localhost:3000  # admin/admin
```

### CloudWatch Setup (AWS)

```bash
# Create CloudWatch dashboard
python -c "
from solution import MLCICDPipeline
pipeline = MLCICDPipeline()
result = pipeline.create_monitoring_dashboard('fraud-detection-pipeline')
print(f'Dashboard created: {result}')
"
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Docker Build Issues
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t ml-cicd-pipeline:latest .
```

#### 2. MLflow Connection Issues
```bash
# Check MLflow server status
curl http://localhost:5000/health

# Reset MLflow database
rm mlflow.db
mlflow db upgrade sqlite:///mlflow.db
```

#### 3. AWS Permissions Issues
```bash
# Check AWS credentials
aws sts get-caller-identity

# Test S3 access
aws s3 ls

# Verify ECS permissions
aws ecs list-clusters
```

#### 4. GitHub Actions Failures
```bash
# Check workflow syntax
yamllint .github/workflows/ml_pipeline.yml

# Test workflow locally with act
act -j data-validation
```

### Performance Optimization

#### 1. Docker Image Optimization
```bash
# Use multi-stage builds
# Add .dockerignore file
echo "*.pyc
__pycache__
.git
.pytest_cache
*.log" > .dockerignore
```

#### 2. Pipeline Optimization
```bash
# Enable caching in GitHub Actions
# Use matrix builds for parallel execution
# Optimize data loading and processing
```

#### 3. AWS Cost Optimization
```bash
# Use spot instances for training
# Implement auto-scaling
# Set up cost alerts
```

## 📚 Additional Resources

### Documentation
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Docker Documentation](https://docs.docker.com/)

### Best Practices
- Always use version pinning in requirements.txt
- Implement proper error handling and logging
- Use secrets management for sensitive data
- Monitor resource usage and costs
- Implement proper testing at all levels
- Use infrastructure as code for reproducibility

### Production Considerations
- Set up proper authentication and authorization
- Implement comprehensive monitoring and alerting
- Use blue-green or canary deployment strategies
- Maintain disaster recovery procedures
- Regular security audits and updates
- Compliance with data protection regulations

## 🆘 Getting Help

If you encounter issues:

1. Check the troubleshooting section above
2. Review logs: `docker-compose logs -f`
3. Check GitHub Actions workflow runs
4. Verify AWS permissions and quotas
5. Test individual components separately

For additional support, refer to the official documentation or community forums for the respective tools.