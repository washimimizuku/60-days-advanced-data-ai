# Day 36: CI/CD for ML - Automated ML Pipelines, Testing, Infrastructure as Code

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Design automated ML pipelines** with continuous integration and deployment
- **Implement comprehensive testing strategies** for ML models and data pipelines
- **Build infrastructure as code** for scalable ML systems deployment
- **Create monitoring and alerting** for production ML workflows
- **Handle model deployment rollbacks** and canary releases safely

**Estimated Time**: 1 hour  
**Difficulty Level**: Advanced ⭐⭐⭐⭐

---

## 🎯 What is CI/CD for ML?

CI/CD for Machine Learning extends traditional software development practices to handle the unique challenges of ML systems, including data versioning, model training, validation, and deployment automation.

### Key Components of ML CI/CD

**1. Continuous Integration for ML**
- Automated data validation and quality checks
- Model training pipeline automation
- Comprehensive testing of ML code and models
- Integration testing with data dependencies

**2. Continuous Deployment for ML**
- Automated model deployment with validation gates
- Blue-green and canary deployment strategies
- Rollback mechanisms for failed deployments
- Infrastructure provisioning and scaling

**3. MLOps Pipeline Orchestration**
- End-to-end workflow automation
- Dependency management and artifact tracking
- Environment consistency across dev/staging/prod
- Monitoring and observability integration

**4. Infrastructure as Code**
- Reproducible environment provisioning
- Scalable compute resource management
- Security and compliance automation
- Cost optimization and resource governance

---

## 🔧 GitHub Actions for ML Workflows

### 1. **ML Pipeline Automation Architecture**

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline CI/CD

on:
  push:
    branches: [ main, develop ]
    paths:
      - 'src/**'
      - 'data/**'
      - 'models/**'
      - 'requirements.txt'
  pull_request:
    branches: [ main ]
  schedule:
    # Run daily at 2 AM UTC for data drift monitoring
    - cron: '0 2 * * *'

env:
  PYTHON_VERSION: '3.9'
  AWS_REGION: 'us-west-2'
  MODEL_REGISTRY: 'ml-model-registry'
  
jobs:
  data-validation:
    runs-on: ubuntu-latest
    outputs:
      data-changed: ${{ steps.check-data.outputs.changed }}
      
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      with:
        fetch-depth: 2
        
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        
    - name: Install dependencies
      run: |
        pip install --upgrade pip
        pip install -r requirements.txt
        pip install great-expectations pandas-profiling
        
    - name: Check data changes
      id: check-data
      run: |
        if git diff --name-only HEAD~1 | grep -q "data/"; then
          echo "changed=true" >> $GITHUB_OUTPUT
        else
          echo "changed=false" >> $GITHUB_OUTPUT
        fi
        
    - name: Validate data schema
      run: |
        python scripts/validate_data_schema.py
        
    - name: Run data quality checks
      run: |
        python scripts/data_quality_checks.py
        
    - name: Generate data profile
      if: steps.check-data.outputs.changed == 'true'
      run: |
        python scripts/generate_data_profile.py
        
    - name: Upload data validation artifacts
      uses: actions/upload-artifact@v3
      with:
        name: data-validation-results
        path: |
          reports/data_validation/
          reports/data_profile/
```
  model-training:
    needs: data-validation
    runs-on: ubuntu-latest
    if: needs.data-validation.outputs.data-changed == 'true' || github.event_name == 'schedule'
    
    strategy:
      matrix:
        model-type: [random_forest, xgboost, neural_network]
        
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
        
    - name: Set up DVC
      uses: iterative/setup-dvc@v1
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install mlflow boto3 sagemaker
        
    - name: Pull data and models
      run: |
        dvc pull
        
    - name: Train model
      env:
        MODEL_TYPE: ${{ matrix.model-type }}
        MLFLOW_TRACKING_URI: ${{ secrets.MLFLOW_TRACKING_URI }}
        MLFLOW_S3_ENDPOINT_URL: ${{ secrets.MLFLOW_S3_ENDPOINT_URL }}
      run: |
        python src/train_model.py --model-type $MODEL_TYPE --experiment-name "ci-cd-${{ github.run_id }}"
        
    - name: Validate model performance
      run: |
        python scripts/validate_model_performance.py --model-type ${{ matrix.model-type }}
        
    - name: Push artifacts
      run: |
        dvc push
        
    - name: Upload training artifacts
      uses: actions/upload-artifact@v3
      with:
        name: model-artifacts-${{ matrix.model-type }}
        path: |
          models/
          metrics/
          reports/

  model-testing:
    needs: [data-validation, model-training]
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
        
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov locust
        
    - name: Download model artifacts
      uses: actions/download-artifact@v3
      with:
        path: artifacts/
        
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
        
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
        
    - name: Run model validation tests
      run: |
        pytest tests/model/ -v
        
    - name: Run performance tests
      run: |
        python tests/performance/load_test.py
        
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results
        path: |
          coverage.xml
          test-results/
          performance-results/

  security-scan:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Run security scan
      uses: pypa/gh-action-pip-audit@v1.0.8
      with:
        inputs: requirements.txt
        
    - name: Run Bandit security scan
      run: |
        pip install bandit
        bandit -r src/ -f json -o security-report.json
        
    - name: Upload security results
      uses: actions/upload-artifact@v3
      with:
        name: security-scan-results
        path: security-report.json

  deploy-staging:
    needs: [model-testing, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
        
    - name: Deploy to staging
      run: |
        python scripts/deploy_model.py --environment staging --approval-required false
        
    - name: Run smoke tests
      run: |
        python tests/smoke/staging_smoke_tests.py
        
    - name: Run integration tests against staging
      run: |
        python tests/integration/staging_integration_tests.py

  deploy-production:
    needs: [model-testing, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    
    steps:
    - name: Checkout code
      uses: actions/checkout@v3
      
    - name: Configure AWS credentials
      uses: aws-actions/configure-aws-credentials@v2
      with:
        aws-access-key-id: ${{ secrets.AWS_ACCESS_KEY_ID }}
        aws-secret-access-key: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        aws-region: ${{ env.AWS_REGION }}
        
    - name: Deploy to production (canary)
      run: |
        python scripts/deploy_model.py --environment production --strategy canary --traffic-percentage 10
        
    - name: Monitor canary deployment
      run: |
        python scripts/monitor_canary.py --duration 300 --success-threshold 0.95
        
    - name: Promote to full production
      run: |
        python scripts/deploy_model.py --environment production --strategy promote --traffic-percentage 100
```
```

### 2. **Infrastructure as Code with Terraform**

```hcl
# infrastructure/main.tf
terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "ml-terraform-state"
    key    = "ml-infrastructure/terraform.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = var.aws_region
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "ml-pipeline"
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

# VPC and Networking
resource "aws_vpc" "ml_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-vpc"
    Environment = var.environment
  }
}

resource "aws_subnet" "ml_subnet" {
  count = 2
  
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-subnet-${count.index + 1}"
    Environment = var.environment
  }
}

# Internet Gateway
resource "aws_internet_gateway" "ml_igw" {
  vpc_id = aws_vpc.ml_vpc.id
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-igw"
    Environment = var.environment
  }
}

# Route Table
resource "aws_route_table" "ml_rt" {
  vpc_id = aws_vpc.ml_vpc.id
  
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.ml_igw.id
  }
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-rt"
    Environment = var.environment
  }
}

resource "aws_route_table_association" "ml_rta" {
  count = length(aws_subnet.ml_subnet)
  
  subnet_id      = aws_subnet.ml_subnet[count.index].id
  route_table_id = aws_route_table.ml_rt.id
}

# Security Groups
resource "aws_security_group" "ml_sg" {
  name_prefix = "${var.project_name}-${var.environment}-"
  vpc_id      = aws_vpc.ml_vpc.id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-sg"
    Environment = var.environment
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "ml_cluster" {
  name = "${var.project_name}-${var.environment}-cluster"
  
  setting {
    name  = "containerInsights"
    value = "enabled"
  }
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-cluster"
    Environment = var.environment
  }
}

# Application Load Balancer
resource "aws_lb" "ml_alb" {
  name               = "${var.project_name}-${var.environment}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.ml_sg.id]
  subnets            = aws_subnet.ml_subnet[*].id
  
  enable_deletion_protection = false
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-alb"
    Environment = var.environment
  }
}

# S3 Buckets
resource "aws_s3_bucket" "ml_artifacts" {
  bucket = "${var.project_name}-${var.environment}-artifacts-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-artifacts"
    Environment = var.environment
  }
}

resource "aws_s3_bucket" "ml_data" {
  bucket = "${var.project_name}-${var.environment}-data-${random_id.bucket_suffix.hex}"
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-data"
    Environment = var.environment
  }
}

resource "random_id" "bucket_suffix" {
  byte_length = 4
}

# S3 Bucket Versioning
resource "aws_s3_bucket_versioning" "ml_artifacts_versioning" {
  bucket = aws_s3_bucket.ml_artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "ml_data_versioning" {
  bucket = aws_s3_bucket.ml_data.id
  versioning_configuration {
    status = "Enabled"
  }
}

# ECR Repository
resource "aws_ecr_repository" "ml_model_repo" {
  name                 = "${var.project_name}-${var.environment}-models"
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-models"
    Environment = var.environment
  }
}

# RDS for MLflow
resource "aws_db_subnet_group" "ml_db_subnet_group" {
  name       = "${var.project_name}-${var.environment}-db-subnet-group"
  subnet_ids = aws_subnet.ml_subnet[*].id
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-db-subnet-group"
    Environment = var.environment
  }
}

resource "aws_db_instance" "mlflow_db" {
  identifier = "${var.project_name}-${var.environment}-mlflow-db"
  
  engine         = "postgres"
  engine_version = "14.9"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  storage_type          = "gp2"
  storage_encrypted     = true
  
  db_name  = "mlflow"
  username = "mlflow"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.ml_sg.id]
  db_subnet_group_name   = aws_db_subnet_group.ml_db_subnet_group.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = true
  deletion_protection = false
  
  tags = {
    Name        = "${var.project_name}-${var.environment}-mlflow-db"
    Environment = var.environment
  }
}

resource "random_password" "db_password" {
  length  = 16
  special = true
}

# Outputs
output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.ml_vpc.id
}

output "subnet_ids" {
  description = "Subnet IDs"
  value       = aws_subnet.ml_subnet[*].id
}

output "security_group_id" {
  description = "Security Group ID"
  value       = aws_security_group.ml_sg.id
}

output "ecs_cluster_name" {
  description = "ECS Cluster Name"
  value       = aws_ecs_cluster.ml_cluster.name
}

output "alb_dns_name" {
  description = "ALB DNS Name"
  value       = aws_lb.ml_alb.dns_name
}

output "s3_artifacts_bucket" {
  description = "S3 Artifacts Bucket Name"
  value       = aws_s3_bucket.ml_artifacts.bucket
}

output "s3_data_bucket" {
  description = "S3 Data Bucket Name"
  value       = aws_s3_bucket.ml_data.bucket
}

output "ecr_repository_url" {
  description = "ECR Repository URL"
  value       = aws_ecr_repository.ml_model_repo.repository_url
}

output "mlflow_db_endpoint" {
  description = "MLflow Database Endpoint"
  value       = aws_db_instance.mlflow_db.endpoint
  sensitive   = true
}
```

---

## 🧪 Comprehensive Testing Strategies

### 1. **ML Model Testing Framework**

```python
import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import joblib
import json
from pathlib import Path

class MLModelTestSuite:
    """Comprehensive test suite for ML models"""
    
    def __init__(self, model_path, test_data_path, config_path):
        self.model_path = Path(model_path)
        self.test_data_path = Path(test_data_path)
        self.config_path = Path(config_path)
        
        # Load model and configuration
        self.model = joblib.load(self.model_path)
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Load test data
        self.test_data = pd.read_csv(self.test_data_path)
        self.X_test = self.test_data.drop(self.config['target_column'], axis=1)
        self.y_test = self.test_data[self.config['target_column']]
    
    def test_model_loading(self):
        """Test that model loads correctly"""
        assert self.model is not None, "Model failed to load"
        assert hasattr(self.model, 'predict'), "Model missing predict method"
        assert hasattr(self.model, 'predict_proba'), "Model missing predict_proba method"
    
    def test_prediction_shape(self):
        """Test prediction output shapes"""
        predictions = self.model.predict(self.X_test)
        probabilities = self.model.predict_proba(self.X_test)
        
        assert len(predictions) == len(self.X_test), "Prediction length mismatch"
        assert probabilities.shape[0] == len(self.X_test), "Probability shape mismatch"
        assert probabilities.shape[1] == len(np.unique(self.y_test)), "Probability classes mismatch"
    
    def test_prediction_types(self):
        """Test prediction data types"""
        predictions = self.model.predict(self.X_test)
        probabilities = self.model.predict_proba(self.X_test)
        
        assert isinstance(predictions, np.ndarray), "Predictions not numpy array"
        assert isinstance(probabilities, np.ndarray), "Probabilities not numpy array"
        assert probabilities.dtype == np.float64, "Probabilities not float type"
    
    def test_prediction_ranges(self):
        """Test prediction value ranges"""
        probabilities = self.model.predict_proba(self.X_test)
        
        # Probabilities should be between 0 and 1
        assert np.all(probabilities >= 0), "Negative probabilities found"
        assert np.all(probabilities <= 1), "Probabilities > 1 found"
        
        # Probabilities should sum to 1 for each sample
        prob_sums = np.sum(probabilities, axis=1)
        assert np.allclose(prob_sums, 1.0, rtol=1e-5), "Probabilities don't sum to 1"
    
    def test_model_performance(self):
        """Test model meets minimum performance requirements"""
        predictions = self.model.predict(self.X_test)
        
        accuracy = accuracy_score(self.y_test, predictions)
        precision = precision_score(self.y_test, predictions, average='weighted')
        recall = recall_score(self.y_test, predictions, average='weighted')
        
        min_accuracy = self.config.get('min_accuracy', 0.7)
        min_precision = self.config.get('min_precision', 0.7)
        min_recall = self.config.get('min_recall', 0.7)
        
        assert accuracy >= min_accuracy, f"Accuracy {accuracy:.3f} below threshold {min_accuracy}"
        assert precision >= min_precision, f"Precision {precision:.3f} below threshold {min_precision}"
        assert recall >= min_recall, f"Recall {recall:.3f} below threshold {min_recall}"
    
    def test_model_stability(self):
        """Test model prediction stability"""
        # Test with same input multiple times
        sample_input = self.X_test.iloc[:10]
        
        predictions_1 = self.model.predict(sample_input)
        predictions_2 = self.model.predict(sample_input)
        
        assert np.array_equal(predictions_1, predictions_2), "Model predictions not stable"
    
    def test_model_robustness(self):
        """Test model robustness to input variations"""
        sample_input = self.X_test.iloc[:10].copy()
        original_predictions = self.model.predict(sample_input)
        
        # Add small noise
        noise_level = 0.01
        for column in sample_input.select_dtypes(include=[np.number]).columns:
            noisy_input = sample_input.copy()
            noise = np.random.normal(0, noise_level, len(noisy_input))
            noisy_input[column] += noise
            
            noisy_predictions = self.model.predict(noisy_input)
            
            # Most predictions should remain the same
            stability_ratio = np.mean(original_predictions == noisy_predictions)
            assert stability_ratio >= 0.8, f"Model not robust to noise in {column}"
    
    def test_feature_importance(self):
        """Test feature importance availability and validity"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            assert len(importances) == len(self.X_test.columns), "Feature importance length mismatch"
            assert np.all(importances >= 0), "Negative feature importances found"
            assert np.sum(importances) > 0, "All feature importances are zero"
    
    def test_cross_validation_performance(self):
        """Test model performance with cross-validation"""
        # Use a subset for faster testing
        X_subset = self.X_test.iloc[:100]
        y_subset = self.y_test.iloc[:100]
        
        cv_scores = cross_val_score(self.model, X_subset, y_subset, cv=3, scoring='accuracy')
        
        mean_cv_score = np.mean(cv_scores)
        std_cv_score = np.std(cv_scores)
        
        min_cv_score = self.config.get('min_cv_score', 0.6)
        max_cv_std = self.config.get('max_cv_std', 0.2)
        
        assert mean_cv_score >= min_cv_score, f"CV score {mean_cv_score:.3f} below threshold"
        assert std_cv_score <= max_cv_std, f"CV std {std_cv_score:.3f} above threshold"

# Pytest fixtures and test functions
@pytest.fixture
def model_test_suite():
    """Fixture to create model test suite"""
    return MLModelTestSuite(
        model_path="models/trained_model.pkl",
        test_data_path="data/test/test_data.csv",
        config_path="config/model_config.json"
    )

def test_model_loading(model_test_suite):
    model_test_suite.test_model_loading()

def test_prediction_shape(model_test_suite):
    model_test_suite.test_prediction_shape()

def test_prediction_types(model_test_suite):
    model_test_suite.test_prediction_types()

def test_prediction_ranges(model_test_suite):
    model_test_suite.test_prediction_ranges()

def test_model_performance(model_test_suite):
    model_test_suite.test_model_performance()

def test_model_stability(model_test_suite):
    model_test_suite.test_model_stability()

def test_model_robustness(model_test_suite):
    model_test_suite.test_model_robustness()

def test_feature_importance(model_test_suite):
    model_test_suite.test_feature_importance()

def test_cross_validation_performance(model_test_suite):
    model_test_suite.test_cross_validation_performance()
```
### 2. **Data Pipeline Testing**

```python
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import great_expectations as ge
from great_expectations.dataset import PandasDataset

class DataPipelineTestSuite:
    """Test suite for data pipeline validation"""
    
    def __init__(self, data_path, schema_path):
        self.data_path = Path(data_path)
        self.schema_path = Path(schema_path)
        
        # Load data and schema
        self.data = pd.read_csv(self.data_path)
        with open(self.schema_path, 'r') as f:
            self.schema = json.load(f)
    
    def test_data_loading(self):
        """Test data loads correctly"""
        assert not self.data.empty, "Data is empty"
        assert len(self.data.columns) > 0, "No columns in data"
    
    def test_schema_compliance(self):
        """Test data matches expected schema"""
        expected_columns = set(self.schema['columns'].keys())
        actual_columns = set(self.data.columns)
        
        missing_columns = expected_columns - actual_columns
        extra_columns = actual_columns - expected_columns
        
        assert len(missing_columns) == 0, f"Missing columns: {missing_columns}"
        assert len(extra_columns) == 0, f"Extra columns: {extra_columns}"
        
        # Test data types
        for column, expected_type in self.schema['columns'].items():
            actual_type = str(self.data[column].dtype)
            
            if expected_type == 'numeric':
                assert pd.api.types.is_numeric_dtype(self.data[column]), f"Column {column} not numeric"
            elif expected_type == 'categorical':
                assert self.data[column].dtype == 'object' or pd.api.types.is_categorical_dtype(self.data[column]), f"Column {column} not categorical"
    
    def test_data_quality(self):
        """Test data quality metrics"""
        # Test for missing values
        missing_threshold = self.schema.get('max_missing_percentage', 0.1)
        
        for column in self.data.columns:
            missing_percentage = self.data[column].isnull().sum() / len(self.data)
            assert missing_percentage <= missing_threshold, f"Column {column} has {missing_percentage:.2%} missing values"
        
        # Test for duplicates
        max_duplicates = self.schema.get('max_duplicate_percentage', 0.05)
        duplicate_percentage = self.data.duplicated().sum() / len(self.data)
        assert duplicate_percentage <= max_duplicates, f"Data has {duplicate_percentage:.2%} duplicates"
    
    def test_data_distribution(self):
        """Test data distribution properties"""
        ge_data = PandasDataset(self.data)
        
        # Test numeric columns
        for column in self.data.select_dtypes(include=[np.number]).columns:
            if column in self.schema.get('numeric_ranges', {}):
                min_val, max_val = self.schema['numeric_ranges'][column]
                
                # Test value ranges
                assert ge_data.expect_column_values_to_be_between(
                    column, min_val, max_val
                ).success, f"Column {column} values outside expected range"
        
        # Test categorical columns
        for column in self.data.select_dtypes(include=['object']).columns:
            if column in self.schema.get('categorical_values', {}):
                expected_values = self.schema['categorical_values'][column]
                
                # Test allowed values
                assert ge_data.expect_column_values_to_be_in_set(
                    column, expected_values
                ).success, f"Column {column} has unexpected values"
    
    def test_data_freshness(self):
        """Test data freshness"""
        if 'timestamp_column' in self.schema:
            timestamp_col = self.schema['timestamp_column']
            max_age_days = self.schema.get('max_age_days', 7)
            
            if timestamp_col in self.data.columns:
                timestamps = pd.to_datetime(self.data[timestamp_col])
                latest_timestamp = timestamps.max()
                age_days = (pd.Timestamp.now() - latest_timestamp).days
                
                assert age_days <= max_age_days, f"Data is {age_days} days old, exceeds {max_age_days} day limit"

# Performance testing
class ModelPerformanceTestSuite:
    """Test suite for model performance and load testing"""
    
    def __init__(self, model_endpoint, test_data_path):
        self.model_endpoint = model_endpoint
        self.test_data = pd.read_csv(test_data_path)
    
    def test_prediction_latency(self):
        """Test prediction latency requirements"""
        import time
        import requests
        
        sample_data = self.test_data.iloc[0].to_dict()
        
        # Measure latency
        start_time = time.time()
        response = requests.post(
            f"{self.model_endpoint}/predict",
            json=sample_data,
            timeout=5
        )
        end_time = time.time()
        
        latency = end_time - start_time
        max_latency = 1.0  # 1 second
        
        assert response.status_code == 200, f"Prediction request failed: {response.status_code}"
        assert latency <= max_latency, f"Prediction latency {latency:.3f}s exceeds {max_latency}s"
    
    def test_throughput(self):
        """Test model throughput under load"""
        import concurrent.futures
        import requests
        import time
        
        def make_prediction(data):
            response = requests.post(
                f"{self.model_endpoint}/predict",
                json=data,
                timeout=10
            )
            return response.status_code == 200
        
        # Test with concurrent requests
        num_requests = 50
        sample_data = self.test_data.iloc[:num_requests].to_dict('records')
        
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_prediction, data) for data in sample_data]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        end_time = time.time()
        
        success_rate = sum(results) / len(results)
        throughput = num_requests / (end_time - start_time)
        
        min_success_rate = 0.95
        min_throughput = 10  # requests per second
        
        assert success_rate >= min_success_rate, f"Success rate {success_rate:.2%} below {min_success_rate:.2%}"
        assert throughput >= min_throughput, f"Throughput {throughput:.1f} RPS below {min_throughput} RPS"
```

---

## 🚀 Deployment Automation and Strategies

### 1. **Blue-Green Deployment Implementation**

```python
import boto3
import json
import time
from datetime import datetime
from typing import Dict, List, Optional

class BlueGreenDeployment:
    """Blue-Green deployment strategy for ML models"""
    
    def __init__(self, aws_region: str = 'us-west-2'):
        self.ecs_client = boto3.client('ecs', region_name=aws_region)
        self.elbv2_client = boto3.client('elbv2', region_name=aws_region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=aws_region)
        
    def deploy_model(self, 
                    cluster_name: str,
                    service_name: str,
                    new_task_definition: str,
                    target_group_arn: str,
                    health_check_path: str = '/health') -> Dict:
        """
        Execute blue-green deployment
        
        Args:
            cluster_name: ECS cluster name
            service_name: ECS service name
            new_task_definition: New task definition ARN
            target_group_arn: ALB target group ARN
            health_check_path: Health check endpoint path
            
        Returns:
            Deployment result dictionary
        """
        
        deployment_id = f"deploy-{int(datetime.now().timestamp())}"
        
        try:
            # Step 1: Create new service (Green)
            green_service_name = f"{service_name}-green-{deployment_id}"
            
            print(f"Creating green service: {green_service_name}")
            green_service = self._create_green_service(
                cluster_name, green_service_name, new_task_definition
            )
            
            # Step 2: Wait for green service to be healthy
            print("Waiting for green service to be healthy...")
            if not self._wait_for_service_healthy(cluster_name, green_service_name):
                raise Exception("Green service failed to become healthy")
            
            # Step 3: Run health checks
            print("Running health checks on green service...")
            green_endpoint = self._get_service_endpoint(cluster_name, green_service_name)
            
            if not self._run_health_checks(green_endpoint, health_check_path):
                raise Exception("Green service failed health checks")
            
            # Step 4: Switch traffic to green service
            print("Switching traffic to green service...")
            self._switch_traffic(target_group_arn, cluster_name, green_service_name)
            
            # Step 5: Monitor green service performance
            print("Monitoring green service performance...")
            if not self._monitor_performance(green_endpoint, duration=300):
                # Rollback if performance issues
                print("Performance issues detected, rolling back...")
                self._rollback_traffic(target_group_arn, cluster_name, service_name)
                raise Exception("Green service performance issues, rolled back")
            
            # Step 6: Cleanup old blue service
            print("Cleaning up old blue service...")
            self._cleanup_old_service(cluster_name, service_name)
            
            # Step 7: Rename green service to blue
            self._rename_service(cluster_name, green_service_name, service_name)
            
            return {
                'status': 'success',
                'deployment_id': deployment_id,
                'green_service': green_service_name,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Deployment failed: {str(e)}")
            
            # Cleanup on failure
            try:
                self._cleanup_failed_deployment(cluster_name, green_service_name)
            except:
                pass
            
            return {
                'status': 'failed',
                'deployment_id': deployment_id,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_green_service(self, cluster_name: str, service_name: str, task_definition: str) -> Dict:
        """Create new green service"""
        
        response = self.ecs_client.create_service(
            cluster=cluster_name,
            serviceName=service_name,
            taskDefinition=task_definition,
            desiredCount=2,
            launchType='FARGATE',
            networkConfiguration={
                'awsvpcConfiguration': {
                    'subnets': self._get_subnet_ids(),
                    'securityGroups': self._get_security_group_ids(),
                    'assignPublicIp': 'ENABLED'
                }
            },
            loadBalancers=[
                {
                    'targetGroupArn': self._create_temp_target_group(),
                    'containerName': 'ml-model',
                    'containerPort': 8080
                }
            ]
        )
        
        return response['service']
    
    def _wait_for_service_healthy(self, cluster_name: str, service_name: str, timeout: int = 600) -> bool:
        """Wait for service to become healthy"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.ecs_client.describe_services(
                cluster=cluster_name,
                services=[service_name]
            )
            
            service = response['services'][0]
            
            if (service['runningCount'] == service['desiredCount'] and 
                service['status'] == 'ACTIVE'):
                return True
            
            time.sleep(30)
        
        return False
    
    def _run_health_checks(self, endpoint: str, health_path: str) -> bool:
        """Run comprehensive health checks"""
        
        import requests
        
        health_checks = [
            self._check_endpoint_health,
            self._check_prediction_functionality,
            self._check_response_time
        ]
        
        for check in health_checks:
            if not check(endpoint, health_path):
                return False
        
        return True
    
    def _check_endpoint_health(self, endpoint: str, health_path: str) -> bool:
        """Check basic endpoint health"""
        
        try:
            response = requests.get(f"{endpoint}{health_path}", timeout=10)
            return response.status_code == 200
        except:
            return False
    
    def _check_prediction_functionality(self, endpoint: str, health_path: str) -> bool:
        """Check prediction functionality"""
        
        try:
            # Test prediction with sample data
            sample_data = {
                "feature1": 1.0,
                "feature2": 2.0,
                "feature3": 3.0
            }
            
            response = requests.post(
                f"{endpoint}/predict",
                json=sample_data,
                timeout=10
            )
            
            return response.status_code == 200 and 'prediction' in response.json()
        except:
            return False
    
    def _check_response_time(self, endpoint: str, health_path: str) -> bool:
        """Check response time requirements"""
        
        try:
            start_time = time.time()
            response = requests.get(f"{endpoint}{health_path}", timeout=5)
            end_time = time.time()
            
            response_time = end_time - start_time
            return response.status_code == 200 and response_time < 2.0
        except:
            return False
    
    def _switch_traffic(self, target_group_arn: str, cluster_name: str, service_name: str):
        """Switch ALB traffic to new service"""
        
        # Get new service target IPs
        new_targets = self._get_service_targets(cluster_name, service_name)
        
        # Register new targets
        self.elbv2_client.register_targets(
            TargetGroupArn=target_group_arn,
            Targets=new_targets
        )
        
        # Wait for targets to be healthy
        self._wait_for_targets_healthy(target_group_arn, new_targets)
        
        # Deregister old targets
        old_targets = self._get_current_targets(target_group_arn)
        if old_targets:
            self.elbv2_client.deregister_targets(
                TargetGroupArn=target_group_arn,
                Targets=old_targets
            )
    
    def _monitor_performance(self, endpoint: str, duration: int = 300) -> bool:
        """Monitor service performance during deployment"""
        
        start_time = time.time()
        error_count = 0
        total_requests = 0
        
        while time.time() - start_time < duration:
            try:
                response = requests.get(f"{endpoint}/health", timeout=5)
                total_requests += 1
                
                if response.status_code != 200:
                    error_count += 1
                
                # Check error rate
                error_rate = error_count / total_requests if total_requests > 0 else 0
                
                if error_rate > 0.05:  # 5% error threshold
                    return False
                
            except:
                error_count += 1
                total_requests += 1
            
            time.sleep(10)
        
        return True
    
    def _rollback_traffic(self, target_group_arn: str, cluster_name: str, original_service: str):
        """Rollback traffic to original service"""
        
        # Get original service targets
        original_targets = self._get_service_targets(cluster_name, original_service)
        
        # Register original targets
        self.elbv2_client.register_targets(
            TargetGroupArn=target_group_arn,
            Targets=original_targets
        )
        
        # Wait for targets to be healthy
        self._wait_for_targets_healthy(target_group_arn, original_targets)
    
    def _cleanup_old_service(self, cluster_name: str, service_name: str):
        """Cleanup old service"""
        
        # Scale down to 0
        self.ecs_client.update_service(
            cluster=cluster_name,
            service=service_name,
            desiredCount=0
        )
        
        # Wait for tasks to stop
        time.sleep(60)
        
        # Delete service
        self.ecs_client.delete_service(
            cluster=cluster_name,
            service=service_name
        )
    
    def _cleanup_failed_deployment(self, cluster_name: str, service_name: str):
        """Cleanup failed deployment"""
        
        try:
            self._cleanup_old_service(cluster_name, service_name)
        except:
            pass
    
    # Helper methods
    def _get_subnet_ids(self) -> List[str]:
        """Get subnet IDs for service"""
        # Implementation depends on your VPC setup
        return ['subnet-12345', 'subnet-67890']
    
    def _get_security_group_ids(self) -> List[str]:
        """Get security group IDs for service"""
        # Implementation depends on your security setup
        return ['sg-12345']
    
    def _create_temp_target_group(self) -> str:
        """Create temporary target group for green service"""
        # Implementation for creating temporary target group
        return 'arn:aws:elasticloadbalancing:us-west-2:123456789012:targetgroup/temp-tg/1234567890123456'
    
    def _get_service_endpoint(self, cluster_name: str, service_name: str) -> str:
        """Get service endpoint URL"""
        # Implementation to get service endpoint
        return 'http://green-service-endpoint.com'
    
    def _get_service_targets(self, cluster_name: str, service_name: str) -> List[Dict]:
        """Get service target definitions"""
        # Implementation to get service targets
        return [{'Id': '10.0.1.100', 'Port': 8080}]
    
    def _get_current_targets(self, target_group_arn: str) -> List[Dict]:
        """Get current target group targets"""
        
        response = self.elbv2_client.describe_target_health(
            TargetGroupArn=target_group_arn
        )
        
        return [{'Id': target['Target']['Id'], 'Port': target['Target']['Port']} 
                for target in response['TargetHealthDescriptions']]
    
    def _wait_for_targets_healthy(self, target_group_arn: str, targets: List[Dict], timeout: int = 300):
        """Wait for targets to become healthy"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            response = self.elbv2_client.describe_target_health(
                TargetGroupArn=target_group_arn,
                Targets=targets
            )
            
            all_healthy = all(
                target['TargetHealth']['State'] == 'healthy'
                for target in response['TargetHealthDescriptions']
            )
            
            if all_healthy:
                return True
            
            time.sleep(30)
        
        return False
    
    def _rename_service(self, cluster_name: str, old_name: str, new_name: str):
        """Rename service (conceptual - ECS doesn't support renaming)"""
        # In practice, you would update your service discovery or DNS records
        pass
```
### 2. **Canary Deployment Strategy**

```python
class CanaryDeployment:
    """Canary deployment strategy for gradual rollout"""
    
    def __init__(self, aws_region: str = 'us-west-2'):
        self.elbv2_client = boto3.client('elbv2', region_name=aws_region)
        self.cloudwatch = boto3.client('cloudwatch', region_name=aws_region)
        
    def deploy_canary(self,
                     target_group_arn: str,
                     canary_service_endpoint: str,
                     production_service_endpoint: str,
                     initial_traffic_percentage: int = 5,
                     monitoring_duration: int = 1800) -> Dict:
        """
        Deploy canary version with gradual traffic increase
        
        Args:
            target_group_arn: ALB target group ARN
            canary_service_endpoint: Canary service endpoint
            production_service_endpoint: Production service endpoint
            initial_traffic_percentage: Initial traffic percentage for canary
            monitoring_duration: Monitoring duration in seconds
            
        Returns:
            Deployment result dictionary
        """
        
        deployment_stages = [
            {'percentage': initial_traffic_percentage, 'duration': 300},
            {'percentage': 10, 'duration': 600},
            {'percentage': 25, 'duration': 900},
            {'percentage': 50, 'duration': 900},
            {'percentage': 100, 'duration': 0}
        ]
        
        for stage in deployment_stages:
            print(f"Deploying {stage['percentage']}% traffic to canary...")
            
            # Update traffic distribution
            self._update_traffic_distribution(
                target_group_arn,
                canary_service_endpoint,
                production_service_endpoint,
                stage['percentage']
            )
            
            # Monitor performance
            if stage['duration'] > 0:
                performance_result = self._monitor_canary_performance(
                    canary_service_endpoint,
                    production_service_endpoint,
                    stage['duration']
                )
                
                if not performance_result['success']:
                    # Rollback on failure
                    print(f"Canary performance issues detected: {performance_result['reason']}")
                    self._rollback_canary(target_group_arn, production_service_endpoint)
                    
                    return {
                        'status': 'failed',
                        'stage': stage['percentage'],
                        'reason': performance_result['reason'],
                        'timestamp': datetime.now().isoformat()
                    }
        
        return {
            'status': 'success',
            'final_traffic_percentage': 100,
            'timestamp': datetime.now().isoformat()
        }
    
    def _update_traffic_distribution(self,
                                   target_group_arn: str,
                                   canary_endpoint: str,
                                   production_endpoint: str,
                                   canary_percentage: int):
        """Update ALB traffic distribution"""
        
        # Implementation would use ALB weighted routing
        # This is a simplified version
        
        canary_weight = canary_percentage
        production_weight = 100 - canary_percentage
        
        # Update ALB listener rules for weighted routing
        # Implementation depends on your ALB configuration
        
        print(f"Traffic distribution: Canary {canary_weight}%, Production {production_weight}%")
    
    def _monitor_canary_performance(self,
                                  canary_endpoint: str,
                                  production_endpoint: str,
                                  duration: int) -> Dict:
        """Monitor canary vs production performance"""
        
        start_time = time.time()
        
        canary_metrics = {'requests': 0, 'errors': 0, 'response_times': []}
        production_metrics = {'requests': 0, 'errors': 0, 'response_times': []}
        
        while time.time() - start_time < duration:
            # Test canary
            canary_result = self._test_endpoint(canary_endpoint)
            canary_metrics['requests'] += 1
            
            if canary_result['success']:
                canary_metrics['response_times'].append(canary_result['response_time'])
            else:
                canary_metrics['errors'] += 1
            
            # Test production
            production_result = self._test_endpoint(production_endpoint)
            production_metrics['requests'] += 1
            
            if production_result['success']:
                production_metrics['response_times'].append(production_result['response_time'])
            else:
                production_metrics['errors'] += 1
            
            time.sleep(10)
        
        # Analyze performance
        return self._analyze_performance_comparison(canary_metrics, production_metrics)
    
    def _test_endpoint(self, endpoint: str) -> Dict:
        """Test endpoint performance"""
        
        import requests
        
        try:
            start_time = time.time()
            response = requests.get(f"{endpoint}/health", timeout=5)
            end_time = time.time()
            
            return {
                'success': response.status_code == 200,
                'response_time': end_time - start_time,
                'status_code': response.status_code
            }
        except Exception as e:
            return {
                'success': False,
                'response_time': 5.0,
                'error': str(e)
            }
    
    def _analyze_performance_comparison(self, canary_metrics: Dict, production_metrics: Dict) -> Dict:
        """Analyze canary vs production performance"""
        
        # Calculate error rates
        canary_error_rate = canary_metrics['errors'] / canary_metrics['requests'] if canary_metrics['requests'] > 0 else 1
        production_error_rate = production_metrics['errors'] / production_metrics['requests'] if production_metrics['requests'] > 0 else 1
        
        # Calculate average response times
        canary_avg_response_time = np.mean(canary_metrics['response_times']) if canary_metrics['response_times'] else float('inf')
        production_avg_response_time = np.mean(production_metrics['response_times']) if production_metrics['response_times'] else float('inf')
        
        # Performance thresholds
        max_error_rate_increase = 0.02  # 2% increase
        max_response_time_increase = 0.5  # 50% increase
        
        # Check for performance degradation
        error_rate_increase = canary_error_rate - production_error_rate
        response_time_increase = (canary_avg_response_time - production_avg_response_time) / production_avg_response_time if production_avg_response_time > 0 else 0
        
        if error_rate_increase > max_error_rate_increase:
            return {
                'success': False,
                'reason': f'Error rate increased by {error_rate_increase:.2%}'
            }
        
        if response_time_increase > max_response_time_increase:
            return {
                'success': False,
                'reason': f'Response time increased by {response_time_increase:.2%}'
            }
        
        return {
            'success': True,
            'canary_error_rate': canary_error_rate,
            'production_error_rate': production_error_rate,
            'canary_avg_response_time': canary_avg_response_time,
            'production_avg_response_time': production_avg_response_time
        }
    
    def _rollback_canary(self, target_group_arn: str, production_endpoint: str):
        """Rollback canary deployment"""
        
        print("Rolling back canary deployment...")
        
        # Set 100% traffic to production
        self._update_traffic_distribution(
            target_group_arn,
            "",  # No canary endpoint
            production_endpoint,
            0  # 0% canary traffic
        )
```

---

## 🔍 Monitoring and Observability

### 1. **ML Pipeline Monitoring**

```python
import boto3
import json
from datetime import datetime, timedelta
from typing import Dict, List

class MLPipelineMonitor:
    """Comprehensive monitoring for ML pipelines"""
    
    def __init__(self, aws_region: str = 'us-west-2'):
        self.cloudwatch = boto3.client('cloudwatch', region_name=aws_region)
        self.sns = boto3.client('sns', region_name=aws_region)
        
    def setup_monitoring(self, pipeline_name: str, notification_topic_arn: str):
        """Setup comprehensive monitoring for ML pipeline"""
        
        # Create custom metrics
        self._create_custom_metrics(pipeline_name)
        
        # Setup alarms
        self._create_pipeline_alarms(pipeline_name, notification_topic_arn)
        
        # Setup dashboard
        self._create_monitoring_dashboard(pipeline_name)
    
    def _create_custom_metrics(self, pipeline_name: str):
        """Create custom CloudWatch metrics"""
        
        metrics = [
            'ModelAccuracy',
            'PredictionLatency',
            'DataDriftScore',
            'ModelDriftScore',
            'PipelineExecutionTime',
            'DataQualityScore'
        ]
        
        for metric in metrics:
            self.cloudwatch.put_metric_data(
                Namespace=f'MLPipeline/{pipeline_name}',
                MetricData=[
                    {
                        'MetricName': metric,
                        'Value': 0,
                        'Unit': 'None',
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
    
    def _create_pipeline_alarms(self, pipeline_name: str, notification_topic_arn: str):
        """Create CloudWatch alarms for pipeline monitoring"""
        
        alarms = [
            {
                'name': f'{pipeline_name}-ModelAccuracy-Low',
                'metric': 'ModelAccuracy',
                'threshold': 0.8,
                'comparison': 'LessThanThreshold',
                'description': 'Model accuracy dropped below threshold'
            },
            {
                'name': f'{pipeline_name}-PredictionLatency-High',
                'metric': 'PredictionLatency',
                'threshold': 2000,  # 2 seconds in milliseconds
                'comparison': 'GreaterThanThreshold',
                'description': 'Prediction latency exceeded threshold'
            },
            {
                'name': f'{pipeline_name}-DataDrift-High',
                'metric': 'DataDriftScore',
                'threshold': 0.3,
                'comparison': 'GreaterThanThreshold',
                'description': 'Data drift detected above threshold'
            },
            {
                'name': f'{pipeline_name}-PipelineExecution-Long',
                'metric': 'PipelineExecutionTime',
                'threshold': 3600,  # 1 hour in seconds
                'comparison': 'GreaterThanThreshold',
                'description': 'Pipeline execution time exceeded threshold'
            }
        ]
        
        for alarm in alarms:
            self.cloudwatch.put_metric_alarm(
                AlarmName=alarm['name'],
                ComparisonOperator=alarm['comparison'],
                EvaluationPeriods=2,
                MetricName=alarm['metric'],
                Namespace=f'MLPipeline/{pipeline_name}',
                Period=300,  # 5 minutes
                Statistic='Average',
                Threshold=alarm['threshold'],
                ActionsEnabled=True,
                AlarmActions=[notification_topic_arn],
                AlarmDescription=alarm['description'],
                Unit='None'
            )
    
    def _create_monitoring_dashboard(self, pipeline_name: str):
        """Create CloudWatch dashboard for pipeline monitoring"""
        
        dashboard_body = {
            "widgets": [
                {
                    "type": "metric",
                    "x": 0,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [f"MLPipeline/{pipeline_name}", "ModelAccuracy"],
                            [".", "DataQualityScore"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": "us-west-2",
                        "title": "Model Performance"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 0,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [f"MLPipeline/{pipeline_name}", "PredictionLatency"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": "us-west-2",
                        "title": "Prediction Latency"
                    }
                },
                {
                    "type": "metric",
                    "x": 0,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [f"MLPipeline/{pipeline_name}", "DataDriftScore"],
                            [".", "ModelDriftScore"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": "us-west-2",
                        "title": "Drift Detection"
                    }
                },
                {
                    "type": "metric",
                    "x": 12,
                    "y": 6,
                    "width": 12,
                    "height": 6,
                    "properties": {
                        "metrics": [
                            [f"MLPipeline/{pipeline_name}", "PipelineExecutionTime"]
                        ],
                        "period": 300,
                        "stat": "Average",
                        "region": "us-west-2",
                        "title": "Pipeline Execution Time"
                    }
                }
            ]
        }
        
        self.cloudwatch.put_dashboard(
            DashboardName=f'{pipeline_name}-MLPipeline-Dashboard',
            DashboardBody=json.dumps(dashboard_body)
        )
    
    def publish_metrics(self, pipeline_name: str, metrics: Dict):
        """Publish custom metrics to CloudWatch"""
        
        metric_data = []
        
        for metric_name, value in metrics.items():
            metric_data.append({
                'MetricName': metric_name,
                'Value': value,
                'Unit': 'None',
                'Timestamp': datetime.utcnow()
            })
        
        self.cloudwatch.put_metric_data(
            Namespace=f'MLPipeline/{pipeline_name}',
            MetricData=metric_data
        )
    
    def get_pipeline_health(self, pipeline_name: str, hours: int = 24) -> Dict:
        """Get pipeline health summary"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours)
        
        metrics_to_check = [
            'ModelAccuracy',
            'PredictionLatency',
            'DataDriftScore',
            'PipelineExecutionTime'
        ]
        
        health_summary = {}
        
        for metric in metrics_to_check:
            response = self.cloudwatch.get_metric_statistics(
                Namespace=f'MLPipeline/{pipeline_name}',
                MetricName=metric,
                Dimensions=[],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,  # 1 hour
                Statistics=['Average', 'Maximum', 'Minimum']
            )
            
            if response['Datapoints']:
                datapoints = sorted(response['Datapoints'], key=lambda x: x['Timestamp'])
                latest = datapoints[-1]
                
                health_summary[metric] = {
                    'current_value': latest['Average'],
                    'max_value': max(dp['Maximum'] for dp in datapoints),
                    'min_value': min(dp['Minimum'] for dp in datapoints),
                    'trend': self._calculate_trend(datapoints)
                }
        
        return health_summary
    
    def _calculate_trend(self, datapoints: List[Dict]) -> str:
        """Calculate metric trend"""
        
        if len(datapoints) < 2:
            return 'insufficient_data'
        
        values = [dp['Average'] for dp in datapoints]
        
        # Simple trend calculation
        recent_avg = np.mean(values[-3:]) if len(values) >= 3 else values[-1]
        older_avg = np.mean(values[:-3]) if len(values) >= 6 else values[0]
        
        if recent_avg > older_avg * 1.05:
            return 'increasing'
        elif recent_avg < older_avg * 0.95:
            return 'decreasing'
        else:
            return 'stable'
```

---

## 🔧 Hands-On Exercise

You'll build a complete CI/CD pipeline for ML that includes automated testing, deployment, and monitoring:

### Exercise Scenario
**Company**: FinTech Solutions Inc.  
**Challenge**: Build a production-ready CI/CD pipeline for fraud detection models
- **Automated Training**: Trigger retraining on data changes
- **Comprehensive Testing**: Unit, integration, and performance tests
- **Safe Deployment**: Blue-green deployment with automated rollback
- **Monitoring**: Real-time performance and drift monitoring
- **Infrastructure**: Terraform-managed AWS infrastructure

### Requirements
1. **GitHub Actions Pipeline**: Complete CI/CD workflow with multiple stages
2. **Infrastructure as Code**: Terraform configuration for AWS resources
3. **Testing Framework**: Comprehensive test suite for models and data
4. **Deployment Strategy**: Blue-green deployment with health checks
5. **Monitoring Setup**: CloudWatch dashboards and alerting

---

## 📚 Key Takeaways

- **Automate everything** - from data validation to model deployment and monitoring
- **Test comprehensively** - models, data, infrastructure, and performance
- **Deploy safely** - use blue-green or canary strategies with automated rollback
- **Monitor continuously** - track model performance, data drift, and system health
- **Infrastructure as code** - version and automate infrastructure provisioning
- **Fail fast and safe** - implement proper validation gates and rollback mechanisms
- **Document processes** - maintain clear deployment and rollback procedures
- **Security first** - integrate security scanning and compliance checks throughout pipeline

---

## 🔄 What's Next?

Tomorrow, we'll explore **Feature Monitoring & Drift** where you'll learn how to:
- Detect and respond to data drift in production
- Monitor feature quality and distribution changes
- Implement automated retraining triggers
- Build comprehensive model performance dashboards