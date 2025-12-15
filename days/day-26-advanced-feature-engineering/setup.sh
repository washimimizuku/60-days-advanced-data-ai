#!/bin/bash

# Day 26: Advanced Feature Engineering Setup Script
# Automated setup for production-ready feature engineering environment

set -e

echo "🚀 Setting up Advanced Feature Engineering Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Check prerequisites
print_header "Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

print_status "Docker and Docker Compose are available"

# Create necessary directories
print_header "Creating directory structure..."

mkdir -p data/{raw,processed,init,models}
mkdir -p notebooks
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
mkdir -p logs
mkdir -p models/{time_series,nlp,selection}

print_status "Directory structure created"

# Create monitoring configuration
print_header "Setting up monitoring configuration..."

# Prometheus configuration
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'feature-api'
    static_configs:
      - targets: ['feature-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s
EOF

# Grafana datasource configuration
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Grafana dashboard configuration
cat > monitoring/grafana/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

print_status "Monitoring configuration created"

# Create database initialization script
print_header "Setting up database initialization..."

cat > data/init/01-init.sql << EOF
-- Initialize FinTech Feature Engineering Database

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS features;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS models;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA features TO feature_user;
GRANT ALL PRIVILEGES ON SCHEMA monitoring TO feature_user;
GRANT ALL PRIVILEGES ON SCHEMA models TO feature_user;

-- Create feature quality monitoring table
CREATE TABLE IF NOT EXISTS monitoring.feature_quality (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_set VARCHAR(255),
    quality_score REAL,
    missing_ratio REAL,
    outlier_ratio REAL,
    drift_score REAL
);

-- Create feature selection results table
CREATE TABLE IF NOT EXISTS monitoring.feature_selection (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    selection_method VARCHAR(100),
    original_features INTEGER,
    selected_features INTEGER,
    selection_time_ms REAL
);

-- Create model performance tracking
CREATE TABLE IF NOT EXISTS monitoring.model_performance (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name VARCHAR(255),
    feature_count INTEGER,
    accuracy REAL,
    precision_score REAL,
    recall REAL,
    f1_score REAL
);

-- Insert sample monitoring data
INSERT INTO monitoring.feature_quality (feature_set, quality_score, missing_ratio, outlier_ratio, drift_score) VALUES
('time_series_features', 0.92, 0.05, 0.02, 0.15),
('nlp_features', 0.88, 0.08, 0.01, 0.12),
('demographic_features', 0.95, 0.02, 0.03, 0.08);

INSERT INTO monitoring.feature_selection (selection_method, original_features, selected_features, selection_time_ms) VALUES
('univariate', 500, 100, 1250.5),
('mutual_info', 500, 85, 2100.3),
('rfe', 500, 75, 5500.8),
('ensemble', 500, 90, 8850.6);

COMMIT;
EOF

print_status "Database initialization script created"

# Create Dockerfiles
print_header "Creating Dockerfiles..."

# API Dockerfile
cat > Dockerfile.api << EOF
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create app user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start command
CMD ["python", "api.py"]
EOF

# Jupyter Dockerfile
cat > Dockerfile.jupyter << EOF
FROM jupyter/scipy-notebook:latest

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

USER jovyan

# Copy requirements and install
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Install additional Jupyter extensions
RUN pip install jupyterlab-git jupyterlab-lsp

# Set working directory
WORKDIR /home/jovyan/work

# Expose port
EXPOSE 8888

# Start Jupyter Lab
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
EOF

# Data generator Dockerfile
cat > Dockerfile.datagen << EOF
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    libpq-dev \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data generator
COPY data_generator.py .

# Create data directory
RUN mkdir -p data

# Run data generator
CMD ["python", "data_generator.py"]
EOF

print_status "Dockerfiles created"

# Create sample notebook
print_header "Creating sample Jupyter notebook..."

cat > notebooks/feature_engineering_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Advanced Feature Engineering Demo\n",
    "\n",
    "This notebook demonstrates advanced feature engineering techniques:\n",
    "1. Time series feature engineering\n",
    "2. NLP feature extraction\n",
    "3. Automated feature selection\n",
    "4. Feature quality monitoring"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from datetime import datetime, timedelta\n",
    "import sys\n",
    "sys.path.append('../')\n",
    "\n",
    "from solution import (\n",
    "    AdvancedTimeSeriesFeatureEngineer,\n",
    "    ProductionNLPFeatureEngineer,\n",
    "    EnsembleFeatureSelector,\n",
    "    AdvancedFeatureGenerator\n",
    ")\n",
    "\n",
    "print(\"🚀 Advanced Feature Engineering Demo\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load sample data\n",
    "customers = pd.read_parquet('../data/customers.parquet')\n",
    "transactions = pd.read_parquet('../data/transactions.parquet')\n",
    "feedback = pd.read_parquet('../data/feedback.parquet')\n",
    "\n",
    "print(f\"Loaded {len(customers)} customers, {len(transactions)} transactions, {len(feedback)} feedback\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Time series feature engineering\n",
    "ts_engineer = AdvancedTimeSeriesFeatureEngineer()\n",
    "\n",
    "# Create temporal features\n",
    "transactions_with_features = ts_engineer.create_temporal_features(transactions, 'timestamp')\n",
    "print(f\"Created {len(transactions_with_features.columns)} temporal features\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
EOF

print_status "Sample notebook created"

# Make scripts executable
chmod +x data_generator.py

print_header "Starting services..."

# Start Docker Compose services
docker-compose up -d

print_status "Waiting for services to start..."
sleep 30

# Check service health
print_header "Checking service health..."

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U feature_user > /dev/null 2>&1; then
    print_status "PostgreSQL is ready"
else
    print_warning "PostgreSQL is not ready yet"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    print_status "Redis is ready"
else
    print_warning "Redis is not ready yet"
fi

# Generate sample data
print_header "Generating sample FinTech data..."
docker-compose run --rm data-generator

print_header "Setup completed successfully! 🎉"

echo ""
echo "📊 Access your services:"
echo "   • Feature Engineering API: http://localhost:8000"
echo "   • Jupyter Lab: http://localhost:8888"
echo "   • Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "   • Prometheus: http://localhost:9090"
echo ""
echo "🚀 Next steps:"
echo "   1. Open Jupyter Lab and run the demo notebook"
echo "   2. Test the feature engineering API: curl http://localhost:8000/health"
echo "   3. Explore Grafana dashboards for monitoring"
echo "   4. Complete the exercises in exercise.py"
echo ""
echo "📚 Documentation:"
echo "   • README.md - Complete guide and exercises"
echo "   • solution.py - Full implementation reference"
echo "   • notebooks/ - Interactive examples"
echo ""