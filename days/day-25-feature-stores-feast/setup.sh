#!/bin/bash

# Day 25: Feature Stores - Feast Setup Script
# Automated setup for production-ready feature store environment

set -e

echo "🚀 Setting up RideShare Feature Store Environment..."

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

mkdir -p data/{raw,processed,init}
mkdir -p feature_repo
mkdir -p notebooks
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
mkdir -p logs

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
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'feast-server'
    static_configs:
      - targets: ['feast-server:8000']
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
-- Initialize RideShare Feature Store Database

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS feast;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA feast TO feast_user;
GRANT ALL PRIVILEGES ON SCHEMA monitoring TO feast_user;

-- Create monitoring table
CREATE TABLE IF NOT EXISTS monitoring.feature_requests (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    feature_view VARCHAR(255),
    latency_ms REAL,
    status VARCHAR(50),
    entity_count INTEGER
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_feature_requests_timestamp ON monitoring.feature_requests(timestamp);
CREATE INDEX IF NOT EXISTS idx_feature_requests_feature_view ON monitoring.feature_requests(feature_view);

-- Insert sample monitoring data
INSERT INTO monitoring.feature_requests (feature_view, latency_ms, status, entity_count) VALUES
('driver_performance_features', 5.2, 'success', 100),
('user_behavior_features', 3.8, 'success', 50),
('location_demand_features', 7.1, 'success', 25);

COMMIT;
EOF

print_status "Database initialization script created"

# Create Dockerfiles
print_header "Creating Dockerfiles..."

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
RUN pip install faker

# Copy data generator
COPY data_generator.py .

# Create data directory
RUN mkdir -p data

# Run data generator
CMD ["python", "data_generator.py"]
EOF

print_status "Dockerfiles created"

# Create feature store configuration
print_header "Setting up Feast configuration..."

mkdir -p feature_repo

cat > feature_repo/feature_store.yaml << EOF
project: rideshare_ml_platform
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: "redis://redis:6379"
offline_store:
  type: file
entity_key_serialization_version: 2
EOF

print_status "Feast configuration created"

# Create sample notebook
print_header "Creating sample Jupyter notebook..."

cat > notebooks/feature_store_demo.ipynb << 'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RideShare Feature Store Demo\n",
    "\n",
    "This notebook demonstrates the complete feature store workflow:\n",
    "1. Feature definition and registration\n",
    "2. Feature materialization\n",
    "3. Training data generation\n",
    "4. Online feature serving\n",
    "5. Feature monitoring and validation"
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
    "from solution import RideShareFeatureStore, FeatureStoreConfig, FeatureMonitor\n",
    "\n",
    "print(\"🚀 RideShare Feature Store Demo\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize feature store\n",
    "config = FeatureStoreConfig()\n",
    "fs = RideShareFeatureStore(config)\n",
    "monitor = FeatureMonitor(fs)\n",
    "\n",
    "print(\"✅ Feature store initialized\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test online feature serving\n",
    "entity_rows = [\n",
    "    {\"driver_id\": 1001},\n",
    "    {\"driver_id\": 1002},\n",
    "    {\"driver_id\": 1003}\n",
    "]\n",
    "\n",
    "feature_refs = [\n",
    "    \"driver_performance_features:acceptance_rate_7d\",\n",
    "    \"driver_performance_features:avg_rating_7d\"\n",
    "]\n",
    "\n",
    "result = fs.get_online_features(entity_rows, feature_refs)\n",
    "print(f\"✅ Features served in {result['latency_ms']:.2f}ms\")\n",
    "print(result['features'])"
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
if docker-compose exec -T postgres pg_isready -U feast_user > /dev/null 2>&1; then
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
print_header "Generating sample data..."
docker-compose run --rm data-generator

print_header "Setup completed successfully! 🎉"

echo ""
echo "📊 Access your services:"
echo "   • Feast Feature Server: http://localhost:6566"
echo "   • Feature Serving API: http://localhost:8000"
echo "   • Jupyter Lab: http://localhost:8888"
echo "   • Grafana Dashboard: http://localhost:3000 (admin/admin123)"
echo "   • Prometheus: http://localhost:9090"
echo ""
echo "🚀 Next steps:"
echo "   1. Open Jupyter Lab and run the demo notebook"
echo "   2. Test the feature serving API: curl http://localhost:8000/health"
echo "   3. Explore Grafana dashboards for monitoring"
echo "   4. Complete the exercises in exercise.py"
echo ""
echo "📚 Documentation:"
echo "   • README.md - Complete guide and exercises"
echo "   • solution.py - Full implementation reference"
echo "   • notebooks/ - Interactive examples"
echo ""