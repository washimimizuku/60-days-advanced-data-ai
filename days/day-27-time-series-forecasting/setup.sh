#!/bin/bash

# Day 27: Time Series Forecasting Setup Script
# Sets up complete forecasting infrastructure with Docker services

set -e

echo "🚀 Setting up Day 27: Time Series Forecasting Environment"
echo "=========================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker is running"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed logs grafana/dashboards grafana/datasources

# Create init.sql for PostgreSQL
echo "📄 Creating database initialization script..."
cat > init.sql << 'EOF'
-- Initialize time series forecasting database

-- Create tables for different time series types
CREATE TABLE IF NOT EXISTS retail_sales (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    sales FLOAT NOT NULL,
    category VARCHAR(50),
    location VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS energy_consumption (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    consumption FLOAT NOT NULL,
    category VARCHAR(50),
    location VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS stock_prices (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    price FLOAT NOT NULL,
    category VARCHAR(50),
    symbol VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS website_traffic (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    visitors INTEGER NOT NULL,
    category VARCHAR(50),
    site VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_retail_timestamp ON retail_sales(timestamp);
CREATE INDEX IF NOT EXISTS idx_energy_timestamp ON energy_consumption(timestamp);
CREATE INDEX IF NOT EXISTS idx_stock_timestamp ON stock_prices(timestamp);
CREATE INDEX IF NOT EXISTS idx_traffic_timestamp ON website_traffic(timestamp);

-- Create model metadata table
CREATE TABLE IF NOT EXISTS forecast_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    series_name VARCHAR(100) NOT NULL,
    parameters JSONB,
    performance_metrics JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create forecasts table
CREATE TABLE IF NOT EXISTS forecasts (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES forecast_models(id),
    series_name VARCHAR(100) NOT NULL,
    forecast_date TIMESTAMP NOT NULL,
    forecast_value FLOAT NOT NULL,
    confidence_lower FLOAT,
    confidence_upper FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO forecast_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO forecast_user;
EOF

# Create Prometheus configuration
echo "📊 Creating Prometheus configuration..."
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'forecast-api'
    static_configs:
      - targets: ['forecast-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']
EOF

# Create Grafana datasource configuration
echo "📈 Creating Grafana configuration..."
mkdir -p grafana/datasources
cat > grafana/datasources/datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: InfluxDB
    type: influxdb
    access: proxy
    url: http://influxdb:8086
    database: timeseries
    user: admin
    password: forecast123

  - name: PostgreSQL
    type: postgres
    access: proxy
    url: postgres:5432
    database: timeseries_db
    user: forecast_user
    password: forecast_pass
EOF

# Create Grafana dashboard
mkdir -p grafana/dashboards
cat > grafana/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# Create API Dockerfile
echo "🐳 Creating API Dockerfile..."
cat > Dockerfile.api << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

# Create Data Generator Dockerfile
echo "🐳 Creating Data Generator Dockerfile..."
cat > Dockerfile.generator << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the data generator
CMD ["python", "data_generator.py"]
EOF

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if command -v python3 &> /dev/null; then
    python3 -m pip install --user -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️  Python3 not found. Dependencies will be installed in Docker containers."
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose down --remove-orphans
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U forecast_user -d timeseries_db > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
fi

# Check InfluxDB
if curl -f http://localhost:8086/ping > /dev/null 2>&1; then
    echo "✅ InfluxDB is ready"
else
    echo "❌ InfluxDB is not ready"
fi

# Generate sample data
echo "📊 Generating sample time series data..."
sleep 10  # Additional wait for data generator

# Display service URLs
echo ""
echo "🎯 Service URLs:"
echo "================================"
echo "📊 Forecasting API:     http://localhost:8000"
echo "📈 Grafana Dashboard:   http://localhost:3000 (admin/forecast123)"
echo "📊 Prometheus:          http://localhost:9090"
echo "🗄️  InfluxDB:            http://localhost:8086"
echo "💾 MLflow:              http://localhost:5000"
echo "📓 Jupyter Notebook:    http://localhost:8888 (token: forecast123)"
echo ""

# Display API endpoints
echo "🔗 API Endpoints:"
echo "================================"
echo "GET  /health              - Health check"
echo "GET  /series              - List available time series"
echo "POST /forecast            - Generate forecasts"
echo "GET  /series/{name}/data  - Get time series data"
echo ""

# Display demo instructions
echo "🚀 Getting Started:"
echo "================================"
echo "1. Run the interactive demo:"
echo "   python demo.py"
echo ""
echo "2. Test the API:"
echo "   curl http://localhost:8000/health"
echo ""
echo "3. Open Jupyter for analysis:"
echo "   http://localhost:8888 (token: forecast123)"
echo ""
echo "4. View monitoring in Grafana:"
echo "   http://localhost:3000 (admin/forecast123)"
echo ""

echo "✅ Day 27 Time Series Forecasting environment is ready!"
echo "🎯 You can now work with ARIMA, Prophet, LSTM, and ensemble forecasting!"