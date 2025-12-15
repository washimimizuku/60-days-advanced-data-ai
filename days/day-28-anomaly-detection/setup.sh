#!/bin/bash

# Day 28: Anomaly Detection Setup Script
# Sets up complete anomaly detection infrastructure with Docker services

set -e

echo "🚀 Setting up Day 28: Anomaly Detection Environment"
echo "=================================================="

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
mkdir -p data/raw data/processed logs models grafana/dashboards grafana/datasources

# Create init.sql for PostgreSQL
echo "📄 Creating database initialization script..."
cat > init.sql << 'EOF'
-- Initialize anomaly detection database

-- Create tables for different data types
CREATE TABLE IF NOT EXISTS financial_transactions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    amount FLOAT NOT NULL,
    hour INTEGER,
    is_international BOOLEAN,
    is_anomaly BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS network_traffic (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    packet_size FLOAT NOT NULL,
    duration FLOAT NOT NULL,
    port INTEGER,
    bytes_sent FLOAT,
    bytes_received FLOAT,
    protocol VARCHAR(10),
    is_anomaly BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sensor_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    temperature FLOAT NOT NULL,
    pressure FLOAT,
    vibration FLOAT,
    power_consumption FLOAT,
    is_anomaly BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS user_behavior (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    session_duration FLOAT NOT NULL,
    page_views INTEGER,
    click_rate FLOAT,
    login_attempts INTEGER,
    location VARCHAR(50),
    is_anomaly BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_financial_timestamp ON financial_transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_network_timestamp ON network_traffic(timestamp);
CREATE INDEX IF NOT EXISTS idx_sensor_timestamp ON sensor_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_timestamp ON user_behavior(timestamp);

-- Create anomaly detection models table
CREATE TABLE IF NOT EXISTS anomaly_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    model_type VARCHAR(50) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    parameters JSONB,
    performance_metrics JSONB,
    model_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create anomaly alerts table
CREATE TABLE IF NOT EXISTS anomaly_alerts (
    id SERIAL PRIMARY KEY,
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    dataset_name VARCHAR(100) NOT NULL,
    anomaly_score FLOAT,
    confidence FLOAT,
    data_point JSONB,
    detection_method VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO anomaly_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO anomaly_user;
EOF

# Create Prometheus configuration
echo "📊 Creating Prometheus configuration..."
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "anomaly_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'anomaly-api'
    static_configs:
      - targets: ['anomaly-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka:9092']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

# Create Prometheus alerting rules
echo "🚨 Creating Prometheus alerting rules..."
cat > anomaly_rules.yml << 'EOF'
groups:
- name: anomaly_detection
  rules:
  - alert: HighAnomalyRate
    expr: rate(anomalies_detected_total[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High anomaly detection rate"
      description: "Anomaly detection rate is {{ $value }} per second"

  - alert: ModelPerformanceDegraded
    expr: model_accuracy < 0.8
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Model performance degraded"
      description: "Model accuracy dropped to {{ $value }}"

  - alert: StreamingDetectorDown
    expr: up{job="stream-detector"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Streaming detector is down"
      description: "The streaming anomaly detector has been down for more than 1 minute"
EOF

# Create Alertmanager configuration
echo "📢 Creating Alertmanager configuration..."
cat > alertmanager.yml << 'EOF'
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@anomaly-detection.local'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  webhook_configs:
  - url: 'http://anomaly-api:8000/webhook/alerts'
    send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'dev', 'instance']
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
    database: anomalies
    user: admin
    password: anomaly123

  - name: PostgreSQL
    type: postgres
    access: proxy
    url: postgres:5432
    database: anomaly_db
    user: anomaly_user
    password: anomaly_pass
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

# Create Stream Detector Dockerfile
echo "🐳 Creating Stream Detector Dockerfile..."
cat > Dockerfile.stream << 'EOF'
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

# Run the stream detector
CMD ["python", "stream_detector.py"]
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
sleep 45

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U anomaly_user -d anomaly_db > /dev/null 2>&1; then
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

# Check Kafka
if docker-compose exec -T kafka kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
    echo "✅ Kafka is ready"
else
    echo "❌ Kafka is not ready"
fi

# Create Kafka topics
echo "📡 Creating Kafka topics..."
docker-compose exec -T kafka kafka-topics --create --bootstrap-server localhost:9092 --topic sensor-data --partitions 3 --replication-factor 1 || true
docker-compose exec -T kafka kafka-topics --create --bootstrap-server localhost:9092 --topic anomaly-alerts --partitions 3 --replication-factor 1 || true

# Generate sample data
echo "📊 Generating sample anomaly detection data..."
sleep 15  # Additional wait for data generator

# Display service URLs
echo ""
echo "🎯 Service URLs:"
echo "================================"
echo "📊 Anomaly Detection API: http://localhost:8000"
echo "📈 Grafana Dashboard:     http://localhost:3000 (admin/anomaly123)"
echo "📊 Prometheus:            http://localhost:9090"
echo "🚨 Alertmanager:          http://localhost:9093"
echo "🗄️  InfluxDB:              http://localhost:8086"
echo "💾 MLflow:                http://localhost:5000"
echo "📓 Jupyter Notebook:      http://localhost:8888 (token: anomaly123)"
echo ""

# Display API endpoints
echo "🔗 API Endpoints:"
echo "================================"
echo "GET  /health                    - Health check"
echo "GET  /datasets                  - List available datasets"
echo "POST /detect                    - Detect anomalies in data"
echo "POST /detect/batch              - Batch anomaly detection"
echo "WS   /ws/realtime              - Real-time anomaly detection"
echo "GET  /datasets/{name}/stats     - Dataset statistics"
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
echo "   http://localhost:8888 (token: anomaly123)"
echo ""
echo "4. View monitoring in Grafana:"
echo "   http://localhost:3000 (admin/anomaly123)"
echo ""
echo "5. Check real-time alerts:"
echo "   http://localhost:9093"
echo ""

echo "✅ Day 28 Anomaly Detection environment is ready!"
echo "🎯 You can now work with statistical, ML, and ensemble anomaly detection methods!"