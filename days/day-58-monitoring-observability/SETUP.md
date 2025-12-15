# Day 58: Monitoring & Observability - Setup Guide

## Overview
This guide helps you set up a comprehensive monitoring and observability stack for ML systems, including Prometheus, Grafana, Jaeger, and the ELK stack for production-ready observability.

## Prerequisites

### Required Software
- **Docker** >= 20.0 (Container runtime)
- **Docker Compose** >= 2.0 (Multi-container orchestration)
- **Python** >= 3.8 (Development environment)
- **Kubernetes** >= 1.20 (Optional, for production deployment)
- **Helm** >= 3.0 (Optional, for Kubernetes deployments)

### System Requirements
- **Memory**: 8GB+ RAM (16GB+ recommended for full stack)
- **Storage**: 20GB+ free disk space
- **CPU**: 4+ cores recommended
- **Network**: Internet access for downloading images

## Installation Steps

### 1. Install Core Dependencies

#### Python Dependencies
```bash
# Create virtual environment
python -m venv observability-env
source observability-env/bin/activate  # On Windows: observability-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import prometheus_client; print('Prometheus client installed')"
python -c "import opentelemetry; print('OpenTelemetry installed')"
```

#### Docker and Docker Compose
```bash
# Verify Docker installation
docker --version
docker-compose --version

# Test Docker functionality
docker run hello-world
```

### 2. Set Up Prometheus

#### Prometheus Configuration
Create `prometheus/prometheus.yml`:
```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  # Prometheus itself
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # ML Model Serving
  - job_name: 'ml-model-serving'
    static_configs:
      - targets: ['ml-service:8080']
    metrics_path: /metrics
    scrape_interval: 10s

  # Node Exporter (System Metrics)
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  # GPU Metrics (if available)
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['gpu-exporter:9445']
```

#### Alert Rules Configuration
Create `prometheus/alert_rules.yml`:
```yaml
groups:
  - name: ml_model_alerts
    rules:
      - alert: HighModelLatency
        expr: histogram_quantile(0.95, rate(ml_model_inference_duration_seconds_bucket[5m])) > 1.0
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "High model inference latency"
          description: "95th percentile latency is {{ $value }}s"

      - alert: ModelAccuracyDrop
        expr: ml_model_accuracy < 0.85
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy below threshold"
          description: "Model accuracy is {{ $value }}"

      - alert: HighErrorRate
        expr: rate(ml_model_predictions_total{status="error"}[5m]) / rate(ml_model_predictions_total[5m]) > 0.05
        for: 3m
        labels:
          severity: warning
        annotations:
          summary: "High model error rate"
          description: "Error rate is {{ $value | humanizePercentage }}"
```

### 3. Set Up Grafana

#### Grafana Configuration
Create `grafana/grafana.ini`:
```ini
[server]
http_port = 3000
domain = localhost

[security]
admin_user = admin
admin_password = admin123

[database]
type = sqlite3
path = grafana.db

[analytics]
reporting_enabled = false

[dashboards]
default_home_dashboard_path = /var/lib/grafana/dashboards/ml-overview.json
```

#### Dashboard Configuration
Create `grafana/dashboards/ml-overview.json`:
```json
{
  "dashboard": {
    "title": "ML System Overview",
    "panels": [
      {
        "title": "Model Prediction Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_model_predictions_total[5m])",
            "legendFormat": "{{model_name}}"
          }
        ]
      },
      {
        "title": "Model Accuracy",
        "type": "singlestat",
        "targets": [
          {
            "expr": "ml_model_accuracy",
            "legendFormat": "{{model_name}}"
          }
        ]
      }
    ]
  }
}
```

### 4. Set Up Jaeger (Distributed Tracing)

#### Jaeger Configuration
Create `jaeger/jaeger.yml`:
```yaml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:1.45
    ports:
      - "16686:16686"  # Jaeger UI
      - "14268:14268"  # Jaeger collector
      - "6831:6831/udp"  # Jaeger agent
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411
      - SPAN_STORAGE_TYPE=elasticsearch
      - ES_SERVER_URLS=http://elasticsearch:9200
```

### 5. Set Up ELK Stack (Logging)

#### Elasticsearch Configuration
Create `elk/elasticsearch.yml`:
```yaml
cluster.name: ml-logging-cluster
node.name: ml-log-node-1
path.data: /usr/share/elasticsearch/data
path.logs: /usr/share/elasticsearch/logs
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node
xpack.security.enabled: false
```

#### Logstash Configuration
Create `elk/logstash.conf`:
```ruby
input {
  beats {
    port => 5044
  }
  
  http {
    port => 8080
    codec => json
  }
}

filter {
  if [message] =~ /^{.*}$/ {
    json {
      source => "message"
    }
  }
  
  # Add computed fields for ML logs
  if [context][latency_ms] {
    ruby {
      code => "
        latency = event.get('[context][latency_ms]')
        if latency > 1000
          event.set('[performance_category]', 'slow')
        elsif latency > 500
          event.set('[performance_category]', 'medium')
        else
          event.set('[performance_category]', 'fast')
        end
      "
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "ml-logs-%{+YYYY.MM.dd}"
  }
}
```

#### Kibana Configuration
Create `elk/kibana.yml`:
```yaml
server.name: kibana
server.host: 0.0.0.0
elasticsearch.hosts: ["http://elasticsearch:9200"]
monitoring.ui.container.elasticsearch.enabled: true
```

### 6. Complete Docker Compose Setup

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  # Prometheus
  prometheus:
    image: prom/prometheus:v2.45.0
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus:/etc/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'

  # Grafana
  grafana:
    image: grafana/grafana:10.0.0
    ports:
      - "3000:3000"
    volumes:
      - ./grafana:/etc/grafana
      - ./grafana/dashboards:/var/lib/grafana/dashboards
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin123

  # Alertmanager
  alertmanager:
    image: prom/alertmanager:v0.25.0
    ports:
      - "9093:9093"
    volumes:
      - ./alertmanager:/etc/alertmanager

  # Jaeger
  jaeger:
    image: jaegertracing/all-in-one:1.45
    ports:
      - "16686:16686"
      - "14268:14268"
      - "6831:6831/udp"
    environment:
      - COLLECTOR_ZIPKIN_HOST_PORT=:9411

  # Elasticsearch
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data

  # Logstash
  logstash:
    image: docker.elastic.co/logstash/logstash:8.11.0
    ports:
      - "5044:5044"
      - "8080:8080"
    volumes:
      - ./elk/logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    depends_on:
      - elasticsearch

  # Kibana
  kibana:
    image: docker.elastic.co/kibana/kibana:8.11.0
    ports:
      - "5601:5601"
    environment:
      - ELASTICSEARCH_HOSTS=http://elasticsearch:9200
    depends_on:
      - elasticsearch

  # Node Exporter (System Metrics)
  node-exporter:
    image: prom/node-exporter:v1.6.0
    ports:
      - "9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - '--path.procfs=/host/proc'
      - '--path.rootfs=/rootfs'
      - '--path.sysfs=/host/sys'
      - '--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)'

volumes:
  elasticsearch_data:
```

### 7. Start the Observability Stack

```bash
# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs if needed
docker-compose logs prometheus
docker-compose logs grafana
docker-compose logs jaeger
```

### 8. Configure Service Discovery

#### For Kubernetes Deployment
Create `k8s/prometheus-config.yaml`:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    
    scrape_configs:
      - job_name: 'kubernetes-apiservers'
        kubernetes_sd_configs:
          - role: endpoints
        scheme: https
        tls_config:
          ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
        
      - job_name: 'kubernetes-pods'
        kubernetes_sd_configs:
          - role: pod
        relabel_configs:
          - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
            action: keep
            regex: true
```

## Configuration and Testing

### 1. Access Web Interfaces

After starting the stack, access these URLs:

- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin123)
- **Jaeger**: http://localhost:16686
- **Kibana**: http://localhost:5601
- **Alertmanager**: http://localhost:9093

### 2. Test Metrics Collection

Create `test_metrics.py`:
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import random

# Create metrics
predictions_total = Counter('ml_model_predictions_total', 'Total predictions', ['model', 'status'])
inference_duration = Histogram('ml_model_inference_duration_seconds', 'Inference time', ['model'])
model_accuracy = Gauge('ml_model_accuracy', 'Model accuracy', ['model'])

# Start metrics server
start_http_server(8000)

# Simulate metrics
while True:
    model = random.choice(['fraud_detector', 'recommender'])
    status = random.choice(['success', 'error'])
    latency = random.uniform(0.1, 2.0)
    accuracy = random.uniform(0.8, 0.95)
    
    predictions_total.labels(model=model, status=status).inc()
    inference_duration.labels(model=model).observe(latency)
    model_accuracy.labels(model=model).set(accuracy)
    
    time.sleep(1)
```

Run the test:
```bash
python test_metrics.py
```

### 3. Test Distributed Tracing

Create `test_tracing.py`:
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
import time

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Create traces
def simulate_ml_pipeline():
    with tracer.start_as_current_span("ml_pipeline") as span:
        span.set_attribute("pipeline.name", "fraud_detection")
        
        with tracer.start_as_current_span("data_preprocessing"):
            time.sleep(0.1)
        
        with tracer.start_as_current_span("model_inference"):
            time.sleep(0.2)
        
        with tracer.start_as_current_span("post_processing"):
            time.sleep(0.05)

# Run simulation
for i in range(10):
    simulate_ml_pipeline()
    time.sleep(1)
```

### 4. Test Structured Logging

Create `test_logging.py`:
```python
import json
import logging
from datetime import datetime

# Configure structured logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def log_ml_event(event_type, **kwargs):
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "service": "ml-test-service",
        "event_type": event_type,
        "context": kwargs
    }
    logger.info(json.dumps(log_entry))

# Test logging
log_ml_event("model_prediction", 
             model="fraud_detector", 
             prediction="legitimate", 
             confidence=0.95, 
             latency_ms=150)

log_ml_event("data_quality_issue", 
             pipeline="user_features", 
             issue_type="missing_values", 
             affected_records=1500)
```

## Validation and Troubleshooting

### 1. Health Checks

Create `health_check.py`:
```python
import requests
import sys

def check_service(name, url):
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ {name}: OK")
            return True
        else:
            print(f"❌ {name}: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ {name}: {e}")
        return False

# Check all services
services = {
    "Prometheus": "http://localhost:9090/-/healthy",
    "Grafana": "http://localhost:3000/api/health",
    "Jaeger": "http://localhost:16686/",
    "Elasticsearch": "http://localhost:9200/_cluster/health",
    "Kibana": "http://localhost:5601/api/status"
}

all_healthy = True
for name, url in services.items():
    if not check_service(name, url):
        all_healthy = False

sys.exit(0 if all_healthy else 1)
```

### 2. Common Issues and Solutions

#### Elasticsearch Memory Issues
```bash
# Increase virtual memory
sudo sysctl -w vm.max_map_count=262144

# Make permanent
echo 'vm.max_map_count=262144' | sudo tee -a /etc/sysctl.conf
```

#### Docker Resource Limits
```bash
# Check Docker resources
docker system df
docker system prune

# Increase Docker memory (Docker Desktop)
# Settings > Resources > Memory: 8GB+
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :9090
lsof -i :3000

# Stop conflicting services
sudo systemctl stop apache2  # If using port 80
```

### 3. Performance Tuning

#### Prometheus Configuration
```yaml
# prometheus.yml - Performance settings
global:
  scrape_interval: 15s
  scrape_timeout: 10s
  evaluation_interval: 15s

# Retention and storage
storage:
  tsdb:
    retention.time: 30d
    retention.size: 10GB
```

#### Elasticsearch Configuration
```yaml
# elasticsearch.yml - Performance settings
indices.memory.index_buffer_size: 30%
indices.memory.min_index_buffer_size: 96mb
thread_pool.write.queue_size: 1000
```

## Production Deployment

### 1. Kubernetes Deployment

Use Helm charts for production deployment:

```bash
# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
helm repo add elastic https://helm.elastic.co

# Install Prometheus stack
helm install prometheus prometheus-community/kube-prometheus-stack

# Install Jaeger
helm install jaeger jaegertracing/jaeger

# Install ELK stack
helm install elasticsearch elastic/elasticsearch
helm install kibana elastic/kibana
```

### 2. Security Configuration

#### Enable Authentication
```yaml
# grafana.ini
[auth]
disable_login_form = false

[auth.basic]
enabled = true

[auth.ldap]
enabled = true
config_file = /etc/grafana/ldap.toml
```

#### TLS Configuration
```yaml
# prometheus.yml
tls_config:
  cert_file: /etc/prometheus/certs/prometheus.crt
  key_file: /etc/prometheus/certs/prometheus.key
```

### 3. Backup and Recovery

```bash
# Backup Prometheus data
docker exec prometheus tar -czf /tmp/prometheus-backup.tar.gz /prometheus

# Backup Grafana dashboards
curl -H "Authorization: Bearer $GRAFANA_API_KEY" \
     http://localhost:3000/api/search?type=dash-db | \
     jq -r '.[] | .uri' > dashboards.txt
```

## Next Steps

1. **Complete the exercises** in `exercise.py`
2. **Import sample dashboards** from Grafana community
3. **Set up alerting rules** for your specific ML models
4. **Configure log shipping** from your applications
5. **Implement custom metrics** for business KPIs
6. **Set up automated backup** procedures
7. **Create runbooks** for alert response

## Resources

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Elastic Stack Documentation](https://www.elastic.co/guide/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [ML Observability Best Practices](https://ml-ops.org/content/monitoring)