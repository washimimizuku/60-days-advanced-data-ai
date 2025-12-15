# Day 58: Monitoring & Observability - Production Systems

## Learning Objectives
By the end of this session, you will be able to:
- Design and implement comprehensive monitoring and observability for ML and data systems
- Deploy and configure Prometheus, Grafana, and distributed tracing solutions
- Create custom metrics, alerts, and dashboards for ML workloads and data pipelines
- Implement SLI/SLO frameworks and error budgets for production ML systems
- Build end-to-end observability with logs, metrics, traces, and business intelligence

## Theory (15 minutes)

### Monitoring & Observability for ML Systems

Monitoring and observability are critical for maintaining reliable, performant ML and data systems in production. While monitoring tells you what's happening, observability helps you understand why it's happening, enabling faster debugging and better system understanding.

### The Three Pillars of Observability

#### 1. Metrics - Quantitative Measurements

**System Metrics**
```yaml
# Prometheus configuration for ML system metrics
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "ml_alerts.yml"
  - "data_pipeline_alerts.yml"

scrape_configs:
  # ML Model Serving Metrics
  - job_name: 'ml-model-serving'
    static_configs:
      - targets: ['model-server:8080']
    metrics_path: /metrics
    scrape_interval: 10s
    
  # Data Pipeline Metrics
  - job_name: 'airflow-metrics'
    static_configs:
      - targets: ['airflow-webserver:8080']
    metrics_path: /admin/metrics
    
  # Kubernetes Cluster Metrics
  - job_name: 'kubernetes-apiservers'
    kubernetes_sd_configs:
      - role: endpoints
    scheme: https
    tls_config:
      ca_file: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
    bearer_token_file: /var/run/secrets/kubernetes.io/serviceaccount/token
    
  # GPU Metrics for ML Training
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['gpu-exporter:9445']
```

**Custom ML Metrics**
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import numpy as np

# ML Model Performance Metrics
model_predictions_total = Counter(
    'ml_model_predictions_total',
    'Total number of model predictions',
    ['model_name', 'model_version', 'status']
)

model_inference_duration = Histogram(
    'ml_model_inference_duration_seconds',
    'Time spent on model inference',
    ['model_name', 'model_version'],
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

model_accuracy = Gauge(
    'ml_model_accuracy',
    'Current model accuracy',
    ['model_name', 'model_version']
)

data_drift_score = Gauge(
    'ml_data_drift_score',
    'Data drift detection score',
    ['model_name', 'feature_name']
)

# Business Metrics
revenue_impact = Counter(
    'ml_revenue_impact_total',
    'Revenue impact from ML predictions',
    ['model_name', 'prediction_type']
)

class MLModelMonitor:
    def __init__(self, model_name: str, model_version: str):
        self.model_name = model_name
        self.model_version = model_version
        
    def record_prediction(self, prediction_time: float, status: str = 'success'):
        """Record model prediction metrics"""
        model_predictions_total.labels(
            model_name=self.model_name,
            model_version=self.model_version,
            status=status
        ).inc()
        
        model_inference_duration.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).observe(prediction_time)
    
    def update_accuracy(self, accuracy: float):
        """Update model accuracy metric"""
        model_accuracy.labels(
            model_name=self.model_name,
            model_version=self.model_version
        ).set(accuracy)
    
    def record_drift(self, feature_name: str, drift_score: float):
        """Record data drift metrics"""
        data_drift_score.labels(
            model_name=self.model_name,
            feature_name=feature_name
        ).set(drift_score)

# Example usage
monitor = MLModelMonitor("text_classifier", "v1.2.0")

# Simulate model serving with metrics
def serve_prediction(input_data):
    start_time = time.time()
    
    try:
        # Simulate model inference
        prediction = np.random.choice(['positive', 'negative'])
        inference_time = time.time() - start_time
        
        # Record successful prediction
        monitor.record_prediction(inference_time, 'success')
        
        return prediction
        
    except Exception as e:
        inference_time = time.time() - start_time
        monitor.record_prediction(inference_time, 'error')
        raise e
```

#### 2. Logs - Contextual Information

**Structured Logging for ML Systems**
```python
import logging
import json
from datetime import datetime
from typing import Dict, Any

class MLStructuredLogger:
    def __init__(self, service_name: str, environment: str):
        self.service_name = service_name
        self.environment = environment
        
        # Configure structured logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s'
        )
        self.logger = logging.getLogger(service_name)
    
    def _create_log_entry(self, level: str, message: str, 
                         context: Dict[str, Any] = None) -> str:
        """Create structured log entry"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "service": self.service_name,
            "environment": self.environment,
            "level": level,
            "message": message,
            "context": context or {}
        }
        return json.dumps(log_entry)
    
    def log_model_prediction(self, model_name: str, model_version: str,
                           input_features: Dict, prediction: Any,
                           confidence: float, latency_ms: float):
        """Log model prediction with context"""
        context = {
            "model_name": model_name,
            "model_version": model_version,
            "prediction": str(prediction),
            "confidence": confidence,
            "latency_ms": latency_ms,
            "input_feature_count": len(input_features),
            "event_type": "model_prediction"
        }
        
        log_entry = self._create_log_entry(
            "INFO", 
            f"Model prediction completed: {model_name}",
            context
        )
        self.logger.info(log_entry)
    
    def log_data_quality_issue(self, pipeline_name: str, issue_type: str,
                              affected_records: int, severity: str):
        """Log data quality issues"""
        context = {
            "pipeline_name": pipeline_name,
            "issue_type": issue_type,
            "affected_records": affected_records,
            "severity": severity,
            "event_type": "data_quality_issue"
        }
        
        log_entry = self._create_log_entry(
            "WARNING" if severity == "medium" else "ERROR",
            f"Data quality issue detected in {pipeline_name}",
            context
        )
        
        if severity == "high":
            self.logger.error(log_entry)
        else:
            self.logger.warning(log_entry)
    
    def log_model_drift(self, model_name: str, drift_type: str,
                       drift_score: float, threshold: float):
        """Log model drift detection"""
        context = {
            "model_name": model_name,
            "drift_type": drift_type,
            "drift_score": drift_score,
            "threshold": threshold,
            "drift_detected": drift_score > threshold,
            "event_type": "model_drift"
        }
        
        level = "WARNING" if drift_score > threshold else "INFO"
        message = f"Model drift {'detected' if drift_score > threshold else 'monitored'}: {model_name}"
        
        log_entry = self._create_log_entry(level, message, context)
        
        if drift_score > threshold:
            self.logger.warning(log_entry)
        else:
            self.logger.info(log_entry)

# Example usage
ml_logger = MLStructuredLogger("ml-prediction-service", "production")

# Log model prediction
ml_logger.log_model_prediction(
    model_name="fraud_detection",
    model_version="v2.1.0",
    input_features={"transaction_amount": 150.0, "merchant_category": "retail"},
    prediction="legitimate",
    confidence=0.95,
    latency_ms=45.2
)
```

#### 3. Traces - Request Flow Tracking

**Distributed Tracing with OpenTelemetry**
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
import time

# Configure OpenTelemetry
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# Configure Jaeger exporter
jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-agent",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Auto-instrument common libraries
RequestsInstrumentor().instrument()
SQLAlchemyInstrumentor().instrument()

class MLPipelineTracer:
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.tracer = trace.get_tracer(pipeline_name)
    
    def trace_data_ingestion(self, source: str, record_count: int):
        """Trace data ingestion process"""
        with self.tracer.start_as_current_span("data_ingestion") as span:
            span.set_attribute("pipeline.name", self.pipeline_name)
            span.set_attribute("data.source", source)
            span.set_attribute("data.record_count", record_count)
            
            # Simulate data ingestion
            time.sleep(0.1)
            
            span.set_attribute("ingestion.status", "completed")
            return f"Ingested {record_count} records from {source}"
    
    def trace_feature_engineering(self, features: list):
        """Trace feature engineering process"""
        with self.tracer.start_as_current_span("feature_engineering") as span:
            span.set_attribute("pipeline.name", self.pipeline_name)
            span.set_attribute("features.count", len(features))
            span.set_attribute("features.names", ",".join(features))
            
            # Simulate feature engineering steps
            for i, feature in enumerate(features):
                with self.tracer.start_as_current_span(f"process_feature_{feature}") as feature_span:
                    feature_span.set_attribute("feature.name", feature)
                    feature_span.set_attribute("feature.index", i)
                    time.sleep(0.05)  # Simulate processing time
            
            span.set_attribute("engineering.status", "completed")
            return f"Processed {len(features)} features"
    
    def trace_model_inference(self, model_name: str, input_size: int):
        """Trace model inference process"""
        with self.tracer.start_as_current_span("model_inference") as span:
            span.set_attribute("model.name", model_name)
            span.set_attribute("model.input_size", input_size)
            
            # Simulate model loading
            with self.tracer.start_as_current_span("model_loading") as load_span:
                load_span.set_attribute("model.name", model_name)
                time.sleep(0.2)  # Simulate model loading time
            
            # Simulate inference
            with self.tracer.start_as_current_span("inference_execution") as inference_span:
                inference_span.set_attribute("input.size", input_size)
                time.sleep(0.1)  # Simulate inference time
                
                prediction = "positive"  # Simulate prediction
                confidence = 0.87
                
                inference_span.set_attribute("prediction.result", prediction)
                inference_span.set_attribute("prediction.confidence", confidence)
            
            span.set_attribute("inference.status", "completed")
            return {"prediction": prediction, "confidence": confidence}

# Example usage
pipeline_tracer = MLPipelineTracer("fraud_detection_pipeline")

# Trace complete ML pipeline
def run_ml_pipeline():
    with tracer.start_as_current_span("ml_pipeline_execution") as root_span:
        root_span.set_attribute("pipeline.name", "fraud_detection_pipeline")
        root_span.set_attribute("pipeline.version", "v1.0.0")
        
        # Trace each step
        ingestion_result = pipeline_tracer.trace_data_ingestion("kafka_stream", 1000)
        features_result = pipeline_tracer.trace_feature_engineering(
            ["transaction_amount", "merchant_category", "time_of_day"]
        )
        inference_result = pipeline_tracer.trace_model_inference("fraud_model_v2", 1000)
        
        root_span.set_attribute("pipeline.status", "completed")
        return inference_result
```

### Prometheus Configuration and Alerting

#### 1. Prometheus Rules and Alerts

**ML System Alert Rules**
```yaml
# ml_alerts.yml
groups:
  - name: ml_model_alerts
    rules:
      # High Model Latency Alert
      - alert: HighModelLatency
        expr: histogram_quantile(0.95, rate(ml_model_inference_duration_seconds_bucket[5m])) > 1.0
        for: 2m
        labels:
          severity: warning
          team: ml-platform
        annotations:
          summary: "High model inference latency detected"
          description: "95th percentile latency for {{ $labels.model_name }} is {{ $value }}s"
          runbook_url: "https://runbooks.company.com/ml-latency"
      
      # Model Accuracy Drop Alert
      - alert: ModelAccuracyDrop
        expr: ml_model_accuracy < 0.85
        for: 5m
        labels:
          severity: critical
          team: ml-platform
        annotations:
          summary: "Model accuracy dropped below threshold"
          description: "Model {{ $labels.model_name }} accuracy is {{ $value }}, below 0.85 threshold"
          runbook_url: "https://runbooks.company.com/model-accuracy"
      
      # High Error Rate Alert
      - alert: HighModelErrorRate
        expr: rate(ml_model_predictions_total{status="error"}[5m]) / rate(ml_model_predictions_total[5m]) > 0.05
        for: 3m
        labels:
          severity: warning
          team: ml-platform
        annotations:
          summary: "High model error rate detected"
          description: "Error rate for {{ $labels.model_name }} is {{ $value | humanizePercentage }}"
      
      # Data Drift Alert
      - alert: DataDriftDetected
        expr: ml_data_drift_score > 0.7
        for: 1m
        labels:
          severity: warning
          team: data-science
        annotations:
          summary: "Data drift detected"
          description: "Data drift score for {{ $labels.model_name }}.{{ $labels.feature_name }} is {{ $value }}"
          runbook_url: "https://runbooks.company.com/data-drift"

  - name: data_pipeline_alerts
    rules:
      # Pipeline Failure Alert
      - alert: DataPipelineFailure
        expr: airflow_dag_run_failed_total > 0
        for: 1m
        labels:
          severity: critical
          team: data-engineering
        annotations:
          summary: "Data pipeline failure detected"
          description: "Pipeline {{ $labels.dag_id }} has failed"
          runbook_url: "https://runbooks.company.com/pipeline-failure"
      
      # Long Running Pipeline Alert
      - alert: LongRunningPipeline
        expr: time() - airflow_dag_run_start_time > 3600
        for: 5m
        labels:
          severity: warning
          team: data-engineering
        annotations:
          summary: "Pipeline running longer than expected"
          description: "Pipeline {{ $labels.dag_id }} has been running for over 1 hour"
      
      # Data Quality Issues Alert
      - alert: DataQualityIssues
        expr: data_quality_failed_checks_total > 5
        for: 2m
        labels:
          severity: warning
          team: data-engineering
        annotations:
          summary: "Multiple data quality checks failing"
          description: "{{ $value }} data quality checks are failing"

  - name: infrastructure_alerts
    rules:
      # High GPU Utilization
      - alert: HighGPUUtilization
        expr: nvidia_gpu_utilization_gpu > 95
        for: 10m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "High GPU utilization detected"
          description: "GPU {{ $labels.gpu }} utilization is {{ $value }}%"
      
      # Low GPU Memory
      - alert: LowGPUMemory
        expr: (nvidia_gpu_memory_free_bytes / nvidia_gpu_memory_total_bytes) * 100 < 10
        for: 5m
        labels:
          severity: critical
          team: infrastructure
        annotations:
          summary: "Low GPU memory available"
          description: "GPU {{ $labels.gpu }} has only {{ $value }}% memory available"
      
      # Kubernetes Pod Crashes
      - alert: PodCrashLooping
        expr: rate(kube_pod_container_status_restarts_total[15m]) > 0
        for: 5m
        labels:
          severity: warning
          team: infrastructure
        annotations:
          summary: "Pod is crash looping"
          description: "Pod {{ $labels.pod }} in namespace {{ $labels.namespace }} is crash looping"
```

#### 2. Grafana Dashboards

**ML Model Performance Dashboard**
```json
{
  "dashboard": {
    "id": null,
    "title": "ML Model Performance Dashboard",
    "tags": ["ml", "monitoring", "production"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Model Prediction Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(ml_model_predictions_total[5m])",
            "legendFormat": "{{model_name}} - {{model_version}}"
          }
        ],
        "yAxes": [
          {
            "label": "Predictions/sec",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "id": 2,
        "title": "Model Inference Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, rate(ml_model_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, rate(ml_model_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.99, rate(ml_model_inference_duration_seconds_bucket[5m]))",
            "legendFormat": "99th percentile"
          }
        ],
        "yAxes": [
          {
            "label": "Latency (seconds)",
            "min": 0
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      },
      {
        "id": 3,
        "title": "Model Accuracy",
        "type": "singlestat",
        "targets": [
          {
            "expr": "ml_model_accuracy",
            "legendFormat": "{{model_name}}"
          }
        ],
        "valueName": "current",
        "format": "percentunit",
        "thresholds": "0.8,0.9",
        "colorBackground": true,
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 0,
          "y": 8
        }
      },
      {
        "id": 4,
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(ml_model_predictions_total{status=\"error\"}[5m]) / rate(ml_model_predictions_total[5m])",
            "legendFormat": "Error Rate"
          }
        ],
        "valueName": "current",
        "format": "percentunit",
        "thresholds": "0.01,0.05",
        "colorBackground": true,
        "gridPos": {
          "h": 4,
          "w": 6,
          "x": 6,
          "y": 8
        }
      },
      {
        "id": 5,
        "title": "Data Drift Scores",
        "type": "graph",
        "targets": [
          {
            "expr": "ml_data_drift_score",
            "legendFormat": "{{model_name}} - {{feature_name}}"
          }
        ],
        "yAxes": [
          {
            "label": "Drift Score",
            "min": 0,
            "max": 1
          }
        ],
        "thresholds": [
          {
            "value": 0.7,
            "colorMode": "critical",
            "op": "gt"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 8
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### SLI/SLO Framework for ML Systems

#### 1. Service Level Indicators (SLIs)

**ML System SLIs**
```python
from dataclasses import dataclass
from typing import Dict, List
import time

@dataclass
class SLI:
    name: str
    description: str
    query: str
    unit: str
    good_threshold: float
    
class MLSystemSLIs:
    """Define SLIs for ML systems"""
    
    def __init__(self):
        self.slis = {
            # Availability SLIs
            "model_availability": SLI(
                name="Model Availability",
                description="Percentage of successful model predictions",
                query="rate(ml_model_predictions_total{status='success'}[5m]) / rate(ml_model_predictions_total[5m])",
                unit="percentage",
                good_threshold=0.999  # 99.9% availability
            ),
            
            # Latency SLIs
            "model_latency_p95": SLI(
                name="Model Latency P95",
                description="95th percentile of model inference latency",
                query="histogram_quantile(0.95, rate(ml_model_inference_duration_seconds_bucket[5m]))",
                unit="seconds",
                good_threshold=0.5  # 500ms P95 latency
            ),
            
            "model_latency_p99": SLI(
                name="Model Latency P99",
                description="99th percentile of model inference latency",
                query="histogram_quantile(0.99, rate(ml_model_inference_duration_seconds_bucket[5m]))",
                unit="seconds",
                good_threshold=1.0  # 1s P99 latency
            ),
            
            # Quality SLIs
            "model_accuracy": SLI(
                name="Model Accuracy",
                description="Current model accuracy on validation set",
                query="ml_model_accuracy",
                unit="percentage",
                good_threshold=0.85  # 85% accuracy
            ),
            
            "data_freshness": SLI(
                name="Data Freshness",
                description="Time since last data update",
                query="time() - data_last_updated_timestamp",
                unit="seconds",
                good_threshold=3600  # 1 hour freshness
            ),
            
            # Throughput SLIs
            "prediction_throughput": SLI(
                name="Prediction Throughput",
                description="Number of predictions per second",
                query="rate(ml_model_predictions_total[5m])",
                unit="requests_per_second",
                good_threshold=100  # 100 RPS minimum
            )
        }
    
    def get_sli_status(self, sli_name: str, current_value: float) -> Dict:
        """Check if SLI meets threshold"""
        sli = self.slis[sli_name]
        
        if sli.unit == "percentage":
            meets_threshold = current_value >= sli.good_threshold
        elif sli.unit == "seconds":
            meets_threshold = current_value <= sli.good_threshold
        else:
            meets_threshold = current_value >= sli.good_threshold
        
        return {
            "sli_name": sli.name,
            "current_value": current_value,
            "threshold": sli.good_threshold,
            "meets_threshold": meets_threshold,
            "unit": sli.unit
        }

# Example SLO definitions
@dataclass
class SLO:
    name: str
    sli_name: str
    target: float  # e.g., 99.9% for availability
    time_window: str  # e.g., "30d" for 30 days
    
class MLSystemSLOs:
    """Define SLOs for ML systems"""
    
    def __init__(self):
        self.slos = {
            "model_availability_slo": SLO(
                name="Model Availability SLO",
                sli_name="model_availability",
                target=0.999,  # 99.9% availability
                time_window="30d"
            ),
            
            "model_latency_slo": SLO(
                name="Model Latency SLO",
                sli_name="model_latency_p95",
                target=0.5,  # 95% of requests under 500ms
                time_window="7d"
            ),
            
            "model_accuracy_slo": SLO(
                name="Model Accuracy SLO",
                sli_name="model_accuracy",
                target=0.85,  # 85% accuracy minimum
                time_window="24h"
            )
        }
    
    def calculate_error_budget(self, slo_name: str, current_performance: float) -> Dict:
        """Calculate error budget for SLO"""
        slo = self.slos[slo_name]
        
        if slo.sli_name == "model_availability":
            # For availability, error budget is (1 - target) * 100
            total_budget = (1 - slo.target) * 100
            used_budget = (1 - current_performance) * 100
        else:
            # For other metrics, calculate based on target
            total_budget = 100 - (slo.target * 100)
            used_budget = max(0, 100 - (current_performance * 100))
        
        remaining_budget = max(0, total_budget - used_budget)
        budget_utilization = (used_budget / total_budget) * 100 if total_budget > 0 else 0
        
        return {
            "slo_name": slo.name,
            "target": slo.target,
            "current_performance": current_performance,
            "total_error_budget": total_budget,
            "used_error_budget": used_budget,
            "remaining_error_budget": remaining_budget,
            "budget_utilization_percent": budget_utilization,
            "time_window": slo.time_window
        }
```

### Log Aggregation and Analysis

#### 1. ELK Stack Configuration

**Elasticsearch Configuration for ML Logs**
```yaml
# elasticsearch.yml
cluster.name: ml-logging-cluster
node.name: ml-log-node-1
path.data: /var/lib/elasticsearch
path.logs: /var/log/elasticsearch
network.host: 0.0.0.0
http.port: 9200
discovery.type: single-node

# Index template for ML logs
PUT _index_template/ml-logs-template
{
  "index_patterns": ["ml-logs-*"],
  "template": {
    "settings": {
      "number_of_shards": 3,
      "number_of_replicas": 1,
      "index.lifecycle.name": "ml-logs-policy",
      "index.lifecycle.rollover_alias": "ml-logs"
    },
    "mappings": {
      "properties": {
        "timestamp": {
          "type": "date",
          "format": "strict_date_optional_time||epoch_millis"
        },
        "service": {
          "type": "keyword"
        },
        "level": {
          "type": "keyword"
        },
        "message": {
          "type": "text",
          "analyzer": "standard"
        },
        "context": {
          "properties": {
            "model_name": {"type": "keyword"},
            "model_version": {"type": "keyword"},
            "prediction": {"type": "keyword"},
            "confidence": {"type": "float"},
            "latency_ms": {"type": "float"},
            "event_type": {"type": "keyword"}
          }
        }
      }
    }
  }
}
```

**Logstash Configuration**
```ruby
# logstash.conf
input {
  beats {
    port => 5044
  }
  
  kafka {
    bootstrap_servers => "kafka:9092"
    topics => ["ml-logs", "data-pipeline-logs"]
    codec => "json"
  }
}

filter {
  # Parse JSON logs
  if [message] =~ /^\{.*\}$/ {
    json {
      source => "message"
    }
  }
  
  # Add computed fields
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
  
  # Geolocate IP addresses if present
  if [client_ip] {
    geoip {
      source => "client_ip"
      target => "geoip"
    }
  }
  
  # Parse user agent if present
  if [user_agent] {
    useragent {
      source => "user_agent"
      target => "ua"
    }
  }
}

output {
  elasticsearch {
    hosts => ["elasticsearch:9200"]
    index => "ml-logs-%{+YYYY.MM.dd}"
  }
  
  # Send alerts to monitoring system
  if [level] == "ERROR" or [level] == "CRITICAL" {
    http {
      url => "http://alertmanager:9093/api/v1/alerts"
      http_method => "post"
      format => "json"
      mapping => {
        "alerts" => [
          {
            "labels" => {
              "alertname" => "LogError",
              "service" => "%{service}",
              "severity" => "%{level}"
            },
            "annotations" => {
              "summary" => "%{message}",
              "description" => "Error in service %{service}: %{message}"
            }
          }
        ]
      }
    }
  }
}
```

### Distributed Tracing Implementation

#### 1. Jaeger Configuration

**Jaeger Deployment**
```yaml
# jaeger-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: jaeger-all-in-one
  labels:
    app: jaeger
spec:
  replicas: 1
  selector:
    matchLabels:
      app: jaeger
  template:
    metadata:
      labels:
        app: jaeger
    spec:
      containers:
      - name: jaeger
        image: jaegertracing/all-in-one:1.45
        ports:
        - containerPort: 16686
          name: ui
        - containerPort: 14268
          name: collector
        - containerPort: 6831
          name: agent-udp
        - containerPort: 6832
          name: agent-binary
        env:
        - name: COLLECTOR_ZIPKIN_HOST_PORT
          value: ":9411"
        - name: SPAN_STORAGE_TYPE
          value: "elasticsearch"
        - name: ES_SERVER_URLS
          value: "http://elasticsearch:9200"
        - name: ES_INDEX_PREFIX
          value: "jaeger"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: jaeger-service
spec:
  selector:
    app: jaeger
  ports:
  - name: ui
    port: 16686
    targetPort: 16686
  - name: collector
    port: 14268
    targetPort: 14268
  - name: agent-udp
    port: 6831
    targetPort: 6831
    protocol: UDP
  - name: agent-binary
    port: 6832
    targetPort: 6832
```

### Why Monitoring & Observability Matters

1. **Proactive Issue Detection**: Identify problems before they impact users
2. **Performance Optimization**: Understand system bottlenecks and optimization opportunities
3. **Reliability Assurance**: Maintain high availability and performance standards
4. **Debugging Efficiency**: Quickly diagnose and resolve issues in complex systems
5. **Business Intelligence**: Understand the business impact of ML systems
6. **Compliance**: Meet regulatory requirements for system monitoring and auditing
7. **Capacity Planning**: Make informed decisions about resource allocation and scaling

### Real-world Use Cases

- **Netflix**: Uses comprehensive observability to maintain 99.99% availability for their recommendation systems
- **Uber**: Monitors ML model performance in real-time to ensure accurate ETAs and pricing
- **Airbnb**: Tracks data quality and model drift to maintain search ranking effectiveness
- **Spotify**: Uses distributed tracing to optimize music recommendation pipeline performance
- **Pinterest**: Monitors visual search model accuracy and latency across global deployments

## Exercise (40 minutes)
Complete the hands-on exercises in `exercise.py` to build production-ready monitoring and observability solutions for ML systems, including Prometheus metrics, Grafana dashboards, distributed tracing, and SLI/SLO frameworks.

## Resources
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [OpenTelemetry Documentation](https://opentelemetry.io/docs/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [ELK Stack Documentation](https://www.elastic.co/guide/)
- [SRE Book - Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)
- [ML Observability Best Practices](https://ml-ops.org/content/monitoring)

## Next Steps
- Complete the monitoring and observability exercises
- Review SLI/SLO framework implementation
- Take the quiz to test your understanding
- Move to Day 59: Cost Optimization