"""
Day 58: Monitoring & Observability - Production Systems
Exercises for building comprehensive monitoring and observability for ML systems
"""

import time
import json
import random
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Represents a single metric data point"""
    timestamp: float
    value: float
    labels: Dict[str, str]


@dataclass
class LogEntry:
    """Represents a structured log entry"""
    timestamp: str
    level: str
    service: str
    message: str
    context: Dict[str, Any]


class MockPrometheusClient:
    """Mock Prometheus client for exercises"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
    
    def counter_inc(self, metric_name: str, labels: Dict[str, str] = None, value: float = 1.0):
        """Increment counter metric"""
        labels = labels or {}
        key = f"{metric_name}_{hash(frozenset(labels.items()))}"
        self.counters[key] += value
        
        self.metrics[metric_name].append(MetricPoint(
            timestamp=time.time(),
            value=self.counters[key],
            labels=labels
        ))
    
    def gauge_set(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge metric value"""
        labels = labels or {}
        key = f"{metric_name}_{hash(frozenset(labels.items()))}"
        self.gauges[key] = value
        
        self.metrics[metric_name].append(MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels
        ))
    
    def histogram_observe(self, metric_name: str, value: float, labels: Dict[str, str] = None):
        """Observe histogram metric value"""
        labels = labels or {}
        key = f"{metric_name}_{hash(frozenset(labels.items()))}"
        self.histograms[key].append(value)
        
        self.metrics[metric_name].append(MetricPoint(
            timestamp=time.time(),
            value=value,
            labels=labels
        ))
    
    def get_metric_values(self, metric_name: str) -> List[MetricPoint]:
        """Get all values for a metric"""
        return self.metrics.get(metric_name, [])
    
    def query(self, query: str) -> List[Dict]:
        """Simulate Prometheus query"""
        # Simple mock implementation
        if "rate(" in query:
            return [{"metric": {"__name__": "rate_result"}, "value": [time.time(), "10.5"]}]
        elif "histogram_quantile(" in query:
            return [{"metric": {"__name__": "quantile_result"}, "value": [time.time(), "0.25"]}]
        else:
            return [{"metric": {"__name__": "mock_result"}, "value": [time.time(), "1.0"]}]


# Exercise 1: ML Model Metrics
def exercise_1_ml_model_metrics():
    """
    Exercise 1: Implement comprehensive ML model metrics
    
    TODO: Complete the MLModelMetrics class
    """
    print("=== Exercise 1: ML Model Metrics ===")
    
    class MLModelMetrics:
        def __init__(self, prometheus_client: MockPrometheusClient):
            self.prometheus = prometheus_client
            self.model_stats = defaultdict(dict)
        
        def record_prediction(self, model_name: str, model_version: str, 
                            prediction_time: float, status: str = "success"):
            """Record model prediction metrics"""
            # Increment prediction counter with labels
            labels = {"model_name": model_name, "model_version": model_version, "status": status}
            self.prometheus.counter_inc("ml_model_predictions_total", labels)
            
            # Record prediction latency in histogram
            latency_labels = {"model_name": model_name, "model_version": model_version}
            self.prometheus.histogram_observe("ml_model_inference_duration_seconds", prediction_time, latency_labels)
            
            # Update model statistics
            key = f"{model_name}_{model_version}"
            if key not in self.model_stats:
                self.model_stats[key] = {"total_predictions": 0, "errors": 0, "avg_latency": 0.0}
            
            self.model_stats[key]["total_predictions"] += 1
            if status == "error":
                self.model_stats[key]["errors"] += 1
            
            # Update average latency
            current_avg = self.model_stats[key]["avg_latency"]
            total = self.model_stats[key]["total_predictions"]
            self.model_stats[key]["avg_latency"] = ((current_avg * (total - 1)) + prediction_time) / total
        
        def update_model_accuracy(self, model_name: str, model_version: str, accuracy: float):
            """Update model accuracy gauge"""
            # Set gauge metric for accuracy
            labels = {"model_name": model_name, "model_version": model_version}
            self.prometheus.gauge_set("ml_model_accuracy", accuracy, labels)
            
            # Store in model statistics
            key = f"{model_name}_{model_version}"
            if key not in self.model_stats:
                self.model_stats[key] = {}
            self.model_stats[key]["accuracy"] = accuracy
            
            # Check if accuracy drops below threshold
            if accuracy < 0.85:
                logger.warning(f"Model accuracy below threshold: {model_name} v{model_version} - {accuracy:.3f}")
        
        def record_data_drift(self, model_name: str, feature_name: str, drift_score: float):
            """Record data drift metrics"""
            # Set gauge for drift score
            labels = {"model_name": model_name, "feature_name": feature_name}
            self.prometheus.gauge_set("ml_data_drift_score", drift_score, labels)
            
            # Track drift trends over time
            key = f"{model_name}_{feature_name}"
            if key not in self.model_stats:
                self.model_stats[key] = {"drift_history": []}
            
            self.model_stats[key]["drift_history"].append({
                "timestamp": time.time(),
                "drift_score": drift_score
            })
            
            # Keep only last 100 drift measurements
            if len(self.model_stats[key]["drift_history"]) > 100:
                self.model_stats[key]["drift_history"] = self.model_stats[key]["drift_history"][-100:]
            
            # Alert if drift exceeds threshold
            if drift_score > 0.7:
                logger.warning(f"High data drift detected: {model_name}.{feature_name} - {drift_score:.3f}")
        
        def get_model_health_summary(self, model_name: str) -> Dict[str, Any]:
            """Generate model health summary"""
            summary = {
                "model_name": model_name,
                "timestamp": datetime.now().isoformat(),
                "health_score": 0.0,
                "metrics": {},
                "alerts": [],
                "recommendations": []
            }
            
            # Aggregate metrics for model
            model_keys = [k for k in self.model_stats.keys() if k.startswith(model_name)]
            
            if not model_keys:
                summary["alerts"].append("No metrics found for model")
                return summary
            
            # Calculate aggregated metrics
            total_predictions = sum(self.model_stats[k].get("total_predictions", 0) for k in model_keys)
            total_errors = sum(self.model_stats[k].get("errors", 0) for k in model_keys)
            
            if total_predictions > 0:
                error_rate = total_errors / total_predictions
                summary["metrics"]["error_rate"] = error_rate
                summary["metrics"]["total_predictions"] = total_predictions
                
                if error_rate > 0.05:
                    summary["alerts"].append(f"High error rate: {error_rate:.2%}")
                    summary["recommendations"].append("Investigate model performance issues")
            
            # Check accuracy
            accuracies = [self.model_stats[k].get("accuracy") for k in model_keys if "accuracy" in self.model_stats[k]]
            if accuracies:
                avg_accuracy = sum(accuracies) / len(accuracies)
                summary["metrics"]["accuracy"] = avg_accuracy
                
                if avg_accuracy < 0.85:
                    summary["alerts"].append(f"Low accuracy: {avg_accuracy:.3f}")
                    summary["recommendations"].append("Consider model retraining")
            
            # Calculate health score (0-1)
            health_factors = []
            if "error_rate" in summary["metrics"]:
                health_factors.append(max(0, 1 - (summary["metrics"]["error_rate"] * 10)))
            if "accuracy" in summary["metrics"]:
                health_factors.append(summary["metrics"]["accuracy"])
            
            if health_factors:
                summary["health_score"] = sum(health_factors) / len(health_factors)
            
            return summary
    
    # Test ML model metrics
    prometheus_client = MockPrometheusClient()
    ml_metrics = MLModelMetrics(prometheus_client)
    
    print("Testing ML Model Metrics...")
    print("\n--- Your implementation should track comprehensive model metrics ---")
    
    # Example model serving simulation
    models = [
        ("fraud_detection", "v2.1.0"),
        ("recommendation_engine", "v1.5.2"),
        ("text_classifier", "v3.0.1")
    ]
    
    print("Model metrics simulation completed")


# Exercise 2: Structured Logging
def exercise_2_structured_logging():
    """
    Exercise 2: Implement structured logging for ML systems
    
    TODO: Complete the MLStructuredLogger class
    """
    print("\n=== Exercise 2: Structured Logging ===")
    
    class MLStructuredLogger:
        def __init__(self, service_name: str, environment: str):
            self.service_name = service_name
            self.environment = environment
            self.log_entries = []
        
        def log_model_prediction(self, model_name: str, model_version: str,
                               input_features: Dict, prediction: Any,
                               confidence: float, latency_ms: float):
            """Log model prediction with structured context"""
            # Create structured log entry with all context
            log_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level="INFO",
                service=self.service_name,
                message=f"Model prediction completed: {model_name}",
                context={
                    "event_type": "model_prediction",
                    "model_name": model_name,
                    "model_version": model_version,
                    "prediction": str(prediction),
                    "confidence": confidence,
                    "latency_ms": latency_ms,
                    "input_feature_count": len(input_features),
                    "input_features_hash": hash(str(sorted(input_features.items()))),
                    "environment": self.environment
                }
            )
            
            # Store log entry and format as JSON
            self.log_entries.append(log_entry)
            json_log = json.dumps({
                "timestamp": log_entry.timestamp,
                "level": log_entry.level,
                "service": log_entry.service,
                "message": log_entry.message,
                "context": log_entry.context
            })
            
            logger.info(json_log)
        
        def log_data_quality_issue(self, pipeline_name: str, issue_type: str,
                                  affected_records: int, severity: str):
            """Log data quality issues"""
            # Determine log level based on severity
            level = "ERROR" if severity == "high" else "WARNING" if severity == "medium" else "INFO"
            
            # Structure log with data quality context
            log_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level=level,
                service=self.service_name,
                message=f"Data quality issue detected in {pipeline_name}: {issue_type}",
                context={
                    "event_type": "data_quality_issue",
                    "pipeline_name": pipeline_name,
                    "issue_type": issue_type,
                    "affected_records": affected_records,
                    "severity": severity,
                    "impact_percentage": (affected_records / 10000) * 100,  # Assuming 10k total records
                    "environment": self.environment,
                    "requires_attention": severity in ["high", "medium"]
                }
            )
            
            # Store log entry and format as JSON
            self.log_entries.append(log_entry)
            json_log = json.dumps({
                "timestamp": log_entry.timestamp,
                "level": log_entry.level,
                "service": log_entry.service,
                "message": log_entry.message,
                "context": log_entry.context
            })
            
            # Log with appropriate level
            if severity == "high":
                logger.error(json_log)
            elif severity == "medium":
                logger.warning(json_log)
            else:
                logger.info(json_log)
        
        def export_logs_for_analysis(self) -> List[Dict]:
            """Export logs in format suitable for analysis"""
            exported_logs = []
            
            for log_entry in self.log_entries:
                # Convert log entries to structured format for ELK stack
                structured_log = {
                    "@timestamp": log_entry.timestamp,
                    "level": log_entry.level,
                    "service": log_entry.service,
                    "environment": self.environment,
                    "message": log_entry.message,
                    "fields": log_entry.context,
                    "correlation_id": f"{self.service_name}_{hash(log_entry.timestamp)}",
                    "source": {
                        "service": self.service_name,
                        "environment": self.environment,
                        "version": "1.0.0"
                    },
                    "tags": [
                        f"service:{self.service_name}",
                        f"environment:{self.environment}",
                        f"level:{log_entry.level.lower()}"
                    ]
                }
                
                # Add event-specific tags
                if "event_type" in log_entry.context:
                    structured_log["tags"].append(f"event_type:{log_entry.context['event_type']}")
                
                # Add severity tag for data quality issues
                if "severity" in log_entry.context:
                    structured_log["tags"].append(f"severity:{log_entry.context['severity']}")
                
                exported_logs.append(structured_log)
            
            return exported_logs
    
    # Test structured logging
    ml_logger = MLStructuredLogger("ml-prediction-service", "production")
    
    print("Testing Structured Logging...")
    print("\n--- Your implementation should create structured, searchable logs ---")
    
    print("Structured logging simulation completed")


def main():
    """Run all monitoring and observability exercises"""
    print("📊 Day 58: Monitoring & Observability - Production Systems")
    print("=" * 70)
    
    exercises = [
        exercise_1_ml_model_metrics,
        exercise_2_structured_logging
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n📋 Starting Exercise {i}")
        try:
            exercise()
            print(f"✅ Exercise {i} setup complete")
        except Exception as e:
            print(f"❌ Exercise {i} error: {e}")
        
        if i < len(exercises):
            input("\nPress Enter to continue to the next exercise...")
    
    print("\n🎉 All exercises completed!")
    print("\nNext steps:")
    print("1. Implement the TODO sections in each exercise")
    print("2. Set up real Prometheus, Grafana, and Jaeger instances")
    print("3. Deploy monitoring to actual ML systems")
    print("4. Review the solution file for complete implementations")
    print("5. Experiment with advanced observability patterns")
    
    print("\n🚀 Production Deployment Checklist:")
    print("• Set up comprehensive metrics collection")
    print("• Implement structured logging with log aggregation")
    print("• Deploy distributed tracing for complex pipelines")
    print("• Define and monitor SLIs/SLOs")
    print("• Configure intelligent alerting with runbooks")
    print("• Create business-focused observability dashboards")


if __name__ == "__main__":
    main()