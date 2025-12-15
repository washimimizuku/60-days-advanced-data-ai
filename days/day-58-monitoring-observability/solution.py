"""
Day 58: Monitoring & Observability - Complete Solutions
Production-ready implementations for comprehensive ML system observability
"""

import time
import json
import random
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
from collections import defaultdict
import uuid

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
class Alert:
    """Represents an alert configuration"""
    name: str
    severity: str
    condition: str
    threshold: float
    duration: str
    labels: Dict[str, str]
    annotations: Dict[str, str]


class ProductionMLMetrics:
    """Production-ready ML metrics collection and management"""
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.metrics = defaultdict(list)
        self.counters = defaultdict(float)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # Start metrics server
        self.start_metrics_server()
    
    def start_metrics_server(self):
        """Start Prometheus metrics server"""
        try:
            # In production, this would start actual Prometheus client
            logger.info(f"Started metrics server for {self.service_name}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
    
    def record_model_prediction(self, model_name: str, model_version: str,
                              prediction_time: float, status: str = "success",
                              prediction_confidence: float = None):
        """Record comprehensive model prediction metrics"""
        
        labels = {
            "model_name": model_name,
            "model_version": model_version,
            "status": status,
            "service": self.service_name
        }
        
        # Increment prediction counter
        counter_key = f"ml_model_predictions_total_{hash(frozenset(labels.items()))}"
        self.counters[counter_key] += 1
        
        # Record prediction latency
        histogram_key = f"ml_model_inference_duration_seconds_{hash(frozenset(labels.items()))}"
        self.histograms[histogram_key].append(prediction_time)
        
        # Record confidence if available
        if prediction_confidence is not None:
            confidence_labels = {k: v for k, v in labels.items() if k != "status"}
            gauge_key = f"ml_model_prediction_confidence_{hash(frozenset(confidence_labels.items()))}"
            self.gauges[gauge_key] = prediction_confidence
        
        # Store metric points for analysis
        self.metrics["ml_model_predictions_total"].append(MetricPoint(
            timestamp=time.time(),
            value=self.counters[counter_key],
            labels=labels
        ))
        
        self.metrics["ml_model_inference_duration_seconds"].append(MetricPoint(
            timestamp=time.time(),
            value=prediction_time,
            labels=labels
        ))
        
        logger.info(f"Recorded prediction metric: {model_name} - {status} - {prediction_time:.3f}s")
    
    def update_model_accuracy(self, model_name: str, model_version: str, 
                            accuracy: float, dataset_type: str = "validation"):
        """Update model accuracy metrics"""
        
        labels = {
            "model_name": model_name,
            "model_version": model_version,
            "dataset_type": dataset_type,
            "service": self.service_name
        }
        
        gauge_key = f"ml_model_accuracy_{hash(frozenset(labels.items()))}"
        self.gauges[gauge_key] = accuracy
        
        self.metrics["ml_model_accuracy"].append(MetricPoint(
            timestamp=time.time(),
            value=accuracy,
            labels=labels
        ))
        
        # Check for accuracy degradation
        if accuracy < 0.85:
            logger.warning(f"Model accuracy below threshold: {model_name} - {accuracy:.3f}")
        
        logger.info(f"Updated model accuracy: {model_name} - {accuracy:.3f}")
    
    def record_data_drift(self, model_name: str, feature_name: str, 
                         drift_score: float, drift_method: str = "ks_test"):
        """Record data drift detection metrics"""
        
        labels = {
            "model_name": model_name,
            "feature_name": feature_name,
            "drift_method": drift_method,
            "service": self.service_name
        }
        
        gauge_key = f"ml_data_drift_score_{hash(frozenset(labels.items()))}"
        self.gauges[gauge_key] = drift_score
        
        self.metrics["ml_data_drift_score"].append(MetricPoint(
            timestamp=time.time(),
            value=drift_score,
            labels=labels
        ))
        
        # Alert on significant drift
        if drift_score > 0.7:
            logger.warning(f"Significant data drift detected: {model_name}.{feature_name} - {drift_score:.3f}")
        
        logger.info(f"Recorded data drift: {model_name}.{feature_name} - {drift_score:.3f}")
    
    def record_business_impact(self, model_name: str, impact_type: str, 
                             value: float, currency: str = "USD"):
        """Record business impact metrics"""
        
        labels = {
            "model_name": model_name,
            "impact_type": impact_type,
            "currency": currency,
            "service": self.service_name
        }
        
        counter_key = f"ml_business_impact_total_{hash(frozenset(labels.items()))}"
        self.counters[counter_key] += value
        
        self.metrics["ml_business_impact_total"].append(MetricPoint(
            timestamp=time.time(),
            value=self.counters[counter_key],
            labels=labels
        ))
        
        logger.info(f"Recorded business impact: {model_name} - {impact_type} - {value} {currency}")
    
    def get_model_health_summary(self, model_name: str) -> Dict[str, Any]:
        """Generate comprehensive model health summary"""
        
        summary = {
            "model_name": model_name,
            "timestamp": datetime.utcnow().isoformat(),
            "health_score": 0.0,
            "metrics": {},
            "alerts": [],
            "recommendations": []
        }
        
        # Calculate prediction rate
        prediction_metrics = [m for m in self.metrics["ml_model_predictions_total"] 
                            if m.labels.get("model_name") == model_name]
        if prediction_metrics:
            recent_predictions = [m for m in prediction_metrics 
                                if time.time() - m.timestamp < 300]  # Last 5 minutes
            prediction_rate = len(recent_predictions) / 5.0  # per minute
            summary["metrics"]["prediction_rate_per_minute"] = prediction_rate
        
        # Get latest accuracy
        accuracy_metrics = [m for m in self.metrics["ml_model_accuracy"] 
                          if m.labels.get("model_name") == model_name]
        if accuracy_metrics:
            latest_accuracy = accuracy_metrics[-1].value
            summary["metrics"]["accuracy"] = latest_accuracy
            
            if latest_accuracy < 0.85:
                summary["alerts"].append("Model accuracy below 85% threshold")
                summary["recommendations"].append("Consider model retraining")
        
        # Calculate average latency
        latency_metrics = [m for m in self.metrics["ml_model_inference_duration_seconds"] 
                         if m.labels.get("model_name") == model_name]
        if latency_metrics:
            recent_latencies = [m.value for m in latency_metrics 
                              if time.time() - m.timestamp < 300]
            if recent_latencies:
                avg_latency = sum(recent_latencies) / len(recent_latencies)
                p95_latency = sorted(recent_latencies)[int(len(recent_latencies) * 0.95)]
                summary["metrics"]["avg_latency_seconds"] = avg_latency
                summary["metrics"]["p95_latency_seconds"] = p95_latency
                
                if p95_latency > 1.0:
                    summary["alerts"].append("High P95 latency detected")
                    summary["recommendations"].append("Investigate model optimization")
        
        # Check data drift
        drift_metrics = [m for m in self.metrics["ml_data_drift_score"] 
                        if m.labels.get("model_name") == model_name]
        if drift_metrics:
            high_drift_features = [m for m in drift_metrics if m.value > 0.7]
            if high_drift_features:
                summary["alerts"].append(f"High data drift in {len(high_drift_features)} features")
                summary["recommendations"].append("Review feature engineering and data sources")
        
        # Calculate overall health score
        health_factors = []
        
        if "accuracy" in summary["metrics"]:
            health_factors.append(min(summary["metrics"]["accuracy"] / 0.9, 1.0))
        
        if "p95_latency_seconds" in summary["metrics"]:
            latency_score = max(0, 1.0 - (summary["metrics"]["p95_latency_seconds"] - 0.5) / 0.5)
            health_factors.append(latency_score)
        
        if health_factors:
            summary["health_score"] = sum(health_factors) / len(health_factors)
        
        return summary


class ProductionStructuredLogger:
    """Production-ready structured logging for ML systems"""
    
    def __init__(self, service_name: str, environment: str):
        self.service_name = service_name
        self.environment = environment
        
        # Configure structured logging
        self.logger = logging.getLogger(f"{service_name}.structured")
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
    
    def _create_base_log_entry(self, level: str, message: str, 
                              event_type: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create base structured log entry"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "service": self.service_name,
            "environment": self.environment,
            "message": message,
            "event_type": event_type,
            "trace_id": str(uuid.uuid4()),
            "context": context or {}
        }
    
    def log_model_prediction(self, model_name: str, model_version: str,
                           input_features: Dict, prediction: Any,
                           confidence: float, latency_ms: float,
                           request_id: str = None):
        """Log model prediction with comprehensive context"""
        
        context = {
            "model_name": model_name,
            "model_version": model_version,
            "prediction": str(prediction),
            "confidence": confidence,
            "latency_ms": latency_ms,
            "input_feature_count": len(input_features),
            "request_id": request_id or str(uuid.uuid4()),
            "input_features_hash": hash(str(sorted(input_features.items())))
        }
        
        log_entry = self._create_base_log_entry(
            "INFO",
            f"Model prediction completed: {model_name}",
            "model_prediction",
            context
        )
        
        self.logger.info(json.dumps(log_entry))
    
    def log_data_quality_issue(self, pipeline_name: str, issue_type: str,
                              affected_records: int, severity: str,
                              dataset_name: str = None, check_name: str = None):
        """Log data quality issues with detailed context"""
        
        context = {
            "pipeline_name": pipeline_name,
            "dataset_name": dataset_name,
            "issue_type": issue_type,
            "affected_records": affected_records,
            "severity": severity,
            "check_name": check_name,
            "impact_percentage": (affected_records / 10000) * 100  # Assuming 10k total records
        }
        
        level = "ERROR" if severity == "high" else "WARNING" if severity == "medium" else "INFO"
        
        log_entry = self._create_base_log_entry(
            level,
            f"Data quality issue detected in {pipeline_name}: {issue_type}",
            "data_quality_issue",
            context
        )
        
        self.logger.log(getattr(logging, level), json.dumps(log_entry))
    
    def log_model_drift_detection(self, model_name: str, drift_type: str,
                                 drift_score: float, threshold: float,
                                 feature_name: str = None, method: str = None):
        """Log model drift detection with statistical context"""
        
        context = {
            "model_name": model_name,
            "feature_name": feature_name,
            "drift_type": drift_type,
            "drift_score": drift_score,
            "threshold": threshold,
            "drift_detected": drift_score > threshold,
            "method": method or "statistical_test",
            "severity": "high" if drift_score > threshold * 1.5 else "medium" if drift_score > threshold else "low"
        }
        
        level = "WARNING" if drift_score > threshold else "INFO"
        message = f"Model drift {'detected' if drift_score > threshold else 'monitored'}: {model_name}"
        
        log_entry = self._create_base_log_entry(level, message, "model_drift", context)
        
        self.logger.log(getattr(logging, level), json.dumps(log_entry))
    
    def log_performance_anomaly(self, component: str, metric_name: str,
                               current_value: float, expected_range: Tuple[float, float],
                               severity: str = "medium"):
        """Log performance anomalies with baseline context"""
        
        context = {
            "component": component,
            "metric_name": metric_name,
            "current_value": current_value,
            "expected_min": expected_range[0],
            "expected_max": expected_range[1],
            "deviation_percentage": abs(current_value - sum(expected_range)/2) / (sum(expected_range)/2) * 100,
            "severity": severity
        }
        
        level = "ERROR" if severity == "high" else "WARNING"
        message = f"Performance anomaly detected in {component}: {metric_name} = {current_value}"
        
        log_entry = self._create_base_log_entry(level, message, "performance_anomaly", context)
        
        self.logger.log(getattr(logging, level), json.dumps(log_entry))


class ProductionSLIManager:
    """Production-ready SLI/SLO management for ML systems"""
    
    def __init__(self, metrics_client: ProductionMLMetrics):
        self.metrics_client = metrics_client
        self.slis = {}
        self.slos = {}
        self.error_budgets = {}
    
    def define_availability_sli(self, name: str, service: str, threshold: float = 0.999):
        """Define availability SLI for ML service"""
        
        self.slis[name] = {
            "name": name,
            "type": "availability",
            "service": service,
            "query": f"rate(ml_model_predictions_total{{service='{service}',status='success'}}[5m]) / rate(ml_model_predictions_total{{service='{service}'}}[5m])",
            "threshold": threshold,
            "unit": "percentage"
        }
        
        logger.info(f"Defined availability SLI: {name} with {threshold*100}% threshold")
    
    def define_latency_sli(self, name: str, service: str, percentile: float = 0.95, 
                          threshold: float = 0.5):
        """Define latency SLI for ML service"""
        
        self.slis[name] = {
            "name": name,
            "type": "latency",
            "service": service,
            "percentile": percentile,
            "query": f"histogram_quantile({percentile}, rate(ml_model_inference_duration_seconds_bucket{{service='{service}'}}[5m]))",
            "threshold": threshold,
            "unit": "seconds"
        }
        
        logger.info(f"Defined latency SLI: {name} with P{int(percentile*100)} < {threshold}s")
    
    def define_accuracy_sli(self, name: str, model: str, threshold: float = 0.85):
        """Define accuracy SLI for ML model"""
        
        self.slis[name] = {
            "name": name,
            "type": "accuracy",
            "model": model,
            "query": f"ml_model_accuracy{{model_name='{model}'}}",
            "threshold": threshold,
            "unit": "percentage"
        }
        
        logger.info(f"Defined accuracy SLI: {name} with {threshold*100}% threshold")
    
    def define_slo(self, name: str, sli_name: str, target: float, time_window: str):
        """Define SLO based on SLI"""
        
        if sli_name not in self.slis:
            raise ValueError(f"SLI {sli_name} not found")
        
        self.slos[name] = {
            "name": name,
            "sli_name": sli_name,
            "target": target,
            "time_window": time_window,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Initialize error budget
        self.error_budgets[name] = {
            "total_budget": (1 - target) * 100,
            "used_budget": 0.0,
            "remaining_budget": (1 - target) * 100,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Defined SLO: {name} with {target*100}% target over {time_window}")
    
    def evaluate_sli(self, sli_name: str) -> Dict[str, Any]:
        """Evaluate SLI against current metrics"""
        
        if sli_name not in self.slis:
            raise ValueError(f"SLI {sli_name} not found")
        
        sli = self.slis[sli_name]
        
        # Simulate metric evaluation (in production, would query Prometheus)
        if sli["type"] == "availability":
            current_value = random.uniform(0.995, 0.9999)
        elif sli["type"] == "latency":
            current_value = random.uniform(0.1, 0.8)
        elif sli["type"] == "accuracy":
            current_value = random.uniform(0.82, 0.95)
        else:
            current_value = random.uniform(0.8, 1.0)
        
        meets_threshold = self._check_sli_threshold(sli, current_value)
        
        result = {
            "sli_name": sli["name"],
            "current_value": current_value,
            "threshold": sli["threshold"],
            "meets_threshold": meets_threshold,
            "unit": sli["unit"],
            "evaluated_at": datetime.utcnow().isoformat()
        }
        
        return result
    
    def _check_sli_threshold(self, sli: Dict, current_value: float) -> bool:
        """Check if SLI meets threshold"""
        if sli["type"] == "latency":
            return current_value <= sli["threshold"]
        else:
            return current_value >= sli["threshold"]
    
    def calculate_error_budget(self, slo_name: str) -> Dict[str, Any]:
        """Calculate error budget for SLO"""
        
        if slo_name not in self.slos:
            raise ValueError(f"SLO {slo_name} not found")
        
        slo = self.slos[slo_name]
        sli_name = slo["sli_name"]
        
        # Evaluate current SLI performance
        sli_result = self.evaluate_sli(sli_name)
        current_performance = sli_result["current_value"]
        
        # Calculate error budget
        total_budget = (1 - slo["target"]) * 100
        
        if sli_result["meets_threshold"]:
            used_budget = 0.0
        else:
            used_budget = (slo["target"] - current_performance) * 100
        
        remaining_budget = max(0, total_budget - used_budget)
        budget_utilization = (used_budget / total_budget) * 100 if total_budget > 0 else 0
        
        # Update error budget tracking
        self.error_budgets[slo_name] = {
            "total_budget": total_budget,
            "used_budget": used_budget,
            "remaining_budget": remaining_budget,
            "budget_utilization_percent": budget_utilization,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        return {
            "slo_name": slo["name"],
            "target": slo["target"],
            "current_performance": current_performance,
            "total_error_budget": total_budget,
            "used_error_budget": used_budget,
            "remaining_error_budget": remaining_budget,
            "budget_utilization_percent": budget_utilization,
            "time_window": slo["time_window"],
            "status": "healthy" if budget_utilization < 50 else "warning" if budget_utilization < 80 else "critical"
        }


def demonstrate_complete_observability():
    """Demonstrate complete observability implementation"""
    print("📊 Complete ML System Observability Demonstration")
    print("=" * 60)
    
    # Initialize observability components
    ml_metrics = ProductionMLMetrics("ml-prediction-service")
    ml_logger = ProductionStructuredLogger("ml-prediction-service", "production")
    sli_manager = ProductionSLIManager(ml_metrics)
    
    print("\n1. ML Model Metrics Collection")
    print("-" * 35)
    
    # Simulate model serving with comprehensive metrics
    models = [
        ("fraud_detection", "v2.1.0"),
        ("recommendation_engine", "v1.5.2"),
        ("text_classifier", "v3.0.1")
    ]
    
    for model_name, version in models:
        # Simulate predictions with metrics
        for _ in range(50):
            latency = random.uniform(0.05, 0.8)
            status = "success" if random.random() > 0.02 else "error"
            confidence = random.uniform(0.7, 0.99) if status == "success" else None
            
            ml_metrics.record_model_prediction(model_name, version, latency, status, confidence)
        
        # Update model accuracy
        accuracy = random.uniform(0.82, 0.95)
        ml_metrics.update_model_accuracy(model_name, version, accuracy)
        
        # Record data drift
        features = ["feature_1", "feature_2", "feature_3"]
        for feature in features:
            drift_score = random.uniform(0.1, 0.9)
            ml_metrics.record_data_drift(model_name, feature, drift_score)
        
        # Record business impact
        revenue_impact = random.uniform(1000, 10000)
        ml_metrics.record_business_impact(model_name, "revenue_increase", revenue_impact)
    
    print("✅ Model metrics collection completed")
    
    print("\n2. Structured Logging")
    print("-" * 20)
    
    # Demonstrate structured logging
    ml_logger.log_model_prediction(
        model_name="fraud_detection",
        model_version="v2.1.0",
        input_features={"transaction_amount": 150.0, "merchant_category": "retail"},
        prediction="legitimate",
        confidence=0.95,
        latency_ms=45.2
    )
    
    ml_logger.log_data_quality_issue(
        pipeline_name="customer_features",
        issue_type="missing_values",
        affected_records=1250,
        severity="medium",
        dataset_name="customer_transactions"
    )
    
    ml_logger.log_model_drift_detection(
        model_name="fraud_detection",
        drift_type="feature_drift",
        drift_score=0.75,
        threshold=0.7,
        feature_name="transaction_amount"
    )
    
    print("✅ Structured logging demonstrated")
    
    print("\n3. SLI/SLO Framework")
    print("-" * 20)
    
    # Define SLIs
    sli_manager.define_availability_sli("model_availability", "ml-prediction-service", 0.999)
    sli_manager.define_latency_sli("model_latency_p95", "ml-prediction-service", 0.95, 0.5)
    sli_manager.define_accuracy_sli("model_accuracy", "fraud_detection", 0.85)
    
    # Define SLOs
    sli_manager.define_slo("availability_slo", "model_availability", 0.999, "30d")
    sli_manager.define_slo("latency_slo", "model_latency_p95", 0.95, "7d")
    sli_manager.define_slo("accuracy_slo", "model_accuracy", 0.85, "24h")
    
    # Evaluate SLOs and error budgets
    for slo_name in sli_manager.slos.keys():
        error_budget = sli_manager.calculate_error_budget(slo_name)
        print(f"📊 {slo_name}: {error_budget['status']} - {error_budget['budget_utilization_percent']:.1f}% budget used")
    
    print("✅ SLI/SLO framework implemented")
    
    print("\n4. Model Health Summary")
    print("-" * 25)
    
    # Generate health summaries
    for model_name, _ in models:
        health_summary = ml_metrics.get_model_health_summary(model_name)
        print(f"\n🏥 {model_name} Health Summary:")
        print(f"   Health Score: {health_summary['health_score']:.2f}")
        print(f"   Alerts: {len(health_summary['alerts'])}")
        print(f"   Recommendations: {len(health_summary['recommendations'])}")
        
        if health_summary['alerts']:
            for alert in health_summary['alerts']:
                print(f"   ⚠️  {alert}")
    
    print("\n🎯 Key Observability Features Demonstrated:")
    print("• Comprehensive ML model metrics (latency, accuracy, drift)")
    print("• Structured logging with rich context and correlation")
    print("• SLI/SLO framework with error budget management")
    print("• Model health monitoring and alerting")
    print("• Business impact tracking and ROI measurement")
    print("• Production-ready observability patterns")


def main():
    """Run complete observability demonstration"""
    print("🚀 Day 58: Monitoring & Observability - Complete Solutions")
    print("=" * 70)
    
    # Run comprehensive demonstration
    demonstrate_complete_observability()
    
    print("\n✅ Demonstration completed successfully!")
    print("\nKey Observability Capabilities:")
    print("• Three pillars: Metrics, Logs, and Traces")
    print("• ML-specific monitoring (accuracy, drift, business impact)")
    print("• Structured logging with correlation and context")
    print("• SLI/SLO framework with error budget management")
    print("• Intelligent alerting with actionable insights")
    print("• Comprehensive dashboards and visualization")
    print("• Production-ready observability infrastructure")
    
    print("\nProduction Deployment Best Practices:")
    print("• Deploy Prometheus for metrics collection and alerting")
    print("• Use ELK stack for log aggregation and analysis")
    print("• Implement distributed tracing with Jaeger or Zipkin")
    print("• Set up Grafana for unified observability dashboards")
    print("• Define SLIs/SLOs aligned with business objectives")
    print("• Create runbooks for alert response procedures")
    print("• Implement automated incident response workflows")


if __name__ == "__main__":
    main()