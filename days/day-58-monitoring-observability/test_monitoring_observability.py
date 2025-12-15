"""
Day 58: Monitoring & Observability - Comprehensive Test Suite
Tests for ML system monitoring, observability, and production readiness
"""

import pytest
import time
import json
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from exercise import MockPrometheusClient, MetricPoint, LogEntry
from solution import (
    ProductionMLMetrics, ProductionStructuredLogger, ProductionSLIManager
)


class TestMockPrometheusClient:
    """Test mock Prometheus client functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.client = MockPrometheusClient()
    
    def test_counter_increment(self):
        """Test counter metric increment"""
        labels = {"model": "test_model", "version": "v1.0"}
        
        self.client.counter_inc("test_counter", labels, 1.0)
        self.client.counter_inc("test_counter", labels, 2.0)
        
        metrics = self.client.get_metric_values("test_counter")
        assert len(metrics) == 2
        assert metrics[0].value == 1.0
        assert metrics[1].value == 3.0  # Cumulative
        assert metrics[0].labels == labels
    
    def test_gauge_set(self):
        """Test gauge metric setting"""
        labels = {"service": "ml_service"}
        
        self.client.gauge_set("test_gauge", 0.85, labels)
        self.client.gauge_set("test_gauge", 0.90, labels)
        
        metrics = self.client.get_metric_values("test_gauge")
        assert len(metrics) == 2
        assert metrics[1].value == 0.90  # Latest value
        assert metrics[1].labels == labels
    
    def test_histogram_observe(self):
        """Test histogram metric observation"""
        labels = {"endpoint": "/predict"}
        
        self.client.histogram_observe("request_duration", 0.1, labels)
        self.client.histogram_observe("request_duration", 0.2, labels)
        self.client.histogram_observe("request_duration", 0.15, labels)
        
        metrics = self.client.get_metric_values("request_duration")
        assert len(metrics) == 3
        assert all(m.labels == labels for m in metrics)
    
    def test_prometheus_query(self):
        """Test Prometheus query simulation"""
        # Test rate query
        result = self.client.query("rate(http_requests_total[5m])")
        assert len(result) == 1
        assert "rate_result" in result[0]["metric"]["__name__"]
        
        # Test histogram quantile query
        result = self.client.query("histogram_quantile(0.95, request_duration_bucket)")
        assert len(result) == 1
        assert "quantile_result" in result[0]["metric"]["__name__"]


class TestProductionMLMetrics:
    """Test production ML metrics collection"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.metrics = ProductionMLMetrics("test-service")
    
    def test_record_model_prediction(self):
        """Test model prediction recording"""
        self.metrics.record_model_prediction(
            model_name="fraud_detector",
            model_version="v2.0",
            prediction_time=0.15,
            status="success",
            prediction_confidence=0.95
        )
        
        # Check prediction metrics
        prediction_metrics = self.metrics.metrics["ml_model_predictions_total"]
        assert len(prediction_metrics) == 1
        assert prediction_metrics[0].labels["model_name"] == "fraud_detector"
        assert prediction_metrics[0].labels["status"] == "success"
        
        # Check latency metrics
        latency_metrics = self.metrics.metrics["ml_model_inference_duration_seconds"]
        assert len(latency_metrics) == 1
        assert latency_metrics[0].value == 0.15
    
    def test_update_model_accuracy(self):
        """Test model accuracy updates"""
        self.metrics.update_model_accuracy(
            model_name="classifier",
            model_version="v1.5",
            accuracy=0.92,
            dataset_type="validation"
        )
        
        accuracy_metrics = self.metrics.metrics["ml_model_accuracy"]
        assert len(accuracy_metrics) == 1
        assert accuracy_metrics[0].value == 0.92
        assert accuracy_metrics[0].labels["dataset_type"] == "validation"
    
    def test_record_data_drift(self):
        """Test data drift recording"""
        self.metrics.record_data_drift(
            model_name="recommender",
            feature_name="user_age",
            drift_score=0.75,
            drift_method="ks_test"
        )
        
        drift_metrics = self.metrics.metrics["ml_data_drift_score"]
        assert len(drift_metrics) == 1
        assert drift_metrics[0].value == 0.75
        assert drift_metrics[0].labels["feature_name"] == "user_age"
        assert drift_metrics[0].labels["drift_method"] == "ks_test"
    
    def test_record_business_impact(self):
        """Test business impact recording"""
        self.metrics.record_business_impact(
            model_name="pricing_model",
            impact_type="revenue_increase",
            value=5000.0,
            currency="USD"
        )
        
        impact_metrics = self.metrics.metrics["ml_business_impact_total"]
        assert len(impact_metrics) == 1
        assert impact_metrics[0].labels["impact_type"] == "revenue_increase"
        assert impact_metrics[0].labels["currency"] == "USD"
    
    def test_model_health_summary(self):
        """Test model health summary generation"""
        model_name = "test_model"
        
        # Record some metrics
        self.metrics.record_model_prediction(model_name, "v1.0", 0.1, "success", 0.9)
        self.metrics.record_model_prediction(model_name, "v1.0", 0.2, "success", 0.85)
        self.metrics.update_model_accuracy(model_name, "v1.0", 0.88)
        self.metrics.record_data_drift(model_name, "feature1", 0.3)
        
        summary = self.metrics.get_model_health_summary(model_name)
        
        assert summary["model_name"] == model_name
        assert "health_score" in summary
        assert "metrics" in summary
        assert "alerts" in summary
        assert "recommendations" in summary
        assert isinstance(summary["health_score"], float)


class TestProductionStructuredLogger:
    """Test production structured logging"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.logger = ProductionStructuredLogger("test-service", "test")
    
    @patch('logging.Logger.info')
    def test_log_model_prediction(self, mock_log):
        """Test model prediction logging"""
        self.logger.log_model_prediction(
            model_name="sentiment_analyzer",
            model_version="v3.0",
            input_features={"text_length": 150, "language": "en"},
            prediction="positive",
            confidence=0.87,
            latency_ms=45.2,
            request_id="req-123"
        )
        
        # Verify log was called
        mock_log.assert_called_once()
        
        # Parse logged JSON
        log_call = mock_log.call_args[0][0]
        log_data = json.loads(log_call)
        
        assert log_data["service"] == "test-service"
        assert log_data["environment"] == "test"
        assert log_data["event_type"] == "model_prediction"
        assert log_data["context"]["model_name"] == "sentiment_analyzer"
        assert log_data["context"]["prediction"] == "positive"
        assert log_data["context"]["confidence"] == 0.87
        assert log_data["context"]["request_id"] == "req-123"
    
    @patch('logging.Logger.warning')
    def test_log_data_quality_issue(self, mock_log):
        """Test data quality issue logging"""
        self.logger.log_data_quality_issue(
            pipeline_name="user_features",
            issue_type="missing_values",
            affected_records=1500,
            severity="medium",
            dataset_name="user_profiles",
            check_name="completeness_check"
        )
        
        mock_log.assert_called_once()
        
        log_call = mock_log.call_args[0][0]
        log_data = json.loads(log_call)
        
        assert log_data["event_type"] == "data_quality_issue"
        assert log_data["context"]["pipeline_name"] == "user_features"
        assert log_data["context"]["issue_type"] == "missing_values"
        assert log_data["context"]["affected_records"] == 1500
        assert log_data["context"]["severity"] == "medium"
    
    @patch('logging.Logger.warning')
    def test_log_model_drift_detection(self, mock_log):
        """Test model drift detection logging"""
        self.logger.log_model_drift_detection(
            model_name="price_predictor",
            drift_type="feature_drift",
            drift_score=0.82,
            threshold=0.7,
            feature_name="market_volatility",
            method="psi"
        )
        
        mock_log.assert_called_once()
        
        log_call = mock_log.call_args[0][0]
        log_data = json.loads(log_call)
        
        assert log_data["event_type"] == "model_drift"
        assert log_data["context"]["model_name"] == "price_predictor"
        assert log_data["context"]["drift_score"] == 0.82
        assert log_data["context"]["drift_detected"] is True
        assert log_data["context"]["method"] == "psi"
    
    @patch('logging.Logger.error')
    def test_log_performance_anomaly(self, mock_log):
        """Test performance anomaly logging"""
        self.logger.log_performance_anomaly(
            component="model_server",
            metric_name="response_time",
            current_value=2.5,
            expected_range=(0.1, 1.0),
            severity="high"
        )
        
        mock_log.assert_called_once()
        
        log_call = mock_log.call_args[0][0]
        log_data = json.loads(log_call)
        
        assert log_data["event_type"] == "performance_anomaly"
        assert log_data["context"]["component"] == "model_server"
        assert log_data["context"]["current_value"] == 2.5
        assert log_data["context"]["expected_min"] == 0.1
        assert log_data["context"]["expected_max"] == 1.0


class TestProductionSLIManager:
    """Test SLI/SLO management"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.metrics = ProductionMLMetrics("test-service")
        self.sli_manager = ProductionSLIManager(self.metrics)
    
    def test_define_availability_sli(self):
        """Test availability SLI definition"""
        self.sli_manager.define_availability_sli(
            name="model_availability",
            service="ml-service",
            threshold=0.999
        )
        
        sli = self.sli_manager.slis["model_availability"]
        assert sli["name"] == "model_availability"
        assert sli["type"] == "availability"
        assert sli["service"] == "ml-service"
        assert sli["threshold"] == 0.999
        assert sli["unit"] == "percentage"
    
    def test_define_latency_sli(self):
        """Test latency SLI definition"""
        self.sli_manager.define_latency_sli(
            name="model_latency_p95",
            service="ml-service",
            percentile=0.95,
            threshold=0.5
        )
        
        sli = self.sli_manager.slis["model_latency_p95"]
        assert sli["name"] == "model_latency_p95"
        assert sli["type"] == "latency"
        assert sli["percentile"] == 0.95
        assert sli["threshold"] == 0.5
        assert sli["unit"] == "seconds"
    
    def test_define_accuracy_sli(self):
        """Test accuracy SLI definition"""
        self.sli_manager.define_accuracy_sli(
            name="model_accuracy",
            model="fraud_detector",
            threshold=0.85
        )
        
        sli = self.sli_manager.slis["model_accuracy"]
        assert sli["name"] == "model_accuracy"
        assert sli["type"] == "accuracy"
        assert sli["model"] == "fraud_detector"
        assert sli["threshold"] == 0.85
    
    def test_define_slo(self):
        """Test SLO definition"""
        # First define an SLI
        self.sli_manager.define_availability_sli("test_availability", "test-service")
        
        # Then define SLO
        self.sli_manager.define_slo(
            name="availability_slo",
            sli_name="test_availability",
            target=0.999,
            time_window="30d"
        )
        
        slo = self.sli_manager.slos["availability_slo"]
        assert slo["name"] == "availability_slo"
        assert slo["sli_name"] == "test_availability"
        assert slo["target"] == 0.999
        assert slo["time_window"] == "30d"
        
        # Check error budget initialization
        error_budget = self.sli_manager.error_budgets["availability_slo"]
        assert error_budget["total_budget"] == 0.1  # (1 - 0.999) * 100
    
    def test_evaluate_sli(self):
        """Test SLI evaluation"""
        self.sli_manager.define_availability_sli("test_sli", "test-service", 0.95)
        
        result = self.sli_manager.evaluate_sli("test_sli")
        
        assert "sli_name" in result
        assert "current_value" in result
        assert "threshold" in result
        assert "meets_threshold" in result
        assert "unit" in result
        assert "evaluated_at" in result
        assert isinstance(result["meets_threshold"], bool)
    
    def test_calculate_error_budget(self):
        """Test error budget calculation"""
        # Define SLI and SLO
        self.sli_manager.define_availability_sli("test_sli", "test-service", 0.95)
        self.sli_manager.define_slo("test_slo", "test_sli", 0.99, "7d")
        
        error_budget = self.sli_manager.calculate_error_budget("test_slo")
        
        assert "slo_name" in error_budget
        assert "target" in error_budget
        assert "current_performance" in error_budget
        assert "total_error_budget" in error_budget
        assert "used_error_budget" in error_budget
        assert "remaining_error_budget" in error_budget
        assert "budget_utilization_percent" in error_budget
        assert "status" in error_budget
        assert error_budget["status"] in ["healthy", "warning", "critical"]
    
    def test_sli_not_found_error(self):
        """Test error when SLI not found"""
        with pytest.raises(ValueError, match="SLI nonexistent not found"):
            self.sli_manager.evaluate_sli("nonexistent")
    
    def test_slo_not_found_error(self):
        """Test error when SLO not found"""
        with pytest.raises(ValueError, match="SLO nonexistent not found"):
            self.sli_manager.calculate_error_budget("nonexistent")


class TestMetricPoint:
    """Test MetricPoint dataclass"""
    
    def test_metric_point_creation(self):
        """Test MetricPoint creation and attributes"""
        labels = {"service": "ml-api", "version": "v1.0"}
        point = MetricPoint(
            timestamp=time.time(),
            value=42.5,
            labels=labels
        )
        
        assert isinstance(point.timestamp, float)
        assert point.value == 42.5
        assert point.labels == labels
        assert point.labels["service"] == "ml-api"


class TestLogEntry:
    """Test LogEntry dataclass"""
    
    def test_log_entry_creation(self):
        """Test LogEntry creation and attributes"""
        context = {"model": "test", "accuracy": 0.95}
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            level="INFO",
            service="ml-service",
            message="Model prediction completed",
            context=context
        )
        
        assert entry.level == "INFO"
        assert entry.service == "ml-service"
        assert entry.message == "Model prediction completed"
        assert entry.context["model"] == "test"
        assert entry.context["accuracy"] == 0.95


class TestObservabilityIntegration:
    """Integration tests for observability components"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.metrics = ProductionMLMetrics("integration-test")
        self.logger = ProductionStructuredLogger("integration-test", "test")
        self.sli_manager = ProductionSLIManager(self.metrics)
    
    def test_end_to_end_monitoring_workflow(self):
        """Test complete monitoring workflow"""
        model_name = "integration_model"
        model_version = "v1.0"
        
        # 1. Record model predictions
        for i in range(10):
            latency = 0.1 + (i * 0.01)  # Increasing latency
            status = "success" if i < 8 else "error"  # 20% error rate
            confidence = 0.9 - (i * 0.01) if status == "success" else None
            
            self.metrics.record_model_prediction(
                model_name, model_version, latency, status, confidence
            )
        
        # 2. Update model accuracy
        self.metrics.update_model_accuracy(model_name, model_version, 0.87)
        
        # 3. Record data drift
        self.metrics.record_data_drift(model_name, "feature1", 0.6)
        
        # 4. Define SLIs and SLOs
        self.sli_manager.define_availability_sli("model_availability", "integration-test", 0.95)
        self.sli_manager.define_latency_sli("model_latency", "integration-test", 0.95, 0.2)
        self.sli_manager.define_slo("availability_slo", "model_availability", 0.99, "24h")
        
        # 5. Generate health summary
        health_summary = self.metrics.get_model_health_summary(model_name)
        
        # Verify workflow results
        assert health_summary["model_name"] == model_name
        assert "health_score" in health_summary
        assert len(self.metrics.metrics["ml_model_predictions_total"]) == 10
        assert len(self.sli_manager.slis) == 2
        assert len(self.sli_manager.slos) == 1
    
    @patch('logging.Logger.info')
    def test_structured_logging_integration(self, mock_log):
        """Test structured logging integration with metrics"""
        # Log model prediction
        self.logger.log_model_prediction(
            model_name="integration_model",
            model_version="v1.0",
            input_features={"feature1": 1.0, "feature2": 2.0},
            prediction="class_a",
            confidence=0.92,
            latency_ms=150.0
        )
        
        # Record corresponding metric
        self.metrics.record_model_prediction(
            "integration_model", "v1.0", 0.15, "success", 0.92
        )
        
        # Verify both logging and metrics
        mock_log.assert_called_once()
        assert len(self.metrics.metrics["ml_model_predictions_total"]) == 1
        
        # Parse log entry
        log_call = mock_log.call_args[0][0]
        log_data = json.loads(log_call)
        
        # Verify correlation between log and metric
        assert log_data["context"]["model_name"] == "integration_model"
        assert log_data["context"]["confidence"] == 0.92
        
        metric = self.metrics.metrics["ml_model_predictions_total"][0]
        assert metric.labels["model_name"] == "integration_model"


def test_observability_configuration_validation():
    """Test observability configuration validation"""
    # Test valid configuration
    config = {
        "metrics": {
            "prometheus_port": 8000,
            "collection_interval": 15
        },
        "logging": {
            "level": "INFO",
            "format": "json"
        },
        "tracing": {
            "enabled": True,
            "jaeger_endpoint": "http://jaeger:14268"
        }
    }
    
    # Validate configuration structure
    assert "metrics" in config
    assert "logging" in config
    assert "tracing" in config
    assert config["metrics"]["prometheus_port"] == 8000
    assert config["logging"]["format"] == "json"
    assert config["tracing"]["enabled"] is True


def test_alert_configuration():
    """Test alert configuration and thresholds"""
    alerts = {
        "high_latency": {
            "condition": "p95_latency > 1.0",
            "severity": "warning",
            "threshold": 1.0,
            "duration": "5m"
        },
        "low_accuracy": {
            "condition": "model_accuracy < 0.85",
            "severity": "critical",
            "threshold": 0.85,
            "duration": "2m"
        },
        "high_error_rate": {
            "condition": "error_rate > 0.05",
            "severity": "warning",
            "threshold": 0.05,
            "duration": "3m"
        }
    }
    
    # Validate alert configurations
    for alert_name, alert_config in alerts.items():
        assert "condition" in alert_config
        assert "severity" in alert_config
        assert "threshold" in alert_config
        assert "duration" in alert_config
        assert alert_config["severity"] in ["info", "warning", "critical"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])