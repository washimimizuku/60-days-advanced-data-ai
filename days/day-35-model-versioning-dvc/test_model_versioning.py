"""
Day 35: Model Versioning with DVC - Comprehensive Test Suite

Tests for model versioning system including DVC pipeline management,
MLflow model registry, deployment strategies, and performance monitoring.
"""

import pytest
import tempfile
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import json
import yaml
from datetime import datetime

# Import classes from solution (in real scenario, these would be separate modules)
from solution import (
    DVCPipelineManager,
    MLflowModelManager, 
    DataLineageTracker,
    ModelDeploymentManager,
    ModelPerformanceMonitor,
    HealthTechMLPipeline
)

class TestDVCPipelineManager:
    """Test DVC pipeline management functionality"""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def dvc_manager(self, temp_project):
        """Create DVC manager instance"""
        return DVCPipelineManager(temp_project)
    
    def test_initialize_dvc_project(self, dvc_manager, temp_project):
        """Test DVC project initialization"""
        result = dvc_manager.initialize_dvc_project()
        
        assert result is True
        assert (Path(temp_project) / ".dvc").exists()
        assert (Path(temp_project) / ".dvcignore").exists()
    
    def test_create_pipeline_stage(self, dvc_manager):
        """Test pipeline stage creation"""
        dvc_manager.create_pipeline_stage(
            stage_name="test_stage",
            command="python test.py",
            dependencies=["data/input.csv"],
            outputs=["data/output.csv"],
            parameters=["params.test_param"],
            metrics=["metrics/test_metrics.json"]
        )
        
        assert "test_stage" in dvc_manager.pipeline_config["stages"]
        stage_config = dvc_manager.pipeline_config["stages"]["test_stage"]
        
        assert stage_config["cmd"] == "python test.py"
        assert "data/input.csv" in stage_config["deps"]
        assert "data/output.csv" in stage_config["outs"]
        assert "params.test_param" in stage_config["params"]
        assert "metrics/test_metrics.json" in stage_config["metrics"]
    
    def test_run_pipeline_success(self, dvc_manager, temp_project):
        """Test successful pipeline execution"""
        # Create test stage
        dvc_manager.create_pipeline_stage(
            stage_name="test_stage",
            command="echo 'test'",
            dependencies=[],
            outputs=["test_output.txt"]
        )
        
        results = dvc_manager.run_pipeline()
        
        assert "test_stage" in results
        assert results["test_stage"]["status"] == "success"
    
    def test_get_pipeline_status(self, dvc_manager):
        """Test pipeline status retrieval"""
        # Create test stage
        dvc_manager.create_pipeline_stage(
            stage_name="test_stage",
            command="echo 'test'",
            dependencies=[],
            outputs=["test_output.txt"]
        )
        
        status = dvc_manager.get_pipeline_status()
        
        assert "stages" in status
        assert "test_stage" in status["stages"]
    
    def test_add_data_file(self, dvc_manager, temp_project):
        """Test adding data file to DVC tracking"""
        # Create test file
        test_file = Path(temp_project) / "test_data.csv"
        pd.DataFrame({"col1": [1, 2, 3]}).to_csv(test_file, index=False)
        
        dvc_manager.add_data_file(str(test_file))
        
        # Check DVC file created
        dvc_file = test_file.with_suffix(test_file.suffix + '.dvc')
        assert dvc_file.exists()
        
        # Check gitignore updated
        gitignore = Path(temp_project) / ".gitignore"
        if gitignore.exists():
            content = gitignore.read_text()
            assert str(test_file) in content

class TestMLflowModelManager:
    """Test MLflow model management functionality"""
    
    @pytest.fixture
    def mlflow_manager(self):
        """Create MLflow manager instance"""
        return MLflowModelManager(experiment_name="test_experiment")
    
    def test_start_experiment_run(self, mlflow_manager):
        """Test starting experiment run"""
        run_id = mlflow_manager.start_experiment_run("test_run")
        
        assert run_id is not None
        assert mlflow_manager.current_run is not None
    
    def test_log_parameters(self, mlflow_manager):
        """Test parameter logging"""
        mlflow_manager.start_experiment_run("test_run")
        
        params = {"learning_rate": 0.01, "n_estimators": 100}
        mlflow_manager.log_parameters(params)
        
        if not hasattr(mlflow_manager, 'client') or mlflow_manager.client is None:
            # Mock implementation
            assert mlflow_manager.current_run["params"] == params
    
    def test_log_metrics(self, mlflow_manager):
        """Test metrics logging"""
        mlflow_manager.start_experiment_run("test_run")
        
        metrics = {"accuracy": 0.95, "f1_score": 0.92}
        mlflow_manager.log_metrics(metrics)
        
        if not hasattr(mlflow_manager, 'client') or mlflow_manager.client is None:
            # Mock implementation
            assert mlflow_manager.current_run["metrics"] == metrics
    
    def test_model_registration(self, mlflow_manager):
        """Test model registration"""
        from sklearn.ensemble import RandomForestClassifier
        
        mlflow_manager.start_experiment_run("test_run")
        
        # Create and log model
        model = RandomForestClassifier(n_estimators=10)
        X_dummy = np.random.random((10, 5))
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        
        mlflow_manager.log_model(model, "test_model", "test_registered_model")
        
        # Test registration
        if not hasattr(mlflow_manager, 'client') or mlflow_manager.client is None:
            # Mock implementation
            assert "test_registered_model" in mlflow_manager.model_registry

class TestDataLineageTracker:
    """Test data lineage tracking functionality"""
    
    @pytest.fixture
    def lineage_tracker(self):
        """Create lineage tracker instance"""
        return DataLineageTracker()
    
    def test_track_transformation(self, lineage_tracker):
        """Test transformation tracking"""
        lineage_tracker.track_transformation(
            input_data="raw_data.csv",
            output_data="processed_data.csv",
            transformation_info={
                "name": "data_cleaning",
                "description": "Remove null values and outliers"
            }
        )
        
        assert len(lineage_tracker.transformations) == 1
        assert "processed_data.csv" in lineage_tracker.lineage_graph
    
    def test_get_lineage(self, lineage_tracker):
        """Test lineage retrieval"""
        # Create transformation chain
        lineage_tracker.track_transformation(
            "raw_data.csv", "clean_data.csv", {"name": "cleaning"}
        )
        lineage_tracker.track_transformation(
            "clean_data.csv", "features.csv", {"name": "feature_engineering"}
        )
        
        lineage = lineage_tracker.get_lineage("features.csv")
        
        assert len(lineage) == 2
        assert lineage[0]["output_data"] == "features.csv"
        assert lineage[1]["output_data"] == "clean_data.csv"
    
    def test_visualize_lineage(self, lineage_tracker):
        """Test lineage visualization"""
        lineage_tracker.track_transformation(
            "raw_data.csv", "processed_data.csv", {"name": "processing"}
        )
        
        visualization = lineage_tracker.visualize_lineage("processed_data.csv")
        
        assert "Data Lineage for processed_data.csv" in visualization
        assert "raw_data.csv" in visualization
        assert "processing" in visualization

class TestModelDeploymentManager:
    """Test model deployment management functionality"""
    
    @pytest.fixture
    def deployment_manager(self):
        """Create deployment manager instance"""
        mlflow_manager = MLflowModelManager()
        return ModelDeploymentManager(mlflow_manager)
    
    def test_validate_model(self, deployment_manager):
        """Test model validation"""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create valid model
        model = RandomForestClassifier(n_estimators=10)
        X_dummy = np.random.random((10, 5))
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        
        validation_result = deployment_manager.validate_model(model)
        
        assert validation_result["is_valid"] is True
        assert len(validation_result["errors"]) == 0
    
    def test_run_smoke_tests(self, deployment_manager):
        """Test smoke test execution"""
        from sklearn.ensemble import RandomForestClassifier
        import tempfile
        import pickle
        
        # Create model and save it
        model = RandomForestClassifier(n_estimators=10)
        X_dummy = np.random.random((10, 10))
        y_dummy = np.random.randint(0, 2, 10)
        model.fit(X_dummy, y_dummy)
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(model, f)
            model_path = f.name
        
        deployment_info = {"model_path": model_path}
        
        smoke_test_result = deployment_manager.run_smoke_tests(deployment_info)
        
        assert "passed" in smoke_test_result
        assert "results" in smoke_test_result
        
        # Cleanup
        Path(model_path).unlink()

class TestModelPerformanceMonitor:
    """Test model performance monitoring functionality"""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor instance"""
        return ModelPerformanceMonitor("test_model")
    
    def test_monitor_performance(self, performance_monitor):
        """Test performance monitoring"""
        # Generate test data
        predictions = np.array([1, 0, 1, 1, 0])
        actuals = np.array([1, 0, 1, 0, 0])
        
        result = performance_monitor.monitor_performance(predictions, actuals)
        
        assert "current_metrics" in result
        assert "accuracy" in result["current_metrics"]
        assert "degradation_detected" in result
    
    def test_detect_data_drift_numerical(self, performance_monitor):
        """Test numerical data drift detection"""
        # Create reference and current data
        reference_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 1000),
            "feature2": np.random.normal(5, 2, 1000)
        })
        
        # Current data with drift
        current_data = pd.DataFrame({
            "feature1": np.random.normal(2, 1, 500),  # Mean shift
            "feature2": np.random.normal(5, 2, 500)   # No drift
        })
        
        drift_result = performance_monitor.detect_data_drift(reference_data, current_data)
        
        assert "drift_detected" in drift_result
        assert "feature_results" in drift_result
    
    def test_detect_data_drift_categorical(self, performance_monitor):
        """Test categorical data drift detection"""
        # Create reference and current data
        reference_data = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C"], 1000, p=[0.5, 0.3, 0.2])
        })
        
        # Current data with different distribution
        current_data = pd.DataFrame({
            "category": np.random.choice(["A", "B", "C"], 500, p=[0.2, 0.3, 0.5])
        })
        
        drift_result = performance_monitor.detect_data_drift(reference_data, current_data)
        
        assert "drift_detected" in drift_result
        assert "feature_results" in drift_result
    
    def test_generate_alert(self, performance_monitor):
        """Test alert generation"""
        performance_monitor.generate_alert(
            "test_alert", 
            "Test alert message", 
            "high"
        )
        
        assert len(performance_monitor.alerts) == 1
        alert = performance_monitor.alerts[0]
        
        assert alert["alert_type"] == "test_alert"
        assert alert["message"] == "Test alert message"
        assert alert["severity"] == "high"
    
    def test_get_performance_report(self, performance_monitor):
        """Test performance report generation"""
        # Add some performance history
        predictions = np.array([1, 0, 1, 1, 0])
        actuals = np.array([1, 0, 1, 0, 0])
        
        performance_monitor.monitor_performance(predictions, actuals)
        
        report = performance_monitor.get_performance_report()
        
        assert "model_name" in report
        assert "performance_summary" in report
        assert "monitoring_period" in report

class TestHealthTechMLPipeline:
    """Test complete ML pipeline functionality"""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def ml_pipeline(self, temp_project):
        """Create ML pipeline instance"""
        return HealthTechMLPipeline(temp_project)
    
    def test_setup_project(self, ml_pipeline):
        """Test project setup"""
        ml_pipeline.setup_project()
        
        # Check directory structure
        project_root = ml_pipeline.project_root
        
        assert (project_root / "data/raw").exists()
        assert (project_root / "data/processed").exists()
        assert (project_root / "models").exists()
        assert (project_root / "params.yaml").exists()
    
    def test_generate_synthetic_data(self, ml_pipeline):
        """Test synthetic data generation"""
        ml_pipeline.setup_project()
        
        data = ml_pipeline.generate_synthetic_data(n_samples=100)
        
        assert len(data) == 100
        assert "cardiovascular_risk" in data.columns
        assert data["cardiovascular_risk"].dtype == int
        
        # Check data file created
        data_path = ml_pipeline.project_root / "data/raw/healthcare_data.csv"
        assert data_path.exists()
    
    def test_create_pipeline(self, ml_pipeline):
        """Test pipeline creation"""
        ml_pipeline.setup_project()
        ml_pipeline.create_pipeline()
        
        # Check DVC pipeline created
        dvc_file = ml_pipeline.project_root / "dvc.yaml"
        assert dvc_file.exists()
        
        # Check pipeline stages
        with open(dvc_file, 'r') as f:
            pipeline_config = yaml.safe_load(f)
        
        assert "stages" in pipeline_config
        assert "generate_data" in pipeline_config["stages"]
        assert "train_model" in pipeline_config["stages"]

# Integration Tests
class TestIntegration:
    """Integration tests for complete workflow"""
    
    @pytest.fixture
    def temp_project(self):
        """Create temporary project directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_complete_ml_workflow(self, temp_project):
        """Test complete ML workflow from setup to monitoring"""
        # Initialize pipeline
        pipeline = HealthTechMLPipeline(temp_project)
        
        # Setup project
        pipeline.setup_project()
        
        # Generate data
        data = pipeline.generate_synthetic_data(n_samples=100)
        assert len(data) == 100
        
        # Create pipeline
        pipeline.create_pipeline()
        
        # Verify lineage tracking
        assert len(pipeline.lineage_tracker.transformations) > 0
        
        # Test performance monitoring
        predictions = np.random.randint(0, 2, 50)
        actuals = np.random.randint(0, 2, 50)
        
        monitor_result = pipeline.performance_monitor.monitor_performance(
            predictions, actuals
        )
        
        assert "current_metrics" in monitor_result

# Performance Tests
class TestPerformance:
    """Performance tests for critical operations"""
    
    def test_large_dataset_lineage_tracking(self):
        """Test lineage tracking with large number of transformations"""
        tracker = DataLineageTracker()
        
        # Track many transformations
        start_time = datetime.now()
        
        for i in range(1000):
            tracker.track_transformation(
                f"input_{i}.csv",
                f"output_{i}.csv",
                {"name": f"transformation_{i}"}
            )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 5.0  # 5 seconds
        assert len(tracker.transformations) == 1000
    
    def test_drift_detection_performance(self):
        """Test drift detection performance with large datasets"""
        monitor = ModelPerformanceMonitor("test_model")
        
        # Create large datasets
        reference_data = pd.DataFrame({
            "feature1": np.random.normal(0, 1, 10000),
            "feature2": np.random.normal(5, 2, 10000)
        })
        
        current_data = pd.DataFrame({
            "feature1": np.random.normal(0.5, 1, 5000),
            "feature2": np.random.normal(5, 2, 5000)
        })
        
        start_time = datetime.now()
        
        drift_result = monitor.detect_data_drift(reference_data, current_data)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 10.0  # 10 seconds
        assert "drift_detected" in drift_result

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])