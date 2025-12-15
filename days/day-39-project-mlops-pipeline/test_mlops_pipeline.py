#!/usr/bin/env python3
"""
Comprehensive test suite for Day 39 MLOps Pipeline

Tests cover:
- Data generation and validation
- Feature store operations
- Model training and evaluation
- Model serving API
- Monitoring and drift detection
- End-to-end pipeline integration

Author: MLOps Engineering Team
Date: December 2024
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
import json
import pickle
from datetime import datetime, timedelta

# Import components to test
from solution import (
    MLOpsConfig, DataGenerator, FeatureStoreManager, 
    AutoMLPipeline, ModelServingAPI, MonitoringSystem, MLOpsPlatform,
    PredictionRequest, PredictionResponse
)

class TestMLOpsConfig:
    """Test MLOps configuration management"""
    
    def test_config_initialization(self):
        """Test configuration initialization with defaults"""
        config = MLOpsConfig()
        
        assert config.POSTGRES_HOST == 'localhost'
        assert config.POSTGRES_PORT == 5432
        assert config.MIN_MODEL_AUC == 0.85
        assert config.MAX_INFERENCE_LATENCY_MS == 100
        assert config.DRIFT_THRESHOLD == 0.1
    
    def test_config_environment_variables(self, monkeypatch):
        """Test configuration from environment variables"""
        monkeypatch.setenv('POSTGRES_HOST', 'test-host')
        monkeypatch.setenv('POSTGRES_PORT', '5433')
        monkeypatch.setenv('MIN_MODEL_AUC', '0.90')
        
        config = MLOpsConfig()
        
        assert config.POSTGRES_HOST == 'test-host'
        assert config.POSTGRES_PORT == 5433
    
    def test_directory_creation(self):
        """Test automatic directory creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = MLOpsConfig()
            config.DATA_PATH = str(Path(temp_dir) / 'data')
            config.MODELS_PATH = str(Path(temp_dir) / 'models')
            
            # Reinitialize to trigger directory creation
            config.__init__()
            
            assert Path(config.DATA_PATH).exists()
            assert Path(config.MODELS_PATH).exists()

class TestDataGenerator:
    """Test data generation functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = MLOpsConfig()
        config.DATA_PATH = tempfile.mkdtemp()
        return config
    
    @pytest.fixture
    def data_generator(self, config):
        """Create data generator instance"""
        return DataGenerator(config)
    
    def test_generate_customer_data(self, data_generator):
        """Test customer data generation"""
        data = data_generator.generate_customer_data(n_customers=1000)
        
        # Check data structure
        assert len(data) == 1000
        assert 'customer_id' in data.columns
        assert 'churn' in data.columns
        assert 'age' in data.columns
        
        # Check data quality
        assert data['customer_id'].nunique() == 1000  # Unique IDs
        assert data['churn'].isin([0, 1]).all()  # Binary labels
        assert (data['age'] >= 18).all() and (data['age'] <= 80).all()  # Age range
        
        # Check churn distribution
        churn_rate = data['churn'].mean()
        assert 0.1 <= churn_rate <= 0.9  # Reasonable churn rate
    
    def test_data_consistency(self, data_generator):
        """Test data generation consistency with seed"""
        data1 = data_generator.generate_customer_data(n_customers=100)
        data2 = data_generator.generate_customer_data(n_customers=100)
        
        # Should be identical due to fixed seed
        pd.testing.assert_frame_equal(data1, data2)
    
    def test_save_data(self, data_generator):
        """Test data saving functionality"""
        data = data_generator.generate_customer_data(n_customers=100)
        filepath = data_generator.save_data(data, 'test_data.csv')
        
        # Check file exists
        assert filepath.exists()
        
        # Check file content
        loaded_data = pd.read_csv(filepath)
        assert len(loaded_data) == 100
        assert list(loaded_data.columns) == list(data.columns)
    
    def teardown_method(self):
        """Clean up test files"""
        # Cleanup is handled by tempfile.mkdtemp()
        pass

class TestFeatureStoreManager:
    """Test feature store management"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = MLOpsConfig()
        config.FEATURE_STORE_PATH = tempfile.mkdtemp()
        config.DATA_PATH = tempfile.mkdtemp()
        return config
    
    @pytest.fixture
    def feature_store(self, config):
        """Create feature store manager"""
        return FeatureStoreManager(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample customer data"""
        return pd.DataFrame({
            'customer_id': [f'CUST_{i:06d}' for i in range(100)],
            'age': np.random.randint(18, 80, 100),
            'account_length_months': np.random.randint(1, 120, 100),
            'monthly_charges': np.random.uniform(20, 150, 100),
            'total_charges': np.random.uniform(100, 5000, 100),
            'support_calls_3m': np.random.poisson(2, 100),
            'login_frequency_30d': np.random.exponential(15, 100),
            'feature_usage_score': np.random.uniform(0, 100, 100),
            'days_since_last_login': np.random.randint(0, 90, 100),
            'avg_session_duration_min': np.random.lognormal(3, 1, 100),
            'late_payments_12m': np.random.poisson(0.5, 100),
            'created_at': pd.Timestamp.now(),
            'churn': np.random.binomial(1, 0.2, 100)
        })
    
    def test_create_feature_store_config(self, feature_store):
        """Test feature store configuration creation"""
        feature_store._create_feature_store_config()
        
        config_path = Path(feature_store.config.FEATURE_STORE_PATH) / 'feature_store.yaml'
        assert config_path.exists()
        
        # Check config content
        with open(config_path, 'r') as f:
            content = f.read()
            assert 'project: mlops_churn_prediction' in content
            assert 'redis://' in content
    
    def test_create_feature_definitions(self, feature_store):
        """Test feature definitions creation"""
        feature_store._create_feature_definitions()
        
        features_path = Path(feature_store.config.FEATURE_STORE_PATH) / 'features.py'
        assert features_path.exists()
        
        # Check features content
        with open(features_path, 'r') as f:
            content = f.read()
            assert 'customer_features' in content
            assert 'Feature(name="age"' in content
    
    def test_prepare_feature_data(self, feature_store, sample_data):
        """Test feature data preparation"""
        feature_path = feature_store.prepare_feature_data(sample_data)
        
        assert feature_path.exists()
        assert feature_path.suffix == '.parquet'
        
        # Check prepared data
        feature_data = pd.read_parquet(feature_path)
        assert len(feature_data) == 100
        assert 'customer_id' in feature_data.columns
        assert 'age' in feature_data.columns

class TestAutoMLPipeline:
    """Test AutoML pipeline functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        config = MLOpsConfig()
        config.DATA_PATH = tempfile.mkdtemp()
        config.MODELS_PATH = tempfile.mkdtemp()
        return config
    
    @pytest.fixture
    def automl_pipeline(self, config):
        """Create AutoML pipeline"""
        with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'):\n            return AutoMLPipeline(config)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 1000
        
        return pd.DataFrame({
            'age': np.random.randint(18, 80, n_samples),
            'account_length_months': np.random.randint(1, 120, n_samples),
            'monthly_charges': np.random.uniform(20, 150, n_samples),
            'total_charges': np.random.uniform(100, 5000, n_samples),
            'support_calls_3m': np.random.poisson(2, n_samples),
            'login_frequency_30d': np.random.exponential(15, n_samples),
            'feature_usage_score': np.random.uniform(0, 100, n_samples),
            'days_since_last_login': np.random.randint(0, 90, n_samples),
            'avg_session_duration_min': np.random.lognormal(3, 1, n_samples),
            'late_payments_12m': np.random.poisson(0.5, n_samples),
            'gender': np.random.choice(['M', 'F'], n_samples),
            'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'payment_method': np.random.choice(['Credit card', 'Bank transfer', 'Electronic check'], n_samples),
            'churn': np.random.binomial(1, 0.2, n_samples)
        })
    
    def test_prepare_data(self, automl_pipeline, sample_data):
        """Test data preparation for ML"""
        X, y = automl_pipeline.prepare_data(sample_data)
        
        # Check shapes
        assert len(X) == len(sample_data)
        assert len(y) == len(sample_data)
        
        # Check feature encoding
        assert X.shape[1] == 13  # 10 numeric + 3 categorical
        assert y.dtype == 'int64'
        
        # Check no missing values
        assert not X.isnull().any().any()
        assert not y.isnull().any()
    
    @patch('optuna.create_study')
    def test_optimize_hyperparameters(self, mock_study, automl_pipeline, sample_data):
        """Test hyperparameter optimization"""
        X, y = automl_pipeline.prepare_data(sample_data)
        
        # Mock Optuna study
        mock_study_instance = Mock()
        mock_study_instance.best_params = {'n_estimators': 100, 'max_depth': 5}
        mock_study_instance.best_value = 0.85
        mock_study.return_value = mock_study_instance
        
        best_params, best_score = automl_pipeline.optimize_hyperparameters(
            X, y, 'random_forest', n_trials=5
        )
        
        assert 'n_estimators' in best_params
        assert best_score == 0.85
        mock_study_instance.optimize.assert_called_once()
    
    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.sklearn.log_model')
    def test_train_models(self, mock_log_model, mock_log_metrics, mock_log_params, 
                         mock_start_run, automl_pipeline, sample_data):
        """Test model training pipeline"""
        # Mock MLflow context
        mock_run = Mock()
        mock_run.info.run_id = 'test_run_id'
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        # Mock hyperparameter optimization
        with patch.object(automl_pipeline, 'optimize_hyperparameters') as mock_optimize:
            mock_optimize.return_value = ({'n_estimators': 100}, 0.85)
            
            X, y = automl_pipeline.prepare_data(sample_data)
            results = automl_pipeline.train_models(X, y)
        
        # Check results structure
        assert isinstance(results, dict)
        assert len(results) > 0
        
        # Check that models were trained
        for model_type, result in results.items():
            assert 'model' in result
            assert 'metrics' in result
            assert 'auc' in result['metrics']

class TestModelServingAPI:
    """Test model serving API functionality"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return MLOpsConfig()
    
    @pytest.fixture
    def serving_api(self, config):
        """Create model serving API"""
        return ModelServingAPI(config)
    
    @pytest.fixture
    def mock_model(self):
        """Create mock ML model"""
        model = Mock()
        model.predict_proba.return_value = np.array([[0.3, 0.7]])
        return model
    
    def test_api_initialization(self, serving_api):
        """Test API initialization"""
        assert serving_api.app is not None
        assert serving_api.current_model_version == "v1"
        assert isinstance(serving_api.models, dict)
    
    def test_load_model(self, serving_api, mock_model):
        """Test model loading"""
        # Create temporary model file
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            pickle.dump(mock_model, f)
            model_path = f.name
        
        try:
            serving_api.load_model(model_path, 'v1')
            assert 'v1' in serving_api.models
        finally:
            Path(model_path).unlink()
    
    def test_ab_test_configuration(self, serving_api):
        """Test A/B testing setup"""
        serving_api.start_ab_test('v1', 'v2', traffic_split=30)
        
        assert serving_api.ab_test_config['enabled'] is True
        assert serving_api.ab_test_config['model_a'] == 'v1'
        assert serving_api.ab_test_config['model_b'] == 'v2'
        assert serving_api.ab_test_config['traffic_split'] == 30
    
    def test_model_version_selection(self, serving_api):
        """Test model version selection for A/B testing"""
        # Test without A/B testing
        version = serving_api._select_model_version('CUST_000001')
        assert version == serving_api.current_model_version
        
        # Test with A/B testing
        serving_api.start_ab_test('v1', 'v2', traffic_split=50)
        version = serving_api._select_model_version('CUST_000001')
        assert version in ['v1', 'v2']
    
    def test_feature_preparation(self, serving_api):
        """Test feature preparation for prediction"""
        features = {
            'age': 35,
            'monthly_charges': 65.5,
            'support_calls_3m': 2
        }
        
        prepared_features = serving_api._prepare_features(features)
        
        assert isinstance(prepared_features, list)
        assert len(prepared_features) == 13  # Expected number of features
        assert prepared_features[0] == 35  # age
        assert prepared_features[2] == 65.5  # monthly_charges

class TestMonitoringSystem:
    """Test monitoring and drift detection"""
    
    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return MLOpsConfig()
    
    @pytest.fixture
    def monitoring_system(self, config):
        """Create monitoring system"""
        return MonitoringSystem(config)
    
    @pytest.fixture
    def reference_data(self):
        """Create reference dataset"""
        np.random.seed(42)
        return pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.normal(5, 2, 1000),
            'feature_3': np.random.exponential(1, 1000),
            'target': np.random.binomial(1, 0.3, 1000)
        })
    
    @pytest.fixture
    def current_data_no_drift(self, reference_data):
        """Create current data without drift"""
        np.random.seed(43)
        return pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 500),
            'feature_2': np.random.normal(5, 2, 500),
            'feature_3': np.random.exponential(1, 500),
            'target': np.random.binomial(1, 0.3, 500)
        })
    
    @pytest.fixture
    def current_data_with_drift(self, reference_data):
        """Create current data with drift"""
        np.random.seed(44)
        return pd.DataFrame({
            'feature_1': np.random.normal(2, 1, 500),  # Mean shift
            'feature_2': np.random.normal(5, 4, 500),  # Variance increase
            'feature_3': np.random.exponential(2, 500),  # Distribution change
            'target': np.random.binomial(1, 0.3, 500)
        })
    
    def test_set_reference_data(self, monitoring_system, reference_data):
        """Test setting reference data"""
        monitoring_system.set_reference_data(reference_data)
        
        assert monitoring_system.reference_data is not None
        assert len(monitoring_system.reference_data) == 1000
    
    @patch('evidently.report.Report')
    def test_detect_data_drift_no_drift(self, mock_report, monitoring_system, 
                                       reference_data, current_data_no_drift):
        """Test drift detection with no drift"""
        # Mock Evidently report
        mock_report_instance = Mock()
        mock_report_instance.as_dict.return_value = {
            'metrics': [{'result': {'drift_score': 0.05}}]
        }
        mock_report.return_value = mock_report_instance
        
        monitoring_system.set_reference_data(reference_data)
        drift_results = monitoring_system.detect_data_drift(current_data_no_drift)
        
        assert 'drift_detected' in drift_results
        assert 'drift_score' in drift_results
        assert drift_results['drift_score'] <= monitoring_system.config.DRIFT_THRESHOLD
    
    def test_track_model_performance_with_actuals(self, monitoring_system):
        """Test performance tracking with actual labels"""
        predictions = [
            {'churn_prediction': 1, 'churn_probability': 0.8},
            {'churn_prediction': 0, 'churn_probability': 0.3},
            {'churn_prediction': 1, 'churn_probability': 0.9},
            {'churn_prediction': 0, 'churn_probability': 0.2}
        ]
        actuals = [1, 0, 1, 0]
        
        metrics = monitoring_system.track_model_performance(predictions, actuals)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc' in metrics
        
        # Perfect predictions should give perfect metrics
        assert metrics['accuracy'] == 1.0
    
    def test_track_model_performance_without_actuals(self, monitoring_system):
        """Test performance tracking without actual labels"""
        predictions = [
            {'churn_prediction': 1, 'churn_probability': 0.8},
            {'churn_prediction': 0, 'churn_probability': 0.3}
        ]
        
        metrics = monitoring_system.track_model_performance(predictions)
        
        assert 'prediction_count' in metrics
        assert 'churn_rate' in metrics
        assert 'avg_churn_probability' in metrics
        assert metrics['prediction_count'] == 2

class TestMLOpsPlatform:
    """Test complete MLOps platform integration"""
    
    @pytest.fixture
    def platform(self):
        """Create MLOps platform"""
        with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'):
            return MLOpsPlatform()
    
    @patch.object(DataGenerator, 'generate_customer_data')
    @patch.object(DataGenerator, 'save_data')
    @patch.object(FeatureStoreManager, 'initialize_feature_store')
    @patch.object(FeatureStoreManager, 'prepare_feature_data')
    @patch.object(FeatureStoreManager, 'deploy_features')
    @patch.object(AutoMLPipeline, 'prepare_data')
    @patch.object(AutoMLPipeline, 'train_models')
    @patch.object(AutoMLPipeline, 'create_ensemble')
    def test_run_complete_pipeline(self, mock_ensemble, mock_train, mock_prepare_data,
                                  mock_deploy, mock_prepare_features, mock_init_fs,
                                  mock_save_data, mock_generate_data, platform):
        """Test complete pipeline execution"""
        # Setup mocks
        sample_data = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'age': [25, 35],
            'churn': [0, 1]
        })
        
        mock_generate_data.return_value = sample_data
        mock_save_data.return_value = Path('/tmp/data.csv')
        mock_prepare_features.return_value = Path('/tmp/features.parquet')
        mock_prepare_data.return_value = (sample_data.drop('churn', axis=1), sample_data['churn'])
        mock_train.return_value = {'rf': {'metrics': {'auc': 0.9}}}
        mock_ensemble.return_value = Mock()
        
        # Run pipeline
        with patch('builtins.open', create=True), \
             patch('pickle.dump'), \
             patch.object(platform.serving_api, 'load_model'), \
             patch.object(platform.monitoring, 'set_reference_data'), \
             patch.object(platform.monitoring, 'detect_data_drift'), \
             patch.object(platform.monitoring, 'track_model_performance'):
            
            results = platform.run_complete_pipeline()
        
        # Verify pipeline execution
        assert results['status'] == 'success'
        assert 'model_performance' in results
        assert 'drift_results' in results
        assert 'model_path' in results
        
        # Verify all components were called
        mock_generate_data.assert_called_once()
        mock_init_fs.assert_called_once()
        mock_train.assert_called_once()

class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_prediction_flow(self):
        """Test end-to-end prediction flow"""
        # This would test the complete flow from data ingestion to prediction
        # In a real scenario, this would involve:
        # 1. Starting the infrastructure (Docker containers)
        # 2. Loading data into feature store
        # 3. Training a model
        # 4. Making predictions via API
        # 5. Monitoring the results
        
        # For now, we'll test the core components integration
        config = MLOpsConfig()
        
        # Generate test data
        data_generator = DataGenerator(config)
        sample_data = data_generator.generate_customer_data(100)
        
        # Test AutoML pipeline
        with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'):
            automl = AutoMLPipeline(config)
            X, y = automl.prepare_data(sample_data)
            
            # Verify data preparation
            assert len(X) == 100
            assert len(y) == 100
            assert X.shape[1] == 13
        
        # Test monitoring
        monitoring = MonitoringSystem(config)
        monitoring.set_reference_data(sample_data)
        
        # Verify monitoring setup
        assert monitoring.reference_data is not None
        assert len(monitoring.reference_data) == 100

def run_performance_tests():
    """Run performance benchmarks"""
    print("🚀 Running Performance Tests...")
    
    # Test data generation performance
    config = MLOpsConfig()
    data_generator = DataGenerator(config)
    
    import time
    start_time = time.time()
    large_dataset = data_generator.generate_customer_data(50000)
    generation_time = time.time() - start_time
    
    print(f"✅ Generated 50K records in {generation_time:.2f} seconds")
    print(f"📊 Generation rate: {50000/generation_time:.0f} records/second")
    
    # Test feature preparation performance
    with patch('mlflow.set_tracking_uri'), patch('mlflow.set_experiment'):
        automl = AutoMLPipeline(config)
        
        start_time = time.time()
        X, y = automl.prepare_data(large_dataset)
        preparation_time = time.time() - start_time
        
        print(f"✅ Prepared 50K records in {preparation_time:.2f} seconds")
        print(f"📊 Preparation rate: {50000/preparation_time:.0f} records/second")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
    
    # Run performance tests
    run_performance_tests()
    
    print("\\n🎉 All tests completed!")