"""
Test suite for feature monitoring and drift detection system
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import tempfile
import os

# Import classes from solution
from solution import (
    PopulationStabilityIndex,
    KolmogorovSmirnovDriftDetector,
    MLBasedDriftDetector,
    RealTimeDriftMonitor,
    AutomatedRetrainingPipeline,
    FeatureMonitoringDashboard,
    create_sample_ecommerce_data,
    create_drifted_data
)

class TestPopulationStabilityIndex:
    """Test PSI drift detection"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'numeric_feature': np.random.normal(100, 15, 1000),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], 1000, p=[0.5, 0.3, 0.2])
        })
        
        self.psi_detector = PopulationStabilityIndex(n_bins=10)
    
    def test_psi_initialization(self):
        """Test PSI detector initialization"""
        assert self.psi_detector.n_bins == 10
        assert self.psi_detector.min_bin_size == 0.05
        assert len(self.psi_detector.reference_stats) == 0
    
    def test_psi_fit_numeric_feature(self):
        """Test fitting PSI on numeric features"""
        self.psi_detector.fit(self.reference_data, ['numeric_feature'])
        
        assert 'numeric_feature' in self.psi_detector.reference_stats
        stats = self.psi_detector.reference_stats['numeric_feature']
        assert stats['type'] == 'numeric'
        assert 'bins' in stats
        assert 'proportions' in stats
        assert len(stats['proportions']) <= self.psi_detector.n_bins
    
    def test_psi_fit_categorical_feature(self):
        """Test fitting PSI on categorical features"""
        self.psi_detector.fit(self.reference_data, ['categorical_feature'])
        
        assert 'categorical_feature' in self.psi_detector.reference_stats
        stats = self.psi_detector.reference_stats['categorical_feature']
        assert stats['type'] == 'categorical'
        assert 'proportions' in stats
        assert 'categories' in stats
    
    def test_psi_calculate_no_drift(self):
        """Test PSI calculation with no drift"""
        self.psi_detector.fit(self.reference_data, ['numeric_feature'])
        
        # Use same data (should have low PSI)
        psi_scores = self.psi_detector.calculate_psi(self.reference_data)
        
        assert 'numeric_feature' in psi_scores
        assert psi_scores['numeric_feature'] < 0.1  # No significant change
    
    def test_psi_calculate_with_drift(self):
        """Test PSI calculation with drift"""
        self.psi_detector.fit(self.reference_data, ['numeric_feature'])
        
        # Create drifted data
        drifted_data = pd.DataFrame({
            'numeric_feature': np.random.normal(120, 20, 1000)  # Different mean and std
        })
        
        psi_scores = self.psi_detector.calculate_psi(drifted_data)
        
        assert 'numeric_feature' in psi_scores
        assert psi_scores['numeric_feature'] > 0.1  # Should detect drift
    
    def test_psi_interpretation(self):
        """Test PSI score interpretation"""
        assert self.psi_detector.interpret_psi(0.05) == "No significant change"
        assert self.psi_detector.interpret_psi(0.15) == "Minor change"
        assert self.psi_detector.interpret_psi(0.35) == "Major change"
        assert self.psi_detector.interpret_psi(0.75) == "Severe change - investigate immediately"

class TestKolmogorovSmirnovDriftDetector:
    """Test KS drift detection"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'feature1': np.random.normal(50, 10, 1000),
            'feature2': np.random.exponential(2, 1000),
            'categorical': np.random.choice(['X', 'Y', 'Z'], 1000)
        })
        
        self.ks_detector = KolmogorovSmirnovDriftDetector(significance_level=0.05)
    
    def test_ks_initialization(self):
        """Test KS detector initialization"""
        assert self.ks_detector.significance_level == 0.05
        assert len(self.ks_detector.reference_distributions) == 0
    
    def test_ks_fit(self):
        """Test fitting KS detector"""
        self.ks_detector.fit(self.reference_data, ['feature1', 'feature2', 'categorical'])
        
        assert 'feature1' in self.ks_detector.reference_distributions
        assert 'feature2' in self.ks_detector.reference_distributions
        assert 'categorical' in self.ks_detector.reference_distributions
        
        # Check numeric feature storage
        assert self.ks_detector.reference_distributions['feature1']['type'] == 'numeric'
        assert 'data' in self.ks_detector.reference_distributions['feature1']
        
        # Check categorical feature storage
        assert self.ks_detector.reference_distributions['categorical']['type'] == 'categorical'
        assert 'value_counts' in self.ks_detector.reference_distributions['categorical']
    
    def test_ks_detect_no_drift(self):
        """Test KS drift detection with no drift"""
        self.ks_detector.fit(self.reference_data, ['feature1'])
        
        # Use same data
        drift_results = self.ks_detector.detect_drift(self.reference_data)
        
        assert 'feature1' in drift_results
        assert not drift_results['feature1']['drift_detected']
        assert drift_results['feature1']['p_value'] > 0.05
    
    def test_ks_detect_with_drift(self):
        """Test KS drift detection with drift"""
        self.ks_detector.fit(self.reference_data, ['feature1'])
        
        # Create significantly different data
        drifted_data = pd.DataFrame({
            'feature1': np.random.normal(80, 15, 1000)  # Different distribution
        })
        
        drift_results = self.ks_detector.detect_drift(drifted_data)
        
        assert 'feature1' in drift_results
        assert drift_results['feature1']['drift_detected']
        assert drift_results['feature1']['p_value'] < 0.05

class TestMLBasedDriftDetector:
    """Test ML-based drift detection"""
    
    def setup_method(self):
        """Setup test data"""
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 1000),
            'feature2': np.random.normal(0, 1, 1000),
            'feature3': np.random.normal(0, 1, 1000)
        })
        
        self.ml_detector = MLBasedDriftDetector(contamination=0.1)
    
    def test_ml_detector_initialization(self):
        """Test ML detector initialization"""
        assert self.ml_detector.contamination == 0.1
        assert self.ml_detector.drift_model is None
        assert self.ml_detector.domain_classifier is None
    
    def test_ml_detector_fit(self):
        """Test fitting ML detector"""
        self.ml_detector.fit(self.reference_data, ['feature1', 'feature2', 'feature3'])
        
        assert self.ml_detector.drift_model is not None
        assert self.ml_detector.reference_stats is not None
        assert 'mean' in self.ml_detector.reference_stats
        assert 'std' in self.ml_detector.reference_stats
    
    def test_ml_detector_detect_no_drift(self):
        """Test ML drift detection with no drift"""
        self.ml_detector.fit(self.reference_data, ['feature1', 'feature2', 'feature3'])
        
        # Use similar data
        current_data = pd.DataFrame({
            'feature1': np.random.normal(0.1, 1.1, 500),
            'feature2': np.random.normal(-0.1, 0.9, 500),
            'feature3': np.random.normal(0.05, 1.05, 500)
        })
        
        drift_results = self.ml_detector.detect_drift(current_data)
        
        assert 'overall_drift_score' in drift_results
        assert 'drift_detected' in drift_results
        # Should detect minimal drift
        assert drift_results['overall_drift_score'] < 0.3
    
    def test_ml_detector_detect_with_drift(self):
        """Test ML drift detection with significant drift"""
        self.ml_detector.fit(self.reference_data, ['feature1', 'feature2', 'feature3'])
        
        # Create significantly different data
        drifted_data = pd.DataFrame({
            'feature1': np.random.normal(5, 2, 500),    # Very different
            'feature2': np.random.normal(-3, 3, 500),   # Very different
            'feature3': np.random.normal(2, 0.5, 500)   # Very different
        })
        
        drift_results = self.ml_detector.detect_drift(drifted_data)
        
        assert drift_results['drift_detected']
        assert drift_results['overall_drift_score'] > 0.2
    
    def test_domain_classifier_training(self):
        """Test domain classifier training"""
        # Create clearly different datasets
        current_data = pd.DataFrame({
            'feature1': np.random.normal(3, 1, 1000),
            'feature2': np.random.normal(3, 1, 1000),
            'feature3': np.random.normal(3, 1, 1000)
        })
        
        auc_score = self.ml_detector.train_domain_classifier(
            self.reference_data, current_data, ['feature1', 'feature2', 'feature3']
        )
        
        # Should be able to distinguish between very different datasets
        assert auc_score > 0.8

class TestRealTimeDriftMonitor:
    """Test real-time drift monitoring"""
    
    def setup_method(self):
        """Setup test data"""
        self.monitor = RealTimeDriftMonitor(window_size=100)
        
        # Create reference data for configuration
        np.random.seed(42)
        self.reference_data = pd.DataFrame({
            'feature1': np.random.normal(100, 15, 1000),
            'feature2': np.random.exponential(2, 1000)
        })
    
    def test_monitor_initialization(self):
        """Test monitor initialization"""
        assert self.monitor.window_size == 100
        assert len(self.monitor.feature_windows) == 0
        assert len(self.monitor.drift_detectors) == 0
    
    def test_setup_feature_monitoring(self):
        """Test feature monitoring setup"""
        feature_config = {
            'feature1': {
                'type': 'psi',
                'reference_data': self.reference_data,
                'alert_threshold': 0.2,
                'n_bins': 10
            }
        }
        
        self.monitor.setup_feature_monitoring(feature_config)
        
        assert 'feature1' in self.monitor.feature_windows
        assert 'feature1' in self.monitor.drift_detectors
        assert 'feature1' in self.monitor.alert_thresholds
        assert self.monitor.alert_thresholds['feature1'] == 0.2
    
    def test_process_new_data(self):
        """Test processing new data points"""
        feature_config = {
            'feature1': {
                'type': 'psi',
                'reference_data': self.reference_data,
                'alert_threshold': 0.2
            }
        }
        
        self.monitor.setup_feature_monitoring(feature_config)
        
        # Add data points
        for i in range(50):
            feature_data = {'feature1': np.random.normal(100, 15)}
            drift_scores = self.monitor.process_new_data(feature_data)
            
            # Should not detect drift yet (window not full)
            assert len(drift_scores) == 0
        
        # Add more points to fill window
        for i in range(60):
            feature_data = {'feature1': np.random.normal(100, 15)}
            drift_scores = self.monitor.process_new_data(feature_data)
        
        # Now should have drift scores
        assert len(self.monitor.feature_windows['feature1']) == 100

class TestAutomatedRetrainingPipeline:
    """Test automated retraining pipeline"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.config = {
            'triggers': {
                'data_drift': {'threshold': 0.2, 'critical_features': ['feature1', 'feature2']},
                'performance_degradation': {'accuracy': 0.8, 'f1_score': 0.7},
                'time_based': {'max_age_days': 7}
            },
            'strategies': {
                'full_retrain': {'drift_threshold': 0.5},
                'incremental_update': {'drift_threshold': 0.3},
                'feature_selection': {'drift_threshold': 0.1}
            }
        }
        
        self.pipeline = AutomatedRetrainingPipeline(self.config)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        assert self.pipeline.config == self.config
        assert len(self.pipeline.retraining_history) == 0
    
    def test_evaluate_triggers_no_activation(self):
        """Test trigger evaluation with no activation"""
        drift_results = {'feature1': 0.1, 'feature2': 0.05}
        performance = {'accuracy': 0.85, 'f1_score': 0.75}
        
        triggers = self.pipeline.evaluate_retraining_triggers(drift_results, performance)
        
        assert not triggers['data_drift']
        assert not triggers['performance_degradation']
    
    def test_evaluate_triggers_drift_activation(self):
        """Test trigger evaluation with drift activation"""
        drift_results = {'feature1': 0.25, 'feature2': 0.15}  # feature1 exceeds threshold
        performance = {'accuracy': 0.85, 'f1_score': 0.75}
        
        triggers = self.pipeline.evaluate_retraining_triggers(drift_results, performance)
        
        assert triggers['data_drift']
        assert not triggers['performance_degradation']
    
    def test_evaluate_triggers_performance_activation(self):
        """Test trigger evaluation with performance activation"""
        drift_results = {'feature1': 0.1, 'feature2': 0.05}
        performance = {'accuracy': 0.75, 'f1_score': 0.65}  # Below thresholds
        
        triggers = self.pipeline.evaluate_retraining_triggers(drift_results, performance)
        
        assert not triggers['data_drift']
        assert triggers['performance_degradation']
    
    def test_determine_retraining_strategy(self):
        """Test retraining strategy determination"""
        # Test full retrain strategy
        triggers = {'data_drift': True, 'performance_degradation': True}
        drift_results = {'feature1': 0.6, 'feature2': 0.4}
        
        strategy = self.pipeline.determine_retraining_strategy(triggers, drift_results)
        assert strategy['type'] == 'full_retrain'
        
        # Test incremental update strategy
        drift_results = {'feature1': 0.35, 'feature2': 0.25}
        strategy = self.pipeline.determine_retraining_strategy(triggers, drift_results)
        assert strategy['type'] == 'incremental_update'
        
        # Test feature selection strategy
        triggers = {'time_based': True}
        drift_results = {'feature1': 0.15, 'feature2': 0.08}
        strategy = self.pipeline.determine_retraining_strategy(triggers, drift_results)
        assert strategy['type'] == 'feature_selection'

class TestFeatureMonitoringDashboard:
    """Test feature monitoring dashboard"""
    
    def setup_method(self):
        """Setup test dashboard"""
        self.drift_detectors = {
            'psi': Mock(),
            'ks': Mock(),
            'ml': Mock()
        }
        
        self.dashboard = FeatureMonitoringDashboard(self.drift_detectors)
    
    def test_dashboard_initialization(self):
        """Test dashboard initialization"""
        assert self.dashboard.drift_detectors == self.drift_detectors
        assert isinstance(self.dashboard.dashboard_data, dict)
    
    def test_create_drift_summary(self):
        """Test drift summary creation"""
        # Test normal status
        summary = self.dashboard.create_drift_summary('feature1', 0.05)
        assert summary['status'] == 'Normal'
        assert summary['color'] == 'green'
        
        # Test minor drift
        summary = self.dashboard.create_drift_summary('feature1', 0.15)
        assert summary['status'] == 'Minor Drift'
        assert summary['color'] == 'orange'
        
        # Test major drift
        summary = self.dashboard.create_drift_summary('feature1', 0.35)
        assert summary['status'] == 'Major Drift'
        assert summary['color'] == 'red'
        
        # Test severe drift
        summary = self.dashboard.create_drift_summary('feature1', 0.75)
        assert summary['status'] == 'Severe Drift'
        assert summary['color'] == 'darkred'
    
    def test_create_distribution_chart(self):
        """Test distribution chart creation"""
        np.random.seed(42)
        reference_data = pd.DataFrame({'feature1': np.random.normal(100, 15, 1000)})
        current_data = pd.DataFrame({'feature1': np.random.normal(110, 20, 1000)})
        
        chart_data = self.dashboard.create_distribution_chart(
            reference_data, current_data, 'feature1'
        )
        
        assert 'feature' in chart_data
        assert 'reference_stats' in chart_data
        assert 'current_stats' in chart_data
        assert 'mean_shift' in chart_data
        assert 'std_ratio' in chart_data
        
        # Check that mean shift is calculated correctly
        expected_shift = current_data['feature1'].mean() - reference_data['feature1'].mean()
        assert abs(chart_data['mean_shift'] - expected_shift) < 0.1

class TestDataGeneration:
    """Test data generation functions"""
    
    def test_create_sample_ecommerce_data(self):
        """Test sample e-commerce data creation"""
        data = create_sample_ecommerce_data()
        
        # Check data structure
        assert isinstance(data, pd.DataFrame)
        assert len(data) == 10000
        
        # Check required columns
        required_columns = [
            'user_age', 'session_duration', 'pages_viewed', 'cart_value',
            'product_price', 'product_rating', 'product_category', 'ctr'
        ]
        
        for col in required_columns:
            assert col in data.columns
        
        # Check data ranges
        assert data['user_age'].min() >= 18
        assert data['user_age'].max() <= 70
        assert data['product_rating'].min() >= 0
        assert data['product_rating'].max() <= 5
        assert data['ctr'].min() >= 0
        assert data['ctr'].max() <= 1
    
    def test_create_drifted_data(self):
        """Test drifted data creation"""
        original_data = create_sample_ecommerce_data()
        
        # Test gradual drift
        gradual_drift = create_drifted_data(original_data, 'gradual')
        assert len(gradual_drift) == len(original_data)
        assert list(gradual_drift.columns) == list(original_data.columns)
        
        # Test sudden drift
        sudden_drift = create_drifted_data(original_data, 'sudden')
        assert len(sudden_drift) == len(original_data)
        
        # Test seasonal drift
        seasonal_drift = create_drifted_data(original_data, 'seasonal')
        assert len(seasonal_drift) == len(original_data)
        
        # Verify drift actually occurred (means should be different)
        original_mean = original_data['user_age'].mean()
        drifted_mean = gradual_drift['user_age'].mean()
        assert abs(original_mean - drifted_mean) > 1  # Should have noticeable difference

# Integration tests
class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_end_to_end_drift_detection(self):
        """Test complete drift detection workflow"""
        # Create data
        reference_data = create_sample_ecommerce_data()
        drifted_data = create_drifted_data(reference_data, 'gradual')
        
        features = ['user_age', 'cart_value', 'session_duration']
        
        # Test PSI detection
        psi_detector = PopulationStabilityIndex()
        psi_detector.fit(reference_data, features)
        psi_scores = psi_detector.calculate_psi(drifted_data)
        
        assert len(psi_scores) == len(features)
        assert all(score >= 0 for score in psi_scores.values())
        
        # Test KS detection
        ks_detector = KolmogorovSmirnovDriftDetector()
        ks_detector.fit(reference_data, features)
        ks_results = ks_detector.detect_drift(drifted_data)
        
        assert len(ks_results) == len(features)
        
        # Test ML detection
        ml_detector = MLBasedDriftDetector()
        ml_detector.fit(reference_data, features)
        ml_results = ml_detector.detect_drift(drifted_data)
        
        assert 'overall_drift_score' in ml_results
        assert 'drift_detected' in ml_results
    
    def test_retraining_pipeline_integration(self):
        """Test retraining pipeline integration"""
        config = {
            'triggers': {
                'data_drift': {'threshold': 0.2, 'critical_features': ['user_age']},
                'performance_degradation': {'accuracy': 0.8},
                'time_based': {'max_age_days': 7}
            },
            'strategies': {
                'full_retrain': {'drift_threshold': 0.5},
                'incremental_update': {'drift_threshold': 0.3},
                'feature_selection': {'drift_threshold': 0.1}
            }
        }
        
        pipeline = AutomatedRetrainingPipeline(config)
        
        # Test with high drift
        drift_results = {'user_age': 0.6}
        performance = {'accuracy': 0.75}
        
        triggers = pipeline.evaluate_retraining_triggers(drift_results, performance)
        strategy = pipeline.determine_retraining_strategy(triggers, drift_results)
        result = pipeline.trigger_retraining(strategy)
        
        assert result['status'] in ['completed', 'failed']
        assert len(pipeline.retraining_history) == 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])