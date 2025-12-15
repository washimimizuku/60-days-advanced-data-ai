#!/usr/bin/env python3
"""
Day 34: A/B Testing Framework - Comprehensive Test Suite

Test suite for validating the A/B testing framework components including
statistical analysis, traffic splitting, multi-armed bandits, and experiment management.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Import framework components
from solution import (
    ExperimentConfig, TrafficSplitter, StatisticalAnalyzer,
    EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit,
    ExperimentManager, ExperimentMonitor, StreamingPlatform
)

class TestExperimentConfig:
    """Test experiment configuration validation"""
    
    def test_valid_config_creation(self):
        """Test creating valid experiment configuration"""
        config = ExperimentConfig(
            experiment_id="test_exp_001",
            name="Test Experiment",
            description="Test description",
            variants={
                'control': {'model': 'baseline'},
                'treatment': {'model': 'new_model'}
            },
            traffic_allocation={'control': 0.5, 'treatment': 0.5},
            primary_metric='conversion_rate'
        )
        
        assert config.experiment_id == "test_exp_001"
        assert config.significance_level == 0.05  # Default value
        assert config.power == 0.8  # Default value
    
    def test_traffic_allocation_validation(self):
        """Test traffic allocation validation"""
        # Valid allocation
        config = ExperimentConfig(
            experiment_id="test_exp_002",
            name="Test",
            description="Test",
            variants={'control': {}, 'treatment_a': {}, 'treatment_b': {}},
            traffic_allocation={'control': 0.4, 'treatment_a': 0.3, 'treatment_b': 0.3},
            primary_metric='ctr'
        )
        
        manager = ExperimentManager()
        # Should not raise exception
        manager._validate_config(config)
        
        # Invalid allocation (doesn't sum to 1)
        config.traffic_allocation = {'control': 0.6, 'treatment_a': 0.3, 'treatment_b': 0.3}
        
        with pytest.raises(ValueError, match="Traffic allocation must sum to 1.0"):
            manager._validate_config(config)

class TestTrafficSplitter:
    """Test traffic splitting and user assignment"""
    
    def setup_method(self):
        """Setup test configuration"""
        self.config = ExperimentConfig(
            experiment_id="traffic_test",
            name="Traffic Test",
            description="Test traffic splitting",
            variants={
                'control': {'model': 'baseline'},
                'treatment_a': {'model': 'model_a'},
                'treatment_b': {'model': 'model_b'}
            },
            traffic_allocation={'control': 0.5, 'treatment_a': 0.3, 'treatment_b': 0.2},
            primary_metric='ctr'
        )
        self.splitter = TrafficSplitter(self.config)
    
    def test_consistent_assignment(self):
        """Test that users get consistent variant assignments"""
        user_id = "user_12345"
        experiment_id = "traffic_test"
        
        # Get assignment multiple times
        assignments = [
            self.splitter.assign_variant(user_id, experiment_id)
            for _ in range(10)
        ]
        
        # All assignments should be the same
        assert len(set(assignments)) == 1
    
    def test_traffic_distribution(self):
        """Test that traffic is distributed according to allocation"""
        experiment_id = "traffic_test"
        n_users = 10000
        
        # Generate assignments for many users
        assignments = []
        for i in range(n_users):
            user_id = f"user_{i:06d}"
            variant = self.splitter.assign_variant(user_id, experiment_id)
            assignments.append(variant)
        
        # Count assignments
        assignment_counts = pd.Series(assignments).value_counts(normalize=True)
        
        # Check that distribution is close to expected
        expected = self.config.traffic_allocation
        for variant, expected_pct in expected.items():
            actual_pct = assignment_counts.get(variant, 0)
            # Allow 2% tolerance
            assert abs(actual_pct - expected_pct) < 0.02, f"Variant {variant}: expected {expected_pct}, got {actual_pct}"
    
    def test_eligibility_criteria(self):
        """Test user eligibility checking"""
        # Add eligibility criteria
        self.config.eligibility_criteria = {
            'allowed_countries': ['US', 'CA'],
            'min_account_age_days': 30
        }
        
        # Eligible user
        eligible_user = {
            'country': 'US',
            'account_created': datetime.now() - timedelta(days=60),
            'is_internal': False
        }
        assert self.splitter.is_user_eligible("user_1", eligible_user)
        
        # Ineligible user (wrong country)
        ineligible_user = {
            'country': 'FR',
            'account_created': datetime.now() - timedelta(days=60),
            'is_internal': False
        }
        assert not self.splitter.is_user_eligible("user_2", ineligible_user)
        
        # Ineligible user (account too new)
        new_user = {
            'country': 'US',
            'account_created': datetime.now() - timedelta(days=10),
            'is_internal': False
        }
        assert not self.splitter.is_user_eligible("user_3", new_user)

class TestStatisticalAnalyzer:
    """Test statistical analysis methods"""
    
    def setup_method(self):
        """Setup analyzer"""
        self.analyzer = StatisticalAnalyzer(significance_level=0.05)
    
    def test_continuous_metric_analysis(self):
        """Test continuous metric analysis"""
        # Generate test data with known difference
        np.random.seed(42)
        control_data = np.random.normal(100, 15, 1000)
        treatment_data = np.random.normal(105, 15, 1000)  # 5% improvement
        
        result = self.analyzer.analyze_continuous_metric(control_data, treatment_data)
        
        # Check basic statistics
        assert abs(result['control_mean'] - 100) < 2  # Should be close to 100
        assert abs(result['treatment_mean'] - 105) < 2  # Should be close to 105
        assert result['relative_improvement_pct'] > 3  # Should detect improvement
        assert result['is_significant']  # Should be significant with large sample
        
        # Check confidence interval
        ci_lower, ci_upper = result['confidence_interval']
        assert ci_lower > 0  # Lower bound should be positive
        assert ci_upper > ci_lower  # Upper bound should be higher
    
    def test_binary_metric_analysis(self):
        """Test binary metric analysis"""
        # Test data: control 10% conversion, treatment 12% conversion
        control_successes = 100
        control_total = 1000
        treatment_successes = 120
        treatment_total = 1000
        
        result = self.analyzer.analyze_binary_metric(
            control_successes, control_total,
            treatment_successes, treatment_total
        )
        
        # Check rates
        assert abs(result['control_rate'] - 0.10) < 0.001
        assert abs(result['treatment_rate'] - 0.12) < 0.001
        assert result['relative_improvement_pct'] > 15  # Should be ~20%
        
        # With large sample, this should be significant
        assert result['is_significant']
    
    def test_sample_size_calculation(self):
        """Test sample size calculation"""
        baseline_rate = 0.10
        mde = 0.20  # 20% relative improvement
        
        sample_size = self.analyzer.calculate_sample_size(
            baseline_rate=baseline_rate,
            minimum_detectable_effect=mde,
            power=0.8,
            significance_level=0.05
        )
        
        # Sample size should be reasonable (not too small or too large)
        assert 100 < sample_size < 100000
        assert isinstance(sample_size, int)

class TestMultiArmedBandits:
    """Test multi-armed bandit algorithms"""
    
    def test_epsilon_greedy_bandit(self):
        """Test epsilon-greedy bandit"""
        bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.1)
        
        # Test initial state
        stats = bandit.get_statistics()
        assert stats['total_pulls'] == 0
        assert len(stats['values']) == 3
        
        # Test arm selection and updates
        for _ in range(100):
            arm = bandit.select_arm()
            assert 0 <= arm < 3
            
            # Simulate different reward rates for different arms
            if arm == 0:
                reward = np.random.binomial(1, 0.1)  # 10% success rate
            elif arm == 1:
                reward = np.random.binomial(1, 0.15)  # 15% success rate
            else:
                reward = np.random.binomial(1, 0.12)  # 12% success rate
            
            bandit.update(arm, reward)
        
        # After many pulls, arm 1 should have highest value
        stats = bandit.get_statistics()
        assert stats['total_pulls'] == 100
        # Arm 1 should likely be the best (though randomness might affect this)
    
    def test_ucb_bandit(self):
        """Test UCB bandit"""
        bandit = UCBBandit(n_arms=3, confidence_level=2.0)
        
        # Test that it tries each arm at least once initially
        selected_arms = set()
        for _ in range(10):
            arm = bandit.select_arm()
            selected_arms.add(arm)
            bandit.update(arm, np.random.random())
        
        # Should have tried multiple arms
        assert len(selected_arms) >= 2
    
    def test_thompson_sampling_bandit(self):
        """Test Thompson Sampling bandit"""
        bandit = ThompsonSamplingBandit(n_arms=3)
        
        # Test updates
        for _ in range(50):
            arm = bandit.select_arm()
            reward = 1 if np.random.random() < 0.5 else 0
            bandit.update(arm, reward)
        
        stats = bandit.get_statistics()
        assert stats['total_pulls'] == 50
        assert len(stats['probabilities']) == 3

class TestExperimentManager:
    """Test experiment management"""
    
    def setup_method(self):
        """Setup experiment manager"""
        self.manager = ExperimentManager()
        self.config = ExperimentConfig(
            experiment_id="manager_test",
            name="Manager Test",
            description="Test experiment management",
            variants={'control': {}, 'treatment': {}},
            traffic_allocation={'control': 0.5, 'treatment': 0.5},
            primary_metric='conversion_rate'
        )
    
    def test_experiment_lifecycle(self):
        """Test complete experiment lifecycle"""
        # Create experiment
        exp_id = self.manager.create_experiment(self.config)
        assert exp_id == "manager_test"
        assert exp_id in self.manager.experiments
        
        # Start experiment
        success = self.manager.start_experiment(exp_id)
        assert success
        assert exp_id in self.manager.active_experiments
        
        # Get user assignment
        user_attributes = {'country': 'US', 'is_internal': False}
        variant = self.manager.get_assignment(exp_id, "user_001", user_attributes)
        assert variant in ['control', 'treatment']
        
        # Record result
        success = self.manager.record_result(exp_id, "user_001", "conversion_rate", 0.15)
        assert success
        
        # Check results
        results = self.manager.get_experiment_results(exp_id)
        assert len(results) == 1
        assert results[0]['metric_name'] == 'conversion_rate'
        assert results[0]['metric_value'] == 0.15

class TestExperimentMonitor:
    """Test experiment monitoring and analysis"""
    
    def setup_method(self):
        """Setup monitor with sample experiment"""
        self.manager = ExperimentManager()
        self.monitor = ExperimentMonitor(self.manager)
        
        self.config = ExperimentConfig(
            experiment_id="monitor_test",
            name="Monitor Test",
            description="Test monitoring",
            variants={'control': {}, 'treatment': {}},
            traffic_allocation={'control': 0.5, 'treatment': 0.5},
            primary_metric='conversion_rate',
            guardrail_metrics={'latency': {'max': 100}}
        )
        
        # Create and start experiment
        self.manager.create_experiment(self.config)
        self.manager.start_experiment("monitor_test")
    
    def test_experiment_status_monitoring(self):
        """Test experiment status monitoring"""
        exp_id = "monitor_test"
        
        # Initially no results
        status = self.monitor.get_experiment_status(exp_id)
        assert status['sample_size'] == 0
        
        # Add some results
        for i in range(100):
            user_id = f"user_{i:03d}"
            variant = self.manager.get_assignment(exp_id, user_id)
            
            # Simulate different conversion rates
            if variant == 'control':
                conversion = 1 if np.random.random() < 0.10 else 0
            else:
                conversion = 1 if np.random.random() < 0.12 else 0
            
            self.manager.record_result(exp_id, user_id, "conversion_rate", conversion)
        
        # Check status with results
        status = self.monitor.get_experiment_status(exp_id)
        assert status['sample_size'] == 100
        assert 'primary_metric_analysis' in status
        assert 'recommendation' in status

class TestStreamingPlatform:
    """Test streaming platform integration"""
    
    def setup_method(self):
        """Setup streaming platform"""
        self.platform = StreamingPlatform()
    
    def test_recommendation_with_experiment(self):
        """Test getting recommendations with A/B testing"""
        # Create experiment
        config = ExperimentConfig(
            experiment_id="rec_test",
            name="Recommendation Test",
            description="Test recommendations",
            variants={
                'control': {'model': 'collaborative_filtering'},
                'treatment_a': {'model': 'deep_learning'}
            },
            traffic_allocation={'control': 0.5, 'treatment_a': 0.5},
            primary_metric='click_through_rate'
        )
        
        exp_id = self.platform.experiment_manager.create_experiment(config)
        self.platform.experiment_manager.start_experiment(exp_id)
        
        # Get recommendations
        user_attributes = {'country': 'US', 'subscription_type': 'premium'}
        result = self.platform.get_recommendations_for_user(
            "test_user", user_attributes, exp_id
        )
        
        assert 'recommendations' in result
        assert 'variant' in result
        assert result['variant'] in ['control', 'treatment_a']
        assert len(result['recommendations']) > 0
    
    def test_user_interaction_recording(self):
        """Test recording user interactions"""
        # Create experiment
        config = ExperimentConfig(
            experiment_id="interaction_test",
            name="Interaction Test",
            description="Test interactions",
            variants={'control': {}, 'treatment': {}},
            traffic_allocation={'control': 0.5, 'treatment': 0.5},
            primary_metric='click_through_rate'
        )
        
        exp_id = self.platform.experiment_manager.create_experiment(config)
        self.platform.experiment_manager.start_experiment(exp_id)
        
        # Get assignment first
        user_attributes = {'country': 'US'}
        self.platform.get_recommendations_for_user("test_user", user_attributes, exp_id)
        
        # Record interaction
        self.platform.record_user_interaction(
            "test_user", "click", "movie_123", 1.0, exp_id
        )
        
        # Check that result was recorded
        results = self.platform.experiment_manager.get_experiment_results(exp_id)
        assert len(results) == 1

def run_performance_tests():
    """Run performance tests for the framework"""
    print("Running performance tests...")
    
    # Test traffic splitting performance
    config = ExperimentConfig(
        experiment_id="perf_test",
        name="Performance Test",
        description="Test performance",
        variants={'control': {}, 'treatment': {}},
        traffic_allocation={'control': 0.5, 'treatment': 0.5},
        primary_metric='ctr'
    )
    
    splitter = TrafficSplitter(config)
    
    import time
    start_time = time.time()
    
    # Assign 10,000 users
    for i in range(10000):
        user_id = f"user_{i:06d}"
        variant = splitter.assign_variant(user_id, "perf_test")
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"Traffic splitting: {10000/duration:.0f} assignments/second")
    
    # Test statistical analysis performance
    analyzer = StatisticalAnalyzer()
    
    # Large datasets
    control_data = np.random.normal(100, 15, 10000)
    treatment_data = np.random.normal(105, 15, 10000)
    
    start_time = time.time()
    result = analyzer.analyze_continuous_metric(control_data, treatment_data)
    end_time = time.time()
    
    print(f"Statistical analysis: {(end_time - start_time)*1000:.1f}ms for 20k samples")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run performance tests
    run_performance_tests()