#!/usr/bin/env python3
"""
Day 40: ML Systems Checkpoint - Test Suite

Tests for the ML system health check and assessment functionality.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import matplotlib.pyplot as plt

from exercise import (
    MLSystemHealthChecker, 
    analyze_system_health, 
    create_monitoring_dashboard,
    generate_optimization_plan,
    calculate_business_impact
)

class TestMLSystemHealthChecker:
    """Test the ML system health checker functionality"""
    
    def test_initialization(self):
        """Test health checker initialization"""
        checker = MLSystemHealthChecker()
        
        assert hasattr(checker, 'system_metrics')
        assert hasattr(checker, 'model_performance')
        assert hasattr(checker, 'feature_metrics')
        assert hasattr(checker, 'business_metrics')
    
    def test_system_metrics_structure(self):
        """Test system metrics have required fields"""
        checker = MLSystemHealthChecker()
        metrics = checker.system_metrics
        
        required_fields = [
            'api_latency_p95_ms', 'error_rate_percent', 
            'cpu_utilization_percent', 'disk_usage_percent'
        ]
        
        for field in required_fields:
            assert field in metrics
            assert isinstance(metrics[field], (int, float))
    
    def test_model_performance_structure(self):
        """Test model performance metrics structure"""
        checker = MLSystemHealthChecker()
        performance = checker.model_performance
        
        assert 'current_auc' in performance
        assert 'baseline_auc' in performance
        assert 'auc_scores' in performance
        assert len(performance['auc_scores']) == 30
    
    def test_feature_metrics_consistency(self):
        """Test feature metrics arrays have consistent lengths"""
        checker = MLSystemHealthChecker()
        metrics = checker.feature_metrics
        
        num_features = len(metrics['feature_names'])
        assert len(metrics['missing_value_rates']) == num_features
        assert len(metrics['drift_scores']) == num_features
        assert len(metrics['feature_importance']) == num_features

class TestSystemHealthAnalysis:
    """Test system health analysis functionality"""
    
    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance for testing"""
        return MLSystemHealthChecker()
    
    def test_analyze_system_health_basic(self, health_checker):
        """Test basic health analysis functionality"""
        result = analyze_system_health(health_checker)
        
        assert isinstance(result, dict)
        assert 'overall_health_score' in result
        assert 'critical_issues' in result
        assert 'recommendations' in result
        assert 'priority_actions' in result
    
    def test_health_score_range(self, health_checker):
        """Test health score is within valid range"""
        result = analyze_system_health(health_checker)
        health_score = result['overall_health_score']
        
        assert 0 <= health_score <= 100
        assert isinstance(health_score, (int, float))
    
    def test_critical_issues_detection(self, health_checker):
        """Test critical issues are properly detected"""
        # Modify metrics to trigger critical issues
        health_checker.system_metrics['disk_usage_percent'] = 95
        health_checker.system_metrics['error_rate_percent'] = 5.0
        
        result = analyze_system_health(health_checker)
        critical_issues = result['critical_issues']
        
        assert len(critical_issues) > 0
        assert any('disk' in issue.lower() for issue in critical_issues)
        assert any('error' in issue.lower() for issue in critical_issues)
    
    def test_recommendations_generation(self, health_checker):
        """Test recommendations are generated appropriately"""
        result = analyze_system_health(health_checker)
        recommendations = result['recommendations']
        
        assert isinstance(recommendations, list)
        # Should have recommendations for the default poor metrics
        assert len(recommendations) > 0

class TestDashboardCreation:
    """Test monitoring dashboard creation"""
    
    @pytest.fixture
    def health_checker(self):
        """Create a health checker instance for testing"""
        return MLSystemHealthChecker()
    
    @patch('matplotlib.pyplot.show')
    def test_dashboard_creation(self, mock_show, health_checker):
        """Test dashboard creation doesn't crash"""
        try:
            create_monitoring_dashboard(health_checker)
            # If we get here, no exception was raised
            assert True
        except Exception as e:
            pytest.fail(f"Dashboard creation failed: {str(e)}")
    
    @patch('matplotlib.pyplot.subplots')
    def test_dashboard_subplot_creation(self, mock_subplots, health_checker):
        """Test dashboard creates proper subplot structure"""
        fig_mock = Mock()
        axes_mock = np.array([[Mock(), Mock()], [Mock(), Mock()]])
        mock_subplots.return_value = (fig_mock, axes_mock)
        
        create_monitoring_dashboard(health_checker)
        
        # Verify subplots was called with correct parameters
        mock_subplots.assert_called_once_with(2, 2, figsize=(15, 10))

class TestOptimizationPlan:
    """Test optimization plan generation"""
    
    @pytest.fixture
    def sample_health_analysis(self):
        """Create sample health analysis for testing"""
        return {
            'overall_health_score': 65,
            'critical_issues': ['Disk usage critical (>90%)', 'High error rate detected'],
            'recommendations': ['Optimize API response time', 'Improve caching strategy'],
            'priority_actions': ['Fix disk space', 'Reduce error rate', 'Optimize API']
        }
    
    def test_optimization_plan_structure(self, sample_health_analysis):
        """Test optimization plan has required structure"""
        plan = generate_optimization_plan(sample_health_analysis)
        
        required_keys = [
            'immediate_fixes', 'short_term_improvements', 
            'long_term_optimizations', 'resource_requirements', 
            'timeline', 'expected_roi', 'risk_assessment'
        ]
        
        for key in required_keys:
            assert key in plan
    
    def test_resource_requirements_structure(self, sample_health_analysis):
        """Test resource requirements have proper structure"""
        plan = generate_optimization_plan(sample_health_analysis)
        resources = plan['resource_requirements']
        
        assert 'immediate_fixes' in resources
        assert 'short_term_improvements' in resources
        assert 'long_term_optimizations' in resources
        
        for category in resources.values():
            assert 'engineering_hours' in category
            assert 'infrastructure_cost' in category
    
    def test_timeline_structure(self, sample_health_analysis):
        """Test timeline has proper structure"""
        plan = generate_optimization_plan(sample_health_analysis)
        timeline = plan['timeline']
        
        assert 'week_1' in timeline
        assert 'weeks_2_4' in timeline
        assert 'months_1_3' in timeline

class TestBusinessImpactCalculation:
    """Test business impact calculation"""
    
    @pytest.fixture
    def sample_optimization_plan(self):
        """Create sample optimization plan for testing"""
        return {
            'resource_requirements': {
                'immediate_fixes': {'engineering_hours': 16, 'infrastructure_cost': 4000},
                'short_term_improvements': {'engineering_hours': 60, 'infrastructure_cost': 15000},
                'long_term_optimizations': {'engineering_hours': 120, 'infrastructure_cost': 30000}
            },
            'expected_roi': 2.5
        }
    
    @pytest.fixture
    def sample_current_metrics(self):
        """Create sample current metrics for testing"""
        return {
            'baseline_monthly_revenue': 150000,
            'operational_cost_monthly': 25000,
            'roi_current': 4.0,
            'roi_target': 6.0
        }
    
    def test_business_impact_structure(self, sample_optimization_plan, sample_current_metrics):
        """Test business impact calculation structure"""
        impact = calculate_business_impact(sample_optimization_plan, sample_current_metrics)
        
        required_keys = [
            'projected_revenue_increase_annual', 'projected_cost_savings_annual',
            'projected_efficiency_improvement_percent', 'total_investment_required',
            'projected_roi', 'payback_period_months'
        ]
        
        for key in required_keys:
            assert key in impact
            assert isinstance(impact[key], (int, float))
    
    def test_roi_calculation(self, sample_optimization_plan, sample_current_metrics):
        """Test ROI calculation is reasonable"""
        impact = calculate_business_impact(sample_optimization_plan, sample_current_metrics)
        
        assert impact['projected_roi'] > 0
        assert impact['payback_period_months'] > 0
        assert impact['total_investment_required'] > 0
    
    def test_revenue_projections(self, sample_optimization_plan, sample_current_metrics):
        """Test revenue projections are positive"""
        impact = calculate_business_impact(sample_optimization_plan, sample_current_metrics)
        
        assert impact['projected_revenue_increase_annual'] > 0
        assert impact['projected_cost_savings_annual'] > 0

class TestIntegration:
    """Integration tests for the complete workflow"""
    
    def test_complete_workflow(self):
        """Test the complete assessment workflow"""
        # Initialize health checker
        checker = MLSystemHealthChecker()
        
        # Run health analysis
        health_analysis = analyze_system_health(checker)
        assert health_analysis is not None
        
        # Generate optimization plan
        optimization_plan = generate_optimization_plan(health_analysis)
        assert optimization_plan is not None
        
        # Calculate business impact
        business_impact = calculate_business_impact(optimization_plan, checker.business_metrics)
        assert business_impact is not None
        
        # Verify the workflow produces reasonable results
        assert 0 <= health_analysis['overall_health_score'] <= 100
        assert business_impact['projected_roi'] > 0
    
    @patch('matplotlib.pyplot.show')
    def test_main_function_execution(self, mock_show):
        """Test main function executes without errors"""
        from exercise import main
        
        try:
            result = main()
            assert result is not None
            assert 'health_analysis' in result
            assert 'optimization_plan' in result
            assert 'business_impact' in result
        except Exception as e:
            pytest.fail(f"Main function execution failed: {str(e)}")

def run_performance_tests():
    """Run performance benchmarks for the assessment"""
    print("🚀 Running Performance Tests...")
    
    import time
    
    # Test health checker initialization time
    start_time = time.time()
    checker = MLSystemHealthChecker()
    init_time = time.time() - start_time
    print(f"✅ Health checker initialization: {init_time:.3f} seconds")
    
    # Test health analysis performance
    start_time = time.time()
    health_analysis = analyze_system_health(checker)
    analysis_time = time.time() - start_time
    print(f"✅ Health analysis: {analysis_time:.3f} seconds")
    
    # Test optimization plan generation
    start_time = time.time()
    optimization_plan = generate_optimization_plan(health_analysis)
    plan_time = time.time() - start_time
    print(f"✅ Optimization plan generation: {plan_time:.3f} seconds")
    
    # Test business impact calculation
    start_time = time.time()
    business_impact = calculate_business_impact(optimization_plan, checker.business_metrics)
    impact_time = time.time() - start_time
    print(f"✅ Business impact calculation: {impact_time:.3f} seconds")
    
    total_time = init_time + analysis_time + plan_time + impact_time
    print(f"📊 Total execution time: {total_time:.3f} seconds")
    
    # Performance assertions
    assert init_time < 1.0, "Initialization too slow"
    assert analysis_time < 0.5, "Analysis too slow"
    assert total_time < 2.0, "Overall execution too slow"
    
    print("🎉 All performance tests passed!")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run performance tests
    run_performance_tests()