"""
Day 36: CI/CD for ML - Comprehensive Test Suite

Tests for ML CI/CD pipeline including data validation, model testing,
deployment strategies, and infrastructure components.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import classes from solution
from solution import (
    MLCICDPipeline,
    create_sample_data,
    train_sample_model
)

class TestDataValidation:
    """Test data validation functionality"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample fraud detection data"""
        return create_sample_data()
    
    @pytest.fixture
    def temp_data_file(self, sample_data):
        """Create temporary data file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        Path(f.name).unlink()
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return MLCICDPipeline()
    
    def test_data_validation_success(self, pipeline, temp_data_file):
        """Test successful data validation"""
        result = pipeline.validate_data_quality(temp_data_file)
        
        assert result['status'] == 'passed'
        assert 'data_not_empty' in result['checks']
        assert result['checks']['data_not_empty'] is True
        assert len(result['errors']) == 0
    
    def test_data_validation_missing_columns(self, pipeline):
        """Test data validation with missing columns"""
        # Create data with missing columns
        incomplete_data = pd.DataFrame({
            'amount': [100, 200, 300],
            'user_age': [25, 30, 35]
        })
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f.name, index=False)
            
            result = pipeline.validate_data_quality(f.name)
            
            assert result['status'] == 'failed'
            assert any('Missing required columns' in error for error in result['errors'])
        
        Path(f.name).unlink()
    
    def test_data_validation_empty_data(self, pipeline):
        """Test data validation with empty data"""
        empty_data = pd.DataFrame()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            empty_data.to_csv(f.name, index=False)
            
            result = pipeline.validate_data_quality(f.name)
            
            assert result['status'] == 'failed'
            assert 'Data is empty' in result['errors']
        
        Path(f.name).unlink()
    
    def test_data_validation_high_missing_values(self, pipeline, sample_data):
        """Test data validation with high missing values"""
        # Introduce high missing values
        corrupted_data = sample_data.copy()
        corrupted_data.loc[:500, 'amount'] = np.nan  # 50% missing
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            corrupted_data.to_csv(f.name, index=False)
            
            result = pipeline.validate_data_quality(f.name)
            
            assert result['status'] == 'failed'
            assert any('missing values' in error for error in result['errors'])
        
        Path(f.name).unlink()

class TestModelTesting:
    """Test model testing functionality"""
    
    @pytest.fixture
    def trained_model_info(self):
        """Create trained model for testing"""
        return train_sample_model()
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return MLCICDPipeline()
    
    def test_model_testing_success(self, pipeline, trained_model_info):
        """Test successful model testing"""
        result = pipeline.run_model_tests(
            trained_model_info['model_path'],
            trained_model_info['test_data_path']
        )
        
        assert result['status'] == 'passed'
        assert 'model_loading' in result['tests']
        assert result['tests']['model_loading'] is True
        assert 'metrics' in result
        assert 'accuracy' in result['metrics']
    
    def test_model_testing_performance_threshold(self, pipeline, trained_model_info):
        """Test model performance threshold validation"""
        # Modify config to have very high thresholds
        pipeline.config['model_performance_thresholds']['min_accuracy'] = 0.99
        
        result = pipeline.run_model_tests(
            trained_model_info['model_path'],
            trained_model_info['test_data_path']
        )
        
        assert result['status'] == 'failed'
        assert any('Accuracy' in error and 'below threshold' in error for error in result['errors'])
    
    def test_model_testing_missing_file(self, pipeline):
        """Test model testing with missing model file"""
        result = pipeline.run_model_tests(
            'nonexistent_model.pkl',
            'nonexistent_data.csv'
        )
        
        assert result['status'] == 'failed'
        assert len(result['errors']) > 0

class TestBlueGreenDeployment:
    """Test blue-green deployment functionality"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return MLCICDPipeline()
    
    @pytest.fixture
    def deployment_config(self):
        """Create deployment configuration"""
        return {
            'cluster_name': 'test-cluster',
            'service_name': 'test-service',
            'task_definition_arn': 'arn:aws:ecs:us-west-2:123456789012:task-definition/test:1',
            'target_group_arn': 'arn:aws:elasticloadbalancing:us-west-2:123456789012:targetgroup/test/1234567890123456'
        }
    
    def test_blue_green_deployment_success(self, pipeline, deployment_config):
        """Test successful blue-green deployment"""
        with patch.object(pipeline, '_create_green_service') as mock_create, \
             patch.object(pipeline, '_wait_for_service_healthy') as mock_wait, \
             patch.object(pipeline, '_run_comprehensive_health_checks') as mock_health, \
             patch.object(pipeline, '_switch_traffic_to_green') as mock_switch, \
             patch.object(pipeline, '_monitor_deployment_performance') as mock_monitor:
            
            # Mock successful responses
            mock_create.return_value = {'success': True}
            mock_wait.return_value = True
            mock_health.return_value = {'success': True, 'errors': []}
            mock_switch.return_value = {'success': True}
            mock_monitor.return_value = {'success': True}
            
            result = pipeline.deploy_model_blue_green('test_model.pkl', deployment_config)
            
            assert result['status'] == 'success'
            assert 'deployment_id' in result
            assert 'green_service_created' in result['steps']
    
    def test_blue_green_deployment_health_check_failure(self, pipeline, deployment_config):
        """Test blue-green deployment with health check failure"""
        with patch.object(pipeline, '_create_green_service') as mock_create, \
             patch.object(pipeline, '_wait_for_service_healthy') as mock_wait, \
             patch.object(pipeline, '_run_comprehensive_health_checks') as mock_health:
            
            # Mock health check failure
            mock_create.return_value = {'success': True}
            mock_wait.return_value = True
            mock_health.return_value = {'success': False, 'errors': ['Health check failed']}
            
            result = pipeline.deploy_model_blue_green('test_model.pkl', deployment_config)
            
            assert result['status'] == 'failed'
            assert 'Health check failed' in result['errors']
    
    def test_blue_green_deployment_performance_monitoring_failure(self, pipeline, deployment_config):
        """Test blue-green deployment with performance monitoring failure"""
        with patch.object(pipeline, '_create_green_service') as mock_create, \
             patch.object(pipeline, '_wait_for_service_healthy') as mock_wait, \
             patch.object(pipeline, '_run_comprehensive_health_checks') as mock_health, \
             patch.object(pipeline, '_switch_traffic_to_green') as mock_switch, \
             patch.object(pipeline, '_monitor_deployment_performance') as mock_monitor, \
             patch.object(pipeline, '_rollback_to_blue_service') as mock_rollback:
            
            # Mock performance monitoring failure
            mock_create.return_value = {'success': True}
            mock_wait.return_value = True
            mock_health.return_value = {'success': True, 'errors': []}
            mock_switch.return_value = {'success': True}
            mock_monitor.return_value = {'success': False, 'reason': 'High error rate'}
            
            result = pipeline.deploy_model_blue_green('test_model.pkl', deployment_config)
            
            assert result['status'] == 'rolled_back'
            assert 'Performance issues: High error rate' in result['errors']
            mock_rollback.assert_called_once()

class TestPerformanceMonitoring:
    """Test performance monitoring functionality"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return MLCICDPipeline()
    
    def test_performance_monitoring_success(self, pipeline):
        """Test successful performance monitoring"""
        with patch('time.time') as mock_time:
            # Mock time progression
            mock_time.side_effect = [0, 30, 60, 90, 120, 150, 180]  # 3 minutes
            
            result = pipeline.monitor_model_performance(
                'http://test-endpoint.com',
                duration_minutes=3
            )
            
            assert 'total_requests' in result
            assert 'error_rate' in result
            assert 'avg_response_time' in result
            assert result['error_rate'] <= 0.05  # Should be low error rate
    
    def test_performance_monitoring_with_errors(self, pipeline):
        """Test performance monitoring handles errors gracefully"""
        result = pipeline.monitor_model_performance(
            'http://invalid-endpoint.com',
            duration_minutes=1
        )
        
        # Should complete without crashing
        assert 'total_requests' in result or 'error' in result

class TestInfrastructureSetup:
    """Test infrastructure setup functionality"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return MLCICDPipeline()
    
    def test_infrastructure_setup_success(self, pipeline):
        """Test successful infrastructure setup"""
        with patch.object(pipeline, '_apply_terraform') as mock_terraform:
            mock_terraform.return_value = {
                'status': 'success',
                'outputs': {
                    'vpc_id': 'vpc-12345678',
                    'ecs_cluster_name': 'test-cluster',
                    's3_artifacts_bucket': 'test-bucket'
                }
            }
            
            result = pipeline.setup_infrastructure('staging')
            
            assert result['status'] == 'success'
            assert 'outputs' in result
            assert result['outputs']['vpc_id'] == 'vpc-12345678'
    
    def test_terraform_config_generation(self, pipeline):
        """Test Terraform configuration generation"""
        config = pipeline._generate_terraform_config('staging')
        
        assert 'terraform {' in config
        assert 'provider "aws"' in config
        assert 'resource "aws_vpc"' in config
        assert 'resource "aws_ecs_cluster"' in config

class TestMonitoringDashboard:
    """Test monitoring dashboard functionality"""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance"""
        return MLCICDPipeline()
    
    def test_dashboard_creation_success(self, pipeline):
        """Test successful dashboard creation"""
        with patch.object(pipeline, '_create_cloudwatch_alarms') as mock_alarms, \
             patch.object(pipeline, '_publish_initial_metrics') as mock_metrics:
            
            mock_alarms.return_value = ['alarm1', 'alarm2', 'alarm3']
            
            result = pipeline.create_monitoring_dashboard('test-pipeline')
            
            assert result['status'] == 'success'
            assert 'dashboard_name' in result
            assert 'dashboard_url' in result
            assert len(result['alarms_created']) == 3
    
    def test_dashboard_creation_handles_errors(self, pipeline):
        """Test dashboard creation handles errors gracefully"""
        with patch.object(pipeline, '_create_cloudwatch_alarms') as mock_alarms:
            mock_alarms.side_effect = Exception('CloudWatch error')
            
            result = pipeline.create_monitoring_dashboard('test-pipeline')
            
            assert result['status'] == 'failed'
            assert 'error' in result

class TestIntegration:
    """Integration tests for complete pipeline"""
    
    def test_complete_pipeline_workflow(self):
        """Test complete pipeline workflow from data to deployment"""
        # Create temporary directory for test
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize pipeline
            pipeline = MLCICDPipeline()
            
            # Generate sample data
            data = create_sample_data()
            data_path = temp_path / 'test_data.csv'
            data.to_csv(data_path, index=False)
            
            # Test data validation
            validation_result = pipeline.validate_data_quality(str(data_path))
            assert validation_result['status'] == 'passed'
            
            # Train model
            model_info = train_sample_model()
            
            # Test model validation
            test_result = pipeline.run_model_tests(
                model_info['model_path'],
                model_info['test_data_path']
            )
            assert test_result['status'] == 'passed'
            
            # Test infrastructure setup
            infra_result = pipeline.setup_infrastructure('test')
            assert infra_result['status'] == 'success'
            
            # Test monitoring dashboard
            dashboard_result = pipeline.create_monitoring_dashboard('test-pipeline')
            assert dashboard_result['status'] == 'success'

class TestPerformance:
    """Performance tests for pipeline components"""
    
    def test_data_validation_performance(self):
        """Test data validation performance with large dataset"""
        # Create large dataset
        large_data = create_sample_data()
        # Replicate to make it larger
        large_data = pd.concat([large_data] * 10, ignore_index=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            large_data.to_csv(f.name, index=False)
            
            pipeline = MLCICDPipeline()
            
            start_time = datetime.now()
            result = pipeline.validate_data_quality(f.name)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            
            # Should complete within reasonable time
            assert duration < 30.0  # 30 seconds
            assert result['status'] == 'passed'
        
        Path(f.name).unlink()
    
    def test_model_testing_performance(self):
        """Test model testing performance"""
        model_info = train_sample_model()
        pipeline = MLCICDPipeline()
        
        start_time = datetime.now()
        result = pipeline.run_model_tests(
            model_info['model_path'],
            model_info['test_data_path']
        )
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        
        # Should complete within reasonable time
        assert duration < 60.0  # 60 seconds
        assert result['status'] == 'passed'

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])