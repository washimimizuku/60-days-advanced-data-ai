"""
Day 55: AWS Deep Dive - Comprehensive Test Suite
Tests for AWS infrastructure components and deployment patterns
"""

import pytest
import boto3
import json
import time
from unittest.mock import Mock, patch, MagicMock
from moto import mock_sagemaker, mock_ecs, mock_lambda, mock_s3, mock_iam
from datetime import datetime, timedelta

# Import the classes from exercise and solution files
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solution import (
    SageMakerPipeline, ECSDeployment, LambdaPipeline,
    AWSClientManager, MockAWSClient
)


class TestAWSClientManager:
    """Test AWS client management functionality"""
    
    def test_client_manager_initialization(self):
        """Test AWS client manager initialization"""
        manager = AWSClientManager(region='us-west-2')
        assert manager.region == 'us-west-2'
        assert manager.clients == {}
    
    def test_get_client_creation(self):
        """Test client creation and caching"""
        manager = AWSClientManager(region='us-west-2')
        
        # Mock the session to avoid credential issues
        with patch.object(manager, 'session') as mock_session:
            mock_client = Mock()
            mock_session.client.return_value = mock_client
            
            client1 = manager.get_client('s3')
            client2 = manager.get_client('s3')
            
            # Should return the same cached client
            assert client1 is client2
            mock_session.client.assert_called_once_with('s3', region_name='us-west-2')
    
    def test_mock_client_fallback(self):
        """Test fallback to mock client when credentials unavailable"""
        manager = AWSClientManager(region='us-west-2')
        
        # Force credential error
        with patch.object(manager.session, 'client', side_effect=Exception("No credentials")):
            client = manager.get_client('sagemaker')
            assert isinstance(client, MockAWSClient)


class TestMockAWSClient:
    """Test mock AWS client functionality"""
    
    def test_mock_client_initialization(self):
        """Test mock client initialization"""
        client = MockAWSClient('sagemaker', 'us-west-2')
        assert client.service_name == 'sagemaker'
        assert client.region == 'us-west-2'
        assert client.call_count == 0
    
    def test_mock_api_calls(self):
        """Test mock API call handling"""
        client = MockAWSClient('sagemaker', 'us-west-2')
        
        # Test training job creation
        response = client.create_training_job(TrainingJobName='test-job')
        assert 'TrainingJobArn' in response
        assert client.call_count == 1
        
        # Test generic call
        response = client.describe_training_job(TrainingJobName='test-job')
        assert response['ResponseMetadata']['HTTPStatusCode'] == 200
        assert client.call_count == 2


@mock_sagemaker
class TestSageMakerPipeline:
    """Test SageMaker ML pipeline functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.role_arn = 'arn:aws:iam::123456789012:role/SageMakerRole'
        self.bucket_name = 'test-ml-bucket'
        self.pipeline = SageMakerPipeline(
            role_arn=self.role_arn,
            bucket_name=self.bucket_name,
            region='us-east-1'  # moto requires us-east-1
        )
    
    def test_pipeline_initialization(self):
        """Test SageMaker pipeline initialization"""
        assert self.pipeline.role_arn == self.role_arn
        assert self.pipeline.bucket_name == self.bucket_name
        assert self.pipeline.region == 'us-east-1'
    
    def test_create_training_job(self):
        """Test training job creation"""
        job_name = 'test-training-job'
        algorithm_spec = {
            'TrainingImage': '382416733822.dkr.ecr.us-east-1.amazonaws.com/sklearn:latest',
            'TrainingInputMode': 'File'
        }
        input_config = {
            'ChannelName': 'training',
            'DataSource': {
                'S3DataSource': {
                    'S3DataType': 'S3Prefix',
                    'S3Uri': 's3://test-bucket/data/',
                    'S3DataDistributionType': 'FullyReplicated'
                }
            }
        }
        output_config = {
            'S3OutputPath': 's3://test-bucket/output/'
        }
        
        # Mock the training job creation
        with patch.object(self.pipeline.sagemaker, 'create_training_job') as mock_create:
            mock_create.return_value = {
                'TrainingJobArn': f'arn:aws:sagemaker:us-east-1:123456789012:training-job/{job_name}'
            }
            
            result = self.pipeline.create_training_job(
                job_name=job_name,
                algorithm_spec=algorithm_spec,
                input_data_config=input_config,
                output_data_config=output_config
            )
            
            assert job_name in result
            mock_create.assert_called_once()
    
    def test_create_model(self):
        """Test model creation"""
        model_name = 'test-model'
        model_artifacts_url = 's3://test-bucket/model.tar.gz'
        image_uri = '382416733822.dkr.ecr.us-east-1.amazonaws.com/sklearn:latest'
        
        with patch.object(self.pipeline.sagemaker, 'create_model') as mock_create:
            mock_create.return_value = {
                'ModelArn': f'arn:aws:sagemaker:us-east-1:123456789012:model/{model_name}'
            }
            
            result = self.pipeline.create_model(
                model_name=model_name,
                model_artifacts_url=model_artifacts_url,
                image_uri=image_uri
            )
            
            assert model_name in result
            mock_create.assert_called_once()
    
    def test_create_endpoint_config(self):
        """Test endpoint configuration creation"""
        config_name = 'test-endpoint-config'
        model_name = 'test-model'
        
        with patch.object(self.pipeline.sagemaker, 'create_endpoint_config') as mock_create:
            mock_create.return_value = {
                'EndpointConfigArn': f'arn:aws:sagemaker:us-east-1:123456789012:endpoint-config/{config_name}'
            }
            
            result = self.pipeline.create_endpoint_config(
                config_name=config_name,
                model_name=model_name
            )
            
            assert config_name in result
            mock_create.assert_called_once()
    
    def test_monitor_training_job(self):
        """Test training job monitoring"""
        job_name = 'test-training-job'
        
        with patch.object(self.pipeline.sagemaker, 'describe_training_job') as mock_describe:
            mock_describe.return_value = {
                'TrainingJobName': job_name,
                'TrainingJobStatus': 'InProgress',
                'CreationTime': datetime.now(),
                'BillableTimeInSeconds': 3600
            }
            
            result = self.pipeline.monitor_training_job(job_name)
            
            assert result['job_name'] == job_name
            assert result['status'] == 'InProgress'
            assert 'billable_time_seconds' in result
            mock_describe.assert_called_once()


@mock_ecs
class TestECSDeployment:
    """Test ECS deployment functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.cluster_name = 'test-cluster'
        self.deployment = ECSDeployment(
            cluster_name=self.cluster_name,
            region='us-east-1'
        )
    
    def test_deployment_initialization(self):
        """Test ECS deployment initialization"""
        assert self.deployment.cluster_name == self.cluster_name
        assert self.deployment.region == 'us-east-1'
    
    def test_create_cluster(self):
        """Test ECS cluster creation"""
        cluster_config = {
            'capacityProviders': ['FARGATE'],
            'settings': [{'name': 'containerInsights', 'value': 'enabled'}]
        }
        
        with patch.object(self.deployment.ecs, 'create_cluster') as mock_create:
            mock_create.return_value = {
                'cluster': {
                    'clusterArn': f'arn:aws:ecs:us-east-1:123456789012:cluster/{self.cluster_name}'
                }
            }
            
            result = self.deployment.create_cluster(cluster_config)
            
            assert self.cluster_name in result
            mock_create.assert_called_once()
    
    def test_create_task_definition(self):
        """Test task definition creation"""
        family = 'test-task'
        container_definitions = [{
            'name': 'test-container',
            'image': 'nginx:latest',
            'memory': 512,
            'essential': True
        }]
        
        with patch.object(self.deployment.ecs, 'register_task_definition') as mock_register:
            mock_register.return_value = {
                'taskDefinition': {
                    'taskDefinitionArn': f'arn:aws:ecs:us-east-1:123456789012:task-definition/{family}:1'
                }
            }
            
            with patch.object(self.deployment, '_create_log_group'):
                result = self.deployment.create_task_definition(
                    family=family,
                    container_definitions=container_definitions
                )
                
                assert family in result
                mock_register.assert_called_once()
    
    def test_create_service(self):
        """Test ECS service creation"""
        service_name = 'test-service'
        task_definition = 'test-task:1'
        subnets = ['subnet-12345']
        security_groups = ['sg-12345']
        
        with patch.object(self.deployment.ecs, 'create_service') as mock_create:
            mock_create.return_value = {
                'service': {
                    'serviceArn': f'arn:aws:ecs:us-east-1:123456789012:service/{service_name}'
                }
            }
            
            with patch.object(self.deployment, 'setup_auto_scaling'):
                result = self.deployment.create_service(
                    service_name=service_name,
                    task_definition=task_definition,
                    desired_count=2,
                    subnets=subnets,
                    security_groups=security_groups
                )
                
                assert service_name in result
                mock_create.assert_called_once()
    
    def test_deploy_application(self):
        """Test complete application deployment"""
        app_config = {
            'service_name': 'test-app',
            'image_uri': 'test-image:latest',
            'subnets': ['subnet-12345'],
            'security_groups': ['sg-12345'],
            'vpc_id': 'vpc-12345'
        }
        
        with patch.object(self.deployment, 'create_cluster') as mock_cluster, \
             patch.object(self.deployment, 'create_load_balancer') as mock_lb, \
             patch.object(self.deployment, 'create_task_definition') as mock_task, \
             patch.object(self.deployment, 'create_service') as mock_service:
            
            mock_cluster.return_value = 'cluster-arn'
            mock_lb.return_value = ('lb-arn', 'tg-arn')
            mock_task.return_value = 'task-arn'
            mock_service.return_value = 'service-arn'
            
            result = self.deployment.deploy_application(app_config)
            
            assert result['status'] == 'deployed'
            assert result['cluster_arn'] == 'cluster-arn'
            assert result['service_arn'] == 'service-arn'


@mock_lambda
@mock_s3
class TestLambdaPipeline:
    """Test Lambda pipeline functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.pipeline = LambdaPipeline(region='us-east-1')
    
    def test_pipeline_initialization(self):
        """Test Lambda pipeline initialization"""
        assert self.pipeline.region == 'us-east-1'
    
    def test_create_lambda_function(self):
        """Test Lambda function creation"""
        function_name = 'test-function'
        code = {'ZipFile': b'test code'}
        handler = 'lambda_function.lambda_handler'
        runtime = 'python3.9'
        role_arn = 'arn:aws:iam::123456789012:role/LambdaRole'
        
        with patch.object(self.pipeline.lambda_client, 'create_function') as mock_create, \
             patch.object(self.pipeline.lambda_client, 'put_reserved_concurrency_config'):
            
            mock_create.return_value = {
                'FunctionArn': f'arn:aws:lambda:us-east-1:123456789012:function:{function_name}'
            }
            
            result = self.pipeline.create_lambda_function(
                function_name=function_name,
                code=code,
                handler=handler,
                runtime=runtime,
                role_arn=role_arn
            )
            
            assert function_name in result
            mock_create.assert_called_once()
    
    def test_create_s3_trigger(self):
        """Test S3 trigger creation"""
        bucket_name = 'test-bucket'
        function_arn = 'arn:aws:lambda:us-east-1:123456789012:function:test-function'
        
        with patch.object(self.pipeline.lambda_client, 'add_permission') as mock_permission, \
             patch.object(self.pipeline.s3, 'put_bucket_notification_configuration') as mock_notification:
            
            result = self.pipeline.create_s3_trigger(
                bucket_name=bucket_name,
                function_arn=function_arn
            )
            
            assert isinstance(result, str)
            mock_permission.assert_called_once()
            mock_notification.assert_called_once()
    
    def test_create_step_function(self):
        """Test Step Functions state machine creation"""
        state_machine_name = 'test-state-machine'
        definition = {
            'Comment': 'Test state machine',
            'StartAt': 'HelloWorld',
            'States': {
                'HelloWorld': {
                    'Type': 'Pass',
                    'Result': 'Hello World!',
                    'End': True
                }
            }
        }
        role_arn = 'arn:aws:iam::123456789012:role/StepFunctionsRole'
        
        with patch.object(self.pipeline.stepfunctions, 'create_state_machine') as mock_create:
            mock_create.return_value = {
                'stateMachineArn': f'arn:aws:states:us-east-1:123456789012:stateMachine:{state_machine_name}'
            }
            
            result = self.pipeline.create_step_function(
                state_machine_name=state_machine_name,
                definition=definition,
                role_arn=role_arn
            )
            
            assert state_machine_name in result
            mock_create.assert_called_once()
    
    def test_deploy_pipeline(self):
        """Test complete pipeline deployment"""
        pipeline_config = {
            'functions': [{
                'name': 'test-function',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'role_arn': 'arn:aws:iam::123456789012:role/LambdaRole'
            }],
            's3_triggers': [{
                'bucket': 'test-bucket',
                'function': 'test-function'
            }]
        }
        
        with patch.object(self.pipeline, 'create_lambda_function') as mock_function, \
             patch.object(self.pipeline, 'create_s3_trigger') as mock_trigger:
            
            mock_function.return_value = 'function-arn'
            mock_trigger.return_value = 'trigger-id'
            
            result = self.pipeline.deploy_pipeline(pipeline_config)
            
            assert result['status'] == 'deployed'
            assert 'test-function' in result['functions']
            assert 'test-bucket' in result['triggers']
    
    def test_monitor_pipeline(self):
        """Test pipeline monitoring"""
        pipeline_name = 'test-pipeline'
        
        with patch.object(self.pipeline.client_manager, 'get_client') as mock_get_client:
            mock_cloudwatch = Mock()
            mock_cloudwatch.get_metric_statistics.return_value = {
                'Datapoints': [
                    {'Timestamp': datetime.now(), 'Sum': 100, 'Average': 1.5}
                ]
            }
            mock_get_client.return_value = mock_cloudwatch
            
            result = self.pipeline.monitor_pipeline(pipeline_name)
            
            assert result['pipeline_name'] == pipeline_name
            assert 'lambda_metrics' in result
            assert 'stepfunctions_metrics' in result


class TestAWSIntegration:
    """Test AWS service integration scenarios"""
    
    def test_sagemaker_ecs_integration(self):
        """Test SageMaker and ECS integration"""
        # Mock SageMaker endpoint
        sagemaker_pipeline = SageMakerPipeline(
            role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
            bucket_name='test-bucket'
        )
        
        # Mock ECS deployment
        ecs_deployment = ECSDeployment('test-cluster')
        
        with patch.object(sagemaker_pipeline.sagemaker, 'create_endpoint') as mock_endpoint, \
             patch.object(ecs_deployment.ecs, 'create_service') as mock_service:
            
            mock_endpoint.return_value = {'EndpointArn': 'endpoint-arn'}
            mock_service.return_value = {'service': {'serviceArn': 'service-arn'}}
            
            # Deploy SageMaker endpoint
            endpoint_arn = sagemaker_pipeline.deploy_endpoint('test-endpoint', 'test-config')
            
            # Deploy ECS service that calls the endpoint
            app_config = {
                'service_name': 'ml-api',
                'image_uri': 'ml-api:latest',
                'subnets': ['subnet-12345'],
                'security_groups': ['sg-12345'],
                'vpc_id': 'vpc-12345',
                'environment_variables': {
                    'SAGEMAKER_ENDPOINT': 'test-endpoint'
                }
            }
            
            with patch.object(ecs_deployment, 'create_cluster'), \
                 patch.object(ecs_deployment, 'create_load_balancer'), \
                 patch.object(ecs_deployment, 'create_task_definition'), \
                 patch.object(ecs_deployment, 'create_service'):
                
                deployment_result = ecs_deployment.deploy_application(app_config)
                
                assert 'endpoint-arn' in endpoint_arn
                assert deployment_result['status'] == 'deployed'
    
    def test_lambda_s3_sagemaker_pipeline(self):
        """Test Lambda, S3, and SageMaker pipeline integration"""
        lambda_pipeline = LambdaPipeline()
        
        pipeline_config = {
            'functions': [{
                'name': 'data-preprocessor',
                'handler': 'preprocess.lambda_handler',
                'runtime': 'python3.9',
                'role_arn': 'arn:aws:iam::123456789012:role/LambdaRole',
                'environment_vars': {
                    'SAGEMAKER_ROLE': 'arn:aws:iam::123456789012:role/SageMakerRole',
                    'S3_BUCKET': 'ml-data-bucket'
                }
            }],
            's3_triggers': [{
                'bucket': 'raw-data-bucket',
                'function': 'data-preprocessor',
                'suffix_filter': '.csv'
            }],
            'state_machine': {
                'name': 'ml-training-pipeline',
                'definition': {
                    'Comment': 'ML training pipeline',
                    'StartAt': 'PreprocessData',
                    'States': {
                        'PreprocessData': {
                            'Type': 'Task',
                            'Resource': 'arn:aws:lambda:us-east-1:123456789012:function:data-preprocessor',
                            'Next': 'TrainModel'
                        },
                        'TrainModel': {
                            'Type': 'Task',
                            'Resource': 'arn:aws:states:::sagemaker:createTrainingJob.sync',
                            'End': True
                        }
                    }
                },
                'role_arn': 'arn:aws:iam::123456789012:role/StepFunctionsRole'
            }
        }
        
        with patch.object(lambda_pipeline, 'create_lambda_function') as mock_function, \
             patch.object(lambda_pipeline, 'create_s3_trigger') as mock_trigger, \
             patch.object(lambda_pipeline, 'create_step_function') as mock_state_machine:
            
            mock_function.return_value = 'function-arn'
            mock_trigger.return_value = 'trigger-id'
            mock_state_machine.return_value = 'state-machine-arn'
            
            result = lambda_pipeline.deploy_pipeline(pipeline_config)
            
            assert result['status'] == 'deployed'
            assert result['state_machine'] == 'state-machine-arn'
            assert 'data-preprocessor' in result['functions']


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_sagemaker_training_job_failure(self):
        """Test handling of training job failures"""
        pipeline = SageMakerPipeline(
            role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
            bucket_name='test-bucket'
        )
        
        with patch.object(pipeline.sagemaker, 'create_training_job') as mock_create:
            mock_create.side_effect = Exception("Training job creation failed")
            
            with pytest.raises(Exception) as exc_info:
                pipeline.create_training_job(
                    job_name='test-job',
                    algorithm_spec={'TrainingImage': 'test-image'},
                    input_data_config={'ChannelName': 'training'},
                    output_data_config={'S3OutputPath': 's3://test/output'}
                )
            
            assert "Training job creation failed" in str(exc_info.value)
    
    def test_ecs_deployment_failure(self):
        """Test handling of ECS deployment failures"""
        deployment = ECSDeployment('test-cluster')
        
        app_config = {
            'service_name': 'test-app',
            'image_uri': 'test-image:latest',
            'subnets': ['subnet-12345'],
            'security_groups': ['sg-12345'],
            'vpc_id': 'vpc-12345'
        }
        
        with patch.object(deployment, 'create_cluster') as mock_cluster:
            mock_cluster.side_effect = Exception("Cluster creation failed")
            
            result = deployment.deploy_application(app_config)
            
            assert result['status'] == 'failed'
            assert 'error' in result
    
    def test_lambda_pipeline_partial_failure(self):
        """Test handling of partial pipeline deployment failures"""
        pipeline = LambdaPipeline()
        
        pipeline_config = {
            'functions': [{
                'name': 'test-function',
                'handler': 'lambda_function.lambda_handler',
                'runtime': 'python3.9',
                'role_arn': 'arn:aws:iam::123456789012:role/LambdaRole'
            }],
            's3_triggers': [{
                'bucket': 'test-bucket',
                'function': 'test-function'
            }]
        }
        
        with patch.object(pipeline, 'create_lambda_function') as mock_function, \
             patch.object(pipeline, 'create_s3_trigger') as mock_trigger:
            
            mock_function.return_value = 'function-arn'
            mock_trigger.side_effect = Exception("S3 trigger creation failed")
            
            result = pipeline.deploy_pipeline(pipeline_config)
            
            assert result['status'] == 'failed'
            assert 'test-function' in result['functions']  # Function should still be created


class TestPerformanceAndScaling:
    """Test performance and scaling scenarios"""
    
    def test_sagemaker_auto_scaling_configuration(self):
        """Test SageMaker endpoint auto-scaling configuration"""
        pipeline = SageMakerPipeline(
            role_arn='arn:aws:iam::123456789012:role/SageMakerRole',
            bucket_name='test-bucket'
        )
        
        with patch.object(pipeline.client_manager, 'get_client') as mock_get_client:
            mock_autoscaling = Mock()
            mock_get_client.return_value = mock_autoscaling
            
            pipeline._setup_endpoint_autoscaling('test-endpoint')
            
            # Verify auto-scaling configuration calls
            mock_autoscaling.register_scalable_target.assert_called_once()
            mock_autoscaling.put_scaling_policy.assert_called_once()
    
    def test_ecs_service_scaling_policy(self):
        """Test ECS service scaling policy configuration"""
        deployment = ECSDeployment('test-cluster')
        
        with patch.object(deployment.client_manager, 'get_client') as mock_get_client:
            mock_autoscaling = Mock()
            mock_autoscaling.put_scaling_policy.return_value = {'PolicyARN': 'policy-arn'}
            mock_get_client.return_value = mock_autoscaling
            
            service_arn = 'arn:aws:ecs:us-east-1:123456789012:service/test-cluster/test-service'
            
            result = deployment.setup_auto_scaling(service_arn, min_capacity=2, max_capacity=20)
            
            assert result == 'policy-arn'
            mock_autoscaling.register_scalable_target.assert_called_once()
            mock_autoscaling.put_scaling_policy.assert_called_once()
    
    def test_lambda_concurrency_configuration(self):
        """Test Lambda concurrency configuration"""
        pipeline = LambdaPipeline()
        
        with patch.object(pipeline.lambda_client, 'create_function') as mock_create, \
             patch.object(pipeline.lambda_client, 'put_reserved_concurrency_config') as mock_concurrency:
            
            mock_create.return_value = {'FunctionArn': 'function-arn'}
            
            pipeline.create_lambda_function(
                function_name='test-function',
                code={'ZipFile': b'test'},
                handler='lambda_function.lambda_handler',
                runtime='python3.9',
                role_arn='arn:aws:iam::123456789012:role/LambdaRole'
            )
            
            # Verify concurrency configuration
            mock_concurrency.assert_called_once_with(
                FunctionName='test-function',
                ReservedConcurrencyLimit=100
            )


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])