"""
Day 55: AWS Deep Dive - Cloud Infrastructure for Data & AI Systems
Exercises for building production-ready AWS infrastructure and services
"""

import boto3
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AWSResource:
    """AWS resource representation"""
    resource_type: str
    resource_id: str
    region: str
    status: str
    created_at: datetime
    tags: Dict[str, str]


class MockAWSClient:
    """Mock AWS client for exercises (simulates AWS API calls)"""
    
    def __init__(self, service_name: str, region: str = 'us-west-2'):
        self.service_name = service_name
        self.region = region
        self.resources = {}
    
    def create_resource(self, resource_type: str, config: Dict) -> str:
        """Simulate resource creation"""
        resource_id = f"{resource_type}-{int(time.time())}"
        self.resources[resource_id] = {
            'type': resource_type,
            'config': config,
            'status': 'creating',
            'created_at': datetime.now()
        }
        logger.info(f"Created {resource_type}: {resource_id}")
        return resource_id
    
    def describe_resource(self, resource_id: str) -> Dict:
        """Simulate resource description"""
        return self.resources.get(resource_id, {})
    
    def list_resources(self) -> List[Dict]:
        """List all resources"""
        return list(self.resources.values())


# Exercise 1: SageMaker ML Pipeline
def exercise_1_sagemaker_pipeline():
    """
    Exercise 1: Build a complete SageMaker ML pipeline
    
    TODO: Complete the SageMakerPipeline class
    """
    print("=== Exercise 1: SageMaker ML Pipeline ===")
    
    class SageMakerPipeline:
        def __init__(self, role_arn: str, bucket_name: str):
            self.role_arn = role_arn
            self.bucket_name = bucket_name
            self.sagemaker_client = MockAWSClient('sagemaker')
            self.s3_client = MockAWSClient('s3')
        
        def create_training_job(self, job_name: str, algorithm_spec: Dict, 
                              input_data_config: Dict, output_data_config: Dict) -> str:
            """Create SageMaker training job"""
            training_config = {
                'TrainingJobName': job_name,
                'RoleArn': self.role_arn,
                'AlgorithmSpecification': algorithm_spec,
                'InputDataConfig': [input_data_config],
                'OutputDataConfig': output_data_config,
                'ResourceConfig': {
                    'InstanceType': 'ml.m5.large',
                    'InstanceCount': 1,
                    'VolumeSizeInGB': 30
                },
                'StoppingCondition': {'MaxRuntimeInSeconds': 3600}
            }
            
            response = self.sagemaker_client.create_resource('training_job', training_config)
            return f"arn:aws:sagemaker:{self.sagemaker_client.region}:123456789012:training-job/{job_name}"
        
        def create_model(self, model_name: str, model_artifacts_url: str, 
                        image_uri: str) -> str:
            """Create SageMaker model from training artifacts"""
            model_config = {
                'ModelName': model_name,
                'PrimaryContainer': {
                    'Image': image_uri,
                    'ModelDataUrl': model_artifacts_url
                },
                'ExecutionRoleArn': self.role_arn
            }
            
            response = self.sagemaker_client.create_resource('model', model_config)
            return f"arn:aws:sagemaker:{self.sagemaker_client.region}:123456789012:model/{model_name}"
        
        def create_endpoint_config(self, config_name: str, model_name: str, 
                                 instance_type: str, instance_count: int) -> str:
            """Create endpoint configuration"""
            config = {
                'EndpointConfigName': config_name,
                'ProductionVariants': [{
                    'VariantName': 'AllTraffic',
                    'ModelName': model_name,
                    'InitialInstanceCount': instance_count,
                    'InstanceType': instance_type
                }]
            }
            
            response = self.sagemaker_client.create_resource('endpoint_config', config)
            return f"arn:aws:sagemaker:{self.sagemaker_client.region}:123456789012:endpoint-config/{config_name}"
        
        def deploy_endpoint(self, endpoint_name: str, config_name: str) -> str:
            """Deploy model to SageMaker endpoint"""
            endpoint_config = {
                'EndpointName': endpoint_name,
                'EndpointConfigName': config_name
            }
            
            response = self.sagemaker_client.create_resource('endpoint', endpoint_config)
            logger.info(f"Endpoint {endpoint_name} deployment started")
            return f"arn:aws:sagemaker:{self.sagemaker_client.region}:123456789012:endpoint/{endpoint_name}"
        
        def create_pipeline(self, pipeline_name: str, steps: List[Dict]) -> str:
            """Create SageMaker Pipeline for MLOps"""
            pipeline_config = {
                'PipelineName': pipeline_name,
                'PipelineDefinition': json.dumps({
                    'Version': '2020-12-01',
                    'Steps': steps
                }),
                'RoleArn': self.role_arn
            }
            
            response = self.sagemaker_client.create_resource('pipeline', pipeline_config)
            return f"arn:aws:sagemaker:{self.sagemaker_client.region}:123456789012:pipeline/{pipeline_name}"
        
        def monitor_training_job(self, job_name: str) -> Dict:
            """Monitor training job progress"""
            job_info = self.sagemaker_client.describe_resource(job_name)
            
            return {
                'job_name': job_name,
                'status': job_info.get('status', 'InProgress'),
                'creation_time': datetime.now(),
                'metrics': {
                    'train_loss': 0.15,
                    'validation_accuracy': 0.92
                }
            }
    
    # Test SageMaker pipeline
    pipeline = SageMakerPipeline(
        role_arn="arn:aws:iam::123456789012:role/SageMakerRole",
        bucket_name="ml-training-bucket"
    )
    
    print("Testing SageMaker ML Pipeline...")
    print("\n--- Your implementation should create ML pipeline ---")
    
    # Example usage
    # training_job_arn = pipeline.create_training_job(
    #     job_name="text-classification-training",
    #     algorithm_spec={
    #         "TrainingImage": "382416733822.dkr.ecr.us-west-2.amazonaws.com/sklearn:latest",
    #         "TrainingInputMode": "File"
    #     },
    #     input_data_config={
    #         "ChannelName": "training",
    #         "DataSource": {
    #             "S3DataSource": {
    #                 "S3DataType": "S3Prefix",
    #                 "S3Uri": "s3://ml-training-bucket/training-data/",
    #                 "S3DataDistributionType": "FullyReplicated"
    #             }
    #         }
    #     },
    #     output_data_config={
    #         "S3OutputPath": "s3://ml-training-bucket/model-artifacts/"
    #     }
    # )
    # print(f"Training job created: {training_job_arn}")


# Exercise 2: ECS Containerized Application
def exercise_2_ecs_deployment():
    """
    Exercise 2: Deploy containerized RAG system using ECS
    
    TODO: Complete the ECSDeployment class
    """
    print("\n=== Exercise 2: ECS Containerized Application ===")
    
    class ECSDeployment:
        def __init__(self, cluster_name: str, region: str = 'us-west-2'):
            self.cluster_name = cluster_name
            self.region = region
            self.ecs_client = MockAWSClient('ecs', region)
            self.elbv2_client = MockAWSClient('elbv2', region)
            self.ec2_client = MockAWSClient('ec2', region)
        
        def create_cluster(self, cluster_config: Dict) -> str:
            """Create ECS cluster"""
            cluster_params = {
                'clusterName': self.cluster_name,
                'capacityProviders': ['FARGATE', 'FARGATE_SPOT'],
                'defaultCapacityProviderStrategy': [{
                    'capacityProvider': 'FARGATE',
                    'weight': 1
                }],
                'settings': [{'name': 'containerInsights', 'value': 'enabled'}]
            }
            
            response = self.ecs_client.create_resource('cluster', cluster_params)
            return f"arn:aws:ecs:{self.ecs_client.region}:123456789012:cluster/{self.cluster_name}"
        
        def create_task_definition(self, family: str, container_definitions: List[Dict],
                                 cpu: str, memory: str) -> str:
            """Create ECS task definition"""
            task_def_params = {
                'family': family,
                'networkMode': 'awsvpc',
                'requiresCompatibilities': ['FARGATE'],
                'cpu': cpu,
                'memory': memory,
                'containerDefinitions': container_definitions,
                'executionRoleArn': 'arn:aws:iam::123456789012:role/ecsTaskExecutionRole'
            }
            
            response = self.ecs_client.create_resource('task_definition', task_def_params)
            return f"arn:aws:ecs:{self.ecs_client.region}:123456789012:task-definition/{family}:1"
        
        def create_service(self, service_name: str, task_definition: str,
                          desired_count: int, subnets: List[str], 
                          security_groups: List[str]) -> str:
            """Create ECS service"""
            service_params = {
                'cluster': self.cluster_name,
                'serviceName': service_name,
                'taskDefinition': task_definition,
                'desiredCount': desired_count,
                'launchType': 'FARGATE',
                'networkConfiguration': {
                    'awsvpcConfiguration': {
                        'subnets': subnets,
                        'securityGroups': security_groups,
                        'assignPublicIp': 'ENABLED'
                    }
                }
            }
            
            response = self.ecs_client.create_resource('service', service_params)
            return f"arn:aws:ecs:{self.ecs_client.region}:123456789012:service/{self.cluster_name}/{service_name}"
        
        def create_load_balancer(self, lb_name: str, subnets: List[str],
                               security_groups: List[str]) -> str:
            """Create Application Load Balancer"""
            lb_params = {
                'Name': lb_name,
                'Subnets': subnets,
                'SecurityGroups': security_groups,
                'Scheme': 'internet-facing',
                'Type': 'application'
            }
            
            response = self.elbv2_client.create_resource('load_balancer', lb_params)
            lb_arn = f"arn:aws:elasticloadbalancing:{self.elbv2_client.region}:123456789012:loadbalancer/app/{lb_name}/1234567890123456"
            logger.info(f"Load balancer created: {lb_arn}")
            return lb_arn
        
        def setup_auto_scaling(self, service_arn: str, min_capacity: int,
                             max_capacity: int, target_cpu: float) -> str:
            """Configure ECS service auto-scaling"""
            autoscaling_client = MockAWSClient('application-autoscaling', self.region)
            
            # Register scalable target
            target_config = {
                'ServiceNamespace': 'ecs',
                'ResourceId': f'service/{self.cluster_name}/service-name',
                'ScalableDimension': 'ecs:service:DesiredCount',
                'MinCapacity': min_capacity,
                'MaxCapacity': max_capacity
            }
            
            response = autoscaling_client.create_resource('scalable_target', target_config)
            logger.info(f"Auto-scaling configured for {service_arn}")
            return f"arn:aws:application-autoscaling:{self.region}:123456789012:scalable-target/1234"
        
        def deploy_application(self, app_config: Dict) -> Dict:
            """Complete application deployment"""
            deployment_result = {
                'cluster_arn': None,
                'service_arn': None,
                'load_balancer_arn': None,
                'status': 'deploying'
            }
            
            try:
                # Create cluster
                cluster_arn = self.create_cluster({})
                deployment_result['cluster_arn'] = cluster_arn
                
                # Create load balancer
                lb_arn = self.create_load_balancer(
                    f"{self.cluster_name}-alb",
                    app_config.get('subnets', ['subnet-12345']),
                    app_config.get('security_groups', ['sg-12345'])
                )
                deployment_result['load_balancer_arn'] = lb_arn
                
                # Create task definition and service
                container_def = {
                    'name': app_config.get('service_name', 'app'),
                    'image': app_config.get('image_uri', 'nginx:latest'),
                    'memory': 512,
                    'essential': True
                }
                
                task_def_arn = self.create_task_definition(
                    app_config.get('service_name', 'app'),
                    [container_def],
                    app_config.get('cpu', '256'),
                    app_config.get('memory', '512')
                )
                
                service_arn = self.create_service(
                    app_config.get('service_name', 'app'),
                    task_def_arn,
                    app_config.get('desired_count', 1),
                    app_config.get('subnets', ['subnet-12345']),
                    app_config.get('security_groups', ['sg-12345'])
                )
                deployment_result['service_arn'] = service_arn
                deployment_result['status'] = 'deployed'
                
            except Exception as e:
                deployment_result['status'] = 'failed'
                deployment_result['error'] = str(e)
            
            return deployment_result
    
    # Test ECS deployment
    ecs_deployment = ECSDeployment("production-cluster")
    
    print("Testing ECS Deployment...")
    print("\n--- Your implementation should deploy containerized application ---")
    
    # Example configuration
    app_config = {
        "image_uri": "123456789012.dkr.ecr.us-west-2.amazonaws.com/rag-system:latest",
        "cpu": "1024",
        "memory": "2048",
        "port": 8000,
        "environment_variables": {
            "OPENAI_API_KEY": "secret-key",
            "DATABASE_URL": "postgresql://..."
        }
    }
    
    # deployment_result = ecs_deployment.deploy_application(app_config)
    # print(f"Deployment result: {deployment_result}")


# Exercise 3: Serverless Data Processing with Lambda
def exercise_3_lambda_pipeline():
    """
    Exercise 3: Build serverless data processing pipeline
    
    TODO: Complete the LambdaPipeline class
    """
    print("\n=== Exercise 3: Serverless Data Processing Pipeline ===")
    
    class LambdaPipeline:
        def __init__(self, region: str = 'us-west-2'):
            self.region = region
            self.lambda_client = MockAWSClient('lambda', region)
            self.s3_client = MockAWSClient('s3', region)
            self.events_client = MockAWSClient('events', region)
            self.stepfunctions_client = MockAWSClient('stepfunctions', region)
        
        def create_lambda_function(self, function_name: str, code: Dict,
                                 handler: str, runtime: str, role_arn: str) -> str:
            """Create Lambda function"""
            function_config = {
                'FunctionName': function_name,
                'Runtime': runtime,
                'Role': role_arn,
                'Handler': handler,
                'Code': code,
                'Timeout': 300,
                'MemorySize': 512
            }
            
            response = self.lambda_client.create_resource('function', function_config)
            function_arn = f"arn:aws:lambda:{self.lambda_client.region}:123456789012:function:{function_name}"
            logger.info(f"Lambda function created: {function_arn}")
            return function_arn
        
        def create_s3_trigger(self, bucket_name: str, function_arn: str,
                            event_type: str = 's3:ObjectCreated:*') -> str:
            """Create S3 event trigger for Lambda"""
            # Add permission for S3 to invoke Lambda
            permission_config = {
                'FunctionName': function_arn,
                'StatementId': f's3-trigger-{int(time.time())}',
                'Action': 'lambda:InvokeFunction',
                'Principal': 's3.amazonaws.com',
                'SourceArn': f'arn:aws:s3:::{bucket_name}'
            }
            
            self.lambda_client.create_resource('permission', permission_config)
            
            # Configure S3 notification
            notification_config = {
                'Bucket': bucket_name,
                'NotificationConfiguration': {
                    'LambdaConfigurations': [{
                        'LambdaFunctionArn': function_arn,
                        'Events': [event_type]
                    }]
                }
            }
            
            response = self.s3_client.create_resource('notification', notification_config)
            logger.info(f"S3 trigger created for {bucket_name} -> {function_arn}")
            return f"s3-trigger-{bucket_name}-{int(time.time())}"
        
        def create_step_function(self, state_machine_name: str,
                               definition: Dict, role_arn: str) -> str:
            """Create Step Functions state machine"""
            state_machine_config = {
                'name': state_machine_name,
                'definition': json.dumps(definition),
                'roleArn': role_arn,
                'type': 'STANDARD'
            }
            
            response = self.stepfunctions_client.create_resource('state_machine', state_machine_config)
            state_machine_arn = f"arn:aws:states:{self.stepfunctions_client.region}:123456789012:stateMachine:{state_machine_name}"
            logger.info(f"Step Functions state machine created: {state_machine_arn}")
            return state_machine_arn
        
        def create_event_rule(self, rule_name: str, schedule_expression: str,
                            targets: List[Dict]) -> str:
            """Create CloudWatch Events rule"""
            rule_config = {
                'Name': rule_name,
                'ScheduleExpression': schedule_expression,
                'State': 'ENABLED'
            }
            
            response = self.events_client.create_resource('rule', rule_config)
            
            # Add targets
            targets_config = {
                'Rule': rule_name,
                'Targets': [{
                    'Id': str(i + 1),
                    'Arn': target['arn']
                } for i, target in enumerate(targets)]
            }
            
            self.events_client.create_resource('targets', targets_config)
            rule_arn = f"arn:aws:events:{self.events_client.region}:123456789012:rule/{rule_name}"
            logger.info(f"CloudWatch Events rule created: {rule_arn}")
            return rule_arn
        
        def deploy_pipeline(self, pipeline_config: Dict) -> Dict:
            """Deploy complete serverless pipeline"""
            deployment_result = {
                'functions': {},
                'triggers': {},
                'state_machine': None,
                'status': 'deploying'
            }
            
            try:
                # Create Lambda functions
                for func_config in pipeline_config.get('functions', []):
                    function_arn = self.create_lambda_function(
                        function_name=func_config['name'],
                        code={'ZipFile': b'placeholder code'},
                        handler=func_config['handler'],
                        runtime=func_config['runtime'],
                        role_arn=func_config.get('role_arn', 'arn:aws:iam::123456789012:role/LambdaRole')
                    )
                    deployment_result['functions'][func_config['name']] = function_arn
                
                # Create S3 triggers
                for trigger_config in pipeline_config.get('triggers', []):
                    if trigger_config['type'] == 's3':
                        trigger_id = self.create_s3_trigger(
                            bucket_name=trigger_config['bucket'],
                            function_arn=deployment_result['functions'][trigger_config['function']]
                        )
                        deployment_result['triggers'][trigger_config['bucket']] = trigger_id
                
                deployment_result['status'] = 'deployed'
                logger.info("Serverless pipeline deployment completed")
                
            except Exception as e:
                deployment_result['status'] = 'failed'
                deployment_result['error'] = str(e)
            
            return deployment_result
        
        def monitor_pipeline(self, pipeline_name: str) -> Dict:
            """Monitor pipeline execution"""
            cloudwatch_client = MockAWSClient('cloudwatch', self.region)
            
            # Get Lambda metrics
            metrics_config = {
                'Namespace': 'AWS/Lambda',
                'MetricName': 'Invocations',
                'Dimensions': [{'Name': 'FunctionName', 'Value': pipeline_name}]
            }
            
            response = cloudwatch_client.create_resource('metrics', metrics_config)
            
            return {
                'pipeline_name': pipeline_name,
                'status': 'healthy',
                'metrics': {
                    'invocations': 150,
                    'errors': 2,
                    'duration_avg': 1.2
                },
                'last_execution': datetime.now().isoformat()
            }
    
    # Test Lambda pipeline
    lambda_pipeline = LambdaPipeline()
    
    print("Testing Serverless Pipeline...")
    print("\n--- Your implementation should create serverless data pipeline ---")
    
    # Example pipeline configuration
    pipeline_config = {
        "functions": [
            {
                "name": "document-processor",
                "handler": "lambda_function.lambda_handler",
                "runtime": "python3.9",
                "memory": 512,
                "timeout": 300
            },
            {
                "name": "embedding-generator",
                "handler": "embeddings.lambda_handler", 
                "runtime": "python3.9",
                "memory": 1024,
                "timeout": 600
            }
        ],
        "triggers": [
            {
                "type": "s3",
                "bucket": "document-uploads",
                "function": "document-processor"
            }
        ]
    }
    
    # deployment_result = lambda_pipeline.deploy_pipeline(pipeline_config)
    # print(f"Pipeline deployed: {deployment_result}")


# Exercise 4: AWS Security and IAM
def exercise_4_security_setup():
    """
    Exercise 4: Implement AWS security best practices
    
    TODO: Complete the SecurityManager class
    """
    print("\n=== Exercise 4: AWS Security and IAM ===")
    
    class SecurityManager:
        def __init__(self, region: str = 'us-west-2'):
            self.region = region
            self.iam_client = MockAWSClient('iam', region)
            self.kms_client = MockAWSClient('kms', region)
            self.secrets_client = MockAWSClient('secretsmanager', region)
            self.ec2_client = MockAWSClient('ec2', region)
        
        def create_iam_role(self, role_name: str, trust_policy: Dict,
                          permissions_policies: List[str]) -> str:
            """Create IAM role with proper permissions"""
            role_config = {
                'RoleName': role_name,
                'AssumeRolePolicyDocument': json.dumps(trust_policy),
                'Description': f'IAM role for {role_name}',
                'Tags': [
                    {'Key': 'Environment', 'Value': 'production'},
                    {'Key': 'Service', 'Value': 'aws-infrastructure'}
                ]
            }
            
            response = self.iam_client.create_resource('role', role_config)
            
            # Attach policies
            for policy_arn in permissions_policies:
                policy_config = {
                    'RoleName': role_name,
                    'PolicyArn': policy_arn
                }
                self.iam_client.create_resource('role_policy_attachment', policy_config)
            
            role_arn = f"arn:aws:iam::123456789012:role/{role_name}"
            logger.info(f"IAM role created: {role_arn}")
            return role_arn
        
        def create_security_group(self, group_name: str, vpc_id: str,
                                ingress_rules: List[Dict]) -> str:
            """Create security group with minimal access"""
            sg_config = {
                'GroupName': group_name,
                'Description': f'Security group for {group_name}',
                'VpcId': vpc_id
            }
            
            response = self.ec2_client.create_resource('security_group', sg_config)
            sg_id = f"sg-{int(time.time())}"
            
            # Add ingress rules
            for rule in ingress_rules:
                rule_config = {
                    'GroupId': sg_id,
                    'IpPermissions': [rule]
                }
                self.ec2_client.create_resource('security_group_ingress', rule_config)
            
            logger.info(f"Security group created: {sg_id}")
            return sg_id
        
        def setup_encryption(self, service_type: str, resource_arn: str) -> str:
            """Set up encryption for AWS resources"""
            key_config = {
                'Description': f'KMS key for {service_type} encryption',
                'KeyUsage': 'ENCRYPT_DECRYPT',
                'KeySpec': 'SYMMETRIC_DEFAULT',
                'Origin': 'AWS_KMS',
                'Policy': json.dumps({
                    'Version': '2012-10-17',
                    'Statement': [{
                        'Effect': 'Allow',
                        'Principal': {'AWS': f'arn:aws:iam::123456789012:root'},
                        'Action': 'kms:*',
                        'Resource': '*'
                    }]
                })
            }
            
            response = self.kms_client.create_resource('key', key_config)
            key_id = f"key-{int(time.time())}"
            
            # Enable key rotation
            rotation_config = {'KeyId': key_id}
            self.kms_client.create_resource('key_rotation', rotation_config)
            
            logger.info(f"KMS key created for {service_type}: {key_id}")
            return key_id
        
        def create_secret(self, secret_name: str, secret_value: Dict,
                        description: str) -> str:
            """Store secrets in AWS Secrets Manager"""
            secret_config = {
                'Name': secret_name,
                'Description': description,
                'SecretString': json.dumps(secret_value),
                'Tags': [
                    {'Key': 'Environment', 'Value': 'production'},
                    {'Key': 'Service', 'Value': 'secrets-management'}
                ]
            }
            
            response = self.secrets_client.create_resource('secret', secret_config)
            secret_arn = f"arn:aws:secretsmanager:{self.secrets_client.region}:123456789012:secret:{secret_name}"
            logger.info(f"Secret created: {secret_arn}")
            return secret_arn
        
        def setup_vpc_security(self, vpc_config: Dict) -> Dict:
            """Set up secure VPC configuration"""
            vpc_result = {
                'vpc_id': None,
                'private_subnets': [],
                'public_subnets': [],
                'nat_gateway': None,
                'status': 'creating'
            }
            
            try:
                # Create VPC
                vpc_params = {
                    'CidrBlock': vpc_config.get('cidr', '10.0.0.0/16'),
                    'EnableDnsHostnames': True,
                    'EnableDnsSupport': True
                }
                
                response = self.ec2_client.create_resource('vpc', vpc_params)
                vpc_id = f"vpc-{int(time.time())}"
                vpc_result['vpc_id'] = vpc_id
                
                # Create subnets
                for i, cidr in enumerate(['10.0.1.0/24', '10.0.2.0/24']):
                    subnet_config = {
                        'VpcId': vpc_id,
                        'CidrBlock': cidr,
                        'AvailabilityZone': f'us-west-2{chr(97+i)}'
                    }
                    subnet_id = f"subnet-{int(time.time())}-{i}"
                    vpc_result['private_subnets'].append(subnet_id)
                
                vpc_result['status'] = 'created'
                logger.info(f"VPC security setup completed: {vpc_id}")
                
            except Exception as e:
                vpc_result['status'] = 'failed'
                vpc_result['error'] = str(e)
            
            return vpc_result
        
        def audit_security_configuration(self, resources: List[str]) -> Dict:
            """Audit security configuration"""
            audit_result = {
                'resources_audited': len(resources),
                'security_score': 85,
                'findings': [
                    {'type': 'INFO', 'message': 'All IAM roles follow least privilege principle'},
                    {'type': 'WARNING', 'message': 'Some security groups allow broad access'},
                    {'type': 'PASS', 'message': 'Encryption enabled for all resources'}
                ],
                'recommendations': [
                    'Enable MFA for all IAM users',
                    'Implement VPC Flow Logs',
                    'Set up AWS Config rules'
                ],
                'audit_timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"Security audit completed for {len(resources)} resources")
            return audit_result
    
    # Test security setup
    security_manager = SecurityManager()
    
    print("Testing Security Configuration...")
    print("\n--- Your implementation should set up secure AWS environment ---")
    
    # Example security configuration
    security_config = {
        "vpc_cidr": "10.0.0.0/16",
        "private_subnets": ["10.0.1.0/24", "10.0.2.0/24"],
        "public_subnets": ["10.0.101.0/24", "10.0.102.0/24"],
        "enable_nat_gateway": True,
        "enable_vpc_endpoints": True
    }
    
    # vpc_result = security_manager.setup_vpc_security(security_config)
    # print(f"VPC security configured: {vpc_result}")


# Exercise 5: Monitoring and Observability
def exercise_5_monitoring_setup():
    """
    Exercise 5: Set up comprehensive monitoring and observability
    
    TODO: Complete the MonitoringSetup class
    """
    print("\n=== Exercise 5: Monitoring and Observability ===")
    
    class MonitoringSetup:
        def __init__(self, region: str = 'us-west-2'):
            self.region = region
            self.cloudwatch_client = MockAWSClient('cloudwatch', region)
            self.xray_client = MockAWSClient('xray', region)
            self.sns_client = MockAWSClient('sns', region)
            self.logs_client = MockAWSClient('logs', region)
        
        def create_custom_metrics(self, namespace: str, metrics: List[Dict]) -> List[str]:
            """Create custom CloudWatch metrics"""
            metric_arns = []
            
            for metric in metrics:
                metric_config = {
                    'Namespace': namespace,
                    'MetricData': [{
                        'MetricName': metric['name'],
                        'Unit': metric.get('unit', 'Count'),
                        'Value': 1.0,
                        'Dimensions': metric.get('dimensions', [])
                    }]
                }
                
                response = self.cloudwatch_client.create_resource('metric_data', metric_config)
                metric_arn = f"arn:aws:cloudwatch:{self.cloudwatch_client.region}:123456789012:metric/{namespace}/{metric['name']}"
                metric_arns.append(metric_arn)
                logger.info(f"Custom metric created: {metric['name']}")
            
            return metric_arns
        
        def setup_alarms(self, alarm_configs: List[Dict]) -> List[str]:
            """Set up CloudWatch alarms"""
            alarm_arns = []
            
            for alarm_config in alarm_configs:
                alarm_params = {
                    'AlarmName': alarm_config['name'],
                    'AlarmDescription': alarm_config.get('description', ''),
                    'MetricName': alarm_config['metric'],
                    'Namespace': alarm_config.get('namespace', 'AWS/ApplicationELB'),
                    'Statistic': alarm_config.get('statistic', 'Average'),
                    'Period': alarm_config.get('period', 300),
                    'EvaluationPeriods': alarm_config.get('evaluation_periods', 2),
                    'Threshold': alarm_config['threshold'],
                    'ComparisonOperator': alarm_config.get('comparison', 'GreaterThanThreshold')
                }
                
                response = self.cloudwatch_client.create_resource('alarm', alarm_params)
                alarm_arn = f"arn:aws:cloudwatch:{self.cloudwatch_client.region}:123456789012:alarm:{alarm_config['name']}"
                alarm_arns.append(alarm_arn)
                logger.info(f"CloudWatch alarm created: {alarm_config['name']}")
            
            return alarm_arns
        
        def create_dashboard(self, dashboard_name: str, widgets: List[Dict]) -> str:
            """Create CloudWatch dashboard"""
            dashboard_body = {
                'widgets': [{
                    'type': 'metric',
                    'properties': {
                        'metrics': widget.get('metrics', []),
                        'period': widget.get('period', 300),
                        'stat': widget.get('stat', 'Average'),
                        'region': self.region,
                        'title': widget.get('title', 'Metric Widget')
                    }
                } for widget in widgets]
            }
            
            dashboard_config = {
                'DashboardName': dashboard_name,
                'DashboardBody': json.dumps(dashboard_body)
            }
            
            response = self.cloudwatch_client.create_resource('dashboard', dashboard_config)
            dashboard_arn = f"arn:aws:cloudwatch::{self.cloudwatch_client.region}:123456789012:dashboard/{dashboard_name}"
            logger.info(f"CloudWatch dashboard created: {dashboard_name}")
            return dashboard_arn
        
        def setup_distributed_tracing(self, service_name: str) -> str:
            """Set up AWS X-Ray tracing"""
            tracing_config = {
                'TracingConfig': {
                    'Mode': 'Active'
                },
                'SamplingRule': {
                    'ruleName': f'{service_name}-sampling',
                    'priority': 9000,
                    'fixedRate': 0.1,
                    'reservoirSize': 1,
                    'serviceName': service_name,
                    'serviceType': '*',
                    'host': '*',
                    'method': '*',
                    'URLPath': '*',
                    'version': 1
                }
            }
            
            response = self.xray_client.create_resource('tracing_config', tracing_config)
            tracing_arn = f"arn:aws:xray:{self.xray_client.region}:123456789012:sampling-rule/{service_name}-sampling"
            logger.info(f"X-Ray tracing configured for {service_name}")
            return tracing_arn
        
        def configure_log_aggregation(self, log_groups: List[str],
                                    retention_days: int) -> Dict:
            """Configure centralized logging"""
            log_config = {
                'log_groups_created': [],
                'retention_policy': retention_days,
                'status': 'configured'
            }
            
            for log_group_name in log_groups:
                log_group_config = {
                    'logGroupName': log_group_name,
                    'retentionInDays': retention_days
                }
                
                response = self.logs_client.create_resource('log_group', log_group_config)
                log_config['log_groups_created'].append(log_group_name)
                logger.info(f"Log group created: {log_group_name}")
            
            return log_config
        
        def setup_alerting(self, notification_config: Dict) -> str:
            """Set up alerting and notifications"""
            topic_config = {
                'Name': notification_config.get('topic_name', 'aws-alerts'),
                'DisplayName': 'AWS Infrastructure Alerts',
                'Attributes': {
                    'DeliveryPolicy': json.dumps({
                        'http': {
                            'defaultHealthyRetryPolicy': {
                                'minDelayTarget': 20,
                                'maxDelayTarget': 20,
                                'numRetries': 3
                            }
                        }
                    })
                }
            }
            
            response = self.sns_client.create_resource('topic', topic_config)
            topic_arn = f"arn:aws:sns:{self.sns_client.region}:123456789012:{notification_config.get('topic_name', 'aws-alerts')}"
            
            # Subscribe email endpoints
            for email in notification_config.get('email_addresses', []):
                subscription_config = {
                    'TopicArn': topic_arn,
                    'Protocol': 'email',
                    'Endpoint': email
                }
                self.sns_client.create_resource('subscription', subscription_config)
            
            logger.info(f"SNS alerting configured: {topic_arn}")
            return topic_arn
    
    # Test monitoring setup
    monitoring = MonitoringSetup()
    
    print("Testing Monitoring Configuration...")
    print("\n--- Your implementation should set up comprehensive monitoring ---")
    
    # Example monitoring configuration
    monitoring_config = {
        "metrics": [
            {"name": "ResponseTime", "unit": "Seconds"},
            {"name": "ErrorRate", "unit": "Percent"},
            {"name": "ThroughputRPS", "unit": "Count/Second"}
        ],
        "alarms": [
            {
                "name": "HighResponseTime",
                "metric": "ResponseTime",
                "threshold": 2.0,
                "comparison": "GreaterThanThreshold"
            }
        ]
    }
    
    # monitoring_result = monitoring.setup_comprehensive_monitoring(monitoring_config)
    # print(f"Monitoring configured: {monitoring_result}")


# Exercise 6: Cost Optimization
def exercise_6_cost_optimization():
    """
    Exercise 6: Implement AWS cost optimization strategies
    
    TODO: Complete the CostOptimizer class
    """
    print("\n=== Exercise 6: Cost Optimization ===")
    
    class CostOptimizer:
        def __init__(self, region: str = 'us-west-2'):
            self.region = region
            self.ce_client = MockAWSClient('ce', region)  # Cost Explorer
            self.autoscaling_client = MockAWSClient('application-autoscaling', region)
            self.ec2_client = MockAWSClient('ec2', region)
        
        def analyze_costs(self, time_period: Dict, group_by: List[str]) -> Dict:
            """Analyze current AWS costs"""
            cost_analysis = {
                'time_period': time_period,
                'total_cost': 245.67,
                'cost_by_service': {
                    'EC2-Instance': 89.23,
                    'S3': 34.56,
                    'Lambda': 12.34,
                    'SageMaker': 78.90,
                    'ECS': 30.64
                },
                'trends': {
                    'month_over_month': '+15%',
                    'projected_monthly': 320.45
                },
                'recommendations': [
                    'Consider Reserved Instances for EC2',
                    'Implement S3 lifecycle policies',
                    'Right-size SageMaker instances'
                ]
            }
            
            logger.info(f"Cost analysis completed for period: {time_period}")
            return cost_analysis
        
        def setup_auto_scaling(self, resource_configs: List[Dict]) -> List[str]:
            """Configure auto-scaling for cost optimization"""
            scaling_policies = []
            
            for config in resource_configs:
                policy_config = {
                    'ServiceNamespace': config.get('service_namespace', 'ecs'),
                    'ResourceId': config['resource_id'],
                    'ScalableDimension': config.get('scalable_dimension', 'ecs:service:DesiredCount'),
                    'MinCapacity': config.get('min_capacity', 1),
                    'MaxCapacity': config.get('max_capacity', 10),
                    'TargetTrackingScalingPolicies': [{
                        'TargetValue': config.get('target_utilization', 70),
                        'PredefinedMetricSpecification': {
                            'PredefinedMetricType': config.get('metric_type', 'ECSServiceAverageCPUUtilization')
                        }
                    }]
                }
                
                response = self.autoscaling_client.create_resource('scaling_policy', policy_config)
                policy_arn = f"arn:aws:application-autoscaling:{self.region}:123456789012:scalable-target/{config['resource_id']}"
                scaling_policies.append(policy_arn)
                logger.info(f"Auto-scaling configured for {config['resource_id']}")
            
            return scaling_policies
        
        def implement_spot_instances(self, workload_configs: List[Dict]) -> Dict:
            """Implement Spot instances for cost savings"""
            spot_result = {
                'spot_fleets_created': [],
                'estimated_savings': '60-70%',
                'interruption_handling': 'configured',
                'status': 'implemented'
            }
            
            for workload in workload_configs:
                spot_config = {
                    'SpotFleetRequestConfig': {
                        'IamFleetRole': workload.get('fleet_role', 'arn:aws:iam::123456789012:role/aws-ec2-spot-fleet-role'),
                        'AllocationStrategy': 'diversified',
                        'TargetCapacity': workload.get('target_capacity', 2),
                        'SpotPrice': workload.get('spot_price', '0.05'),
                        'LaunchSpecifications': [{
                            'ImageId': workload.get('ami_id', 'ami-12345678'),
                            'InstanceType': workload.get('instance_type', 'm5.large'),
                            'KeyName': workload.get('key_name', 'my-key'),
                            'SecurityGroups': [{'GroupId': workload.get('security_group', 'sg-12345')}]
                        }]
                    }
                }
                
                response = self.ec2_client.create_resource('spot_fleet_request', spot_config)
                fleet_id = f"sfr-{int(time.time())}"
                spot_result['spot_fleets_created'].append(fleet_id)
                logger.info(f"Spot fleet created: {fleet_id}")
            
            return spot_result
        
        def setup_reserved_capacity(self, usage_analysis: Dict) -> Dict:
            """Recommend and purchase Reserved Instances"""
            ri_recommendations = {
                'recommendations': [
                    {
                        'instance_type': 'm5.large',
                        'quantity': 3,
                        'term': '1year',
                        'payment_option': 'partial_upfront',
                        'estimated_savings': '$1,234/year'
                    },
                    {
                        'instance_type': 'ml.m5.xlarge',
                        'quantity': 2,
                        'term': '1year',
                        'payment_option': 'no_upfront',
                        'estimated_savings': '$2,456/year'
                    }
                ],
                'total_estimated_savings': '$3,690/year',
                'utilization_threshold': '75%',
                'analysis_period': usage_analysis.get('period', '30 days'),
                'status': 'recommendations_generated'
            }
            
            logger.info(f"Reserved Instance recommendations generated: {len(ri_recommendations['recommendations'])} recommendations")
            return ri_recommendations
        
        def optimize_storage_costs(self, storage_analysis: Dict) -> Dict:
            """Optimize S3 and EBS storage costs"""
            storage_optimization = {
                's3_lifecycle_policies': [
                    {
                        'bucket': 'data-lake-bucket',
                        'rules': [
                            {'transition_to_ia': '30 days'},
                            {'transition_to_glacier': '90 days'},
                            {'transition_to_deep_archive': '365 days'}
                        ]
                    }
                ],
                'intelligent_tiering_enabled': True,
                'ebs_optimizations': [
                    {
                        'volume_type': 'gp3',
                        'estimated_savings': '20%',
                        'performance_improvement': 'baseline 3000 IOPS'
                    }
                ],
                'total_estimated_savings': '$450/month',
                'status': 'optimized'
            }
            
            # Simulate lifecycle policy creation
            for policy in storage_optimization['s3_lifecycle_policies']:
                lifecycle_config = {
                    'Bucket': policy['bucket'],
                    'LifecycleConfiguration': {
                        'Rules': [{
                            'Status': 'Enabled',
                            'Transitions': policy['rules']
                        }]
                    }
                }
                logger.info(f"S3 lifecycle policy configured for {policy['bucket']}")
            
            return storage_optimization
        
        def create_cost_budget(self, budget_config: Dict) -> str:
            """Create cost budgets and alerts"""
            budget_params = {
                'BudgetName': budget_config.get('name', 'monthly-budget'),
                'BudgetLimit': {
                    'Amount': str(budget_config.get('amount', 1000)),
                    'Unit': 'USD'
                },
                'TimeUnit': 'MONTHLY',
                'BudgetType': 'COST',
                'CostFilters': {
                    'Service': budget_config.get('services', ['Amazon Elastic Compute Cloud - Compute'])
                },
                'Subscribers': [{
                    'SubscriptionType': 'EMAIL',
                    'Address': budget_config.get('email', 'admin@example.com')
                }],
                'ThresholdPercentages': budget_config.get('thresholds', [50, 80, 100])
            }
            
            # Mock budget creation
            budget_arn = f"arn:aws:budgets::123456789012:budget/{budget_config.get('name', 'monthly-budget')}"
            logger.info(f"Cost budget created: {budget_arn}")
            
            return budget_arn
    
    # Test cost optimization
    cost_optimizer = CostOptimizer()
    
    print("Testing Cost Optimization...")
    print("\n--- Your implementation should optimize AWS costs ---")
    
    # Example cost optimization configuration
    cost_config = {
        "budget_amount": 1000,
        "alert_thresholds": [50, 80, 100],
        "auto_scaling_targets": [
            {"service": "ecs", "target_utilization": 70},
            {"service": "sagemaker", "target_utilization": 80}
        ]
    }
    
    # optimization_result = cost_optimizer.optimize_infrastructure(cost_config)
    # print(f"Cost optimization result: {optimization_result}")


# Exercise 7: Complete AWS Infrastructure Deployment
def exercise_7_infrastructure_deployment():
    """
    Exercise 7: Deploy complete AWS infrastructure for production RAG system
    
    TODO: Complete the InfrastructureDeployment class
    """
    print("\n=== Exercise 7: Complete Infrastructure Deployment ===")
    
    class InfrastructureDeployment:
        def __init__(self, environment: str = 'production'):
            self.environment = environment
            self.cloudformation_client = MockAWSClient('cloudformation')
            self.deployed_resources = {}
        
        def create_vpc_stack(self, vpc_config: Dict) -> str:
            """Deploy VPC infrastructure using CloudFormation"""
            stack_config = {
                'StackName': f'{self.environment}-vpc-stack',
                'TemplateBody': json.dumps({
                    'AWSTemplateFormatVersion': '2010-09-09',
                    'Resources': {
                        'VPC': {
                            'Type': 'AWS::EC2::VPC',
                            'Properties': {
                                'CidrBlock': vpc_config.get('cidr', '10.0.0.0/16'),
                                'EnableDnsHostnames': True,
                                'EnableDnsSupport': True
                            }
                        },
                        'PublicSubnet1': {
                            'Type': 'AWS::EC2::Subnet',
                            'Properties': {
                                'VpcId': {'Ref': 'VPC'},
                                'CidrBlock': '10.0.1.0/24',
                                'AvailabilityZone': 'us-west-2a'
                            }
                        },
                        'PrivateSubnet1': {
                            'Type': 'AWS::EC2::Subnet',
                            'Properties': {
                                'VpcId': {'Ref': 'VPC'},
                                'CidrBlock': '10.0.2.0/24',
                                'AvailabilityZone': 'us-west-2b'
                            }
                        }
                    }
                }),
                'Capabilities': ['CAPABILITY_IAM']
            }
            
            response = self.cloudformation_client.create_resource('stack', stack_config)
            stack_arn = f"arn:aws:cloudformation:us-west-2:123456789012:stack/{self.environment}-vpc-stack/12345"
            self.deployed_resources['vpc_stack'] = stack_arn
            logger.info(f"VPC stack deployed: {stack_arn}")
            return stack_arn
        
        def deploy_data_layer(self, data_config: Dict) -> Dict:
            """Deploy data storage and processing layer"""
            data_resources = {
                'rds_instances': [],
                's3_buckets': [],
                'dynamodb_tables': [],
                'status': 'deployed'
            }
            
            # Deploy RDS instances
            for rds_config in data_config.get('rds_instances', []):
                rds_params = {
                    'DBInstanceIdentifier': rds_config.get('identifier', 'production-db'),
                    'DBInstanceClass': rds_config.get('instance_class', 'db.t3.micro'),
                    'Engine': rds_config.get('engine', 'postgres'),
                    'AllocatedStorage': rds_config.get('storage', 20),
                    'VpcSecurityGroupIds': rds_config.get('security_groups', ['sg-12345'])
                }
                
                rds_arn = f"arn:aws:rds:us-west-2:123456789012:db:{rds_params['DBInstanceIdentifier']}"
                data_resources['rds_instances'].append(rds_arn)
                logger.info(f"RDS instance deployed: {rds_arn}")
            
            # Deploy S3 buckets
            for bucket_name in data_config.get('s3_buckets', []):
                bucket_config = {
                    'Bucket': bucket_name,
                    'VersioningConfiguration': {'Status': 'Enabled'},
                    'LifecycleConfiguration': {
                        'Rules': [{
                            'Status': 'Enabled',
                            'Transitions': [{
                                'Days': 30,
                                'StorageClass': 'STANDARD_IA'
                            }]
                        }]
                    }
                }
                
                bucket_arn = f"arn:aws:s3:::{bucket_name}"
                data_resources['s3_buckets'].append(bucket_arn)
                logger.info(f"S3 bucket deployed: {bucket_arn}")
            
            # Deploy DynamoDB tables
            for table_config in data_config.get('dynamodb_tables', []):
                table_params = {
                    'TableName': table_config.get('name', 'sessions'),
                    'KeySchema': table_config.get('key_schema', [{'AttributeName': 'id', 'KeyType': 'HASH'}]),
                    'AttributeDefinitions': table_config.get('attributes', [{'AttributeName': 'id', 'AttributeType': 'S'}]),
                    'BillingMode': 'PAY_PER_REQUEST'
                }
                
                table_arn = f"arn:aws:dynamodb:us-west-2:123456789012:table/{table_params['TableName']}"
                data_resources['dynamodb_tables'].append(table_arn)
                logger.info(f"DynamoDB table deployed: {table_arn}")
            
            self.deployed_resources['data_layer'] = data_resources
            return data_resources
        
        def deploy_compute_layer(self, compute_config: Dict) -> Dict:
            """Deploy compute infrastructure"""
            compute_resources = {
                'ecs_clusters': [],
                'lambda_functions': [],
                'sagemaker_endpoints': [],
                'load_balancers': [],
                'status': 'deployed'
            }
            
            # Deploy ECS clusters
            if compute_config.get('ecs_cluster'):
                cluster_arn = f"arn:aws:ecs:us-west-2:123456789012:cluster/{self.environment}-cluster"
                compute_resources['ecs_clusters'].append(cluster_arn)
                logger.info(f"ECS cluster deployed: {cluster_arn}")
            
            # Deploy Lambda functions
            for func_name in compute_config.get('lambda_functions', []):
                function_arn = f"arn:aws:lambda:us-west-2:123456789012:function:{func_name}"
                compute_resources['lambda_functions'].append(function_arn)
                logger.info(f"Lambda function deployed: {function_arn}")
            
            # Deploy SageMaker endpoints
            for endpoint_name in compute_config.get('sagemaker_endpoints', []):
                endpoint_arn = f"arn:aws:sagemaker:us-west-2:123456789012:endpoint/{endpoint_name}"
                compute_resources['sagemaker_endpoints'].append(endpoint_arn)
                logger.info(f"SageMaker endpoint deployed: {endpoint_arn}")
            
            # Deploy load balancers
            if compute_config.get('load_balancer'):
                lb_arn = f"arn:aws:elasticloadbalancing:us-west-2:123456789012:loadbalancer/app/{self.environment}-alb/1234567890123456"
                compute_resources['load_balancers'].append(lb_arn)
                logger.info(f"Load balancer deployed: {lb_arn}")
            
            self.deployed_resources['compute_layer'] = compute_resources
            return compute_resources
        
        def deploy_monitoring_layer(self, monitoring_config: Dict) -> Dict:
            """Deploy monitoring and observability"""
            monitoring_resources = {
                'dashboards': [],
                'alarms': [],
                'log_groups': [],
                'xray_tracing': False,
                'status': 'deployed'
            }
            
            # Deploy CloudWatch dashboards
            if monitoring_config.get('cloudwatch_dashboards'):
                dashboard_arn = f"arn:aws:cloudwatch::us-west-2:123456789012:dashboard/{self.environment}-dashboard"
                monitoring_resources['dashboards'].append(dashboard_arn)
                logger.info(f"CloudWatch dashboard deployed: {dashboard_arn}")
            
            # Set up X-Ray tracing
            if monitoring_config.get('xray_tracing'):
                monitoring_resources['xray_tracing'] = True
                logger.info("X-Ray tracing enabled")
            
            # Create custom metrics and alarms
            if monitoring_config.get('custom_metrics'):
                alarm_names = ['HighCPUUtilization', 'HighMemoryUsage', 'HighErrorRate']
                for alarm_name in alarm_names:
                    alarm_arn = f"arn:aws:cloudwatch:us-west-2:123456789012:alarm:{alarm_name}"
                    monitoring_resources['alarms'].append(alarm_arn)
                    logger.info(f"CloudWatch alarm created: {alarm_name}")
            
            # Set up log groups
            log_groups = ['/aws/ecs/rag-api', '/aws/lambda/document-processor', '/aws/sagemaker/training']
            for log_group in log_groups:
                log_group_arn = f"arn:aws:logs:us-west-2:123456789012:log-group:{log_group}"
                monitoring_resources['log_groups'].append(log_group_arn)
                logger.info(f"Log group created: {log_group}")
            
            self.deployed_resources['monitoring_layer'] = monitoring_resources
            return monitoring_resources
        
        def deploy_security_layer(self, security_config: Dict) -> Dict:
            """Deploy security infrastructure"""
            security_resources = {
                'iam_roles': [],
                'kms_keys': [],
                'secrets': [],
                'security_groups': [],
                'status': 'deployed'
            }
            
            # Deploy IAM roles
            role_names = ['ECSTaskRole', 'LambdaExecutionRole', 'SageMakerRole']
            for role_name in role_names:
                role_arn = f"arn:aws:iam::123456789012:role/{self.environment}-{role_name}"
                security_resources['iam_roles'].append(role_arn)
                logger.info(f"IAM role deployed: {role_arn}")
            
            # Deploy KMS keys
            if security_config.get('encryption_enabled'):
                key_id = f"key-{int(time.time())}"
                key_arn = f"arn:aws:kms:us-west-2:123456789012:key/{key_id}"
                security_resources['kms_keys'].append(key_arn)
                logger.info(f"KMS key deployed: {key_arn}")
            
            # Deploy secrets
            secret_names = ['database-credentials', 'api-keys', 'certificates']
            for secret_name in secret_names:
                secret_arn = f"arn:aws:secretsmanager:us-west-2:123456789012:secret:{self.environment}-{secret_name}"
                security_resources['secrets'].append(secret_arn)
                logger.info(f"Secret deployed: {secret_arn}")
            
            # Deploy security groups
            sg_names = ['web-tier-sg', 'app-tier-sg', 'db-tier-sg']
            for sg_name in sg_names:
                sg_id = f"sg-{int(time.time())}-{sg_name}"
                security_resources['security_groups'].append(sg_id)
                logger.info(f"Security group deployed: {sg_id}")
            
            self.deployed_resources['security_layer'] = security_resources
            return security_resources
        
        def validate_deployment(self) -> Dict:
            """Validate complete deployment"""
            validation_result = {
                'health_checks': {
                    'vpc_connectivity': 'PASS',
                    'load_balancer_health': 'PASS',
                    'database_connectivity': 'PASS',
                    'api_endpoints': 'PASS'
                },
                'security_validation': {
                    'iam_policies': 'PASS',
                    'security_groups': 'PASS',
                    'encryption': 'PASS',
                    'secrets_access': 'PASS'
                },
                'performance_tests': {
                    'response_time': '< 200ms',
                    'throughput': '1000 RPS',
                    'error_rate': '< 0.1%'
                },
                'deployment_summary': {
                    'total_resources': len([item for sublist in self.deployed_resources.values() if isinstance(sublist, list) for item in sublist]),
                    'deployment_time': '45 minutes',
                    'status': 'SUCCESS'
                },
                'validation_timestamp': datetime.now().isoformat()
            }
            
            logger.info("Deployment validation completed successfully")
            return validation_result
        
        def deploy_complete_infrastructure(self, config: Dict) -> Dict:
            """Deploy complete production infrastructure"""
            deployment_result = {
                'deployment_id': f"deploy-{int(time.time())}",
                'environment': self.environment,
                'layers_deployed': [],
                'total_resources': 0,
                'status': 'in_progress',
                'start_time': datetime.now().isoformat()
            }
            
            try:
                # Deploy VPC layer
                logger.info("Deploying VPC infrastructure...")
                vpc_stack_arn = self.create_vpc_stack(config.get('vpc', {}))
                deployment_result['layers_deployed'].append('vpc')
                
                # Deploy data layer
                logger.info("Deploying data layer...")
                data_resources = self.deploy_data_layer(config.get('data', {}))
                deployment_result['layers_deployed'].append('data')
                
                # Deploy compute layer
                logger.info("Deploying compute layer...")
                compute_resources = self.deploy_compute_layer(config.get('compute', {}))
                deployment_result['layers_deployed'].append('compute')
                
                # Deploy monitoring layer
                logger.info("Deploying monitoring layer...")
                monitoring_resources = self.deploy_monitoring_layer(config.get('monitoring', {}))
                deployment_result['layers_deployed'].append('monitoring')
                
                # Deploy security layer
                logger.info("Deploying security layer...")
                security_resources = self.deploy_security_layer(config.get('security', {}))
                deployment_result['layers_deployed'].append('security')
                
                # Validate deployment
                logger.info("Validating deployment...")
                validation_result = self.validate_deployment()
                
                # Calculate total resources
                total_resources = sum(len(resources) if isinstance(resources, list) else 1 
                                    for layer in self.deployed_resources.values() 
                                    for resources in (layer.values() if isinstance(layer, dict) else [layer]))
                
                deployment_result.update({
                    'total_resources': total_resources,
                    'status': 'completed',
                    'end_time': datetime.now().isoformat(),
                    'validation_result': validation_result,
                    'deployed_resources': self.deployed_resources
                })
                
                logger.info(f"Complete infrastructure deployment successful: {deployment_result['deployment_id']}")
                
            except Exception as e:
                deployment_result.update({
                    'status': 'failed',
                    'error': str(e),
                    'end_time': datetime.now().isoformat()
                })
                logger.error(f"Infrastructure deployment failed: {e}")
            
            return deployment_result
    
    # Test complete infrastructure deployment
    infrastructure = InfrastructureDeployment('production')
    
    print("Testing Complete Infrastructure Deployment...")
    print("\n--- Your implementation should deploy production-ready infrastructure ---")
    
    # Example infrastructure configuration
    infra_config = {
        "vpc": {
            "cidr": "10.0.0.0/16",
            "availability_zones": 2,
            "nat_gateways": True
        },
        "compute": {
            "ecs_cluster": True,
            "lambda_functions": ["document-processor", "embedding-generator"],
            "sagemaker_endpoints": ["text-classifier", "embedding-model"]
        },
        "data": {
            "rds_instance": "db.r5.large",
            "s3_buckets": ["documents", "models", "logs"],
            "dynamodb_tables": ["sessions", "user-preferences"]
        },
        "monitoring": {
            "cloudwatch_dashboards": True,
            "xray_tracing": True,
            "custom_metrics": True
        }
    }
    
    # deployment_result = infrastructure.deploy_complete_infrastructure(infra_config)
    # print(f"Infrastructure deployment: {deployment_result}")


def main():
    """Run all AWS Deep Dive exercises"""
    print("☁️ Day 55: AWS Deep Dive - Cloud Infrastructure for Data & AI Systems")
    print("=" * 80)
    
    exercises = [
        exercise_1_sagemaker_pipeline,
        exercise_2_ecs_deployment,
        exercise_3_lambda_pipeline,
        exercise_4_security_setup,
        exercise_5_monitoring_setup,
        exercise_6_cost_optimization,
        exercise_7_infrastructure_deployment
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
    print("2. Test your implementations with AWS CLI/SDK")
    print("3. Deploy actual resources in a test AWS account")
    print("4. Review the solution file for complete implementations")
    print("5. Practice with AWS Free Tier resources")
    
    print("\n🚀 Production Deployment Checklist:")
    print("• Set up proper IAM roles and policies")
    print("• Configure VPC with private/public subnets")
    print("• Implement encryption at rest and in transit")
    print("• Set up comprehensive monitoring and alerting")
    print("• Configure auto-scaling and cost optimization")
    print("• Implement disaster recovery and backup strategies")
    print("• Set up CI/CD pipelines for infrastructure")


if __name__ == "__main__":
    main()
