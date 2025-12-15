# Day 55: AWS Deep Dive - Cloud Infrastructure for Data & AI Systems

## Learning Objectives
By the end of this session, you will be able to:
- Design and deploy scalable data and AI infrastructure on AWS
- Implement production ML workflows using SageMaker, ECS, and Lambda
- Build secure, cost-effective cloud architectures for enterprise workloads
- Configure monitoring, logging, and observability for cloud-native applications
- Apply AWS best practices for high availability, disaster recovery, and compliance

## Theory (15 minutes)

### AWS Cloud Infrastructure for Data & AI

Amazon Web Services (AWS) provides a comprehensive suite of cloud services that enable organizations to build, deploy, and scale data and AI applications efficiently. This deep dive focuses on the core services essential for production data and AI systems.

### Core AWS Services for Data & AI

#### 1. Amazon SageMaker - Complete ML Platform

**SageMaker Studio**
```python
# SageMaker training job configuration
import boto3
import sagemaker
from sagemaker.sklearn.estimator import SKLearn

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Create estimator
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='0.23-1',
    py_version='py3',
    script_mode=True,
    hyperparameters={
        'n_estimators': 100,
        'max_depth': 5
    }
)

# Start training
sklearn_estimator.fit({'training': 's3://bucket/training-data'})
```

**SageMaker Endpoints for Real-time Inference**
```python
# Deploy model to endpoint
predictor = sklearn_estimator.deploy(
    initial_instance_count=1,
    instance_type='ml.m5.large',
    endpoint_name='ml-model-endpoint'
)

# Auto-scaling configuration
import boto3
autoscaling = boto3.client('application-autoscaling')

autoscaling.register_scalable_target(
    ServiceNamespace='sagemaker',
    ResourceId='endpoint/ml-model-endpoint/variant/AllTraffic',
    ScalableDimension='sagemaker:variant:DesiredInstanceCount',
    MinCapacity=1,
    MaxCapacity=10
)
```

**SageMaker Pipelines for MLOps**
```python
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import TrainingStep, CreateModelStep
from sagemaker.workflow.step_collections import RegisterModel

# Define pipeline steps
training_step = TrainingStep(
    name="TrainModel",
    estimator=sklearn_estimator,
    inputs={'training': training_input}
)

model_step = CreateModelStep(
    name="CreateModel",
    model=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    inputs=sagemaker.inputs.CreateModelInput(
        instance_type="ml.m5.large"
    )
)

# Create pipeline
pipeline = Pipeline(
    name="MLPipeline",
    steps=[training_step, model_step]
)

pipeline.create(role_arn=role)
```

#### 2. Amazon ECS - Container Orchestration

**ECS Fargate for Serverless Containers**
```yaml
# ECS Task Definition
version: '3'
services:
  rag-api:
    image: your-account.dkr.ecr.region.amazonaws.com/rag-system:latest
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - DATABASE_URL=${DATABASE_URL}
    logging:
      driver: awslogs
      options:
        awslogs-group: /ecs/rag-system
        awslogs-region: us-west-2
        awslogs-stream-prefix: ecs
```

**ECS Service with Load Balancer**
```python
import boto3

ecs = boto3.client('ecs')
elbv2 = boto3.client('elbv2')

# Create ECS service
service_response = ecs.create_service(
    cluster='production-cluster',
    serviceName='rag-api-service',
    taskDefinition='rag-api:1',
    desiredCount=3,
    launchType='FARGATE',
    networkConfiguration={
        'awsvpcConfiguration': {
            'subnets': ['subnet-12345', 'subnet-67890'],
            'securityGroups': ['sg-abcdef'],
            'assignPublicIp': 'ENABLED'
        }
    },
    loadBalancers=[
        {
            'targetGroupArn': 'arn:aws:elasticloadbalancing:...',
            'containerName': 'rag-api',
            'containerPort': 8000
        }
    ]
)
```

#### 3. AWS Lambda - Serverless Computing

**Lambda for Event-Driven Processing**
```python
import json
import boto3
from typing import Dict, Any

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Process document upload events
    """
    # Parse S3 event
    s3_event = event['Records'][0]['s3']
    bucket = s3_event['bucket']['name']
    key = s3_event['object']['key']
    
    # Initialize AWS clients
    s3 = boto3.client('s3')
    sagemaker = boto3.client('sagemaker-runtime')
    
    try:
        # Download document from S3
        response = s3.get_object(Bucket=bucket, Key=key)
        document_content = response['Body'].read()
        
        # Process document (extract text, generate embeddings)
        processed_data = process_document(document_content)
        
        # Store embeddings in vector database
        store_embeddings(processed_data)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Successfully processed {key}',
                'document_id': processed_data['id']
            })
        }
        
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def process_document(content: bytes) -> Dict[str, Any]:
    """Process document content"""
    # Implementation for document processing
    pass

def store_embeddings(data: Dict[str, Any]):
    """Store embeddings in vector database"""
    # Implementation for storing embeddings
    pass
```

#### 4. Amazon RDS & DynamoDB - Database Services

**RDS for Relational Data**
```python
import boto3
import psycopg2
from sqlalchemy import create_engine

# RDS connection
def create_rds_connection():
    rds = boto3.client('rds')
    
    # Get RDS endpoint
    response = rds.describe_db_instances(DBInstanceIdentifier='production-db')
    endpoint = response['DBInstances'][0]['Endpoint']['Address']
    
    # Create connection
    engine = create_engine(
        f'postgresql://username:password@{endpoint}:5432/database'
    )
    return engine

# Query execution
engine = create_rds_connection()
with engine.connect() as conn:
    result = conn.execute("""
        SELECT document_id, title, created_at 
        FROM documents 
        WHERE category = 'technical'
        ORDER BY created_at DESC
        LIMIT 100
    """)
```

**DynamoDB for NoSQL Data**
```python
import boto3
from boto3.dynamodb.conditions import Key, Attr

dynamodb = boto3.resource('dynamodb')
table = dynamodb.Table('user-sessions')

# Put item
table.put_item(
    Item={
        'session_id': 'session-123',
        'user_id': 'user-456',
        'created_at': '2024-12-12T10:00:00Z',
        'query_history': [
            {'query': 'What is machine learning?', 'timestamp': '2024-12-12T10:01:00Z'},
            {'query': 'How to deploy models?', 'timestamp': '2024-12-12T10:05:00Z'}
        ],
        'ttl': 1734019200  # 30 days from now
    }
)

# Query items
response = table.query(
    KeyConditionExpression=Key('user_id').eq('user-456'),
    FilterExpression=Attr('created_at').gte('2024-12-01T00:00:00Z')
)
```

#### 5. Amazon S3 - Object Storage

**S3 for Data Lake Architecture**
```python
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client('s3')

# Upload with metadata
def upload_document(file_path: str, bucket: str, key: str, metadata: dict):
    try:
        s3.upload_file(
            file_path, 
            bucket, 
            key,
            ExtraArgs={
                'Metadata': metadata,
                'ServerSideEncryption': 'AES256',
                'StorageClass': 'STANDARD_IA'  # Infrequent Access
            }
        )
        print(f"Successfully uploaded {key}")
    except ClientError as e:
        print(f"Upload failed: {e}")

# S3 lifecycle policy
lifecycle_policy = {
    'Rules': [
        {
            'ID': 'DocumentArchiving',
            'Status': 'Enabled',
            'Filter': {'Prefix': 'documents/'},
            'Transitions': [
                {
                    'Days': 30,
                    'StorageClass': 'STANDARD_IA'
                },
                {
                    'Days': 90,
                    'StorageClass': 'GLACIER'
                },
                {
                    'Days': 365,
                    'StorageClass': 'DEEP_ARCHIVE'
                }
            ]
        }
    ]
}

s3.put_bucket_lifecycle_configuration(
    Bucket='data-lake-bucket',
    LifecycleConfiguration=lifecycle_policy
)
```

### AWS Architecture Patterns

#### 1. Microservices Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │   Application   │    │   Application   │
│  Load Balancer  │    │  Load Balancer  │    │  Load Balancer  │
│      (ALB)      │    │      (ALB)      │    │      (ALB)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   ECS Service   │    │   ECS Service   │    │   ECS Service   │
│   (RAG API)     │    │ (Document Proc) │    │  (Evaluation)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      RDS        │    │       S3        │    │   DynamoDB      │
│   (Metadata)    │    │  (Documents)    │    │   (Sessions)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### 2. Event-Driven Architecture

```python
# CloudWatch Events + Lambda
import boto3

events = boto3.client('events')

# Create rule for S3 uploads
events.put_rule(
    Name='DocumentUploadRule',
    EventPattern=json.dumps({
        'source': ['aws.s3'],
        'detail-type': ['Object Created'],
        'detail': {
            'bucket': {'name': ['documents-bucket']}
        }
    }),
    State='ENABLED'
)

# Add Lambda target
events.put_targets(
    Rule='DocumentUploadRule',
    Targets=[
        {
            'Id': '1',
            'Arn': 'arn:aws:lambda:us-west-2:123456789012:function:ProcessDocument'
        }
    ]
)
```

#### 3. Data Pipeline Architecture

```python
# Step Functions for orchestration
import boto3

stepfunctions = boto3.client('stepfunctions')

state_machine_definition = {
    "Comment": "ML Pipeline",
    "StartAt": "DataPreprocessing",
    "States": {
        "DataPreprocessing": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-west-2:123456789012:function:PreprocessData",
            "Next": "TrainModel"
        },
        "TrainModel": {
            "Type": "Task",
            "Resource": "arn:aws:states:::sagemaker:createTrainingJob.sync",
            "Parameters": {
                "TrainingJobName.$": "$.training_job_name",
                "AlgorithmSpecification": {
                    "TrainingImage": "382416733822.dkr.ecr.us-west-2.amazonaws.com/sklearn:latest",
                    "TrainingInputMode": "File"
                }
            },
            "Next": "EvaluateModel"
        },
        "EvaluateModel": {
            "Type": "Task",
            "Resource": "arn:aws:lambda:us-west-2:123456789012:function:EvaluateModel",
            "End": true
        }
    }
}
```

### Security and Compliance

#### 1. IAM Roles and Policies

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject"
      ],
      "Resource": "arn:aws:s3:::data-bucket/*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreateTrainingJob",
        "sagemaker:DescribeTrainingJob"
      ],
      "Resource": "*"
    }
  ]
}
```

#### 2. VPC and Network Security

```python
# VPC configuration
import boto3

ec2 = boto3.client('ec2')

# Create VPC
vpc_response = ec2.create_vpc(CidrBlock='10.0.0.0/16')
vpc_id = vpc_response['Vpc']['VpcId']

# Create subnets
private_subnet = ec2.create_subnet(
    VpcId=vpc_id,
    CidrBlock='10.0.1.0/24',
    AvailabilityZone='us-west-2a'
)

public_subnet = ec2.create_subnet(
    VpcId=vpc_id,
    CidrBlock='10.0.2.0/24',
    AvailabilityZone='us-west-2b'
)

# Security group for ECS tasks
security_group = ec2.create_security_group(
    GroupName='ecs-tasks-sg',
    Description='Security group for ECS tasks',
    VpcId=vpc_id
)

# Allow inbound HTTP traffic
ec2.authorize_security_group_ingress(
    GroupId=security_group['GroupId'],
    IpPermissions=[
        {
            'IpProtocol': 'tcp',
            'FromPort': 8000,
            'ToPort': 8000,
            'IpRanges': [{'CidrIp': '10.0.0.0/16'}]
        }
    ]
)
```

### Monitoring and Observability

#### 1. CloudWatch Metrics and Alarms

```python
import boto3

cloudwatch = boto3.client('cloudwatch')

# Create custom metric
cloudwatch.put_metric_data(
    Namespace='RAG/System',
    MetricData=[
        {
            'MetricName': 'ResponseTime',
            'Value': 1.5,
            'Unit': 'Seconds',
            'Dimensions': [
                {
                    'Name': 'Service',
                    'Value': 'RAG-API'
                }
            ]
        }
    ]
)

# Create alarm
cloudwatch.put_metric_alarm(
    AlarmName='HighResponseTime',
    ComparisonOperator='GreaterThanThreshold',
    EvaluationPeriods=2,
    MetricName='ResponseTime',
    Namespace='RAG/System',
    Period=300,
    Statistic='Average',
    Threshold=2.0,
    ActionsEnabled=True,
    AlarmActions=[
        'arn:aws:sns:us-west-2:123456789012:alerts-topic'
    ],
    AlarmDescription='Alert when response time exceeds 2 seconds'
)
```

#### 2. AWS X-Ray for Distributed Tracing

```python
from aws_xray_sdk.core import xray_recorder
from aws_xray_sdk.core import patch_all

# Patch AWS SDK calls
patch_all()

@xray_recorder.capture('process_query')
def process_query(query: str):
    # Add metadata
    xray_recorder.current_subsegment().put_metadata('query_length', len(query))
    
    # Add annotation for filtering
    xray_recorder.current_subsegment().put_annotation('query_type', 'technical')
    
    # Your processing logic here
    result = perform_retrieval(query)
    return result

@xray_recorder.capture('perform_retrieval')
def perform_retrieval(query: str):
    # Retrieval logic with tracing
    pass
```

### Cost Optimization Strategies

#### 1. Right-sizing and Auto-scaling

```python
# ECS auto-scaling
autoscaling = boto3.client('application-autoscaling')

# Register scalable target
autoscaling.register_scalable_target(
    ServiceNamespace='ecs',
    ResourceId='service/production-cluster/rag-api-service',
    ScalableDimension='ecs:service:DesiredCount',
    MinCapacity=2,
    MaxCapacity=20
)

# Create scaling policy
autoscaling.put_scaling_policy(
    PolicyName='cpu-scaling-policy',
    ServiceNamespace='ecs',
    ResourceId='service/production-cluster/rag-api-service',
    ScalableDimension='ecs:service:DesiredCount',
    PolicyType='TargetTrackingScaling',
    TargetTrackingScalingPolicyConfiguration={
        'TargetValue': 70.0,
        'PredefinedMetricSpecification': {
            'PredefinedMetricType': 'ECSServiceAverageCPUUtilization'
        },
        'ScaleOutCooldown': 300,
        'ScaleInCooldown': 300
    }
)
```

#### 2. Spot Instances and Reserved Capacity

```python
# SageMaker training with Spot instances
sklearn_estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    use_spot_instances=True,
    max_wait=3600,  # Wait up to 1 hour for Spot capacity
    max_run=1800,   # Maximum training time
    framework_version='0.23-1'
)
```

### Why AWS Deep Dive Matters

1. **Scalability**: AWS services automatically scale to handle varying workloads
2. **Cost Efficiency**: Pay-as-you-go pricing with optimization opportunities
3. **Security**: Enterprise-grade security with compliance certifications
4. **Global Reach**: Deploy applications worldwide with low latency
5. **Innovation Speed**: Rapidly prototype and deploy new features
6. **Operational Excellence**: Managed services reduce operational overhead

### Real-world Use Cases

- **Netflix**: Uses AWS for global content delivery and recommendation systems
- **Airbnb**: Leverages AWS for data processing and machine learning at scale
- **Capital One**: Built cloud-native banking applications on AWS
- **NASA JPL**: Processes space mission data using AWS analytics services
- **Moderna**: Accelerated vaccine development using AWS compute resources

## Exercise (40 minutes)
Complete the hands-on exercises in `exercise.py` to build production-ready AWS infrastructure for data and AI applications, including SageMaker ML pipelines, ECS containerized services, and serverless architectures.

## Resources
- [AWS Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/)
- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Amazon ECS Best Practices](https://docs.aws.amazon.com/AmazonECS/latest/bestpracticesguide/)
- [AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [AWS Cost Optimization](https://aws.amazon.com/aws-cost-management/)
- [AWS Security Best Practices](https://aws.amazon.com/architecture/security-identity-compliance/)

## Next Steps
- Complete the AWS infrastructure exercises
- Review cloud architecture patterns
- Take the quiz to test your understanding
- Move to Day 56: Kubernetes for ML & Data Workloads
