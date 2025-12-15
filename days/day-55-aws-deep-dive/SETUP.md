# Day 55: AWS Deep Dive - Setup Guide

## Overview
This guide helps you set up the environment for AWS Deep Dive exercises, including AWS CLI configuration, IAM setup, and local development environment.

## Prerequisites

### Required Software
- Python 3.9+ with pip
- AWS CLI v2
- Docker Desktop
- Git
- Code editor (VS Code recommended)

### AWS Account Requirements
- Active AWS account with billing enabled
- Administrative access or appropriate IAM permissions
- AWS Free Tier eligible (recommended for cost control)

## Installation Steps

### 1. Install Dependencies

```bash
# Navigate to day 55 directory
cd days/day-55-aws-deep-dive

# Install Python dependencies
pip install -r requirements.txt

# Verify installations
aws --version
docker --version
python --version
```

### 2. AWS CLI Configuration

#### Option A: Using AWS Configure (Recommended for beginners)
```bash
# Configure AWS CLI with your credentials
aws configure

# Enter when prompted:
# AWS Access Key ID: [Your access key]
# AWS Secret Access Key: [Your secret key]
# Default region name: us-west-2
# Default output format: json
```

#### Option B: Using Environment Variables
```bash
# Set environment variables (Linux/macOS)
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-west-2

# For Windows PowerShell
$env:AWS_ACCESS_KEY_ID="your_access_key_here"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key_here"
$env:AWS_DEFAULT_REGION="us-west-2"
```

#### Option C: Using AWS Profiles
```bash
# Create a profile for this bootcamp
aws configure --profile bootcamp

# Use the profile
export AWS_PROFILE=bootcamp
```

### 3. Verify AWS Access

```bash
# Test AWS connectivity
aws sts get-caller-identity

# Expected output:
# {
#     "UserId": "AIDACKCEVSQ6C2EXAMPLE",
#     "Account": "123456789012",
#     "Arn": "arn:aws:iam::123456789012:user/your-username"
# }
```

### 4. Create IAM Roles (Required for Exercises)

#### SageMaker Execution Role
```bash
# Create trust policy for SageMaker
cat > sagemaker-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "sagemaker.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
    --role-name SageMakerExecutionRole \
    --assume-role-policy-document file://sagemaker-trust-policy.json

# Attach managed policies
aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy \
    --role-name SageMakerExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

#### ECS Task Execution Role
```bash
# Create trust policy for ECS
cat > ecs-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "ecs-tasks.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
    --role-name ECSTaskExecutionRole \
    --assume-role-policy-document file://ecs-trust-policy.json

# Attach managed policy
aws iam attach-role-policy \
    --role-name ECSTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
```

#### Lambda Execution Role
```bash
# Create trust policy for Lambda
cat > lambda-trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

# Create the role
aws iam create-role \
    --role-name LambdaExecutionRole \
    --assume-role-policy-document file://lambda-trust-policy.json

# Attach managed policies
aws iam attach-role-policy \
    --role-name LambdaExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

aws iam attach-role-policy \
    --role-name LambdaExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### 5. Create S3 Buckets

```bash
# Get your AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create unique bucket names
ML_BUCKET="ml-training-bucket-${ACCOUNT_ID}"
DATA_BUCKET="data-processing-bucket-${ACCOUNT_ID}"
LOGS_BUCKET="aws-logs-bucket-${ACCOUNT_ID}"

# Create buckets
aws s3 mb s3://${ML_BUCKET} --region us-west-2
aws s3 mb s3://${DATA_BUCKET} --region us-west-2
aws s3 mb s3://${LOGS_BUCKET} --region us-west-2

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket ${ML_BUCKET} \
    --versioning-configuration Status=Enabled

# Set up lifecycle policy for cost optimization
cat > lifecycle-policy.json << EOF
{
    "Rules": [
        {
            "ID": "CostOptimization",
            "Status": "Enabled",
            "Filter": {"Prefix": ""},
            "Transitions": [
                {
                    "Days": 30,
                    "StorageClass": "STANDARD_IA"
                },
                {
                    "Days": 90,
                    "StorageClass": "GLACIER"
                }
            ]
        }
    ]
}
EOF

aws s3api put-bucket-lifecycle-configuration \
    --bucket ${DATA_BUCKET} \
    --lifecycle-configuration file://lifecycle-policy.json
```

### 6. Set Up VPC (Optional but Recommended)

```bash
# Create VPC
VPC_ID=$(aws ec2 create-vpc \
    --cidr-block 10.0.0.0/16 \
    --query 'Vpc.VpcId' \
    --output text)

# Create subnets
SUBNET_1=$(aws ec2 create-subnet \
    --vpc-id ${VPC_ID} \
    --cidr-block 10.0.1.0/24 \
    --availability-zone us-west-2a \
    --query 'Subnet.SubnetId' \
    --output text)

SUBNET_2=$(aws ec2 create-subnet \
    --vpc-id ${VPC_ID} \
    --cidr-block 10.0.2.0/24 \
    --availability-zone us-west-2b \
    --query 'Subnet.SubnetId' \
    --output text)

# Create internet gateway
IGW_ID=$(aws ec2 create-internet-gateway \
    --query 'InternetGateway.InternetGatewayId' \
    --output text)

# Attach internet gateway to VPC
aws ec2 attach-internet-gateway \
    --vpc-id ${VPC_ID} \
    --internet-gateway-id ${IGW_ID}

# Create security group
SG_ID=$(aws ec2 create-security-group \
    --group-name bootcamp-sg \
    --description "Security group for bootcamp exercises" \
    --vpc-id ${VPC_ID} \
    --query 'GroupId' \
    --output text)

# Add inbound rules
aws ec2 authorize-security-group-ingress \
    --group-id ${SG_ID} \
    --protocol tcp \
    --port 80 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id ${SG_ID} \
    --protocol tcp \
    --port 443 \
    --cidr 0.0.0.0/0

aws ec2 authorize-security-group-ingress \
    --group-id ${SG_ID} \
    --protocol tcp \
    --port 8000 \
    --cidr 0.0.0.0/0

echo "VPC ID: ${VPC_ID}"
echo "Subnet 1: ${SUBNET_1}"
echo "Subnet 2: ${SUBNET_2}"
echo "Security Group: ${SG_ID}"
```

### 7. Environment Configuration

Create a `.env` file with your AWS resources:

```bash
# Create environment file
cat > .env << EOF
# AWS Configuration
AWS_REGION=us-west-2
AWS_ACCOUNT_ID=${ACCOUNT_ID}

# S3 Buckets
ML_TRAINING_BUCKET=${ML_BUCKET}
DATA_PROCESSING_BUCKET=${DATA_BUCKET}
LOGS_BUCKET=${LOGS_BUCKET}

# IAM Roles
SAGEMAKER_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/SageMakerExecutionRole
ECS_TASK_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/ECSTaskExecutionRole
LAMBDA_ROLE_ARN=arn:aws:iam::${ACCOUNT_ID}:role/LambdaExecutionRole

# VPC Configuration (if created)
VPC_ID=${VPC_ID}
SUBNET_1=${SUBNET_1}
SUBNET_2=${SUBNET_2}
SECURITY_GROUP_ID=${SG_ID}

# Application Configuration
CLUSTER_NAME=bootcamp-cluster
SERVICE_NAME=rag-api
FUNCTION_NAME_PREFIX=bootcamp
EOF
```

### 8. Test Setup

```bash
# Run the setup test
python test_setup.py

# Expected output:
# ✅ AWS CLI configured correctly
# ✅ AWS credentials valid
# ✅ Required IAM roles exist
# ✅ S3 buckets accessible
# ✅ VPC configuration valid
# ✅ All dependencies installed
```

## Local Development with LocalStack (Optional)

For development without AWS costs, you can use LocalStack:

```bash
# Install LocalStack
pip install localstack

# Start LocalStack
localstack start -d

# Set LocalStack endpoint
export AWS_ENDPOINT_URL=http://localhost:4566

# Run exercises against LocalStack
python exercise.py --local
```

## Docker Setup for Container Exercises

```bash
# Build sample container image
cat > Dockerfile << EOF
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

# Build image
docker build -t rag-api:latest .

# Test locally
docker run -p 8000:8000 rag-api:latest
```

## Cost Management

### Set Up Billing Alerts
```bash
# Create billing alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "BillingAlert" \
    --alarm-description "Alert when charges exceed $10" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 86400 \
    --threshold 10 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=Currency,Value=USD \
    --evaluation-periods 1
```

### Resource Cleanup Script
```bash
# Create cleanup script
cat > cleanup.sh << 'EOF'
#!/bin/bash

echo "Cleaning up AWS resources..."

# Delete S3 buckets (empty them first)
aws s3 rm s3://${ML_TRAINING_BUCKET} --recursive
aws s3 rb s3://${ML_TRAINING_BUCKET}

aws s3 rm s3://${DATA_PROCESSING_BUCKET} --recursive
aws s3 rb s3://${DATA_PROCESSING_BUCKET}

aws s3 rm s3://${LOGS_BUCKET} --recursive
aws s3 rb s3://${LOGS_BUCKET}

# Delete IAM roles
aws iam detach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
aws iam detach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam delete-role --role-name SageMakerExecutionRole

aws iam detach-role-policy --role-name ECSTaskExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
aws iam delete-role --role-name ECSTaskExecutionRole

aws iam detach-role-policy --role-name LambdaExecutionRole --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
aws iam detach-role-policy --role-name LambdaExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
aws iam delete-role --role-name LambdaExecutionRole

# Delete VPC resources
aws ec2 detach-internet-gateway --vpc-id ${VPC_ID} --internet-gateway-id ${IGW_ID}
aws ec2 delete-internet-gateway --internet-gateway-id ${IGW_ID}
aws ec2 delete-security-group --group-id ${SG_ID}
aws ec2 delete-subnet --subnet-id ${SUBNET_1}
aws ec2 delete-subnet --subnet-id ${SUBNET_2}
aws ec2 delete-vpc --vpc-id ${VPC_ID}

echo "Cleanup completed!"
EOF

chmod +x cleanup.sh
```

## Troubleshooting

### Common Issues

1. **AWS CLI not configured**
   ```bash
   # Error: Unable to locate credentials
   # Solution: Run aws configure or set environment variables
   ```

2. **Insufficient permissions**
   ```bash
   # Error: AccessDenied
   # Solution: Ensure your IAM user has necessary permissions
   ```

3. **Region mismatch**
   ```bash
   # Error: InvalidParameterValue
   # Solution: Ensure all resources are in the same region
   ```

4. **Docker not running**
   ```bash
   # Error: Cannot connect to Docker daemon
   # Solution: Start Docker Desktop
   ```

### Verification Commands

```bash
# Check AWS configuration
aws configure list

# Check IAM roles
aws iam list-roles --query 'Roles[?contains(RoleName, `SageMaker`) || contains(RoleName, `ECS`) || contains(RoleName, `Lambda`)].RoleName'

# Check S3 buckets
aws s3 ls

# Check VPC resources
aws ec2 describe-vpcs --query 'Vpcs[?Tags[?Key==`Name`]].VpcId'
```

## Next Steps

1. Complete the setup verification
2. Review the exercise.py file
3. Start with Exercise 1: SageMaker Pipeline
4. Progress through all 7 exercises
5. Review the solution.py for complete implementations
6. Take the quiz to test your understanding

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review AWS documentation for specific services
3. Ensure you have the latest versions of all tools
4. Verify your AWS account has sufficient permissions and credits

Remember to clean up resources after completing the exercises to avoid unnecessary charges!