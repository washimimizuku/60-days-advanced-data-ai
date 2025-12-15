#!/usr/bin/env python3
"""
Day 22: AWS Glue & Data Catalog - AWS Resources Initialization
Initialize LocalStack with required AWS resources for the exercise
"""

import boto3
import os
import json
from botocore.exceptions import ClientError

def get_aws_clients():
    """Initialize AWS clients for LocalStack"""
    endpoint_url = os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566')
    
    return {
        's3': boto3.client('s3', endpoint_url=endpoint_url),
        'glue': boto3.client('glue', endpoint_url=endpoint_url),
        'iam': boto3.client('iam', endpoint_url=endpoint_url)
    }

def create_s3_buckets(s3_client):
    """Create required S3 buckets"""
    buckets = [
        os.getenv('S3_DATA_BUCKET', 'serverlessdata-datalake'),
        os.getenv('S3_SCRIPTS_BUCKET', 'serverlessdata-glue-scripts'),
        os.getenv('S3_ATHENA_RESULTS', 'serverlessdata-athena-results')
    ]
    
    for bucket in buckets:
        try:
            s3_client.create_bucket(Bucket=bucket)
            print(f"✅ Created S3 bucket: {bucket}")
        except ClientError as e:
            if e.response['Error']['Code'] == 'BucketAlreadyOwnedByYou':
                print(f"ℹ️  S3 bucket already exists: {bucket}")
            else:
                print(f"❌ Error creating bucket {bucket}: {e}")

def create_iam_role(iam_client):
    """Create IAM role for Glue"""
    role_name = 'GlueServiceRole'
    
    trust_policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "glue.amazonaws.com"},
                "Action": "sts:AssumeRole"
            }
        ]
    }
    
    try:
        iam_client.create_role(
            RoleName=role_name,
            AssumeRolePolicyDocument=json.dumps(trust_policy),
            Description='IAM role for AWS Glue service'
        )
        print(f"✅ Created IAM role: {role_name}")
    except ClientError as e:
        if e.response['Error']['Code'] == 'EntityAlreadyExists':
            print(f"ℹ️  IAM role already exists: {role_name}")
        else:
            print(f"❌ Error creating IAM role: {e}")

def main():
    """Initialize all AWS resources"""
    print("🔧 Initializing AWS resources in LocalStack...")
    
    clients = get_aws_clients()
    
    # Create S3 buckets
    create_s3_buckets(clients['s3'])
    
    # Create IAM role
    create_iam_role(clients['iam'])
    
    print("✅ AWS resources initialization complete!")

if __name__ == '__main__':
    main()