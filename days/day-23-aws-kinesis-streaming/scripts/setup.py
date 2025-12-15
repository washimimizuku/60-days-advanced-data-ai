#!/usr/bin/env python3
"""
Setup script for Day 23: AWS Kinesis & Streaming
"""

import boto3
import json
import time
from faker import Faker
import random
import os
from dotenv import load_dotenv

load_dotenv()

def setup_customer_profiles():
    """Generate sample customer profiles in DynamoDB"""
    
    dynamodb = boto3.resource(
        'dynamodb',
        endpoint_url=os.getenv('AWS_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )
    
    table = dynamodb.Table('customer_profiles')
    fake = Faker()
    
    print("📊 Generating customer profiles...")
    
    segments = ['premium', 'standard', 'basic', 'new']
    
    with table.batch_writer() as batch:
        for i in range(1000):
            customer_id = f'CUST{i:06d}'
            segment = random.choice(segments)
            
            if segment == 'premium':
                avg_amount = random.uniform(500, 2000)
                risk_score = random.uniform(0.1, 0.3)
                lifetime_value = random.uniform(10000, 100000)
            elif segment == 'standard':
                avg_amount = random.uniform(100, 500)
                risk_score = random.uniform(0.2, 0.5)
                lifetime_value = random.uniform(1000, 10000)
            else:
                avg_amount = random.uniform(20, 200)
                risk_score = random.uniform(0.3, 0.8)
                lifetime_value = random.uniform(100, 1000)
            
            batch.put_item(Item={
                'customer_id': customer_id,
                'segment': segment,
                'avg_transaction_amount': round(avg_amount, 2),
                'risk_score': round(risk_score, 3),
                'lifetime_value': round(lifetime_value, 2),
                'account_age_days': random.randint(1, 1000),
                'fraud_incidents': random.randint(0, 2) if risk_score > 0.6 else 0
            })
    
    print("✅ Customer profiles generated successfully!")

def verify_setup():
    """Verify all AWS resources are properly configured"""
    
    # Test Kinesis
    kinesis = boto3.client(
        'kinesis',
        endpoint_url=os.getenv('AWS_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )
    
    try:
        response = kinesis.describe_stream(StreamName='realtimedata-transactions')
        print(f"✅ Kinesis stream: {response['StreamDescription']['StreamStatus']}")
    except Exception as e:
        print(f"❌ Kinesis stream error: {e}")
    
    # Test S3
    s3 = boto3.client(
        's3',
        endpoint_url=os.getenv('AWS_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )
    
    try:
        s3.head_bucket(Bucket='realtimedata-streaming-bucket')
        print("✅ S3 bucket: Available")
    except Exception as e:
        print(f"❌ S3 bucket error: {e}")

if __name__ == "__main__":
    print("🚀 Setting up Day 23: AWS Kinesis & Streaming")
    
    # Wait for LocalStack to be ready
    time.sleep(10)
    
    setup_customer_profiles()
    verify_setup()
    
    print("✅ Setup complete! Ready for streaming exercises.")