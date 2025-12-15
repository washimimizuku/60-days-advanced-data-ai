#!/bin/bash

echo "🚀 Initializing AWS resources in LocalStack..."

# Create S3 bucket
awslocal s3 mb s3://realtimedata-streaming-bucket

# Create Kinesis stream
awslocal kinesis create-stream --stream-name realtimedata-transactions --shard-count 3

# Create SNS topic
awslocal sns create-topic --name fraud-alerts

# Create DynamoDB table for customer profiles
awslocal dynamodb create-table \
    --table-name customer_profiles \
    --attribute-definitions AttributeName=customer_id,AttributeType=S \
    --key-schema AttributeName=customer_id,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST

# Wait for stream to be active
echo "⏳ Waiting for Kinesis stream to be active..."
awslocal kinesis wait stream-exists --stream-name realtimedata-transactions

echo "✅ AWS resources initialized successfully!"