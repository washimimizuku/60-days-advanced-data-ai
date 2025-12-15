#!/bin/bash

echo "🚀 Initializing AWS resources in LocalStack..."

# Create S3 bucket
awslocal s3 mb s3://dataflow-production-data

# Create Kinesis stream
awslocal kinesis create-stream --stream-name dataflow_events --shard-count 2

echo "✅ AWS resources initialized successfully!"