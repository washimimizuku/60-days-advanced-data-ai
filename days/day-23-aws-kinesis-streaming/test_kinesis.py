#!/usr/bin/env python3
"""
Day 23: AWS Kinesis & Streaming - Tests
"""

import pytest
import boto3
import json
import os
from dotenv import load_dotenv
from exercise import KinesisStreamsManager, TransactionDataGenerator

load_dotenv()

@pytest.fixture
def kinesis_client():
    return boto3.client(
        'kinesis',
        endpoint_url=os.getenv('AWS_ENDPOINT_URL'),
        aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_DEFAULT_REGION')
    )

@pytest.fixture
def streams_manager():
    return KinesisStreamsManager()

@pytest.fixture
def data_generator():
    return TransactionDataGenerator()

def test_kinesis_connection(kinesis_client):
    """Test basic Kinesis connectivity"""
    response = kinesis_client.list_streams()
    assert 'StreamNames' in response

def test_stream_exists(kinesis_client):
    """Test that the main stream exists"""
    stream_name = os.getenv('KINESIS_STREAM_NAME')
    response = kinesis_client.describe_stream(StreamName=stream_name)
    assert response['StreamDescription']['StreamStatus'] == 'ACTIVE'

def test_transaction_generation(data_generator):
    """Test transaction data generation"""
    transactions = data_generator.generate_transaction_batch(batch_size=10)
    
    assert len(transactions) == 10
    assert all('transaction_id' in t for t in transactions)
    assert all('customer_id' in t for t in transactions)
    assert all('amount' in t for t in transactions)

def test_fraud_transaction_generation(data_generator):
    """Test fraud transaction generation"""
    transactions = data_generator.generate_transaction_batch(
        batch_size=100, 
        fraud_rate=0.1
    )
    
    fraud_count = sum(1 for t in transactions if t.get('is_fraud_simulation', False))
    assert fraud_count == 10  # 10% of 100

def test_kinesis_put_records(streams_manager, data_generator):
    """Test putting records to Kinesis"""
    transactions = data_generator.generate_transaction_batch(batch_size=5)
    
    result = streams_manager.put_records_with_batching(
        os.getenv('KINESIS_STREAM_NAME'), 
        transactions
    )
    
    assert result['total_records'] == 5
    assert result['successful_records'] >= 0
    assert result['success_rate'] >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])