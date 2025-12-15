#!/usr/bin/env python3
"""
Day 22: AWS Glue & Data Catalog - Sample Data Generator
Generate realistic sample data for the exercise
"""

import pandas as pd
import numpy as np
import boto3
import os
from datetime import datetime, timedelta
from io import StringIO

def get_s3_client():
    """Get S3 client for LocalStack"""
    endpoint_url = os.getenv('AWS_ENDPOINT_URL', 'http://localhost:4566')
    return boto3.client('s3', endpoint_url=endpoint_url)

def generate_customer_data(num_customers=1000):
    """Generate customer profile data"""
    np.random.seed(42)
    
    segments = ['premium', 'standard', 'basic']
    
    data = {
        'customer_id': [f'CUST{i:06d}' for i in range(1, num_customers + 1)],
        'first_name': [f'FirstName{i}' for i in range(1, num_customers + 1)],
        'last_name': [f'LastName{i}' for i in range(1, num_customers + 1)],
        'email_hash': [f'hash_{i:06d}' for i in range(1, num_customers + 1)],
        'registration_date': pd.date_range('2020-01-01', periods=num_customers, freq='D')[:num_customers],
        'customer_segment': np.random.choice(segments, num_customers, p=[0.2, 0.5, 0.3]),
        'lifetime_value': np.random.lognormal(6, 1, num_customers).round(2)
    }
    
    return pd.DataFrame(data)

def generate_transaction_data(num_transactions=10000):
    """Generate transaction data"""
    np.random.seed(42)
    
    payment_methods = ['credit_card', 'debit_card', 'paypal', 'bank_transfer']
    categories = ['electronics', 'clothing', 'books', 'home', 'sports']
    currencies = ['USD', 'EUR', 'GBP']
    
    data = {
        'transaction_id': [f'TXN{i:08d}' for i in range(1, num_transactions + 1)],
        'customer_id': [f'CUST{np.random.randint(1, 1001):06d}' for _ in range(num_transactions)],
        'product_id': [f'PROD{np.random.randint(1, 5001):06d}' for _ in range(num_transactions)],
        'transaction_amount': np.random.lognormal(4, 1, num_transactions).round(2),
        'transaction_date': pd.date_range('2024-01-01', periods=num_transactions, freq='5min')[:num_transactions],
        'payment_method': np.random.choice(payment_methods, num_transactions),
        'merchant_category': np.random.choice(categories, num_transactions),
        'currency_code': np.random.choice(currencies, num_transactions, p=[0.7, 0.2, 0.1])
    }
    
    return pd.DataFrame(data)

def upload_to_s3(s3_client, df, bucket, key):
    """Upload DataFrame to S3 as CSV"""
    csv_buffer = StringIO()
    df.to_csv(csv_buffer, index=False)
    
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=csv_buffer.getvalue()
        )
        print(f"✅ Uploaded {len(df)} records to s3://{bucket}/{key}")
    except Exception as e:
        print(f"❌ Error uploading to S3: {e}")

def main():
    """Generate and upload sample data"""
    print("📊 Generating sample data...")
    
    s3_client = get_s3_client()
    bucket = os.getenv('S3_DATA_BUCKET', 'serverlessdata-datalake')
    
    # Generate customer data
    customers_df = generate_customer_data(1000)
    
    # Upload customers by segment (partitioned)
    for segment in customers_df['customer_segment'].unique():
        segment_df = customers_df[customers_df['customer_segment'] == segment]
        key = f"raw/customers/segment={segment}/customers.csv"
        upload_to_s3(s3_client, segment_df, bucket, key)
    
    # Generate transaction data
    transactions_df = generate_transaction_data(10000)
    
    # Upload transactions by date (partitioned)
    transactions_df['year'] = transactions_df['transaction_date'].dt.year
    transactions_df['month'] = transactions_df['transaction_date'].dt.month.astype(str).str.zfill(2)
    transactions_df['day'] = transactions_df['transaction_date'].dt.day.astype(str).str.zfill(2)
    
    for (year, month, day), group in transactions_df.groupby(['year', 'month', 'day']):
        key = f"raw/transactions/year={year}/month={month}/day={day}/transactions.csv"
        upload_to_s3(s3_client, group.drop(['year', 'month', 'day'], axis=1), bucket, key)
    
    print("✅ Sample data generation complete!")

if __name__ == '__main__':
    main()