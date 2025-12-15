#!/usr/bin/env python3
"""
Day 23: AWS Kinesis & Streaming - Interactive Demo
"""

import boto3
import json
import time
import os
from datetime import datetime
from faker import Faker
import random
from dotenv import load_dotenv

load_dotenv()

class KinesisStreamingDemo:
    def __init__(self):
        self.kinesis = boto3.client(
            'kinesis',
            endpoint_url=os.getenv('AWS_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )
        
        self.s3 = boto3.client(
            's3',
            endpoint_url=os.getenv('AWS_ENDPOINT_URL'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION')
        )
        
        self.stream_name = os.getenv('KINESIS_STREAM_NAME')
        self.fake = Faker()
    
    def generate_sample_transaction(self, is_fraud=False):
        """Generate a sample transaction"""
        
        transaction = {
            'transaction_id': f'TXN{random.randint(100000, 999999)}',
            'customer_id': f'CUST{random.randint(1, 1000):06d}',
            'merchant_id': f'MERCHANT{random.randint(1, 100):04d}',
            'amount': random.uniform(10, 1000) if not is_fraud else random.uniform(5000, 50000),
            'timestamp': datetime.now().isoformat(),
            'payment_method': random.choice(['credit_card', 'debit_card', 'bank_transfer']),
            'merchant_category': 'high_risk' if is_fraud else random.choice(['retail', 'grocery', 'gas']),
            'is_fraud_simulation': is_fraud
        }
        
        return transaction
    
    def send_to_kinesis(self, records):
        """Send records to Kinesis stream"""
        
        kinesis_records = []
        for record in records:
            kinesis_records.append({
                'Data': json.dumps(record),
                'PartitionKey': record['customer_id']
            })
        
        try:
            response = self.kinesis.put_records(
                Records=kinesis_records,
                StreamName=self.stream_name
            )
            
            failed_count = response['FailedRecordCount']
            successful_count = len(records) - failed_count
            
            print(f"📊 Sent {successful_count}/{len(records)} records to Kinesis")
            return successful_count
            
        except Exception as e:
            print(f"❌ Error sending to Kinesis: {e}")
            return 0
    
    def read_from_kinesis(self, limit=10):
        """Read records from Kinesis stream"""
        
        try:
            # Get stream description
            stream_desc = self.kinesis.describe_stream(StreamName=self.stream_name)
            shards = stream_desc['StreamDescription']['Shards']
            
            records = []
            for shard in shards[:1]:  # Read from first shard only
                shard_id = shard['ShardId']
                
                # Get shard iterator
                iterator_response = self.kinesis.get_shard_iterator(
                    StreamName=self.stream_name,
                    ShardId=shard_id,
                    ShardIteratorType='TRIM_HORIZON'
                )
                
                shard_iterator = iterator_response['ShardIterator']
                
                # Get records
                records_response = self.kinesis.get_records(
                    ShardIterator=shard_iterator,
                    Limit=limit
                )
                
                for record in records_response['Records']:
                    data = json.loads(record['Data'])
                    records.append(data)
            
            print(f"📖 Read {len(records)} records from Kinesis")
            return records
            
        except Exception as e:
            print(f"❌ Error reading from Kinesis: {e}")
            return []
    
    def run_demo(self):
        """Run interactive streaming demo"""
        
        print("🚀 AWS Kinesis Streaming Demo")
        print("=" * 50)
        
        while True:
            print("\nChoose an option:")
            print("1. Generate and send sample transactions")
            print("2. Generate fraud transactions")
            print("3. Read records from stream")
            print("4. Stream status")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                count = int(input("Number of transactions to generate (default 10): ") or 10)
                transactions = [self.generate_sample_transaction() for _ in range(count)]
                self.send_to_kinesis(transactions)
                
            elif choice == '2':
                count = int(input("Number of fraud transactions (default 5): ") or 5)
                fraud_transactions = [self.generate_sample_transaction(is_fraud=True) for _ in range(count)]
                self.send_to_kinesis(fraud_transactions)
                
            elif choice == '3':
                limit = int(input("Number of records to read (default 10): ") or 10)
                records = self.read_from_kinesis(limit)
                
                if records:
                    print("\n📋 Recent transactions:")
                    for i, record in enumerate(records[:5], 1):
                        fraud_flag = "🚨 FRAUD" if record.get('is_fraud_simulation') else "✅ NORMAL"
                        print(f"{i}. {record['transaction_id']}: ${record['amount']:.2f} - {fraud_flag}")
                
            elif choice == '4':
                try:
                    response = self.kinesis.describe_stream(StreamName=self.stream_name)
                    status = response['StreamDescription']['StreamStatus']
                    shard_count = len(response['StreamDescription']['Shards'])
                    print(f"📊 Stream Status: {status}")
                    print(f"📊 Shard Count: {shard_count}")
                except Exception as e:
                    print(f"❌ Error getting stream status: {e}")
                
            elif choice == '5':
                print("👋 Demo completed!")
                break
                
            else:
                print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    demo = KinesisStreamingDemo()
    demo.run_demo()