#!/usr/bin/env python3
"""
FinTech Sample Data Generator for Advanced Feature Engineering

Generates realistic financial data including:
- Customer demographics and behavior
- Transaction history with temporal patterns
- Customer feedback and support tickets
- Account information and credit data
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import psycopg2
import redis
from faker import Faker
from tqdm import tqdm
import json

fake = Faker()

class FinTechDataGenerator:
    """Generate realistic FinTech data for feature engineering"""
    
    def __init__(self):
        self.fake = Faker()
        
        # Configuration from environment
        self.num_customers = int(os.getenv('SAMPLE_CUSTOMERS', 10000))
        self.num_transactions = int(os.getenv('SAMPLE_TRANSACTIONS', 100000))
        self.days_of_data = int(os.getenv('DAYS_OF_DATA', 90))
        
        # Database connections
        self.postgres_conn = None
        self.redis_client = None
        
        self._setup_connections()
    
    def _setup_connections(self):
        """Setup database connections"""
        
        # PostgreSQL connection
        try:
            self.postgres_conn = psycopg2.connect(
                host=os.getenv('POSTGRES_HOST', 'localhost'),
                database=os.getenv('POSTGRES_DB', 'fintech_features'),
                user=os.getenv('POSTGRES_USER', 'feature_user'),
                password=os.getenv('POSTGRES_PASSWORD', 'feature_password'),
                port=int(os.getenv('POSTGRES_PORT', 5432))
            )
            print("✅ Connected to PostgreSQL")
        except Exception as e:
            print(f"❌ PostgreSQL connection failed: {e}")
            sys.exit(1)
        
        # Redis connection
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                decode_responses=True
            )
            self.redis_client.ping()
            print("✅ Connected to Redis")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            sys.exit(1)
    
    def generate_customers(self) -> pd.DataFrame:
        """Generate customer demographic and profile data"""
        
        print(f"👥 Generating {self.num_customers} customer profiles...")
        
        customers = []
        
        for customer_id in tqdm(range(1, self.num_customers + 1)):
            # Demographics
            age = np.random.normal(40, 15)
            age = max(18, min(80, int(age)))
            
            # Income based on age and education
            education_level = np.random.choice(['high_school', 'bachelor', 'master', 'phd'], 
                                             p=[0.3, 0.4, 0.25, 0.05])
            
            base_income = {
                'high_school': 35000,
                'bachelor': 55000,
                'master': 75000,
                'phd': 95000
            }[education_level]
            
            # Age factor for income
            age_factor = 1 + (age - 25) * 0.02
            income = base_income * age_factor * np.random.lognormal(0, 0.3)
            income = max(20000, min(500000, income))
            
            # Credit score based on age and income
            base_credit = 600 + (income - 35000) * 0.0003 + (age - 25) * 2
            credit_score = int(np.random.normal(base_credit, 50))
            credit_score = max(300, min(850, credit_score))
            
            # Account information
            account_age = np.random.exponential(3)  # Years
            account_age = max(0.1, min(20, account_age))
            
            customer_data = {
                'customer_id': customer_id,
                'age': age,
                'income': income,
                'education_level': education_level,
                'credit_score': credit_score,
                'account_age_years': account_age,
                'account_type': np.random.choice(['basic', 'premium', 'business'], p=[0.6, 0.3, 0.1]),
                'city': self.fake.city(),
                'state': self.fake.state_abbr(),
                'employment_status': np.random.choice(['employed', 'self_employed', 'unemployed', 'retired'], 
                                                    p=[0.7, 0.15, 0.05, 0.1]),
                'marital_status': np.random.choice(['single', 'married', 'divorced'], p=[0.4, 0.5, 0.1]),
                'num_dependents': np.random.poisson(1.2),
                'created_at': datetime.now() - timedelta(days=account_age * 365)
            }
            
            customers.append(customer_data)
        
        return pd.DataFrame(customers)
    
    def generate_transactions(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate transaction history with temporal patterns"""
        
        print(f"💳 Generating {self.num_transactions} transactions...")
        
        transactions = []
        
        # Transaction categories and their typical amounts
        categories = {
            'groceries': (20, 150),
            'gas': (25, 80),
            'restaurants': (15, 200),
            'shopping': (30, 500),
            'utilities': (50, 300),
            'entertainment': (20, 150),
            'healthcare': (50, 1000),
            'travel': (100, 2000),
            'transfer': (50, 5000),
            'atm': (20, 500)
        }
        
        for _ in tqdm(range(self.num_transactions)):
            # Select random customer
            customer = customers_df.sample(1).iloc[0]
            
            # Generate timestamp with realistic patterns
            days_ago = np.random.exponential(self.days_of_data / 4)
            days_ago = min(days_ago, self.days_of_data)
            
            base_time = datetime.now() - timedelta(days=days_ago)
            
            # Add time of day patterns
            if np.random.random() < 0.7:  # Business hours
                hour = np.random.normal(14, 4)  # Peak around 2 PM
                hour = max(6, min(22, int(hour)))
            else:  # Off hours
                hour = np.random.choice([0, 1, 2, 3, 4, 5, 23])
            
            timestamp = base_time.replace(
                hour=hour,
                minute=np.random.randint(0, 60),
                second=np.random.randint(0, 60)
            )
            
            # Select category and amount
            category = np.random.choice(list(categories.keys()))
            min_amt, max_amt = categories[category]
            
            # Adjust amount based on customer income
            income_factor = customer['income'] / 50000
            amount = np.random.uniform(min_amt, max_amt) * income_factor
            
            # Add some randomness
            amount *= np.random.lognormal(0, 0.3)
            amount = max(1, round(amount, 2))
            
            # Transaction type
            transaction_type = np.random.choice(['debit', 'credit', 'transfer'], p=[0.6, 0.3, 0.1])
            
            # Merchant information
            merchant_name = f"{category.title()} Store {np.random.randint(1, 1000)}"
            
            transaction_data = {
                'transaction_id': f"txn_{len(transactions) + 1}",
                'customer_id': customer['customer_id'],
                'timestamp': timestamp,
                'amount': amount,
                'category': category,
                'transaction_type': transaction_type,
                'merchant_name': merchant_name,
                'merchant_city': self.fake.city(),
                'merchant_state': self.fake.state_abbr(),
                'is_weekend': timestamp.weekday() >= 5,
                'hour_of_day': timestamp.hour,
                'day_of_week': timestamp.weekday(),
                'is_holiday': self._is_holiday(timestamp),
                'channel': np.random.choice(['online', 'atm', 'pos', 'mobile'], p=[0.4, 0.1, 0.4, 0.1])
            }
            
            transactions.append(transaction_data)
        
        return pd.DataFrame(transactions)
    
    def generate_feedback(self, customers_df: pd.DataFrame) -> pd.DataFrame:
        """Generate customer feedback and support tickets"""
        
        print("💬 Generating customer feedback...")
        
        feedback_data = []
        
        # Feedback templates by sentiment
        positive_templates = [
            "Excellent service! The {feature} works perfectly and the {aspect} is outstanding.",
            "I love using this {product}. The {feature} is amazing and {aspect} exceeded my expectations.",
            "Great experience with {feature}. Very satisfied with the {aspect} and overall service.",
            "Outstanding {product}! The {feature} is user-friendly and {aspect} is top-notch.",
            "Highly recommend this service. The {feature} is reliable and {aspect} is excellent."
        ]
        
        negative_templates = [
            "Terrible experience with {feature}. The {aspect} is disappointing and needs improvement.",
            "Very frustrated with {product}. The {feature} doesn't work properly and {aspect} is awful.",
            "Poor service quality. The {feature} is confusing and {aspect} is unacceptable.",
            "Disappointed with {feature}. The {aspect} is slow and unreliable.",
            "Worst {product} experience. The {feature} is broken and {aspect} is horrible."
        ]
        
        neutral_templates = [
            "The {feature} is okay but the {aspect} could be better.",
            "Average experience with {product}. The {feature} works but {aspect} needs work.",
            "The {feature} is decent. Some issues with {aspect} but generally acceptable.",
            "Standard service. The {feature} is fine and {aspect} is average.",
            "The {product} is okay. {feature} works as expected and {aspect} is reasonable."
        ]
        
        features = ['mobile app', 'website', 'customer service', 'ATM network', 'online banking']
        aspects = ['user interface', 'response time', 'security', 'functionality', 'support']
        products = ['banking service', 'credit card', 'loan process', 'investment platform']
        
        # Generate feedback for subset of customers
        feedback_customers = customers_df.sample(n=min(5000, len(customers_df)))
        
        for _, customer in tqdm(feedback_customers.iterrows(), total=len(feedback_customers)):
            # Generate 1-3 feedback entries per customer
            num_feedback = np.random.poisson(1.5)
            num_feedback = max(1, min(3, num_feedback))
            
            for _ in range(num_feedback):
                # Determine sentiment based on customer satisfaction
                satisfaction_score = np.random.beta(2, 1)  # Skewed towards positive
                
                if satisfaction_score > 0.7:
                    sentiment = 'positive'
                    template = np.random.choice(positive_templates)
                elif satisfaction_score < 0.3:
                    sentiment = 'negative'
                    template = np.random.choice(negative_templates)
                else:
                    sentiment = 'neutral'
                    template = np.random.choice(neutral_templates)
                
                # Fill template
                feedback_text = template.format(
                    feature=np.random.choice(features),
                    aspect=np.random.choice(aspects),
                    product=np.random.choice(products)
                )
                
                # Add timestamp
                days_ago = np.random.exponential(30)  # More recent feedback
                days_ago = min(days_ago, self.days_of_data)
                timestamp = datetime.now() - timedelta(days=days_ago)
                
                feedback_entry = {
                    'feedback_id': f"fb_{len(feedback_data) + 1}",
                    'customer_id': customer['customer_id'],
                    'timestamp': timestamp,
                    'feedback_text': feedback_text,
                    'sentiment': sentiment,
                    'satisfaction_score': satisfaction_score,
                    'channel': np.random.choice(['email', 'phone', 'chat', 'survey'], p=[0.3, 0.2, 0.3, 0.2]),
                    'category': np.random.choice(['complaint', 'suggestion', 'compliment', 'question'], 
                                               p=[0.3, 0.2, 0.3, 0.2]),
                    'resolved': np.random.choice([True, False], p=[0.8, 0.2])
                }
                
                feedback_data.append(feedback_entry)
        
        return pd.DataFrame(feedback_data)
    
    def _is_holiday(self, date: datetime) -> bool:
        """Simple holiday detection"""
        # Major US holidays (simplified)
        holidays = [
            (1, 1),   # New Year
            (7, 4),   # Independence Day
            (12, 25), # Christmas
            (11, 24), # Thanksgiving (approximate)
        ]
        
        return (date.month, date.day) in holidays
    
    def save_to_postgres(self, df: pd.DataFrame, table_name: str):
        """Save DataFrame to PostgreSQL"""
        
        try:
            cursor = self.postgres_conn.cursor()
            
            # Create table schema based on DataFrame
            columns = []
            for col, dtype in df.dtypes.items():
                if 'int' in str(dtype):
                    pg_type = 'INTEGER'
                elif 'float' in str(dtype):
                    pg_type = 'REAL'
                elif 'datetime' in str(dtype):
                    pg_type = 'TIMESTAMP'
                elif 'bool' in str(dtype):
                    pg_type = 'BOOLEAN'
                else:
                    pg_type = 'TEXT'
                
                columns.append(f"{col} {pg_type}")
            
            # Drop and create table
            cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
            create_table_sql = f"""
            CREATE TABLE {table_name} (
                {', '.join(columns)}
            )
            """
            cursor.execute(create_table_sql)
            
            # Insert data in batches
            batch_size = 1000
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                for _, row in batch.iterrows():
                    placeholders = ', '.join(['%s'] * len(row))
                    insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
                    cursor.execute(insert_sql, tuple(row))
            
            self.postgres_conn.commit()
            cursor.close()
            
            print(f"💾 Saved {len(df)} records to PostgreSQL table {table_name}")
            
        except Exception as e:
            print(f"❌ Failed to save to PostgreSQL: {e}")
            self.postgres_conn.rollback()
    
    def save_to_parquet(self, df: pd.DataFrame, filename: str):
        """Save DataFrame to Parquet format"""
        
        os.makedirs('data', exist_ok=True)
        filepath = f"data/{filename}.parquet"
        
        df.to_parquet(filepath, index=False)
        print(f"💾 Saved {len(df)} records to {filepath}")
    
    def cache_to_redis(self, key: str, data: Dict[str, Any]):
        """Cache summary data to Redis"""
        
        try:
            self.redis_client.setex(key, 3600, json.dumps(data, default=str))
            print(f"📦 Cached data to Redis key: {key}")
        except Exception as e:
            print(f"❌ Failed to cache to Redis: {e}")
    
    def generate_all_data(self):
        """Generate complete FinTech dataset"""
        
        print("🎯 Starting comprehensive FinTech data generation...")
        
        # Generate datasets
        customers_df = self.generate_customers()
        transactions_df = self.generate_transactions(customers_df)
        feedback_df = self.generate_feedback(customers_df)
        
        # Save to Parquet files
        self.save_to_parquet(customers_df, 'customers')
        self.save_to_parquet(transactions_df, 'transactions')
        self.save_to_parquet(feedback_df, 'feedback')
        
        # Save to PostgreSQL
        self.save_to_postgres(customers_df, 'customers')
        self.save_to_postgres(transactions_df, 'transactions')
        self.save_to_postgres(feedback_df, 'feedback')
        
        # Cache summary statistics to Redis
        summary_stats = {
            'customers_count': len(customers_df),
            'transactions_count': len(transactions_df),
            'feedback_count': len(feedback_df),
            'date_range': {
                'start': transactions_df['timestamp'].min().isoformat(),
                'end': transactions_df['timestamp'].max().isoformat()
            },
            'total_transaction_volume': float(transactions_df['amount'].sum()),
            'avg_transaction_amount': float(transactions_df['amount'].mean()),
            'generated_at': datetime.now().isoformat()
        }
        
        self.cache_to_redis('fintech:summary', summary_stats)
        
        print("\n✅ Data generation completed successfully!")
        print(f"📊 Generated datasets:")
        print(f"   • Customers: {len(customers_df):,} records")
        print(f"   • Transactions: {len(transactions_df):,} records")
        print(f"   • Feedback: {len(feedback_df):,} records")
        print(f"   • Date range: {summary_stats['date_range']['start'][:10]} to {summary_stats['date_range']['end'][:10]}")
        print(f"   • Total volume: ${summary_stats['total_transaction_volume']:,.2f}")
        
        return {
            'customers': customers_df,
            'transactions': transactions_df,
            'feedback': feedback_df,
            'summary': summary_stats
        }

if __name__ == "__main__":
    generator = FinTechDataGenerator()
    datasets = generator.generate_all_data()