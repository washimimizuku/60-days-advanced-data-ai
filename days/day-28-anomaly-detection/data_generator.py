#!/usr/bin/env python3
"""
Anomaly Detection Data Generator
Generates realistic data with injected anomalies for various scenarios
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import redis
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
from kafka import KafkaProducer
import json
import time
import logging
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDataGenerator:
    """Generate realistic data with controlled anomaly injection"""
    
    def __init__(self):
        self.setup_connections()
        self.anomaly_injection_rate = float(os.getenv('DEFAULT_CONTAMINATION', 0.1))
    
    def setup_connections(self):
        """Setup database and streaming connections"""
        # PostgreSQL
        self.pg_conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB', 'anomaly_db'),
            user=os.getenv('POSTGRES_USER', 'anomaly_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'anomaly_pass')
        )
        
        # Redis
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )
        
        # InfluxDB
        self.influx_client = InfluxDBClient(
            url=f"http://{os.getenv('INFLUXDB_HOST', 'localhost')}:8086",
            token=os.getenv('INFLUXDB_TOKEN', 'anomaly-token-12345'),
            org=os.getenv('INFLUXDB_ORG', 'anomaly-org')
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
        
        # Kafka
        try:
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092').split(','),
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
        except Exception as e:
            logger.warning(f"Kafka not available: {e}")
            self.kafka_producer = None
    
    def generate_financial_transactions(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate financial transaction data with fraud anomalies"""
        
        np.random.seed(42)
        
        # Normal transaction patterns
        amounts = np.random.lognormal(mean=3, sigma=1, size=n_samples)
        amounts = np.clip(amounts, 1, 10000)
        
        # Transaction times (business hours bias)
        hours = np.random.choice(range(24), size=n_samples, 
                                p=[0.01]*6 + [0.05]*2 + [0.15]*10 + [0.08]*4 + [0.02]*2)
        
        # Merchant categories
        categories = np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], 
                                    size=n_samples, p=[0.3, 0.2, 0.25, 0.15, 0.1])
        
        # Geographic locations (simplified)
        locations = np.random.choice(['domestic', 'international'], 
                                   size=n_samples, p=[0.95, 0.05])
        
        # Create base dataframe
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='5min'),
            'amount': amounts,
            'hour': hours,
            'category': categories,
            'location': locations,
            'is_anomaly': False
        })
        
        # Inject anomalies
        n_anomalies = int(n_samples * self.anomaly_injection_rate)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['amount', 'time', 'location', 'frequency'])
            
            if anomaly_type == 'amount':
                # Unusually high amounts
                df.loc[idx, 'amount'] = np.random.uniform(5000, 50000)
            elif anomaly_type == 'time':
                # Unusual transaction times
                df.loc[idx, 'hour'] = np.random.choice([2, 3, 4, 5])
            elif anomaly_type == 'location':
                # Suspicious international transactions
                df.loc[idx, 'location'] = 'international'
                df.loc[idx, 'amount'] = np.random.uniform(1000, 5000)
            elif anomaly_type == 'frequency':
                # Multiple rapid transactions
                base_time = df.loc[idx, 'timestamp']
                for i in range(5):
                    if idx + i < len(df):
                        df.loc[idx + i, 'timestamp'] = base_time + timedelta(minutes=i)
                        df.loc[idx + i, 'is_anomaly'] = True
            
            df.loc[idx, 'is_anomaly'] = True
        
        return df
    
    def generate_network_traffic(self, n_samples: int = 50000) -> pd.DataFrame:
        """Generate network traffic data with intrusion anomalies"""
        
        np.random.seed(123)
        
        # Normal traffic patterns
        packet_sizes = np.random.exponential(scale=500, size=n_samples)
        packet_sizes = np.clip(packet_sizes, 64, 1500)
        
        # Connection durations
        durations = np.random.exponential(scale=30, size=n_samples)
        durations = np.clip(durations, 0.1, 300)
        
        # Protocols
        protocols = np.random.choice(['TCP', 'UDP', 'ICMP'], 
                                   size=n_samples, p=[0.7, 0.25, 0.05])
        
        # Ports
        common_ports = [80, 443, 22, 21, 25, 53, 110, 143]
        ports = np.random.choice(common_ports + list(range(1024, 65536)), 
                               size=n_samples, p=[0.1]*8 + [0.2/64512]*64512)
        
        # Create base dataframe
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1s'),
            'packet_size': packet_sizes,
            'duration': durations,
            'protocol': protocols,
            'port': ports,
            'bytes_sent': np.random.exponential(1000, n_samples),
            'bytes_received': np.random.exponential(800, n_samples),
            'is_anomaly': False
        })
        
        # Inject network anomalies
        n_anomalies = int(n_samples * self.anomaly_injection_rate)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['ddos', 'port_scan', 'data_exfiltration', 'unusual_protocol'])
            
            if anomaly_type == 'ddos':
                # High packet rate, small packets
                df.loc[idx, 'packet_size'] = np.random.uniform(64, 128)
                df.loc[idx, 'duration'] = np.random.uniform(0.001, 0.1)
            elif anomaly_type == 'port_scan':
                # Multiple connections to different ports
                df.loc[idx, 'port'] = np.random.randint(1, 1024)
                df.loc[idx, 'duration'] = np.random.uniform(0.1, 1)
            elif anomaly_type == 'data_exfiltration':
                # Large data transfers
                df.loc[idx, 'bytes_sent'] = np.random.uniform(10000, 100000)
                df.loc[idx, 'duration'] = np.random.uniform(60, 300)
            elif anomaly_type == 'unusual_protocol':
                # Rare protocols
                df.loc[idx, 'protocol'] = 'ICMP'
                df.loc[idx, 'packet_size'] = np.random.uniform(1000, 1500)
            
            df.loc[idx, 'is_anomaly'] = True
        
        return df
    
    def generate_sensor_data(self, n_samples: int = 100000) -> pd.DataFrame:
        """Generate IoT sensor data with equipment failure anomalies"""
        
        np.random.seed(456)
        
        # Normal sensor readings with daily patterns
        timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1min')
        
        # Temperature with daily cycle
        base_temp = 20 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60))
        temperature = base_temp + np.random.normal(0, 2, n_samples)
        
        # Pressure (correlated with temperature)
        pressure = 1013 + 0.5 * (temperature - 20) + np.random.normal(0, 5, n_samples)
        
        # Vibration (equipment health indicator)
        vibration = np.random.exponential(scale=0.5, size=n_samples)
        
        # Power consumption
        power = 100 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / (24 * 60)) + np.random.normal(0, 5, n_samples)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'pressure': pressure,
            'vibration': vibration,
            'power_consumption': power,
            'is_anomaly': False
        })
        
        # Inject equipment anomalies
        n_anomalies = int(n_samples * self.anomaly_injection_rate)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['overheating', 'pressure_drop', 'vibration_spike', 'power_surge'])
            
            if anomaly_type == 'overheating':
                df.loc[idx, 'temperature'] = np.random.uniform(60, 80)
                df.loc[idx, 'power_consumption'] *= 1.5
            elif anomaly_type == 'pressure_drop':
                df.loc[idx, 'pressure'] = np.random.uniform(900, 950)
            elif anomaly_type == 'vibration_spike':
                df.loc[idx, 'vibration'] = np.random.uniform(5, 15)
            elif anomaly_type == 'power_surge':
                df.loc[idx, 'power_consumption'] = np.random.uniform(200, 300)
            
            df.loc[idx, 'is_anomaly'] = True
        
        return df
    
    def generate_user_behavior(self, n_samples: int = 20000) -> pd.DataFrame:
        """Generate user behavior data with suspicious activity anomalies"""
        
        np.random.seed(789)
        
        # Normal user behavior patterns
        session_durations = np.random.lognormal(mean=3, sigma=1, size=n_samples)
        session_durations = np.clip(session_durations, 1, 300)  # 1 second to 5 minutes
        
        # Page views per session
        page_views = np.random.poisson(lam=5, size=n_samples)
        page_views = np.clip(page_views, 1, 50)
        
        # Click rates
        click_rates = np.random.beta(a=2, b=5, size=n_samples)
        
        # Login attempts
        login_attempts = np.random.choice([1, 2, 3], size=n_samples, p=[0.8, 0.15, 0.05])
        
        # Geographic locations
        locations = np.random.choice(['US', 'EU', 'ASIA', 'OTHER'], 
                                   size=n_samples, p=[0.5, 0.3, 0.15, 0.05])
        
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='30s'),
            'session_duration': session_durations,
            'page_views': page_views,
            'click_rate': click_rates,
            'login_attempts': login_attempts,
            'location': locations,
            'is_anomaly': False
        })
        
        # Inject behavioral anomalies
        n_anomalies = int(n_samples * self.anomaly_injection_rate)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['bot_behavior', 'brute_force', 'scraping', 'location_anomaly'])
            
            if anomaly_type == 'bot_behavior':
                df.loc[idx, 'session_duration'] = np.random.uniform(0.1, 2)
                df.loc[idx, 'page_views'] = np.random.randint(20, 100)
                df.loc[idx, 'click_rate'] = 0
            elif anomaly_type == 'brute_force':
                df.loc[idx, 'login_attempts'] = np.random.randint(10, 50)
                df.loc[idx, 'session_duration'] = np.random.uniform(0.5, 5)
            elif anomaly_type == 'scraping':
                df.loc[idx, 'page_views'] = np.random.randint(100, 1000)
                df.loc[idx, 'session_duration'] = np.random.uniform(10, 60)
                df.loc[idx, 'click_rate'] = 0
            elif anomaly_type == 'location_anomaly':
                df.loc[idx, 'location'] = 'OTHER'
                df.loc[idx, 'login_attempts'] = np.random.randint(5, 15)
            
            df.loc[idx, 'is_anomaly'] = True
        
        return df
    
    def store_to_postgres(self, df: pd.DataFrame, table_name: str):
        """Store data to PostgreSQL"""
        try:
            df.to_sql(table_name, self.pg_conn, if_exists='replace', index=False, method='multi')
            logger.info(f"Stored {len(df)} records to PostgreSQL table: {table_name}")
        except Exception as e:
            logger.error(f"Error storing to PostgreSQL: {e}")
    
    def store_to_influxdb(self, df: pd.DataFrame, measurement: str):
        """Store data to InfluxDB"""
        try:
            points = []
            for _, row in df.iterrows():
                point = Point(measurement)
                
                # Add tags
                for col in ['category', 'location', 'protocol']:
                    if col in row and pd.notna(row[col]):
                        point = point.tag(col, str(row[col]))
                
                # Add anomaly tag
                point = point.tag("is_anomaly", str(row['is_anomaly']))
                
                # Add fields (numeric values)
                for col in df.columns:
                    if col not in ['timestamp', 'category', 'location', 'protocol', 'is_anomaly']:
                        if pd.notna(row[col]) and isinstance(row[col], (int, float)):
                            point = point.field(col, float(row[col]))
                
                # Set timestamp
                point = point.time(row['timestamp'])
                points.append(point)
            
            self.write_api.write(
                bucket=os.getenv('INFLUXDB_BUCKET', 'anomalies'),
                record=points
            )
            logger.info(f"Stored {len(points)} points to InfluxDB measurement: {measurement}")
        except Exception as e:
            logger.error(f"Error storing to InfluxDB: {e}")
    
    def stream_to_kafka(self, df: pd.DataFrame, topic: str):
        """Stream data to Kafka for real-time processing"""
        if not self.kafka_producer:
            logger.warning("Kafka producer not available")
            return
        
        try:
            for _, row in df.iterrows():
                message = {
                    'timestamp': row['timestamp'].isoformat(),
                    'data': row.drop('timestamp').to_dict(),
                    'is_anomaly': bool(row['is_anomaly'])
                }
                
                self.kafka_producer.send(topic, value=message)
                time.sleep(0.01)  # Small delay for realistic streaming
            
            self.kafka_producer.flush()
            logger.info(f"Streamed {len(df)} messages to Kafka topic: {topic}")
        except Exception as e:
            logger.error(f"Error streaming to Kafka: {e}")
    
    def cache_statistics(self, df: pd.DataFrame, dataset_name: str):
        """Cache dataset statistics to Redis"""
        try:
            stats = {
                'total_records': len(df),
                'anomaly_count': int(df['is_anomaly'].sum()),
                'anomaly_rate': float(df['is_anomaly'].mean()),
                'last_updated': datetime.now().isoformat()
            }
            
            # Add numeric column statistics
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'is_anomaly':
                    stats[f'{col}_mean'] = float(df[col].mean())
                    stats[f'{col}_std'] = float(df[col].std())
                    stats[f'{col}_min'] = float(df[col].min())
                    stats[f'{col}_max'] = float(df[col].max())
            
            # Store in Redis
            for key, value in stats.items():
                self.redis_client.set(f"{dataset_name}:{key}", str(value))
            
            logger.info(f"Cached statistics for {dataset_name} to Redis")
        except Exception as e:
            logger.error(f"Error caching statistics: {e}")
    
    def generate_all_datasets(self):
        """Generate all anomaly detection datasets"""
        logger.info("Starting anomaly detection data generation...")
        
        datasets = {
            'financial_transactions': self.generate_financial_transactions(10000),
            'network_traffic': self.generate_network_traffic(50000),
            'sensor_data': self.generate_sensor_data(100000),
            'user_behavior': self.generate_user_behavior(20000)
        }
        
        for name, df in datasets.items():
            logger.info(f"Processing dataset: {name}")
            logger.info(f"  Total records: {len(df)}")
            logger.info(f"  Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean():.2%})")
            
            # Store to all systems
            self.store_to_postgres(df, name)
            self.store_to_influxdb(df, name)
            self.cache_statistics(df, name)
            
            # Stream subset to Kafka for real-time demo
            sample_df = df.sample(min(1000, len(df)))
            self.stream_to_kafka(sample_df, 'sensor-data')
            
            # Save to CSV for backup
            df.to_csv(f"{name}.csv", index=False)
            logger.info(f"Saved {name} to CSV")
        
        logger.info("Data generation completed!")
    
    def close_connections(self):
        """Close all connections"""
        self.pg_conn.close()
        self.redis_client.close()
        self.influx_client.close()
        if self.kafka_producer:
            self.kafka_producer.close()

def main():
    """Main execution function"""
    generator = AnomalyDataGenerator()
    
    try:
        # Wait for services to be ready
        logger.info("Waiting for services to be ready...")
        time.sleep(30)
        
        # Generate data
        generator.generate_all_datasets()
        
    except Exception as e:
        logger.error(f"Error in data generation: {e}")
    finally:
        generator.close_connections()

if __name__ == "__main__":
    main()