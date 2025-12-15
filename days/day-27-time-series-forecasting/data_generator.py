#!/usr/bin/env python3
"""
Time Series Data Generator for Forecasting
Generates realistic multi-seasonal time series data for various business scenarios
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import psycopg2
import redis
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import time
import logging
from typing import Dict, List, Tuple
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesGenerator:
    """Generate realistic time series data for forecasting scenarios"""
    
    def __init__(self):
        self.setup_connections()
    
    def setup_connections(self):
        """Setup database connections"""
        # PostgreSQL
        self.pg_conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB', 'timeseries_db'),
            user=os.getenv('POSTGRES_USER', 'forecast_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'forecast_pass')
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
            token=os.getenv('INFLUXDB_TOKEN', 'forecast-token-12345'),
            org=os.getenv('INFLUXDB_ORG', 'forecast-org')
        )
        self.write_api = self.influx_client.write_api(write_options=SYNCHRONOUS)
    
    def generate_retail_sales(self, start_date: str = '2020-01-01', periods: int = 1095) -> pd.DataFrame:
        """Generate retail sales data with seasonality and trends"""
        dates = pd.date_range(start_date, periods=periods, freq='D')
        
        # Base trend
        trend = np.linspace(1000, 1500, periods)
        
        # Seasonal patterns
        yearly_season = 200 * np.sin(2 * np.pi * np.arange(periods) / 365.25)
        weekly_season = 100 * np.sin(2 * np.pi * np.arange(periods) / 7)
        
        # Holiday effects
        holiday_boost = np.zeros(periods)
        for i, date in enumerate(dates):
            if date.month == 12 and date.day in [24, 25, 31]:  # Christmas/New Year
                holiday_boost[i] = 300
            elif date.month == 11 and 22 <= date.day <= 28 and date.weekday() == 4:  # Black Friday
                holiday_boost[i] = 400
        
        # Random noise
        noise = np.random.normal(0, 50, periods)
        
        # Combine components
        sales = trend + yearly_season + weekly_season + holiday_boost + noise
        sales = np.maximum(sales, 0)  # Ensure non-negative
        
        return pd.DataFrame({
            'timestamp': dates,
            'sales': sales,
            'category': 'retail',
            'location': 'store_001'
        })
    
    def generate_energy_consumption(self, start_date: str = '2020-01-01', periods: int = 8760) -> pd.DataFrame:
        """Generate hourly energy consumption data"""
        dates = pd.date_range(start_date, periods=periods, freq='H')
        
        # Base load
        base_load = 500
        
        # Daily pattern (higher during day)
        daily_pattern = 200 * np.sin(2 * np.pi * (dates.hour - 6) / 24)
        
        # Weekly pattern (lower on weekends)
        weekly_pattern = 100 * (1 - 0.3 * (dates.weekday >= 5))
        
        # Seasonal pattern (higher in summer/winter)
        seasonal_pattern = 150 * np.sin(2 * np.pi * dates.dayofyear / 365.25 + np.pi/2)
        
        # Temperature effect
        temp_effect = 100 * np.random.normal(0, 1, periods)
        
        # Random noise
        noise = np.random.normal(0, 25, periods)
        
        consumption = base_load + daily_pattern + weekly_pattern + seasonal_pattern + temp_effect + noise
        consumption = np.maximum(consumption, 50)
        
        return pd.DataFrame({
            'timestamp': dates,
            'consumption': consumption,
            'category': 'energy',
            'location': 'grid_001'
        })
    
    def generate_stock_prices(self, start_date: str = '2020-01-01', periods: int = 1095) -> pd.DataFrame:
        """Generate stock price data with volatility clustering"""
        dates = pd.date_range(start_date, periods=periods, freq='D')
        
        # Initial price
        price = 100.0
        prices = [price]
        
        # Generate returns with volatility clustering
        for i in range(1, periods):
            # Volatility clustering (GARCH-like)
            vol = 0.02 + 0.1 * abs(np.random.normal(0, 0.1))
            
            # Random walk with drift
            drift = 0.0005
            shock = np.random.normal(drift, vol)
            
            price = price * (1 + shock)
            prices.append(price)
        
        return pd.DataFrame({
            'timestamp': dates,
            'price': prices,
            'category': 'finance',
            'symbol': 'TECH001'
        })
    
    def generate_website_traffic(self, start_date: str = '2020-01-01', periods: int = 2190) -> pd.DataFrame:
        """Generate website traffic data with multiple seasonalities"""
        dates = pd.date_range(start_date, periods=periods, freq='12H')  # Twice daily
        
        # Base traffic
        base_traffic = 10000
        
        # Growth trend
        growth = np.linspace(0, 5000, periods)
        
        # Daily pattern (higher during business hours)
        daily_pattern = 3000 * np.sin(2 * np.pi * (dates.hour - 12) / 24)
        
        # Weekly pattern (lower on weekends)
        weekly_pattern = 2000 * (1 - 0.4 * (dates.weekday >= 5))
        
        # Monthly pattern (campaign effects)
        monthly_pattern = 1000 * np.sin(2 * np.pi * dates.day / 30)
        
        # Random events (viral content, outages)
        events = np.random.choice([0, 5000, -3000], periods, p=[0.95, 0.03, 0.02])
        
        # Noise
        noise = np.random.normal(0, 500, periods)
        
        traffic = base_traffic + growth + daily_pattern + weekly_pattern + monthly_pattern + events + noise
        traffic = np.maximum(traffic, 100)
        
        return pd.DataFrame({
            'timestamp': dates,
            'visitors': traffic.astype(int),
            'category': 'web',
            'site': 'main_site'
        })
    
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
                if 'category' in row:
                    point = point.tag("category", row['category'])
                if 'location' in row:
                    point = point.tag("location", row['location'])
                if 'symbol' in row:
                    point = point.tag("symbol", row['symbol'])
                if 'site' in row:
                    point = point.tag("site", row['site'])
                
                # Add fields (numeric values)
                for col in df.columns:
                    if col not in ['timestamp', 'category', 'location', 'symbol', 'site']:
                        point = point.field(col, float(row[col]))
                
                # Set timestamp
                point = point.time(row['timestamp'])
                points.append(point)
            
            self.write_api.write(
                bucket=os.getenv('INFLUXDB_BUCKET', 'timeseries'),
                record=points
            )
            logger.info(f"Stored {len(points)} points to InfluxDB measurement: {measurement}")
        except Exception as e:
            logger.error(f"Error storing to InfluxDB: {e}")
    
    def cache_latest_values(self, df: pd.DataFrame, prefix: str):
        """Cache latest values to Redis"""
        try:
            latest = df.iloc[-1]
            for col in df.columns:
                if col != 'timestamp':
                    key = f"{prefix}:latest:{col}"
                    self.redis_client.set(key, str(latest[col]))
            
            # Cache last update time
            self.redis_client.set(f"{prefix}:last_update", str(latest['timestamp']))
            logger.info(f"Cached latest values to Redis with prefix: {prefix}")
        except Exception as e:
            logger.error(f"Error caching to Redis: {e}")
    
    def generate_all_datasets(self):
        """Generate all time series datasets"""
        logger.info("Starting time series data generation...")
        
        # Generate datasets
        datasets = {
            'retail_sales': self.generate_retail_sales(),
            'energy_consumption': self.generate_energy_consumption(),
            'stock_prices': self.generate_stock_prices(),
            'website_traffic': self.generate_website_traffic()
        }
        
        # Store to all systems
        for name, df in datasets.items():
            logger.info(f"Processing dataset: {name}")
            
            # PostgreSQL
            self.store_to_postgres(df, name)
            
            # InfluxDB
            self.store_to_influxdb(df, name)
            
            # Redis cache
            self.cache_latest_values(df, name)
            
            # Save to CSV for backup
            df.to_csv(f"{name}.csv", index=False)
            logger.info(f"Saved {name} to CSV")
        
        logger.info("Data generation completed!")
    
    def close_connections(self):
        """Close all connections"""
        self.pg_conn.close()
        self.redis_client.close()
        self.influx_client.close()

def main():
    """Main execution function"""
    generator = TimeSeriesGenerator()
    
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