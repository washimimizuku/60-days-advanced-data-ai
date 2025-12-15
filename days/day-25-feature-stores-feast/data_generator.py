#!/usr/bin/env python3
"""
Sample Data Generator for RideShare Feature Store

Generates realistic sample data for:
- Driver statistics and performance metrics
- User behavior and preferences
- Ride history and patterns
- Location demand and supply data
- Vehicle telemetry and maintenance data
"""

import os
import sys
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import psycopg2
import redis
from faker import Faker
from tqdm import tqdm

fake = Faker()

class RideShareDataGenerator:
    """Generate realistic sample data for ride-sharing feature store"""
    
    def __init__(self):
        self.fake = Faker()
        
        # Configuration from environment
        self.num_drivers = int(os.getenv('SAMPLE_DRIVERS', 10000))
        self.num_users = int(os.getenv('SAMPLE_USERS', 50000))
        self.num_rides = int(os.getenv('SAMPLE_RIDES', 100000))
        self.days_of_data = int(os.getenv('DAYS_OF_DATA', 30))
        
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
                database=os.getenv('POSTGRES_DB', 'rideshare_features'),
                user=os.getenv('POSTGRES_USER', 'feast_user'),
                password=os.getenv('POSTGRES_PASSWORD', 'feast_password'),
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
    
    def generate_driver_data(self) -> pd.DataFrame:
        """Generate driver statistics and performance data"""
        
        print(f"🚗 Generating data for {self.num_drivers} drivers...")
        
        drivers = []
        
        for driver_id in tqdm(range(1, self.num_drivers + 1)):
            # Generate multiple records for time series
            for days_back in range(self.days_of_data):
                event_time = datetime.now() - timedelta(days=days_back)
                
                # Driver performance varies over time
                base_acceptance = random.uniform(0.7, 0.95)
                base_rating = random.uniform(4.2, 4.9)
                
                # Add some temporal variation
                time_factor = 1 + 0.1 * np.sin(days_back * 0.1)
                
                driver_data = {
                    'driver_id': driver_id,
                    'event_timestamp': event_time,
                    'created_timestamp': event_time + timedelta(minutes=random.randint(1, 60)),
                    
                    # Performance metrics
                    'acceptance_rate_7d': max(0.5, min(1.0, base_acceptance * time_factor)),
                    'cancellation_rate_7d': random.uniform(0.01, 0.08),
                    'avg_rating_7d': max(3.0, min(5.0, base_rating * time_factor)),
                    'total_rides_7d': random.randint(20, 150),
                    'total_earnings_7d': random.uniform(800, 2500),
                    
                    # Activity metrics
                    'avg_trip_duration_7d': random.uniform(15, 45),
                    'avg_trip_distance_7d': random.uniform(5, 25),
                    'peak_hours_active_7d': random.randint(20, 60),
                    'weekend_activity_ratio': random.uniform(0.3, 0.8),
                    'late_night_rides_7d': random.randint(0, 20),
                    
                    # Quality metrics
                    'complaint_count_30d': random.randint(0, 5),
                    'safety_score': random.uniform(0.85, 1.0)
                }
                
                drivers.append(driver_data)
        
        return pd.DataFrame(drivers)
    
    def generate_user_data(self) -> pd.DataFrame:
        """Generate user behavior and preference data"""
        
        print(f"👥 Generating data for {self.num_users} users...")
        
        users = []
        payment_methods = ['credit_card', 'debit_card', 'digital_wallet', 'cash']
        vehicle_types = ['economy', 'comfort', 'premium', 'shared']
        
        for user_id in tqdm(range(1, self.num_users + 1)):
            for days_back in range(0, self.days_of_data, 7):  # Weekly snapshots
                event_time = datetime.now() - timedelta(days=days_back)
                
                # User behavior evolves over time
                base_frequency = random.randint(5, 30)
                loyalty_trend = 1 + (self.days_of_data - days_back) * 0.01
                
                user_data = {
                    'user_id': user_id,
                    'event_timestamp': event_time,
                    'created_timestamp': event_time + timedelta(minutes=random.randint(1, 30)),
                    
                    # Trip patterns
                    'avg_trip_distance_30d': random.uniform(3, 30),
                    'ride_frequency_30d': max(1, int(base_frequency * loyalty_trend)),
                    'avg_rating_given_30d': random.uniform(4.0, 5.0),
                    'cancellation_rate_30d': random.uniform(0.02, 0.12),
                    
                    # Preferences
                    'preferred_payment_method': random.choice(payment_methods),
                    'preferred_vehicle_type': random.choice(vehicle_types),
                    'peak_usage_hours': random.choice(['morning', 'afternoon', 'evening', 'night']),
                    
                    # Financial metrics
                    'avg_tip_percentage': random.uniform(0.1, 0.25),
                    'total_spent_30d': random.uniform(100, 800),
                    'price_sensitivity_score': random.uniform(0.3, 0.9),
                    
                    # Engagement metrics
                    'loyalty_score': min(1.0, loyalty_trend * random.uniform(0.6, 0.95)),
                    'days_since_last_ride': random.randint(0, 14),
                    'business_vs_personal_ratio': random.uniform(0.2, 0.8),
                    
                    # Location preferences
                    'favorite_pickup_locations': f\"zone_{random.randint(1, 50)},zone_{random.randint(1, 50)}\"\n                }\n                \n                users.append(user_data)\n        \n        return pd.DataFrame(users)\n    \n    def generate_location_data(self) -> pd.DataFrame:\n        \"\"\"Generate location demand and supply data\"\"\"\n        \n        print(\"📍 Generating location demand data...\")\n        \n        locations = []\n        location_ids = [f\"zone_{i}\" for i in range(1, 101)]  # 100 zones\n        \n        for location_id in tqdm(location_ids):\n            for hours_back in range(0, self.days_of_data * 24, 2):  # Every 2 hours\n                event_time = datetime.now() - timedelta(hours=hours_back)\n                \n                # Demand varies by time of day and location\n                hour = event_time.hour\n                is_peak = hour in [7, 8, 9, 17, 18, 19]  # Rush hours\n                is_weekend = event_time.weekday() >= 5\n                \n                base_demand = random.uniform(0.3, 0.9)\n                if is_peak:\n                    base_demand *= 1.5\n                if is_weekend:\n                    base_demand *= 0.8\n                \n                location_data = {\n                    'location_id': location_id,\n                    'event_timestamp': event_time,\n                    'created_timestamp': event_time + timedelta(minutes=random.randint(1, 10)),\n                    \n                    # Demand metrics\n                    'current_demand_score': min(1.0, base_demand),\n                    'supply_ratio': random.uniform(0.5, 2.0),\n                    'avg_wait_time_minutes': random.uniform(2, 15),\n                    'surge_multiplier': 1.0 + (base_demand - 0.5) * 2,\n                    \n                    # Context factors\n                    'weather_impact_factor': random.uniform(0.8, 1.2),\n                    'event_impact_score': random.uniform(0.9, 1.1),\n                    'traffic_congestion_level': random.randint(1, 5),\n                    \n                    # Performance metrics\n                    'avg_trip_value': random.uniform(15, 60),\n                    'pickup_success_rate': random.uniform(0.85, 0.98),\n                    'safety_score': random.uniform(0.9, 1.0),\n                    \n                    # Categorical features\n                    'historical_demand_pattern': random.choice(['low', 'medium', 'high', 'variable']),\n                    'competitor_presence': random.choice([True, False])\n                }\n                \n                locations.append(location_data)\n        \n        return pd.DataFrame(locations)\n    \n    def generate_ride_data(self) -> pd.DataFrame:\n        \"\"\"Generate ride history data\"\"\"\n        \n        print(f\"🚕 Generating {self.num_rides} ride records...\")\n        \n        rides = []\n        \n        for ride_id in tqdm(range(1, self.num_rides + 1)):\n            event_time = datetime.now() - timedelta(\n                days=random.randint(0, self.days_of_data),\n                hours=random.randint(0, 23),\n                minutes=random.randint(0, 59)\n            )\n            \n            ride_data = {\n                'ride_id': f\"ride_{ride_id}\",\n                'driver_id': random.randint(1, self.num_drivers),\n                'user_id': random.randint(1, self.num_users),\n                'pickup_location_id': f\"zone_{random.randint(1, 100)}\",\n                'dropoff_location_id': f\"zone_{random.randint(1, 100)}\",\n                'event_timestamp': event_time,\n                'created_timestamp': event_time + timedelta(minutes=random.randint(1, 5)),\n                \n                # Trip metrics\n                'trip_distance_km': random.uniform(2, 50),\n                'trip_duration_minutes': random.uniform(10, 90),\n                'trip_fare': random.uniform(8, 120),\n                'surge_applied': random.uniform(1.0, 3.0),\n                \n                # Status\n                'ride_status': random.choice(['completed', 'cancelled_driver', 'cancelled_user']),\n                'payment_method': random.choice(['credit_card', 'debit_card', 'digital_wallet', 'cash']),\n                \n                # Ratings\n                'driver_rating': random.uniform(3.5, 5.0),\n                'user_rating': random.uniform(3.5, 5.0)\n            }\n            \n            rides.append(ride_data)\n        \n        return pd.DataFrame(rides)\n    \n    def generate_vehicle_data(self) -> pd.DataFrame:\n        \"\"\"Generate vehicle performance and telemetry data\"\"\"\n        \n        print(\"🚙 Generating vehicle telemetry data...\")\n        \n        vehicles = []\n        vehicle_ids = [f\"vehicle_{i}\" for i in range(1, self.num_drivers + 1)]\n        \n        for vehicle_id in tqdm(vehicle_ids):\n            for days_back in range(0, self.days_of_data, 3):  # Every 3 days\n                event_time = datetime.now() - timedelta(days=days_back)\n                \n                # Vehicle condition degrades over time\n                age_factor = 1 - (days_back / (self.days_of_data * 10))\n                \n                vehicle_data = {\n                    'vehicle_id': vehicle_id,\n                    'event_timestamp': event_time,\n                    'created_timestamp': event_time + timedelta(minutes=random.randint(1, 30)),\n                    \n                    # Performance metrics\n                    'fuel_efficiency_7d': random.uniform(8, 15) * age_factor,\n                    'maintenance_score': random.uniform(0.7, 1.0) * age_factor,\n                    'breakdown_risk_score': random.uniform(0.1, 0.4) * (1 - age_factor),\n                    'total_mileage': random.uniform(50000, 200000) + days_back * 100,\n                    \n                    # Driving behavior\n                    'avg_speed_7d': random.uniform(25, 45),\n                    'harsh_braking_events_7d': random.randint(0, 15),\n                    'rapid_acceleration_events_7d': random.randint(0, 10),\n                    'idle_time_percentage_7d': random.uniform(0.15, 0.35),\n                    \n                    # Maintenance\n                    'days_since_last_service': random.randint(0, 180),\n                    'predicted_service_date': (event_time + timedelta(days=random.randint(30, 120))).strftime('%Y-%m-%d')\n                }\n                \n                vehicles.append(vehicle_data)\n        \n        return pd.DataFrame(vehicles)\n    \n    def save_to_parquet(self, df: pd.DataFrame, filename: str):\n        \"\"\"Save DataFrame to Parquet format\"\"\"\n        \n        os.makedirs('data', exist_ok=True)\n        filepath = f\"data/{filename}.parquet\"\n        \n        df.to_parquet(filepath, index=False)\n        print(f\"💾 Saved {len(df)} records to {filepath}\")\n    \n    def save_to_postgres(self, df: pd.DataFrame, table_name: str):\n        \"\"\"Save DataFrame to PostgreSQL\"\"\"\n        \n        try:\n            cursor = self.postgres_conn.cursor()\n            \n            # Create table if not exists\n            columns = []\n            for col, dtype in df.dtypes.items():\n                if 'int' in str(dtype):\n                    pg_type = 'INTEGER'\n                elif 'float' in str(dtype):\n                    pg_type = 'REAL'\n                elif 'datetime' in str(dtype):\n                    pg_type = 'TIMESTAMP'\n                elif 'bool' in str(dtype):\n                    pg_type = 'BOOLEAN'\n                else:\n                    pg_type = 'TEXT'\n                \n                columns.append(f\"{col} {pg_type}\")\n            \n            create_table_sql = f\"\"\"\n            CREATE TABLE IF NOT EXISTS {table_name} (\n                {', '.join(columns)}\n            )\n            \"\"\"\n            \n            cursor.execute(create_table_sql)\n            \n            # Insert data\n            for _, row in df.iterrows():\n                placeholders = ', '.join(['%s'] * len(row))\n                insert_sql = f\"INSERT INTO {table_name} VALUES ({placeholders})\"\n                cursor.execute(insert_sql, tuple(row))\n            \n            self.postgres_conn.commit()\n            cursor.close()\n            \n            print(f\"💾 Saved {len(df)} records to PostgreSQL table {table_name}\")\n            \n        except Exception as e:\n            print(f\"❌ Failed to save to PostgreSQL: {e}\")\n            self.postgres_conn.rollback()\n    \n    def generate_all_data(self):\n        \"\"\"Generate all sample datasets\"\"\"\n        \n        print(\"🎯 Starting comprehensive data generation...\")\n        \n        # Generate datasets\n        datasets = {\n            'driver_stats': self.generate_driver_data(),\n            'user_behavior': self.generate_user_data(),\n            'location_demand': self.generate_location_data(),\n            'ride_history': self.generate_ride_data(),\n            'vehicle_telemetry': self.generate_vehicle_data()\n        }\n        \n        # Save to Parquet files\n        for name, df in datasets.items():\n            self.save_to_parquet(df, name)\n        \n        # Save to PostgreSQL\n        for name, df in datasets.items():\n            self.save_to_postgres(df, name)\n        \n        print(\"\\n✅ Data generation completed successfully!\")\n        print(f\"📊 Generated datasets:\")\n        for name, df in datasets.items():\n            print(f\"   • {name}: {len(df):,} records\")\n        \n        return datasets\n\nif __name__ == \"__main__\":\n    generator = RideShareDataGenerator()\n    datasets = generator.generate_all_data()\n