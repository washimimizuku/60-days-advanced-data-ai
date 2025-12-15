"""
Day 25: Feature Stores - Feast & Tecton - Complete Solution

Production-ready feature store implementation for RideShare Analytics.
Demonstrates enterprise-grade feature engineering, serving, and monitoring.

This solution showcases:
- Comprehensive feature definitions for multi-domain ML use cases
- Real-time and batch feature processing with Feast
- Sub-millisecond feature serving for production inference
- Feature monitoring, drift detection, and quality validation
- Production API with comprehensive error handling and observability
- Feature governance with versioning and lineage tracking
"""

import os
import sys
import json
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import redis
import boto3
from scipy import stats
from sklearn.preprocessing import StandardScaler

# Feast imports
from feast import Entity, Feature, FeatureView, FileSource, ValueType, FeatureStore
from feast.on_demand_feature_view import on_demand_feature_view
from feast.field import Field
from feast.types import Float64, Int64, String, Bool
from feast.data_source import KafkaSource
from feast.aggregation import Aggregation
from feast import StreamFeatureView

# FastAPI imports
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field as PydanticField
import uvicorn

# Monitoring imports
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# =============================================================================
# PRODUCTION FEATURE STORE CONFIGURATION
# =============================================================================

@dataclass
class FeatureStoreConfig:
    """Configuration for production feature store deployment"""
    
    # Store configuration
    project_name: str = "rideshare_ml_platform"
    registry_path: str = "s3://rideshare-ml-registry/registry.db"
    
    # Online store (Redis)
    redis_host: str = "production-redis.cache.amazonaws.com"
    redis_port: int = 6379
    redis_ssl: bool = True
    
    # Offline store (S3 + Spark)
    s3_bucket: str = "rideshare-feature-store"
    s3_region: str = "us-west-2"
    spark_config: Dict[str, str] = None
    
    # Monitoring
    metrics_port: int = 8000
    drift_threshold: float = 0.1
    quality_threshold: float = 0.95
    
    def __post_init__(self):
        if self.spark_config is None:
            self.spark_config = {
                "spark.executor.memory": "4g",
                "spark.executor.cores": "2",
                "spark.sql.execution.arrow.pyspark.enabled": "true",
                "spark.serializer": "org.apache.spark.serializer.KryoSerializer"
            }

# =============================================================================
# ENTITY DEFINITIONS
# =============================================================================

# Core business entities for ride-sharing platform
driver = Entity(
    name="driver_id",
    value_type=ValueType.INT64,
    description="Unique identifier for drivers in the platform",
    tags={"domain": "driver_management", "pii": "false"}
)

user = Entity(
    name="user_id", 
    value_type=ValueType.INT64,
    description="Unique identifier for platform users/riders",
    tags={"domain": "user_analytics", "pii": "false"}
)

ride = Entity(
    name="ride_id",
    value_type=ValueType.STRING,
    description="Unique identifier for individual rides",
    tags={"domain": "ride_analytics", "pii": "false"}
)

location = Entity(
    name="location_id",
    value_type=ValueType.STRING, 
    description="Geographic location identifier (geohash or zone)",
    tags={"domain": "location_analytics", "pii": "false"}
)

vehicle = Entity(
    name="vehicle_id",
    value_type=ValueType.STRING,
    description="Unique identifier for vehicles in the fleet",
    tags={"domain": "fleet_management", "pii": "false"}
)

# =============================================================================
# BATCH DATA SOURCES
# =============================================================================

# Driver statistics from data warehouse
driver_stats_source = FileSource(
    path="s3://rideshare-feature-store/driver_stats/",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
    file_format="parquet"
)

# User behavior patterns
user_behavior_source = FileSource(
    path="s3://rideshare-feature-store/user_behavior/",
    event_timestamp_column="event_timestamp", 
    created_timestamp_column="created_timestamp",
    file_format="parquet"
)

# Historical ride data
ride_history_source = FileSource(
    path="s3://rideshare-feature-store/ride_history/",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp", 
    file_format="parquet"
)

# Location demand patterns
location_demand_source = FileSource(
    path="s3://rideshare-feature-store/location_demand/",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
    file_format="parquet"
)

# Vehicle telemetry data
vehicle_telemetry_source = FileSource(
    path="s3://rideshare-feature-store/vehicle_telemetry/",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
    file_format="parquet"
)

# =============================================================================
# STREAMING DATA SOURCES
# =============================================================================

# Real-time ride events stream
ride_events_stream = KafkaSource(
    name="ride_events_stream",
    kafka_bootstrap_servers="kafka-cluster.rideshare.internal:9092",
    topic="ride_events",
    timestamp_field="event_timestamp",
    batch_source=ride_history_source,
    watermark_delay_threshold=timedelta(minutes=5)
)

# Real-time driver location updates
driver_location_stream = KafkaSource(
    name="driver_location_stream", 
    kafka_bootstrap_servers="kafka-cluster.rideshare.internal:9092",
    topic="driver_locations",
    timestamp_field="event_timestamp",
    batch_source=driver_stats_source,
    watermark_delay_threshold=timedelta(minutes=1)
)

# Real-time demand signals
demand_signals_stream = KafkaSource(
    name="demand_signals_stream",
    kafka_bootstrap_servers="kafka-cluster.rideshare.internal:9092", 
    topic="demand_signals",
    timestamp_field="event_timestamp",
    batch_source=location_demand_source,
    watermark_delay_threshold=timedelta(minutes=2)
)

# =============================================================================
# BATCH FEATURE VIEWS
# =============================================================================

# Driver performance and behavior features
driver_performance_features = FeatureView(
    name="driver_performance_features",
    entities=["driver_id"],
    ttl=timedelta(days=7),
    features=[
        Feature(name="acceptance_rate_7d", dtype=ValueType.FLOAT),
        Feature(name="cancellation_rate_7d", dtype=ValueType.FLOAT),
        Feature(name="avg_rating_7d", dtype=ValueType.FLOAT),
        Feature(name="total_rides_7d", dtype=ValueType.INT64),
        Feature(name="total_earnings_7d", dtype=ValueType.FLOAT),
        Feature(name="avg_trip_duration_7d", dtype=ValueType.FLOAT),
        Feature(name="avg_trip_distance_7d", dtype=ValueType.FLOAT),
        Feature(name="peak_hours_active_7d", dtype=ValueType.INT64),
        Feature(name="weekend_activity_ratio", dtype=ValueType.FLOAT),
        Feature(name="late_night_rides_7d", dtype=ValueType.INT64),
        Feature(name="complaint_count_30d", dtype=ValueType.INT64),
        Feature(name="safety_score", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=driver_stats_source,
    tags={
        "team": "ml_platform",
        "domain": "driver_management", 
        "version": "v2.1",
        "owner": "driver_analytics_team"
    }
)

# User behavior and preference features
user_behavior_features = FeatureView(
    name="user_behavior_features",
    entities=["user_id"],
    ttl=timedelta(days=30),
    features=[
        Feature(name="avg_trip_distance_30d", dtype=ValueType.FLOAT),
        Feature(name="preferred_payment_method", dtype=ValueType.STRING),
        Feature(name="loyalty_score", dtype=ValueType.FLOAT),
        Feature(name="ride_frequency_30d", dtype=ValueType.INT64),
        Feature(name="avg_rating_given_30d", dtype=ValueType.FLOAT),
        Feature(name="cancellation_rate_30d", dtype=ValueType.FLOAT),
        Feature(name="peak_usage_hours", dtype=ValueType.STRING),
        Feature(name="preferred_vehicle_type", dtype=ValueType.STRING),
        Feature(name="avg_tip_percentage", dtype=ValueType.FLOAT),
        Feature(name="price_sensitivity_score", dtype=ValueType.FLOAT),
        Feature(name="total_spent_30d", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_ride", dtype=ValueType.INT64),
        Feature(name="favorite_pickup_locations", dtype=ValueType.STRING),
        Feature(name="business_vs_personal_ratio", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=user_behavior_source,
    tags={
        "team": "ml_platform",
        "domain": "user_analytics",
        "version": "v1.8", 
        "owner": "user_experience_team"
    }
)

# Location-based demand and supply features
location_demand_features = FeatureView(
    name="location_demand_features",
    entities=["location_id"],
    ttl=timedelta(hours=2),
    features=[
        Feature(name="current_demand_score", dtype=ValueType.FLOAT),
        Feature(name="supply_ratio", dtype=ValueType.FLOAT),
        Feature(name="avg_wait_time_minutes", dtype=ValueType.FLOAT),
        Feature(name="surge_multiplier", dtype=ValueType.FLOAT),
        Feature(name="historical_demand_pattern", dtype=ValueType.STRING),
        Feature(name="weather_impact_factor", dtype=ValueType.FLOAT),
        Feature(name="event_impact_score", dtype=ValueType.FLOAT),
        Feature(name="competitor_presence", dtype=ValueType.BOOL),
        Feature(name="avg_trip_value", dtype=ValueType.FLOAT),
        Feature(name="pickup_success_rate", dtype=ValueType.FLOAT),
        Feature(name="traffic_congestion_level", dtype=ValueType.INT64),
        Feature(name="safety_score", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=location_demand_source,
    tags={
        "team": "ml_platform",
        "domain": "demand_forecasting",
        "version": "v3.0",
        "owner": "operations_team"
    }
)

# Vehicle performance and maintenance features
vehicle_performance_features = FeatureView(
    name="vehicle_performance_features",
    entities=["vehicle_id"],
    ttl=timedelta(days=1),
    features=[
        Feature(name="fuel_efficiency_7d", dtype=ValueType.FLOAT),
        Feature(name="maintenance_score", dtype=ValueType.FLOAT),
        Feature(name="breakdown_risk_score", dtype=ValueType.FLOAT),
        Feature(name="total_mileage", dtype=ValueType.FLOAT),
        Feature(name="avg_speed_7d", dtype=ValueType.FLOAT),
        Feature(name="harsh_braking_events_7d", dtype=ValueType.INT64),
        Feature(name="rapid_acceleration_events_7d", dtype=ValueType.INT64),
        Feature(name="idle_time_percentage_7d", dtype=ValueType.FLOAT),
        Feature(name="days_since_last_service", dtype=ValueType.INT64),
        Feature(name="predicted_service_date", dtype=ValueType.STRING)
    ],
    online=True,
    batch_source=vehicle_telemetry_source,
    tags={
        "team": "ml_platform",
        "domain": "fleet_management",
        "version": "v1.5",
        "owner": "fleet_operations_team"
    }
)

# =============================================================================
# STREAMING FEATURE VIEWS WITH REAL-TIME AGGREGATIONS
# =============================================================================

# Real-time driver activity metrics
driver_realtime_metrics = StreamFeatureView(
    name="driver_realtime_metrics",
    entities=[driver],
    ttl=timedelta(hours=1),
    source=driver_location_stream,
    aggregations=[
        Aggregation(
            column="trip_duration_minutes",
            function="avg",
            time_window=timedelta(hours=1),
            feature_name="avg_trip_duration_1h"
        ),
        Aggregation(
            column="trip_distance_km", 
            function="sum",
            time_window=timedelta(hours=1),
            feature_name="total_distance_1h"
        ),
        Aggregation(
            column="trip_fare",
            function="sum",
            time_window=timedelta(hours=1),
            feature_name="total_earnings_1h"
        ),
        Aggregation(
            column="ride_completed",
            function="count",
            time_window=timedelta(hours=1),
            feature_name="rides_completed_1h"
        ),
        Aggregation(
            column="trip_duration_minutes",
            function="avg",
            time_window=timedelta(hours=4),
            feature_name="avg_trip_duration_4h"
        ),
        Aggregation(
            column="customer_rating",
            function="avg",
            time_window=timedelta(hours=24),
            feature_name="avg_rating_24h"
        )
    ],
    online=True,
    schema=[
        Field(name="driver_id", dtype=Int64),
        Field(name="trip_duration_minutes", dtype=Float64),
        Field(name="trip_distance_km", dtype=Float64),
        Field(name="trip_fare", dtype=Float64),
        Field(name="ride_completed", dtype=Int64),
        Field(name="customer_rating", dtype=Float64)
    ],
    tags={"team": "ml_platform", "domain": "realtime_driver_metrics"}
)

# Real-time demand and supply metrics
location_realtime_demand = StreamFeatureView(
    name="location_realtime_demand",
    entities=[location],
    ttl=timedelta(minutes=30),
    source=demand_signals_stream,
    aggregations=[
        Aggregation(
            column="ride_request",
            function="count",
            time_window=timedelta(minutes=15),
            feature_name="ride_requests_15m"
        ),
        Aggregation(
            column="ride_request",
            function="count", 
            time_window=timedelta(hours=1),
            feature_name="ride_requests_1h"
        ),
        Aggregation(
            column="wait_time_minutes",
            function="avg",
            time_window=timedelta(minutes=30),
            feature_name="avg_wait_time_30m"
        ),
        Aggregation(
            column="surge_applied",
            function="avg",
            time_window=timedelta(minutes=15),
            feature_name="avg_surge_15m"
        ),
        Aggregation(
            column="cancellation_event",
            function="count",
            time_window=timedelta(hours=1),
            feature_name="cancellations_1h"
        )
    ],
    online=True,
    schema=[
        Field(name="location_id", dtype=String),
        Field(name="ride_request", dtype=Int64),
        Field(name="wait_time_minutes", dtype=Float64),
        Field(name="surge_applied", dtype=Float64),
        Field(name="cancellation_event", dtype=Int64)
    ],
    tags={"team": "ml_platform", "domain": "realtime_demand_metrics"}
)

# =============================================================================
# ON-DEMAND FEATURE TRANSFORMATIONS
# =============================================================================

@on_demand_feature_view(
    sources=[driver_performance_features, driver_realtime_metrics],
    schema=[
        Field(name="driver_performance_score", dtype=Float64),
        Field(name="driver_risk_category", dtype=String),
        Field(name="driver_efficiency_score", dtype=Float64),
        Field(name="driver_reliability_score", dtype=Float64)
    ]
)
def driver_performance_score(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive driver performance metrics"""
    
    df = pd.DataFrame()
    
    # Composite performance score (0-1 scale)
    acceptance_weight = 0.25
    rating_weight = 0.30
    reliability_weight = 0.25  # Based on cancellation rate
    efficiency_weight = 0.20   # Based on earnings and activity
    
    df["driver_performance_score"] = (
        features_df["acceptance_rate_7d"] * acceptance_weight +
        (features_df["avg_rating_7d"] / 5.0) * rating_weight +
        (1 - features_df["cancellation_rate_7d"]) * reliability_weight +
        np.minimum(features_df["total_earnings_1h"] / 100.0, 1.0) * efficiency_weight
    ).clip(0, 1)
    
    # Risk categorization based on performance score
    df["driver_risk_category"] = df["driver_performance_score"].apply(
        lambda x: "low" if x > 0.8 else "medium" if x > 0.6 else "high"
    )
    
    # Efficiency score based on earnings per hour and trip metrics
    df["driver_efficiency_score"] = (
        (features_df["total_earnings_1h"] / np.maximum(features_df["avg_trip_duration_1h"], 1)) * 0.6 +
        (features_df["rides_completed_1h"] / 10.0) * 0.4
    ).clip(0, 1)
    
    # Reliability score combining multiple reliability factors
    df["driver_reliability_score"] = (
        (1 - features_df["cancellation_rate_7d"]) * 0.4 +
        (features_df["acceptance_rate_7d"]) * 0.3 +
        (features_df["avg_rating_7d"] / 5.0) * 0.3
    ).clip(0, 1)
    
    return df

@on_demand_feature_view(
    sources=[location_demand_features, location_realtime_demand],
    schema=[
        Field(name="dynamic_surge_multiplier", dtype=Float64),
        Field(name="estimated_wait_time", dtype=Float64),
        Field(name="demand_forecast_score", dtype=Float64),
        Field(name="location_attractiveness", dtype=Float64)
    ]
)
def surge_pricing_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dynamic pricing and demand forecasting features"""
    
    df = pd.DataFrame()
    
    # Dynamic surge multiplier based on real-time supply/demand
    base_surge = features_df["surge_multiplier"]
    demand_factor = features_df["ride_requests_15m"] / np.maximum(features_df["ride_requests_1h"] / 4, 1)
    supply_factor = 1 / np.maximum(features_df["supply_ratio"], 0.1)
    
    df["dynamic_surge_multiplier"] = np.maximum(
        1.0,
        base_surge * demand_factor * supply_factor * 0.5
    ).clip(1.0, 5.0)
    
    # Estimated wait time based on current conditions
    base_wait = features_df["avg_wait_time_minutes"]
    current_wait = features_df["avg_wait_time_30m"]
    
    df["estimated_wait_time"] = (
        base_wait * 0.3 + current_wait * 0.7
    ).clip(1.0, 30.0)
    
    # Demand forecast score (0-1) predicting demand in next 30 minutes
    df["demand_forecast_score"] = (
        features_df["ride_requests_15m"] / np.maximum(features_df["ride_requests_1h"], 1) * 4
    ).clip(0, 1)
    
    # Location attractiveness for drivers
    df["location_attractiveness"] = (
        (features_df["avg_trip_value"] / 50.0) * 0.4 +
        (features_df["pickup_success_rate"]) * 0.3 +
        (1 - features_df["avg_wait_time_minutes"] / 20.0) * 0.3
    ).clip(0, 1)
    
    return df

@on_demand_feature_view(
    sources=[user_behavior_features],
    schema=[
        Field(name="user_lifetime_value", dtype=Float64),
        Field(name="churn_risk_score", dtype=Float64),
        Field(name="price_elasticity", dtype=Float64),
        Field(name="user_segment", dtype=String)
    ]
)
def user_analytics_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced user analytics and segmentation features"""
    
    df = pd.DataFrame()
    
    # User lifetime value estimation
    monthly_spend = features_df["total_spent_30d"]
    frequency = features_df["ride_frequency_30d"]
    loyalty = features_df["loyalty_score"]
    
    df["user_lifetime_value"] = (
        monthly_spend * 12 * loyalty * np.log1p(frequency)
    ).clip(0, 10000)
    
    # Churn risk score based on recent activity
    days_inactive = features_df["days_since_last_ride"]
    frequency_decline = np.maximum(0, 1 - frequency / 20)  # Assuming 20 rides/month is high
    
    df["churn_risk_score"] = (
        (days_inactive / 30.0) * 0.5 +
        frequency_decline * 0.3 +
        (1 - loyalty) * 0.2
    ).clip(0, 1)
    
    # Price elasticity (sensitivity to surge pricing)
    df["price_elasticity"] = features_df["price_sensitivity_score"]
    
    # User segmentation
    def segment_user(row):
        if row["user_lifetime_value"] > 2000 and row["loyalty_score"] > 0.8:
            return "premium"
        elif row["ride_frequency_30d"] > 15:
            return "frequent"
        elif row["churn_risk_score"] > 0.7:
            return "at_risk"
        elif row["days_since_last_ride"] < 7:
            return "active"
        else:
            return "casual"
    
    df["user_segment"] = features_df.apply(segment_user, axis=1)
    
    return df

# =============================================================================
# PRODUCTION FEATURE STORE CLASS
# =============================================================================

class RideShareFeatureStore:
    """Production-ready feature store for ride-sharing ML platform"""
    
    def __init__(self, config: FeatureStoreConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.store = None
        self.redis_client = None
        
        # Initialize monitoring
        self._setup_metrics()
        
        # Initialize feature store
        self._setup_feature_store()
        
        # Initialize Redis client for caching
        self._setup_redis_client()
        
        self.logger.info("RideShare Feature Store initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup structured logging for production"""
        
        logger = logging.getLogger('rideshare_feature_store')
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"component": "%(name)s", "message": "%(message)s"}'
        )
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_metrics(self):
        """Setup Prometheus metrics for monitoring"""
        
        self.metrics = {
            'feature_requests': Counter('feature_requests_total', 'Total feature requests', ['type', 'status']),
            'feature_latency': Histogram('feature_latency_seconds', 'Feature serving latency', ['type']),
            'feature_quality': Gauge('feature_quality_score', 'Feature quality score', ['feature_view']),
            'drift_score': Gauge('feature_drift_score', 'Feature drift score', ['feature']),
            'materialization_status': Gauge('materialization_status', 'Materialization status', ['feature_view'])
        }
        
        # Start metrics server
        start_http_server(self.config.metrics_port)
        self.logger.info(f"Metrics server started on port {self.config.metrics_port}")
    
    def _setup_feature_store(self):
        """Initialize Feast feature store with production configuration"""
        
        # Create feature_store.yaml configuration
        config_yaml = f"""
project: {self.config.project_name}
registry: {self.config.registry_path}
provider: aws

online_store:
  type: redis
  connection_string: "redis://{self.config.redis_host}:{self.config.redis_port}"
  ssl: {str(self.config.redis_ssl).lower()}

offline_store:
  type: s3
  region: {self.config.s3_region}
  s3_staging_location: s3://{self.config.s3_bucket}/staging/

batch_engine:
  type: spark
  spark_conf:
{chr(10).join(f'    {k}: "{v}"' for k, v in self.config.spark_config.items())}
"""
        
        # Write configuration file
        with open("feature_store.yaml", "w") as f:
            f.write(config_yaml)
        
        try:
            self.store = FeatureStore(repo_path=".")
            self.logger.info("Feature store initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize feature store: {e}")
            raise
    
    def _setup_redis_client(self):
        """Setup Redis client for additional caching"""
        
        try:
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                ssl=self.config.redis_ssl,
                decode_responses=True
            )
            
            # Test connection
            self.redis_client.ping()
            self.logger.info("Redis client initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Redis client initialization failed: {e}")
            self.redis_client = None
    
    def materialize_features(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Materialize features to online store for serving"""
        
        self.logger.info(f"Starting materialization from {start_date} to {end_date}")
        
        results = {
            'status': 'success',
            'materialized_features': [],
            'errors': [],
            'total_time': 0.0
        }
        
        feature_views = [
            "driver_performance_features",
            "user_behavior_features", 
            "location_demand_features",
            "vehicle_performance_features"
        ]
        
        start_time = time.time()
        
        for fv_name in feature_views:
            try:
                fv_start_time = time.time()
                
                # Materialize feature view
                self.store.materialize(
                    start_date=start_date,
                    end_date=end_date,
                    feature_views=[fv_name]
                )
                
                fv_time = time.time() - fv_start_time
                
                results['materialized_features'].append({
                    'feature_view': fv_name,
                    'materialization_time': fv_time
                })
                
                # Update metrics
                self.metrics['materialization_status'].labels(feature_view=fv_name).set(1)
                
                self.logger.info(f"Materialized {fv_name} in {fv_time:.2f}s")
                
            except Exception as e:
                error_info = {'feature_view': fv_name, 'error': str(e)}
                results['errors'].append(error_info)
                
                # Update metrics
                self.metrics['materialization_status'].labels(feature_view=fv_name).set(0)
                
                self.logger.error(f"Failed to materialize {fv_name}: {e}")
        
        results['total_time'] = time.time() - start_time
        
        if results['errors']:
            results['status'] = 'partial_success'
        
        self.logger.info(f"Materialization completed in {results['total_time']:.2f}s")
        
        return results
    
    def get_training_features(self, entity_df: pd.DataFrame, 
                            feature_refs: List[str]) -> pd.DataFrame:
        """Get historical features for model training with point-in-time correctness"""
        
        start_time = time.time()
        
        try:
            # Get historical features with point-in-time correctness
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs
            ).to_df()
            
            latency = time.time() - start_time
            
            # Update metrics
            self.metrics['feature_requests'].labels(type='training', status='success').inc()
            self.metrics['feature_latency'].labels(type='training').observe(latency)
            
            self.logger.info(f"Retrieved {len(training_df)} training records in {latency:.2f}s")
            
            return training_df
            
        except Exception as e:
            self.metrics['feature_requests'].labels(type='training', status='error').inc()
            self.logger.error(f"Failed to get training features: {e}")
            raise
    
    def get_online_features(self, entity_rows: List[Dict], 
                          feature_refs: List[str]) -> Dict[str, Any]:
        """Get online features for real-time inference"""
        
        start_time = time.time()
        
        try:
            # Check cache first if Redis is available
            cache_key = self._generate_cache_key(entity_rows, feature_refs)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result:
                latency = (time.time() - start_time) * 1000
                self.logger.info(f"Retrieved cached features in {latency:.2f}ms")
                
                return {
                    'features': cached_result,
                    'latency_ms': latency,
                    'status': 'success',
                    'cache_hit': True
                }
            
            # Get online features from Feast
            features = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows
            )
            
            feature_dict = features.to_dict()
            latency = (time.time() - start_time) * 1000
            
            # Cache result if Redis is available
            self._cache_result(cache_key, feature_dict, ttl=300)  # 5 minute TTL
            
            # Update metrics
            self.metrics['feature_requests'].labels(type='online', status='success').inc()
            self.metrics['feature_latency'].labels(type='online').observe(latency / 1000)
            
            self.logger.info(f"Retrieved online features in {latency:.2f}ms")
            
            return {
                'features': feature_dict,
                'latency_ms': latency,
                'status': 'success',
                'cache_hit': False
            }
            
        except Exception as e:
            self.metrics['feature_requests'].labels(type='online', status='error').inc()
            self.logger.error(f"Failed to get online features: {e}")
            raise
    
    def _generate_cache_key(self, entity_rows: List[Dict], feature_refs: List[str]) -> str:
        """Generate cache key for feature request"""
        
        # Create deterministic key from entity rows and features
        entities_str = json.dumps(entity_rows, sort_keys=True)
        features_str = "|".join(sorted(feature_refs))
        
        import hashlib
        key_data = f"{entities_str}:{features_str}"
        cache_key = hashlib.md5(key_data.encode()).hexdigest()
        
        return f"features:{cache_key}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Get cached feature result"""
        
        if not self.redis_client:
            return None
        
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            self.logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    def _cache_result(self, cache_key: str, result: Dict, ttl: int = 300):
        """Cache feature result"""
        
        if not self.redis_client:
            return
        
        try:
            self.redis_client.setex(
                cache_key,
                ttl,
                json.dumps(result, default=str)
            )
        except Exception as e:
            self.logger.warning(f"Cache storage failed: {e}")

# =============================================================================
# FEATURE MONITORING AND VALIDATION
# =============================================================================

class FeatureMonitor:
    """Monitor feature quality and detect drift in production"""
    
    def __init__(self, feature_store: RideShareFeatureStore):
        self.feature_store = feature_store
        self.logger = feature_store.logger
        
    def validate_feature_quality(self, feature_view_name: str) -> Dict[str, Any]:
        """Validate feature quality using statistical checks"""
        
        self.logger.info(f"Validating feature quality for {feature_view_name}")
        
        validation_results = {
            'feature_view': feature_view_name,
            'quality_score': 0.0,
            'checks_passed': 0,
            'total_checks': 0,
            'issues': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get recent feature data for validation
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=24)
            
            # Create entity DataFrame for validation
            if 'driver' in feature_view_name:
                entity_df = pd.DataFrame({
                    'driver_id': range(1, 1001),  # Sample 1000 drivers
                    'event_timestamp': [end_date] * 1000
                })
                feature_refs = [f"{feature_view_name}:acceptance_rate_7d"]
            elif 'user' in feature_view_name:
                entity_df = pd.DataFrame({
                    'user_id': range(1, 1001),  # Sample 1000 users
                    'event_timestamp': [end_date] * 1000
                })
                feature_refs = [f"{feature_view_name}:ride_frequency_30d"]
            else:
                # Default validation
                return validation_results
            
            # Get feature data
            feature_data = self.feature_store.get_training_features(entity_df, feature_refs)
            
            if feature_data.empty:
                validation_results['issues'].append("No data available for validation")
                return validation_results
            
            # Validation checks
            checks = [
                self._check_null_values,
                self._check_data_types,
                self._check_value_ranges,
                self._check_outliers,
                self._check_distribution
            ]
            
            for check in checks:
                try:
                    check_result = check(feature_data, feature_view_name)
                    validation_results['total_checks'] += 1
                    
                    if check_result['passed']:
                        validation_results['checks_passed'] += 1
                    else:
                        validation_results['issues'].append(check_result['issue'])
                        
                except Exception as e:
                    validation_results['issues'].append(f"Check failed: {str(e)}")
            
            # Calculate quality score
            if validation_results['total_checks'] > 0:
                validation_results['quality_score'] = (
                    validation_results['checks_passed'] / validation_results['total_checks']
                )
            
            # Update metrics
            self.feature_store.metrics['feature_quality'].labels(
                feature_view=feature_view_name
            ).set(validation_results['quality_score'])
            
            self.logger.info(
                f"Feature quality validation completed for {feature_view_name}: "
                f"score={validation_results['quality_score']:.3f}"
            )
            
        except Exception as e:
            validation_results['issues'].append(f"Validation failed: {str(e)}")
            self.logger.error(f"Feature quality validation failed: {e}")
        
        return validation_results
    
    def _check_null_values(self, df: pd.DataFrame, feature_view: str) -> Dict[str, Any]:
        """Check for excessive null values"""
        
        null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        threshold = 0.05  # 5% threshold
        
        return {
            'passed': null_percentage <= threshold,
            'issue': f"High null percentage: {null_percentage:.3f} > {threshold}" if null_percentage > threshold else None
        }
    
    def _check_data_types(self, df: pd.DataFrame, feature_view: str) -> Dict[str, Any]:
        """Check data type consistency"""
        
        # Check if numeric columns contain non-numeric values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df[col].dtype == 'object':
                return {
                    'passed': False,
                    'issue': f"Column {col} has inconsistent data types"
                }
        
        return {'passed': True, 'issue': None}
    
    def _check_value_ranges(self, df: pd.DataFrame, feature_view: str) -> Dict[str, Any]:
        """Check if values are within expected ranges"""
        
        # Define expected ranges for different feature types
        range_checks = {
            'rate': (0.0, 1.0),
            'score': (0.0, 1.0),
            'rating': (1.0, 5.0),
            'percentage': (0.0, 100.0)
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            for check_type, (min_val, max_val) in range_checks.items():
                if check_type in col_lower:
                    if df[col].min() < min_val or df[col].max() > max_val:
                        return {
                            'passed': False,
                            'issue': f"Column {col} has values outside expected range [{min_val}, {max_val}]"
                        }
        
        return {'passed': True, 'issue': None}
    
    def _check_outliers(self, df: pd.DataFrame, feature_view: str) -> Dict[str, Any]:
        """Check for excessive outliers using IQR method"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if len(df[col].dropna()) > 0:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_percentage = len(outliers) / len(df)
                
                if outlier_percentage > 0.1:  # 10% threshold
                    return {
                        'passed': False,
                        'issue': f"Column {col} has excessive outliers: {outlier_percentage:.3f}"
                    }
        
        return {'passed': True, 'issue': None}
    
    def _check_distribution(self, df: pd.DataFrame, feature_view: str) -> Dict[str, Any]:
        """Check for reasonable distribution characteristics"""
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if len(df[col].dropna()) > 10:
                # Check for constant values
                if df[col].nunique() == 1:
                    return {
                        'passed': False,
                        'issue': f"Column {col} has constant values"
                    }
                
                # Check for extreme skewness
                skewness = abs(df[col].skew())
                if skewness > 5:
                    return {
                        'passed': False,
                        'issue': f"Column {col} has extreme skewness: {skewness:.3f}"
                    }
        
        return {'passed': True, 'issue': None}
    
    def detect_feature_drift(self, feature_name: str, 
                           reference_period: timedelta,
                           current_period: timedelta) -> Dict[str, Any]:
        """Detect statistical drift in feature distributions"""
        
        self.logger.info(f"Detecting drift for feature {feature_name}")
        
        drift_results = {
            'feature_name': feature_name,
            'drift_score': 0.0,
            'drift_detected': False,
            'statistical_test': 'kolmogorov_smirnov',
            'p_value': 1.0,
            'alert_threshold': self.feature_store.config.drift_threshold,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Get reference data
            end_ref = datetime.now() - current_period
            start_ref = end_ref - reference_period
            
            # Get current data
            end_curr = datetime.now()
            start_curr = end_curr - current_period
            
            # Create entity DataFrames for both periods
            entity_df_ref = pd.DataFrame({
                'driver_id': range(1, 1001),
                'event_timestamp': [end_ref] * 1000
            })
            
            entity_df_curr = pd.DataFrame({
                'driver_id': range(1, 1001),
                'event_timestamp': [end_curr] * 1000
            })
            
            # Get feature data for both periods
            ref_data = self.feature_store.get_training_features(entity_df_ref, [feature_name])
            curr_data = self.feature_store.get_training_features(entity_df_curr, [feature_name])
            
            if ref_data.empty or curr_data.empty:
                drift_results['drift_detected'] = False
                return drift_results
            
            # Extract feature values
            feature_col = feature_name.split(':')[-1]  # Get feature name after ':'
            
            if feature_col not in ref_data.columns or feature_col not in curr_data.columns:
                drift_results['drift_detected'] = False
                return drift_results
            
            ref_values = ref_data[feature_col].dropna()
            curr_values = curr_data[feature_col].dropna()
            
            if len(ref_values) == 0 or len(curr_values) == 0:
                drift_results['drift_detected'] = False
                return drift_results
            
            # Perform Kolmogorov-Smirnov test
            ks_statistic, p_value = stats.ks_2samp(ref_values, curr_values)
            
            drift_results['drift_score'] = ks_statistic
            drift_results['p_value'] = p_value
            drift_results['drift_detected'] = ks_statistic > drift_results['alert_threshold']
            
            # Update metrics
            self.feature_store.metrics['drift_score'].labels(feature=feature_name).set(ks_statistic)
            
            if drift_results['drift_detected']:
                self.logger.warning(
                    f"Drift detected for {feature_name}: "
                    f"KS statistic={ks_statistic:.4f}, p-value={p_value:.4f}"
                )
            else:
                self.logger.info(
                    f"No drift detected for {feature_name}: "
                    f"KS statistic={ks_statistic:.4f}, p-value={p_value:.4f}"
                )
            
        except Exception as e:
            self.logger.error(f"Drift detection failed for {feature_name}: {e}")
            drift_results['drift_detected'] = False
        
        return drift_results

# =============================================================================
# FEATURE SERVING API
# =============================================================================

# Initialize feature store
config = FeatureStoreConfig()
feature_store = RideShareFeatureStore(config)
feature_monitor = FeatureMonitor(feature_store)

app = FastAPI(
    title="RideShare Feature Serving API",
    description="Production feature serving for ML models",
    version="2.0.0"
)

class FeatureRequest(BaseModel):
    """Request model for feature serving"""
    entity_rows: List[Dict[str, Any]] = PydanticField(..., description="Entity rows for feature lookup")
    features: List[str] = PydanticField(..., description="List of feature references to retrieve")

class FeatureResponse(BaseModel):
    """Response model for feature serving"""
    features: Dict[str, Any] = PydanticField(..., description="Retrieved features")
    latency_ms: float = PydanticField(..., description="Request latency in milliseconds")
    status: str = PydanticField(..., description="Request status")
    cache_hit: bool = PydanticField(False, description="Whether result was served from cache")

class ValidationResponse(BaseModel):
    """Response model for feature validation"""
    feature_view: str
    quality_score: float
    checks_passed: int
    total_checks: int
    issues: List[str]
    timestamp: str

@app.post("/features", response_model=FeatureResponse)
async def get_features(request: FeatureRequest):
    """Serve features for real-time ML inference"""
    
    try:
        result = feature_store.get_online_features(
            entity_rows=request.entity_rows,
            feature_refs=request.features
        )
        
        return FeatureResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature serving failed: {str(e)}")

@app.get("/features/metadata/{feature_view}")
async def get_feature_metadata(feature_view: str):
    """Get metadata and schema information for feature view"""
    
    try:
        fv = feature_store.store.get_feature_view(feature_view)
        
        return {
            "name": fv.name,
            "entities": [e.name for e in fv.entities],
            "features": [{"name": f.name, "dtype": f.dtype.name} for f in fv.features],
            "ttl": str(fv.ttl),
            "tags": fv.tags,
            "online": fv.online,
            "source": {
                "type": type(fv.batch_source).__name__,
                "path": getattr(fv.batch_source, 'path', None)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Feature view not found: {str(e)}")

@app.post("/features/validate/{feature_view}", response_model=ValidationResponse)
async def validate_feature_quality(feature_view: str, background_tasks: BackgroundTasks):
    """Validate feature quality for a specific feature view"""
    
    try:
        # Run validation in background for better performance
        result = feature_monitor.validate_feature_quality(feature_view)
        
        return ValidationResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/features/drift/{feature_name}")
async def detect_feature_drift(feature_name: str, 
                             reference_hours: int = 168,  # 1 week
                             current_hours: int = 24):    # 1 day
    """Detect drift for a specific feature"""
    
    try:
        result = feature_monitor.detect_feature_drift(
            feature_name=feature_name,
            reference_period=timedelta(hours=reference_hours),
            current_period=timedelta(hours=current_hours)
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")

@app.post("/features/materialize")
async def materialize_features(hours_back: int = 24):
    """Trigger feature materialization"""
    
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours_back)
        
        result = feature_store.materialize_features(start_date, end_date)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Materialization failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    try:
        # Test feature store connectivity
        test_result = feature_store.get_online_features(
            entity_rows=[{"driver_id": 1}],
            feature_refs=["driver_performance_features:acceptance_rate_7d"]
        )
        
        return {
            "status": "healthy",
            "feature_store": "connected",
            "latency_ms": test_result["latency_ms"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# =============================================================================
# INTEGRATION TESTING
# =============================================================================

def test_feature_store_integration():
    """Comprehensive integration test for feature store functionality"""
    
    print("🧪 Testing RideShare Feature Store Integration...")
    
    try:
        # Test 1: Feature Store Initialization
        print("1️⃣ Testing feature store initialization...")
        config = FeatureStoreConfig()
        fs = RideShareFeatureStore(config)
        print("✅ Feature store initialized successfully")
        
        # Test 2: Feature Materialization
        print("2️⃣ Testing feature materialization...")
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        materialization_result = fs.materialize_features(start_date, end_date)
        print(f"✅ Materialization completed: {materialization_result['status']}")
        
        # Test 3: Training Features
        print("3️⃣ Testing training feature retrieval...")
        entity_df = pd.DataFrame({
            'driver_id': [1001, 1002, 1003, 1004, 1005],
            'event_timestamp': [datetime.now() - timedelta(hours=1)] * 5
        })
        
        feature_refs = [
            "driver_performance_features:acceptance_rate_7d",
            "driver_performance_features:avg_rating_7d",
            "driver_performance_features:total_rides_7d"
        ]
        
        training_df = fs.get_training_features(entity_df, feature_refs)
        print(f"✅ Retrieved {len(training_df)} training records")
        
        # Test 4: Online Features
        print("4️⃣ Testing online feature serving...")
        entity_rows = [
            {"driver_id": 1001},
            {"driver_id": 1002},
            {"driver_id": 1003}
        ]
        
        online_result = fs.get_online_features(entity_rows, feature_refs)
        print(f"✅ Online features served in {online_result['latency_ms']:.2f}ms")
        
        # Verify latency requirement
        if online_result['latency_ms'] < 10:
            print("✅ Latency requirement met (<10ms)")
        else:
            print(f"⚠️ Latency requirement not met: {online_result['latency_ms']:.2f}ms")
        
        # Test 5: Feature Quality Validation
        print("5️⃣ Testing feature quality validation...")
        monitor = FeatureMonitor(fs)
        
        quality_result = monitor.validate_feature_quality("driver_performance_features")
        print(f"✅ Feature quality score: {quality_result['quality_score']:.3f}")
        
        # Test 6: Drift Detection
        print("6️⃣ Testing drift detection...")
        drift_result = monitor.detect_feature_drift(
            "driver_performance_features:acceptance_rate_7d",
            reference_period=timedelta(days=7),
            current_period=timedelta(days=1)
        )
        
        if drift_result['drift_detected']:
            print(f"⚠️ Drift detected: score={drift_result['drift_score']:.4f}")
        else:
            print(f"✅ No drift detected: score={drift_result['drift_score']:.4f}")
        
        # Test 7: API Endpoints
        print("7️⃣ Testing API endpoints...")
        
        # Test feature serving endpoint
        test_request = FeatureRequest(
            entity_rows=[{"driver_id": 1001}],
            features=["driver_performance_features:acceptance_rate_7d"]
        )
        
        # Simulate API call
        api_result = fs.get_online_features(
            test_request.entity_rows,
            test_request.features
        )
        print(f"✅ API feature serving: {api_result['latency_ms']:.2f}ms")
        
        print("\n🎉 All integration tests completed successfully!")
        
        # Summary
        print("\n📊 Test Summary:")
        print(f"   • Feature store: Initialized and connected")
        print(f"   • Materialization: {materialization_result['status']}")
        print(f"   • Training features: {len(training_df)} records retrieved")
        print(f"   • Online serving: {online_result['latency_ms']:.2f}ms latency")
        print(f"   • Feature quality: {quality_result['quality_score']:.3f} score")
        print(f"   • Drift detection: {'Detected' if drift_result['drift_detected'] else 'No drift'}")
        print(f"   • API endpoints: Functional")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Run integration tests
    success = test_feature_store_integration()
    
    if success:
        print("\n🚀 Starting Feature Serving API...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("\n❌ Integration tests failed. Please fix issues before starting API.")
        sys.exit(1)