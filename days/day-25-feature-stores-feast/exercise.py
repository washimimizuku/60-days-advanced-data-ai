"""
Day 25: Feature Stores - Feast & Tecton - Exercise

Business Scenario:
You're the ML Platform Engineer at "RideShare Analytics", a ride-sharing company. 
You need to build a comprehensive feature store to support multiple ML models including:
- Driver performance prediction
- Demand forecasting  
- Fraud detection
- Dynamic pricing optimization

Your mission is to implement a production-ready feature store using Feast that serves 
features for both training and real-time inference with sub-millisecond latency.

Requirements:
1. Create feature definitions for drivers, users, and rides
2. Configure both online (Redis) and offline (S3/Parquet) stores
3. Implement streaming features with real-time aggregations
4. Build feature serving API for real-time inference
5. Add feature monitoring and quality validation
6. Implement feature versioning and governance

Success Criteria:
- Feature store serves features in <10ms for real-time inference
- Historical features maintain point-in-time correctness for training
- Feature drift detection alerts on statistical changes
- Comprehensive feature documentation and lineage tracking
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Feast imports
from feast import Entity, Feature, FeatureView, FileSource, ValueType, FeatureStore
from feast.on_demand_feature_view import on_demand_feature_view
from feast.field import Field
from feast.types import Float64, Int64, String
from feast.data_source import KafkaSource
from feast.aggregation import Aggregation
from feast import StreamFeatureView
from scipy import stats

# =============================================================================
# EXERCISE 1: ENTITY AND DATA SOURCE DEFINITIONS
# =============================================================================

# TODO: Define entities for the ride-sharing domain
# HINT: Think about the key business objects (driver, user, ride, location)

driver = Entity(
    name="driver_id",
    value_type=ValueType.INT64,
    description="Unique identifier for drivers in the platform"
)

# TODO: Define user entity
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="Unique identifier for platform users/riders"
)

# TODO: Define ride entity  
ride = Entity(
    name="ride_id",
    value_type=ValueType.STRING,
    description="Unique identifier for individual rides"
)

# TODO: Define location entity for geographic features
location = Entity(
    name="location_id",
    value_type=ValueType.STRING,
    description="Geographic location identifier (geohash or zone)"
)

# =============================================================================
# EXERCISE 2: BATCH DATA SOURCES
# =============================================================================

# TODO: Define driver statistics data source
driver_stats_source = FileSource(
    path="/data/driver_stats.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# TODO: Define user behavior data source
user_behavior_source = FileSource(
    path=os.getenv('USER_BEHAVIOR_PATH', '/data/user_behavior.parquet'),
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# TODO: Define ride history data source
ride_history_source = FileSource(
    path=os.getenv('RIDE_HISTORY_PATH', '/data/ride_history.parquet'),
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# TODO: Define location demand data source
location_demand_source = FileSource(
    path=os.getenv('LOCATION_DEMAND_PATH', '/data/location_demand.parquet'),
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# =============================================================================
# EXERCISE 3: STREAMING DATA SOURCES
# =============================================================================

# TODO: Define real-time ride events stream
ride_events_stream = KafkaSource(
    name="ride_events_stream",
    kafka_bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    topic="ride_events",
    timestamp_field="event_timestamp",
    batch_source=ride_history_source
)

# TODO: Define real-time driver location stream
driver_location_stream = KafkaSource(
    name="driver_location_stream",
    kafka_bootstrap_servers=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    topic="driver_locations",
    timestamp_field="event_timestamp",
    batch_source=driver_stats_source
)

# =============================================================================
# EXERCISE 4: BATCH FEATURE VIEWS
# =============================================================================

# TODO: Create driver performance feature view
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
        Feature(name="avg_trip_duration_7d", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=driver_stats_source,
    tags={"team": "ml_platform", "domain": "driver_management"}
)

# TODO: Create user behavior feature view
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
        Feature(name="cancellation_rate_30d", dtype=ValueType.FLOAT)
    ],
    online=True,
    batch_source=user_behavior_source,
    tags={"team": "ml_platform", "domain": "user_analytics"}
)

# TODO: Create location demand feature view
location_demand_features = FeatureView(
    name="location_demand_features",
    entities=["location_id"],
    ttl=timedelta(hours=2),
    features=[
        Feature(name="current_demand_score", dtype=ValueType.FLOAT),
        Feature(name="supply_ratio", dtype=ValueType.FLOAT),
        Feature(name="avg_wait_time_minutes", dtype=ValueType.FLOAT),
        Feature(name="surge_multiplier", dtype=ValueType.FLOAT),
        Feature(name="historical_demand_pattern", dtype=ValueType.STRING)
    ],
    online=True,
    batch_source=location_demand_source,
    tags={"team": "ml_platform", "domain": "demand_forecasting"}
)

# =============================================================================
# EXERCISE 5: STREAMING FEATURE VIEWS WITH AGGREGATIONS
# =============================================================================

# TODO: Create real-time driver metrics with streaming aggregations
driver_realtime_metrics = StreamFeatureView(
    name="driver_realtime_metrics",
    entities=[driver],
    ttl=timedelta(hours=1),
    source=driver_location_stream,
    aggregations=[
        Aggregation(
            column="trip_duration",
            function="avg",
            time_window=timedelta(hours=1),
            feature_name="avg_trip_duration_1h"
        ),
        Aggregation(
            column="trip_distance",
            function="sum",
            time_window=timedelta(hours=1),
            feature_name="total_distance_1h"
        ),
        Aggregation(
            column="trip_fare",
            function="sum",
            time_window=timedelta(hours=1),
            feature_name="total_earnings_1h"
        )
    ],
    online=True,
    schema=[
        Field(name="driver_id", dtype=Int64),
        Field(name="trip_duration", dtype=Float64),
        Field(name="trip_distance", dtype=Float64),
        Field(name="trip_fare", dtype=Float64)
    ]
)

# TODO: Create real-time ride demand metrics
ride_demand_metrics = StreamFeatureView(
    name="ride_demand_metrics",
    entities=[location],
    ttl=timedelta(minutes=30),
    source=ride_events_stream,
    aggregations=[
        Aggregation(
            column="ride_request",
            function="count",
            time_window=timedelta(hours=1),
            feature_name="requests_per_hour"
        ),
        Aggregation(
            column="wait_time",
            function="avg",
            time_window=timedelta(minutes=30),
            feature_name="avg_wait_time_30m"
        )
    ],
    online=True,
    schema=[
        Field(name="location_id", dtype=String),
        Field(name="ride_request", dtype=Int64),
        Field(name="wait_time", dtype=Float64)
    ]
)

# =============================================================================
# EXERCISE 6: ON-DEMAND FEATURE TRANSFORMATIONS
# =============================================================================

# TODO: Create on-demand driver performance score
@on_demand_feature_view(
    sources=[driver_performance_features],
    schema=[
        Field(name="driver_performance_score", dtype=Float64),
        Field(name="driver_risk_category", dtype=String)
    ]
)
def driver_performance_score(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived driver performance metrics"""
    
    df = pd.DataFrame()
    
    # Calculate composite performance score
    df["driver_performance_score"] = (
        features_df["acceptance_rate_7d"] * 0.4 +
        (features_df["avg_rating_7d"] / 5.0) * 0.3 +
        (1 - features_df["cancellation_rate_7d"]) * 0.3
    ).clip(0, 1)
    
    # Categorize risk level based on performance score
    df["driver_risk_category"] = df["driver_performance_score"].apply(
        lambda x: "low" if x > 0.8 else "medium" if x > 0.6 else "high"
    )
    
    return df

# TODO: Create on-demand surge pricing features
@on_demand_feature_view(
    sources=[location_demand_features, driver_realtime_metrics],
    schema=[
        Field(name="dynamic_surge_multiplier", dtype=Float64),
        Field(name="estimated_wait_time", dtype=Float64)
    ]
)
def surge_pricing_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate dynamic pricing and wait time estimates"""
    
    df = pd.DataFrame()
    
    # Calculate dynamic surge multiplier
    df["dynamic_surge_multiplier"] = np.maximum(
        1.0,
        features_df["surge_multiplier"] * 
        (features_df["current_demand_score"] / np.maximum(features_df["supply_ratio"], 0.1))
    ).clip(1.0, 5.0)
    
    # Estimate wait time based on driver availability
    df["estimated_wait_time"] = (
        features_df["avg_wait_time_minutes"] * 
        (2.0 - features_df["supply_ratio"])
    ).clip(1.0, 30.0)
    
    return df

# =============================================================================
# EXERCISE 7: FEATURE STORE CONFIGURATION
# =============================================================================

class RideShareFeatureStore:
    """Production feature store for ride-sharing ML platform"""
    
    def __init__(self, config_path: str = "feature_store.yaml"):
        self.config_path = config_path
        self.store = None
        self._setup_feature_store()
    
    def _setup_feature_store(self):
        """Initialize Feast feature store with production configuration"""
        
        # Create feature store configuration
        config_yaml = f"""
project: {self.config_path}
registry: data/registry.db
provider: local
online_store:
  type: redis
  connection_string: "redis://{os.getenv('REDIS_HOST', 'localhost')}:{os.getenv('REDIS_PORT', '6379')}"
offline_store:
  type: file
"""
        
        # Write configuration file
        with open("feature_store.yaml", "w") as f:
            f.write(config_yaml)
        
        try:
            self.store = FeatureStore(repo_path=".")
            print("✅ Feature store initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize feature store: {e}")
            raise
    
    def materialize_features(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        """Materialize features to online store for serving"""
        
        # TODO: Materialize all feature views to online store
        # HINT: Use store.materialize() with date range
        # Handle errors and return status information
        
        results = {
            'status': 'success',
            'materialized_features': [],
            'errors': []
        }
        
        feature_views = [
            "driver_performance_features",
            "user_behavior_features",
            "location_demand_features"
        ]
        
        for fv_name in feature_views:
            try:
                # Materialize individual feature view
                self.store.materialize(
                    start_date=start_date,
                    end_date=end_date,
                    feature_views=[fv_name]
                )
                
                results['materialized_features'].append(fv_name)
                print(f"✅ Materialized {fv_name}")
                
            except Exception as e:
                error_info = {'feature_view': fv_name, 'error': str(e)}
                results['errors'].append(error_info)
                print(f"❌ Failed to materialize {fv_name}: {e}")
        
        return results
    
    def get_training_features(self, entity_df: pd.DataFrame, 
                            feature_refs: List[str]) -> pd.DataFrame:
        """Get historical features for model training with point-in-time correctness"""
        
        # TODO: Retrieve historical features for training
        # HINT: Use store.get_historical_features()
        # Ensure point-in-time correctness for training data
        
        try:
            # Get historical features
            training_df = self.store.get_historical_features(
                entity_df=entity_df,
                features=feature_refs
            ).to_df()
            
            print(f"✅ Retrieved {len(training_df)} training records")
            return training_df
            
        except Exception as e:
            print(f"❌ Failed to get training features: {e}")
            raise
    
    def get_online_features(self, entity_rows: List[Dict], 
                          feature_refs: List[str]) -> Dict[str, Any]:
        """Get online features for real-time inference"""
        
        # TODO: Retrieve online features for serving
        # HINT: Use store.get_online_features()
        # Measure and log latency for performance monitoring
        
        start_time = time.time()
        
        try:
            # Get online features
            features = self.store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows
            )
            
            latency = (time.time() - start_time) * 1000  # Convert to milliseconds
            
            print(f"✅ Retrieved online features in {latency:.2f}ms")
            
            return {
                'features': features.to_dict(),
                'latency_ms': latency,
                'status': 'success'
            }
            
        except Exception as e:
            print(f"❌ Failed to get online features: {e}")
            raise

# =============================================================================
# EXERCISE 8: FEATURE MONITORING AND VALIDATION
# =============================================================================

class FeatureMonitor:
    """Monitor feature quality and detect drift in production"""
    
    def __init__(self, feature_store: RideShareFeatureStore):
        self.feature_store = feature_store
        
    def validate_feature_quality(self, feature_view_name: str) -> Dict[str, Any]:
        """Validate feature quality using statistical checks"""
        
        # TODO: Implement feature quality validation
        # HINT: Check for null values, outliers, data types
        # Calculate quality score based on validation results
        
        validation_results = {
            'feature_view': feature_view_name,
            'quality_score': 0.0,
            'checks_passed': 0,
            'total_checks': 0,
            'issues': []
        }
        
        # Validation checks
        checks = [
            self._check_null_values,
            self._check_data_types,
            self._check_value_ranges,
            self._check_outliers
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
        
        return validation_results
    
    def _check_null_values(self, df: pd.DataFrame, feature_view: str) -> Dict[str, Any]:
        """Check for excessive null values"""
        null_percentage = df.isnull().sum().sum() / (len(df) * len(df.columns))
        threshold = 0.05
        return {
            'passed': null_percentage <= threshold,
            'issue': f"High null percentage: {null_percentage:.3f}" if null_percentage > threshold else None
        }
    
    def _check_data_types(self, df: pd.DataFrame, feature_view: str) -> Dict[str, Any]:
        """Check data type consistency"""
        return {'passed': True, 'issue': None}
    
    def _check_value_ranges(self, df: pd.DataFrame, feature_view: str) -> Dict[str, Any]:
        """Check if values are within expected ranges"""
        return {'passed': True, 'issue': None}
    
    def _check_outliers(self, df: pd.DataFrame, feature_view: str) -> Dict[str, Any]:
        """Check for excessive outliers"""
        return {'passed': True, 'issue': None}
    
    def detect_feature_drift(self, feature_name: str, 
                           reference_period: timedelta,
                           current_period: timedelta) -> Dict[str, Any]:
        """Detect statistical drift in feature distributions"""
        
        # TODO: Implement feature drift detection
        # HINT: Compare current vs reference distributions
        # Use statistical tests (KS test, Chi-square, etc.)
        # Calculate drift score and alert if threshold exceeded
        
        drift_results = {
            'feature_name': feature_name,
            'drift_score': 0.0,
            'drift_detected': False,
            'statistical_test': 'kolmogorov_smirnov',
            'p_value': 1.0,
            'alert_threshold': 0.05
        }
        
        # Get reference data (mock implementation)
        ref_data = np.random.normal(0.8, 0.1, 1000)  # Mock reference distribution
        curr_data = np.random.normal(0.75, 0.12, 1000)  # Mock current distribution
        
        # Calculate Kolmogorov-Smirnov test
        from scipy import stats
        ks_statistic, p_value = stats.ks_2samp(ref_data, curr_data)
        
        drift_results['drift_score'] = ks_statistic
        drift_results['p_value'] = p_value
        drift_results['drift_detected'] = p_value < drift_results['alert_threshold']
        
        return drift_results

# =============================================================================
# EXERCISE 9: FEATURE SERVING API
# =============================================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="RideShare Feature Serving API")

# Initialize feature store
feature_store = RideShareFeatureStore()

class FeatureRequest(BaseModel):
    """Request model for feature serving"""
    entity_rows: List[Dict[str, Any]]
    features: List[str]

class FeatureResponse(BaseModel):
    """Response model for feature serving"""
    features: Dict[str, Any]
    latency_ms: float
    status: str

@app.post("/features", response_model=FeatureResponse)
async def get_features(request: FeatureRequest):
    """Serve features for real-time ML inference"""
    
    # TODO: Implement feature serving endpoint
    # HINT: Use feature_store.get_online_features()
    # Add error handling and performance monitoring
    # Return features in standardized format
    
    try:
        # Get online features
        result = feature_store.get_online_features(
            entity_rows=request.entity_rows,
            feature_refs=request.features
        )
        
        return FeatureResponse(
            features=result['features'],
            latency_ms=result['latency_ms'],
            status=result['status']
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features/metadata/{feature_view}")
async def get_feature_metadata(feature_view: str):
    """Get metadata and schema information for feature view"""
    
    # Return feature view metadata
    metadata = {
        "feature_view": feature_view,
        "features": [
            {"name": "acceptance_rate_7d", "type": "FLOAT"},
            {"name": "avg_rating_7d", "type": "FLOAT"},
            {"name": "total_rides_7d", "type": "INT64"}
        ],
        "entities": ["driver_id"],
        "tags": {"team": "ml_platform", "domain": "driver_management"},
        "ttl": "7 days",
        "online": True
    }
    
    return metadata

# =============================================================================
# EXERCISE 10: INTEGRATION TESTING
# =============================================================================

def test_feature_store_integration():
    """Test complete feature store functionality"""
    
    print("🧪 Testing RideShare Feature Store Integration...")
    
    try:
        # Initialize feature store
        feature_store = RideShareFeatureStore()
        print("✅ Feature store initialized")
        
        # Test 1 - Materialize features
        end_date = datetime.now()
        start_date = end_date - timedelta(days=1)
        
        materialization_result = feature_store.materialize_features(start_date, end_date)
        print(f"✅ Materialization: {materialization_result['status']}")
        
        # Test 2 - Get training features
        entity_df = pd.DataFrame({
            'driver_id': [1001, 1002, 1003],
            'event_timestamp': [datetime.now()] * 3
        })
        
        feature_refs = [
            "driver_performance_features:acceptance_rate_7d",
            "driver_performance_features:avg_rating_7d"
        ]
        
        training_df = feature_store.get_training_features(entity_df, feature_refs)
        print(f"✅ Training features: {len(training_df)} records")
        
        # Test 3 - Get online features
        entity_rows = [{"driver_id": 1001}, {"driver_id": 1002}]
        
        online_result = feature_store.get_online_features(entity_rows, feature_refs)
        print(f"✅ Online features: {online_result['latency_ms']:.2f}ms")
        
        # Test 4 - Feature monitoring
        monitor = FeatureMonitor(feature_store)
        
        quality_result = monitor.validate_feature_quality("driver_performance_features")
        print(f"✅ Feature quality: {quality_result['quality_score']:.3f}")
        
        drift_result = monitor.detect_feature_drift(
            "driver_performance_features:acceptance_rate_7d",
            timedelta(days=7),
            timedelta(days=1)
        )
        print(f"✅ Drift detection: {'detected' if drift_result['drift_detected'] else 'no drift'}")
        
        print("✅ All integration tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

if __name__ == "__main__":
    # Run the integration test
    success = test_feature_store_integration()
    
    if success:
        # Start the feature serving API
        import uvicorn
        print("🚀 Starting Feature Serving API...")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("❌ Please fix integration test issues before starting API")
        sys.exit(1)
