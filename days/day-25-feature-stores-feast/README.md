# Day 25: Feature Stores - Feast & Tecton

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** feature stores as the foundation of production ML systems
- **Implement** Feast for open-source feature store management with comprehensive feature engineering
- **Build** scalable feature pipelines that ensure consistency between training and serving
- **Deploy** production-ready feature serving infrastructure with sub-millisecond latency
- **Establish** feature monitoring, versioning, and governance practices for enterprise ML operations

⭐ **Difficulty**: Advanced ML Infrastructure (1 hour)

---

## Theory

### What are Feature Stores?

A feature store is a centralized repository for storing, managing, and serving machine learning features. It acts as the bridge between raw data and ML models, ensuring consistency, reusability, and reliability in feature engineering workflows.

**Why Feature Stores are Critical for Production ML**:
- **Training-Serving Consistency**: Eliminate feature skew between training and inference
- **Feature Reusability**: Share features across multiple models and teams
- **Operational Efficiency**: Reduce feature engineering duplication and maintenance overhead
- **Performance**: Sub-millisecond feature retrieval for real-time inference
- **Governance**: Version control, lineage tracking, and access control for features
- **Monitoring**: Detect feature drift and data quality issues proactively

### The Feature Store Problem

```python
# The Traditional Problem: Feature Engineering Duplication
# Training Pipeline (Batch)
def create_training_features(user_data, transaction_data):
    # Complex feature engineering logic
    user_features = user_data.groupby('user_id').agg({
        'transaction_amount': ['sum', 'mean', 'count'],
        'days_since_last_purchase': 'min'
    })
    return user_features

# Serving Pipeline (Real-time) - DIFFERENT IMPLEMENTATION!
def create_serving_features(user_id, redis_client):
    # Reimplemented logic - potential for bugs and inconsistency
    total_spent = redis_client.get(f"user:{user_id}:total_spent")
    avg_spent = redis_client.get(f"user:{user_id}:avg_spent")
    # ... different logic, different bugs, different results
    return features

# Result: Training/serving skew, model performance degradation
```

### Feature Store Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Feature Store Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Data Sources   │    │ Feature Store   │    │   Consumers     │         │
│  │                 │    │                 │    │                 │         │
│  │ • Databases     │───▶│ • Feature Repo  │───▶│ • Training      │         │
│  │ • Streams       │    │ • Offline Store │    │ • Serving       │         │
│  │ • APIs          │    │ • Online Store  │    │ • Analytics     │         │
│  │ • Files         │    │ • Registry      │    │ • Monitoring    │         │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘         │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    Feature Engineering Pipeline                     │   │
│  │                                                                     │   │
│  │  Raw Data → Transform → Validate → Store → Serve → Monitor         │   │
│  │     ↓           ↓          ↓         ↓       ↓        ↓            │   │
│  │  Ingestion   Features   Quality   Offline  Online   Drift          │   │
│  │              Creation   Checks    Store    Store    Detection       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Feast: Open-Source Feature Store

Feast is the leading open-source feature store that provides:

#### Core Components

**1. Feature Repository**
```python
# feature_repo/features.py
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from datetime import timedelta

# Define entities (primary keys for features)
user = Entity(
    name="user_id",
    value_type=ValueType.INT64,
    description="User identifier"
)

driver = Entity(
    name="driver_id", 
    value_type=ValueType.INT64,
    description="Driver identifier"
)

# Define data sources
driver_stats_source = FileSource(
    path="/data/driver_stats.parquet",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# Define feature views
driver_hourly_stats = FeatureView(
    name="driver_hourly_stats",
    entities=["driver_id"],
    ttl=timedelta(hours=2),
    features=[
        Feature(name="conv_rate", dtype=ValueType.FLOAT),
        Feature(name="acc_rate", dtype=ValueType.FLOAT),
        Feature(name="avg_daily_trips", dtype=ValueType.INT64),
    ],
    online=True,
    batch_source=driver_stats_source,
    tags={"team": "driver_management"}
)
```

**2. Offline Store (Historical Features)**
```python
# Training data retrieval with point-in-time correctness
from feast import FeatureStore
import pandas as pd

store = FeatureStore(repo_path=".")

# Entity DataFrame with timestamps
entity_df = pd.DataFrame({
    "driver_id": [1001, 1002, 1003],
    "event_timestamp": [
        datetime(2023, 9, 1, 10, 0, 0),
        datetime(2023, 9, 1, 11, 0, 0), 
        datetime(2023, 9, 1, 12, 0, 0)
    ]
})

# Get historical features for training
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate", 
        "driver_hourly_stats:avg_daily_trips"
    ]
).to_df()

print("Training features with point-in-time correctness:")
print(training_df.head())
```

**3. Online Store (Real-time Serving)**
```python
# Real-time feature retrieval for inference
online_features = store.get_online_features(
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate"
    ],
    entity_rows=[
        {"driver_id": 1001},
        {"driver_id": 1002}
    ]
).to_dict()

print("Online features for serving:")
print(online_features)
# Output: {'driver_id': [1001, 1002], 'conv_rate': [0.85, 0.92], 'acc_rate': [0.78, 0.88]}
```

### Advanced Feature Engineering Patterns

#### 1. Streaming Features with Real-time Aggregations
```python
from feast import StreamFeatureView
from feast.data_source import KafkaSource
from feast.aggregation import Aggregation
from feast.field import Field
from feast.types import Float64, Int64

# Real-time streaming source
stream_source = KafkaSource(
    name="transaction_stream",
    kafka_bootstrap_servers="localhost:9092",
    topic="transactions",
    timestamp_field="event_timestamp",
    batch_source=FileSource(path="/data/transactions.parquet")
)

# Streaming feature view with real-time aggregations
transaction_stats = StreamFeatureView(
    name="transaction_stats",
    entities=[user],
    ttl=timedelta(days=1),
    source=stream_source,
    aggregations=[
        Aggregation(
            column="transaction_amount",
            function="sum",
            time_window=timedelta(hours=1),
            feature_name="total_spent_1h"
        ),
        Aggregation(
            column="transaction_amount", 
            function="count",
            time_window=timedelta(hours=24),
            feature_name="transaction_count_24h"
        ),
        Aggregation(
            column="transaction_amount",
            function="avg",
            time_window=timedelta(days=7),
            feature_name="avg_transaction_7d"
        )
    ],
    online=True,
    schema=[
        Field(name="user_id", dtype=Int64),
        Field(name="transaction_amount", dtype=Float64),
        Field(name="merchant_category", dtype=String)
    ]
)
```

#### 2. Feature Transformations and Business Logic
```python
from feast.on_demand_feature_view import on_demand_feature_view
from feast.field import Field
from feast.types import Float64

# On-demand feature transformations
@on_demand_feature_view(
    sources=[driver_hourly_stats],
    schema=[
        Field(name="driver_performance_score", dtype=Float64),
        Field(name="risk_category", dtype=String)
    ]
)
def driver_performance_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate derived features from base features"""
    
    # Complex business logic for performance scoring
    df = pd.DataFrame()
    df["driver_performance_score"] = (
        features_df["conv_rate"] * 0.4 + 
        features_df["acc_rate"] * 0.3 + 
        (features_df["avg_daily_trips"] / 20) * 0.3
    )
    
    # Risk categorization
    df["risk_category"] = df["driver_performance_score"].apply(
        lambda x: "low" if x > 0.8 else "medium" if x > 0.6 else "high"
    )
    
    return df
```

#### 3. Feature Validation and Quality Monitoring
```python
from feast.dqm.profilers.ge_profiler import ge_profiler
from great_expectations.core.expectation_suite import ExpectationSuite

# Data quality monitoring with Great Expectations
@ge_profiler
def validate_driver_features(df: pd.DataFrame) -> ExpectationSuite:
    """Validate driver features for quality and consistency"""
    
    suite = ExpectationSuite(expectation_suite_name="driver_features_suite")
    
    # Feature value ranges
    suite.add_expectation({
        "expectation_type": "expect_column_values_to_be_between",
        "kwargs": {
            "column": "conv_rate",
            "min_value": 0.0,
            "max_value": 1.0
        }
    })
    
    # Feature completeness
    suite.add_expectation({
        "expectation_type": "expect_column_values_to_not_be_null",
        "kwargs": {"column": "driver_id"}
    })
    
    # Statistical expectations
    suite.add_expectation({
        "expectation_type": "expect_column_mean_to_be_between",
        "kwargs": {
            "column": "avg_daily_trips",
            "min_value": 5,
            "max_value": 50
        }
    })
    
    return suite
```

### Production Feature Store Deployment

#### 1. Infrastructure Configuration
```yaml
# feature_store.yaml
project: production_ml_platform
registry: s3://ml-feature-registry/registry.db
provider: aws

online_store:
  type: redis
  connection_string: "redis://production-redis:6379"
  
offline_store:
  type: redshift
  cluster_id: ml-redshift-cluster
  region: us-west-2
  database: feature_store
  user: feast_user
  s3_staging_location: s3://ml-feature-staging/

batch_engine:
  type: spark
  spark_conf:
    spark.executor.memory: "4g"
    spark.executor.cores: "2"
    spark.sql.execution.arrow.pyspark.enabled: "true"
```

#### 2. Feature Serving API
```python
from fastapi import FastAPI
from feast import FeatureStore
import uvicorn

app = FastAPI(title="Feature Serving API")
store = FeatureStore(repo_path="/opt/feature_repo")

@app.post("/get_features")
async def get_features(request: FeatureRequest):
    """Serve features for real-time inference"""
    
    try:
        # Get online features
        features = store.get_online_features(
            features=request.features,
            entity_rows=request.entities
        ).to_dict()
        
        return {
            "status": "success",
            "features": features,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/feature_metadata/{feature_view}")
async def get_feature_metadata(feature_view: str):
    """Get metadata for a feature view"""
    
    try:
        fv = store.get_feature_view(feature_view)
        return {
            "name": fv.name,
            "entities": [e.name for e in fv.entities],
            "features": [f.name for f in fv.features],
            "ttl": str(fv.ttl),
            "tags": fv.tags
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Feature Store Best Practices

#### 1. Feature Naming and Organization
```python
# Good: Descriptive, hierarchical naming
user_transaction_stats_7d = FeatureView(
    name="user_transaction_stats_7d",
    entities=["user_id"],
    features=[
        Feature(name="total_amount_7d", dtype=ValueType.FLOAT),
        Feature(name="transaction_count_7d", dtype=ValueType.INT64),
        Feature(name="avg_amount_7d", dtype=ValueType.FLOAT),
        Feature(name="unique_merchants_7d", dtype=ValueType.INT64)
    ],
    tags={
        "team": "risk_modeling",
        "domain": "transactions", 
        "window": "7d",
        "version": "v2"
    }
)

# Bad: Unclear naming
user_stats = FeatureView(
    name="stats",
    features=[
        Feature(name="amt", dtype=ValueType.FLOAT),
        Feature(name="cnt", dtype=ValueType.INT64)
    ]
)
```

#### 2. Feature Versioning and Backward Compatibility
```python
# Version 1: Original features
user_features_v1 = FeatureView(
    name="user_features_v1",
    entities=["user_id"],
    features=[
        Feature(name="total_spent", dtype=ValueType.FLOAT),
        Feature(name="transaction_count", dtype=ValueType.INT64)
    ],
    tags={"version": "v1", "deprecated": "2024-01-01"}
)

# Version 2: Enhanced features with backward compatibility
user_features_v2 = FeatureView(
    name="user_features_v2", 
    entities=["user_id"],
    features=[
        Feature(name="total_spent", dtype=ValueType.FLOAT),  # Kept for compatibility
        Feature(name="transaction_count", dtype=ValueType.INT64),  # Kept for compatibility
        Feature(name="avg_transaction_amount", dtype=ValueType.FLOAT),  # New
        Feature(name="days_since_last_transaction", dtype=ValueType.INT64),  # New
        Feature(name="preferred_category", dtype=ValueType.STRING)  # New
    ],
    tags={"version": "v2", "active": "true"}
)
```

#### 3. Feature Monitoring and Alerting
```python
from feast.feature_logging import LoggingConfig, LoggingSource
import pandas as pd

class FeatureMonitor:
    """Monitor feature drift and quality in production"""
    
    def __init__(self, feature_store: FeatureStore):
        self.store = feature_store
        
    def monitor_feature_drift(self, feature_view_name: str, 
                            reference_period: timedelta,
                            current_period: timedelta) -> Dict[str, float]:
        """Detect statistical drift in features"""
        
        # Get reference data
        end_ref = datetime.now() - current_period
        start_ref = end_ref - reference_period
        
        reference_data = self._get_feature_data(
            feature_view_name, start_ref, end_ref
        )
        
        # Get current data  
        end_curr = datetime.now()
        start_curr = end_curr - current_period
        
        current_data = self._get_feature_data(
            feature_view_name, start_curr, end_curr
        )
        
        # Calculate drift metrics
        drift_scores = {}
        
        for feature in reference_data.columns:
            if feature in current_data.columns:
                # KL divergence for drift detection
                drift_score = self._calculate_kl_divergence(
                    reference_data[feature], 
                    current_data[feature]
                )
                drift_scores[feature] = drift_score
                
                # Alert if drift exceeds threshold
                if drift_score > 0.1:
                    self._send_drift_alert(feature, drift_score)
        
        return drift_scores
    
    def _calculate_kl_divergence(self, ref_data: pd.Series, 
                               curr_data: pd.Series) -> float:
        """Calculate KL divergence between two distributions"""
        
        # Simplified KL divergence calculation
        ref_hist, bins = np.histogram(ref_data, bins=50, density=True)
        curr_hist, _ = np.histogram(curr_data, bins=bins, density=True)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        ref_hist += epsilon
        curr_hist += epsilon
        
        # Normalize
        ref_hist /= ref_hist.sum()
        curr_hist /= curr_hist.sum()
        
        # Calculate KL divergence
        kl_div = np.sum(curr_hist * np.log(curr_hist / ref_hist))
        
        return kl_div
    
    def _send_drift_alert(self, feature_name: str, drift_score: float):
        """Send alert for feature drift"""
        
        alert_message = {
            "alert_type": "feature_drift",
            "feature_name": feature_name,
            "drift_score": drift_score,
            "timestamp": datetime.now().isoformat(),
            "severity": "high" if drift_score > 0.2 else "medium"
        }
        
        # In production: send to Slack, PagerDuty, etc.
        print(f"🚨 FEATURE DRIFT ALERT: {alert_message}")
```

---

## 🚀 Quick Start

### Option 1: Full Infrastructure Setup (Recommended)

1. **Prerequisites**:
   - Docker and Docker Compose installed
   - 8GB+ RAM available
   - Ports 3000, 5432, 6379, 6566, 8000, 8888, 9090 available

2. **One-command setup**:
   ```bash
   ./setup.sh
   ```

3. **Access services**:
   - Feature Serving API: http://localhost:8000
   - Jupyter Lab: http://localhost:8888
   - Grafana Dashboard: http://localhost:3000 (admin/admin123)
   - Prometheus: http://localhost:9090

4. **Run interactive demo**:
   ```bash
   python demo.py
   ```

### Option 2: Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run exercises**:
   ```bash
   python exercise.py
   ```

3. **Check solution**:
   ```bash
   python solution.py
   ```

## 📁 Project Structure

```
day-25-feature-stores-feast/
├── README.md                    # Complete guide and documentation
├── exercise.py                  # Hands-on exercises with TODOs
├── solution.py                  # Complete production implementation
├── quiz.md                     # Knowledge assessment
│
├── setup.sh                    # Automated environment setup
├── demo.py                     # Interactive demonstration
├── data_generator.py           # Sample data generation
├── docker-compose.yml          # Complete infrastructure
├── requirements.txt            # Python dependencies
├── .env                       # Environment configuration
│
├── Dockerfile.feast           # Feast server container
├── Dockerfile.jupyter         # Jupyter development environment
├── Dockerfile.datagen         # Data generation container
│
├── data/                      # Generated sample data
│   ├── raw/                   # Original datasets
│   ├── processed/             # Processed features
│   └── init/                  # Database initialization
│
├── feature_repo/              # Feast feature definitions
│   └── feature_store.yaml     # Feast configuration
│
├── notebooks/                 # Interactive Jupyter notebooks
│   └── feature_store_demo.ipynb
│
├── monitoring/                # Observability configuration
│   ├── prometheus.yml         # Metrics collection
│   └── grafana/              # Dashboard configuration
│       ├── dashboards/
│       └── datasources/
│
└── logs/                     # Application logs
```

## 🎯 Learning Objectives

By completing this day, you will:

✅ **Understand** feature store architecture and production patterns  
✅ **Implement** comprehensive Feast feature definitions and transformations  
✅ **Configure** Redis online store and PostgreSQL offline store  
✅ **Build** sub-10ms feature serving APIs with caching and monitoring  
✅ **Monitor** feature quality, drift detection, and observability  
✅ **Deploy** complete production infrastructure with Docker and monitoring  
✅ **Integrate** feature stores with ML pipelines and real-time systems  
✅ **Optimize** feature serving performance and cost efficiency

---

## 💻 Hands-On Exercise (40 min)

### Exercise Overview

**Business Scenario**: You're the ML Platform Engineer at "RideShare Analytics", a ride-sharing company. You need to build a feature store to support multiple ML models including driver performance prediction, demand forecasting, and fraud detection.

**Your Mission**: Implement a production-ready feature store using Feast that serves features for both training and real-time inference.

### Requirements

1. **Feature Engineering**: Create features for drivers, users, and rides
2. **Online/Offline Stores**: Configure Redis and PostgreSQL for different use cases  
3. **Real-time Serving**: Build API for sub-10ms feature retrieval
4. **Monitoring**: Implement feature drift detection and quality monitoring
5. **Governance**: Add versioning, documentation, and access control

### Exercise Steps

1. **Setup Environment**:
   ```bash
   ./setup.sh  # Full infrastructure
   # OR
   pip install -r requirements.txt  # Local only
   ```

2. **Complete TODOs in exercise.py**:
   - Entity and data source definitions
   - Batch and streaming feature views
   - On-demand feature transformations
   - Feature store configuration
   - Feature monitoring and validation
   - API endpoints and integration testing

3. **Test Your Implementation**:
   ```bash
   python exercise.py  # Run your implementation
   python demo.py      # Interactive demonstration
   ```

4. **Verify with Infrastructure**:
   - Check feature serving API: http://localhost:8000/health
   - Explore Jupyter notebooks: http://localhost:8888
   - Monitor with Grafana: http://localhost:3000

### Success Criteria

- ✅ Feature store serves features in <10ms
- ✅ Historical features maintain point-in-time correctness
- ✅ Feature drift detection alerts on statistical changes
- ✅ Complete infrastructure runs successfully
- ✅ All integration tests pass

---

## 🛠️ Infrastructure Components

### Services Included

| Service | Port | Purpose | Access |
|---------|------|---------|--------|
| **Feature Serving API** | 8000 | Real-time feature serving | http://localhost:8000 |
| **Feast Server** | 6566 | Feast feature server | Internal |
| **PostgreSQL** | 5432 | Offline feature store | Internal |
| **Redis** | 6379 | Online feature store | Internal |
| **Jupyter Lab** | 8888 | Interactive development | http://localhost:8888 |
| **Grafana** | 3000 | Monitoring dashboards | http://localhost:3000 |
| **Prometheus** | 9090 | Metrics collection | http://localhost:9090 |

### Key Features

- **🚀 Complete Infrastructure**: Docker Compose with all services
- **📊 Sample Data**: 10K drivers, 50K users, 100K rides, 30 days of data
- **⚡ Sub-10ms Serving**: Redis-backed online feature store
- **📈 Monitoring**: Prometheus metrics + Grafana dashboards
- **🔍 Interactive Demo**: Rich CLI demo with performance benchmarks
- **📓 Jupyter Notebooks**: Interactive development environment
- **🧪 Integration Tests**: Comprehensive test suite

### Monitoring & Observability

- **Feature Serving Latency**: P50, P95, P99 percentiles
- **Feature Quality Metrics**: Null rates, outliers, drift scores
- **System Health**: Redis/PostgreSQL connection status
- **API Performance**: Request rates, error rates, throughput
- **Data Quality**: Feature validation results and alerts

## 📚 Resources

- **Feast Documentation**: [docs.feast.dev](https://docs.feast.dev/) - Comprehensive feature store guide
- **Feature Store Concepts**: [featurestore.org](https://www.featurestore.org/) - Industry best practices
- **MLOps Feature Stores**: [ml-ops.org/content/feature-stores](https://ml-ops.org/content/feature-stores) - Production patterns
- **Tecton Platform**: [tecton.ai](https://www.tecton.ai/) - Enterprise feature store platform
- **Feature Engineering**: [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) - O'Reilly book

---

## 🎯 Key Takeaways

- **Feature stores eliminate training-serving skew** by providing a single source of truth for ML features
- **Feast provides production-ready infrastructure** for both batch and real-time feature serving
- **Point-in-time correctness** ensures historical accuracy for training data generation
- **Online stores enable sub-millisecond serving** for real-time inference applications
- **Feature monitoring and governance** are essential for maintaining model performance in production
- **Streaming features** enable real-time aggregations and up-to-date feature values
- **Feature reusability** accelerates ML development and ensures consistency across teams
- **Proper versioning and documentation** enable safe feature evolution and team collaboration

---

## 🚀 What's Next?

Tomorrow (Day 26), you'll learn **Advanced Feature Engineering** techniques including time series features, NLP embeddings, and automated feature selection.

**Preview**: You'll explore sophisticated feature engineering patterns, automated feature discovery, feature selection algorithms, and advanced transformation techniques that build on the feature store foundation you've established today!

---

## ✅ Before Moving On

- [ ] Understand the critical role of feature stores in production ML systems
- [ ] Can implement Feast for both offline training and online serving
- [ ] Know how to configure and deploy feature store infrastructure
- [ ] Understand feature monitoring, versioning, and governance practices
- [ ] Can build real-time feature serving APIs with sub-millisecond latency
- [ ] Complete the comprehensive feature store implementation exercise
- [ ] Review feature engineering best practices and production patterns

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐ (Advanced ML Infrastructure)

**Welcome to Phase 3: Advanced ML & MLOps!** 🚀  
You're now building the infrastructure foundation for production ML systems! 🤖
