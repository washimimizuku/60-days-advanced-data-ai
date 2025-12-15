# Day 33: Model Serving at Scale - REST APIs, Batch, Streaming

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Design and implement** scalable model serving architectures for different use cases
- **Build production-ready REST APIs** for real-time model inference with FastAPI
- **Create efficient batch processing systems** for large-scale model predictions
- **Implement streaming inference pipelines** for real-time data processing
- **Apply performance optimization techniques** including caching, load balancing, and auto-scaling

**Estimated Time**: 1 hour  
**Difficulty Level**: Advanced ⭐⭐⭐⭐

---

## 🎯 What is Model Serving at Scale?

Model serving at scale is the practice of deploying machine learning models to production environments where they can handle high volumes of prediction requests efficiently and reliably. It encompasses three primary serving patterns:

### 1. **Real-time Serving (REST APIs)**
- **Synchronous predictions** with low latency (<100ms)
- **Interactive applications** like fraud detection, recommendation systems
- **High availability** requirements (99.9%+ uptime)
- **Auto-scaling** based on request volume

### 2. **Batch Serving**
- **Asynchronous processing** of large datasets
- **Scheduled predictions** for reporting, analytics, risk assessment
- **Cost-efficient** for non-time-sensitive workloads
- **High throughput** over low latency

### 3. **Streaming Serving**
- **Real-time processing** of continuous data streams
- **Event-driven predictions** for IoT, financial markets, monitoring
- **Low latency** with high throughput
- **Fault tolerance** and exactly-once processing

---

## 🏗️ Model Serving Architecture Patterns

### Microservices Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   API Gateway   │    │  Load Balancer  │    │  Model Services │
│                 │    │                 │    │                 │
│ • Authentication│───▶│ • Health Checks │───▶│ • Model A (v1)  │
│ • Rate Limiting │    │ • Routing       │    │ • Model B (v2)  │
│ • Monitoring    │    │ • Failover      │    │ • Model C (v1)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Caching Layer │    │   Feature Store │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Redis Cache   │    │ • Online Store  │    │ • Metrics       │
│ • TTL Policies  │    │ • Feature Eng   │    │ • Logging       │
│ • Invalidation  │    │ • Validation    │    │ • Alerting      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Batch Processing Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Sources  │    │  Batch Scheduler│    │  Compute Cluster│
│                 │    │                 │    │                 │
│ • Data Lake     │───▶│ • Airflow DAGs  │───▶│ • Spark/Dask    │
│ • Data Warehouse│    │ • Cron Jobs     │    │ • Kubernetes    │
│ • File Systems  │    │ • Event Triggers│    │ • Auto-scaling  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Output Store  │    │   Model Registry│    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Predictions   │    │ • Model Versions│    │ • Job Status    │
│ • Metadata      │    │ • A/B Testing   │    │ • Performance   │
│ • Audit Logs    │    │ • Rollback      │    │ • Data Quality  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Streaming Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Event Sources  │    │  Stream Processor│    │  Model Services │
│                 │    │                 │    │                 │
│ • Kafka Topics  │───▶│ • Kafka Streams │───▶│ • Stateless     │
│ • Kinesis       │    │ • Apache Flink  │    │ • Containerized │
│ • Event Hubs    │    │ • Spark Stream  │    │ • Auto-scaling  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Output Sinks  │    │   State Store   │    │   Monitoring    │
│                 │    │                 │    │                 │
│ • Databases     │    │ • Feature Cache │    │ • Latency       │
│ • Message Queues│    │ • Model State   │    │ • Throughput    │
│ • Notifications │    │ • Checkpoints   │    │ • Error Rates   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 🚀 Real-time Model Serving with FastAPI

### Production-Ready API Design

```python
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
import asyncio
import redis
import pickle
from typing import List, Dict, Any
import time

app = FastAPI(
    title="ML Model Serving API",
    description="Production-ready model serving with caching and monitoring",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    features: Dict[str, Any] = Field(..., description="Input features for prediction")
    model_name: str = Field("default", description="Model name to use")
    use_cache: bool = Field(True, description="Whether to use caching")

class PredictionResponse(BaseModel):
    prediction: Any = Field(..., description="Model prediction")
    probability: float = Field(None, description="Prediction probability")
    model_version: str = Field(..., description="Model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    cached: bool = Field(False, description="Whether result was cached")

# Model registry and caching
model_registry = {}
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    start_time = time.time()
    
    # Check cache first
    if request.use_cache:
        cache_key = f"prediction:{hash(str(request.features))}"
        cached_result = redis_client.get(cache_key)
        if cached_result:
            result = pickle.loads(cached_result)
            result.cached = True
            return result
    
    # Load model
    model = model_registry.get(request.model_name)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Make prediction
    prediction = await make_prediction(model, request.features)
    
    processing_time = (time.time() - start_time) * 1000
    
    response = PredictionResponse(
        prediction=prediction["prediction"],
        probability=prediction.get("probability"),
        model_version=model["version"],
        processing_time_ms=processing_time,
        cached=False
    )
    
    # Cache result
    if request.use_cache:
        redis_client.setex(cache_key, 300, pickle.dumps(response))  # 5 min TTL
    
    return response

async def make_prediction(model, features):
    # Simulate async model inference
    await asyncio.sleep(0.01)  # Simulate processing time
    return {"prediction": 0.85, "probability": 0.85}
```

### Load Balancing and Auto-scaling

```python
# Kubernetes deployment configuration
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-serving-api
  template:
    metadata:
      labels:
        app: model-serving-api
    spec:
      containers:
      - name: api
        image: model-serving-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        env:
        - name: REDIS_URL
          value: "redis://redis-service:6379"
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-serving-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

---

## 📊 Batch Model Serving

### Apache Spark Batch Processing

```python
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType
import mlflow

class BatchModelServing:
    def __init__(self, spark_config=None):
        self.spark = SparkSession.builder \
            .appName("BatchModelServing") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
            .getOrCreate()
        
        self.models = {}
    
    def load_model(self, model_name, model_uri):
        """Load model from MLflow registry"""
        model = mlflow.pyfunc.load_model(model_uri)
        self.models[model_name] = model
        return model
    
    def batch_predict(self, input_path, output_path, model_name, batch_size=10000):
        """Process large datasets in batches"""
        
        # Read input data
        df = self.spark.read.parquet(input_path)
        
        # Repartition for optimal processing
        df = df.repartition(200)  # Adjust based on cluster size
        
        # Define prediction UDF
        model = self.models[model_name]
        
        def predict_batch(features):
            import pandas as pd
            # Convert Spark DataFrame partition to Pandas
            pdf = pd.DataFrame(features)
            predictions = model.predict(pdf)
            return predictions.tolist()
        
        predict_udf = udf(predict_batch, DoubleType())
        
        # Apply predictions
        result_df = df.withColumn("prediction", predict_udf(col("features")))
        
        # Write results with partitioning
        result_df.write \
            .mode("overwrite") \
            .partitionBy("date") \
            .parquet(output_path)
        
        return result_df.count()

# Airflow DAG for batch processing
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def run_batch_predictions(**context):
    batch_serving = BatchModelServing()
    batch_serving.load_model("credit_risk", "models:/credit_risk_model/Production")
    
    input_path = f"s3://data-lake/credit-applications/{context['ds']}/"
    output_path = f"s3://predictions/credit-risk/{context['ds']}/"
    
    count = batch_serving.batch_predict(input_path, output_path, "credit_risk")
    print(f"Processed {count} predictions")

dag = DAG(
    'batch_model_serving',
    default_args={
        'owner': 'ml-team',
        'depends_on_past': False,
        'start_date': datetime(2023, 1, 1),
        'email_on_failure': True,
        'email_on_retry': False,
        'retries': 2,
        'retry_delay': timedelta(minutes=5)
    },
    description='Daily batch model predictions',
    schedule_interval='0 2 * * *',  # Run at 2 AM daily
    catchup=False
)

batch_task = PythonOperator(
    task_id='run_batch_predictions',
    python_callable=run_batch_predictions,
    dag=dag
)
```

---

## 🌊 Streaming Model Serving

### Kafka Streams Processing

```python
from kafka import KafkaConsumer, KafkaProducer
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

class StreamingModelServing:
    def __init__(self, bootstrap_servers, model_registry):
        self.bootstrap_servers = bootstrap_servers
        self.model_registry = model_registry
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            batch_size=16384,
            linger_ms=10,
            compression_type='gzip'
        )
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def process_stream(self, input_topic, output_topic, model_name):
        """Process streaming data with model predictions"""
        
        consumer = KafkaConsumer(
            input_topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=f'model-serving-{model_name}',
            enable_auto_commit=True,
            auto_offset_reset='latest'
        )
        
        model = self.model_registry.get_model(model_name)
        
        async for message in self._async_consumer(consumer):
            try:
                # Extract features
                features = message.value
                
                # Make prediction asynchronously
                prediction = await self._predict_async(model, features)
                
                # Prepare output message
                output_message = {
                    'input': features,
                    'prediction': prediction,
                    'model_name': model_name,
                    'timestamp': message.timestamp,
                    'partition': message.partition,
                    'offset': message.offset
                }
                
                # Send to output topic
                self.producer.send(output_topic, output_message)
                
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                # Send to dead letter queue
                self.producer.send(f"{output_topic}-dlq", {
                    'error': str(e),
                    'original_message': message.value
                })
    
    async def _async_consumer(self, consumer):
        """Convert synchronous Kafka consumer to async"""
        loop = asyncio.get_event_loop()
        while True:
            messages = await loop.run_in_executor(
                self.executor, 
                lambda: consumer.poll(timeout_ms=1000)
            )
            for topic_partition, msgs in messages.items():
                for message in msgs:
                    yield message
    
    async def _predict_async(self, model, features):
        """Make async prediction"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            model.predict,
            features
        )

# Apache Flink streaming job (PyFlink)
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import StreamTableEnvironment
from pyflink.datastream.connectors import FlinkKafkaConsumer, FlinkKafkaProducer

def create_flink_streaming_job():
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(4)
    
    # Kafka source
    kafka_source = FlinkKafkaConsumer(
        topics=['input-events'],
        deserialization_schema=...,
        properties={
            'bootstrap.servers': 'localhost:9092',
            'group.id': 'flink-ml-serving'
        }
    )
    
    # Process stream
    stream = env.add_source(kafka_source) \
        .map(lambda x: predict_with_model(x)) \
        .add_sink(FlinkKafkaProducer(...))
    
    env.execute("ML Model Streaming Job")
```

---

## ⚡ Performance Optimization Techniques

### 1. Model Optimization

```python
# Model quantization for faster inference
import torch
import torch.quantization

def quantize_model(model):
    """Quantize PyTorch model for faster inference"""
    model.eval()
    
    # Post-training quantization
    quantized_model = torch.quantization.quantize_dynamic(
        model, 
        {torch.nn.Linear}, 
        dtype=torch.qint8
    )
    
    return quantized_model

# ONNX conversion for cross-platform optimization
import onnx
import onnxruntime as ort

def convert_to_onnx(model, input_shape):
    """Convert model to ONNX for optimized inference"""
    dummy_input = torch.randn(input_shape)
    
    torch.onnx.export(
        model,
        dummy_input,
        "model.onnx",
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    
    # Create ONNX Runtime session
    session = ort.InferenceSession("model.onnx")
    return session
```

### 2. Caching Strategies

```python
import redis
import hashlib
from functools import wraps

class PredictionCache:
    def __init__(self, redis_client, ttl=300):
        self.redis = redis_client
        self.ttl = ttl
    
    def cache_prediction(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            cache_key = self._create_cache_key(args, kwargs)
            
            # Check cache
            cached_result = self.redis.get(cache_key)
            if cached_result:
                return pickle.loads(cached_result)
            
            # Compute and cache
            result = func(*args, **kwargs)
            self.redis.setex(cache_key, self.ttl, pickle.dumps(result))
            
            return result
        return wrapper
    
    def _create_cache_key(self, args, kwargs):
        """Create deterministic cache key"""
        key_data = str(args) + str(sorted(kwargs.items()))
        return hashlib.md5(key_data.encode()).hexdigest()

# Multi-level caching
class MultiLevelCache:
    def __init__(self):
        self.l1_cache = {}  # In-memory cache
        self.l2_cache = redis.Redis()  # Redis cache
        self.l1_max_size = 1000
    
    def get(self, key):
        # Check L1 cache first
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Check L2 cache
        result = self.l2_cache.get(key)
        if result:
            result = pickle.loads(result)
            # Promote to L1 cache
            if len(self.l1_cache) < self.l1_max_size:
                self.l1_cache[key] = result
            return result
        
        return None
    
    def set(self, key, value, ttl=300):
        # Set in both caches
        if len(self.l1_cache) < self.l1_max_size:
            self.l1_cache[key] = value
        self.l2_cache.setex(key, ttl, pickle.dumps(value))
```

### 3. Connection Pooling and Resource Management

```python
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool
import asyncpg
import asyncio

class DatabaseConnectionManager:
    def __init__(self, database_url, pool_size=20):
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600
        )
    
    async def get_async_connection_pool(self, database_url):
        """Create async connection pool for high-performance queries"""
        return await asyncpg.create_pool(
            database_url,
            min_size=10,
            max_size=50,
            command_timeout=60
        )

# Resource monitoring and auto-scaling
class ResourceMonitor:
    def __init__(self):
        self.metrics = {
            'cpu_usage': 0,
            'memory_usage': 0,
            'request_rate': 0,
            'response_time': 0
        }
    
    def should_scale_up(self):
        return (
            self.metrics['cpu_usage'] > 70 or
            self.metrics['memory_usage'] > 80 or
            self.metrics['response_time'] > 500
        )
    
    def should_scale_down(self):
        return (
            self.metrics['cpu_usage'] < 30 and
            self.metrics['memory_usage'] < 40 and
            self.metrics['response_time'] < 100
        )
```

---

## 📊 Monitoring and Observability

### Comprehensive Monitoring Setup

```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import logging

# Prometheus metrics
REQUEST_COUNT = Counter('ml_requests_total', 'Total ML requests', ['model', 'status'])
REQUEST_DURATION = Histogram('ml_request_duration_seconds', 'Request duration', ['model'])
MODEL_ACCURACY = Gauge('ml_model_accuracy', 'Model accuracy', ['model', 'version'])
ACTIVE_CONNECTIONS = Gauge('ml_active_connections', 'Active connections')

class ModelServingMonitor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def monitor_prediction(self, func):
        @wraps(func)
        async def wrapper(model_name, *args, **kwargs):
            start_time = time.time()
            
            try:
                result = await func(model_name, *args, **kwargs)
                
                # Record success metrics
                REQUEST_COUNT.labels(model=model_name, status='success').inc()
                REQUEST_DURATION.labels(model=model_name).observe(time.time() - start_time)
                
                return result
                
            except Exception as e:
                # Record error metrics
                REQUEST_COUNT.labels(model=model_name, status='error').inc()
                self.logger.error(f"Prediction error for {model_name}: {e}")
                raise
                
        return wrapper
    
    def update_model_accuracy(self, model_name, version, accuracy):
        MODEL_ACCURACY.labels(model=model_name, version=version).set(accuracy)

# Distributed tracing with OpenTelemetry
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

def setup_tracing():
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)
    
    jaeger_exporter = JaegerExporter(
        agent_host_name="localhost",
        agent_port=6831,
    )
    
    span_processor = BatchSpanProcessor(jaeger_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)
    
    return tracer

# Usage in API
@app.middleware("http")
async def add_tracing(request, call_next):
    tracer = trace.get_tracer(__name__)
    
    with tracer.start_as_current_span("http_request") as span:
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", str(request.url))
        
        response = await call_next(request)
        
        span.set_attribute("http.status_code", response.status_code)
        return response
```

---

## 🔧 Hands-On Exercise

You'll implement a complete model serving system that demonstrates all three serving patterns:

### Exercise Scenario
**Company**: TechCorp Financial Services  
**Challenge**: Deploy a fraud detection model that can handle:
- **Real-time transactions** (REST API) - 1000+ RPS with <50ms latency
- **Daily batch processing** - 10M+ transactions for risk analysis
- **Streaming alerts** - Real-time fraud detection from transaction streams

### Requirements
1. **REST API Service**:
   - FastAPI with async endpoints
   - Redis caching with TTL
   - Prometheus monitoring
   - Auto-scaling configuration

2. **Batch Processing**:
   - Spark job for large-scale processing
   - Airflow orchestration
   - Partitioned output storage

3. **Streaming Pipeline**:
   - Kafka consumer/producer
   - Real-time fraud scoring
   - Alert generation

4. **Performance Optimization**:
   - Model quantization
   - Connection pooling
   - Multi-level caching

---

## 📚 Key Takeaways

- **Choose the right serving pattern** based on latency, throughput, and cost requirements
- **Implement comprehensive monitoring** for production reliability and performance optimization
- **Use caching strategically** to reduce latency and computational costs
- **Design for scalability** with auto-scaling, load balancing, and resource management
- **Optimize models** for inference performance through quantization and format conversion
- **Implement proper error handling** and circuit breaker patterns for resilience
- **Monitor business metrics** alongside technical metrics for complete observability
- **Plan for model versioning** and A/B testing in production environments

---

## 🔄 What's Next?

Tomorrow, we'll explore **A/B Testing for ML** where you'll learn how to:
- Design and implement A/B testing frameworks for ML models
- Measure statistical significance and business impact
- Handle multi-armed bandit problems for dynamic model selection
- Build experimentation platforms for continuous model improvement

The model serving infrastructure you build today will serve as the foundation for sophisticated experimentation and optimization strategies.

---

## 📖 Additional Resources

### Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Apache Spark ML Guide](https://spark.apache.org/docs/latest/ml-guide.html)
- [Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)

### Performance Optimization
- [ONNX Runtime Performance Tuning](https://onnxruntime.ai/docs/performance/)
- [PyTorch Model Optimization](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [Redis Performance Best Practices](https://redis.io/topics/benchmarks)

### Monitoring and Observability
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [OpenTelemetry Python Guide](https://opentelemetry.io/docs/instrumentation/python/)
- [Grafana Dashboard Design](https://grafana.com/docs/grafana/latest/best-practices/)

### Production Deployment
- [Kubernetes ML Workloads](https://kubernetes.io/docs/concepts/workloads/)
- [MLflow Model Serving](https://mlflow.org/docs/latest/models.html#built-in-deployment-tools)
- [AWS SageMaker Endpoints](https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html)