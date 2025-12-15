"""
Day 33: Model Serving at Scale - Exercise

Business Scenario:
You're the Senior ML Engineer at TechCorp Financial Services. The company processes
over 50 million financial transactions daily and needs a comprehensive model serving
infrastructure that can handle:

1. Real-time fraud detection (REST API) - 1000+ RPS with <50ms latency
2. Daily batch risk assessment - 10M+ transactions for compliance reporting  
3. Streaming transaction monitoring - Real-time alerts for suspicious patterns

Your task is to implement a complete model serving system that demonstrates all three
serving patterns with production-ready performance, monitoring, and scalability.

Requirements:
1. REST API Service with FastAPI, caching, and monitoring
2. Batch Processing with Spark and Airflow orchestration
3. Streaming Pipeline with Kafka for real-time processing
4. Performance optimization with caching and model optimization
5. Comprehensive monitoring and alerting

Success Criteria:
- REST API: <50ms p95 latency, >1000 RPS throughput
- Batch: Process 10M records in <2 hours
- Streaming: <100ms end-to-end latency
- Monitoring: Full observability with metrics and tracing
"""

import asyncio
import time
import json
import pickle
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

# FastAPI and async libraries
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn

# Caching and monitoring
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Kafka for streaming (mock implementation)
class MockKafkaProducer:
    def __init__(self, bootstrap_servers):
        self.bootstrap_servers = bootstrap_servers
        
    def send(self, topic, value):
        print(f"Sending to {topic}: {value}")

class MockKafkaConsumer:
    def __init__(self, topic, bootstrap_servers, group_id):
        self.topic = topic
        self.bootstrap_servers = bootstrap_servers
        self.group_id = group_id
        self.messages = []
        
    def poll(self, timeout_ms=1000):
        # Simulate receiving messages
        return {}

class FraudDetectionModel:
    """Simple rule-based fraud detection model for learning purposes"""
    
    def __init__(self):
        self.model_version = "1.0.0"
        self.thresholds = {
            'high_amount': 1000,
            'suspicious_hours': [0, 1, 2, 3, 22, 23],
            'high_risk_categories': ['online', 'international'],
            'young_age_threshold': 25
        }
        print(f"Fraud detection model v{self.model_version} initialized")
    
    def predict_proba(self, features: Dict[str, Any]) -> float:
        """
        Predict fraud probability for a transaction
        
        Args:
            features: Dictionary containing transaction features
            
        Returns:
            Fraud probability (0-1)
        """
        risk_score = 0.0
        
        # Amount-based risk
        amount = features.get('amount', 0)
        if amount > self.thresholds['high_amount']:
            risk_score += 0.3
        
        # Time-based risk
        timestamp = pd.to_datetime(features.get('timestamp', datetime.now()))
        hour = timestamp.hour
        if hour in self.thresholds['suspicious_hours']:
            risk_score += 0.2
        
        # Category-based risk
        category = features.get('merchant_category', '')
        if category in self.thresholds['high_risk_categories']:
            risk_score += 0.25
        
        # Location-based risk
        location = features.get('location', 'domestic')
        if location == 'international':
            risk_score += 0.15
        
        # Age-based risk
        age = features.get('user_age', 35)
        if age < self.thresholds['young_age_threshold']:
            risk_score += 0.1
        
        # Convert to probability and add some randomness
        probability = min(1.0, max(0.0, risk_score))
        probability += np.random.normal(0, 0.05)
        return min(1.0, max(0.0, probability))
    
    def predict(self, features: Dict[str, Any]) -> int:
        """
        Predict binary fraud outcome
        
        Args:
            features: Dictionary containing transaction features
            
        Returns:
            1 for fraud, 0 for legitimate
        """
        probability = self.predict_proba(features)
        return 1 if probability > 0.5 else 0

class TransactionRequest(BaseModel):
    """Request model for fraud prediction"""
    transaction_id: str = Field(..., description="Unique transaction identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    merchant_category: str = Field(..., description="Merchant category")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_id: int = Field(..., gt=0, description="User identifier")
    location: str = Field(default="domestic", description="Transaction location")
    user_age: Optional[int] = Field(35, ge=18, le=100, description="User age")
    use_cache: bool = Field(True, description="Whether to use caching")

class FraudPredictionResponse(BaseModel):
    """Response model for fraud prediction"""
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    fraud_prediction: int = Field(..., ge=0, le=1)
    model_version: str
    processing_time_ms: float
    cached: bool = Field(False)
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

class PredictionCache:
    """Multi-level caching system for predictions"""
    
    def __init__(self, redis_client, ttl=300):
        self.redis_client = redis_client or MockRedis()
        self.ttl = ttl
        self.memory_cache = {}  # L1 cache
        self.max_memory_size = 1000
        self.stats = {'hits': 0, 'misses': 0}
    
    def get_cache_key(self, features: Dict[str, Any]) -> str:
        """
        Generate a deterministic cache key from features
        
        Args:
            features: Transaction features
            
        Returns:
            Cache key string
        """
        # Sort features for consistent hashing
        sorted_features = json.dumps(features, sort_keys=True)
        return hashlib.md5(sorted_features.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Dict]:
        """
        Retrieve cached prediction
        
        Args:
            cache_key: Cache key to lookup
            
        Returns:
            Cached prediction or None
        """
        # Check L1 cache (memory) first
        if cache_key in self.memory_cache:
            self.stats['hits'] += 1
            return self.memory_cache[cache_key]
        
        # Check L2 cache (Redis)
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                result = pickle.loads(cached_data)
                # Promote to L1 cache
                if len(self.memory_cache) < self.max_memory_size:
                    self.memory_cache[cache_key] = result
                self.stats['hits'] += 1
                return result
        except Exception as e:
            print(f"Cache error: {e}")
        
        self.stats['misses'] += 1
        return None
    
    def set(self, cache_key: str, prediction: Dict):
        """
        Store prediction in cache
        
        Args:
            cache_key: Cache key
            prediction: Prediction to cache
        """
        try:
            # Store in L1 cache
            if len(self.memory_cache) < self.max_memory_size:
                self.memory_cache[cache_key] = prediction
            
            # Store in L2 cache
            serialized = pickle.dumps(prediction)
            self.redis_client.setex(cache_key, self.ttl, serialized)
        except Exception as e:
            print(f"Cache storage error: {e}")
    
    def get_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / total if total > 0 else 0.0

# TODO: Implement monitoring metrics
class ModelServingMonitor:
    def __init__(self):
        # TODO: Initialize Prometheus metrics
        # HINT: Create Counter for requests, Histogram for latency, Gauge for active connections
        self.request_count = None  # Counter('ml_requests_total', 'Total requests', ['status'])
        self.request_duration = None  # Histogram('ml_request_duration_seconds', 'Request duration')
        self.active_connections = None  # Gauge('ml_active_connections', 'Active connections')
        
    def record_request(self, status: str, duration: float):
        """
        Record request metrics
        
        Args:
            status: Request status (success/error)
            duration: Request duration in seconds
        """
        # TODO: Implement metrics recording
        # HINT: Increment counter and observe histogram
        pass

# TODO: Implement the main FastAPI application
class ModelServingAPI:
    def __init__(self):
        self.app = FastAPI(
            title="Fraud Detection API",
            description="Production-ready fraud detection model serving",
            version="1.0.0"
        )
        
        # TODO: Initialize components
        self.model = None  # FraudDetectionModel()
        self.cache = None  # PredictionCache(redis_client)
        self.monitor = None  # ModelServingMonitor()
        
        # TODO: Setup API routes
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.post("/predict", response_model=FraudPredictionResponse)
        async def predict_fraud(request: TransactionRequest):
            """
            Predict fraud for a transaction
            """
            # TODO: Implement the main prediction endpoint
            # HINT: 
            # 1. Start timing
            # 2. Check cache first
            # 3. If not cached, make prediction
            # 4. Cache the result
            # 5. Record metrics
            # 6. Return response
            pass
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            # TODO: Implement health check
            # HINT: Return status of model, cache, and other dependencies
            pass
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get performance metrics"""
            # TODO: Return current performance metrics
            pass

# TODO: Implement batch processing system
class BatchFraudProcessor:
    def __init__(self, model: FraudDetectionModel):
        # TODO: Initialize batch processor
        self.model = model
        
    def process_batch(self, input_file: str, output_file: str, batch_size: int = 10000):
        """
        Process large batch of transactions
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            batch_size: Number of records to process at once
        """
        # TODO: Implement batch processing
        # HINT:
        # 1. Read input file in chunks
        # 2. Apply model to each chunk
        # 3. Write results to output file
        # 4. Track progress and performance
        pass
    
    def schedule_daily_batch(self):
        """
        Schedule daily batch processing job
        """
        # TODO: Implement job scheduling logic
        # HINT: This would typically integrate with Airflow or similar scheduler
        pass

# TODO: Implement streaming processor
class StreamingFraudProcessor:
    def __init__(self, model: FraudDetectionModel, kafka_config: Dict):
        # TODO: Initialize streaming processor
        self.model = model
        self.kafka_config = kafka_config
        self.producer = None  # MockKafkaProducer(kafka_config['bootstrap_servers'])
        self.consumer = None  # MockKafkaConsumer(...)
        
    async def process_stream(self, input_topic: str, output_topic: str):
        """
        Process streaming transactions
        
        Args:
            input_topic: Kafka topic for incoming transactions
            output_topic: Kafka topic for fraud predictions
        """
        # TODO: Implement streaming processing
        # HINT:
        # 1. Create Kafka consumer for input topic
        # 2. Process messages in real-time
        # 3. Make predictions
        # 4. Send results to output topic
        # 5. Handle errors and retries
        pass
    
    def generate_alerts(self, prediction: Dict):
        """
        Generate alerts for high-risk transactions
        
        Args:
            prediction: Fraud prediction result
        """
        # TODO: Implement alerting logic
        # HINT: Send alerts for high fraud probability transactions
        pass

# TODO: Implement performance optimization
class ModelOptimizer:
    def __init__(self):
        pass
    
    def optimize_model_for_inference(self, model):
        """
        Optimize model for faster inference
        
        Args:
            model: Model to optimize
            
        Returns:
            Optimized model
        """
        # TODO: Implement model optimization
        # HINT: Consider quantization, pruning, or format conversion (ONNX)
        pass
    
    def setup_connection_pooling(self):
        """
        Setup database connection pooling
        """
        # TODO: Implement connection pooling
        # HINT: Use SQLAlchemy or asyncpg for efficient database connections
        pass

# TODO: Implement comprehensive testing
class ModelServingTester:
    def __init__(self, api: ModelServingAPI):
        self.api = api
        
    async def test_api_performance(self, num_requests: int = 1000):
        """
        Test API performance under load
        
        Args:
            num_requests: Number of concurrent requests to send
        """
        # TODO: Implement load testing
        # HINT:
        # 1. Generate test transactions
        # 2. Send concurrent requests
        # 3. Measure latency and throughput
        # 4. Check for errors
        pass
    
    def test_batch_processing(self):
        """
        Test batch processing performance
        """
        # TODO: Implement batch processing test
        # HINT: Create test dataset and measure processing time
        pass
    
    def test_streaming_latency(self):
        """
        Test streaming processing latency
        """
        # TODO: Implement streaming latency test
        # HINT: Measure end-to-end latency from input to output
        pass

# TODO: Main integration function
async def main():
    """
    Main function to demonstrate complete model serving system
    """
    print("Day 33: Model Serving at Scale - Exercise")
    print("=" * 60)
    
    # TODO: Initialize all components
    print("1. Initializing components...")
    
    # TODO: Setup Redis connection
    # redis_client = redis.Redis(host='localhost', port=6379, db=0)
    
    # TODO: Initialize model serving API
    # api = ModelServingAPI()
    
    # TODO: Initialize batch processor
    # batch_processor = BatchFraudProcessor(api.model)
    
    # TODO: Initialize streaming processor
    # streaming_processor = StreamingFraudProcessor(api.model, kafka_config)
    
    # TODO: Run performance tests
    print("2. Running performance tests...")
    
    # TODO: Test API performance
    # tester = ModelServingTester(api)
    # await tester.test_api_performance()
    
    # TODO: Test batch processing
    # tester.test_batch_processing()
    
    # TODO: Test streaming latency
    # tester.test_streaming_latency()
    
    # TODO: Start monitoring
    print("3. Starting monitoring...")
    # start_http_server(8000)  # Prometheus metrics endpoint
    
    # TODO: Demonstrate all serving patterns
    print("4. Demonstrating serving patterns...")
    
    # TODO: Show real-time API usage
    print("   - Real-time API serving...")
    
    # TODO: Show batch processing
    print("   - Batch processing...")
    
    # TODO: Show streaming processing
    print("   - Streaming processing...")
    
    print("\nExercise completed! Check the implementation above.")
    print("\nNext steps:")
    print("1. Implement the FraudDetectionModel class")
    print("2. Complete the caching layer")
    print("3. Add comprehensive monitoring")
    print("4. Implement batch and streaming processors")
    print("5. Add performance optimization")
    print("6. Run load tests and measure performance")

if __name__ == "__main__":
    asyncio.run(main())