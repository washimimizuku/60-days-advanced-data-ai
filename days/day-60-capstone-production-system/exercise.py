#!/usr/bin/env python3
"""
Day 60: Capstone - Production System Integration
Exercise Implementation

Build the complete Intelligent Customer Analytics Platform by implementing
the missing components and integrating all technologies from the 60-day curriculum.

This exercise provides a structured approach to building the production system
with clear TODO sections for hands-on implementation.

Author: 60 Days Advanced Data and AI Curriculum
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Core dependencies
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field
import asyncpg
import redis.asyncio as redis
from kafka import KafkaProducer, KafkaConsumer
import mlflow
import mlflow.sklearn
from feast import FeatureStore
from sklearn.ensemble import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn

# GenAI and LangChain
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import structlog

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Metrics for monitoring
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions', ['model_name'])
FEATURE_STORE_REQUESTS = Counter('feature_store_requests_total', 'Feature store requests')
GENAI_INSIGHTS = Counter('genai_insights_generated_total', 'GenAI insights generated')

# Configuration
@dataclass
class Config:
    """Application configuration"""
    # Database connections
    postgres_url: str = os.getenv("POSTGRES_URL", "postgresql://user:pass@localhost:5432/customers")
    mongodb_url: str = os.getenv("MONGODB_URL", "mongodb://localhost:27017/events")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    
    # Kafka configuration
    kafka_bootstrap_servers: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    
    # ML platform
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    feast_repo_path: str = os.getenv("FEAST_REPO_PATH", "/app/feature_repo")
    
    # GenAI configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    chroma_host: str = os.getenv("CHROMA_HOST", "localhost:8000")
    
    # Application settings
    environment: str = os.getenv("ENVIRONMENT", "development")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

config = Config()

# Data Models
class CustomerProfile(BaseModel):
    """Customer profile data model"""
    customer_id: str
    email: str
    first_name: str
    last_name: str
    segment: str
    lifetime_value: float
    registration_date: datetime
    total_purchases: int
    avg_order_value: float
    days_since_last_purchase: int
    churn_probability: Optional[float] = None

class PredictionRequest(BaseModel):
    """ML prediction request model"""
    customer_ids: List[str] = Field(..., description="List of customer IDs for prediction")
    model_name: str = Field(default="churn_prediction", description="Model to use for prediction")

class PredictionResponse(BaseModel):
    """ML prediction response model"""
    customer_id: str
    churn_probability: float
    recommendation_score: float
    segment: str
    confidence: float
    timestamp: datetime

class InsightRequest(BaseModel):
    """GenAI insight generation request"""
    customer_id: str
    include_recommendations: bool = True
    context: Optional[str] = None

class InsightResponse(BaseModel):
    """GenAI insight response"""
    customer_id: str
    insights: str
    recommendations: str
    confidence_score: float
    processing_time_ms: int
    timestamp: datetime

# Database Connection Manager
class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self):
        self.postgres_pool = None
        self.redis_client = None
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            # TODO: Initialize PostgreSQL connection pool
            # Hint: Use asyncpg.create_pool() with config.postgres_url
            # Set min_size=5, max_size=20, command_timeout=60
            pass
            
            # TODO: Initialize Redis connection
            # Hint: Use redis.from_url() with config.redis_url
            # Set encoding="utf-8", decode_responses=True
            pass
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """Retrieve customer profile from PostgreSQL"""
        try:
            # TODO: Implement customer profile retrieval
            # 1. Acquire connection from postgres_pool
            # 2. Execute SQL query to get customer data
            # 3. Return CustomerProfile object if found, None otherwise
            # 
            # SQL Query:
            # SELECT customer_id, email, first_name, last_name, segment,
            #        lifetime_value, registration_date, total_purchases,
            #        avg_order_value, days_since_last_purchase
            # FROM customers WHERE customer_id = $1
            pass
                
        except Exception as e:
            logger.error(f"Error fetching customer profile: {e}")
            raise
    
    async def cache_features(self, customer_id: str, features: Dict[str, Any]):
        """Cache customer features in Redis"""
        try:
            # TODO: Implement feature caching
            # 1. Create cache key: f"features:{customer_id}"
            # 2. Use redis_client.hset() to store features
            # 3. Set expiration with redis_client.expire() (3600 seconds)
            pass
            
        except Exception as e:
            logger.error(f"Error caching features: {e}")
    
    async def get_cached_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached features from Redis"""
        try:
            # TODO: Implement cached feature retrieval
            # 1. Create cache key: f"features:{customer_id}"
            # 2. Use redis_client.hgetall() to get features
            # 3. Return features dict if exists, None otherwise
            pass
            
        except Exception as e:
            logger.error(f"Error retrieving cached features: {e}")
            return None

# ML Model Manager
class MLModelManager:
    """Manages ML models and predictions"""
    
    def __init__(self):
        self.models = {}
        self.feature_store = None
        
    async def initialize(self):
        """Initialize ML components"""
        try:
            # TODO: Initialize MLflow
            # Hint: Use mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            pass
            
            # TODO: Initialize Feast feature store
            # Hint: Use FeatureStore(repo_path=config.feast_repo_path)
            pass
            
            # TODO: Load models
            # Hint: Call self.load_models()
            pass
            
            logger.info("ML components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
            raise
    
    async def load_models(self):
        """Load ML models from MLflow"""
        try:
            # TODO: Load churn prediction model
            # 1. Set model URI: "models:/churn_prediction/production"
            # 2. Use mlflow.sklearn.load_model() to load model
            # 3. Store in self.models["churn_prediction"]
            # 
            # For demo purposes, if model loading fails, create a dummy XGBClassifier
            pass
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Use dummy models for demo
            self.models["churn_prediction"] = XGBClassifier()
    
    async def get_features(self, customer_ids: List[str]) -> pd.DataFrame:
        """Retrieve features from feature store"""
        try:
            # TODO: Get features from Feast
            # 1. Increment FEATURE_STORE_REQUESTS counter
            # 2. Use feature_store.get_online_features() with:
            #    - features: ["customer_features:age", "customer_features:total_purchases", 
            #                "customer_features:avg_order_value", "customer_features:days_since_last_purchase"]
            #    - entity_rows: [{"customer_id": cid} for cid in customer_ids]
            # 3. Convert to DataFrame with .to_df()
            # 4. Return the DataFrame
            
            # For demo purposes, return dummy features if feature store fails
            return pd.DataFrame({
                "customer_id": customer_ids,
                "age": np.random.randint(18, 80, len(customer_ids)),
                "total_purchases": np.random.randint(1, 100, len(customer_ids)),
                "avg_order_value": np.random.uniform(20, 500, len(customer_ids)),
                "days_since_last_purchase": np.random.randint(0, 365, len(customer_ids))
            })
            
        except Exception as e:
            logger.error(f"Error retrieving features: {e}")
            # Return dummy features for demo
            return pd.DataFrame({
                "customer_id": customer_ids,
                "age": np.random.randint(18, 80, len(customer_ids)),
                "total_purchases": np.random.randint(1, 100, len(customer_ids)),
                "avg_order_value": np.random.uniform(20, 500, len(customer_ids)),
                "days_since_last_purchase": np.random.randint(0, 365, len(customer_ids))
            })
    
    async def predict_churn(self, customer_ids: List[str]) -> List[PredictionResponse]:
        """Predict customer churn probability"""
        try:
            # TODO: Implement churn prediction
            # 1. Increment MODEL_PREDICTIONS counter with model_name="churn_prediction"
            # 2. Get features using self.get_features(customer_ids)
            # 3. Prepare feature matrix X from columns: ["age", "total_purchases", "avg_order_value", "days_since_last_purchase"]
            # 4. Fill NaN values with 0
            # 5. Get model from self.models["churn_prediction"]
            # 6. Make predictions using model.predict_proba(X)[:, 1] if available, else use dummy predictions
            # 7. Create PredictionResponse objects for each customer
            # 8. Use helper methods: _calculate_recommendation_score() and _determine_segment()
            
            # For now, return dummy predictions
            predictions = []
            for customer_id in customer_ids:
                churn_prob = np.random.uniform(0, 1)
                predictions.append(PredictionResponse(
                    customer_id=customer_id,
                    churn_probability=churn_prob,
                    recommendation_score=np.random.uniform(0.5, 1.0),
                    segment=np.random.choice(["Premium", "Standard", "Basic"]),
                    confidence=np.random.uniform(0.6, 0.95),
                    timestamp=datetime.utcnow()
                ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in churn prediction: {e}")
            raise
    
    def _calculate_recommendation_score(self, features: pd.Series) -> float:
        """Calculate recommendation relevance score"""
        # TODO: Implement recommendation scoring logic
        # 1. Start with base_score = 0.5
        # 2. Add 0.2 if total_purchases > 10
        # 3. Add 0.2 if avg_order_value > 100
        # 4. Add 0.1 if days_since_last_purchase < 30
        # 5. Return min(1.0, base_score)
        return 0.7  # Placeholder
    
    def _determine_segment(self, features: pd.Series) -> str:
        """Determine customer segment based on features"""
        # TODO: Implement segmentation logic
        # 1. Calculate LTV = avg_order_value * total_purchases
        # 2. Return "Premium" if LTV > 5000
        # 3. Return "Standard" if LTV > 1000
        # 4. Return "Basic" otherwise
        return "Standard"  # Placeholder

# GenAI Insight Generator
class GenAIInsightGenerator:
    """Generates AI-powered customer insights using RAG"""
    
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        
    async def initialize(self):
        """Initialize GenAI components"""
        try:
            if not config.openai_api_key:
                logger.warning("OpenAI API key not provided, using mock responses")
                return
            
            # TODO: Initialize GenAI components
            # 1. Create OpenAIEmbeddings with openai_api_key=config.openai_api_key
            # 2. Create OpenAI LLM with temperature=0.1, max_tokens=500, openai_api_key=config.openai_api_key
            # 3. Call self.setup_knowledge_base()
            pass
            
            logger.info("GenAI components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GenAI components: {e}")
    
    async def setup_knowledge_base(self):
        """Setup vector database with customer knowledge"""
        try:
            # TODO: Create knowledge base
            # 1. Create DataFrame with customer insights knowledge (see solution.py for examples)
            # 2. Use DataFrameLoader to load documents
            # 3. Use RecursiveCharacterTextSplitter to split text (chunk_size=500, chunk_overlap=50)
            # 4. Create Chroma vector store from documents with embeddings
            # 5. Create RetrievalQA chain with LLM and vector store retriever
            pass
            
        except Exception as e:
            logger.error(f"Error setting up knowledge base: {e}")
    
    async def generate_insights(self, customer_profile: CustomerProfile) -> InsightResponse:
        """Generate AI-powered customer insights"""
        start_time = time.time()
        
        try:
            # TODO: Increment GENAI_INSIGHTS counter
            pass
            
            if not self.qa_chain:
                # TODO: Generate mock insights when GenAI is not available
                # Use _generate_mock_insights() and _generate_mock_recommendations()
                insights = "Mock insights: Customer analysis pending GenAI setup"
                recommendations = "Mock recommendations: Configure OpenAI API key for full functionality"
            else:
                # TODO: Generate insights using RAG
                # 1. Format customer context using _format_customer_context()
                # 2. Create insights query asking for customer analysis
                # 3. Create recommendations query asking for specific recommendations
                # 4. Use self.qa_chain.run() for both queries
                pass
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return InsightResponse(
                customer_id=customer_profile.customer_id,
                insights=insights,
                recommendations=recommendations,
                confidence_score=self._calculate_confidence_score(customer_profile),
                processing_time_ms=processing_time,
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            raise
    
    def _format_customer_context(self, customer_profile: CustomerProfile) -> str:
        """Format customer data for LLM context"""
        # TODO: Create formatted context string with customer profile information
        return f"Customer ID: {customer_profile.customer_id}, Segment: {customer_profile.segment}"
    
    def _generate_mock_insights(self, customer_profile: CustomerProfile) -> str:
        """Generate mock insights when GenAI is not available"""
        # TODO: Generate mock insights based on customer profile
        # Include analysis of segment, churn risk, and value
        return "Mock customer insights generated"
    
    def _generate_mock_recommendations(self, customer_profile: CustomerProfile) -> str:
        """Generate mock recommendations when GenAI is not available"""
        # TODO: Generate mock recommendations based on customer profile
        # Include specific actions and strategies
        return "Mock recommendations generated"
    
    def _calculate_confidence_score(self, customer_profile: CustomerProfile) -> float:
        """Calculate confidence score based on data completeness"""
        # TODO: Calculate confidence based on available data fields
        # Check for non-null values in key fields and return score 0.6-0.95
        return 0.8  # Placeholder

# Initialize managers
db_manager = DatabaseManager()
ml_manager = MLModelManager()
genai_manager = GenAIInsightGenerator()

# FastAPI Application
app = FastAPI(
    title="Customer Analytics Platform - Exercise",
    description="Intelligent Customer Analytics Platform - Day 60 Exercise",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Simple authentication (replace with proper auth in production)"""
    if config.environment == "development":
        return {"user_id": "demo_user", "role": "admin"}
    
    # TODO: Implement proper JWT token validation
    # For now, accept any token in development
    return {"user_id": "authenticated_user", "role": "user"}

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "service": "Customer Analytics Platform - Exercise",
        "version": "1.0.0",
        "status": "operational",
        "environment": config.environment,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # TODO: Implement health checks
        # 1. Check database connectivity (postgres and redis)
        # 2. Return health status with component status
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": "unknown",
                "cache": "unknown",
                "ml_models": "unknown",
                "genai": "unknown"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

@app.get("/customers/{customer_id}", response_model=CustomerProfile)
async def get_customer(
    customer_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get customer profile"""
    # TODO: Implement customer retrieval
    # 1. Use db_manager.get_customer_profile(customer_id)
    # 2. Return 404 if customer not found
    # 3. Update REQUEST_COUNT metrics
    
    raise HTTPException(status_code=501, detail="Not implemented - complete the TODO")

@app.post("/predict/churn", response_model=List[PredictionResponse])
async def predict_churn(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Predict customer churn probability"""
    # TODO: Implement churn prediction endpoint
    # 1. Use ml_manager.predict_churn(request.customer_ids)
    # 2. Add background task to cache predictions
    # 3. Update REQUEST_COUNT metrics
    # 4. Handle exceptions appropriately
    
    raise HTTPException(status_code=501, detail="Not implemented - complete the TODO")

@app.post("/insights/generate", response_model=InsightResponse)
async def generate_insights(
    request: InsightRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Generate AI-powered customer insights"""
    # TODO: Implement insight generation endpoint
    # 1. Get customer profile using db_manager.get_customer_profile()
    # 2. Get churn prediction if not available
    # 3. Generate insights using genai_manager.generate_insights()
    # 4. Add background task for logging
    # 5. Update REQUEST_COUNT metrics
    
    raise HTTPException(status_code=501, detail="Not implemented - complete the TODO")

@app.get("/analytics/dashboard")
async def get_dashboard_data(current_user: dict = Depends(get_current_user)):
    """Get dashboard analytics data"""
    # TODO: Implement dashboard data endpoint
    # Return mock dashboard data with customer metrics
    
    return {
        "total_customers": 0,
        "active_customers": 0,
        "churn_rate": 0.0,
        "avg_ltv": 0.0,
        "message": "Complete the TODO to implement dashboard data"
    }

# Background Tasks

async def cache_predictions(predictions: List[PredictionResponse]):
    """Cache predictions in Redis"""
    # TODO: Implement prediction caching
    # Use db_manager.cache_features() to store prediction data
    pass

async def log_insight_generation(customer_id: str, processing_time_ms: int):
    """Log insight generation for analytics"""
    # TODO: Implement insight logging
    # Log the insight generation event with customer_id and processing_time
    pass

# Application startup
@app.on_event("startup")
async def startup_event():
    """Initialize application components"""
    logger.info("Starting Customer Analytics Platform - Exercise...")
    
    # TODO: Initialize all managers
    # 1. await db_manager.initialize()
    # 2. await ml_manager.initialize()
    # 3. await genai_manager.initialize()
    
    logger.info("Application startup complete - ready for implementation!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup application resources"""
    logger.info("Shutting down Customer Analytics Platform...")
    
    # TODO: Cleanup resources
    # await db_manager.close()

# Main application entry point
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("🚀 Starting Customer Analytics Platform - Exercise Mode")
    print("📝 Complete the TODO sections to build the full system")
    print("📖 See SETUP.md for detailed instructions")
    print("🔧 Run 'python test_setup.py' to verify your environment")
    
    # Run the application
    uvicorn.run(
        "exercise:app",
        host="0.0.0.0",
        port=8080,
        reload=config.environment == "development",
        log_level=config.log_level.lower()
    )

"""
Exercise Instructions:

1. **Database Integration** (20 minutes):
   - Complete DatabaseManager.initialize()
   - Implement get_customer_profile()
   - Add feature caching methods

2. **ML Pipeline** (25 minutes):
   - Complete MLModelManager.initialize()
   - Implement predict_churn()
   - Add feature retrieval and scoring logic

3. **GenAI Integration** (20 minutes):
   - Complete GenAIInsightGenerator.initialize()
   - Implement generate_insights()
   - Add RAG system setup

4. **API Endpoints** (15 minutes):
   - Complete all API endpoint implementations
   - Add proper error handling and metrics
   - Implement background tasks

5. **Testing and Validation** (10 minutes):
   - Run test_setup.py to verify setup
   - Test API endpoints with sample data
   - Verify monitoring metrics

Success Criteria:
✅ All TODO sections completed
✅ API endpoints return valid responses
✅ Database connections working
✅ ML predictions generated
✅ GenAI insights created (or mock mode)
✅ Metrics collected and exposed
✅ Tests passing

This exercise demonstrates the integration of all 60 days of curriculum
into a production-ready customer analytics platform.
"""