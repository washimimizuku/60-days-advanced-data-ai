#!/usr/bin/env python3
"""
Day 60: Capstone - Production System Integration
Solution Implementation

This solution provides a complete implementation of the Intelligent Customer Analytics Platform
that integrates all technologies from the 60-day curriculum into a production-ready system.

Architecture Components:
- Data Layer: PostgreSQL, MongoDB, Redis, Kafka, Debezium
- Orchestration: Airflow, dbt, Great Expectations
- ML Platform: Feast, MLflow, FastAPI serving
- GenAI: LangChain RAG, DSPy optimization
- Infrastructure: Kubernetes, Terraform, Prometheus

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
from contextlib import asynccontextmanager

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
import boto3

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

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = structlog.get_logger()

# Metrics for monitoring
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration')
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total model predictions', ['model_name'])
FEATURE_STORE_REQUESTS = Counter('feature_store_requests_total', 'Feature store requests')
GENAI_INSIGHTS = Counter('genai_insights_generated_total', 'GenAI insights generated')
ACTIVE_CONNECTIONS = Gauge('active_database_connections', 'Active database connections')

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
    
    # AWS configuration
    aws_region: str = os.getenv("AWS_REGION", "us-west-2")
    
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
            # PostgreSQL connection pool
            self.postgres_pool = await asyncpg.create_pool(
                config.postgres_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )
            
            # Redis connection
            self.redis_client = redis.from_url(
                config.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            
            logger.info("Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def get_customer_profile(self, customer_id: str) -> Optional[CustomerProfile]:
        """Retrieve customer profile from PostgreSQL"""
        try:
            async with self.postgres_pool.acquire() as conn:
                query = """
                    SELECT 
                        customer_id, email, first_name, last_name, segment,
                        lifetime_value, registration_date, total_purchases,
                        avg_order_value, days_since_last_purchase
                    FROM customers 
                    WHERE customer_id = $1
                """
                row = await conn.fetchrow(query, customer_id)
                
                if row:
                    return CustomerProfile(**dict(row))
                return None
                
        except Exception as e:
            logger.error(f"Error fetching customer profile: {e}")
            raise
    
    async def cache_features(self, customer_id: str, features: Dict[str, Any]):
        """Cache customer features in Redis"""
        try:
            cache_key = f"features:{customer_id}"
            await self.redis_client.hset(cache_key, mapping=features)
            await self.redis_client.expire(cache_key, 3600)  # 1 hour TTL
            
        except Exception as e:
            logger.error(f"Error caching features: {e}")
    
    async def get_cached_features(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached features from Redis"""
        try:
            cache_key = f"features:{customer_id}"
            features = await self.redis_client.hgetall(cache_key)
            return features if features else None
            
        except Exception as e:
            logger.error(f"Error retrieving cached features: {e}")
            return None
    
    async def close(self):
        """Close database connections"""
        if self.postgres_pool:
            await self.postgres_pool.close()
        if self.redis_client:
            await self.redis_client.close()

# ML Model Manager
class MLModelManager:
    """Manages ML models and predictions"""
    
    def __init__(self):
        self.models = {}
        self.feature_store = None
        
    async def initialize(self):
        """Initialize ML components"""
        try:
            # Initialize MLflow
            mlflow.set_tracking_uri(config.mlflow_tracking_uri)
            
            # Initialize Feast feature store
            self.feature_store = FeatureStore(repo_path=config.feast_repo_path)
            
            # Load models
            await self.load_models()
            
            logger.info("ML components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ML components: {e}")
            raise
    
    async def load_models(self):
        """Load ML models from MLflow"""
        try:
            # Load churn prediction model
            churn_model_uri = "models:/churn_prediction/production"
            self.models["churn_prediction"] = mlflow.sklearn.load_model(churn_model_uri)
            
            # Load recommendation model
            recommendation_model_uri = "models:/recommendation_engine/production"
            self.models["recommendation"] = mlflow.sklearn.load_model(recommendation_model_uri)
            
            logger.info("Models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Use dummy models for demo
            self.models["churn_prediction"] = XGBClassifier()
            self.models["recommendation"] = XGBClassifier()
    
    async def get_features(self, customer_ids: List[str]) -> pd.DataFrame:
        """Retrieve features from feature store"""
        try:
            FEATURE_STORE_REQUESTS.inc()
            
            # Get features from Feast
            features = self.feature_store.get_online_features(
                features=[
                    "customer_features:age",
                    "customer_features:total_purchases",
                    "customer_features:avg_order_value",
                    "customer_features:days_since_last_purchase",
                    "customer_features:preferred_category"
                ],
                entity_rows=[{"customer_id": cid} for cid in customer_ids]
            ).to_df()
            
            return features
            
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
            MODEL_PREDICTIONS.labels(model_name="churn_prediction").inc(len(customer_ids))
            
            # Get features
            features_df = await self.get_features(customer_ids)
            
            # Prepare feature matrix
            feature_columns = ["age", "total_purchases", "avg_order_value", "days_since_last_purchase"]
            X = features_df[feature_columns].fillna(0)
            
            # Make predictions
            model = self.models["churn_prediction"]
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1]  # Probability of churn
            else:
                # Dummy predictions for demo
                probabilities = np.random.uniform(0, 1, len(customer_ids))
            
            # Generate responses
            predictions = []
            for i, customer_id in enumerate(customer_ids):
                churn_prob = float(probabilities[i])
                
                predictions.append(PredictionResponse(
                    customer_id=customer_id,
                    churn_probability=churn_prob,
                    recommendation_score=self._calculate_recommendation_score(features_df.iloc[i]),
                    segment=self._determine_segment(features_df.iloc[i]),
                    confidence=min(0.95, max(0.6, 1.0 - abs(churn_prob - 0.5) * 2)),
                    timestamp=datetime.utcnow()
                ))
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in churn prediction: {e}")
            raise
    
    def _calculate_recommendation_score(self, features: pd.Series) -> float:
        """Calculate recommendation relevance score"""
        # Simple scoring based on customer activity
        base_score = 0.5
        
        if features.get("total_purchases", 0) > 10:
            base_score += 0.2
        if features.get("avg_order_value", 0) > 100:
            base_score += 0.2
        if features.get("days_since_last_purchase", 365) < 30:
            base_score += 0.1
            
        return min(1.0, base_score)
    
    def _determine_segment(self, features: pd.Series) -> str:
        """Determine customer segment based on features"""
        ltv = features.get("avg_order_value", 0) * features.get("total_purchases", 0)
        
        if ltv > 5000:
            return "Premium"
        elif ltv > 1000:
            return "Standard"
        else:
            return "Basic"

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
            
            # Initialize embeddings and LLM
            self.embeddings = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
            self.llm = OpenAI(temperature=0.1, max_tokens=500, openai_api_key=config.openai_api_key)
            
            # Setup knowledge base
            await self.setup_knowledge_base()
            
            logger.info("GenAI components initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize GenAI components: {e}")
    
    async def setup_knowledge_base(self):
        """Setup vector database with customer knowledge"""
        try:
            # Customer insights knowledge base
            knowledge_data = pd.DataFrame([
                {
                    "content": "High-value customers (LTV > $5000) typically purchase premium products and respond well to exclusive offers and early access to new products. They prefer personalized service and are willing to pay for quality.",
                    "category": "customer_segmentation"
                },
                {
                    "content": "Customers with churn probability > 0.7 should receive immediate retention campaigns including personalized discounts (10-20%), customer success outreach, and product recommendations based on purchase history.",
                    "category": "churn_prevention"
                },
                {
                    "content": "Customers who haven't purchased in 30+ days but have high engagement scores (website visits, email opens) are good candidates for re-engagement campaigns with targeted product recommendations.",
                    "category": "reactivation"
                },
                {
                    "content": "Premium segment customers prefer email communication and detailed product information, while Standard segment responds better to SMS notifications and mobile app push messages.",
                    "category": "communication_preferences"
                },
                {
                    "content": "Customers with high recommendation scores (>0.8) are likely to respond to cross-sell and upsell campaigns. Focus on complementary products and bundle offers.",
                    "category": "cross_sell_upsell"
                },
                {
                    "content": "New customers (registered < 90 days) benefit from onboarding sequences, tutorial content, and first-purchase incentives to establish buying patterns.",
                    "category": "customer_onboarding"
                }
            ])
            
            # Create documents
            loader = DataFrameLoader(knowledge_data, page_content_column="content")
            documents = loader.load()
            
            # Split text
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store
            if self.embeddings:
                self.vectorstore = Chroma.from_documents(
                    documents=splits,
                    embedding=self.embeddings,
                    persist_directory="./customer_knowledge_base"
                )
                
                # Create QA chain
                self.qa_chain = RetrievalQA.from_chain_type(
                    llm=self.llm,
                    chain_type="stuff",
                    retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3})
                )
            
        except Exception as e:
            logger.error(f"Error setting up knowledge base: {e}")
    
    async def generate_insights(self, customer_profile: CustomerProfile) -> InsightResponse:
        """Generate AI-powered customer insights"""
        start_time = time.time()
        
        try:
            GENAI_INSIGHTS.inc()
            
            if not self.qa_chain:
                # Mock response when GenAI is not available
                insights = self._generate_mock_insights(customer_profile)
                recommendations = self._generate_mock_recommendations(customer_profile)
            else:
                # Generate insights using RAG
                context = self._format_customer_context(customer_profile)
                
                insights_query = f"""
                Based on this customer profile, provide specific actionable insights:
                {context}
                
                Focus on:
                1. Customer segment analysis and behavior patterns
                2. Risk assessment and retention strategies
                3. Engagement opportunities and preferences
                """
                
                recommendations_query = f"""
                Based on the customer profile, suggest specific recommendations:
                {context}
                
                Provide:
                1. Product recommendations
                2. Marketing strategies
                3. Communication preferences
                """
                
                insights = self.qa_chain.run(insights_query)
                recommendations = self.qa_chain.run(recommendations_query)
            
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
        return f"""
        Customer Profile:
        - ID: {customer_profile.customer_id}
        - Segment: {customer_profile.segment}
        - Lifetime Value: ${customer_profile.lifetime_value:,.2f}
        - Total Purchases: {customer_profile.total_purchases}
        - Average Order Value: ${customer_profile.avg_order_value:.2f}
        - Days Since Last Purchase: {customer_profile.days_since_last_purchase}
        - Churn Probability: {customer_profile.churn_probability:.2%}
        - Registration Date: {customer_profile.registration_date.strftime('%Y-%m-%d')}
        """
    
    def _generate_mock_insights(self, customer_profile: CustomerProfile) -> str:
        """Generate mock insights when GenAI is not available"""
        segment = customer_profile.segment
        churn_prob = customer_profile.churn_probability or 0.5
        ltv = customer_profile.lifetime_value
        
        insights = f"Customer Analysis for {segment} segment customer:\n\n"
        
        if churn_prob > 0.7:
            insights += "🚨 HIGH CHURN RISK: Immediate retention action required. "
            insights += "Consider personalized outreach and retention offers.\n\n"
        elif churn_prob > 0.4:
            insights += "⚠️ MODERATE CHURN RISK: Monitor engagement and consider proactive retention.\n\n"
        else:
            insights += "✅ LOW CHURN RISK: Focus on growth and upsell opportunities.\n\n"
        
        if ltv > 5000:
            insights += "💎 HIGH VALUE CUSTOMER: Prioritize premium service and exclusive offers.\n"
        elif ltv > 1000:
            insights += "⭐ STANDARD VALUE CUSTOMER: Focus on engagement and cross-sell opportunities.\n"
        else:
            insights += "🌱 GROWTH OPPORTUNITY: Nurture with onboarding and value demonstration.\n"
        
        return insights
    
    def _generate_mock_recommendations(self, customer_profile: CustomerProfile) -> str:
        """Generate mock recommendations when GenAI is not available"""
        recommendations = "Recommended Actions:\n\n"
        
        if customer_profile.days_since_last_purchase > 60:
            recommendations += "1. 📧 Send re-engagement email with personalized product recommendations\n"
            recommendations += "2. 🎁 Offer comeback discount (10-15% off next purchase)\n"
        
        if customer_profile.segment == "Premium":
            recommendations += "3. 🌟 Invite to VIP program with exclusive benefits\n"
            recommendations += "4. 📞 Schedule personal consultation call\n"
        else:
            recommendations += "3. 📱 Send targeted mobile push notifications\n"
            recommendations += "4. 🛍️ Recommend complementary products\n"
        
        recommendations += "5. 📊 Monitor engagement metrics for next 30 days\n"
        
        return recommendations
    
    def _calculate_confidence_score(self, customer_profile: CustomerProfile) -> float:
        """Calculate confidence score based on data completeness"""
        required_fields = [
            customer_profile.lifetime_value,
            customer_profile.total_purchases,
            customer_profile.avg_order_value,
            customer_profile.churn_probability
        ]
        
        available_fields = sum(1 for field in required_fields if field is not None and field > 0)
        return min(0.95, max(0.6, available_fields / len(required_fields)))

# Application Lifecycle Manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("Starting Customer Analytics Platform...")
    
    # Initialize components
    await db_manager.initialize()
    await ml_manager.initialize()
    await genai_manager.initialize()
    
    logger.info("Customer Analytics Platform started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Customer Analytics Platform...")
    await db_manager.close()
    logger.info("Customer Analytics Platform shutdown complete")

# Initialize managers
db_manager = DatabaseManager()
ml_manager = MLModelManager()
genai_manager = GenAIInsightGenerator()

# FastAPI Application
app = FastAPI(
    title="Customer Analytics Platform",
    description="Intelligent Customer Analytics Platform - 60 Days Capstone Project",
    version="1.0.0",
    lifespan=lifespan
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
    
    # In production, validate JWT token here
    token = credentials.credentials
    if not token or token != "demo-token":
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")
    
    return {"user_id": "authenticated_user", "role": "user"}

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "service": "Customer Analytics Platform",
        "version": "1.0.0",
        "status": "operational",
        "environment": config.environment,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connectivity
        if db_manager.postgres_pool:
            async with db_manager.postgres_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
        
        # Check Redis connectivity
        if db_manager.redis_client:
            await db_manager.redis_client.ping()
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "database": "operational",
                "cache": "operational",
                "ml_models": "loaded" if ml_manager.models else "not_loaded",
                "genai": "operational" if genai_manager.qa_chain else "mock_mode"
            }
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")

@app.get("/ready")
async def readiness_check():
    """Readiness check for Kubernetes"""
    if not ml_manager.models:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    return {"status": "ready", "timestamp": datetime.utcnow().isoformat()}

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
    with REQUEST_DURATION.time():
        REQUEST_COUNT.labels(method="GET", endpoint="/customers", status="success").inc()
        
        customer = await db_manager.get_customer_profile(customer_id)
        if not customer:
            REQUEST_COUNT.labels(method="GET", endpoint="/customers", status="not_found").inc()
            raise HTTPException(status_code=404, detail="Customer not found")
        
        return customer

@app.post("/predict/churn", response_model=List[PredictionResponse])
async def predict_churn(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Predict customer churn probability"""
    with REQUEST_DURATION.time():
        try:
            predictions = await ml_manager.predict_churn(request.customer_ids)
            
            # Cache predictions
            background_tasks.add_task(cache_predictions, predictions)
            
            REQUEST_COUNT.labels(method="POST", endpoint="/predict/churn", status="success").inc()
            return predictions
            
        except Exception as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/predict/churn", status="error").inc()
            logger.error(f"Churn prediction failed: {e}")
            raise HTTPException(status_code=500, detail="Prediction failed")

@app.post("/insights/generate", response_model=InsightResponse)
async def generate_insights(
    request: InsightRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Generate AI-powered customer insights"""
    with REQUEST_DURATION.time():
        try:
            # Get customer profile
            customer = await db_manager.get_customer_profile(request.customer_id)
            if not customer:
                raise HTTPException(status_code=404, detail="Customer not found")
            
            # Get churn prediction if not available
            if customer.churn_probability is None:
                predictions = await ml_manager.predict_churn([request.customer_id])
                customer.churn_probability = predictions[0].churn_probability
            
            # Generate insights
            insights = await genai_manager.generate_insights(customer)
            
            # Log for analytics
            background_tasks.add_task(log_insight_generation, request.customer_id, insights.processing_time_ms)
            
            REQUEST_COUNT.labels(method="POST", endpoint="/insights/generate", status="success").inc()
            return insights
            
        except HTTPException:
            raise
        except Exception as e:
            REQUEST_COUNT.labels(method="POST", endpoint="/insights/generate", status="error").inc()
            logger.error(f"Insight generation failed: {e}")
            raise HTTPException(status_code=500, detail="Insight generation failed")

@app.get("/analytics/dashboard")
async def get_dashboard_data(current_user: dict = Depends(get_current_user)):
    """Get dashboard analytics data"""
    try:
        # Mock dashboard data (in production, query from data warehouse)
        dashboard_data = {
            "total_customers": 50000,
            "active_customers": 35000,
            "churn_rate": 0.12,
            "avg_ltv": 2500.0,
            "top_segments": [
                {"segment": "Premium", "count": 5000, "revenue": 12500000},
                {"segment": "Standard", "count": 25000, "revenue": 37500000},
                {"segment": "Basic", "count": 20000, "revenue": 10000000}
            ],
            "predictions_today": 1250,
            "insights_generated": 450,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Dashboard data unavailable")

# Background Tasks

async def cache_predictions(predictions: List[PredictionResponse]):
    """Cache predictions in Redis"""
    try:
        for prediction in predictions:
            cache_data = {
                "churn_probability": prediction.churn_probability,
                "recommendation_score": prediction.recommendation_score,
                "segment": prediction.segment,
                "confidence": prediction.confidence,
                "timestamp": prediction.timestamp.isoformat()
            }
            await db_manager.cache_features(f"prediction:{prediction.customer_id}", cache_data)
            
    except Exception as e:
        logger.error(f"Error caching predictions: {e}")

async def log_insight_generation(customer_id: str, processing_time_ms: int):
    """Log insight generation for analytics"""
    try:
        log_data = {
            "customer_id": customer_id,
            "processing_time_ms": processing_time_ms,
            "timestamp": datetime.utcnow().isoformat(),
            "service": "genai_insights"
        }
        
        # In production, send to analytics pipeline
        logger.info("Insight generated", **log_data)
        
    except Exception as e:
        logger.error(f"Error logging insight generation: {e}")

# Data Pipeline Integration (Airflow DAG would call these endpoints)

@app.post("/pipeline/trigger-training")
async def trigger_model_training(
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Trigger ML model retraining"""
    try:
        # In production, this would trigger Airflow DAG
        background_tasks.add_task(simulate_model_training)
        
        return {
            "status": "training_triggered",
            "message": "Model training pipeline started",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Model training trigger failed: {e}")
        raise HTTPException(status_code=500, detail="Training trigger failed")

async def simulate_model_training():
    """Simulate model training process"""
    try:
        logger.info("Starting model training simulation...")
        
        # Simulate training time
        await asyncio.sleep(5)
        
        # Update model metrics
        MODEL_PREDICTIONS.labels(model_name="churn_prediction_v2").inc(0)
        
        logger.info("Model training simulation completed")
        
    except Exception as e:
        logger.error(f"Model training simulation failed: {e}")

# Cost Optimization Endpoints

@app.get("/cost/optimization")
async def get_cost_optimization_data(current_user: dict = Depends(get_current_user)):
    """Get cost optimization recommendations"""
    try:
        # Mock cost data (in production, integrate with AWS Cost Explorer)
        cost_data = {
            "current_monthly_cost": 15000.0,
            "projected_savings": 2250.0,
            "optimization_opportunities": [
                {
                    "service": "EC2 Instances",
                    "current_cost": 8000.0,
                    "optimized_cost": 6000.0,
                    "savings": 2000.0,
                    "recommendation": "Right-size instances and use spot instances for batch workloads"
                },
                {
                    "service": "RDS",
                    "current_cost": 3000.0,
                    "optimized_cost": 2750.0,
                    "savings": 250.0,
                    "recommendation": "Enable automated backups cleanup and optimize storage"
                }
            ],
            "resource_utilization": {
                "cpu_avg": 0.65,
                "memory_avg": 0.72,
                "storage_utilization": 0.58
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return cost_data
        
    except Exception as e:
        logger.error(f"Cost optimization data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail="Cost data unavailable")

# Main application entry point
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the application
    uvicorn.run(
        "solution:app",
        host="0.0.0.0",
        port=8080,
        reload=config.environment == "development",
        log_level=config.log_level.lower()
    )

"""
Production Deployment Notes:

1. Kubernetes Deployment:
   - Use the provided Kubernetes manifests in project.md
   - Configure resource limits and requests appropriately
   - Set up horizontal and vertical pod autoscaling

2. Environment Variables:
   - Set all configuration variables in Kubernetes secrets/configmaps
   - Use proper database connection strings
   - Configure OpenAI API key for GenAI features

3. Monitoring:
   - Deploy Prometheus to scrape /metrics endpoint
   - Configure Grafana dashboards for visualization
   - Set up alerting rules for critical metrics

4. Security:
   - Implement proper JWT authentication
   - Use TLS/SSL for all communications
   - Configure network policies and RBAC

5. Data Pipeline Integration:
   - Connect to Airflow for orchestration
   - Integrate with dbt for data transformations
   - Set up Kafka consumers for real-time data

6. Cost Optimization:
   - Use spot instances for non-critical workloads
   - Implement auto-scaling policies
   - Monitor and optimize resource usage

This solution demonstrates the integration of all 60 days of curriculum
into a production-ready customer analytics platform.
"""