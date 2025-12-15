#!/usr/bin/env python3
"""
Day 32: Project - ML Model with Feature Store - Production API

FastAPI server for the complete ML platform serving credit risk, fraud detection,
market forecasting, and recommendation models with feature store integration.
"""

import os
import sys
import json
import time
import uuid
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import asyncio
import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager

# FastAPI and Pydantic
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Database and caching
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from dotenv import load_dotenv

# ML libraries
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# Monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import psutil

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=getattr(logging, os.getenv('LOG_LEVEL', 'INFO')),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('ml_platform_predictions_total', 'Total predictions', ['model_name', 'status'])
PREDICTION_LATENCY = Histogram('ml_platform_prediction_duration_seconds', 'Prediction latency', ['model_name'])
FEATURE_STORE_LATENCY = Histogram('ml_platform_feature_store_duration_seconds', 'Feature store latency')
ACTIVE_MODELS = Gauge('ml_platform_active_models', 'Number of active models')
API_REQUESTS = Counter('ml_platform_api_requests_total', 'Total API requests', ['endpoint', 'method', 'status'])

# Security
security = HTTPBearer()

# Pydantic models
class CreditRiskRequest(BaseModel):
    """Credit risk prediction request"""
    age: float = Field(..., ge=18, le=100, description="Age in years")
    income: float = Field(..., ge=0, description="Annual income")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score")
    credit_history_length: float = Field(..., ge=0, description="Credit history length in years")
    loan_amount: float = Field(..., ge=0, description="Requested loan amount")
    loan_purpose: str = Field(..., description="Purpose of loan")
    employment_length: float = Field(..., ge=0, description="Employment length in years")
    debt_to_income_ratio: float = Field(..., ge=0, le=1, description="Debt to income ratio")
    explain: bool = Field(False, description="Include explanation")

class FraudDetectionRequest(BaseModel):
    """Fraud detection request"""
    amount: float = Field(..., ge=0, description="Transaction amount")
    timestamp: str = Field(..., description="Transaction timestamp")
    merchant_category: str = Field(..., description="Merchant category")
    location_type: str = Field(..., description="Location type")
    user_age: float = Field(..., ge=18, le=100, description="User age")
    user_income: float = Field(..., ge=0, description="User income")
    explain: bool = Field(False, description="Include explanation")

class MarketForecastRequest(BaseModel):
    """Market forecast request"""
    periods: int = Field(7, ge=1, le=30, description="Number of periods to forecast")
    confidence_interval: float = Field(0.95, ge=0.8, le=0.99, description="Confidence interval")

class RecommendationRequest(BaseModel):
    """Product recommendation request"""
    user_id: int = Field(..., ge=0, description="User ID")
    n_recommendations: int = Field(10, ge=1, le=50, description="Number of recommendations")
    exclude_categories: List[str] = Field([], description="Categories to exclude")

class PredictionResponse(BaseModel):
    """Generic prediction response"""
    request_id: str
    model_name: str
    prediction: Union[int, float, List[Any]]
    probability: Optional[float] = None
    confidence_interval: Optional[Dict[str, float]] = None
    explanation: Optional[Dict[str, Any]] = None
    processing_time_ms: float
    timestamp: str
    model_version: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: int
    feature_store_connected: bool
    database_connected: bool
    redis_connected: bool
    system_metrics: Dict[str, float]

# Global variables
models_cache = {}
feature_store_client = None
db_pool = None
redis_client = None

class MLPlatformService:
    """
    Core ML platform service managing all models and predictions
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.model_metadata = {}
        self.prediction_history = []
        
    def load_models(self):
        """Load all trained models"""
        logger.info("Loading ML models")
        
        try:
            # Load models from filesystem (in production, load from model registry)
            model_paths = {
                'credit_risk': './models/credit_risk_model.pkl',
                'fraud_detection': './models/fraud_detection_model.pkl',
                'market_forecasting': './models/market_forecasting_model.pkl',
                'recommendations': './models/recommendation_model.pkl'
            }
            
            for model_name, model_path in model_paths.items():
                try:
                    if os.path.exists(model_path):
                        model_data = joblib.load(model_path)
                        self.models[model_name] = model_data.get('model')
                        self.scalers[model_name] = model_data.get('scaler')
                        self.label_encoders[model_name] = model_data.get('label_encoders', {})
                        self.model_metadata[model_name] = model_data.get('metadata', {})
                        logger.info(f"Loaded model: {model_name}")
                    else:
                        # Create dummy models for demo
                        self._create_dummy_model(model_name)
                        
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
                    self._create_dummy_model(model_name)
            
            ACTIVE_MODELS.set(len(self.models))
            logger.info(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Create dummy models for demo
            for model_name in ['credit_risk', 'fraud_detection', 'market_forecasting', 'recommendations']:
                self._create_dummy_model(model_name)
    
    def _create_dummy_model(self, model_name: str):
        """Create dummy model for demonstration"""
        logger.info(f"Creating dummy model: {model_name}")
        
        if model_name == 'credit_risk':
            from sklearn.datasets import make_classification
            X_dummy, y_dummy = make_classification(n_samples=1000, n_features=8, random_state=42)
            
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_dummy, y_dummy)
            
            self.models[model_name] = model
            self.scalers[model_name] = StandardScaler().fit(X_dummy)
            self.model_metadata[model_name] = {
                'version': 'v1.0-demo',
                'accuracy': 0.85,
                'auc': 0.87
            }
            
        elif model_name == 'fraud_detection':
            X_dummy = np.random.random((1000, 12))
            
            model = IsolationForest(contamination=0.05, random_state=42)
            model.fit(X_dummy)
            
            self.models[model_name] = model
            self.scalers[model_name] = StandardScaler().fit(X_dummy)
            self.model_metadata[model_name] = {
                'version': 'v1.0-demo',
                'f1_score': 0.92
            }
            
        else:
            # Placeholder for other models
            self.models[model_name] = None
            self.model_metadata[model_name] = {'version': 'v1.0-demo'}
    
    def predict_credit_risk(self, request: CreditRiskRequest) -> Dict[str, Any]:
        """Predict credit risk"""
        start_time = time.time()
        
        try:
            # Prepare features
            features = np.array([[
                request.age, request.income, request.credit_score,
                request.credit_history_length, request.loan_amount,
                request.employment_length, request.debt_to_income_ratio,
                hash(request.loan_purpose) % 10  # Simple encoding for demo
            ]])
            
            # Scale features
            if 'credit_risk' in self.scalers and self.scalers['credit_risk']:
                features = self.scalers['credit_risk'].transform(features)
            
            # Make prediction
            model = self.models.get('credit_risk')
            if model:
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0, 1] if hasattr(model, 'predict_proba') else 0.5
            else:
                # Fallback prediction based on simple rules
                risk_score = (
                    0.3 * (1 - request.credit_score / 850) +
                    0.2 * request.debt_to_income_ratio +
                    0.2 * (1 - min(request.employment_length, 10) / 10) +
                    0.3 * (request.loan_amount / max(request.income, 1))
                )
                probability = min(max(risk_score, 0), 1)
                prediction = 1 if probability > 0.5 else 0
            
            processing_time = (time.time() - start_time) * 1000
            
            # Generate explanation if requested
            explanation = None
            if request.explain:
                explanation = self._generate_credit_explanation(request, probability)
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'explanation': explanation,
                'processing_time_ms': processing_time
            }
            
            PREDICTION_COUNTER.labels(model_name='credit_risk', status='success').inc()
            PREDICTION_LATENCY.labels(model_name='credit_risk').observe(time.time() - start_time)
            
            return result
            
        except Exception as e:
            PREDICTION_COUNTER.labels(model_name='credit_risk', status='error').inc()
            logger.error(f"Credit risk prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def predict_fraud(self, request: FraudDetectionRequest) -> Dict[str, Any]:
        """Predict fraud"""
        start_time = time.time()
        
        try:
            # Parse timestamp
            try:
                timestamp = pd.to_datetime(request.timestamp)
                hour = timestamp.hour
                day_of_week = timestamp.dayofweek
            except:
                hour = 12
                day_of_week = 1
            
            # Prepare features
            features = np.array([[
                request.amount, np.log1p(request.amount), hour, day_of_week,
                int(hour < 6 or hour > 22), int(day_of_week >= 5),
                hash(request.merchant_category) % 10, hash(request.location_type) % 5,
                request.user_age, request.user_income, 0, 0  # Padding for demo
            ]])
            
            # Scale features
            if 'fraud_detection' in self.scalers and self.scalers['fraud_detection']:
                features = self.scalers['fraud_detection'].transform(features)
            
            # Make prediction
            model = self.models.get('fraud_detection')
            if model:
                anomaly_score = model.decision_function(features)[0]
                probability = 1 / (1 + np.exp(anomaly_score))  # Convert to probability
                prediction = 1 if probability > 0.5 else 0
            else:
                # Fallback prediction based on simple rules
                risk_factors = (
                    0.4 * (request.amount > 1000) +
                    0.3 * (hour < 6 or hour > 22) +
                    0.2 * (request.location_type == 'international') +
                    0.1 * (request.user_age < 25)
                )
                probability = min(max(risk_factors, 0), 1)
                prediction = 1 if probability > 0.5 else 0
            
            processing_time = (time.time() - start_time) * 1000
            
            # Generate explanation if requested
            explanation = None
            if request.explain:
                explanation = self._generate_fraud_explanation(request, probability)
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'explanation': explanation,
                'processing_time_ms': processing_time
            }
            
            PREDICTION_COUNTER.labels(model_name='fraud_detection', status='success').inc()
            PREDICTION_LATENCY.labels(model_name='fraud_detection').observe(time.time() - start_time)
            
            return result
            
        except Exception as e:
            PREDICTION_COUNTER.labels(model_name='fraud_detection', status='error').inc()
            logger.error(f"Fraud detection error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def forecast_market(self, request: MarketForecastRequest) -> Dict[str, Any]:
        """Generate market forecast"""
        start_time = time.time()
        
        try:
            # Generate dummy forecast for demo
            base_value = 150.0
            dates = pd.date_range(start=datetime.now(), periods=request.periods, freq='D')
            
            # Simple trend + noise
            trend = np.linspace(0, request.periods * 0.1, request.periods)
            noise = np.random.normal(0, 2, request.periods)
            values = base_value + trend + noise
            
            # Confidence intervals
            ci_width = 1.96 * 2  # 95% confidence interval
            lower_bound = values - ci_width
            upper_bound = values + ci_width
            
            forecast = []
            for i, date in enumerate(dates):
                forecast.append({
                    'date': date.isoformat(),
                    'value': float(values[i]),
                    'lower_bound': float(lower_bound[i]),
                    'upper_bound': float(upper_bound[i])
                })
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'prediction': forecast,
                'confidence_interval': {'level': request.confidence_interval},
                'processing_time_ms': processing_time
            }
            
            PREDICTION_COUNTER.labels(model_name='market_forecasting', status='success').inc()
            PREDICTION_LATENCY.labels(model_name='market_forecasting').observe(time.time() - start_time)
            
            return result
            
        except Exception as e:
            PREDICTION_COUNTER.labels(model_name='market_forecasting', status='error').inc()
            logger.error(f"Market forecasting error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def recommend_products(self, request: RecommendationRequest) -> Dict[str, Any]:
        """Generate product recommendations"""
        start_time = time.time()
        
        try:
            # Generate dummy recommendations for demo
            all_products = [
                'savings_account', 'checking_account', 'credit_card', 'personal_loan',
                'mortgage', 'investment_account', 'insurance', 'retirement_plan'
            ]
            
            # Filter out excluded categories
            available_products = [p for p in all_products if p not in request.exclude_categories]
            
            # Simple recommendation based on user_id
            np.random.seed(request.user_id)
            recommended_products = np.random.choice(
                available_products, 
                size=min(request.n_recommendations, len(available_products)),
                replace=False
            ).tolist()
            
            # Add confidence scores
            recommendations = []
            for i, product in enumerate(recommended_products):
                score = 0.9 - (i * 0.1)  # Decreasing confidence
                recommendations.append({
                    'product_id': product,
                    'confidence_score': max(score, 0.1),
                    'reason': f'Based on user profile and similar users'
                })
            
            processing_time = (time.time() - start_time) * 1000
            
            result = {
                'prediction': recommendations,
                'processing_time_ms': processing_time
            }
            
            PREDICTION_COUNTER.labels(model_name='recommendations', status='success').inc()
            PREDICTION_LATENCY.labels(model_name='recommendations').observe(time.time() - start_time)
            
            return result
            
        except Exception as e:
            PREDICTION_COUNTER.labels(model_name='recommendations', status='error').inc()
            logger.error(f"Recommendation error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def _generate_credit_explanation(self, request: CreditRiskRequest, probability: float) -> Dict[str, Any]:
        """Generate explanation for credit risk prediction"""
        factors = []
        
        if request.credit_score < 650:
            factors.append({'factor': 'Low credit score', 'impact': 'increases_risk', 'value': request.credit_score})
        
        if request.debt_to_income_ratio > 0.4:
            factors.append({'factor': 'High debt-to-income ratio', 'impact': 'increases_risk', 'value': request.debt_to_income_ratio})
        
        if request.employment_length < 2:
            factors.append({'factor': 'Short employment history', 'impact': 'increases_risk', 'value': request.employment_length})
        
        if request.credit_score > 750:
            factors.append({'factor': 'Excellent credit score', 'impact': 'decreases_risk', 'value': request.credit_score})
        
        return {
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'key_factors': factors,
            'recommendation': 'Approve' if probability < 0.5 else 'Review' if probability < 0.7 else 'Decline'
        }
    
    def _generate_fraud_explanation(self, request: FraudDetectionRequest, probability: float) -> Dict[str, Any]:
        """Generate explanation for fraud prediction"""
        factors = []
        
        if request.amount > 1000:
            factors.append({'factor': 'High transaction amount', 'impact': 'increases_risk', 'value': request.amount})
        
        try:
            hour = pd.to_datetime(request.timestamp).hour
            if hour < 6 or hour > 22:
                factors.append({'factor': 'Unusual transaction time', 'impact': 'increases_risk', 'value': f'{hour}:00'})
        except:
            pass
        
        if request.location_type == 'international':
            factors.append({'factor': 'International transaction', 'impact': 'increases_risk', 'value': request.location_type})
        
        return {
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low',
            'key_factors': factors,
            'recommendation': 'Block' if probability > 0.8 else 'Review' if probability > 0.5 else 'Allow'
        }

# Initialize ML platform service
ml_service = MLPlatformService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting ML Platform API")
    
    # Initialize connections
    global redis_client
    try:
        redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Load models
    ml_service.load_models()
    
    yield
    
    # Shutdown
    logger.info("Shutting down ML Platform API")
    if redis_client:
        redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="ML Platform API",
    description="Production ML platform for FinTech applications",
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

# Middleware for request logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    
    # Log API request
    API_REQUESTS.labels(
        endpoint=request.url.path,
        method=request.method,
        status=response.status_code
    ).inc()
    
    return response

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "ML Platform API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    
    # Check database connection
    db_connected = False
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        conn.close()
        db_connected = True
    except:
        pass
    
    # Check Redis connection
    redis_connected = False
    if redis_client:
        try:
            redis_client.ping()
            redis_connected = True
        except:
            pass
    
    return HealthResponse(
        status="healthy" if db_connected else "degraded",
        timestamp=datetime.now().isoformat(),
        models_loaded=len(ml_service.models),
        feature_store_connected=True,  # Simplified for demo
        database_connected=db_connected,
        redis_connected=redis_connected,
        system_metrics={
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(),
            "disk_percent": psutil.disk_usage('/').percent
        }
    )

@app.post("/predict/credit-risk", response_model=PredictionResponse)
async def predict_credit_risk(request: CreditRiskRequest):
    """Predict credit risk for loan application"""
    
    request_id = str(uuid.uuid4())
    
    try:
        result = ml_service.predict_credit_risk(request)
        
        return PredictionResponse(
            request_id=request_id,
            model_name="credit_risk_ensemble",
            prediction=result['prediction'],
            probability=result['probability'],
            explanation=result['explanation'],
            processing_time_ms=result['processing_time_ms'],
            timestamp=datetime.now().isoformat(),
            model_version=ml_service.model_metadata.get('credit_risk', {}).get('version', 'v1.0')
        )
        
    except Exception as e:
        logger.error(f"Credit risk prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/fraud-detection", response_model=PredictionResponse)
async def predict_fraud(request: FraudDetectionRequest):
    """Detect fraud in transaction"""
    
    request_id = str(uuid.uuid4())
    
    try:
        result = ml_service.predict_fraud(request)
        
        return PredictionResponse(
            request_id=request_id,
            model_name="fraud_detection_anomaly",
            prediction=result['prediction'],
            probability=result['probability'],
            explanation=result['explanation'],
            processing_time_ms=result['processing_time_ms'],
            timestamp=datetime.now().isoformat(),
            model_version=ml_service.model_metadata.get('fraud_detection', {}).get('version', 'v1.0')
        )
        
    except Exception as e:
        logger.error(f"Fraud detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/market-forecast", response_model=PredictionResponse)
async def forecast_market(request: MarketForecastRequest):
    """Generate market forecast"""
    
    request_id = str(uuid.uuid4())
    
    try:
        result = ml_service.forecast_market(request)
        
        return PredictionResponse(
            request_id=request_id,
            model_name="market_forecasting_prophet",
            prediction=result['prediction'],
            confidence_interval=result['confidence_interval'],
            processing_time_ms=result['processing_time_ms'],
            timestamp=datetime.now().isoformat(),
            model_version=ml_service.model_metadata.get('market_forecasting', {}).get('version', 'v1.0')
        )
        
    except Exception as e:
        logger.error(f"Market forecasting failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/recommendations", response_model=PredictionResponse)
async def recommend_products(request: RecommendationRequest):
    """Generate product recommendations"""
    
    request_id = str(uuid.uuid4())
    
    try:
        result = ml_service.recommend_products(request)
        
        return PredictionResponse(
            request_id=request_id,
            model_name="product_recommendations_cf",
            prediction=result['prediction'],
            processing_time_ms=result['processing_time_ms'],
            timestamp=datetime.now().isoformat(),
            model_version=ml_service.model_metadata.get('recommendations', {}).get('version', 'v1.0')
        )
        
    except Exception as e:
        logger.error(f"Product recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get available models and their metadata"""
    
    models_info = []
    
    for model_name, model in ml_service.models.items():
        metadata = ml_service.model_metadata.get(model_name, {})
        
        info = {
            'name': model_name,
            'type': model.__class__.__name__ if model else 'Unknown',
            'version': metadata.get('version', 'unknown'),
            'status': 'active' if model else 'inactive',
            'metadata': metadata
        }
        models_info.append(info)
    
    return models_info

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(generate_latest(), media_type="text/plain")

@app.get("/performance")
async def get_performance():
    """Get performance metrics"""
    
    return {
        'models_loaded': len(ml_service.models),
        'total_predictions': len(ml_service.prediction_history),
        'system_metrics': {
            'memory_percent': psutil.virtual_memory().percent,
            'cpu_percent': psutil.cpu_percent(),
            'disk_percent': psutil.disk_usage('/').percent
        },
        'timestamp': datetime.now().isoformat()
    }

def main():
    """
    Main function to run the API server
    """
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    workers = int(os.getenv('API_WORKERS', 1))
    
    logger.info(f"Starting ML Platform API on {host}:{port}")
    
    uvicorn.run(
        "api:app",
        host=host,
        port=port,
        workers=workers,
        reload=os.getenv('DEBUG', 'false').lower() == 'true',
        log_level=os.getenv('LOG_LEVEL', 'info').lower()
    )

if __name__ == "__main__":
    main()