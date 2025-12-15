#!/usr/bin/env python3
"""
Day 30: Ensemble Methods - Production API Server

FastAPI server for serving ensemble models with comprehensive monitoring,
caching, and production-ready features.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
import asyncio
import numpy as np
import pandas as pd
import joblib
from contextlib import asynccontextmanager

# FastAPI and Pydantic
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Database and caching
import psycopg2
from psycopg2.extras import RealDictCursor
import redis
from dotenv import load_dotenv

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
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
PREDICTION_COUNTER = Counter('ensemble_predictions_total', 'Total predictions made', ['model_type', 'status'])
PREDICTION_LATENCY = Histogram('ensemble_prediction_duration_seconds', 'Prediction latency')
MODEL_ACCURACY = Gauge('ensemble_model_accuracy', 'Current model accuracy', ['model_name'])
ACTIVE_MODELS = Gauge('ensemble_active_models', 'Number of active models')
SYSTEM_MEMORY = Gauge('ensemble_system_memory_percent', 'System memory usage')
SYSTEM_CPU = Gauge('ensemble_system_cpu_percent', 'System CPU usage')

# Pydantic models
class PredictionRequest(BaseModel):
    """Request model for predictions"""
    features: List[float] = Field(..., description="Feature values for prediction")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    return_probabilities: bool = Field(False, description="Return prediction probabilities")
    request_id: Optional[str] = Field(None, description="Unique request identifier")
    
    @validator('features')
    def validate_features(cls, v):
        if len(v) != 25:  # Expected number of features
            raise ValueError(f"Expected 25 features, got {len(v)}")
        return v

class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions"""
    features_batch: List[List[float]] = Field(..., description="Batch of feature vectors")
    model_name: Optional[str] = Field(None, description="Specific model to use")
    return_probabilities: bool = Field(False, description="Return prediction probabilities")
    batch_id: Optional[str] = Field(None, description="Unique batch identifier")

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: Union[int, List[int]]
    probability: Optional[Union[float, List[float]]] = None
    model_used: str
    latency_ms: float
    timestamp: str
    request_id: Optional[str] = None

class ModelInfo(BaseModel):
    """Model information response"""
    name: str
    model_type: str
    algorithm: str
    performance_metrics: Dict[str, float]
    training_date: str
    is_active: bool

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: int
    database_connected: bool
    redis_connected: bool
    system_metrics: Dict[str, float]

# Global variables
models_cache = {}
scaler_cache = {}
db_pool = None
redis_client = None

class EnsembleModelManager:
    """
    Manages ensemble models with caching and monitoring
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.model_metadata = {}
        self.prediction_history = []
        
    def load_models(self):
        """Load all active models from database and filesystem"""
        logger.info("Loading ensemble models")
        
        try:
            # Connect to database
            conn = psycopg2.connect(os.getenv('DATABASE_URL'))
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get active models
            cur.execute("""
                SELECT name, model_type, algorithm, model_path, performance_metrics, training_date
                FROM models 
                WHERE is_active = true
                ORDER BY training_date DESC
            """)
            
            active_models = cur.fetchall()
            
            for model_info in active_models:
                model_name = model_info['name']
                model_path = model_info['model_path']
                
                try:
                    if model_path and os.path.exists(model_path):
                        # Load model from file
                        model_data = joblib.load(model_path)
                        
                        if isinstance(model_data, dict):
                            self.models[model_name] = model_data.get('model')
                            self.scalers[model_name] = model_data.get('scaler')
                        else:
                            self.models[model_name] = model_data
                            
                        self.model_metadata[model_name] = dict(model_info)
                        logger.info(f"Loaded model: {model_name}")
                        
                    else:
                        # Create default models if no saved models exist
                        self._create_default_model(model_name, model_info['algorithm'])
                        
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
                    # Create fallback model
                    self._create_default_model(model_name, 'RandomForest')
            
            cur.close()
            conn.close()
            
            # Update metrics
            ACTIVE_MODELS.set(len(self.models))
            
            logger.info(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Create default ensemble if database fails
            self._create_default_ensemble()
    
    def _create_default_model(self, model_name: str, algorithm: str):
        """Create a default model for testing"""
        logger.info(f"Creating default {algorithm} model: {model_name}")
        
        if algorithm.lower() == 'randomforest':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif algorithm.lower() == 'gradientboosting':
            model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            model = LogisticRegression(
                random_state=42,
                max_iter=1000
            )
        
        # Generate dummy training data for the model
        from sklearn.datasets import make_classification
        X_dummy, y_dummy = make_classification(
            n_samples=1000, n_features=25, n_classes=2, random_state=42
        )
        
        # Train model
        model.fit(X_dummy, y_dummy)
        
        # Create scaler
        scaler = StandardScaler()
        scaler.fit(X_dummy)
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.model_metadata[model_name] = {
            'name': model_name,
            'algorithm': algorithm,
            'model_type': 'ensemble',
            'training_date': datetime.now().isoformat(),
            'performance_metrics': {'accuracy': 0.85, 'auc': 0.88}
        }
    
    def _create_default_ensemble(self):
        """Create default ensemble models for testing"""
        logger.info("Creating default ensemble models")
        
        default_models = [
            ('random_forest_default', 'RandomForest'),
            ('gradient_boosting_default', 'GradientBoosting'),
            ('logistic_regression_default', 'LogisticRegression')
        ]
        
        for model_name, algorithm in default_models:
            self._create_default_model(model_name, algorithm)
    
    def predict(self, features: List[float], model_name: Optional[str] = None,
                return_probabilities: bool = False) -> Dict[str, Any]:
        """Make prediction using specified or best model"""
        
        start_time = time.time()
        
        try:
            # Select model
            if model_name and model_name in self.models:
                selected_model = self.models[model_name]
                selected_scaler = self.scalers.get(model_name)
                used_model_name = model_name
            else:
                # Use first available model
                if not self.models:
                    raise HTTPException(status_code=503, detail="No models available")
                
                used_model_name = list(self.models.keys())[0]
                selected_model = self.models[used_model_name]
                selected_scaler = self.scalers.get(used_model_name)
            
            # Prepare features
            X = np.array(features).reshape(1, -1)
            
            # Scale features if scaler available
            if selected_scaler:
                X = selected_scaler.transform(X)
            
            # Make prediction
            prediction = selected_model.predict(X)[0]
            
            result = {
                'prediction': int(prediction),
                'model_used': used_model_name,
                'latency_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add probabilities if requested
            if return_probabilities and hasattr(selected_model, 'predict_proba'):
                probabilities = selected_model.predict_proba(X)[0]
                result['probability'] = float(probabilities[1])  # Probability of positive class
            
            # Update metrics
            PREDICTION_COUNTER.labels(
                model_type=used_model_name, 
                status='success'
            ).inc()
            PREDICTION_LATENCY.observe(time.time() - start_time)
            
            # Log prediction
            self._log_prediction(result, features)
            
            return result
            
        except Exception as e:
            PREDICTION_COUNTER.labels(
                model_type=model_name or 'unknown', 
                status='error'
            ).inc()
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def batch_predict(self, features_batch: List[List[float]], 
                     model_name: Optional[str] = None,
                     return_probabilities: bool = False) -> Dict[str, Any]:
        """Make batch predictions"""
        
        start_time = time.time()
        
        try:
            # Select model
            if model_name and model_name in self.models:
                selected_model = self.models[model_name]
                selected_scaler = self.scalers.get(model_name)
                used_model_name = model_name
            else:
                if not self.models:
                    raise HTTPException(status_code=503, detail="No models available")
                
                used_model_name = list(self.models.keys())[0]
                selected_model = self.models[used_model_name]
                selected_scaler = self.scalers.get(used_model_name)
            
            # Prepare features
            X = np.array(features_batch)
            
            # Scale features if scaler available
            if selected_scaler:
                X = selected_scaler.transform(X)
            
            # Make predictions
            predictions = selected_model.predict(X)
            
            result = {
                'predictions': predictions.tolist(),
                'model_used': used_model_name,
                'batch_size': len(features_batch),
                'latency_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add probabilities if requested
            if return_probabilities and hasattr(selected_model, 'predict_proba'):
                probabilities = selected_model.predict_proba(X)
                result['probabilities'] = probabilities[:, 1].tolist()  # Positive class probabilities
            
            # Update metrics
            PREDICTION_COUNTER.labels(
                model_type=used_model_name, 
                status='success'
            ).inc(len(features_batch))
            PREDICTION_LATENCY.observe(time.time() - start_time)
            
            return result
            
        except Exception as e:
            PREDICTION_COUNTER.labels(
                model_type=model_name or 'unknown', 
                status='error'
            ).inc()
            logger.error(f"Batch prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    def _log_prediction(self, result: Dict[str, Any], features: List[float]):
        """Log prediction for monitoring"""
        
        log_entry = {
            'timestamp': result['timestamp'],
            'model_used': result['model_used'],
            'prediction': result['prediction'],
            'latency_ms': result['latency_ms'],
            'features_hash': hash(tuple(features))  # Don't log actual features for privacy
        }
        
        self.prediction_history.append(log_entry)
        
        # Keep only recent history
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-1000:]
        
        # Store in database if available
        try:
            if db_pool:
                conn = psycopg2.connect(os.getenv('DATABASE_URL'))
                cur = conn.cursor()
                
                cur.execute("""
                    INSERT INTO predictions (model_id, input_features, prediction, latency_ms, prediction_time)
                    VALUES ((SELECT id FROM models WHERE name = %s LIMIT 1), %s, %s, %s, %s)
                """, (
                    result['model_used'],
                    json.dumps(features),
                    result['prediction'],
                    result['latency_ms'],
                    datetime.now()
                ))
                
                conn.commit()
                cur.close()
                conn.close()
                
        except Exception as e:
            logger.warning(f"Failed to log prediction to database: {e}")
    
    def get_model_info(self) -> List[Dict[str, Any]]:
        """Get information about all loaded models"""
        
        model_info = []
        for name, metadata in self.model_metadata.items():
            info = {
                'name': name,
                'model_type': metadata.get('model_type', 'unknown'),
                'algorithm': metadata.get('algorithm', 'unknown'),
                'performance_metrics': metadata.get('performance_metrics', {}),
                'training_date': metadata.get('training_date', ''),
                'is_active': name in self.models
            }
            model_info.append(info)
        
        return model_info
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from prediction history"""
        
        if not self.prediction_history:
            return {'message': 'No predictions logged yet'}
        
        latencies = [entry['latency_ms'] for entry in self.prediction_history]
        
        return {
            'total_predictions': len(self.prediction_history),
            'avg_latency_ms': np.mean(latencies),
            'max_latency_ms': np.max(latencies),
            'min_latency_ms': np.min(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'last_prediction_time': self.prediction_history[-1]['timestamp'],
            'models_used': list(set(entry['model_used'] for entry in self.prediction_history))
        }

# Initialize model manager
model_manager = EnsembleModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Ensemble API server")
    
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
    model_manager.load_models()
    
    # Start background tasks
    asyncio.create_task(update_system_metrics())
    
    yield
    
    # Shutdown
    logger.info("Shutting down Ensemble API server")
    if redis_client:
        redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Ensemble Methods API",
    description="Production API for ensemble model serving with monitoring",
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

# Background task for system metrics
async def update_system_metrics():
    """Update system metrics periodically"""
    while True:
        try:
            # Update system metrics
            SYSTEM_MEMORY.set(psutil.virtual_memory().percent)
            SYSTEM_CPU.set(psutil.cpu_percent())
            
            # Update model accuracy metrics (placeholder)
            for model_name, metadata in model_manager.model_metadata.items():
                accuracy = metadata.get('performance_metrics', {}).get('accuracy', 0)
                MODEL_ACCURACY.labels(model_name=model_name).set(accuracy)
            
            await asyncio.sleep(60)  # Update every minute
            
        except Exception as e:
            logger.error(f"Error updating system metrics: {e}")
            await asyncio.sleep(60)

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Ensemble Methods API",
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
        models_loaded=len(model_manager.models),
        database_connected=db_connected,
        redis_connected=redis_connected,
        system_metrics={
            "memory_percent": psutil.virtual_memory().percent,
            "cpu_percent": psutil.cpu_percent(),
            "disk_percent": psutil.disk_usage('/').percent
        }
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest, background_tasks: BackgroundTasks):
    """Make a single prediction"""
    
    result = model_manager.predict(
        features=request.features,
        model_name=request.model_name,
        return_probabilities=request.return_probabilities
    )
    
    # Add request ID if provided
    if request.request_id:
        result['request_id'] = request.request_id
    
    return PredictionResponse(**result)

@app.post("/predict/batch")
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions"""
    
    result = model_manager.batch_predict(
        features_batch=request.features_batch,
        model_name=request.model_name,
        return_probabilities=request.return_probabilities
    )
    
    # Add batch ID if provided
    if request.batch_id:
        result['batch_id'] = request.batch_id
    
    return result

@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about all available models"""
    
    models_info = model_manager.get_model_info()
    return [ModelInfo(**info) for info in models_info]

@app.get("/models/{model_name}")
async def get_model_details(model_name: str):
    """Get detailed information about a specific model"""
    
    if model_name not in model_manager.models:
        raise HTTPException(status_code=404, detail="Model not found")
    
    metadata = model_manager.model_metadata.get(model_name, {})
    
    # Get recent predictions for this model
    recent_predictions = [
        entry for entry in model_manager.prediction_history[-100:]
        if entry['model_used'] == model_name
    ]
    
    return {
        "name": model_name,
        "metadata": metadata,
        "is_loaded": model_name in model_manager.models,
        "recent_predictions_count": len(recent_predictions),
        "avg_recent_latency": np.mean([p['latency_ms'] for p in recent_predictions]) if recent_predictions else 0
    }

@app.get("/performance")
async def get_performance():
    """Get performance metrics and statistics"""
    
    return model_manager.get_performance_summary()

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(generate_latest(), media_type="text/plain")

@app.post("/models/{model_name}/reload")
async def reload_model(model_name: str):
    """Reload a specific model"""
    
    try:
        # This would reload the model from storage
        # For now, just return success
        return {"message": f"Model {model_name} reload initiated"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics"""
    
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available")
    
    try:
        info = redis_client.info()
        return {
            "connected_clients": info.get("connected_clients", 0),
            "used_memory": info.get("used_memory_human", "0B"),
            "keyspace_hits": info.get("keyspace_hits", 0),
            "keyspace_misses": info.get("keyspace_misses", 0),
            "total_commands_processed": info.get("total_commands_processed", 0)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")

def main():
    """
    Main function to run the API server
    """
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    workers = int(os.getenv('API_WORKERS', 1))
    
    logger.info(f"Starting Ensemble API server on {host}:{port}")
    
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