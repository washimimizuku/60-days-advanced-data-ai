#!/usr/bin/env python3
"""
Day 31: Model Explainability - Production API Server

FastAPI server for serving model explanations with comprehensive monitoring,
caching, and production-ready features for healthcare applications.
"""

import os
import sys
import json
import time
import hashlib
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

# ML and explainability libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    import lime
    import lime.lime_tabular
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

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
EXPLANATION_COUNTER = Counter('explainability_requests_total', 'Total explanation requests', ['model_name', 'explanation_type', 'status'])
EXPLANATION_LATENCY = Histogram('explainability_request_duration_seconds', 'Explanation request latency')
CACHE_HIT_COUNTER = Counter('explainability_cache_hits_total', 'Cache hits')
CACHE_MISS_COUNTER = Counter('explainability_cache_misses_total', 'Cache misses')
ACTIVE_MODELS = Gauge('explainability_active_models', 'Number of active models')

# Pydantic models
class ExplanationRequest(BaseModel):
    """Request model for explanations"""
    model_name: str = Field(..., description="Name of the model to explain")
    instance_data: Dict[str, float] = Field(..., description="Patient data for explanation")
    explanation_type: str = Field("shap", description="Type of explanation (shap, lime, permutation)")
    num_features: int = Field(10, description="Number of top features to return")
    return_probabilities: bool = Field(True, description="Return prediction probabilities")
    patient_id: Optional[str] = Field(None, description="Patient identifier")
    
    @validator('explanation_type')
    def validate_explanation_type(cls, v):
        if v not in ['shap', 'lime', 'permutation']:
            raise ValueError("explanation_type must be one of: shap, lime, permutation")
        return v

class ClinicalExplanationRequest(BaseModel):
    """Request model for clinical explanations"""
    model_name: str = Field(..., description="Name of the model to explain")
    patient_data: Dict[str, float] = Field(..., description="Patient clinical data")
    patient_id: str = Field(..., description="Patient identifier")
    include_recommendations: bool = Field(True, description="Include clinical recommendations")

class ExplanationResponse(BaseModel):
    """Response model for explanations"""
    request_id: str
    model_name: str
    explanation_type: str
    prediction: float
    probability: Optional[float] = None
    explanations: List[Dict[str, Any]]
    computation_time_ms: float
    timestamp: str
    cached: bool = False

class ClinicalExplanationResponse(BaseModel):
    """Response model for clinical explanations"""
    patient_id: str
    risk_probability: float
    risk_level: str
    risk_color: str
    feature_contributions: List[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    models_loaded: int
    explainers_ready: int
    database_connected: bool
    redis_connected: bool

# Global variables
models_cache = {}
explainers_cache = {}
scalers_cache = {}
db_pool = None
redis_client = None

class ExplainabilityManager:
    """
    Manages explainability models and services
    """
    
    def __init__(self):
        self.models = {}
        self.explainers = {}
        self.scalers = {}
        self.feature_names = []
        self.explanation_cache = {}
        
        # Clinical mappings
        self.clinical_mappings = {
            'age': 'Patient Age',
            'length_of_stay': 'Length of Hospital Stay',
            'num_medications': 'Number of Medications',
            'charlson_comorbidity_index': 'Comorbidity Burden',
            'diabetes_severity': 'Diabetes Severity',
            'heart_failure_severity': 'Heart Failure Severity',
            'renal_disease_severity': 'Kidney Disease Severity',
            'liver_disease_severity': 'Liver Disease Severity',
            'hemoglobin_level': 'Hemoglobin Level',
            'creatinine_level': 'Creatinine Level',
            'sodium_level': 'Sodium Level',
            'glucose_level': 'Glucose Level',
            'emergency_admission': 'Emergency Admission',
            'icu_stay': 'ICU Stay',
            'surgical_procedure': 'Surgical Procedure'
        }
        
        self.risk_thresholds = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8
        }
    
    def load_models(self):
        """Load all active models and create explainers"""
        logger.info("Loading explainability models")
        
        try:
            # Connect to database
            conn = psycopg2.connect(os.getenv('DATABASE_URL'))
            cur = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get active models
            cur.execute("""
                SELECT name, model_type, algorithm, model_path, explainability_config
                FROM models 
                WHERE is_active = true
                ORDER BY name
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
                            self.feature_names = model_data.get('feature_names', [])
                        else:
                            self.models[model_name] = model_data
                            
                        # Create explainers
                        self._create_explainers(model_name, model_info['algorithm'])
                        
                        logger.info(f"Loaded model and explainers: {model_name}")
                        
                    else:
                        # Create default models if no saved models exist
                        self._create_default_model(model_name, model_info['algorithm'])
                        
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {e}")
                    continue
            
            cur.close()
            conn.close()
            
            # Update metrics
            ACTIVE_MODELS.set(len(self.models))
            
            logger.info(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            # Create default models if database fails
            self._create_default_models()
    
    def _create_default_models(self):
        """Create default models for testing"""
        logger.info("Creating default models for testing")
        
        # Generate sample data
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=15, random_state=42)
        
        self.feature_names = [
            'age', 'length_of_stay', 'num_medications', 'charlson_comorbidity_index',
            'diabetes_severity', 'heart_failure_severity', 'renal_disease_severity',
            'liver_disease_severity', 'hemoglobin_level', 'creatinine_level',
            'sodium_level', 'glucose_level', 'emergency_admission', 'icu_stay',
            'surgical_procedure'
        ]
        
        # Create models
        models_config = {
            'patient_readmission_rf': RandomForestClassifier(n_estimators=50, random_state=42),
            'patient_readmission_gb': GradientBoostingClassifier(n_estimators=50, random_state=42),
            'patient_readmission_lr': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        for name, model in models_config.items():
            if name == 'patient_readmission_lr':
                # Scale data for logistic regression
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                model.fit(X_scaled, y)
                self.scalers[name] = scaler
            else:
                model.fit(X, y)
                self.scalers[name] = None
            
            self.models[name] = model
            self._create_explainers(name, model.__class__.__name__)
    
    def _create_default_model(self, model_name: str, algorithm: str):
        """Create a default model for testing"""
        logger.info(f"Creating default {algorithm} model: {model_name}")
        
        # Generate dummy training data
        from sklearn.datasets import make_classification
        X_dummy, y_dummy = make_classification(n_samples=1000, n_features=15, random_state=42)
        
        if algorithm.lower() == 'randomforest':
            model = RandomForestClassifier(n_estimators=50, random_state=42)
            model.fit(X_dummy, y_dummy)
            scaler = None
        elif algorithm.lower() == 'gradientboosting':
            model = GradientBoostingClassifier(n_estimators=50, random_state=42)
            model.fit(X_dummy, y_dummy)
            scaler = None
        else:  # LogisticRegression
            model = LogisticRegression(random_state=42, max_iter=1000)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_dummy)
            model.fit(X_scaled, y_dummy)
        
        self.models[model_name] = model
        self.scalers[model_name] = scaler
        self.feature_names = [f'feature_{i}' for i in range(15)]
        
        self._create_explainers(model_name, algorithm)
    
    def _create_explainers(self, model_name: str, algorithm: str):
        """Create explainers for a model"""
        model = self.models[model_name]
        
        explainers = {}
        
        # SHAP explainers
        if SHAP_AVAILABLE:
            try:
                if algorithm.lower() in ['randomforest', 'gradientboosting']:
                    explainers['shap'] = shap.TreeExplainer(model)
                elif algorithm.lower() == 'logisticregression':
                    # Need background data for linear explainer
                    from sklearn.datasets import make_classification
                    X_bg, _ = make_classification(n_samples=100, n_features=15, random_state=42)
                    if self.scalers[model_name]:
                        X_bg = self.scalers[model_name].transform(X_bg)
                    explainers['shap'] = shap.LinearExplainer(model, X_bg)
                
                logger.info(f"Created SHAP explainer for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to create SHAP explainer for {model_name}: {e}")
        
        # LIME explainers
        if LIME_AVAILABLE:
            try:
                from sklearn.datasets import make_classification
                X_bg, _ = make_classification(n_samples=100, n_features=15, random_state=42)
                if self.scalers[model_name]:
                    X_bg = self.scalers[model_name].transform(X_bg)
                
                explainers['lime'] = lime.lime_tabular.LimeTabularExplainer(
                    X_bg,
                    feature_names=self.feature_names,
                    class_names=['No Readmission', 'Readmission'],
                    mode='classification'
                )
                
                logger.info(f"Created LIME explainer for {model_name}")
            except Exception as e:
                logger.warning(f"Failed to create LIME explainer for {model_name}: {e}")
        
        self.explainers[model_name] = explainers
    
    def generate_explanation(self, model_name: str, instance_data: Dict[str, float], 
                           explanation_type: str = 'shap', num_features: int = 10) -> Dict[str, Any]:
        """Generate explanation for an instance"""
        
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(model_name, instance_data, explanation_type)
            cached_result = self._get_cached_explanation(cache_key)
            
            if cached_result:
                CACHE_HIT_COUNTER.inc()
                cached_result['cached'] = True
                return cached_result
            
            CACHE_MISS_COUNTER.inc()
            
            # Get model and explainer
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")
            
            model = self.models[model_name]
            scaler = self.scalers.get(model_name)
            
            # Prepare instance data
            instance_array = np.array([list(instance_data.values())])
            
            if scaler:
                instance_array = scaler.transform(instance_array)
            
            # Make prediction
            prediction = model.predict(instance_array)[0]
            probability = model.predict_proba(instance_array)[0, 1] if hasattr(model, 'predict_proba') else None
            
            # Generate explanation
            explanations = []
            
            if explanation_type == 'shap' and model_name in self.explainers and 'shap' in self.explainers[model_name]:
                explanations = self._generate_shap_explanation(
                    model_name, instance_array, num_features
                )
            elif explanation_type == 'lime' and model_name in self.explainers and 'lime' in self.explainers[model_name]:
                explanations = self._generate_lime_explanation(
                    model_name, instance_array, num_features
                )
            elif explanation_type == 'permutation':
                explanations = self._generate_permutation_explanation(
                    model_name, instance_array, num_features
                )
            
            computation_time = (time.time() - start_time) * 1000
            
            result = {
                'prediction': float(prediction),
                'probability': float(probability) if probability is not None else None,
                'explanations': explanations,
                'computation_time_ms': computation_time,
                'cached': False
            }
            
            # Cache result
            self._cache_explanation(cache_key, result)
            
            # Update metrics
            EXPLANATION_COUNTER.labels(
                model_name=model_name,
                explanation_type=explanation_type,
                status='success'
            ).inc()
            EXPLANATION_LATENCY.observe(time.time() - start_time)
            
            return result
            
        except Exception as e:
            EXPLANATION_COUNTER.labels(
                model_name=model_name,
                explanation_type=explanation_type,
                status='error'
            ).inc()
            logger.error(f"Explanation error: {e}")
            raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")
    
    def _generate_shap_explanation(self, model_name: str, instance: np.ndarray, num_features: int) -> List[Dict[str, Any]]:
        """Generate SHAP explanation"""
        explainer = self.explainers[model_name]['shap']
        
        shap_values = explainer.shap_values(instance)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Binary classification, class 1
        
        if len(shap_values.shape) > 1:
            shap_values = shap_values[0]  # Single instance
        
        explanations = []
        for i, (feature, importance) in enumerate(zip(self.feature_names, shap_values)):
            explanations.append({
                'feature': feature,
                'clinical_name': self.clinical_mappings.get(feature, feature),
                'importance': float(importance),
                'feature_value': float(instance[0, i]),
                'impact': 'Increases Risk' if importance > 0 else 'Decreases Risk'
            })
        
        # Sort by absolute importance
        explanations.sort(key=lambda x: abs(x['importance']), reverse=True)
        
        return explanations[:num_features]
    
    def _generate_lime_explanation(self, model_name: str, instance: np.ndarray, num_features: int) -> List[Dict[str, Any]]:
        """Generate LIME explanation"""
        explainer = self.explainers[model_name]['lime']
        model = self.models[model_name]
        
        # LIME expects 1D array
        instance_1d = instance.flatten()
        
        explanation = explainer.explain_instance(
            instance_1d,
            model.predict_proba,
            num_features=num_features
        )
        
        explanations = []
        for feature_desc, importance in explanation.as_list():
            # Parse feature description
            if '=' in feature_desc:
                feature_name = feature_desc.split('=')[0].strip()
                feature_idx = int(feature_name) if feature_name.isdigit() else 0
            else:
                feature_idx = 0
            
            if feature_idx < len(self.feature_names):
                feature = self.feature_names[feature_idx]
                explanations.append({
                    'feature': feature,
                    'clinical_name': self.clinical_mappings.get(feature, feature),
                    'importance': float(importance),
                    'feature_value': float(instance_1d[feature_idx]),
                    'impact': 'Increases Risk' if importance > 0 else 'Decreases Risk'
                })
        
        return explanations
    
    def _generate_permutation_explanation(self, model_name: str, instance: np.ndarray, num_features: int) -> List[Dict[str, Any]]:
        """Generate permutation-based explanation (simplified)"""
        model = self.models[model_name]
        
        # Original prediction
        original_pred = model.predict_proba(instance)[0, 1]
        
        explanations = []
        
        # Generate background data for permutation
        from sklearn.datasets import make_classification
        X_bg, _ = make_classification(n_samples=100, n_features=len(self.feature_names), random_state=42)
        
        if self.scalers[model_name]:
            X_bg = self.scalers[model_name].transform(X_bg)
        
        for i, feature in enumerate(self.feature_names):
            # Replace feature with background mean
            instance_modified = instance.copy()
            instance_modified[0, i] = X_bg[:, i].mean()
            
            # New prediction
            new_pred = model.predict_proba(instance_modified)[0, 1]
            
            # Importance is the change in prediction
            importance = original_pred - new_pred
            
            explanations.append({
                'feature': feature,
                'clinical_name': self.clinical_mappings.get(feature, feature),
                'importance': float(importance),
                'feature_value': float(instance[0, i]),
                'impact': 'Increases Risk' if importance > 0 else 'Decreases Risk'
            })
        
        # Sort by absolute importance
        explanations.sort(key=lambda x: abs(x['importance']), reverse=True)
        
        return explanations[:num_features]
    
    def generate_clinical_explanation(self, model_name: str, patient_data: Dict[str, float], 
                                    patient_id: str) -> Dict[str, Any]:
        """Generate clinical explanation with recommendations"""
        
        # Generate base explanation
        explanation_result = self.generate_explanation(model_name, patient_data, 'shap', 10)
        
        risk_probability = explanation_result['probability']
        
        # Determine risk level
        if risk_probability < self.risk_thresholds['low']:
            risk_level = 'Low'
            risk_color = 'green'
        elif risk_probability < self.risk_thresholds['moderate']:
            risk_level = 'Moderate'
            risk_color = 'yellow'
        elif risk_probability < self.risk_thresholds['high']:
            risk_level = 'High'
            risk_color = 'orange'
        else:
            risk_level = 'Very High'
            risk_color = 'red'
        
        # Generate recommendations
        recommendations = self._generate_clinical_recommendations(
            explanation_result['explanations'], risk_level
        )
        
        return {
            'patient_id': patient_id,
            'risk_probability': risk_probability,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'feature_contributions': explanation_result['explanations'],
            'recommendations': recommendations,
            'model_used': model_name,
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_clinical_recommendations(self, explanations: List[Dict], risk_level: str) -> List[Dict[str, str]]:
        """Generate clinical recommendations based on explanations"""
        recommendations = []
        
        # Get top risk factors
        top_risk_factors = [exp for exp in explanations[:5] if exp['importance'] > 0]
        
        for factor in top_risk_factors:
            feature = factor['feature']
            value = factor['feature_value']
            
            if feature == 'length_of_stay' and value > 7:
                recommendations.append({
                    'category': 'Discharge Planning',
                    'recommendation': 'Consider enhanced discharge planning and home health services',
                    'priority': 'High'
                })
            
            elif feature == 'num_medications' and value > 10:
                recommendations.append({
                    'category': 'Medication Management',
                    'recommendation': 'Conduct medication reconciliation and consider polypharmacy review',
                    'priority': 'High'
                })
            
            elif feature == 'charlson_comorbidity_index' and value > 5:
                recommendations.append({
                    'category': 'Comorbidity Management',
                    'recommendation': 'Coordinate care with specialists for comorbidity management',
                    'priority': 'Medium'
                })
        
        # General recommendations based on risk level
        if risk_level in ['High', 'Very High']:
            recommendations.extend([
                {
                    'category': 'Follow-up Care',
                    'recommendation': 'Schedule follow-up appointment within 7 days of discharge',
                    'priority': 'High'
                },
                {
                    'category': 'Patient Education',
                    'recommendation': 'Provide comprehensive discharge education and written instructions',
                    'priority': 'High'
                }
            ])
        
        return recommendations
    
    def _generate_cache_key(self, model_name: str, instance_data: Dict, explanation_type: str) -> str:
        """Generate cache key for explanation"""
        data_str = json.dumps(instance_data, sort_keys=True)
        key_str = f"{model_name}:{explanation_type}:{data_str}"
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _get_cached_explanation(self, cache_key: str) -> Optional[Dict]:
        """Get cached explanation"""
        if redis_client:
            try:
                cached = redis_client.get(f"explanation:{cache_key}")
                if cached:
                    return json.loads(cached)
            except Exception as e:
                logger.warning(f"Cache get error: {e}")
        return None
    
    def _cache_explanation(self, cache_key: str, result: Dict):
        """Cache explanation result"""
        if redis_client:
            try:
                ttl = int(os.getenv('SHAP_CACHE_TTL', 3600))
                redis_client.setex(
                    f"explanation:{cache_key}",
                    ttl,
                    json.dumps(result)
                )
            except Exception as e:
                logger.warning(f"Cache set error: {e}")

# Initialize explainability manager
explainability_manager = ExplainabilityManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Explainability API server")
    
    # Initialize connections
    global redis_client
    try:
        redis_client = redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        redis_client = None
    
    # Load models and explainers
    explainability_manager.load_models()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Explainability API server")
    if redis_client:
        redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Model Explainability API",
    description="Production API for model explanations in healthcare applications",
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

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Model Explainability API",
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
        models_loaded=len(explainability_manager.models),
        explainers_ready=len(explainability_manager.explainers),
        database_connected=db_connected,
        redis_connected=redis_connected
    )

@app.post("/explain", response_model=ExplanationResponse)
async def explain_prediction(request: ExplanationRequest):
    """Generate explanation for a prediction"""
    
    request_id = f"req_{int(time.time() * 1000)}"
    
    try:
        result = explainability_manager.generate_explanation(
            model_name=request.model_name,
            instance_data=request.instance_data,
            explanation_type=request.explanation_type,
            num_features=request.num_features
        )
        
        return ExplanationResponse(
            request_id=request_id,
            model_name=request.model_name,
            explanation_type=request.explanation_type,
            prediction=result['prediction'],
            probability=result['probability'],
            explanations=result['explanations'],
            computation_time_ms=result['computation_time_ms'],
            timestamp=datetime.now().isoformat(),
            cached=result['cached']
        )
        
    except Exception as e:
        logger.error(f"Explanation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain/clinical", response_model=ClinicalExplanationResponse)
async def explain_clinical(request: ClinicalExplanationRequest):
    """Generate clinical explanation with recommendations"""
    
    try:
        result = explainability_manager.generate_clinical_explanation(
            model_name=request.model_name,
            patient_data=request.patient_data,
            patient_id=request.patient_id
        )
        
        return ClinicalExplanationResponse(**result)
        
    except Exception as e:
        logger.error(f"Clinical explanation request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get available models and their explainability capabilities"""
    
    models_info = []
    
    for model_name, model in explainability_manager.models.items():
        explainers = explainability_manager.explainers.get(model_name, {})
        
        info = {
            'name': model_name,
            'type': model.__class__.__name__,
            'available_explanations': list(explainers.keys()),
            'feature_count': len(explainability_manager.feature_names),
            'has_scaler': explainability_manager.scalers.get(model_name) is not None
        }
        models_info.append(info)
    
    return models_info

@app.get("/features")
async def get_features():
    """Get available features and their clinical mappings"""
    
    features = []
    for feature in explainability_manager.feature_names:
        features.append({
            'name': feature,
            'clinical_name': explainability_manager.clinical_mappings.get(feature, feature),
            'type': 'continuous'  # Simplified for this example
        })
    
    return features

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(generate_latest(), media_type="text/plain")

def main():
    """
    Main function to run the API server
    """
    
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', 8000))
    workers = int(os.getenv('API_WORKERS', 1))
    
    logger.info(f"Starting Explainability API server on {host}:{port}")
    
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