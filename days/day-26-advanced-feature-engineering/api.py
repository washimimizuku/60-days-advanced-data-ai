#!/usr/bin/env python3
"""
Advanced Feature Engineering API Server

Production API for feature engineering services including:
- Time series feature engineering
- NLP feature extraction
- Automated feature selection
- Feature quality monitoring
"""

import os
import sys
import time
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest
from fastapi.responses import Response

# Import feature engineering components
from solution import (
    AdvancedTimeSeriesFeatureEngineer,
    ProductionNLPFeatureEngineer,
    EnsembleFeatureSelector,
    AdvancedFeatureGenerator,
    FeatureQualityValidator
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
feature_requests = Counter('feature_requests_total', 'Total feature engineering requests', ['type', 'status'])
feature_latency = Histogram('feature_latency_seconds', 'Feature engineering latency', ['type'])
feature_quality_score = Gauge('feature_quality_score', 'Feature quality score', ['feature_set'])

# Initialize FastAPI app
app = FastAPI(
    title="Advanced Feature Engineering API",
    description="Production API for advanced feature engineering",
    version="1.0.0"
)

# Initialize feature engineering components
ts_engineer = AdvancedTimeSeriesFeatureEngineer()
nlp_engineer = ProductionNLPFeatureEngineer()
feature_selector = EnsembleFeatureSelector()
feature_generator = AdvancedFeatureGenerator()
validator = FeatureQualityValidator()

# Request/Response models
class TimeSeriesRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Time series data")
    timestamp_col: str = Field(..., description="Timestamp column name")
    target_col: str = Field(..., description="Target column for features")
    entity_col: Optional[str] = Field(None, description="Entity column for grouping")
    lags: List[int] = Field([1, 6, 24], description="Lag periods")
    windows: List[int] = Field([6, 24, 168], description="Rolling window sizes")

class NLPRequest(BaseModel):
    texts: List[str] = Field(..., description="Text data for feature extraction")
    max_tfidf_features: int = Field(100, description="Maximum TF-IDF features")
    extract_linguistic: bool = Field(True, description="Extract linguistic features")

class FeatureSelectionRequest(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="Feature data")
    target: List[float] = Field(..., description="Target variable")
    task_type: str = Field("classification", description="Task type")
    voting_threshold: float = Field(0.4, description="Ensemble voting threshold")

class FeatureResponse(BaseModel):
    features: List[Dict[str, Any]] = Field(..., description="Generated features")
    feature_names: List[str] = Field(..., description="Feature names")
    processing_time: float = Field(..., description="Processing time in seconds")
    status: str = Field(..., description="Processing status")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "components": {
            "time_series_engineer": "ready",
            "nlp_engineer": "ready",
            "feature_selector": "ready",
            "feature_generator": "ready",
            "validator": "ready"
        }
    }

@app.post("/features/time-series", response_model=FeatureResponse)
async def create_time_series_features(request: TimeSeriesRequest):
    """Create time series features from temporal data"""
    
    start_time = time.time()
    
    try:
        # Convert request data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Create temporal features
        df_temporal = ts_engineer.create_temporal_features(df, request.timestamp_col)
        
        # Create lag features if entity column provided
        if request.entity_col and request.entity_col in df.columns:
            df_temporal = ts_engineer.create_lag_features(
                df_temporal, request.target_col, request.entity_col, request.lags
            )
            
            # Create rolling features
            df_temporal = ts_engineer.create_rolling_features(
                df_temporal, request.target_col, request.entity_col, request.windows
            )
        
        processing_time = time.time() - start_time
        
        # Update metrics
        feature_requests.labels(type='time_series', status='success').inc()
        feature_latency.labels(type='time_series').observe(processing_time)
        
        return FeatureResponse(
            features=df_temporal.to_dict('records'),
            feature_names=df_temporal.columns.tolist(),
            processing_time=processing_time,
            status="success"
        )
        
    except Exception as e:
        feature_requests.labels(type='time_series', status='error').inc()
        logger.error(f"Time series feature engineering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/features/nlp", response_model=FeatureResponse)
async def create_nlp_features(request: NLPRequest):
    """Extract NLP features from text data"""
    
    start_time = time.time()
    
    try:
        features_list = []
        
        if request.extract_linguistic:
            # Extract linguistic features
            for text in request.texts:
                linguistic_features = nlp_engineer.extract_linguistic_features(text)
                features_list.append(linguistic_features)
        
        # Create TF-IDF features
        tfidf_features = nlp_engineer.create_tfidf_features(
            request.texts, max_features=request.max_tfidf_features
        )
        
        # Combine features
        if features_list:
            linguistic_df = pd.DataFrame(features_list)
            # Add TF-IDF features as additional columns
            for i in range(tfidf_features.shape[1]):
                linguistic_df[f'tfidf_{i}'] = tfidf_features[:, i]
            
            combined_features = linguistic_df.to_dict('records')
            feature_names = linguistic_df.columns.tolist()
        else:
            # Only TF-IDF features
            tfidf_df = pd.DataFrame(tfidf_features, columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
            combined_features = tfidf_df.to_dict('records')
            feature_names = tfidf_df.columns.tolist()
        
        processing_time = time.time() - start_time
        
        # Update metrics
        feature_requests.labels(type='nlp', status='success').inc()
        feature_latency.labels(type='nlp').observe(processing_time)
        
        return FeatureResponse(
            features=combined_features,
            feature_names=feature_names,
            processing_time=processing_time,
            status="success"
        )
        
    except Exception as e:
        feature_requests.labels(type='nlp', status='error').inc()
        logger.error(f"NLP feature engineering failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/features/select")
async def select_features(request: FeatureSelectionRequest):
    """Perform automated feature selection"""
    
    start_time = time.time()
    
    try:
        # Convert request data to DataFrame
        X = pd.DataFrame(request.data)
        y = pd.Series(request.target)
        
        # Initialize selector with task type
        selector = EnsembleFeatureSelector(task_type=request.task_type)
        
        # Perform ensemble selection
        selected_features = selector.ensemble_selection(
            X, y, voting_threshold=request.voting_threshold
        )
        
        processing_time = time.time() - start_time
        
        # Update metrics
        feature_requests.labels(type='selection', status='success').inc()
        feature_latency.labels(type='selection').observe(processing_time)
        
        return {
            "selected_features": selected_features,
            "original_count": len(X.columns),
            "selected_count": len(selected_features),
            "reduction_ratio": 1 - (len(selected_features) / len(X.columns)),
            "feature_scores": selector.feature_scores,
            "processing_time": processing_time,
            "status": "success"
        }
        
    except Exception as e:
        feature_requests.labels(type='selection', status='error').inc()
        logger.error(f"Feature selection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/features/generate")
async def generate_features(data: List[Dict[str, Any]]):
    """Generate new features from existing ones"""
    
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Generate features
        enhanced_df = feature_generator.generate_all_features(
            df,
            include_polynomial=False,  # Skip to avoid explosion
            include_ratios=True,
            include_binning=True,
            include_statistical=True
        )
        
        processing_time = time.time() - start_time
        
        # Update metrics
        feature_requests.labels(type='generation', status='success').inc()
        feature_latency.labels(type='generation').observe(processing_time)
        
        return {
            "features": enhanced_df.to_dict('records'),
            "feature_names": enhanced_df.columns.tolist(),
            "original_count": len(df.columns),
            "generated_count": len(enhanced_df.columns),
            "expansion_ratio": len(enhanced_df.columns) / len(df.columns),
            "processing_time": processing_time,
            "status": "success"
        }
        
    except Exception as e:
        feature_requests.labels(type='generation', status='error').inc()
        logger.error(f"Feature generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/features/validate")
async def validate_features(data: List[Dict[str, Any]]):
    """Validate feature quality"""
    
    start_time = time.time()
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Validate feature quality
        validation_results = validator.validate_feature_quality(df)
        
        processing_time = time.time() - start_time
        
        # Update quality metric
        feature_quality_score.labels(feature_set='user_data').set(validation_results['quality_score'])
        
        validation_results['processing_time'] = processing_time
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Feature validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

@app.get("/features/stats")
async def get_feature_stats():
    """Get feature engineering statistics"""
    
    try:
        # Load sample data for stats
        customers = pd.read_parquet('data/customers.parquet')
        transactions = pd.read_parquet('data/transactions.parquet')
        feedback = pd.read_parquet('data/feedback.parquet')
        
        stats = {
            "datasets": {
                "customers": {
                    "count": len(customers),
                    "features": len(customers.columns),
                    "numeric_features": len(customers.select_dtypes(include=[np.number]).columns),
                    "categorical_features": len(customers.select_dtypes(include=['object']).columns)
                },
                "transactions": {
                    "count": len(transactions),
                    "features": len(transactions.columns),
                    "date_range": {
                        "start": transactions['timestamp'].min().isoformat(),
                        "end": transactions['timestamp'].max().isoformat()
                    },
                    "total_volume": float(transactions['amount'].sum())
                },
                "feedback": {
                    "count": len(feedback),
                    "features": len(feedback.columns),
                    "avg_text_length": float(feedback['feedback_text'].str.len().mean())
                }
            },
            "feature_engineering": {
                "time_series_capabilities": [
                    "temporal_features", "lag_features", "rolling_statistics", 
                    "seasonality_decomposition"
                ],
                "nlp_capabilities": [
                    "linguistic_features", "tfidf_features", "sentiment_analysis",
                    "readability_scores"
                ],
                "selection_methods": [
                    "univariate", "mutual_information", "rfe", "lasso", "ensemble"
                ],
                "generation_methods": [
                    "polynomial", "ratios", "binning", "statistical_transforms"
                ]
            }
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get feature stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start Prometheus metrics server
    start_http_server(8001)
    
    # Start FastAPI server
    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
        log_level="info"
    )