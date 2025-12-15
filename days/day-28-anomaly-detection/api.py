#!/usr/bin/env python3
"""
Anomaly Detection API
Production FastAPI server for real-time anomaly detection with multiple methods
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import asyncio
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import psycopg2
import redis
from influxdb_client import InfluxDBClient
import mlflow
import mlflow.sklearn

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class AnomalyDetectionRequest(BaseModel):
    data: List[Dict[str, Union[float, int, str]]] = Field(..., description="Input data for anomaly detection")
    method: str = Field("isolation_forest", description="Detection method: isolation_forest, one_class_svm, ensemble")
    contamination: float = Field(0.1, description="Expected proportion of anomalies", ge=0.01, le=0.5)
    threshold: float = Field(0.95, description="Anomaly threshold", ge=0.5, le=0.99)

class BatchDetectionRequest(BaseModel):
    dataset_name: str = Field(..., description="Name of the dataset to analyze")
    method: str = Field("ensemble", description="Detection method")
    contamination: float = Field(0.1, description="Expected contamination rate")

class AnomalyDetectionResponse(BaseModel):
    anomalies: List[bool]
    anomaly_scores: List[float]
    method_used: str
    total_anomalies: int
    anomaly_rate: float
    model_metrics: Dict[str, float]
    generated_at: datetime

class ModelTrainRequest(BaseModel):
    dataset_name: str
    method: str
    parameters: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]
    models_loaded: List[str]

# FastAPI app
app = FastAPI(
    title="Anomaly Detection API",
    description="Production API for real-time anomaly detection with multiple methods",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnomalyDetectionService:
    """Main anomaly detection service with multiple methods"""
    
    def __init__(self):
        self.setup_connections()
        self.models = {}
        self.scalers = {}
        self.load_pretrained_models()
        
    def setup_connections(self):
        """Setup database connections"""
        # PostgreSQL
        self.pg_conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB', 'anomaly_db'),
            user=os.getenv('POSTGRES_USER', 'anomaly_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'anomaly_pass')
        )
        
        # Redis
        self.redis_client = redis.Redis(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0)),
            decode_responses=True
        )
        
        # InfluxDB
        self.influx_client = InfluxDBClient(
            url=f"http://{os.getenv('INFLUXDB_HOST', 'localhost')}:8086",
            token=os.getenv('INFLUXDB_TOKEN', 'anomaly-token-12345'),
            org=os.getenv('INFLUXDB_ORG', 'anomaly-org')
        )
        
        # MLflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment("anomaly_detection")
    
    def load_pretrained_models(self):
        """Load pre-trained models if available"""
        try:
            # Try to load models from MLflow or local storage
            model_names = ['isolation_forest', 'one_class_svm']
            for name in model_names:
                try:
                    model_path = f"models/{name}_model.joblib"
                    if os.path.exists(model_path):
                        self.models[name] = joblib.load(model_path)
                        logger.info(f"Loaded pre-trained {name} model")
                except Exception as e:
                    logger.warning(f"Could not load {name} model: {e}")
        except Exception as e:
            logger.warning(f"Error loading pre-trained models: {e}")
    
    def get_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Retrieve dataset from storage"""
        try:
            query = f"SELECT * FROM {dataset_name} ORDER BY timestamp"
            df = pd.read_sql(query, self.pg_conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Error retrieving dataset {dataset_name}: {e}")
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
    
    def prepare_features(self, data: Union[pd.DataFrame, List[Dict]]) -> np.ndarray:
        """Prepare features for anomaly detection"""
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Select numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'is_anomaly']
        
        if len(numeric_cols) == 0:
            raise ValueError("No numeric features found for anomaly detection")
        
        features = df[numeric_cols].fillna(0)
        return features.values
    
    def train_isolation_forest(self, X: np.ndarray, contamination: float = 0.1) -> Dict:
        """Train Isolation Forest model"""
        with mlflow.start_run(nested=True):
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100
            )
            
            model.fit(X)
            
            # Log parameters
            mlflow.log_params({
                'model_type': 'IsolationForest',
                'contamination': contamination,
                'n_estimators': 100
            })
            
            # Calculate anomaly scores
            scores = model.decision_function(X)
            predictions = model.predict(X)
            
            return {
                'model': model,
                'type': 'isolation_forest',
                'scores': scores,
                'predictions': predictions
            }
    
    def train_one_class_svm(self, X: np.ndarray, contamination: float = 0.1) -> Dict:
        """Train One-Class SVM model"""
        with mlflow.start_run(nested=True):
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            model = OneClassSVM(
                kernel='rbf',
                gamma='scale',
                nu=contamination
            )
            
            model.fit(X_scaled)
            
            mlflow.log_params({
                'model_type': 'OneClassSVM',
                'kernel': 'rbf',
                'nu': contamination
            })
            
            scores = model.decision_function(X_scaled)
            predictions = model.predict(X_scaled)
            
            return {
                'model': model,
                'scaler': scaler,
                'type': 'one_class_svm',
                'scores': scores,
                'predictions': predictions
            }
    
    def ensemble_detection(self, X: np.ndarray, contamination: float = 0.1) -> Dict:
        """Ensemble anomaly detection combining multiple methods"""
        methods = []
        
        # Isolation Forest
        try:
            iso_result = self.train_isolation_forest(X, contamination)
            methods.append(('isolation_forest', iso_result, 0.4))
        except Exception as e:
            logger.warning(f"Isolation Forest failed: {e}")
        
        # One-Class SVM
        try:
            svm_result = self.train_one_class_svm(X, contamination)
            methods.append(('one_class_svm', svm_result, 0.4))
        except Exception as e:
            logger.warning(f"One-Class SVM failed: {e}")
        
        # Statistical method (Z-score)
        try:
            from scipy import stats
            z_scores = np.abs(stats.zscore(X, axis=0, nan_policy='omit'))
            z_anomalies = (z_scores > 3).any(axis=1)
            z_scores_combined = np.mean(z_scores, axis=1)
            
            methods.append(('zscore', {
                'scores': z_scores_combined,
                'predictions': z_anomalies.astype(int) * 2 - 1  # Convert to -1/1 format
            }, 0.2))
        except Exception as e:
            logger.warning(f"Z-score method failed: {e}")
        
        if not methods:
            raise ValueError("No anomaly detection methods succeeded")
        
        # Combine predictions using weighted voting
        ensemble_scores = np.zeros(len(X))
        ensemble_predictions = np.zeros(len(X))
        total_weight = sum(weight for _, _, weight in methods)
        
        for method_name, result, weight in methods:
            # Normalize scores to [0, 1]
            scores = result['scores']
            normalized_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
            
            ensemble_scores += normalized_scores * weight
            ensemble_predictions += (result['predictions'] == -1).astype(float) * weight
        
        ensemble_scores /= total_weight
        ensemble_predictions = (ensemble_predictions / total_weight) > 0.5
        
        return {
            'type': 'ensemble',
            'scores': ensemble_scores,
            'predictions': ensemble_predictions.astype(int) * 2 - 1,  # Convert to -1/1 format
            'methods_used': [name for name, _, _ in methods]
        }
    
    def detect_anomalies(self, data: Union[pd.DataFrame, List[Dict]], 
                        method: str = 'isolation_forest', 
                        contamination: float = 0.1) -> Dict:
        """Detect anomalies using specified method"""
        
        X = self.prepare_features(data)
        
        if method == 'isolation_forest':
            result = self.train_isolation_forest(X, contamination)
        elif method == 'one_class_svm':
            result = self.train_one_class_svm(X, contamination)
        elif method == 'ensemble':
            result = self.ensemble_detection(X, contamination)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convert predictions to boolean anomalies
        anomalies = result['predictions'] == -1
        scores = result['scores']
        
        # Calculate metrics
        total_anomalies = int(np.sum(anomalies))
        anomaly_rate = float(np.mean(anomalies))
        
        return {
            'anomalies': anomalies.tolist(),
            'anomaly_scores': scores.tolist(),
            'method_used': result['type'],
            'total_anomalies': total_anomalies,
            'anomaly_rate': anomaly_rate,
            'model_metrics': {
                'contamination': contamination,
                'data_points': len(X),
                'features': X.shape[1]
            }
        }
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Evaluate anomaly detection performance"""
        try:
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            
            return {
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
        except Exception as e:
            logger.warning(f"Could not calculate evaluation metrics: {e}")
            return {}

# Global service instance
anomaly_service = AnomalyDetectionService()

# WebSocket connection manager for real-time detection
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check PostgreSQL
    try:
        cursor = anomaly_service.pg_conn.cursor()
        cursor.execute("SELECT 1")
        services["postgresql"] = "healthy"
    except:
        services["postgresql"] = "unhealthy"
    
    # Check Redis
    try:
        anomaly_service.redis_client.ping()
        services["redis"] = "healthy"
    except:
        services["redis"] = "unhealthy"
    
    # Check InfluxDB
    try:
        anomaly_service.influx_client.ping()
        services["influxdb"] = "healthy"
    except:
        services["influxdb"] = "unhealthy"
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        timestamp=datetime.now(),
        services=services,
        models_loaded=list(anomaly_service.models.keys())
    )

@app.get("/datasets")
async def list_datasets():
    """List available datasets"""
    try:
        cursor = anomaly_service.pg_conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        return {"datasets": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies_endpoint(request: AnomalyDetectionRequest):
    """Detect anomalies in provided data"""
    try:
        result = anomaly_service.detect_anomalies(
            data=request.data,
            method=request.method,
            contamination=request.contamination
        )
        
        return AnomalyDetectionResponse(
            anomalies=result['anomalies'],
            anomaly_scores=result['anomaly_scores'],
            method_used=result['method_used'],
            total_anomalies=result['total_anomalies'],
            anomaly_rate=result['anomaly_rate'],
            model_metrics=result['model_metrics'],
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch")
async def batch_detection(request: BatchDetectionRequest):
    """Batch anomaly detection on stored dataset"""
    try:
        # Get dataset
        df = anomaly_service.get_dataset(request.dataset_name)
        
        # Detect anomalies
        result = anomaly_service.detect_anomalies(
            data=df,
            method=request.method,
            contamination=request.contamination
        )
        
        # Evaluate if ground truth available
        evaluation_metrics = {}
        if 'is_anomaly' in df.columns:
            y_true = df['is_anomaly'].values
            y_pred = result['anomalies']
            evaluation_metrics = anomaly_service.evaluate_model(y_true, y_pred)
        
        return {
            "dataset_name": request.dataset_name,
            "detection_results": result,
            "evaluation_metrics": evaluation_metrics,
            "dataset_info": {
                "total_records": len(df),
                "features": df.select_dtypes(include=[np.number]).columns.tolist()
            }
        }
        
    except Exception as e:
        logger.error(f"Batch detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/realtime")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time anomaly detection"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive data from client
            data = await websocket.receive_json()
            
            # Detect anomalies
            result = anomaly_service.detect_anomalies(
                data=data.get('data', []),
                method=data.get('method', 'isolation_forest'),
                contamination=data.get('contamination', 0.1)
            )
            
            # Send results back
            await websocket.send_json({
                "type": "anomaly_detection",
                "result": result,
                "timestamp": datetime.now().isoformat()
            })
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/datasets/{dataset_name}/stats")
async def get_dataset_stats(dataset_name: str):
    """Get dataset statistics"""
    try:
        # Get cached stats from Redis
        stats = {}
        keys = anomaly_service.redis_client.keys(f"{dataset_name}:*")
        
        for key in keys:
            stat_name = key.split(':', 1)[1]
            value = anomaly_service.redis_client.get(key)
            try:
                stats[stat_name] = float(value)
            except:
                stats[stat_name] = value
        
        if not stats:
            # Fallback to database query
            df = anomaly_service.get_dataset(dataset_name)
            stats = {
                'total_records': len(df),
                'anomaly_count': int(df['is_anomaly'].sum()) if 'is_anomaly' in df.columns else 0,
                'anomaly_rate': float(df['is_anomaly'].mean()) if 'is_anomaly' in df.columns else 0
            }
        
        return {"dataset": dataset_name, "statistics": stats}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Anomaly Detection API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/datasets",
            "/detect",
            "/detect/batch",
            "/ws/realtime",
            "/datasets/{name}/stats"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=os.getenv('API_HOST', '0.0.0.0'),
        port=int(os.getenv('API_PORT', 8000)),
        workers=int(os.getenv('API_WORKERS', 1)),
        reload=True
    )