#!/usr/bin/env python3
"""
Time Series Forecasting API
Production FastAPI server for time series forecasting with multiple models
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import asyncio
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

import psycopg2
import redis
from influxdb_client import InfluxDBClient
import mlflow
import mlflow.sklearn
import mlflow.tensorflow

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ForecastRequest(BaseModel):
    series_name: str = Field(..., description="Name of the time series")
    horizon: int = Field(30, description="Forecast horizon in periods", ge=1, le=365)
    model_type: str = Field("auto", description="Model type: auto, arima, sarima, prophet")
    confidence_level: float = Field(0.95, description="Confidence level for intervals", ge=0.5, le=0.99)

class ForecastResponse(BaseModel):
    series_name: str
    model_type: str
    forecast_horizon: int
    forecast_values: List[float]
    confidence_intervals: Dict[str, List[float]]
    model_metrics: Dict[str, float]
    generated_at: datetime

class ModelTrainRequest(BaseModel):
    series_name: str
    model_type: str
    parameters: Optional[Dict] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, str]

# FastAPI app
app = FastAPI(
    title="Time Series Forecasting API",
    description="Production API for time series forecasting with multiple models",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ForecastingService:
    """Main forecasting service with multiple models"""
    
    def __init__(self):
        self.setup_connections()
        self.models = {}
        
    def setup_connections(self):
        """Setup database connections"""
        # PostgreSQL
        self.pg_conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB', 'timeseries_db'),
            user=os.getenv('POSTGRES_USER', 'forecast_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'forecast_pass')
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
            token=os.getenv('INFLUXDB_TOKEN', 'forecast-token-12345'),
            org=os.getenv('INFLUXDB_ORG', 'forecast-org')
        )
        
        # MLflow
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        mlflow.set_experiment("time_series_forecasting")
    
    def get_time_series_data(self, series_name: str) -> pd.DataFrame:
        """Retrieve time series data from storage"""
        try:
            # Try PostgreSQL first
            query = f"SELECT * FROM {series_name} ORDER BY timestamp"
            df = pd.read_sql(query, self.pg_conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            logger.error(f"Error retrieving data for {series_name}: {e}")
            raise HTTPException(status_code=404, detail=f"Series {series_name} not found")
    
    def prepare_series_for_modeling(self, df: pd.DataFrame, series_name: str) -> pd.Series:
        """Prepare time series for modeling"""
        # Determine value column based on series type
        value_columns = {
            'retail_sales': 'sales',
            'energy_consumption': 'consumption',
            'stock_prices': 'price',
            'website_traffic': 'visitors'
        }
        
        value_col = value_columns.get(series_name, df.columns[-1])
        
        if value_col not in df.columns:
            # Use first numeric column
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                raise ValueError("No numeric columns found")
            value_col = numeric_cols[0]
        
        ts = pd.Series(df[value_col].values, index=df['timestamp'])
        return ts.sort_index()
    
    def fit_arima_model(self, ts: pd.Series) -> Dict:
        """Fit ARIMA model with automatic parameter selection"""
        try:
            from pmdarima import auto_arima
            
            with mlflow.start_run(nested=True):
                # Auto ARIMA
                model = auto_arima(
                    ts,
                    start_p=0, start_q=0, max_p=5, max_q=5,
                    seasonal=False,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                
                # Log parameters
                mlflow.log_params({
                    'model_type': 'ARIMA',
                    'order': str(model.order),
                    'aic': model.aic()
                })
                
                return {
                    'model': model,
                    'type': 'ARIMA',
                    'order': model.order,
                    'aic': model.aic()
                }
        except Exception as e:
            logger.error(f"ARIMA fitting error: {e}")
            raise
    
    def fit_sarima_model(self, ts: pd.Series) -> Dict:
        """Fit SARIMA model"""
        try:
            from pmdarima import auto_arima
            
            with mlflow.start_run(nested=True):
                # Auto SARIMA with seasonality
                model = auto_arima(
                    ts,
                    start_p=0, start_q=0, max_p=3, max_q=3,
                    start_P=0, start_Q=0, max_P=2, max_Q=2,
                    seasonal=True, m=12,  # Assume monthly seasonality
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore'
                )
                
                mlflow.log_params({
                    'model_type': 'SARIMA',
                    'order': str(model.order),
                    'seasonal_order': str(model.seasonal_order),
                    'aic': model.aic()
                })
                
                return {
                    'model': model,
                    'type': 'SARIMA',
                    'order': model.order,
                    'seasonal_order': model.seasonal_order,
                    'aic': model.aic()
                }
        except Exception as e:
            logger.error(f"SARIMA fitting error: {e}")
            raise
    
    def fit_prophet_model(self, ts: pd.Series) -> Dict:
        """Fit Prophet model"""
        if not PROPHET_AVAILABLE:
            raise HTTPException(status_code=400, detail="Prophet not available")
        
        try:
            with mlflow.start_run(nested=True):
                # Prepare data for Prophet
                df = pd.DataFrame({
                    'ds': ts.index,
                    'y': ts.values
                })
                
                # Create and fit model
                model = Prophet(
                    yearly_seasonality=True,
                    weekly_seasonality=True,
                    daily_seasonality=False
                )
                model.fit(df)
                
                mlflow.log_params({
                    'model_type': 'Prophet',
                    'yearly_seasonality': True,
                    'weekly_seasonality': True
                })
                
                return {
                    'model': model,
                    'type': 'Prophet',
                    'components': ['trend', 'yearly', 'weekly']
                }
        except Exception as e:
            logger.error(f"Prophet fitting error: {e}")
            raise
    
    def select_best_model(self, ts: pd.Series) -> Dict:
        """Automatically select best model based on validation"""
        models = []
        
        # Split data for validation
        train_size = int(len(ts) * 0.8)
        train_ts = ts[:train_size]
        test_ts = ts[train_size:]
        
        # Try different models
        try:
            arima_model = self.fit_arima_model(train_ts)
            arima_forecast = arima_model['model'].predict(n_periods=len(test_ts))
            arima_mae = np.mean(np.abs(test_ts.values - arima_forecast))
            models.append(('ARIMA', arima_model, arima_mae))
        except:
            pass
        
        try:
            sarima_model = self.fit_sarima_model(train_ts)
            sarima_forecast = sarima_model['model'].predict(n_periods=len(test_ts))
            sarima_mae = np.mean(np.abs(test_ts.values - sarima_forecast))
            models.append(('SARIMA', sarima_model, sarima_mae))
        except:
            pass
        
        if PROPHET_AVAILABLE:
            try:
                prophet_model = self.fit_prophet_model(train_ts)
                future = prophet_model['model'].make_future_dataframe(periods=len(test_ts), freq='D')
                prophet_forecast = prophet_model['model'].predict(future)
                prophet_mae = np.mean(np.abs(test_ts.values - prophet_forecast['yhat'].tail(len(test_ts)).values))
                models.append(('Prophet', prophet_model, prophet_mae))
            except:
                pass
        
        if not models:
            raise HTTPException(status_code=500, detail="No models could be fitted")
        
        # Select best model (lowest MAE)
        best_model = min(models, key=lambda x: x[2])
        logger.info(f"Selected {best_model[0]} model with MAE: {best_model[2]:.2f}")
        
        return best_model[1]
    
    def generate_forecast(self, model_info: Dict, horizon: int, confidence_level: float) -> Dict:
        """Generate forecast with confidence intervals"""
        model = model_info['model']
        model_type = model_info['type']
        
        if model_type in ['ARIMA', 'SARIMA']:
            # pmdarima forecast
            forecast, conf_int = model.predict(n_periods=horizon, return_conf_int=True, alpha=1-confidence_level)
            
            return {
                'forecast': forecast.tolist(),
                'lower_bound': conf_int[:, 0].tolist(),
                'upper_bound': conf_int[:, 1].tolist()
            }
        
        elif model_type == 'Prophet':
            # Prophet forecast
            future = model.make_future_dataframe(periods=horizon, freq='D')
            forecast = model.predict(future)
            
            forecast_values = forecast['yhat'].tail(horizon).tolist()
            lower_bound = forecast['yhat_lower'].tail(horizon).tolist()
            upper_bound = forecast['yhat_upper'].tail(horizon).tolist()
            
            return {
                'forecast': forecast_values,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")

# Global service instance
forecasting_service = ForecastingService()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    services = {}
    
    # Check PostgreSQL
    try:
        cursor = forecasting_service.pg_conn.cursor()
        cursor.execute("SELECT 1")
        services["postgresql"] = "healthy"
    except:
        services["postgresql"] = "unhealthy"
    
    # Check Redis
    try:
        forecasting_service.redis_client.ping()
        services["redis"] = "healthy"
    except:
        services["redis"] = "unhealthy"
    
    # Check InfluxDB
    try:
        forecasting_service.influx_client.ping()
        services["influxdb"] = "healthy"
    except:
        services["influxdb"] = "unhealthy"
    
    return HealthResponse(
        status="healthy" if all(s == "healthy" for s in services.values()) else "degraded",
        timestamp=datetime.now(),
        services=services
    )

@app.get("/series")
async def list_series():
    """List available time series"""
    try:
        cursor = forecasting_service.pg_conn.cursor()
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        tables = [row[0] for row in cursor.fetchall()]
        return {"series": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    """Generate time series forecast"""
    try:
        # Get data
        df = forecasting_service.get_time_series_data(request.series_name)
        ts = forecasting_service.prepare_series_for_modeling(df, request.series_name)
        
        # Select and fit model
        if request.model_type == "auto":
            model_info = forecasting_service.select_best_model(ts)
        elif request.model_type == "arima":
            model_info = forecasting_service.fit_arima_model(ts)
        elif request.model_type == "sarima":
            model_info = forecasting_service.fit_sarima_model(ts)
        elif request.model_type == "prophet":
            model_info = forecasting_service.fit_prophet_model(ts)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        # Generate forecast
        forecast_result = forecasting_service.generate_forecast(
            model_info, request.horizon, request.confidence_level
        )
        
        # Calculate metrics
        metrics = {
            'model_type': model_info['type'],
            'data_points': len(ts),
            'forecast_horizon': request.horizon
        }
        
        if 'aic' in model_info:
            metrics['aic'] = model_info['aic']
        
        return ForecastResponse(
            series_name=request.series_name,
            model_type=model_info['type'],
            forecast_horizon=request.horizon,
            forecast_values=forecast_result['forecast'],
            confidence_intervals={
                'lower': forecast_result['lower_bound'],
                'upper': forecast_result['upper_bound']
            },
            model_metrics=metrics,
            generated_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/series/{series_name}/data")
async def get_series_data(series_name: str, limit: int = 100):
    """Get time series data"""
    try:
        df = forecasting_service.get_time_series_data(series_name)
        
        if limit:
            df = df.tail(limit)
        
        return {
            "series_name": series_name,
            "data_points": len(df),
            "data": df.to_dict('records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Time Series Forecasting API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/series",
            "/forecast",
            "/series/{series_name}/data"
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