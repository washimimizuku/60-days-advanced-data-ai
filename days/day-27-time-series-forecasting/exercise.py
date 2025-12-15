"""
Day 27: Time Series Forecasting - ARIMA, Prophet & Neural Networks - Exercise

Business Scenario:
You're the Lead Data Scientist at "Global Supply Chain Analytics", a company that provides 
demand forecasting solutions for retail and manufacturing clients. You need to build a 
comprehensive forecasting system that handles multiple time series patterns including:
- Seasonal retail demand with holiday effects
- Manufacturing capacity planning with trend analysis
- Inventory optimization with multi-step forecasting
- Supply chain disruption prediction and recovery planning

Your mission is to implement a production-ready forecasting system that combines classical 
statistical methods, modern Prophet forecasting, and neural network approaches to provide 
robust demand predictions for various business scenarios.

Requirements:
1. Implement ARIMA and SARIMA with automatic parameter selection
2. Build Prophet models with seasonality, holidays, and external regressors
3. Create LSTM-based forecasting for complex non-linear patterns
4. Develop ensemble methods combining multiple forecasting approaches
5. Build automated forecasting pipeline with model selection and validation
6. Add comprehensive evaluation metrics and confidence intervals

Success Criteria:
- Forecasting system handles multiple time series patterns accurately
- Automatic model selection chooses best approach for each series
- Ensemble methods improve forecast accuracy over individual models
- Production pipeline processes forecasts efficiently at scale
- Comprehensive evaluation provides actionable business insights
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Statistical forecasting libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Statsmodels not available. Install with: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

# Prophet forecasting
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not available. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

# Neural network forecasting
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import MinMaxScaler
    TF_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Install with: pip install tensorflow")
    TF_AVAILABLE = False

# Core ML libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# =============================================================================
# EXERCISE 1: CLASSICAL TIME SERIES FORECASTING (ARIMA/SARIMA)
# =============================================================================

class ClassicalForecaster:
    """Advanced ARIMA and SARIMA forecasting with automatic parameter selection"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.is_fitted = False
    
    def check_stationarity(self, ts: pd.Series) -> Dict[str, Any]:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        
        if not STATSMODELS_AVAILABLE:
            return {'is_stationary': False, 'p_value': 1.0}
        
        result = adfuller(ts.dropna())
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def auto_arima_selection(self, ts: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3) -> Dict[str, Any]:
        """Automatic ARIMA parameter selection using grid search and AIC"""
        
        if not STATSMODELS_AVAILABLE:
            return {'best_order': (1, 1, 1), 'best_aic': np.inf}
        
        best_aic = np.inf
        best_order = None
        best_model = None
        
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            best_model = fitted_model
                    except:
                        continue
        
        return {
            'best_order': best_order or (1, 1, 1),
            'best_aic': best_aic,
            'best_model': best_model
        }
    
    def auto_sarima_selection(self, ts: pd.Series, seasonal_period: int = 12) -> Dict[str, Any]:
        """Automatic SARIMA parameter selection"""
        
        # TODO: Implement automatic SARIMA parameter selection
        # HINT: Grid search over both non-seasonal and seasonal parameters
        # Consider computational efficiency with reduced parameter space
        
        if not STATSMODELS_AVAILABLE:
            return {'best_order': (1, 1, 1), 'best_seasonal_order': (1, 1, 1, 12)}
        
        # TODO: SARIMA grid search
        # Reduced parameter space for efficiency
        # p_values = [0, 1, 2]
        # d_values = [0, 1]
        # q_values = [0, 1, 2]
        # P_values = [0, 1]
        # D_values = [0, 1]
        # Q_values = [0, 1]
        
        return {
            'best_order': (1, 1, 1),
            'best_seasonal_order': (1, 1, 1, 12),
            'best_aic': np.inf,
            'best_model': None
        }
    
    def fit(self, ts: pd.Series, seasonal_period: int = None) -> 'ClassicalForecaster':
        """Fit both ARIMA and SARIMA models and select the best"""
        
        print("Fitting classical forecasting models...")
        
        arima_result = self.auto_arima_selection(ts)
        self.models['arima'] = arima_result
        
        if seasonal_period:
            sarima_result = self.auto_sarima_selection(ts, seasonal_period)
            self.models['sarima'] = sarima_result
            
            # Select best model based on AIC
            if sarima_result['best_aic'] < arima_result['best_aic']:
                self.best_model = sarima_result['best_model']
            else:
                self.best_model = arima_result['best_model']
        else:
            self.best_model = arima_result['best_model']
        
        self.is_fitted = True
        return self
    
    def forecast(self, steps: int, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Generate forecasts with confidence intervals"""
        
        if not self.is_fitted or self.best_model is None:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.best_model.forecast(steps=steps, alpha=1-confidence_level)
        conf_int = self.best_model.get_forecast(steps=steps, alpha=1-confidence_level).conf_int()
        
        return {
            'forecast': forecast_result,
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1],
            'model_type': 'arima'
        }

# TODO: Test the ClassicalForecaster
def test_classical_forecasting():
    """Test classical forecasting methods"""
    
    print("🧪 Testing Classical Forecasting (ARIMA/SARIMA)...")
    
    # TODO: Create sample time series data
    # HINT: Generate data with trend, seasonality, and noise
    
    # Sample structure:
    # dates = pd.date_range('2020-01-01', periods=365*2, freq='D')
    # trend = np.linspace(100, 200, len(dates))
    # seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    # noise = np.random.normal(0, 5, len(dates))
    # ts_data = trend + seasonal + noise
    # ts = pd.Series(ts_data, index=dates)
    
    # TODO: Test classical forecasting
    # forecaster = ClassicalForecaster()
    # forecaster.fit(ts, seasonal_period=365)
    # forecast_result = forecaster.forecast(steps=30)
    
    print("✅ Classical forecasting test completed")

# =============================================================================
# EXERCISE 2: FACEBOOK PROPHET FORECASTING
# =============================================================================

class ProphetForecaster:
    """Advanced Prophet forecasting with seasonality and external regressors"""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.regressors = []
    
    def setup_model(self, 
                   growth: str = 'linear',
                   seasonality_mode: str = 'additive',
                   yearly_seasonality: bool = True,
                   weekly_seasonality: bool = True,
                   daily_seasonality: bool = False) -> 'ProphetForecaster':
        """Setup Prophet model with configuration"""
        
        if not PROPHET_AVAILABLE:
            print("Prophet not available, using placeholder")
            return self
        
        self.model = Prophet(
            growth=growth,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        
        return self
    
    def add_holidays(self, country: str = 'US') -> 'ProphetForecaster':
        """Add country-specific holidays"""
        
        if PROPHET_AVAILABLE and self.model:
            self.model.add_country_holidays(country_name=country)
        
        return self
    
    def add_custom_seasonality(self, name: str, period: float, fourier_order: int) -> 'ProphetForecaster':
        """Add custom seasonality patterns"""
        
        # TODO: Add custom seasonality
        # HINT: Use add_seasonality method
        # Examples: monthly (30.5 days), quarterly (91.25 days)
        
        if PROPHET_AVAILABLE and self.model:
            # self.model.add_seasonality(name=name, period=period, fourier_order=fourier_order)
            pass
        
        return self
    
    def add_regressor(self, name: str, prior_scale: float = 10.0) -> 'ProphetForecaster':
        """Add external regressor"""
        
        # TODO: Add external regressor
        # HINT: Use add_regressor method and track regressor names
        
        if PROPHET_AVAILABLE and self.model:
            # self.model.add_regressor(name, prior_scale=prior_scale)
            # self.regressors.append(name)
            pass
        
        return self
    
    def fit(self, df: pd.DataFrame) -> 'ProphetForecaster':
        """Fit Prophet model"""
        
        if not PROPHET_AVAILABLE or self.model is None:
            print("Prophet not available")
            return self
        
        required_cols = ['ds', 'y']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        df_clean = df.copy()
        df_clean['ds'] = pd.to_datetime(df_clean['ds'])
        self.model.fit(df_clean)
        self.is_fitted = True
        
        return self
    
    def forecast(self, periods: int, freq: str = 'D', future_regressors: pd.DataFrame = None) -> pd.DataFrame:
        """Generate Prophet forecasts"""
        
        if not PROPHET_AVAILABLE or not self.is_fitted:
            dates = pd.date_range(start=datetime.now(), periods=periods, freq=freq)
            return pd.DataFrame({
                'ds': dates,
                'yhat': np.zeros(periods),
                'yhat_lower': np.zeros(periods),
                'yhat_upper': np.zeros(periods)
            })
        
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        if future_regressors is not None and self.regressors:
            for regressor in self.regressors:
                if regressor in future_regressors.columns:
                    future[regressor] = future_regressors[regressor]
        
        forecast = self.model.predict(future)
        return forecast
    
    def detect_changepoints(self) -> Dict[str, Any]:
        """Detect significant changepoints in the time series"""
        
        # TODO: Implement changepoint detection
        # HINT: Analyze Prophet's changepoint detection results
        # Identify significant changes in trend
        
        if not PROPHET_AVAILABLE or not self.is_fitted:
            return {'changepoints': [], 'significant_changepoints': []}
        
        # TODO: Get changepoints from fitted model
        # changepoints = self.model.changepoints
        # changepoint_effects = self.model.params['delta'].mean(axis=0)
        
        return {
            'changepoints': [],
            'significant_changepoints': []
        }

# TODO: Test the ProphetForecaster
def test_prophet_forecasting():
    """Test Prophet forecasting"""
    
    print("🧪 Testing Prophet Forecasting...")
    
    # TODO: Create sample data for Prophet
    # HINT: Create DataFrame with 'ds' and 'y' columns
    
    # TODO: Test Prophet forecasting
    # prophet_forecaster = ProphetForecaster()
    # prophet_forecaster.setup_model()
    # prophet_forecaster.add_holidays('US')
    # prophet_forecaster.fit(prophet_df)
    # forecast = prophet_forecaster.forecast(periods=30)
    
    print("✅ Prophet forecasting test completed")

# =============================================================================
# EXERCISE 3: NEURAL NETWORK FORECASTING (LSTM)
# =============================================================================

class LSTMForecaster:
    """Advanced LSTM forecasting with sequence modeling"""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2):
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = None
        self.is_fitted = False
    
    def prepare_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        
        # TODO: Create sequences for LSTM
        # HINT: Create sliding windows of sequence_length
        # Return X (sequences) and y (targets)
        
        X, y = [], []
        
        # TODO: Create sequences
        # for i in range(self.sequence_length, len(data)):
        #     X.append(data[i-self.sequence_length:i])
        #     y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM model architecture"""
        
        # TODO: Build LSTM model
        # HINT: Use Sequential model with LSTM layers
        # Add dropout for regularization
        # Include dense layers for output
        
        if not TF_AVAILABLE:
            print("TensorFlow not available")
            return
        
        # TODO: Create model architecture
        # self.model = Sequential([
        #     Input(shape=input_shape),
        #     LSTM(self.lstm_units, return_sequences=True),
        #     Dropout(self.dropout_rate),
        #     LSTM(self.lstm_units),
        #     Dropout(self.dropout_rate),
        #     Dense(25),
        #     Dense(1)
        # ])
        
        # TODO: Compile model
        # self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    def fit(self, ts: pd.Series, 
            validation_split: float = 0.2,
            epochs: int = 50,
            batch_size: int = 32) -> 'LSTMForecaster':
        """Fit LSTM model to time series"""
        
        # TODO: Implement LSTM training
        # HINT: Scale data, create sequences, train model
        # Use validation split and early stopping
        
        if not TF_AVAILABLE:
            print("TensorFlow not available")
            return self
        
        # TODO: Scale data
        # from sklearn.preprocessing import MinMaxScaler
        # self.scaler = MinMaxScaler()
        # scaled_data = self.scaler.fit_transform(ts.values.reshape(-1, 1))
        
        # TODO: Create sequences
        # X, y = self.prepare_sequences(scaled_data.flatten())
        # X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # TODO: Build and train model
        # self.build_model((X.shape[1], 1))
        # self.model.fit(X, y, validation_split=validation_split, epochs=epochs, batch_size=batch_size)
        # self.is_fitted = True
        
        return self
    
    def forecast(self, ts: pd.Series, steps: int) -> np.ndarray:
        """Generate LSTM forecasts"""
        
        # TODO: Generate forecasts using LSTM
        # HINT: Use last sequence to predict iteratively
        # Scale inputs and inverse transform outputs
        
        if not TF_AVAILABLE or not self.is_fitted:
            return np.zeros(steps)
        
        # TODO: Prepare last sequence
        # scaled_data = self.scaler.transform(ts.values.reshape(-1, 1))
        # last_sequence = scaled_data[-self.sequence_length:].flatten()
        
        # TODO: Generate forecasts iteratively
        # forecasts = []
        # current_sequence = last_sequence.copy()
        # for _ in range(steps):
        #     X_pred = current_sequence.reshape((1, self.sequence_length, 1))
        #     next_pred = self.model.predict(X_pred)[0, 0]
        #     forecasts.append(next_pred)
        #     current_sequence = np.append(current_sequence[1:], next_pred)
        
        # TODO: Inverse transform
        # forecasts = self.scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
        
        return np.zeros(steps)  # Placeholder

# TODO: Test the LSTMForecaster
def test_lstm_forecasting():
    """Test LSTM forecasting"""
    
    print("🧪 Testing LSTM Forecasting...")
    
    # TODO: Create sample data for LSTM
    # TODO: Test LSTM forecasting
    # lstm_forecaster = LSTMForecaster(sequence_length=30)
    # lstm_forecaster.fit(sample_ts, epochs=10)
    # forecast = lstm_forecaster.forecast(sample_ts, steps=30)
    
    print("✅ LSTM forecasting test completed")

# =============================================================================
# EXERCISE 4: ENSEMBLE FORECASTING
# =============================================================================

class EnsembleForecaster:
    """Ensemble forecasting combining multiple methods"""
    
    def __init__(self):
        self.forecasters = {}
        self.weights = {}
        self.is_fitted = False
    
    def add_forecaster(self, name: str, forecaster: Any, weight: float = 1.0):
        """Add forecaster to ensemble"""
        
        # TODO: Add forecaster to ensemble
        # HINT: Store forecaster and its weight
        
        self.forecasters[name] = forecaster
        self.weights[name] = weight
    
    def fit(self, ts: pd.Series, **kwargs) -> 'EnsembleForecaster':
        """Fit all forecasters in ensemble"""
        
        # TODO: Fit all forecasters
        # HINT: Iterate through forecasters and fit each one
        # Handle different fitting requirements for each method
        
        print("Fitting ensemble forecasters...")
        
        for name, forecaster in self.forecasters.items():
            try:
                print(f"Fitting {name}...")
                # TODO: Fit individual forecaster
                # Handle different fit signatures for different forecasters
                
                if hasattr(forecaster, 'fit'):
                    forecaster.fit(ts, **kwargs)
                
            except Exception as e:
                print(f"Failed to fit {name}: {e}")
        
        self.is_fitted = True
        return self
    
    def forecast(self, steps: int, **kwargs) -> Dict[str, Any]:
        """Generate ensemble forecasts"""
        
        # TODO: Generate ensemble forecasts
        # HINT: Get forecasts from all methods and combine using weights
        # Handle different forecast formats from different methods
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before forecasting")
        
        individual_forecasts = {}
        valid_forecasts = []
        valid_weights = []
        
        # TODO: Get forecasts from each method
        for name, forecaster in self.forecasters.items():
            try:
                # TODO: Generate forecast from individual method
                # forecast = forecaster.forecast(steps, **kwargs)
                # individual_forecasts[name] = forecast
                # valid_forecasts.append(forecast_values)
                # valid_weights.append(self.weights[name])
                pass
            except Exception as e:
                print(f"Failed to get forecast from {name}: {e}")
        
        # TODO: Combine forecasts using weighted average
        if valid_forecasts:
            # ensemble_forecast = np.average(valid_forecasts, axis=0, weights=valid_weights)
            ensemble_forecast = np.zeros(steps)  # Placeholder
        else:
            ensemble_forecast = np.zeros(steps)
        
        return {
            'ensemble_forecast': ensemble_forecast,
            'individual_forecasts': individual_forecasts,
            'weights_used': dict(zip(self.forecasters.keys(), valid_weights)) if valid_forecasts else {}
        }
    
    def evaluate_forecasters(self, ts: pd.Series, test_size: int = 30) -> Dict[str, Dict[str, float]]:
        """Evaluate individual forecasters on test set"""
        
        # TODO: Implement forecaster evaluation
        # HINT: Split data, fit on train, forecast on test, calculate metrics
        
        train_ts = ts[:-test_size]
        test_ts = ts[-test_size:]
        
        evaluation_results = {}
        
        for name, forecaster in self.forecasters.items():
            try:
                # TODO: Fit on training data and forecast
                # forecaster.fit(train_ts)
                # forecast = forecaster.forecast(test_size)
                
                # TODO: Calculate evaluation metrics
                # mae = mean_absolute_error(test_ts.values, forecast)
                # mse = mean_squared_error(test_ts.values, forecast)
                # rmse = np.sqrt(mse)
                
                evaluation_results[name] = {
                    'mae': 0.0,  # Placeholder
                    'mse': 0.0,
                    'rmse': 0.0
                }
                
            except Exception as e:
                print(f"Failed to evaluate {name}: {e}")
                evaluation_results[name] = {'mae': np.inf, 'mse': np.inf, 'rmse': np.inf}
        
        return evaluation_results

# TODO: Test the EnsembleForecaster
def test_ensemble_forecasting():
    """Test ensemble forecasting"""
    
    print("🧪 Testing Ensemble Forecasting...")
    
    # TODO: Create ensemble with multiple forecasters
    # ensemble = EnsembleForecaster()
    # ensemble.add_forecaster('arima', ClassicalForecaster(), weight=0.4)
    # ensemble.add_forecaster('prophet', ProphetForecaster(), weight=0.4)
    # ensemble.add_forecaster('lstm', LSTMForecaster(), weight=0.2)
    
    # TODO: Test ensemble forecasting
    # ensemble.fit(sample_ts)
    # ensemble_result = ensemble.forecast(steps=30)
    
    print("✅ Ensemble forecasting test completed")

# =============================================================================
# EXERCISE 5: PRODUCTION FORECASTING PIPELINE
# =============================================================================

class ProductionForecastingPipeline:
    """Production-ready forecasting pipeline with automated model selection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.best_model = None
        self.model_performance = {}
        self.is_fitted = False
    
    def detect_patterns(self, ts: pd.Series) -> Dict[str, Any]:
        """Detect time series patterns to guide model selection"""
        
        # TODO: Implement pattern detection
        # HINT: Detect trend, seasonality, stationarity
        # Use statistical tests and decomposition
        
        patterns = {
            'has_trend': False,
            'has_seasonality': False,
            'is_stationary': False,
            'seasonal_period': None,
            'trend_strength': 0.0,
            'seasonal_strength': 0.0
        }
        
        # TODO: Detect trend
        # Use linear regression or Mann-Kendall test
        
        # TODO: Detect seasonality
        # Use FFT or autocorrelation analysis
        
        # TODO: Check stationarity
        # Use ADF test
        
        return patterns
    
    def select_best_model(self, ts: pd.Series, patterns: Dict[str, Any]) -> str:
        """Select best forecasting model based on data patterns"""
        
        # TODO: Implement model selection logic
        # HINT: Use patterns to choose appropriate model
        # Consider data size, seasonality, trend, etc.
        
        # Decision logic examples:
        # - Strong seasonality + holidays -> Prophet
        # - Stationary with clear patterns -> ARIMA
        # - Complex non-linear patterns -> LSTM
        # - Multiple patterns -> Ensemble
        
        if patterns['has_seasonality'] and len(ts) > 365:
            return 'prophet'
        elif patterns['is_stationary']:
            return 'arima'
        elif len(ts) > 500:
            return 'lstm'
        else:
            return 'ensemble'
    
    def fit(self, ts: pd.Series) -> 'ProductionForecastingPipeline':
        """Fit production forecasting pipeline"""
        
        # TODO: Implement production pipeline fitting
        # HINT: Detect patterns, select model, fit chosen model
        
        print("Fitting production forecasting pipeline...")
        
        # TODO: Detect patterns
        patterns = self.detect_patterns(ts)
        print(f"Detected patterns: {patterns}")
        
        # TODO: Select best model
        best_model_name = self.select_best_model(ts, patterns)
        print(f"Selected model: {best_model_name}")
        
        # TODO: Fit selected model
        if best_model_name == 'arima':
            self.best_model = ClassicalForecaster()
        elif best_model_name == 'prophet':
            self.best_model = ProphetForecaster()
        elif best_model_name == 'lstm':
            self.best_model = LSTMForecaster()
        else:
            self.best_model = EnsembleForecaster()
        
        # TODO: Fit the selected model
        # self.best_model.fit(ts)
        self.is_fitted = True
        
        return self
    
    def forecast(self, steps: int, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Generate production forecasts"""
        
        # TODO: Generate production forecasts
        # HINT: Use fitted model to generate forecasts
        # Include confidence intervals and metadata
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before forecasting")
        
        # TODO: Generate forecast
        # forecast_result = self.best_model.forecast(steps)
        
        return {
            'forecast': np.zeros(steps),  # Placeholder
            'confidence_intervals': {
                'lower': np.zeros(steps),
                'upper': np.zeros(steps)
            },
            'model_used': type(self.best_model).__name__,
            'forecast_horizon': steps,
            'confidence_level': confidence_level
        }
    
    def validate_forecast(self, ts: pd.Series, test_size: int = 30) -> Dict[str, float]:
        """Validate forecasting performance"""
        
        # TODO: Implement forecast validation
        # HINT: Use time series cross-validation
        # Calculate multiple evaluation metrics
        
        # TODO: Split data for validation
        train_ts = ts[:-test_size]
        test_ts = ts[-test_size:]
        
        # TODO: Fit on training data and forecast
        # self.fit(train_ts)
        # forecast_result = self.forecast(test_size)
        
        # TODO: Calculate metrics
        metrics = {
            'mae': 0.0,  # Placeholder
            'mse': 0.0,
            'rmse': 0.0,
            'mape': 0.0,
            'smape': 0.0
        }
        
        return metrics

# TODO: Test the ProductionForecastingPipeline
def test_production_pipeline():
    """Test production forecasting pipeline"""
    
    print("🧪 Testing Production Forecasting Pipeline...")
    
    # TODO: Test production pipeline
    # config = {'model_selection': 'auto', 'validation_size': 30}
    # pipeline = ProductionForecastingPipeline(config)
    # pipeline.fit(sample_ts)
    # forecast_result = pipeline.forecast(steps=30)
    # validation_metrics = pipeline.validate_forecast(sample_ts)
    
    print("✅ Production pipeline test completed")

# =============================================================================
# EXERCISE 6: COMPREHENSIVE EVALUATION AND VISUALIZATION
# =============================================================================

class ForecastEvaluator:
    """Comprehensive forecast evaluation and visualization"""
    
    def __init__(self):
        self.evaluation_results = {}
    
    def calculate_metrics(self, actual: np.ndarray, forecast: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive evaluation metrics"""
        
        # TODO: Implement evaluation metrics
        # HINT: Calculate MAE, MSE, RMSE, MAPE, SMAPE
        # Handle edge cases (zero values, etc.)
        
        metrics = {}
        
        # TODO: Mean Absolute Error
        # metrics['mae'] = mean_absolute_error(actual, forecast)
        
        # TODO: Mean Squared Error
        # metrics['mse'] = mean_squared_error(actual, forecast)
        
        # TODO: Root Mean Squared Error
        # metrics['rmse'] = np.sqrt(metrics['mse'])
        
        # TODO: Mean Absolute Percentage Error
        # metrics['mape'] = np.mean(np.abs((actual - forecast) / actual)) * 100
        
        # TODO: Symmetric Mean Absolute Percentage Error
        # metrics['smape'] = np.mean(2 * np.abs(actual - forecast) / (np.abs(actual) + np.abs(forecast))) * 100
        
        return {
            'mae': 0.0,  # Placeholder
            'mse': 0.0,
            'rmse': 0.0,
            'mape': 0.0,
            'smape': 0.0
        }
    
    def plot_forecast_results(self, ts: pd.Series, forecast: np.ndarray, 
                            confidence_intervals: Dict[str, np.ndarray] = None):
        """Plot forecast results with confidence intervals"""
        
        # TODO: Create comprehensive forecast visualization
        # HINT: Plot historical data, forecast, and confidence intervals
        # Use different colors and styles for clarity
        
        plt.figure(figsize=(12, 6))
        
        # TODO: Plot historical data
        # plt.plot(ts.index, ts.values, label='Historical', color='blue')
        
        # TODO: Plot forecast
        # forecast_dates = pd.date_range(start=ts.index[-1], periods=len(forecast)+1, freq='D')[1:]
        # plt.plot(forecast_dates, forecast, label='Forecast', color='red', linestyle='--')
        
        # TODO: Plot confidence intervals if provided
        # if confidence_intervals:
        #     plt.fill_between(forecast_dates, 
        #                     confidence_intervals['lower'], 
        #                     confidence_intervals['upper'], 
        #                     alpha=0.3, color='red')
        
        plt.title('Time Series Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.show()

# TODO: Test the ForecastEvaluator
def test_forecast_evaluation():
    """Test forecast evaluation"""
    
    print("🧪 Testing Forecast Evaluation...")
    
    # TODO: Test evaluation metrics and visualization
    # evaluator = ForecastEvaluator()
    # metrics = evaluator.calculate_metrics(actual_values, forecast_values)
    # evaluator.plot_forecast_results(sample_ts, forecast_values)
    
    print("✅ Forecast evaluation test completed")

# =============================================================================
# EXERCISE 7: INTEGRATION TEST
# =============================================================================

def run_comprehensive_forecasting_test():
    """Run comprehensive test of all forecasting components"""
    
    print("🚀 Running Comprehensive Time Series Forecasting Test")
    print("=" * 60)
    
    # TODO: Create comprehensive test dataset
    print("📊 Creating comprehensive time series dataset...")
    
    # TODO: Generate synthetic time series with multiple patterns
    # HINT: Include trend, multiple seasonalities, noise, and changepoints
    
    # Sample structure:
    # np.random.seed(42)
    # dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
    # 
    # # Base trend
    # trend = np.linspace(100, 300, len(dates))
    # 
    # # Multiple seasonalities
    # yearly_seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    # weekly_seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    # 
    # # Noise and changepoints
    # noise = np.random.normal(0, 10, len(dates))
    # changepoint = len(dates) // 2
    # trend[changepoint:] += 50  # Structural break
    # 
    # ts_data = trend + yearly_seasonal + weekly_seasonal + noise
    # ts = pd.Series(ts_data, index=dates)
    
    print("✅ Test dataset created")
    
    # TODO: Test each component
    print("\n🔧 Testing individual components...")
    
    # Test 1: Classical Forecasting
    test_classical_forecasting()
    
    # Test 2: Prophet Forecasting
    test_prophet_forecasting()
    
    # Test 3: LSTM Forecasting
    test_lstm_forecasting()
    
    # Test 4: Ensemble Forecasting
    test_ensemble_forecasting()
    
    # Test 5: Production Pipeline
    test_production_pipeline()
    
    # Test 6: Evaluation
    test_forecast_evaluation()
    
    print("\n🎯 Integration Test Summary:")
    print("✅ Classical Forecasting (ARIMA/SARIMA): Implemented")
    print("✅ Prophet Forecasting: Implemented")
    print("✅ LSTM Neural Network Forecasting: Implemented")
    print("✅ Ensemble Forecasting: Implemented")
    print("✅ Production Pipeline: Implemented")
    print("✅ Comprehensive Evaluation: Implemented")
    
    print("\n🚀 All forecasting components ready for production!")

if __name__ == "__main__":
    print("🎯 Day 27: Time Series Forecasting - Exercise")
    print("Building comprehensive forecasting systems for supply chain analytics")
    print()
    
    # TODO: Run the comprehensive test
    # run_comprehensive_forecasting_test()
    
    print("💡 Complete all TODO items to build production-ready forecasting systems!")
    print("📈 Focus on ARIMA, Prophet, LSTM, and ensemble methods!")
    print("🔍 Remember to include proper evaluation and confidence intervals!")
