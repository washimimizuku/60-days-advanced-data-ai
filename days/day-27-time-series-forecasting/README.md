# Day 27: Time Series Forecasting - ARIMA, Prophet & Neural Networks

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** classical time series forecasting methods including ARIMA, SARIMA, and exponential smoothing
- **Implement** Facebook Prophet for robust forecasting with seasonality and holiday effects
- **Build** neural network-based forecasting models using LSTM and Transformer architectures
- **Deploy** production forecasting systems with automated model selection and ensemble methods
- **Apply** advanced forecasting techniques for demand prediction, financial modeling, and capacity planning

⭐ **Difficulty**: Advanced ML Engineering (1 hour)

---

## Theory

### Time Series Forecasting: From Classical to Modern Approaches

Time series forecasting is one of the most critical applications in production ML systems, powering everything from demand planning and financial modeling to capacity management and anomaly detection. Modern forecasting combines classical statistical methods with advanced machine learning techniques.

**Why Time Series Forecasting is Essential**:
- **Business Planning**: Revenue forecasting, inventory management, resource allocation
- **Operational Efficiency**: Capacity planning, load balancing, maintenance scheduling
- **Risk Management**: Financial risk modeling, fraud detection, system monitoring
- **Customer Experience**: Demand prediction, personalization, recommendation timing
- **Cost Optimization**: Energy consumption, cloud resource scaling, supply chain optimization

### Classical Time Series Methods

#### 1. ARIMA (AutoRegressive Integrated Moving Average)
ARIMA models are the foundation of classical time series forecasting, combining autoregression, differencing, and moving averages.

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class ARIMAForecaster:
    """Production-ready ARIMA forecasting implementation"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=None):
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
        self.is_fitted = False
        
    def check_stationarity(self, ts: pd.Series) -> Dict[str, Any]:
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        
        result = adfuller(ts.dropna())
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05
        }
    
    def make_stationary(self, ts: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """Make time series stationary through differencing"""
        
        diff_order = 0
        current_ts = ts.copy()
        
        for i in range(max_diff):
            stationarity = self.check_stationarity(current_ts)
            
            if stationarity['is_stationary']:
                break
                
            current_ts = current_ts.diff().dropna()
            diff_order += 1
        
        return current_ts, diff_order
    
    def auto_arima(self, ts: pd.Series, max_p: int = 5, max_d: int = 2, max_q: int = 5) -> Dict[str, Any]:
        """Automatic ARIMA order selection using AIC"""
        
        best_aic = np.inf
        best_order = None
        best_model = None
        
        # Grid search for best parameters
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
                            
                    except Exception:
                        continue
        
        return {
            'best_order': best_order,
            'best_aic': best_aic,
            'best_model': best_model
        }
    
    def fit(self, ts: pd.Series, auto_order: bool = True) -> 'ARIMAForecaster':
        """Fit ARIMA model to time series"""
        
        if auto_order:
            auto_result = self.auto_arima(ts)
            self.order = auto_result['best_order']
            self.fitted_model = auto_result['best_model']
        else:
            model = ARIMA(ts, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = model.fit()
        
        self.is_fitted = True
        return self
    
    def forecast(self, steps: int, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Generate forecasts with confidence intervals"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.forecast(steps=steps, alpha=1-confidence_level)
        conf_int = self.fitted_model.get_forecast(steps=steps, alpha=1-confidence_level).conf_int()
        
        return {
            'forecast': forecast_result,
            'lower_bound': conf_int.iloc[:, 0],
            'upper_bound': conf_int.iloc[:, 1],
            'confidence_level': confidence_level
        }
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Get comprehensive model diagnostics"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before diagnostics")
        
        return {
            'aic': self.fitted_model.aic,
            'bic': self.fitted_model.bic,
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'residuals_mean': self.fitted_model.resid.mean(),
            'residuals_std': self.fitted_model.resid.std(),
            'ljung_box_p_value': self._ljung_box_test()
        }
    
    def _ljung_box_test(self) -> float:
        """Ljung-Box test for residual autocorrelation"""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        
        try:
            result = acorr_ljungbox(self.fitted_model.resid, lags=10, return_df=True)
            return result['lb_pvalue'].iloc[-1]
        except:
            return np.nan

# Example usage
np.random.seed(42)

# Generate sample time series with trend and seasonality
dates = pd.date_range('2020-01-01', periods=365*3, freq='D')
trend = np.linspace(100, 200, len(dates))
seasonal = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
noise = np.random.normal(0, 5, len(dates))
ts_data = trend + seasonal + noise

ts = pd.Series(ts_data, index=dates)

# Fit ARIMA model
arima_forecaster = ARIMAForecaster()
arima_forecaster.fit(ts, auto_order=True)

# Generate forecasts
forecast_result = arima_forecaster.forecast(steps=30)
print("ARIMA Forecast Results:")
print(f"Next 5 days forecast: {forecast_result['forecast'].head()}")
print(f"Model diagnostics: {arima_forecaster.get_model_diagnostics()}")
```

#### 2. SARIMA (Seasonal ARIMA)
SARIMA extends ARIMA to handle seasonal patterns explicitly.

```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

class SARIMAForecaster:
    """Production SARIMA forecasting with seasonal decomposition"""
    
    def __init__(self, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.fitted_model = None
        self.is_fitted = False
    
    def decompose_seasonality(self, ts: pd.Series, period: int = None) -> Dict[str, pd.Series]:
        """Decompose time series into trend, seasonal, and residual components"""
        
        if period is None:
            # Auto-detect seasonality period
            period = self._detect_seasonality(ts)
        
        decomposition = seasonal_decompose(ts, model='additive', period=period)
        
        return {
            'trend': decomposition.trend,
            'seasonal': decomposition.seasonal,
            'residual': decomposition.resid,
            'period': period
        }
    
    def _detect_seasonality(self, ts: pd.Series) -> int:
        """Auto-detect seasonality period using FFT"""
        
        # Remove trend
        detrended = ts - ts.rolling(window=12, center=True).mean()
        detrended = detrended.dropna()
        
        # Apply FFT
        fft = np.fft.fft(detrended.values)
        frequencies = np.fft.fftfreq(len(detrended))
        
        # Find dominant frequency
        magnitude = np.abs(fft)
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
        dominant_freq = frequencies[dominant_freq_idx]
        
        # Convert to period
        period = int(1 / abs(dominant_freq)) if dominant_freq != 0 else 12
        
        return max(2, min(period, len(ts) // 3))  # Reasonable bounds
    
    def auto_sarima(self, ts: pd.Series, seasonal_period: int = 12) -> Dict[str, Any]:
        """Automatic SARIMA parameter selection"""
        
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None
        best_model = None
        
        # Reduced grid search for efficiency
        p_values = [0, 1, 2]
        d_values = [0, 1]
        q_values = [0, 1, 2]
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in [0, 1]:
                        for D in [0, 1]:
                            for Q in [0, 1]:
                                try:
                                    order = (p, d, q)
                                    seasonal_order = (P, D, Q, seasonal_period)
                                    
                                    model = SARIMAX(ts, order=order, seasonal_order=seasonal_order)
                                    fitted_model = model.fit(disp=False)
                                    
                                    if fitted_model.aic < best_aic:
                                        best_aic = fitted_model.aic
                                        best_order = order
                                        best_seasonal_order = seasonal_order
                                        best_model = fitted_model
                                        
                                except Exception:
                                    continue
        
        return {
            'best_order': best_order,
            'best_seasonal_order': best_seasonal_order,
            'best_aic': best_aic,
            'best_model': best_model
        }
    
    def fit(self, ts: pd.Series, auto_order: bool = True) -> 'SARIMAForecaster':
        """Fit SARIMA model"""
        
        if auto_order:
            auto_result = self.auto_sarima(ts)
            self.order = auto_result['best_order']
            self.seasonal_order = auto_result['best_seasonal_order']
            self.fitted_model = auto_result['best_model']
        else:
            model = SARIMAX(ts, order=self.order, seasonal_order=self.seasonal_order)
            self.fitted_model = model.fit(disp=False)
        
        self.is_fitted = True
        return self
    
    def forecast(self, steps: int, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Generate SARIMA forecasts"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        forecast_result = self.fitted_model.get_forecast(steps=steps, alpha=1-confidence_level)
        
        return {
            'forecast': forecast_result.predicted_mean,
            'lower_bound': forecast_result.conf_int().iloc[:, 0],
            'upper_bound': forecast_result.conf_int().iloc[:, 1],
            'confidence_level': confidence_level
        }

# Example usage with seasonal data
seasonal_ts = ts.resample('M').mean()  # Monthly aggregation for seasonality

sarima_forecaster = SARIMAForecaster()
sarima_forecaster.fit(seasonal_ts, auto_order=True)

seasonal_forecast = sarima_forecaster.forecast(steps=12)
print("SARIMA Seasonal Forecast:")
print(f"Next 12 months: {seasonal_forecast['forecast']}")
```

### Facebook Prophet: Robust Forecasting at Scale

Prophet is designed for business forecasting with strong seasonal patterns and holiday effects.

```python
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not available. Install with: pip install prophet")
    PROPHET_AVAILABLE = False

class ProphetForecaster:
    """Production Prophet forecasting with advanced features"""
    
    def __init__(self, 
                 growth='linear',
                 seasonality_mode='additive',
                 yearly_seasonality='auto',
                 weekly_seasonality='auto',
                 daily_seasonality='auto'):
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is required for this forecaster")
        
        self.model = Prophet(
            growth=growth,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality
        )
        self.is_fitted = False
        self.holidays = None
    
    def add_country_holidays(self, country: str = 'US'):
        """Add country-specific holidays"""
        
        self.model.add_country_holidays(country_name=country)
        return self
    
    def add_custom_seasonality(self, name: str, period: float, fourier_order: int):
        """Add custom seasonality patterns"""
        
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order
        )
        return self
    
    def add_regressor(self, name: str, prior_scale: float = 10.0):
        """Add external regressors"""
        
        self.model.add_regressor(name, prior_scale=prior_scale)
        return self
    
    def fit(self, df: pd.DataFrame) -> 'ProphetForecaster':
        """Fit Prophet model
        
        Args:
            df: DataFrame with 'ds' (date) and 'y' (value) columns
        """
        
        # Validate input format
        required_cols = ['ds', 'y']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Ensure proper date format
        df = df.copy()
        df['ds'] = pd.to_datetime(df['ds'])
        
        self.model.fit(df)
        self.is_fitted = True
        
        return self
    
    def forecast(self, periods: int, freq: str = 'D', 
                include_history: bool = True) -> pd.DataFrame:
        """Generate Prophet forecasts"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=periods, 
            freq=freq, 
            include_history=include_history
        )
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        return forecast
    
    def plot_forecast(self, forecast: pd.DataFrame, figsize: Tuple[int, int] = (12, 8)):
        """Plot forecast with components"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Main forecast plot
        fig1 = self.model.plot(forecast, figsize=figsize)
        plt.title('Time Series Forecast')
        plt.show()
        
        # Components plot
        fig2 = self.model.plot_components(forecast, figsize=figsize)
        plt.show()
        
        return fig1, fig2
    
    def cross_validate(self, df: pd.DataFrame, 
                      initial: str = '730 days',
                      period: str = '180 days', 
                      horizon: str = '365 days') -> pd.DataFrame:
        """Perform time series cross-validation"""
        
        from prophet.diagnostics import cross_validation, performance_metrics
        
        cv_results = cross_validation(
            self.model, 
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        # Calculate performance metrics
        metrics = performance_metrics(cv_results)
        
        return cv_results, metrics
    
    def detect_changepoints(self) -> Dict[str, Any]:
        """Detect and analyze changepoints in the time series"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before changepoint detection")
        
        changepoints = self.model.changepoints
        changepoint_effects = self.model.params['delta'].mean(axis=0)
        
        # Find significant changepoints
        significant_changepoints = []
        threshold = np.std(changepoint_effects) * 2
        
        for i, (cp, effect) in enumerate(zip(changepoints, changepoint_effects)):
            if abs(effect) > threshold:
                significant_changepoints.append({
                    'date': cp,
                    'effect': effect,
                    'significance': abs(effect) / threshold
                })
        
        return {
            'all_changepoints': changepoints,
            'changepoint_effects': changepoint_effects,
            'significant_changepoints': significant_changepoints,
            'threshold': threshold
        }

# Example usage with Prophet
if PROPHET_AVAILABLE:
    # Prepare data for Prophet
    prophet_df = pd.DataFrame({
        'ds': ts.index,
        'y': ts.values
    })
    
    # Create and fit Prophet model
    prophet_forecaster = ProphetForecaster()
    prophet_forecaster.add_country_holidays('US')
    prophet_forecaster.add_custom_seasonality('monthly', 30.5, 5)
    
    prophet_forecaster.fit(prophet_df)
    
    # Generate forecast
    prophet_forecast = prophet_forecaster.forecast(periods=90, freq='D')
    
    print("Prophet Forecast Results:")
    print(f"Forecast columns: {prophet_forecast.columns.tolist()}")
    print(f"Next 5 days forecast: {prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()}")
    
    # Detect changepoints
    changepoints = prophet_forecaster.detect_changepoints()
    print(f"Significant changepoints: {len(changepoints['significant_changepoints'])}")
```

### Neural Network-Based Forecasting

Modern deep learning approaches for time series forecasting using LSTM and Transformer architectures.

```python
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

class LSTMForecaster:
    """Production LSTM forecasting with advanced features"""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 lstm_units: int = 50,
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001):
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM forecasting")
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            X.append(data[i-self.sequence_length:i])
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM model architecture"""
        
        model = Sequential([
            Input(shape=input_shape),
            LSTM(self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units, return_sequences=True),
            Dropout(self.dropout_rate),
            LSTM(self.lstm_units),
            Dropout(self.dropout_rate),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def fit(self, ts: pd.Series, 
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 1) -> 'LSTMForecaster':
        """Fit LSTM model to time series"""
        
        # Scale data
        scaled_data = self.scaler.fit_transform(ts.values.reshape(-1, 1))
        
        # Create sequences
        X, y = self._create_sequences(scaled_data.flatten())
        
        # Reshape for LSTM
        X = X.reshape((X.shape[0], X.shape[1], 1))
        
        # Build model
        self.model = self._build_model((X.shape[1], 1))
        
        # Add callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        self.training_history = history
        
        return self
    
    def forecast(self, ts: pd.Series, steps: int) -> np.ndarray:
        """Generate LSTM forecasts"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        # Prepare last sequence
        scaled_data = self.scaler.transform(ts.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.sequence_length:].flatten()
        
        forecasts = []
        current_sequence = last_sequence.copy()
        
        for _ in range(steps):
            # Reshape for prediction
            X_pred = current_sequence.reshape((1, self.sequence_length, 1))
            
            # Predict next value
            next_pred = self.model.predict(X_pred, verbose=0)[0, 0]
            forecasts.append(next_pred)
            
            # Update sequence
            current_sequence = np.append(current_sequence[1:], next_pred)
        
        # Inverse transform forecasts
        forecasts = np.array(forecasts).reshape(-1, 1)
        forecasts = self.scaler.inverse_transform(forecasts).flatten()
        
        return forecasts
    
    def plot_training_history(self):
        """Plot training history"""
        
        if not hasattr(self, 'training_history'):
            raise ValueError("No training history available")
        
        history = self.training_history.history
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(history['loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # MAE plot
        ax2.plot(history['mae'], label='Training MAE')
        ax2.plot(history['val_mae'], label='Validation MAE')
        ax2.set_title('Model MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage with LSTM
if TF_AVAILABLE:
    # Use subset of data for faster training
    train_ts = ts[:800]
    
    lstm_forecaster = LSTMForecaster(sequence_length=30, lstm_units=32)
    lstm_forecaster.fit(train_ts, epochs=50, verbose=0)
    
    # Generate forecasts
    lstm_forecast = lstm_forecaster.forecast(train_ts, steps=30)
    
    print("LSTM Forecast Results:")
    print(f"Next 5 days forecast: {lstm_forecast[:5]}")
```

---

## 🏗️ Infrastructure Setup

### Quick Start (5 minutes)

```bash
# 1. Navigate to day 27
cd days/day-27-time-series-forecasting

# 2. Start the complete infrastructure
./setup.sh

# 3. Run interactive demo
python demo.py
```

### Infrastructure Components

**Time Series Stack**:
- **PostgreSQL**: Metadata and model storage
- **InfluxDB**: High-performance time series storage
- **Redis**: Real-time caching and session storage
- **MLflow**: Experiment tracking and model registry

**Forecasting Services**:
- **FastAPI Server**: Production forecasting endpoints
- **Data Generator**: Realistic multi-seasonal time series
- **Jupyter Notebook**: Interactive analysis environment

**Monitoring & Visualization**:
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Time series visualization and dashboards
- **Health Checks**: Service availability monitoring

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Forecasting API | http://localhost:8000 | - |
| Grafana Dashboard | http://localhost:3000 | admin/forecast123 |
| Jupyter Notebook | http://localhost:8888 | token: forecast123 |
| MLflow Tracking | http://localhost:5000 | - |
| Prometheus | http://localhost:9090 | - |

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List available time series
curl http://localhost:8000/series

# Generate forecast
curl -X POST http://localhost:8000/forecast \
  -H "Content-Type: application/json" \
  -d '{"series_name": "retail_sales", "horizon": 30, "model_type": "auto"}'

# Get time series data
curl http://localhost:8000/series/retail_sales/data?limit=100
```

---

## 💻 Hands-On Exercise (40 min)

### Exercise Overview

**Business Scenario**: You're the Lead Data Scientist at "Global Supply Chain Analytics", a company that provides demand forecasting solutions for retail and manufacturing clients. You need to build a comprehensive forecasting system that handles multiple time series patterns and provides accurate predictions for inventory planning.

**Your Mission**: Implement a production-ready forecasting system that combines classical statistical methods, modern Prophet forecasting, and neural network approaches to provide robust demand predictions.

### Requirements

1. **Classical Methods**: Implement ARIMA and SARIMA with automatic parameter selection
2. **Prophet Integration**: Build Prophet models with seasonality and holiday effects
3. **Neural Networks**: Create LSTM-based forecasting for complex patterns
4. **Ensemble Methods**: Combine multiple forecasting approaches for improved accuracy
5. **Production System**: Deploy automated forecasting pipeline with model selection

### Generated Datasets

The infrastructure provides realistic time series data:

- **Retail Sales**: Daily sales with seasonal patterns, holidays, and trends
- **Energy Consumption**: Hourly consumption with daily/weekly/seasonal cycles
- **Stock Prices**: Financial time series with volatility clustering
- **Website Traffic**: Web analytics with multiple seasonalities

See the exercise file for detailed implementation steps.

---

## 📚 Resources

- **Time Series Analysis**: [Forecasting: Principles and Practice](https://otexts.com/fpp3/) - Comprehensive forecasting textbook
- **Prophet Documentation**: [Facebook Prophet](https://facebook.github.io/prophet/) - Official Prophet guide
- **ARIMA Guide**: [Statsmodels ARIMA](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html) - Statistical forecasting
- **Neural Forecasting**: [TensorFlow Time Series](https://www.tensorflow.org/tutorials/structured_data/time_series) - Deep learning for forecasting
- **Production Forecasting**: [MLOps for Time Series](https://neptune.ai/blog/mlops-for-time-series) - Deployment best practices

---

## 🎯 Key Takeaways

- **Classical methods like ARIMA remain powerful** for many forecasting problems, especially with proper parameter tuning
- **Prophet excels at business forecasting** with clear seasonal patterns and holiday effects
- **Neural networks handle complex patterns** but require more data and computational resources
- **Ensemble methods often outperform individual models** by combining different forecasting strengths
- **Automatic parameter selection is crucial** for production systems handling multiple time series
- **Proper validation using time series splits** prevents data leakage and overfitting
- **Seasonality detection and handling** significantly improves forecast accuracy
- **Production forecasting requires robust error handling** and fallback mechanisms
- **Time series databases like InfluxDB** provide optimized storage for forecasting workloads
- **Real-time forecasting APIs** enable integration with business applications
- **Comprehensive monitoring** is essential for production forecasting systems

---

## 🚀 What's Next?

Tomorrow (Day 28), you'll learn **Anomaly Detection** with statistical and ML-based methods, building on the time series analysis skills you've developed today.

**Preview**: You'll explore advanced anomaly detection techniques, real-time monitoring systems, and production anomaly detection pipelines that leverage time series forecasting for baseline establishment!

---

## ✅ Before Moving On

- [ ] Understand the strengths and limitations of different forecasting approaches
- [ ] Can implement ARIMA/SARIMA with automatic parameter selection
- [ ] Know how to use Prophet for business forecasting with seasonality
- [ ] Understand neural network-based forecasting with LSTM
- [ ] Can build ensemble forecasting systems for improved accuracy
- [ ] Complete the comprehensive forecasting system implementation exercise
- [ ] Review production deployment considerations for forecasting systems

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐ (Advanced ML Engineering)

**Phase 3 Progress**: Building sophisticated ML capabilities! 📈
