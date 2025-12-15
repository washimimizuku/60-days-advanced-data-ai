"""
Day 27: Time Series Forecasting - ARIMA, Prophet & Neural Networks - Complete Solution

Production-ready forecasting implementation for supply chain analytics.
Demonstrates enterprise-grade forecasting across multiple methodologies.

This solution showcases:
- Classical statistical forecasting with ARIMA and SARIMA
- Modern Prophet forecasting with seasonality and holiday effects
- Neural network forecasting using LSTM architectures
- Ensemble methods combining multiple forecasting approaches
- Production pipeline with automated model selection
- Comprehensive evaluation and confidence interval estimation
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
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    print("Statsmodels not available. Install with: pip install statsmodels")
    STATSMODELS_AVAILABLE = False

# Prophet forecasting
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
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

# Core libraries
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import logging
import json
import time
# =============================================================================
# PRODUCTION CLASSICAL FORECASTING (ARIMA/SARIMA)
# =============================================================================

class ProductionClassicalForecaster:
    """Production-grade ARIMA and SARIMA forecasting with comprehensive features"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.is_fitted = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for forecasting"""
        logger = logging.getLogger('classical_forecaster')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def check_stationarity(self, ts: pd.Series) -> Dict[str, Any]:
        """Comprehensive stationarity check using multiple tests"""
        
        if not STATSMODELS_AVAILABLE:
            return {'is_stationary': False, 'p_value': 1.0}
        
        # Augmented Dickey-Fuller test
        adf_result = adfuller(ts.dropna())
        
        # KPSS test for trend stationarity
        try:
            from statsmodels.tsa.stattools import kpss
            kpss_result = kpss(ts.dropna(), regression='c')
            kpss_stationary = kpss_result[1] > 0.05  # Null hypothesis: stationary
        except:
            kpss_stationary = None
        
        return {
            'adf_statistic': adf_result[0],
            'adf_p_value': adf_result[1],
            'adf_critical_values': adf_result[4],
            'is_stationary_adf': adf_result[1] < 0.05,
            'is_stationary_kpss': kpss_stationary,
            'is_stationary': adf_result[1] < 0.05
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
            
            self.logger.info(f"Applied differencing order {diff_order}, p-value: {stationarity['adf_p_value']:.4f}")
        
        return current_ts, diff_order
    
    def auto_arima_selection(self, ts: pd.Series, 
                           max_p: int = 5, max_d: int = 2, max_q: int = 5,
                           information_criterion: str = 'aic') -> Dict[str, Any]:
        """Comprehensive ARIMA parameter selection with multiple criteria"""
        
        if not STATSMODELS_AVAILABLE:
            return {'best_order': (1, 1, 1), 'best_ic': np.inf}
        
        self.logger.info(f"Starting ARIMA grid search with max_p={max_p}, max_d={max_d}, max_q={max_q}")
        
        best_ic = np.inf
        best_order = None
        best_model = None
        results = []
        
        # Grid search for best parameters
        for p in range(max_p + 1):
            for d in range(max_d + 1):
                for q in range(max_q + 1):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted_model = model.fit()
                        
                        # Get information criterion
                        if information_criterion == 'aic':
                            ic_value = fitted_model.aic
                        elif information_criterion == 'bic':
                            ic_value = fitted_model.bic
                        else:
                            ic_value = fitted_model.aic
                        
                        results.append({
                            'order': (p, d, q),
                            'aic': fitted_model.aic,
                            'bic': fitted_model.bic,
                            'llf': fitted_model.llf
                        })
                        
                        if ic_value < best_ic:
                            best_ic = ic_value
                            best_order = (p, d, q)
                            best_model = fitted_model
                            
                    except Exception as e:
                        self.logger.debug(f"Failed to fit ARIMA{(p, d, q)}: {e}")
                        continue
        
        self.logger.info(f"Best ARIMA order: {best_order} with {information_criterion}={best_ic:.4f}")
        
        return {
            'best_order': best_order,
            'best_ic': best_ic,
            'best_model': best_model,
            'all_results': results
        }
    
    def auto_sarima_selection(self, ts: pd.Series, seasonal_period: int = 12) -> Dict[str, Any]:
        """Comprehensive SARIMA parameter selection"""
        
        if not STATSMODELS_AVAILABLE:
            return {'best_order': (1, 1, 1), 'best_seasonal_order': (1, 1, 1, 12)}
        
        self.logger.info(f"Starting SARIMA grid search with seasonal period {seasonal_period}")
        
        best_aic = np.inf
        best_order = None
        best_seasonal_order = None
        best_model = None
        
        # Reduced parameter space for computational efficiency
        p_values = [0, 1, 2]
        d_values = [0, 1]
        q_values = [0, 1, 2]
        P_values = [0, 1]
        D_values = [0, 1]
        Q_values = [0, 1]
        
        total_combinations = len(p_values) * len(d_values) * len(q_values) * len(P_values) * len(D_values) * len(Q_values)
        self.logger.info(f"Testing {total_combinations} SARIMA parameter combinations")
        
        for p in p_values:
            for d in d_values:
                for q in q_values:
                    for P in P_values:
                        for D in D_values:
                            for Q in Q_values:
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
                                        
                                except Exception as e:
                                    self.logger.debug(f"Failed to fit SARIMA{order}x{seasonal_order}: {e}")
                                    continue
        
        self.logger.info(f"Best SARIMA: {best_order}x{best_seasonal_order} with AIC={best_aic:.4f}")
        
        return {
            'best_order': best_order,
            'best_seasonal_order': best_seasonal_order,
            'best_aic': best_aic,
            'best_model': best_model
        }
    
    def detect_seasonality(self, ts: pd.Series) -> Dict[str, Any]:
        """Detect seasonality using multiple methods"""
        
        # Method 1: Autocorrelation analysis
        autocorr_lags = []
        if len(ts) > 50:
            autocorr = [ts.autocorr(lag=i) for i in range(1, min(len(ts)//3, 100))]
            # Find peaks in autocorrelation
            for i in range(1, len(autocorr)-1):
                if autocorr[i] > 0.3 and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                    autocorr_lags.append(i+1)
        
        # Method 2: FFT-based seasonality detection
        fft_period = None
        if len(ts) > 100:
            try:
                # Remove trend
                detrended = ts - ts.rolling(window=min(12, len(ts)//4), center=True).mean()
                detrended = detrended.dropna()
                
                # Apply FFT
                fft = np.fft.fft(detrended.values)
                frequencies = np.fft.fftfreq(len(detrended))
                
                # Find dominant frequency
                magnitude = np.abs(fft)
                dominant_freq_idx = np.argmax(magnitude[1:len(magnitude)//2]) + 1
                dominant_freq = frequencies[dominant_freq_idx]
                
                if dominant_freq != 0:
                    fft_period = int(1 / abs(dominant_freq))
                    fft_period = max(2, min(fft_period, len(ts) // 3))
            except:
                pass
        
        # Method 3: Seasonal decomposition
        seasonal_strength = 0.0
        if len(ts) > 24:
            try:
                period = fft_period if fft_period else (12 if len(ts) > 24 else len(ts)//2)
                decomposition = seasonal_decompose(ts, model='additive', period=period)
                
                # Calculate seasonal strength
                seasonal_var = np.var(decomposition.seasonal.dropna())
                residual_var = np.var(decomposition.resid.dropna())
                seasonal_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0
                
            except:
                pass
        
        return {
            'autocorr_periods': autocorr_lags[:3],  # Top 3 periods
            'fft_period': fft_period,
            'seasonal_strength': seasonal_strength,
            'has_seasonality': seasonal_strength > 0.3 or len(autocorr_lags) > 0,
            'recommended_period': fft_period if fft_period else (autocorr_lags[0] if autocorr_lags else 12)
        }
    
    def fit(self, ts: pd.Series, seasonal_period: int = None, auto_seasonal: bool = True) -> 'ProductionClassicalForecaster':
        """Fit both ARIMA and SARIMA models and select the best"""
        
        self.logger.info("Starting classical forecasting model fitting...")
        
        # Detect seasonality if not provided
        if seasonal_period is None and auto_seasonal:
            seasonality_info = self.detect_seasonality(ts)
            if seasonality_info['has_seasonality']:
                seasonal_period = seasonality_info['recommended_period']
                self.logger.info(f"Auto-detected seasonal period: {seasonal_period}")
        
        # Fit ARIMA model
        self.logger.info("Fitting ARIMA model...")
        arima_result = self.auto_arima_selection(ts)
        self.models['arima'] = arima_result
        
        # Fit SARIMA model if seasonality detected
        if seasonal_period and seasonal_period > 1:
            self.logger.info(f"Fitting SARIMA model with seasonal period {seasonal_period}...")
            sarima_result = self.auto_sarima_selection(ts, seasonal_period)
            self.models['sarima'] = sarima_result
        
        # Select best model based on AIC
        best_aic = np.inf
        for model_name, model_info in self.models.items():
            model_aic = model_info.get('best_aic', model_info.get('best_ic', np.inf))
            if model_aic < best_aic:
                best_aic = model_aic
                self.best_model = model_info['best_model']
                self.best_model_name = model_name
        
        self.is_fitted = True
        self.logger.info(f"Best model selected: {self.best_model_name} with AIC={best_aic:.4f}")
        
        return self
    
    def forecast(self, steps: int, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Generate forecasts with comprehensive confidence intervals"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before forecasting")
        
        self.logger.info(f"Generating {steps}-step forecast with {confidence_level*100}% confidence")
        
        # Generate forecast
        forecast_result = self.best_model.get_forecast(steps=steps, alpha=1-confidence_level)
        
        return {
            'forecast': forecast_result.predicted_mean,
            'lower_bound': forecast_result.conf_int().iloc[:, 0],
            'upper_bound': forecast_result.conf_int().iloc[:, 1],
            'confidence_level': confidence_level,
            'model_type': self.best_model_name,
            'model_order': getattr(self.best_model, 'model_orders', None)
        }
    
    def get_model_diagnostics(self) -> Dict[str, Any]:
        """Comprehensive model diagnostics"""
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before diagnostics")
        
        diagnostics = {
            'model_type': self.best_model_name,
            'aic': self.best_model.aic,
            'bic': self.best_model.bic,
            'llf': self.best_model.llf,
            'residuals_mean': self.best_model.resid.mean(),
            'residuals_std': self.best_model.resid.std(),
        }
        
        # Ljung-Box test for residual autocorrelation
        try:
            ljung_box_result = acorr_ljungbox(self.best_model.resid, lags=10, return_df=True)
            diagnostics['ljung_box_p_value'] = ljung_box_result['lb_pvalue'].iloc[-1]
        except:
            diagnostics['ljung_box_p_value'] = np.nan
        
        # Jarque-Bera test for normality
        try:
            from scipy.stats import jarque_bera
            jb_stat, jb_p_value = jarque_bera(self.best_model.resid.dropna())
            diagnostics['jarque_bera_p_value'] = jb_p_value
        except:
            diagnostics['jarque_bera_p_value'] = np.nan
        
        return diagnostics
# =============================================================================
# PRODUCTION PROPHET FORECASTING
# =============================================================================

class ProductionProphetForecaster:
    """Production-grade Prophet forecasting with advanced features"""
    
    def __init__(self):
        self.model = None
        self.is_fitted = False
        self.regressors = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for Prophet forecasting"""
        logger = logging.getLogger('prophet_forecaster')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def setup_model(self, 
                   growth: str = 'linear',
                   seasonality_mode: str = 'additive',
                   yearly_seasonality: str = 'auto',
                   weekly_seasonality: str = 'auto',
                   daily_seasonality: str = 'auto',
                   seasonality_prior_scale: float = 10.0,
                   holidays_prior_scale: float = 10.0,
                   changepoint_prior_scale: float = 0.05,
                   changepoint_range: float = 0.8) -> 'ProductionProphetForecaster':
        """Setup Prophet model with comprehensive configuration"""
        
        if not PROPHET_AVAILABLE:
            self.logger.error("Prophet not available")
            return self
        
        self.logger.info("Setting up Prophet model with advanced configuration")
        
        self.model = Prophet(
            growth=growth,
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale,
            changepoint_prior_scale=changepoint_prior_scale,
            changepoint_range=changepoint_range
        )
        
        return self
    
    def add_holidays(self, country: str = 'US', custom_holidays: pd.DataFrame = None) -> 'ProductionProphetForecaster':
        """Add comprehensive holiday effects"""
        
        if not PROPHET_AVAILABLE or not self.model:
            return self
        
        # Add country holidays
        if country:
            self.model.add_country_holidays(country_name=country)
            self.logger.info(f"Added {country} holidays")
        
        # Add custom holidays
        if custom_holidays is not None:
            for _, holiday in custom_holidays.iterrows():
                self.model.holidays = pd.concat([
                    self.model.holidays if self.model.holidays is not None else pd.DataFrame(),
                    pd.DataFrame({
                        'holiday': [holiday['holiday']],
                        'ds': [holiday['ds']],
                        'lower_window': [holiday.get('lower_window', 0)],
                        'upper_window': [holiday.get('upper_window', 0)]
                    })
                ])
            self.logger.info(f"Added {len(custom_holidays)} custom holidays")
        
        return self
    
    def add_custom_seasonality(self, name: str, period: float, fourier_order: int, 
                             prior_scale: float = 10.0, mode: str = None) -> 'ProductionProphetForecaster':
        """Add custom seasonality patterns"""
        
        if not PROPHET_AVAILABLE or not self.model:
            return self
        
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale,
            mode=mode
        )
        
        self.logger.info(f"Added custom seasonality: {name} (period={period}, fourier_order={fourier_order})")
        
        return self
    
    def add_regressor(self, name: str, prior_scale: float = 10.0, 
                     standardize: str = 'auto', mode: str = None) -> 'ProductionProphetForecaster':
        """Add external regressor with advanced options"""
        
        if not PROPHET_AVAILABLE or not self.model:
            return self
        
        self.model.add_regressor(
            name=name,
            prior_scale=prior_scale,
            standardize=standardize,
            mode=mode
        )
        
        self.regressors.append(name)
        self.logger.info(f"Added regressor: {name} (prior_scale={prior_scale})")
        
        return self
    
    def fit(self, df: pd.DataFrame) -> 'ProductionProphetForecaster':
        """Fit Prophet model with comprehensive validation"""
        
        if not PROPHET_AVAILABLE:
            self.logger.error("Prophet not available")
            return self
        
        # Validate input format
        required_cols = ['ds', 'y']
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"DataFrame must contain columns: {required_cols}")
        
        # Validate regressor columns
        for regressor in self.regressors:
            if regressor not in df.columns:
                raise ValueError(f"Regressor '{regressor}' not found in DataFrame")
        
        # Prepare data
        df_clean = df.copy()
        df_clean['ds'] = pd.to_datetime(df_clean['ds'])
        
        # Remove outliers (optional)
        if len(df_clean) > 50:
            Q1 = df_clean['y'].quantile(0.25)
            Q3 = df_clean['y'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df_clean['y'] < lower_bound) | (df_clean['y'] > upper_bound)
            if outliers.sum() > 0:
                self.logger.info(f"Detected {outliers.sum()} outliers, capping values")
                df_clean.loc[df_clean['y'] < lower_bound, 'y'] = lower_bound
                df_clean.loc[df_clean['y'] > upper_bound, 'y'] = upper_bound
        
        self.logger.info(f"Fitting Prophet model on {len(df_clean)} data points")
        
        # Fit model
        self.model.fit(df_clean)
        self.is_fitted = True
        
        self.logger.info("Prophet model fitted successfully")
        
        return self
    
    def forecast(self, periods: int, freq: str = 'D', 
                include_history: bool = True,
                future_regressors: pd.DataFrame = None) -> pd.DataFrame:
        """Generate comprehensive Prophet forecasts"""
        
        if not PROPHET_AVAILABLE or not self.is_fitted:
            self.logger.error("Prophet not available or not fitted")
            return pd.DataFrame()
        
        self.logger.info(f"Generating {periods}-period forecast")
        
        # Create future dataframe
        future = self.model.make_future_dataframe(
            periods=periods, 
            freq=freq, 
            include_history=include_history
        )
        
        # Add regressor values
        if future_regressors is not None and self.regressors:
            for regressor in self.regressors:
                if regressor in future_regressors.columns:
                    # Align future regressors with future dataframe
                    future = future.merge(
                        future_regressors[['ds', regressor]], 
                        on='ds', 
                        how='left'
                    )
                    
                    # Forward fill missing values
                    future[regressor] = future[regressor].fillna(method='ffill')
                else:
                    self.logger.warning(f"Regressor '{regressor}' not found in future_regressors")
        
        # Generate forecast
        forecast = self.model.predict(future)
        
        self.logger.info("Prophet forecast generated successfully")
        
        return forecast
    
    def cross_validate_performance(self, df: pd.DataFrame, 
                                 initial: str = '730 days',
                                 period: str = '180 days', 
                                 horizon: str = '365 days') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Comprehensive time series cross-validation"""
        
        if not PROPHET_AVAILABLE or not self.is_fitted:
            self.logger.error("Prophet not available or not fitted")
            return pd.DataFrame(), pd.DataFrame()
        
        self.logger.info("Starting cross-validation...")
        
        try:
            cv_results = cross_validation(
                self.model, 
                initial=initial,
                period=period,
                horizon=horizon
            )
            
            # Calculate performance metrics
            metrics = performance_metrics(cv_results)
            
            self.logger.info(f"Cross-validation completed: {len(cv_results)} forecasts evaluated")
            
            return cv_results, metrics
            
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            return pd.DataFrame(), pd.DataFrame()
    
    def detect_changepoints(self) -> Dict[str, Any]:
        """Advanced changepoint detection and analysis"""
        
        if not PROPHET_AVAILABLE or not self.is_fitted:
            return {'changepoints': [], 'significant_changepoints': []}
        
        changepoints = self.model.changepoints
        
        if len(changepoints) == 0:
            return {'changepoints': [], 'significant_changepoints': []}
        
        # Get changepoint effects
        changepoint_effects = self.model.params['delta'].mean(axis=0)
        
        # Calculate significance threshold
        threshold = np.std(changepoint_effects) * 2
        
        # Find significant changepoints
        significant_changepoints = []
        for i, (cp, effect) in enumerate(zip(changepoints, changepoint_effects)):
            if abs(effect) > threshold:
                significant_changepoints.append({
                    'date': cp,
                    'effect': effect,
                    'significance': abs(effect) / threshold,
                    'direction': 'increase' if effect > 0 else 'decrease'
                })
        
        self.logger.info(f"Detected {len(significant_changepoints)} significant changepoints")
        
        return {
            'all_changepoints': changepoints.tolist(),
            'changepoint_effects': changepoint_effects.tolist(),
            'significant_changepoints': significant_changepoints,
            'threshold': threshold
        }
    
    def plot_components(self, forecast: pd.DataFrame, figsize: Tuple[int, int] = (12, 10)):
        """Plot comprehensive forecast components"""
        
        if not PROPHET_AVAILABLE or not self.is_fitted:
            return None
        
        fig = self.model.plot_components(forecast, figsize=figsize)
        plt.tight_layout()
        plt.show()
        
        return fig
# =============================================================================
# PRODUCTION LSTM FORECASTING
# =============================================================================

class ProductionLSTMForecaster:
    """Production-grade LSTM forecasting with advanced neural architectures"""
    
    def __init__(self, 
                 sequence_length: int = 60,
                 lstm_units: List[int] = [50, 50],
                 dropout_rate: float = 0.2,
                 learning_rate: float = 0.001,
                 batch_size: int = 32):
        
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.training_history = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for LSTM forecasting"""
        logger = logging.getLogger('lstm_forecaster')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def prepare_sequences(self, data: np.ndarray, 
                         include_features: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training with optional feature engineering"""
        
        X, y = [], []
        
        for i in range(self.sequence_length, len(data)):
            sequence = data[i-self.sequence_length:i]
            
            if include_features:
                # Add simple technical indicators
                sequence_features = self._add_sequence_features(sequence)
                X.append(sequence_features)
            else:
                X.append(sequence)
            
            y.append(data[i])
        
        return np.array(X), np.array(y)
    
    def _add_sequence_features(self, sequence: np.ndarray) -> np.ndarray:
        """Add technical indicators to sequence"""
        
        # Simple moving averages
        sma_5 = np.convolve(sequence, np.ones(5)/5, mode='same')
        sma_10 = np.convolve(sequence, np.ones(10)/10, mode='same')
        
        # Price changes
        price_change = np.diff(sequence, prepend=sequence[0])
        
        # Volatility (rolling std)
        volatility = pd.Series(sequence).rolling(window=5, min_periods=1).std().values
        
        # Combine features
        features = np.column_stack([
            sequence,
            sma_5,
            sma_10,
            price_change,
            volatility
        ])
        
        return features
    
    def build_model(self, input_shape: Tuple[int, int], 
                   architecture: str = 'stacked') -> tf.keras.Model:
        """Build advanced LSTM model architecture"""
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM forecasting")
        
        self.logger.info(f"Building {architecture} LSTM model with input shape {input_shape}")
        
        if architecture == 'stacked':
            model = self._build_stacked_lstm(input_shape)
        elif architecture == 'bidirectional':
            model = self._build_bidirectional_lstm(input_shape)
        elif architecture == 'attention':
            model = self._build_attention_lstm(input_shape)
        else:
            model = self._build_stacked_lstm(input_shape)
        
        # Compile model with advanced optimizer
        optimizer = Adam(
            learning_rate=self.learning_rate,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        return model
    
    def _build_stacked_lstm(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build stacked LSTM architecture"""
        
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Stacked LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(LSTM(units, return_sequences=return_sequences))
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(1))
        
        return model
    
    def _build_bidirectional_lstm(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build bidirectional LSTM architecture"""
        
        from tensorflow.keras.layers import Bidirectional
        
        model = Sequential()
        model.add(Input(shape=input_shape))
        
        # Bidirectional LSTM layers
        for i, units in enumerate(self.lstm_units):
            return_sequences = i < len(self.lstm_units) - 1
            model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            model.add(Dropout(self.dropout_rate))
        
        # Dense layers
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        
        return model
    
    def _build_attention_lstm(self, input_shape: Tuple[int, int]) -> tf.keras.Model:
        """Build LSTM with attention mechanism"""
        
        from tensorflow.keras.layers import Attention, Concatenate
        
        # Input
        inputs = Input(shape=input_shape)
        
        # LSTM layers
        lstm_out = inputs
        for i, units in enumerate(self.lstm_units):
            lstm_out = LSTM(units, return_sequences=True)(lstm_out)
            lstm_out = Dropout(self.dropout_rate)(lstm_out)
        
        # Attention mechanism
        attention = Attention()([lstm_out, lstm_out])
        
        # Global average pooling
        from tensorflow.keras.layers import GlobalAveragePooling1D
        pooled = GlobalAveragePooling1D()(attention)
        
        # Dense layers
        dense = Dense(50, activation='relu')(pooled)
        dense = Dropout(self.dropout_rate)(dense)
        outputs = Dense(1)(dense)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        return model
    
    def fit(self, ts: pd.Series, 
            validation_split: float = 0.2,
            epochs: int = 100,
            architecture: str = 'stacked',
            include_features: bool = False,
            verbose: int = 1) -> 'ProductionLSTMForecaster':
        """Fit LSTM model with comprehensive training pipeline"""
        
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow not available")
            return self
        
        self.logger.info(f"Starting LSTM training with {len(ts)} data points")
        
        # Scale data
        scaled_data = self.scaler.fit_transform(ts.values.reshape(-1, 1))
        
        # Create sequences
        X, y = self.prepare_sequences(scaled_data.flatten(), include_features)
        
        if len(X) == 0:
            raise ValueError(f"Not enough data for sequence length {self.sequence_length}")
        
        # Reshape for LSTM
        if include_features:
            input_shape = (X.shape[1], X.shape[2])
        else:
            X = X.reshape((X.shape[0], X.shape[1], 1))
            input_shape = (X.shape[1], 1)
        
        self.logger.info(f"Prepared {len(X)} sequences with shape {input_shape}")
        
        # Build model
        self.model = self.build_model(input_shape, architecture)
        
        # Advanced callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=0
            )
        ]
        
        # Train model
        self.logger.info("Starting model training...")
        
        self.training_history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=verbose
        )
        
        self.is_fitted = True
        self.logger.info("LSTM model training completed")
        
        return self
    
    def forecast(self, ts: pd.Series, steps: int, 
                confidence_level: float = 0.95,
                monte_carlo_samples: int = 100) -> Dict[str, Any]:
        """Generate LSTM forecasts with uncertainty estimation"""
        
        if not TF_AVAILABLE or not self.is_fitted:
            self.logger.error("TensorFlow not available or model not fitted")
            return {'forecast': np.zeros(steps)}
        
        self.logger.info(f"Generating {steps}-step LSTM forecast")
        
        # Prepare last sequence
        scaled_data = self.scaler.transform(ts.values.reshape(-1, 1))
        last_sequence = scaled_data[-self.sequence_length:].flatten()
        
        # Monte Carlo dropout for uncertainty estimation
        forecasts_mc = []
        
        for _ in range(monte_carlo_samples):
            forecasts = []
            current_sequence = last_sequence.copy()
            
            for step in range(steps):
                # Reshape for prediction
                X_pred = current_sequence.reshape((1, self.sequence_length, 1))
                
                # Predict with dropout enabled (for uncertainty)
                next_pred = self.model(X_pred, training=True).numpy()[0, 0]
                forecasts.append(next_pred)
                
                # Update sequence
                current_sequence = np.append(current_sequence[1:], next_pred)
            
            forecasts_mc.append(forecasts)
        
        # Calculate statistics
        forecasts_mc = np.array(forecasts_mc)
        mean_forecast = np.mean(forecasts_mc, axis=0)
        std_forecast = np.std(forecasts_mc, axis=0)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        lower_bound = np.percentile(forecasts_mc, lower_percentile, axis=0)
        upper_bound = np.percentile(forecasts_mc, upper_percentile, axis=0)
        
        # Inverse transform
        mean_forecast = self.scaler.inverse_transform(mean_forecast.reshape(-1, 1)).flatten()
        lower_bound = self.scaler.inverse_transform(lower_bound.reshape(-1, 1)).flatten()
        upper_bound = self.scaler.inverse_transform(upper_bound.reshape(-1, 1)).flatten()
        
        return {
            'forecast': mean_forecast,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'std': self.scaler.inverse_transform(std_forecast.reshape(-1, 1)).flatten(),
            'confidence_level': confidence_level,
            'monte_carlo_samples': monte_carlo_samples
        }
    
    def plot_training_history(self, figsize: Tuple[int, int] = (12, 4)):
        """Plot comprehensive training history"""
        
        if not hasattr(self, 'training_history') or self.training_history is None:
            self.logger.warning("No training history available")
            return
        
        history = self.training_history.history
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Loss plot
        axes[0].plot(history['loss'], label='Training Loss', color='blue')
        axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0].set_title('Model Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # MAE plot
        axes[1].plot(history['mae'], label='Training MAE', color='blue')
        axes[1].plot(history['val_mae'], label='Validation MAE', color='red')
        axes[1].set_title('Mean Absolute Error')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True)
        
        # Learning rate plot (if available)
        if 'lr' in history:
            axes[2].plot(history['lr'], label='Learning Rate', color='green')
            axes[2].set_title('Learning Rate')
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('Learning Rate')
            axes[2].set_yscale('log')
            axes[2].legend()
            axes[2].grid(True)
        else:
            axes[2].text(0.5, 0.5, 'Learning Rate\nNot Available', 
                        ha='center', va='center', transform=axes[2].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        
        if not self.is_fitted:
            return {'error': 'Model not fitted'}
        
        summary = {
            'architecture': 'LSTM',
            'sequence_length': self.sequence_length,
            'lstm_units': self.lstm_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'total_parameters': self.model.count_params(),
            'trainable_parameters': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        }
        
        if self.training_history:
            history = self.training_history.history
            summary.update({
                'epochs_trained': len(history['loss']),
                'final_train_loss': history['loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'best_val_loss': min(history['val_loss']),
                'best_epoch': np.argmin(history['val_loss']) + 1
            })
        
        return summary
# =============================================================================
# ENSEMBLE FORECASTING SYSTEM
# =============================================================================

class ProductionEnsembleForecaster:
    """Production ensemble forecasting with intelligent model combination"""
    
    def __init__(self, combination_method: str = 'weighted_average'):
        self.forecasters = {}
        self.weights = {}
        self.performance_history = {}
        self.combination_method = combination_method
        self.is_fitted = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for ensemble forecasting"""
        logger = logging.getLogger('ensemble_forecaster')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def add_forecaster(self, name: str, forecaster: Any, weight: float = 1.0):
        """Add forecaster to ensemble with initial weight"""
        
        self.forecasters[name] = forecaster
        self.weights[name] = weight
        self.performance_history[name] = []
        
        self.logger.info(f"Added forecaster '{name}' with weight {weight}")
    
    def fit(self, ts: pd.Series, **kwargs) -> 'ProductionEnsembleForecaster':
        """Fit all forecasters in ensemble with error handling"""
        
        self.logger.info(f"Fitting ensemble with {len(self.forecasters)} forecasters")
        
        successful_fits = 0
        
        for name, forecaster in self.forecasters.items():
            try:
                self.logger.info(f"Fitting {name}...")
                start_time = time.time()
                
                # Handle different forecaster types
                if isinstance(forecaster, ProductionProphetForecaster):
                    # Prophet expects DataFrame with 'ds' and 'y' columns
                    prophet_df = pd.DataFrame({
                        'ds': ts.index,
                        'y': ts.values
                    })
                    forecaster.fit(prophet_df)
                else:
                    # Other forecasters expect Series
                    forecaster.fit(ts, **kwargs)
                
                fit_time = time.time() - start_time
                self.logger.info(f"Successfully fitted {name} in {fit_time:.2f}s")
                successful_fits += 1
                
            except Exception as e:
                self.logger.error(f"Failed to fit {name}: {e}")
                # Remove failed forecaster from ensemble
                self.weights[name] = 0.0
        
        if successful_fits == 0:
            raise ValueError("No forecasters could be fitted successfully")
        
        self.is_fitted = True
        self.logger.info(f"Ensemble fitting completed: {successful_fits}/{len(self.forecasters)} successful")
        
        return self
    
    def forecast(self, steps: int, **kwargs) -> Dict[str, Any]:
        """Generate ensemble forecasts with multiple combination methods"""
        
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before forecasting")
        
        self.logger.info(f"Generating ensemble forecast for {steps} steps")
        
        individual_forecasts = {}
        valid_forecasts = []
        valid_weights = []
        forecast_metadata = {}
        
        # Get forecasts from each method
        for name, forecaster in self.forecasters.items():
            if self.weights[name] == 0.0:
                continue
                
            try:
                start_time = time.time()
                
                # Get forecast based on forecaster type
                if isinstance(forecaster, ProductionProphetForecaster):
                    forecast_df = forecaster.forecast(periods=steps, **kwargs)
                    forecast_values = forecast_df['yhat'].tail(steps).values
                    
                    forecast_metadata[name] = {
                        'lower_bound': forecast_df['yhat_lower'].tail(steps).values,
                        'upper_bound': forecast_df['yhat_upper'].tail(steps).values
                    }
                    
                elif hasattr(forecaster, 'forecast'):
                    forecast_result = forecaster.forecast(steps, **kwargs)
                    
                    if isinstance(forecast_result, dict):
                        forecast_values = forecast_result['forecast']
                        if 'lower_bound' in forecast_result:
                            forecast_metadata[name] = {
                                'lower_bound': forecast_result['lower_bound'],
                                'upper_bound': forecast_result['upper_bound']
                            }
                    else:
                        forecast_values = forecast_result
                else:
                    self.logger.warning(f"Forecaster {name} has no forecast method")
                    continue
                
                individual_forecasts[name] = forecast_values
                valid_forecasts.append(forecast_values)
                valid_weights.append(self.weights[name])
                
                forecast_time = time.time() - start_time
                self.logger.info(f"Generated forecast from {name} in {forecast_time:.2f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to get forecast from {name}: {e}")
                continue
        
        if not valid_forecasts:
            raise ValueError("No valid forecasts could be generated")
        
        # Combine forecasts
        ensemble_result = self._combine_forecasts(
            valid_forecasts, valid_weights, forecast_metadata
        )
        
        ensemble_result.update({
            'individual_forecasts': individual_forecasts,
            'weights_used': dict(zip([name for name in self.forecasters.keys() 
                                    if name in individual_forecasts], valid_weights)),
            'combination_method': self.combination_method,
            'num_models': len(valid_forecasts)
        })
        
        return ensemble_result
    
    def _combine_forecasts(self, forecasts: List[np.ndarray], 
                          weights: List[float],
                          metadata: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine forecasts using specified method"""
        
        forecasts_array = np.array(forecasts)
        weights_array = np.array(weights)
        
        if self.combination_method == 'simple_average':
            ensemble_forecast = np.mean(forecasts_array, axis=0)
            
        elif self.combination_method == 'weighted_average':
            # Normalize weights
            weights_normalized = weights_array / np.sum(weights_array)
            ensemble_forecast = np.average(forecasts_array, axis=0, weights=weights_normalized)
            
        elif self.combination_method == 'median':
            ensemble_forecast = np.median(forecasts_array, axis=0)
            
        elif self.combination_method == 'trimmed_mean':
            # Remove top and bottom 20% and average the rest
            sorted_forecasts = np.sort(forecasts_array, axis=0)
            trim_count = max(1, len(forecasts) // 5)
            trimmed_forecasts = sorted_forecasts[trim_count:-trim_count]
            ensemble_forecast = np.mean(trimmed_forecasts, axis=0)
            
        else:
            # Default to weighted average
            weights_normalized = weights_array / np.sum(weights_array)
            ensemble_forecast = np.average(forecasts_array, axis=0, weights=weights_normalized)
        
        # Calculate ensemble confidence intervals
        ensemble_std = np.std(forecasts_array, axis=0)
        ensemble_lower = ensemble_forecast - 1.96 * ensemble_std
        ensemble_upper = ensemble_forecast + 1.96 * ensemble_std
        
        return {
            'forecast': ensemble_forecast,
            'lower_bound': ensemble_lower,
            'upper_bound': ensemble_upper,
            'forecast_std': ensemble_std
        }
    
    def evaluate_forecasters(self, ts: pd.Series, test_size: int = 30, 
                           metrics: List[str] = None) -> Dict[str, Dict[str, float]]:
        """Comprehensive evaluation of individual forecasters"""
        
        if metrics is None:
            metrics = ['mae', 'mse', 'rmse', 'mape', 'smape']
        
        self.logger.info(f"Evaluating forecasters on {test_size} test samples")
        
        # Split data
        train_ts = ts[:-test_size]
        test_ts = ts[-test_size:]
        
        evaluation_results = {}
        
        for name, forecaster in self.forecasters.items():
            try:
                self.logger.info(f"Evaluating {name}...")
                
                # Fit on training data
                if isinstance(forecaster, ProductionProphetForecaster):
                    train_df = pd.DataFrame({'ds': train_ts.index, 'y': train_ts.values})
                    forecaster.fit(train_df)
                    forecast_df = forecaster.forecast(periods=test_size)
                    forecast_values = forecast_df['yhat'].tail(test_size).values
                else:
                    forecaster.fit(train_ts)
                    forecast_result = forecaster.forecast(test_size)
                    
                    if isinstance(forecast_result, dict):
                        forecast_values = forecast_result['forecast']
                    else:
                        forecast_values = forecast_result
                
                # Calculate metrics
                results = {}
                
                if 'mae' in metrics:
                    results['mae'] = mean_absolute_error(test_ts.values, forecast_values)
                
                if 'mse' in metrics:
                    results['mse'] = mean_squared_error(test_ts.values, forecast_values)
                
                if 'rmse' in metrics:
                    results['rmse'] = np.sqrt(mean_squared_error(test_ts.values, forecast_values))
                
                if 'mape' in metrics:
                    results['mape'] = np.mean(np.abs((test_ts.values - forecast_values) / test_ts.values)) * 100
                
                if 'smape' in metrics:
                    results['smape'] = np.mean(2 * np.abs(test_ts.values - forecast_values) / 
                                            (np.abs(test_ts.values) + np.abs(forecast_values))) * 100
                
                evaluation_results[name] = results
                self.performance_history[name].append(results)
                
                self.logger.info(f"Evaluated {name}: MAE={results.get('mae', 0):.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {name}: {e}")
                evaluation_results[name] = {metric: np.inf for metric in metrics}
        
        return evaluation_results
    
    def update_weights(self, evaluation_results: Dict[str, Dict[str, float]], 
                      metric: str = 'mae', method: str = 'inverse_error'):
        """Update ensemble weights based on performance"""
        
        self.logger.info(f"Updating weights based on {metric} using {method} method")
        
        if method == 'inverse_error':
            # Weight inversely proportional to error
            errors = [results.get(metric, np.inf) for results in evaluation_results.values()]
            
            # Avoid division by zero
            errors = [max(error, 1e-10) for error in errors]
            
            # Calculate inverse weights
            inverse_weights = [1.0 / error for error in errors]
            total_weight = sum(inverse_weights)
            
            # Normalize weights
            for i, name in enumerate(evaluation_results.keys()):
                self.weights[name] = inverse_weights[i] / total_weight
                
        elif method == 'rank_based':
            # Weight based on ranking
            sorted_forecasters = sorted(evaluation_results.items(), 
                                      key=lambda x: x[1].get(metric, np.inf))
            
            total_forecasters = len(sorted_forecasters)
            for i, (name, _) in enumerate(sorted_forecasters):
                # Higher rank gets higher weight
                self.weights[name] = (total_forecasters - i) / sum(range(1, total_forecasters + 1))
        
        self.logger.info(f"Updated weights: {self.weights}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble summary"""
        
        summary = {
            'num_forecasters': len(self.forecasters),
            'combination_method': self.combination_method,
            'is_fitted': self.is_fitted,
            'forecasters': {},
            'total_weight': sum(self.weights.values())
        }
        
        for name, forecaster in self.forecasters.items():
            forecaster_info = {
                'type': type(forecaster).__name__,
                'weight': self.weights[name],
                'weight_percentage': (self.weights[name] / sum(self.weights.values())) * 100 if sum(self.weights.values()) > 0 else 0
            }
            
            # Add performance history if available
            if self.performance_history[name]:
                latest_performance = self.performance_history[name][-1]
                forecaster_info['latest_performance'] = latest_performance
            
            summary['forecasters'][name] = forecaster_info
        
        return summary
# =============================================================================
# PRODUCTION FORECASTING PIPELINE
# =============================================================================

class ProductionForecastingPipeline:
    """Comprehensive production forecasting pipeline with automated model selection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.best_model = None
        self.model_performance = {}
        self.data_characteristics = {}
        self.is_fitted = False
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for forecasting pipeline"""
        logger = logging.getLogger('forecasting_pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_data_characteristics(self, ts: pd.Series) -> Dict[str, Any]:
        """Comprehensive analysis of time series characteristics"""
        
        self.logger.info("Analyzing time series characteristics...")
        
        characteristics = {
            'length': len(ts),
            'frequency': self._detect_frequency(ts),
            'missing_values': ts.isnull().sum(),
            'missing_percentage': (ts.isnull().sum() / len(ts)) * 100
        }
        
        # Statistical properties
        characteristics.update({
            'mean': ts.mean(),
            'std': ts.std(),
            'min': ts.min(),
            'max': ts.max(),
            'skewness': ts.skew(),
            'kurtosis': ts.kurtosis()
        })
        
        # Trend analysis
        trend_info = self._analyze_trend(ts)
        characteristics.update(trend_info)
        
        # Seasonality analysis
        seasonality_info = self._analyze_seasonality(ts)
        characteristics.update(seasonality_info)
        
        # Stationarity analysis
        stationarity_info = self._analyze_stationarity(ts)
        characteristics.update(stationarity_info)
        
        # Volatility analysis
        volatility_info = self._analyze_volatility(ts)
        characteristics.update(volatility_info)
        
        self.data_characteristics = characteristics
        
        return characteristics
    
    def _detect_frequency(self, ts: pd.Series) -> str:
        """Detect time series frequency"""
        
        if not isinstance(ts.index, pd.DatetimeIndex):
            return 'unknown'
        
        try:
            inferred_freq = pd.infer_freq(ts.index)
            return inferred_freq if inferred_freq else 'irregular'
        except:
            return 'irregular'
    
    def _analyze_trend(self, ts: pd.Series) -> Dict[str, Any]:
        """Analyze trend characteristics"""
        
        # Linear trend using least squares
        x = np.arange(len(ts))
        y = ts.values
        
        # Remove NaN values
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        if len(x_valid) < 2:
            return {'has_trend': False, 'trend_strength': 0.0, 'trend_direction': 'none'}
        
        # Linear regression
        slope, intercept = np.polyfit(x_valid, y_valid, 1)
        
        # Calculate R-squared
        y_pred = slope * x_valid + intercept
        ss_res = np.sum((y_valid - y_pred) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Trend strength and direction
        trend_strength = abs(r_squared)
        trend_direction = 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'flat'
        
        return {
            'has_trend': trend_strength > 0.1,
            'trend_strength': trend_strength,
            'trend_direction': trend_direction,
            'trend_slope': slope
        }
    
    def _analyze_seasonality(self, ts: pd.Series) -> Dict[str, Any]:
        """Analyze seasonality characteristics"""
        
        if len(ts) < 24:
            return {'has_seasonality': False, 'seasonal_periods': [], 'seasonal_strength': 0.0}
        
        # Autocorrelation analysis
        max_lag = min(len(ts) // 3, 100)
        autocorr_values = [ts.autocorr(lag=i) for i in range(1, max_lag)]
        
        # Find peaks in autocorrelation
        seasonal_periods = []
        for i in range(1, len(autocorr_values) - 1):
            if (autocorr_values[i] > 0.3 and 
                autocorr_values[i] > autocorr_values[i-1] and 
                autocorr_values[i] > autocorr_values[i+1]):
                seasonal_periods.append(i + 1)
        
        # FFT-based seasonality detection
        fft_periods = []
        if len(ts) > 50:
            try:
                # Remove trend
                detrended = ts - ts.rolling(window=min(12, len(ts)//4), center=True).mean()
                detrended = detrended.dropna()
                
                # Apply FFT
                fft = np.fft.fft(detrended.values)
                frequencies = np.fft.fftfreq(len(detrended))
                
                # Find significant frequencies
                magnitude = np.abs(fft)
                threshold = np.mean(magnitude) + 2 * np.std(magnitude)
                
                significant_freqs = frequencies[magnitude > threshold]
                for freq in significant_freqs:
                    if freq > 0:
                        period = int(1 / freq)
                        if 2 <= period <= len(ts) // 3:
                            fft_periods.append(period)
            except:
                pass
        
        # Combine results
        all_periods = list(set(seasonal_periods + fft_periods))
        
        # Calculate seasonal strength using decomposition
        seasonal_strength = 0.0
        if all_periods and len(ts) > 24:
            try:
                period = all_periods[0] if all_periods else 12
                decomposition = seasonal_decompose(ts, model='additive', period=period)
                
                seasonal_var = np.var(decomposition.seasonal.dropna())
                residual_var = np.var(decomposition.resid.dropna())
                seasonal_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0
            except:
                pass
        
        return {
            'has_seasonality': seasonal_strength > 0.2 or len(all_periods) > 0,
            'seasonal_periods': all_periods[:3],  # Top 3 periods
            'seasonal_strength': seasonal_strength,
            'primary_seasonal_period': all_periods[0] if all_periods else None
        }
    
    def _analyze_stationarity(self, ts: pd.Series) -> Dict[str, Any]:
        """Analyze stationarity characteristics"""
        
        if not STATSMODELS_AVAILABLE:
            return {'is_stationary': False}
        
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(ts.dropna())
            
            return {
                'is_stationary': adf_result[1] < 0.05,
                'adf_p_value': adf_result[1],
                'adf_statistic': adf_result[0]
            }
        except:
            return {'is_stationary': False, 'adf_p_value': 1.0}
    
    def _analyze_volatility(self, ts: pd.Series) -> Dict[str, Any]:
        """Analyze volatility characteristics"""
        
        # Calculate rolling volatility
        returns = ts.pct_change().dropna()
        
        if len(returns) < 10:
            return {'volatility': 0.0, 'volatility_clustering': False}
        
        # Rolling standard deviation
        rolling_vol = returns.rolling(window=min(30, len(returns)//3)).std()
        
        # Volatility of volatility (volatility clustering)
        vol_of_vol = rolling_vol.std()
        
        return {
            'volatility': returns.std(),
            'volatility_clustering': vol_of_vol > rolling_vol.mean() * 0.5,
            'volatility_of_volatility': vol_of_vol
        }
    
    def select_optimal_model(self, characteristics: Dict[str, Any]) -> str:
        """Select optimal forecasting model based on data characteristics"""
        
        self.logger.info("Selecting optimal forecasting model...")
        
        # Decision tree for model selection
        data_length = characteristics['length']
        has_seasonality = characteristics.get('has_seasonality', False)
        seasonal_strength = characteristics.get('seasonal_strength', 0.0)
        has_trend = characteristics.get('has_trend', False)
        is_stationary = characteristics.get('is_stationary', False)
        
        # Model selection logic
        if data_length < 100:
            # Limited data - use simple methods
            if has_seasonality:
                selected_model = 'sarima'
            else:
                selected_model = 'arima'
                
        elif data_length < 500:
            # Medium data - classical or Prophet
            if has_seasonality and seasonal_strength > 0.3:
                selected_model = 'prophet'
            elif is_stationary:
                selected_model = 'arima'
            else:
                selected_model = 'sarima'
                
        else:
            # Large data - can use neural networks or ensemble
            if has_seasonality and seasonal_strength > 0.4:
                selected_model = 'prophet'
            elif not is_stationary and has_trend:
                selected_model = 'lstm'
            else:
                selected_model = 'ensemble'
        
        # Override with config if specified
        if 'force_model' in self.config:
            selected_model = self.config['force_model']
        
        self.logger.info(f"Selected model: {selected_model}")
        
        return selected_model
    
    def fit(self, ts: pd.Series) -> 'ProductionForecastingPipeline':
        """Fit production forecasting pipeline with automated model selection"""
        
        self.logger.info("Starting production forecasting pipeline...")
        
        # Analyze data characteristics
        characteristics = self.analyze_data_characteristics(ts)
        
        # Select optimal model
        model_type = self.select_optimal_model(characteristics)
        
        # Initialize and fit selected model
        if model_type == 'arima':
            self.best_model = ProductionClassicalForecaster()
            self.best_model.fit(ts, seasonal_period=None)
            
        elif model_type == 'sarima':
            self.best_model = ProductionClassicalForecaster()
            seasonal_period = characteristics.get('primary_seasonal_period', 12)
            self.best_model.fit(ts, seasonal_period=seasonal_period)
            
        elif model_type == 'prophet':
            self.best_model = ProductionProphetForecaster()
            self.best_model.setup_model()
            self.best_model.add_holidays('US')
            
            # Add custom seasonality if detected
            if characteristics.get('primary_seasonal_period'):
                period = characteristics['primary_seasonal_period']
                self.best_model.add_custom_seasonality(
                    name=f'custom_{period}',
                    period=period,
                    fourier_order=min(5, period//2)
                )
            
            prophet_df = pd.DataFrame({'ds': ts.index, 'y': ts.values})
            self.best_model.fit(prophet_df)
            
        elif model_type == 'lstm':
            self.best_model = ProductionLSTMForecaster(
                sequence_length=min(60, len(ts)//4),
                lstm_units=[50, 50],
                dropout_rate=0.2
            )
            self.best_model.fit(ts, epochs=50, verbose=0)
            
        elif model_type == 'ensemble':
            self.best_model = ProductionEnsembleForecaster()
            
            # Add multiple forecasters to ensemble
            arima_forecaster = ProductionClassicalForecaster()
            self.best_model.add_forecaster('arima', arima_forecaster, weight=0.3)
            
            if PROPHET_AVAILABLE:
                prophet_forecaster = ProductionProphetForecaster()
                prophet_forecaster.setup_model()
                self.best_model.add_forecaster('prophet', prophet_forecaster, weight=0.4)
            
            if TF_AVAILABLE and len(ts) > 200:
                lstm_forecaster = ProductionLSTMForecaster(sequence_length=30)
                self.best_model.add_forecaster('lstm', lstm_forecaster, weight=0.3)
            
            self.best_model.fit(ts)
        
        self.is_fitted = True
        self.logger.info(f"Pipeline fitted successfully with {model_type} model")
        
        return self
    
    def forecast(self, steps: int, confidence_level: float = 0.95) -> Dict[str, Any]:
        """Generate production forecasts with comprehensive metadata"""
        
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before forecasting")
        
        self.logger.info(f"Generating {steps}-step forecast")
        
        # Generate forecast
        if isinstance(self.best_model, ProductionProphetForecaster):
            forecast_df = self.best_model.forecast(periods=steps)
            forecast_result = {
                'forecast': forecast_df['yhat'].tail(steps).values,
                'lower_bound': forecast_df['yhat_lower'].tail(steps).values,
                'upper_bound': forecast_df['yhat_upper'].tail(steps).values
            }
        else:
            forecast_result = self.best_model.forecast(steps, confidence_level=confidence_level)
        
        # Add metadata
        forecast_result.update({
            'model_type': type(self.best_model).__name__,
            'forecast_horizon': steps,
            'confidence_level': confidence_level,
            'data_characteristics': self.data_characteristics,
            'timestamp': datetime.now().isoformat()
        })
        
        return forecast_result
    
    def validate_performance(self, ts: pd.Series, test_size: int = 30) -> Dict[str, float]:
        """Validate forecasting performance using time series cross-validation"""
        
        self.logger.info(f"Validating performance on {test_size} test samples")
        
        # Time series split
        train_ts = ts[:-test_size]
        test_ts = ts[-test_size:]
        
        # Fit on training data
        temp_pipeline = ProductionForecastingPipeline(self.config)
        temp_pipeline.fit(train_ts)
        
        # Generate forecast
        forecast_result = temp_pipeline.forecast(test_size)
        forecast_values = forecast_result['forecast']
        
        # Calculate comprehensive metrics
        metrics = {
            'mae': mean_absolute_error(test_ts.values, forecast_values),
            'mse': mean_squared_error(test_ts.values, forecast_values),
            'rmse': np.sqrt(mean_squared_error(test_ts.values, forecast_values)),
        }
        
        # MAPE (handle zero values)
        non_zero_mask = test_ts.values != 0
        if non_zero_mask.any():
            mape = np.mean(np.abs((test_ts.values[non_zero_mask] - forecast_values[non_zero_mask]) / 
                                test_ts.values[non_zero_mask])) * 100
            metrics['mape'] = mape
        
        # SMAPE
        smape = np.mean(2 * np.abs(test_ts.values - forecast_values) / 
                       (np.abs(test_ts.values) + np.abs(forecast_values))) * 100
        metrics['smape'] = smape
        
        # Directional accuracy
        actual_direction = np.diff(test_ts.values) > 0
        forecast_direction = np.diff(forecast_values) > 0
        directional_accuracy = np.mean(actual_direction == forecast_direction) * 100
        metrics['directional_accuracy'] = directional_accuracy
        
        self.logger.info(f"Validation completed: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
        
        return metrics

    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get comprehensive pipeline summary"""
        
        summary = {
            'is_fitted': self.is_fitted,
            'model_type': type(self.best_model).__name__ if self.best_model else None,
            'data_characteristics': self.data_characteristics,
            'config': self.config
        }
        
        # Add model-specific information
        if self.best_model and hasattr(self.best_model, 'get_model_summary'):
            summary['model_details'] = self.best_model.get_model_summary()
        elif self.best_model and hasattr(self.best_model, 'get_ensemble_summary'):
            summary['model_details'] = self.best_model.get_ensemble_summary()
        
        return summary

# =============================================================================
# COMPREHENSIVE INTEGRATION TEST
# =============================================================================

def comprehensive_forecasting_test():
    """Comprehensive test of all forecasting methods with synthetic data"""
    
    print("=" * 80)
    print("COMPREHENSIVE TIME SERIES FORECASTING TEST")
    print("=" * 80)
    
    # Generate synthetic time series data
    print("\n1. Generating synthetic time series data...")
    
    # Create date range
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    # Generate complex time series with trend, seasonality, and noise
    np.random.seed(42)
    
    # Base trend
    trend = np.linspace(100, 200, len(dates))
    
    # Annual seasonality
    annual_season = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365.25)
    
    # Weekly seasonality
    weekly_season = 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 7)
    
    # Random noise
    noise = np.random.normal(0, 5, len(dates))
    
    # Combine components
    values = trend + annual_season + weekly_season + noise
    
    # Create time series
    ts = pd.Series(values, index=dates, name='synthetic_demand')
    
    print(f"Generated time series: {len(ts)} observations from {ts.index[0]} to {ts.index[-1]}")
    print(f"Statistics: Mean={ts.mean():.2f}, Std={ts.std():.2f}, Min={ts.min():.2f}, Max={ts.max():.2f}")
    
    # Split data
    split_date = '2023-10-01'
    train_ts = ts[ts.index < split_date]
    test_ts = ts[ts.index >= split_date]
    
    print(f"Training data: {len(train_ts)} observations")
    print(f"Test data: {len(test_ts)} observations")
    
    # Test 1: Classical Forecasting (ARIMA/SARIMA)
    print("\n2. Testing Classical Forecasting (ARIMA/SARIMA)...")
    
    if STATSMODELS_AVAILABLE:
        try:
            classical_forecaster = ProductionClassicalForecaster()
            classical_forecaster.fit(train_ts, seasonal_period=7)
            
            classical_forecast = classical_forecaster.forecast(len(test_ts))
            classical_mae = mean_absolute_error(test_ts.values, classical_forecast['forecast'])
            
            print(f"✅ Classical Forecasting: MAE = {classical_mae:.4f}")
            print(f"   Model: {classical_forecaster.best_model_name}")
            
            # Get diagnostics
            diagnostics = classical_forecaster.get_model_diagnostics()
            print(f"   AIC: {diagnostics['aic']:.2f}, BIC: {diagnostics['bic']:.2f}")
            
        except Exception as e:
            print(f"❌ Classical Forecasting failed: {e}")
    else:
        print("⚠️  Statsmodels not available - skipping classical forecasting")
    
    # Test 2: Prophet Forecasting
    print("\n3. Testing Prophet Forecasting...")
    
    if PROPHET_AVAILABLE:
        try:
            prophet_forecaster = ProductionProphetForecaster()
            prophet_forecaster.setup_model(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False
            )
            prophet_forecaster.add_holidays('US')
            prophet_forecaster.add_custom_seasonality('weekly', period=7, fourier_order=3)
            
            # Prepare Prophet data
            prophet_df = pd.DataFrame({
                'ds': train_ts.index,
                'y': train_ts.values
            })
            
            prophet_forecaster.fit(prophet_df)
            prophet_forecast_df = prophet_forecaster.forecast(len(test_ts))
            prophet_forecast = prophet_forecast_df['yhat'].tail(len(test_ts)).values
            
            prophet_mae = mean_absolute_error(test_ts.values, prophet_forecast)
            
            print(f"✅ Prophet Forecasting: MAE = {prophet_mae:.4f}")
            
            # Detect changepoints
            changepoints = prophet_forecaster.detect_changepoints()
            print(f"   Detected {len(changepoints['significant_changepoints'])} significant changepoints")
            
        except Exception as e:
            print(f"❌ Prophet Forecasting failed: {e}")
    else:
        print("⚠️  Prophet not available - skipping Prophet forecasting")
    
    # Test 3: LSTM Forecasting
    print("\n4. Testing LSTM Forecasting...")
    
    if TF_AVAILABLE:
        try:
            lstm_forecaster = ProductionLSTMForecaster(
                sequence_length=30,
                lstm_units=[32, 16],
                dropout_rate=0.2,
                learning_rate=0.001
            )
            
            lstm_forecaster.fit(train_ts, epochs=20, verbose=0)
            lstm_forecast = lstm_forecaster.forecast(len(test_ts))
            lstm_mae = mean_absolute_error(test_ts.values, lstm_forecast['forecast'])
            
            print(f"✅ LSTM Forecasting: MAE = {lstm_mae:.4f}")
            
            # Get model summary
            model_summary = lstm_forecaster.get_model_summary()
            print(f"   Parameters: {model_summary['total_parameters']}")
            print(f"   Best epoch: {model_summary.get('best_epoch', 'N/A')}")
            
        except Exception as e:
            print(f"❌ LSTM Forecasting failed: {e}")
    else:
        print("⚠️  TensorFlow not available - skipping LSTM forecasting")
    
    # Test 4: Ensemble Forecasting
    print("\n5. Testing Ensemble Forecasting...")
    
    try:
        ensemble_forecaster = ProductionEnsembleForecaster(combination_method='weighted_average')
        
        # Add available forecasters
        if STATSMODELS_AVAILABLE:
            arima_forecaster = ProductionClassicalForecaster()
            ensemble_forecaster.add_forecaster('arima', arima_forecaster, weight=0.4)
        
        if PROPHET_AVAILABLE:
            prophet_forecaster = ProductionProphetForecaster()
            prophet_forecaster.setup_model()
            ensemble_forecaster.add_forecaster('prophet', prophet_forecaster, weight=0.6)
        
        if len(ensemble_forecaster.forecasters) > 0:
            ensemble_forecaster.fit(train_ts)
            ensemble_forecast = ensemble_forecaster.forecast(len(test_ts))
            ensemble_mae = mean_absolute_error(test_ts.values, ensemble_forecast['forecast'])
            
            print(f"✅ Ensemble Forecasting: MAE = {ensemble_mae:.4f}")
            print(f"   Models used: {ensemble_forecast['num_models']}")
            print(f"   Combination method: {ensemble_forecast['combination_method']}")
            
            # Get ensemble summary
            summary = ensemble_forecaster.get_ensemble_summary()
            for name, info in summary['forecasters'].items():
                print(f"   {name}: weight={info['weight']:.3f} ({info['weight_percentage']:.1f}%)")
        else:
            print("⚠️  No forecasters available for ensemble")
            
    except Exception as e:
        print(f"❌ Ensemble Forecasting failed: {e}")
    
    # Test 5: Production Pipeline
    print("\n6. Testing Production Forecasting Pipeline...")
    
    try:
        config = {
            'auto_model_selection': True,
            'validation_size': 30,
            'confidence_level': 0.95
        }
        
        pipeline = ProductionForecastingPipeline(config)
        pipeline.fit(train_ts)
        
        # Generate forecast
        pipeline_forecast = pipeline.forecast(len(test_ts))
        pipeline_mae = mean_absolute_error(test_ts.values, pipeline_forecast['forecast'])
        
        print(f"✅ Production Pipeline: MAE = {pipeline_mae:.4f}")
        print(f"   Selected model: {pipeline_forecast['model_type']}")
        
        # Validate performance
        validation_metrics = pipeline.validate_performance(train_ts, test_size=30)
        print(f"   Validation RMSE: {validation_metrics['rmse']:.4f}")
        print(f"   Directional accuracy: {validation_metrics['directional_accuracy']:.1f}%")
        
        # Get pipeline summary
        summary = pipeline.get_pipeline_summary()
        characteristics = summary['data_characteristics']
        print(f"   Data length: {characteristics['length']}")
        print(f"   Has seasonality: {characteristics.get('has_seasonality', False)}")
        print(f"   Has trend: {characteristics.get('has_trend', False)}")
        print(f"   Is stationary: {characteristics.get('is_stationary', False)}")
        
    except Exception as e:
        print(f"❌ Production Pipeline failed: {e}")
    
    # Performance Comparison
    print("\n7. Performance Summary...")
    print("-" * 50)
    
    results = []
    
    if STATSMODELS_AVAILABLE:
        try:
            results.append(('Classical (ARIMA/SARIMA)', classical_mae))
        except:
            pass
    
    if PROPHET_AVAILABLE:
        try:
            results.append(('Prophet', prophet_mae))
        except:
            pass
    
    if TF_AVAILABLE:
        try:
            results.append(('LSTM', lstm_mae))
        except:
            pass
    
    try:
        results.append(('Ensemble', ensemble_mae))
    except:
        pass
    
    try:
        results.append(('Production Pipeline', pipeline_mae))
    except:
        pass
    
    if results:
        # Sort by performance
        results.sort(key=lambda x: x[1])
        
        print("Model Performance Ranking (by MAE):")
        for i, (model_name, mae) in enumerate(results, 1):
            print(f"{i}. {model_name}: {mae:.4f}")
        
        best_model = results[0][0]
        best_mae = results[0][1]
        print(f"\n🏆 Best performing model: {best_model} (MAE: {best_mae:.4f})")
    
    print("\n" + "=" * 80)
    print("FORECASTING TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution demonstrating comprehensive time series forecasting
    """
    
    print("Time Series Forecasting - Production Implementation")
    print("=" * 60)
    
    # Check library availability
    print("\nLibrary Availability Check:")
    print(f"📊 Statsmodels (ARIMA/SARIMA): {'✅ Available' if STATSMODELS_AVAILABLE else '❌ Not Available'}")
    print(f"🔮 Prophet: {'✅ Available' if PROPHET_AVAILABLE else '❌ Not Available'}")
    print(f"🧠 TensorFlow (LSTM): {'✅ Available' if TF_AVAILABLE else '❌ Not Available'}")
    
    if not any([STATSMODELS_AVAILABLE, PROPHET_AVAILABLE, TF_AVAILABLE]):
        print("\n⚠️  Warning: No forecasting libraries available!")
        print("Install with: pip install statsmodels prophet tensorflow")
        sys.exit(1)
    
    # Run comprehensive test
    try:
        success = comprehensive_forecasting_test()
        
        if success:
            print("\n🎉 All tests completed successfully!")
            print("\nKey Takeaways:")
            print("• Classical methods (ARIMA/SARIMA) work well for stationary data")
            print("• Prophet excels with strong seasonality and holidays")
            print("• LSTM captures complex non-linear patterns")
            print("• Ensemble methods combine strengths of multiple approaches")
            print("• Production pipelines automate model selection and validation")
            
            print("\nNext Steps:")
            print("• Experiment with different model configurations")
            print("• Add external regressors for improved accuracy")
            print("• Implement real-time forecasting updates")
            print("• Set up monitoring and alerting for forecast quality")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)