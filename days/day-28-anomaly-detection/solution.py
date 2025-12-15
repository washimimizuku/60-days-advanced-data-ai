"""
Day 28: Anomaly Detection - Statistical & ML-based Methods - Complete Solution

Production-ready anomaly detection system for SecureBank's fraud detection platform.
Demonstrates comprehensive anomaly detection across multiple methodologies.

This solution showcases:
- Statistical anomaly detection (Z-score, IQR, Modified Z-score)
- Machine learning methods (Isolation Forest, One-Class SVM, Autoencoders)
- Time series anomaly detection (SPC, seasonal decomposition)
- Ensemble methods with adaptive weighting
- Real-time processing pipeline with concept drift detection
- Comprehensive evaluation and benchmarking framework
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core libraries
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical libraries
from scipy import stats
from scipy.stats import zscore, ks_2samp
import logging
import time
import json

# Deep learning (handle gracefully if not available)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    from tensorflow.keras.optimizers import Adam
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Time series libraries
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# =============================================================================
# STATISTICAL ANOMALY DETECTION METHODS
# =============================================================================

class ProductionStatisticalDetector:
    """Production-grade statistical anomaly detection with comprehensive methods"""
    
    def __init__(self):
        self.fitted_params = {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for statistical detector"""
        logger = logging.getLogger('statistical_detector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def zscore_detection(self, data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Z-score based anomaly detection with robust statistics
        
        Args:
            data: Input data array
            threshold: Z-score threshold for anomaly detection
            
        Returns:
            Tuple of (anomaly_flags, z_scores)
        """
        self.logger.info(f"Applying Z-score detection with threshold {threshold}")
        
        # Handle edge cases
        if len(data) == 0:
            return np.array([]), np.array([])
        
        # Remove NaN values for calculation
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return np.full(len(data), False), np.full(len(data), np.nan)
        
        # Calculate robust statistics
        mean_val = np.mean(clean_data)
        std_val = np.std(clean_data)
        
        # Handle constant data
        if std_val == 0:
            self.logger.warning("Standard deviation is zero - no anomalies detected")
            return np.full(len(data), False), np.zeros(len(data))
        
        # Calculate Z-scores
        z_scores = np.abs((data - mean_val) / std_val)
        
        # Identify anomalies
        anomalies = z_scores > threshold
        
        # Store parameters for future use
        self.fitted_params['zscore'] = {
            'mean': mean_val,
            'std': std_val,
            'threshold': threshold
        }
        
        self.logger.info(f"Detected {np.sum(anomalies)} anomalies ({np.mean(anomalies)*100:.2f}%)")
        
        return anomalies, z_scores
    
    def iqr_detection(self, data: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        IQR-based anomaly detection with comprehensive bounds information
        
        Args:
            data: Input data array
            factor: IQR multiplication factor for outlier bounds
            
        Returns:
            Tuple of (anomaly_flags, bounds_info)
        """
        self.logger.info(f"Applying IQR detection with factor {factor}")
        
        # Handle edge cases
        if len(data) == 0:
            return np.array([]), {}
        
        # Remove NaN values for calculation
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return np.full(len(data), False), {}
        
        # Calculate quartiles
        Q1 = np.percentile(clean_data, 25)
        Q3 = np.percentile(clean_data, 75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Identify anomalies
        anomalies = (data < lower_bound) | (data > upper_bound)
        
        # Bounds information
        bounds_info = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'factor': factor
        }
        
        # Store parameters
        self.fitted_params['iqr'] = bounds_info
        
        self.logger.info(f"Detected {np.sum(anomalies)} anomalies ({np.mean(anomalies)*100:.2f}%)")
        self.logger.info(f"Bounds: [{lower_bound:.4f}, {upper_bound:.4f}]")
        
        return anomalies, bounds_info
    
    def modified_zscore_detection(self, data: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Modified Z-score using Median Absolute Deviation (MAD) for robust detection
        
        Args:
            data: Input data array
            threshold: Modified Z-score threshold
            
        Returns:
            Tuple of (anomaly_flags, modified_z_scores)
        """
        self.logger.info(f"Applying Modified Z-score detection with threshold {threshold}")
        
        # Handle edge cases
        if len(data) == 0:
            return np.array([]), np.array([])
        
        # Remove NaN values for calculation
        clean_data = data[~np.isnan(data)]
        
        if len(clean_data) == 0:
            return np.full(len(data), False), np.full(len(data), np.nan)
        
        # Calculate median and MAD
        median_val = np.median(clean_data)
        mad = np.median(np.abs(clean_data - median_val))
        
        # Handle case where MAD = 0 (all values are the same)
        if mad == 0:
            self.logger.warning("MAD is zero - using fallback method")
            # Use mean absolute deviation as fallback
            mad = np.mean(np.abs(clean_data - median_val))
            if mad == 0:
                return np.full(len(data), False), np.zeros(len(data))
        
        # Calculate modified Z-scores
        # 0.6745 is the 75th percentile of the standard normal distribution
        modified_z_scores = 0.6745 * (data - median_val) / mad
        
        # Identify anomalies
        anomalies = np.abs(modified_z_scores) > threshold
        
        # Store parameters
        self.fitted_params['modified_zscore'] = {
            'median': median_val,
            'mad': mad,
            'threshold': threshold
        }
        
        self.logger.info(f"Detected {np.sum(anomalies)} anomalies ({np.mean(anomalies)*100:.2f}%)")
        
        return anomalies, modified_z_scores
    
    def ensemble_statistical_detection(self, data: np.ndarray, 
                                     methods: List[str] = None,
                                     voting_threshold: float = 0.5,
                                     weights: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Ensemble statistical anomaly detection with weighted voting
        
        Args:
            data: Input data array
            methods: List of methods to use ['zscore', 'iqr', 'modified_zscore']
            voting_threshold: Fraction of methods that must agree for anomaly
            weights: Optional weights for each method
            
        Returns:
            Dictionary with ensemble results and individual method results
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'modified_zscore']
        
        if weights is None:
            weights = {method: 1.0 for method in methods}
        
        self.logger.info(f"Applying ensemble detection with methods: {methods}")
        
        individual_results = {}
        votes = np.zeros(len(data))
        total_weight = 0
        
        # Apply each method
        for method in methods:
            try:
                if method == 'zscore':
                    anomalies, scores = self.zscore_detection(data)
                elif method == 'iqr':
                    anomalies, bounds = self.iqr_detection(data)
                    scores = np.where(anomalies, 1.0, 0.0)  # Binary scores for IQR
                elif method == 'modified_zscore':
                    anomalies, scores = self.modified_zscore_detection(data)
                else:
                    self.logger.warning(f"Unknown method: {method}")
                    continue
                
                # Store individual results
                individual_results[method] = {
                    'anomalies': anomalies,
                    'scores': scores
                }
                
                # Add weighted votes
                method_weight = weights.get(method, 1.0)
                votes += anomalies.astype(float) * method_weight
                total_weight += method_weight
                
            except Exception as e:
                self.logger.error(f"Method {method} failed: {e}")
                continue
        
        # Calculate ensemble results
        if total_weight > 0:
            normalized_votes = votes / total_weight
            ensemble_anomalies = normalized_votes > voting_threshold
        else:
            ensemble_anomalies = np.full(len(data), False)
            normalized_votes = np.zeros(len(data))
        
        ensemble_results = {
            'ensemble_anomalies': ensemble_anomalies,
            'ensemble_scores': normalized_votes,
            'individual_results': individual_results,
            'methods_used': methods,
            'voting_threshold': voting_threshold,
            'weights': weights,
            'total_anomalies': np.sum(ensemble_anomalies),
            'anomaly_rate': np.mean(ensemble_anomalies)
        }
        
        self.logger.info(f"Ensemble detected {np.sum(ensemble_anomalies)} anomalies ({np.mean(ensemble_anomalies)*100:.2f}%)")
        
        return ensemble_results
# =============================================================================
# MACHINE LEARNING ANOMALY DETECTION METHODS
# =============================================================================

class ProductionMLDetector:
    """Production-grade ML-based anomaly detection with multiple algorithms"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_fitted = {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for ML detector"""
        logger = logging.getLogger('ml_detector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def isolation_forest_detection(self, X_train: np.ndarray, X_test: np.ndarray = None,
                                 contamination: float = 0.1, 
                                 n_estimators: int = 100,
                                 random_state: int = 42) -> Dict[str, Any]:
        """
        Isolation Forest anomaly detection with comprehensive results
        
        Args:
            X_train: Training data (normal data)
            X_test: Test data (optional, if None use X_train)
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with predictions, scores, and model info
        """
        self.logger.info(f"Training Isolation Forest with {n_estimators} estimators")
        
        # Use training data for testing if no test data provided
        if X_test is None:
            X_test = X_train
        
        # Ensure 2D arrays
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        # Initialize and fit Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        
        start_time = time.time()
        iso_forest.fit(X_train)
        fit_time = time.time() - start_time
        
        # Generate predictions and scores
        start_time = time.time()
        predictions = iso_forest.predict(X_test)
        anomaly_scores = iso_forest.decision_function(X_test)
        predict_time = time.time() - start_time
        
        # Convert predictions (-1 for anomaly, 1 for normal) to boolean
        anomalies = predictions == -1
        
        # Store model
        self.models['isolation_forest'] = iso_forest
        self.is_fitted['isolation_forest'] = True
        
        results = {
            'anomalies': anomalies,
            'predictions': predictions,
            'anomaly_scores': anomaly_scores,
            'model_params': {
                'contamination': contamination,
                'n_estimators': n_estimators,
                'random_state': random_state
            },
            'performance': {
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_anomalies': np.sum(anomalies),
                'anomaly_rate': np.mean(anomalies)
            }
        }
        
        self.logger.info(f"Isolation Forest: {np.sum(anomalies)} anomalies ({np.mean(anomalies)*100:.2f}%)")
        self.logger.info(f"Training time: {fit_time:.4f}s, Prediction time: {predict_time:.4f}s")
        
        return results
    
    def one_class_svm_detection(self, X_train: np.ndarray, X_test: np.ndarray = None,
                              kernel: str = 'rbf', nu: float = 0.1,
                              gamma: str = 'scale') -> Dict[str, Any]:
        """
        One-Class SVM anomaly detection with proper scaling
        
        Args:
            X_train: Training data (normal data)
            X_test: Test data (optional)
            kernel: SVM kernel type
            nu: Upper bound on fraction of training errors
            gamma: Kernel coefficient
            
        Returns:
            Dictionary with predictions, scores, and model info
        """
        self.logger.info(f"Training One-Class SVM with {kernel} kernel")
        
        # Use training data for testing if no test data provided
        if X_test is None:
            X_test = X_train
        
        # Ensure 2D arrays
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        # Scale the data
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize and fit One-Class SVM
        oc_svm = OneClassSVM(
            kernel=kernel,
            nu=nu,
            gamma=gamma
        )
        
        start_time = time.time()
        oc_svm.fit(X_train_scaled)
        fit_time = time.time() - start_time
        
        # Generate predictions and scores
        start_time = time.time()
        predictions = oc_svm.predict(X_test_scaled)
        decision_scores = oc_svm.decision_function(X_test_scaled)
        predict_time = time.time() - start_time
        
        # Convert predictions (-1 for anomaly, 1 for normal) to boolean
        anomalies = predictions == -1
        
        # Store model and scaler
        self.models['one_class_svm'] = oc_svm
        self.scalers['one_class_svm'] = scaler
        self.is_fitted['one_class_svm'] = True
        
        results = {
            'anomalies': anomalies,
            'predictions': predictions,
            'decision_scores': decision_scores,
            'model_params': {
                'kernel': kernel,
                'nu': nu,
                'gamma': gamma
            },
            'performance': {
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_anomalies': np.sum(anomalies),
                'anomaly_rate': np.mean(anomalies)
            }
        }
        
        self.logger.info(f"One-Class SVM: {np.sum(anomalies)} anomalies ({np.mean(anomalies)*100:.2f}%)")
        self.logger.info(f"Training time: {fit_time:.4f}s, Prediction time: {predict_time:.4f}s")
        
        return results
    
    def autoencoder_detection(self, X_train: np.ndarray, X_test: np.ndarray = None,
                            encoding_dim: int = 32, epochs: int = 100,
                            batch_size: int = 32, validation_split: float = 0.2,
                            threshold_percentile: float = 95) -> Dict[str, Any]:
        """
        Autoencoder-based anomaly detection with comprehensive training
        
        Args:
            X_train: Training data (normal data)
            X_test: Test data (optional)
            encoding_dim: Dimension of encoded representation
            epochs: Training epochs
            batch_size: Training batch size
            validation_split: Fraction of training data for validation
            threshold_percentile: Percentile for reconstruction error threshold
            
        Returns:
            Dictionary with predictions, reconstruction errors, and model info
        """
        if not TF_AVAILABLE:
            self.logger.error("TensorFlow not available for autoencoder")
            return {'error': 'TensorFlow not available'}
        
        self.logger.info(f"Training Autoencoder with encoding dimension {encoding_dim}")
        
        # Use training data for testing if no test data provided
        if X_test is None:
            X_test = X_train
        
        # Ensure 2D arrays
        if X_train.ndim == 1:
            X_train = X_train.reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        input_dim = X_train.shape[1]
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Build autoencoder architecture
        input_layer = layers.Input(shape=(input_dim,))
        
        # Encoder
        encoded = layers.Dense(min(64, input_dim * 2), activation='relu')(input_layer)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(min(64, input_dim * 2), activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        # Train the model
        start_time = time.time()
        history = autoencoder.fit(
            X_train_scaled, X_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
        )
        fit_time = time.time() - start_time
        
        # Generate reconstructions and calculate errors
        start_time = time.time()
        reconstructions = autoencoder.predict(X_test_scaled, verbose=0)
        reconstruction_errors = np.mean(np.square(X_test_scaled - reconstructions), axis=1)
        predict_time = time.time() - start_time
        
        # Determine threshold from training data reconstruction errors
        train_reconstructions = autoencoder.predict(X_train_scaled, verbose=0)
        train_errors = np.mean(np.square(X_train_scaled - train_reconstructions), axis=1)
        threshold = np.percentile(train_errors, threshold_percentile)
        
        # Identify anomalies
        anomalies = reconstruction_errors > threshold
        
        # Store model and scaler
        self.models['autoencoder'] = autoencoder
        self.scalers['autoencoder'] = scaler
        self.is_fitted['autoencoder'] = True
        
        results = {
            'anomalies': anomalies,
            'reconstruction_errors': reconstruction_errors,
            'threshold': threshold,
            'model_params': {
                'encoding_dim': encoding_dim,
                'epochs': len(history.history['loss']),
                'batch_size': batch_size,
                'threshold_percentile': threshold_percentile
            },
            'training_history': {
                'loss': history.history['loss'],
                'val_loss': history.history['val_loss']
            },
            'performance': {
                'fit_time': fit_time,
                'predict_time': predict_time,
                'total_anomalies': np.sum(anomalies),
                'anomaly_rate': np.mean(anomalies),
                'final_loss': history.history['loss'][-1],
                'final_val_loss': history.history['val_loss'][-1]
            }
        }
        
        self.logger.info(f"Autoencoder: {np.sum(anomalies)} anomalies ({np.mean(anomalies)*100:.2f}%)")
        self.logger.info(f"Training time: {fit_time:.4f}s, Prediction time: {predict_time:.4f}s")
        self.logger.info(f"Threshold: {threshold:.6f}, Final loss: {history.history['loss'][-1]:.6f}")
        
        return results
# =============================================================================
# TIME SERIES ANOMALY DETECTION METHODS
# =============================================================================

class ProductionTimeSeriesDetector:
    """Production-grade time series anomaly detection methods"""
    
    def __init__(self):
        self.models = {}
        self.fitted_params = {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for time series detector"""
        logger = logging.getLogger('timeseries_detector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def statistical_process_control(self, data: pd.Series, 
                                  window_size: int = 30,
                                  n_sigma: float = 3.0) -> Dict[str, Any]:
        """
        Statistical Process Control (SPC) anomaly detection with control charts
        
        Args:
            data: Time series data with datetime index
            window_size: Rolling window size for control limits
            n_sigma: Number of standard deviations for control limits
            
        Returns:
            Dictionary with anomaly flags, control limits, and statistics
        """
        self.logger.info(f"Applying SPC with window size {window_size} and {n_sigma} sigma limits")
        
        if len(data) < window_size:
            self.logger.warning(f"Data length ({len(data)}) less than window size ({window_size})")
            return {
                'anomalies': np.full(len(data), False),
                'control_limits': None,
                'error': 'Insufficient data for SPC analysis'
            }
        
        # Calculate rolling statistics
        rolling_mean = data.rolling(window=window_size, min_periods=1).mean()
        rolling_std = data.rolling(window=window_size, min_periods=1).std()
        
        # Calculate control limits
        upper_control_limit = rolling_mean + n_sigma * rolling_std
        lower_control_limit = rolling_mean - n_sigma * rolling_std
        
        # Identify anomalies (points outside control limits)
        anomalies = (data > upper_control_limit) | (data < lower_control_limit)
        
        # Handle initial NaN values
        anomalies = anomalies.fillna(False)
        
        # Calculate additional statistics
        center_line = rolling_mean
        
        # Store fitted parameters
        self.fitted_params['spc'] = {
            'window_size': window_size,
            'n_sigma': n_sigma,
            'final_mean': rolling_mean.iloc[-1],
            'final_std': rolling_std.iloc[-1]
        }
        
        results = {
            'anomalies': anomalies.values,
            'anomaly_indices': data.index[anomalies].tolist(),
            'control_limits': {
                'upper': upper_control_limit,
                'lower': lower_control_limit,
                'center': center_line
            },
            'statistics': {
                'total_anomalies': anomalies.sum(),
                'anomaly_rate': anomalies.mean(),
                'window_size': window_size,
                'n_sigma': n_sigma
            },
            'rolling_stats': {
                'mean': rolling_mean,
                'std': rolling_std
            }
        }
        
        self.logger.info(f"SPC detected {anomalies.sum()} anomalies ({anomalies.mean()*100:.2f}%)")
        
        return results
    
    def seasonal_decomposition_detection(self, data: pd.Series,
                                       period: int = 24,
                                       model: str = 'additive',
                                       anomaly_method: str = 'iqr') -> Dict[str, Any]:
        """
        Seasonal decomposition-based anomaly detection
        
        Args:
            data: Time series data with datetime index
            period: Seasonal period (e.g., 24 for hourly data with daily seasonality)
            model: 'additive' or 'multiplicative' decomposition
            anomaly_method: Method to apply to residuals ('iqr', 'zscore', 'modified_zscore')
            
        Returns:
            Dictionary with anomaly flags, decomposition components, and residual analysis
        """
        if not STATSMODELS_AVAILABLE:
            self.logger.error("Statsmodels not available for seasonal decomposition")
            return {'error': 'Statsmodels not available'}
        
        self.logger.info(f"Applying seasonal decomposition with period {period}")
        
        if len(data) < 2 * period:
            self.logger.warning(f"Data length ({len(data)}) less than 2 periods ({2*period})")
            return {
                'anomalies': np.full(len(data), False),
                'error': 'Insufficient data for seasonal decomposition'
            }
        
        try:
            # Perform seasonal decomposition
            decomposition = seasonal_decompose(
                data, 
                model=model, 
                period=period,
                extrapolate_trend='freq'
            )
            
            # Extract components
            trend = decomposition.trend
            seasonal = decomposition.seasonal
            residual = decomposition.resid
            
            # Apply anomaly detection to residuals
            residual_clean = residual.dropna()
            
            if len(residual_clean) == 0:
                return {
                    'anomalies': np.full(len(data), False),
                    'error': 'No valid residuals for anomaly detection'
                }
            
            # Apply specified anomaly detection method to residuals
            if anomaly_method == 'iqr':
                Q1 = residual_clean.quantile(0.25)
                Q3 = residual_clean.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                residual_anomalies = (residual < lower_bound) | (residual > upper_bound)
                
            elif anomaly_method == 'zscore':
                z_scores = np.abs(stats.zscore(residual_clean))
                threshold = 3.0
                residual_anomalies = pd.Series(False, index=data.index)
                residual_anomalies.loc[residual_clean.index] = z_scores > threshold
                
            elif anomaly_method == 'modified_zscore':
                median_val = residual_clean.median()
                mad = np.median(np.abs(residual_clean - median_val))
                if mad == 0:
                    mad = np.mean(np.abs(residual_clean - median_val))
                if mad > 0:
                    modified_z_scores = 0.6745 * (residual_clean - median_val) / mad
                    residual_anomalies = pd.Series(False, index=data.index)
                    residual_anomalies.loc[residual_clean.index] = np.abs(modified_z_scores) > 3.5
                else:
                    residual_anomalies = pd.Series(False, index=data.index)
            else:
                raise ValueError(f"Unknown anomaly method: {anomaly_method}")
            
            # Fill NaN values with False
            residual_anomalies = residual_anomalies.fillna(False)
            
            # Store fitted parameters
            self.fitted_params['seasonal_decomposition'] = {
                'period': period,
                'model': model,
                'anomaly_method': anomaly_method
            }
            
            results = {
                'anomalies': residual_anomalies.values,
                'anomaly_indices': data.index[residual_anomalies].tolist(),
                'decomposition': {
                    'trend': trend,
                    'seasonal': seasonal,
                    'residual': residual,
                    'observed': data
                },
                'residual_analysis': {
                    'residual_mean': residual_clean.mean(),
                    'residual_std': residual_clean.std(),
                    'residual_min': residual_clean.min(),
                    'residual_max': residual_clean.max()
                },
                'statistics': {
                    'total_anomalies': residual_anomalies.sum(),
                    'anomaly_rate': residual_anomalies.mean(),
                    'period': period,
                    'model': model,
                    'anomaly_method': anomaly_method
                }
            }
            
            self.logger.info(f"Seasonal decomposition detected {residual_anomalies.sum()} anomalies ({residual_anomalies.mean()*100:.2f}%)")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Seasonal decomposition failed: {e}")
            return {
                'anomalies': np.full(len(data), False),
                'error': f'Decomposition failed: {str(e)}'
            }
    
    def sliding_window_detection(self, data: pd.Series,
                               window_size: int = 50,
                               step_size: int = 1,
                               method: str = 'isolation_forest',
                               **method_kwargs) -> Dict[str, Any]:
        """
        Sliding window anomaly detection for streaming data simulation
        
        Args:
            data: Time series data
            window_size: Size of sliding window
            step_size: Step size for window movement
            method: Anomaly detection method to apply in each window
            **method_kwargs: Additional arguments for the detection method
            
        Returns:
            Dictionary with anomaly flags, window-wise results, and method info
        """
        self.logger.info(f"Applying sliding window detection with {method}")
        
        if len(data) < window_size:
            self.logger.warning(f"Data length ({len(data)}) less than window size ({window_size})")
            return {
                'anomalies': np.full(len(data), False),
                'error': 'Insufficient data for sliding window analysis'
            }
        
        # Initialize results
        anomaly_flags = np.zeros(len(data), dtype=bool)
        window_results = []
        
        # Create ML detector for window-based detection
        ml_detector = ProductionMLDetector()
        
        # Slide window through data
        for start_idx in range(0, len(data) - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_data = data.iloc[start_idx:end_idx].values.reshape(-1, 1)
            
            try:
                if method == 'isolation_forest':
                    contamination = method_kwargs.get('contamination', 0.1)
                    result = ml_detector.isolation_forest_detection(
                        window_data, 
                        contamination=contamination
                    )
                    window_anomalies = result['anomalies']
                    
                elif method == 'one_class_svm':
                    nu = method_kwargs.get('nu', 0.1)
                    result = ml_detector.one_class_svm_detection(
                        window_data,
                        nu=nu
                    )
                    window_anomalies = result['anomalies']
                    
                else:
                    # Fallback to statistical method
                    stat_detector = ProductionStatisticalDetector()
                    window_anomalies, _ = stat_detector.zscore_detection(
                        window_data.flatten(),
                        threshold=method_kwargs.get('threshold', 3.0)
                    )
                
                # Map window anomalies back to global indices
                global_indices = range(start_idx, end_idx)
                for i, is_anomaly in enumerate(window_anomalies):
                    if is_anomaly:
                        anomaly_flags[global_indices[i]] = True
                
                # Store window results
                window_results.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'anomalies_in_window': np.sum(window_anomalies),
                    'anomaly_rate': np.mean(window_anomalies)
                })
                
            except Exception as e:
                self.logger.warning(f"Window {start_idx}-{end_idx} failed: {e}")
                continue
        
        results = {
            'anomalies': anomaly_flags,
            'anomaly_indices': data.index[anomaly_flags].tolist(),
            'window_results': window_results,
            'statistics': {
                'total_anomalies': np.sum(anomaly_flags),
                'anomaly_rate': np.mean(anomaly_flags),
                'windows_processed': len(window_results),
                'window_size': window_size,
                'step_size': step_size,
                'method': method
            }
        }
        
        self.logger.info(f"Sliding window detected {np.sum(anomaly_flags)} anomalies ({np.mean(anomaly_flags)*100:.2f}%)")
        
        return results
# =============================================================================
# ENSEMBLE ANOMALY DETECTION SYSTEM
# =============================================================================

class ProductionEnsembleDetector:
    """Production ensemble anomaly detection with intelligent model combination"""
    
    def __init__(self, combination_method: str = 'weighted_voting'):
        self.detectors = {}
        self.weights = {}
        self.combination_method = combination_method
        self.performance_history = {}
        self.is_fitted = {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for ensemble detector"""
        logger = logging.getLogger('ensemble_detector')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def add_detector(self, name: str, detector: Any, weight: float = 1.0):
        """
        Add an anomaly detector to the ensemble
        
        Args:
            name: Unique name for the detector
            detector: Anomaly detector instance
            weight: Weight for ensemble combination
        """
        self.detectors[name] = detector
        self.weights[name] = weight
        self.performance_history[name] = []
        self.is_fitted[name] = False
        
        self.logger.info(f"Added detector '{name}' with weight {weight}")
    
    def fit_ensemble(self, X_train: np.ndarray, y_train: np.ndarray = None) -> 'ProductionEnsembleDetector':
        """
        Fit all detectors in the ensemble
        
        Args:
            X_train: Training data
            y_train: Training labels (optional, for supervised methods)
            
        Returns:
            Self for method chaining
        """
        self.logger.info(f"Fitting ensemble with {len(self.detectors)} detectors")
        
        successful_fits = 0
        
        for name, detector in self.detectors.items():
            try:
                self.logger.info(f"Fitting detector: {name}")
                start_time = time.time()
                
                # Handle different detector types
                if hasattr(detector, 'fit_ensemble'):
                    # For nested ensemble detectors
                    detector.fit_ensemble(X_train, y_train)
                elif hasattr(detector, 'isolation_forest_detection'):
                    # For ML detectors - fit using isolation forest as default
                    detector.isolation_forest_detection(X_train)
                elif hasattr(detector, 'zscore_detection'):
                    # For statistical detectors - just mark as fitted
                    pass
                else:
                    # Try generic fit method
                    if hasattr(detector, 'fit'):
                        if y_train is not None:
                            detector.fit(X_train, y_train)
                        else:
                            detector.fit(X_train)
                
                fit_time = time.time() - start_time
                self.is_fitted[name] = True
                successful_fits += 1
                
                self.logger.info(f"Successfully fitted {name} in {fit_time:.4f}s")
                
            except Exception as e:
                self.logger.error(f"Failed to fit {name}: {e}")
                self.weights[name] = 0.0  # Disable failed detector
                self.is_fitted[name] = False
        
        if successful_fits == 0:
            raise ValueError("No detectors could be fitted successfully")
        
        self.logger.info(f"Ensemble fitting completed: {successful_fits}/{len(self.detectors)} successful")
        
        return self
    
    def predict_ensemble(self, X_test: np.ndarray) -> Dict[str, Any]:
        """
        Generate ensemble predictions using all fitted detectors
        
        Args:
            X_test: Test data for anomaly detection
            
        Returns:
            Dictionary with ensemble predictions, individual results, and confidence scores
        """
        self.logger.info(f"Generating ensemble predictions for {len(X_test)} samples")
        
        individual_results = {}
        valid_predictions = []
        valid_scores = []
        valid_weights = []
        detector_names = []
        
        # Get predictions from each fitted detector
        for name, detector in self.detectors.items():
            if not self.is_fitted[name] or self.weights[name] == 0.0:
                continue
                
            try:
                start_time = time.time()
                
                # Get predictions based on detector type
                if hasattr(detector, 'predict_ensemble'):
                    # For nested ensemble detectors
                    result = detector.predict_ensemble(X_test)
                    predictions = result['ensemble_predictions']
                    scores = result.get('ensemble_scores', predictions.astype(float))
                    
                elif hasattr(detector, 'isolation_forest_detection'):
                    # For ML detectors
                    result = detector.isolation_forest_detection(X_test, X_test)
                    predictions = result['anomalies']
                    scores = result.get('anomaly_scores', predictions.astype(float))
                    
                elif hasattr(detector, 'zscore_detection'):
                    # For statistical detectors
                    if X_test.ndim > 1:
                        # Use first column for univariate methods
                        test_data = X_test[:, 0]
                    else:
                        test_data = X_test
                    
                    predictions, z_scores = detector.zscore_detection(test_data)
                    scores = z_scores
                    
                else:
                    # Try generic predict method
                    if hasattr(detector, 'predict'):
                        predictions = detector.predict(X_test)
                        if hasattr(detector, 'decision_function'):
                            scores = detector.decision_function(X_test)
                        else:
                            scores = predictions.astype(float)
                    else:
                        self.logger.warning(f"Detector {name} has no predict method")
                        continue
                
                predict_time = time.time() - start_time
                
                # Store individual results
                individual_results[name] = {
                    'predictions': predictions,
                    'scores': scores,
                    'predict_time': predict_time,
                    'total_anomalies': np.sum(predictions),
                    'anomaly_rate': np.mean(predictions)
                }
                
                # Add to ensemble voting
                valid_predictions.append(predictions.astype(float))
                valid_scores.append(scores)
                valid_weights.append(self.weights[name])
                detector_names.append(name)
                
                self.logger.info(f"Got predictions from {name}: {np.sum(predictions)} anomalies ({np.mean(predictions)*100:.2f}%)")
                
            except Exception as e:
                self.logger.error(f"Failed to get predictions from {name}: {e}")
                continue
        
        if not valid_predictions:
            raise ValueError("No valid predictions could be generated")
        
        # Combine predictions using specified method
        ensemble_result = self._combine_predictions(
            valid_predictions, valid_scores, valid_weights, detector_names
        )
        
        # Add individual results to ensemble result
        ensemble_result['individual_results'] = individual_results
        ensemble_result['detectors_used'] = detector_names
        ensemble_result['combination_method'] = self.combination_method
        
        return ensemble_result
    
    def _combine_predictions(self, predictions: List[np.ndarray], 
                           scores: List[np.ndarray],
                           weights: List[float],
                           names: List[str]) -> Dict[str, Any]:
        """
        Combine predictions using specified ensemble method
        
        Args:
            predictions: List of prediction arrays
            scores: List of score arrays
            weights: List of weights for each detector
            names: List of detector names
            
        Returns:
            Dictionary with combined predictions and metadata
        """
        predictions_array = np.array(predictions)
        scores_array = np.array(scores)
        weights_array = np.array(weights)
        
        if self.combination_method == 'simple_voting':
            # Simple majority voting
            ensemble_predictions = np.mean(predictions_array, axis=0) > 0.5
            ensemble_scores = np.mean(scores_array, axis=0)
            
        elif self.combination_method == 'weighted_voting':
            # Weighted voting
            weights_normalized = weights_array / np.sum(weights_array)
            weighted_predictions = np.average(predictions_array, axis=0, weights=weights_normalized)
            ensemble_predictions = weighted_predictions > 0.5
            ensemble_scores = np.average(scores_array, axis=0, weights=weights_normalized)
            
        elif self.combination_method == 'consensus':
            # Require all detectors to agree
            ensemble_predictions = np.all(predictions_array, axis=0)
            ensemble_scores = np.min(scores_array, axis=0)
            
        elif self.combination_method == 'any':
            # Any detector flags as anomaly
            ensemble_predictions = np.any(predictions_array, axis=0)
            ensemble_scores = np.max(scores_array, axis=0)
            
        else:
            # Default to weighted voting
            weights_normalized = weights_array / np.sum(weights_array)
            weighted_predictions = np.average(predictions_array, axis=0, weights=weights_normalized)
            ensemble_predictions = weighted_predictions > 0.5
            ensemble_scores = np.average(scores_array, axis=0, weights=weights_normalized)
        
        # Calculate confidence based on agreement
        agreement = np.std(predictions_array, axis=0)
        confidence = 1.0 - agreement  # Higher confidence when detectors agree
        
        return {
            'ensemble_predictions': ensemble_predictions,
            'ensemble_scores': ensemble_scores,
            'confidence': confidence,
            'agreement_std': agreement,
            'weights_used': dict(zip(names, weights)),
            'total_anomalies': np.sum(ensemble_predictions),
            'anomaly_rate': np.mean(ensemble_predictions)
        }
    
    def evaluate_ensemble(self, X_test: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate ensemble performance and individual detector performance
        
        Args:
            X_test: Test data
            y_true: True anomaly labels
            
        Returns:
            Dictionary with performance metrics for ensemble and individual detectors
        """
        self.logger.info("Evaluating ensemble performance")
        
        # Generate ensemble predictions
        ensemble_result = self.predict_ensemble(X_test)
        ensemble_predictions = ensemble_result['ensemble_predictions']
        
        # Calculate ensemble metrics
        ensemble_metrics = self._calculate_metrics(y_true, ensemble_predictions)
        
        # Calculate individual detector metrics
        individual_metrics = {}
        for name, result in ensemble_result['individual_results'].items():
            individual_predictions = result['predictions']
            individual_metrics[name] = self._calculate_metrics(y_true, individual_predictions)
        
        # Update performance history
        for name, metrics in individual_metrics.items():
            self.performance_history[name].append(metrics)
        
        evaluation_results = {
            'ensemble_metrics': ensemble_metrics,
            'individual_metrics': individual_metrics,
            'ensemble_predictions': ensemble_predictions,
            'individual_results': ensemble_result['individual_results'],
            'performance_comparison': self._compare_performance(ensemble_metrics, individual_metrics)
        }
        
        self.logger.info(f"Ensemble F1: {ensemble_metrics['f1_score']:.4f}")
        
        return evaluation_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive performance metrics"""
        
        try:
            metrics = {
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1_score': f1_score(y_true, y_pred, zero_division=0),
                'accuracy': np.mean(y_true == y_pred),
                'true_positives': np.sum((y_true == 1) & (y_pred == 1)),
                'false_positives': np.sum((y_true == 0) & (y_pred == 1)),
                'true_negatives': np.sum((y_true == 0) & (y_pred == 0)),
                'false_negatives': np.sum((y_true == 1) & (y_pred == 0))
            }
            
            # Calculate specificity
            if metrics['true_negatives'] + metrics['false_positives'] > 0:
                metrics['specificity'] = metrics['true_negatives'] / (metrics['true_negatives'] + metrics['false_positives'])
            else:
                metrics['specificity'] = 0.0
                
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            metrics = {key: 0.0 for key in ['precision', 'recall', 'f1_score', 'accuracy', 'specificity']}
        
        return metrics
    
    def _compare_performance(self, ensemble_metrics: Dict[str, float], 
                           individual_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compare ensemble performance to individual detectors"""
        
        comparison = {
            'ensemble_rank': 1,  # Start assuming ensemble is best
            'best_individual': None,
            'worst_individual': None,
            'improvement_over_best': 0.0,
            'improvement_over_average': 0.0
        }
        
        if not individual_metrics:
            return comparison
        
        # Find best and worst individual detectors
        f1_scores = {name: metrics['f1_score'] for name, metrics in individual_metrics.items()}
        best_name = max(f1_scores, key=f1_scores.get)
        worst_name = min(f1_scores, key=f1_scores.get)
        
        comparison['best_individual'] = best_name
        comparison['worst_individual'] = worst_name
        
        # Calculate improvements
        best_f1 = f1_scores[best_name]
        average_f1 = np.mean(list(f1_scores.values()))
        ensemble_f1 = ensemble_metrics['f1_score']
        
        comparison['improvement_over_best'] = ensemble_f1 - best_f1
        comparison['improvement_over_average'] = ensemble_f1 - average_f1
        
        # Calculate ensemble rank
        all_f1_scores = list(f1_scores.values()) + [ensemble_f1]
        all_f1_scores.sort(reverse=True)
        comparison['ensemble_rank'] = all_f1_scores.index(ensemble_f1) + 1
        
        return comparison
    
    def update_weights(self, performance_metrics: Dict[str, Dict[str, float]], 
                      metric: str = 'f1_score',
                      adaptation_rate: float = 0.1):
        """
        Update detector weights based on recent performance
        
        Args:
            performance_metrics: Recent performance metrics for each detector
            metric: Metric to use for weight updates
            adaptation_rate: Rate of weight adaptation
        """
        self.logger.info(f"Updating weights based on {metric}")
        
        # Extract metric values
        metric_values = {}
        for name, metrics in performance_metrics.items():
            if name in self.weights:
                metric_values[name] = metrics.get(metric, 0.0)
        
        if not metric_values:
            return
        
        # Calculate new weights based on performance
        total_performance = sum(metric_values.values())
        
        if total_performance > 0:
            for name in metric_values:
                # Performance-based weight
                performance_weight = metric_values[name] / total_performance
                
                # Exponential moving average for smooth adaptation
                current_weight = self.weights[name]
                new_weight = (1 - adaptation_rate) * current_weight + adaptation_rate * performance_weight
                
                # Ensure minimum weight for diversity
                self.weights[name] = max(new_weight, 0.01)
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            for name in self.weights:
                self.weights[name] /= total_weight
        
        self.logger.info(f"Updated weights: {self.weights}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get comprehensive ensemble summary"""
        
        summary = {
            'num_detectors': len(self.detectors),
            'combination_method': self.combination_method,
            'detectors': {},
            'total_weight': sum(self.weights.values())
        }
        
        for name, detector in self.detectors.items():
            detector_info = {
                'type': type(detector).__name__,
                'weight': self.weights[name],
                'is_fitted': self.is_fitted[name],
                'weight_percentage': (self.weights[name] / sum(self.weights.values())) * 100 if sum(self.weights.values()) > 0 else 0
            }
            
            # Add performance history if available
            if self.performance_history[name]:
                latest_performance = self.performance_history[name][-1]
                detector_info['latest_performance'] = latest_performance
            
            summary['detectors'][name] = detector_info
        
        return summary
# =============================================================================
# REAL-TIME ANOMALY DETECTION PIPELINE
# =============================================================================

class ProductionRealTimePipeline:
    """Production-ready real-time anomaly detection pipeline with concept drift handling"""
    
    def __init__(self, buffer_size: int = 1000, update_frequency: int = 100):
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.data_buffer = []
        self.anomaly_buffer = []
        self.label_buffer = []  # For supervised learning when labels are available
        self.models = {}
        self.update_counter = 0
        self.performance_metrics = {}
        self.drift_detector = None
        self.baseline_distribution = None
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for real-time pipeline"""
        logger = logging.getLogger('realtime_pipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def initialize_pipeline(self, initial_data: np.ndarray, 
                          detector_configs: Dict[str, Dict] = None):
        """
        Initialize the real-time pipeline with baseline data
        
        Args:
            initial_data: Initial training data for baseline models
            detector_configs: Configuration for different detectors
        """
        self.logger.info("Initializing real-time anomaly detection pipeline")
        
        if detector_configs is None:
            detector_configs = {
                'isolation_forest': {'contamination': 0.1, 'n_estimators': 50},
                'one_class_svm': {'nu': 0.1, 'kernel': 'rbf'},
                'statistical': {'method': 'zscore', 'threshold': 3.0}
            }
        
        # Initialize detectors
        self.models['ml_detector'] = ProductionMLDetector()
        self.models['stat_detector'] = ProductionStatisticalDetector()
        
        # Fit initial models
        if len(initial_data) > 0:
            # Fit ML models
            if 'isolation_forest' in detector_configs:
                config = detector_configs['isolation_forest']
                self.models['ml_detector'].isolation_forest_detection(
                    initial_data, contamination=config.get('contamination', 0.1)
                )
            
            # Store baseline distribution for drift detection
            self.baseline_distribution = {
                'mean': np.mean(initial_data, axis=0),
                'std': np.std(initial_data, axis=0),
                'quantiles': np.percentile(initial_data, [25, 50, 75], axis=0)
            }
        
        self.logger.info("Pipeline initialized successfully")
    
    def process_data_point(self, data_point: np.ndarray, 
                          timestamp: datetime = None,
                          true_label: bool = None) -> Dict[str, Any]:
        """
        Process a single data point in real-time
        
        Args:
            data_point: Feature vector for the data point
            timestamp: Timestamp of the data point
            true_label: True anomaly label (if available for evaluation)
            
        Returns:
            Dictionary with anomaly detection results and metadata
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Ensure data point is 2D array
        if data_point.ndim == 1:
            data_point = data_point.reshape(1, -1)
        
        # Add to buffer
        self.data_buffer.append({
            'data': data_point,
            'timestamp': timestamp,
            'true_label': true_label
        })
        
        # Maintain buffer size
        if len(self.data_buffer) > self.buffer_size:
            self.data_buffer.pop(0)
        
        # Apply anomaly detection
        detection_results = self._detect_anomaly(data_point, timestamp)
        
        # Store anomaly result
        self.anomaly_buffer.append({
            'timestamp': timestamp,
            'is_anomaly': detection_results['is_anomaly'],
            'confidence': detection_results['confidence'],
            'scores': detection_results['scores']
        })
        
        # Maintain anomaly buffer size
        if len(self.anomaly_buffer) > self.buffer_size:
            self.anomaly_buffer.pop(0)
        
        # Update models periodically
        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self._update_models()
            self.update_counter = 0
        
        # Add performance metrics if true label is available
        if true_label is not None:
            self._update_performance_metrics(detection_results['is_anomaly'], true_label)
        
        return detection_results
    
    def _detect_anomaly(self, data_point: np.ndarray, timestamp: datetime) -> Dict[str, Any]:
        """Apply anomaly detection to a single data point"""
        
        results = {
            'timestamp': timestamp,
            'is_anomaly': False,
            'confidence': 0.0,
            'scores': {},
            'methods_used': []
        }
        
        anomaly_votes = []
        confidence_scores = []
        
        # Apply ML-based detection
        try:
            if 'ml_detector' in self.models and hasattr(self.models['ml_detector'], 'models'):
                if 'isolation_forest' in self.models['ml_detector'].models:
                    iso_model = self.models['ml_detector'].models['isolation_forest']
                    prediction = iso_model.predict(data_point)[0]
                    score = iso_model.decision_function(data_point)[0]
                    
                    is_anomaly = prediction == -1
                    anomaly_votes.append(is_anomaly)
                    confidence_scores.append(abs(score))
                    results['scores']['isolation_forest'] = score
                    results['methods_used'].append('isolation_forest')
        except Exception as e:
            self.logger.debug(f"Isolation Forest detection failed: {e}")
        
        # Apply statistical detection
        try:
            if len(self.data_buffer) > 10:
                recent_data = np.array([item['data'].flatten() for item in self.data_buffer[-50:]])
                if data_point.ndim > 1:
                    current_value = data_point.flatten()[0]
                else:
                    current_value = data_point[0]
                
                # Z-score detection
                anomaly_flag, z_score = self.models['stat_detector'].zscore_detection(
                    np.append(recent_data[:, 0], current_value)
                )
                
                is_anomaly = anomaly_flag[-1]  # Last element is current point
                anomaly_votes.append(is_anomaly)
                confidence_scores.append(abs(z_score[-1]))
                results['scores']['zscore'] = z_score[-1]
                results['methods_used'].append('zscore')
        except Exception as e:
            self.logger.debug(f"Statistical detection failed: {e}")
        
        # Combine results
        if anomaly_votes:
            # Majority voting
            results['is_anomaly'] = sum(anomaly_votes) > len(anomaly_votes) / 2
            results['confidence'] = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return results
    
    def _update_models(self, retrain: bool = False):
        """Update anomaly detection models with recent data"""
        
        if len(self.data_buffer) < 50:
            return
        
        self.logger.info("Updating models with recent data")
        
        try:
            # Extract recent normal data (assuming most data is normal)
            recent_data = np.array([item['data'] for item in self.data_buffer[-200:]])
            recent_data = recent_data.reshape(len(recent_data), -1)
            
            # Filter out likely anomalies for training
            if len(self.anomaly_buffer) >= len(recent_data):
                recent_anomalies = [item['is_anomaly'] for item in self.anomaly_buffer[-len(recent_data):]]
                normal_mask = np.array([not anomaly for anomaly in recent_anomalies])
                if np.sum(normal_mask) > 10:  # Ensure we have enough normal data
                    normal_data = recent_data[normal_mask]
                else:
                    normal_data = recent_data
            else:
                normal_data = recent_data
            
            # Update Isolation Forest
            if retrain or 'isolation_forest' not in self.models['ml_detector'].models:
                self.models['ml_detector'].isolation_forest_detection(
                    normal_data, contamination=0.1, n_estimators=50
                )
            
            self.logger.info(f"Models updated with {len(normal_data)} normal samples")
            
        except Exception as e:
            self.logger.error(f"Model update failed: {e}")
    
    def detect_concept_drift(self) -> Dict[str, Any]:
        """
        Detect if the data distribution has changed significantly
        
        Returns:
            Dictionary with drift detection results and recommended actions
        """
        if len(self.data_buffer) < 100 or self.baseline_distribution is None:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Extract recent data
        recent_data = np.array([item['data'] for item in self.data_buffer[-100:]])
        recent_data = recent_data.reshape(len(recent_data), -1)
        
        drift_results = {
            'drift_detected': False,
            'drift_score': 0.0,
            'statistical_tests': {},
            'recommendations': []
        }
        
        try:
            # Statistical tests for drift detection
            baseline_mean = self.baseline_distribution['mean']
            baseline_std = self.baseline_distribution['std']
            
            recent_mean = np.mean(recent_data, axis=0)
            recent_std = np.std(recent_data, axis=0)
            
            # Mean shift detection
            mean_shift = np.abs(recent_mean - baseline_mean) / (baseline_std + 1e-8)
            mean_drift_score = np.max(mean_shift)
            
            # Variance change detection
            variance_ratio = recent_std / (baseline_std + 1e-8)
            variance_drift_score = np.max(np.abs(np.log(variance_ratio + 1e-8)))
            
            # Kolmogorov-Smirnov test for distribution change
            ks_statistics = []
            for i in range(min(recent_data.shape[1], len(baseline_mean))):
                baseline_sample = np.random.normal(
                    baseline_mean[i], baseline_std[i], 1000
                )
                ks_stat, ks_p_value = ks_2samp(baseline_sample, recent_data[:, i])
                ks_statistics.append(ks_stat)
            
            ks_drift_score = np.max(ks_statistics) if ks_statistics else 0.0
            
            # Combine drift scores
            drift_results['drift_score'] = max(mean_drift_score, variance_drift_score, ks_drift_score)
            drift_results['statistical_tests'] = {
                'mean_shift_score': mean_drift_score,
                'variance_drift_score': variance_drift_score,
                'ks_drift_score': ks_drift_score
            }
            
            # Determine if drift is significant
            drift_threshold = 2.0  # Configurable threshold
            if drift_results['drift_score'] > drift_threshold:
                drift_results['drift_detected'] = True
                drift_results['recommendations'] = [
                    'Retrain models with recent data',
                    'Update baseline distribution',
                    'Adjust detection thresholds',
                    'Increase monitoring frequency'
                ]
            
            self.logger.info(f"Drift detection: score={drift_results['drift_score']:.4f}, detected={drift_results['drift_detected']}")
            
        except Exception as e:
            self.logger.error(f"Drift detection failed: {e}")
            drift_results['error'] = str(e)
        
        return drift_results
    
    def _update_performance_metrics(self, predicted: bool, actual: bool):
        """Update performance metrics with new prediction"""
        
        if 'predictions' not in self.performance_metrics:
            self.performance_metrics['predictions'] = []
            self.performance_metrics['actuals'] = []
        
        self.performance_metrics['predictions'].append(predicted)
        self.performance_metrics['actuals'].append(actual)
        
        # Keep only recent predictions for performance calculation
        max_history = 1000
        if len(self.performance_metrics['predictions']) > max_history:
            self.performance_metrics['predictions'] = self.performance_metrics['predictions'][-max_history:]
            self.performance_metrics['actuals'] = self.performance_metrics['actuals'][-max_history:]
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of the real-time pipeline
        
        Returns:
            Dictionary with pipeline health, performance, and operational metrics
        """
        status = {
            'buffer_status': {
                'data_buffer_size': len(self.data_buffer),
                'anomaly_buffer_size': len(self.anomaly_buffer),
                'buffer_capacity': self.buffer_size,
                'buffer_utilization': len(self.data_buffer) / self.buffer_size
            },
            'processing_stats': {
                'update_counter': self.update_counter,
                'update_frequency': self.update_frequency,
                'updates_until_next': self.update_frequency - self.update_counter
            },
            'model_status': {
                'models_loaded': list(self.models.keys()),
                'baseline_available': self.baseline_distribution is not None
            }
        }
        
        # Add performance metrics if available
        if self.performance_metrics and len(self.performance_metrics['predictions']) > 0:
            predictions = np.array(self.performance_metrics['predictions'])
            actuals = np.array(self.performance_metrics['actuals'])
            
            status['performance_metrics'] = {
                'total_predictions': len(predictions),
                'accuracy': np.mean(predictions == actuals),
                'precision': precision_score(actuals, predictions, zero_division=0),
                'recall': recall_score(actuals, predictions, zero_division=0),
                'f1_score': f1_score(actuals, predictions, zero_division=0)
            }
        
        # Add recent anomaly statistics
        if self.anomaly_buffer:
            recent_anomalies = [item['is_anomaly'] for item in self.anomaly_buffer[-100:]]
            status['anomaly_stats'] = {
                'recent_anomaly_rate': np.mean(recent_anomalies),
                'total_anomalies_detected': sum(item['is_anomaly'] for item in self.anomaly_buffer),
                'average_confidence': np.mean([item['confidence'] for item in self.anomaly_buffer])
            }
        
        # Check for concept drift
        drift_status = self.detect_concept_drift()
        status['drift_status'] = drift_status
        
        return status
# =============================================================================
# COMPREHENSIVE EVALUATION AND TESTING FRAMEWORK
# =============================================================================

def generate_synthetic_anomaly_data(n_samples: int = 10000, 
                                  anomaly_rate: float = 0.05,
                                  data_type: str = 'multivariate',
                                  random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with known anomalies for comprehensive testing
    
    Args:
        n_samples: Number of samples to generate
        anomaly_rate: Proportion of anomalies in the data
        data_type: Type of data ('univariate', 'multivariate', 'timeseries')
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (data, labels) where labels indicate anomalies (1=anomaly, 0=normal)
    """
    np.random.seed(random_state)
    
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies
    
    if data_type == 'univariate':
        # Generate 1D data
        normal_data = np.random.normal(0, 1, n_normal)
        
        # Generate anomalies with different patterns
        anomaly_data = []
        # Point anomalies (extreme values)
        anomaly_data.extend(np.random.normal(0, 1, n_anomalies // 3) + np.random.choice([-5, 5], n_anomalies // 3))
        # Shift anomalies
        anomaly_data.extend(np.random.normal(3, 0.5, n_anomalies // 3))
        # Scale anomalies
        remaining = n_anomalies - len(anomaly_data)
        anomaly_data.extend(np.random.normal(0, 3, remaining))
        
        # Combine and shuffle
        data = np.concatenate([normal_data, anomaly_data])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices].reshape(-1, 1)
        labels = labels[indices]
        
    elif data_type == 'multivariate':
        # Generate correlated multivariate normal data
        n_features = 5
        
        # Normal data with correlation structure
        correlation_matrix = np.random.rand(n_features, n_features)
        correlation_matrix = correlation_matrix @ correlation_matrix.T  # Make positive definite
        normal_data = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=correlation_matrix,
            size=n_normal
        )
        
        # Generate different types of anomalies
        anomaly_data = []
        
        # Point anomalies (extreme values in one or more dimensions)
        n_point = n_anomalies // 3
        point_anomalies = np.random.multivariate_normal(
            mean=np.zeros(n_features),
            cov=correlation_matrix,
            size=n_point
        )
        # Add extreme values to random dimensions
        for i in range(n_point):
            n_extreme_dims = np.random.randint(1, n_features + 1)
            extreme_dims = np.random.choice(n_features, n_extreme_dims, replace=False)
            point_anomalies[i, extreme_dims] += np.random.choice([-3, 3], n_extreme_dims) * np.random.uniform(2, 4, n_extreme_dims)
        anomaly_data.append(point_anomalies)
        
        # Contextual anomalies (break correlation structure)
        n_contextual = n_anomalies // 3
        contextual_anomalies = np.random.normal(0, 1, (n_contextual, n_features))
        anomaly_data.append(contextual_anomalies)
        
        # Collective anomalies (shifted distribution)
        n_collective = n_anomalies - n_point - n_contextual
        collective_anomalies = np.random.multivariate_normal(
            mean=np.ones(n_features) * 2,  # Shifted mean
            cov=correlation_matrix * 0.5,  # Different covariance
            size=n_collective
        )
        anomaly_data.append(collective_anomalies)
        
        # Combine all data
        anomaly_data = np.vstack(anomaly_data)
        data = np.vstack([normal_data, anomaly_data])
        labels = np.concatenate([np.zeros(n_normal), np.ones(n_anomalies)])
        
        # Shuffle
        indices = np.random.permutation(len(data))
        data = data[indices]
        labels = labels[indices]
        
    elif data_type == 'timeseries':
        # Generate time series with trend, seasonality, and anomalies
        time_points = np.arange(n_samples)
        
        # Base signal with trend and seasonality
        trend = 0.001 * time_points
        seasonality = 2 * np.sin(2 * np.pi * time_points / 100) + np.sin(2 * np.pi * time_points / 20)
        noise = np.random.normal(0, 0.5, n_samples)
        
        normal_signal = trend + seasonality + noise
        
        # Inject anomalies
        labels = np.zeros(n_samples)
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['point', 'level_shift', 'trend_change'])
            
            if anomaly_type == 'point':
                # Point anomaly
                normal_signal[idx] += np.random.choice([-1, 1]) * np.random.uniform(3, 6)
            elif anomaly_type == 'level_shift':
                # Level shift anomaly
                shift_duration = min(20, n_samples - idx)
                shift_magnitude = np.random.choice([-1, 1]) * np.random.uniform(2, 4)
                normal_signal[idx:idx+shift_duration] += shift_magnitude
            elif anomaly_type == 'trend_change':
                # Trend change anomaly
                change_duration = min(30, n_samples - idx)
                trend_change = np.random.choice([-1, 1]) * np.random.uniform(0.05, 0.1)
                normal_signal[idx:idx+change_duration] += trend_change * np.arange(change_duration)
            
            labels[idx] = 1
        
        data = normal_signal.reshape(-1, 1)
        
    else:
        raise ValueError(f"Unknown data type: {data_type}")
    
    return data, labels.astype(int)

def comprehensive_evaluation(detectors: List[Tuple[str, Any]], 
                           X_test: np.ndarray, 
                           y_true: np.ndarray,
                           include_timing: bool = True) -> pd.DataFrame:
    """
    Perform comprehensive evaluation of multiple anomaly detectors
    
    Args:
        detectors: List of (name, detector) tuples
        X_test: Test data
        y_true: True anomaly labels
        include_timing: Whether to measure execution time
        
    Returns:
        DataFrame with performance metrics for each detector
    """
    results = []
    
    for name, detector in detectors:
        try:
            start_time = time.time()
            
            # Get predictions based on detector type
            if hasattr(detector, 'predict_ensemble'):
                # Ensemble detector
                result = detector.predict_ensemble(X_test)
                predictions = result['ensemble_predictions']
                
            elif hasattr(detector, 'isolation_forest_detection'):
                # ML detector
                result = detector.isolation_forest_detection(X_test, X_test)
                predictions = result['anomalies']
                
            elif hasattr(detector, 'zscore_detection'):
                # Statistical detector
                if X_test.ndim > 1:
                    test_data = X_test[:, 0]  # Use first column for univariate methods
                else:
                    test_data = X_test
                predictions, _ = detector.zscore_detection(test_data)
                
            else:
                # Try generic predict method
                predictions = detector.predict(X_test)
                if hasattr(predictions, 'flatten'):
                    predictions = predictions.flatten()
                # Convert to boolean if needed
                if predictions.dtype != bool:
                    predictions = predictions == -1  # Assume -1 indicates anomaly
            
            execution_time = time.time() - start_time
            
            # Calculate metrics
            precision = precision_score(y_true, predictions, zero_division=0)
            recall = recall_score(y_true, predictions, zero_division=0)
            f1 = f1_score(y_true, predictions, zero_division=0)
            accuracy = np.mean(y_true == predictions)
            
            # Calculate additional metrics
            tn = np.sum((y_true == 0) & (predictions == 0))
            fp = np.sum((y_true == 0) & (predictions == 1))
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            result_row = {
                'Detector': name,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Accuracy': accuracy,
                'Specificity': specificity,
                'True Positives': np.sum((y_true == 1) & (predictions == 1)),
                'False Positives': fp,
                'True Negatives': tn,
                'False Negatives': np.sum((y_true == 1) & (predictions == 0)),
                'Predicted Anomalies': np.sum(predictions),
                'Actual Anomalies': np.sum(y_true)
            }
            
            if include_timing:
                result_row['Execution Time (s)'] = execution_time
                result_row['Samples per Second'] = len(X_test) / execution_time if execution_time > 0 else np.inf
            
            results.append(result_row)
            
        except Exception as e:
            print(f"Error evaluating {name}: {e}")
            # Add error row
            error_row = {
                'Detector': name,
                'Error': str(e)
            }
            results.append(error_row)
    
    return pd.DataFrame(results)

def plot_anomaly_detection_results(data: np.ndarray, 
                                  anomalies: np.ndarray, 
                                  scores: np.ndarray = None,
                                  true_anomalies: np.ndarray = None,
                                  title: str = "Anomaly Detection Results",
                                  figsize: Tuple[int, int] = (15, 10)):
    """
    Create comprehensive visualization of anomaly detection results
    
    Args:
        data: Original data (1D or 2D)
        anomalies: Boolean array indicating detected anomalies
        scores: Anomaly scores (optional)
        true_anomalies: True anomaly labels (optional)
        title: Plot title
        figsize: Figure size
    """
    if data.ndim > 1 and data.shape[1] > 2:
        # For high-dimensional data, use PCA for visualization
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
    elif data.ndim > 1 and data.shape[1] == 2:
        data_2d = data
    else:
        # 1D data - create time series plot
        data_2d = np.column_stack([np.arange(len(data)), data.flatten()])
    
    # Determine number of subplots
    n_plots = 2 if true_anomalies is not None else 1
    if scores is not None:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=figsize)
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Plot detected anomalies
    ax = axes[plot_idx]
    normal_mask = ~anomalies
    anomaly_mask = anomalies
    
    ax.scatter(data_2d[normal_mask, 0], data_2d[normal_mask, 1], 
              c='blue', alpha=0.6, s=20, label='Normal')
    ax.scatter(data_2d[anomaly_mask, 0], data_2d[anomaly_mask, 1], 
              c='red', alpha=0.8, s=50, label='Detected Anomalies', marker='x')
    
    ax.set_title(f'{title} - Detected Anomalies')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # Plot true anomalies if available
    if true_anomalies is not None:
        ax = axes[plot_idx]
        true_normal_mask = ~true_anomalies.astype(bool)
        true_anomaly_mask = true_anomalies.astype(bool)
        
        ax.scatter(data_2d[true_normal_mask, 0], data_2d[true_normal_mask, 1], 
                  c='lightblue', alpha=0.6, s=20, label='True Normal')
        ax.scatter(data_2d[true_anomaly_mask, 0], data_2d[true_anomaly_mask, 1], 
                  c='darkred', alpha=0.8, s=50, label='True Anomalies', marker='s')
        
        # Highlight detection performance
        tp_mask = anomalies & true_anomalies.astype(bool)
        fp_mask = anomalies & ~true_anomalies.astype(bool)
        fn_mask = ~anomalies & true_anomalies.astype(bool)
        
        if np.any(tp_mask):
            ax.scatter(data_2d[tp_mask, 0], data_2d[tp_mask, 1], 
                      c='green', alpha=1.0, s=100, label='True Positives', marker='o', 
                      facecolors='none', edgecolors='green', linewidth=2)
        if np.any(fp_mask):
            ax.scatter(data_2d[fp_mask, 0], data_2d[fp_mask, 1], 
                      c='orange', alpha=1.0, s=100, label='False Positives', marker='o',
                      facecolors='none', edgecolors='orange', linewidth=2)
        if np.any(fn_mask):
            ax.scatter(data_2d[fn_mask, 0], data_2d[fn_mask, 1], 
                      c='purple', alpha=1.0, s=100, label='False Negatives', marker='o',
                      facecolors='none', edgecolors='purple', linewidth=2)
        
        ax.set_title(f'{title} - True vs Detected Anomalies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # Plot anomaly scores if available
    if scores is not None:
        ax = axes[plot_idx]
        
        if data.ndim == 1 or data.shape[1] == 1:
            # Time series plot for scores
            ax.plot(scores, alpha=0.7, label='Anomaly Scores')
            ax.scatter(np.where(anomalies)[0], scores[anomalies], 
                      c='red', s=50, label='Detected Anomalies', marker='x')
            ax.set_xlabel('Time/Index')
            ax.set_ylabel('Anomaly Score')
        else:
            # Scatter plot with color-coded scores
            scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], 
                               c=scores, cmap='viridis', alpha=0.7, s=30)
            plt.colorbar(scatter, ax=ax, label='Anomaly Score')
            
            # Highlight detected anomalies
            ax.scatter(data_2d[anomalies, 0], data_2d[anomalies, 1], 
                      c='red', alpha=1.0, s=100, label='Detected Anomalies', 
                      marker='o', facecolors='none', edgecolors='red', linewidth=2)
        
        ax.set_title(f'{title} - Anomaly Scores')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# COMPREHENSIVE INTEGRATION TEST
# =============================================================================

def comprehensive_anomaly_detection_test():
    """Comprehensive test of all anomaly detection methods"""
    
    print("=" * 80)
    print("COMPREHENSIVE ANOMALY DETECTION TEST")
    print("=" * 80)
    
    # Test 1: Statistical Methods
    print("\n1. Testing Statistical Anomaly Detection Methods...")
    print("-" * 50)
    
    # Generate test data
    data_1d, labels_1d = generate_synthetic_anomaly_data(
        n_samples=1000, anomaly_rate=0.05, data_type='univariate'
    )
    
    stat_detector = ProductionStatisticalDetector()
    
    # Test individual methods
    print("Testing Z-score detection...")
    anomalies_z, scores_z = stat_detector.zscore_detection(data_1d.flatten())
    precision_z = precision_score(labels_1d, anomalies_z, zero_division=0)
    recall_z = recall_score(labels_1d, anomalies_z, zero_division=0)
    print(f"Z-score: Precision={precision_z:.3f}, Recall={recall_z:.3f}")
    
    print("Testing IQR detection...")
    anomalies_iqr, bounds_iqr = stat_detector.iqr_detection(data_1d.flatten())
    precision_iqr = precision_score(labels_1d, anomalies_iqr, zero_division=0)
    recall_iqr = recall_score(labels_1d, anomalies_iqr, zero_division=0)
    print(f"IQR: Precision={precision_iqr:.3f}, Recall={recall_iqr:.3f}")
    
    print("Testing Modified Z-score detection...")
    anomalies_mod, scores_mod = stat_detector.modified_zscore_detection(data_1d.flatten())
    precision_mod = precision_score(labels_1d, anomalies_mod, zero_division=0)
    recall_mod = recall_score(labels_1d, anomalies_mod, zero_division=0)
    print(f"Modified Z-score: Precision={precision_mod:.3f}, Recall={recall_mod:.3f}")
    
    # Test ensemble statistical methods
    print("Testing ensemble statistical detection...")
    ensemble_stat_result = stat_detector.ensemble_statistical_detection(data_1d.flatten())
    ensemble_anomalies = ensemble_stat_result['ensemble_anomalies']
    precision_ens = precision_score(labels_1d, ensemble_anomalies, zero_division=0)
    recall_ens = recall_score(labels_1d, ensemble_anomalies, zero_division=0)
    print(f"Statistical Ensemble: Precision={precision_ens:.3f}, Recall={recall_ens:.3f}")
    
    # Test 2: Machine Learning Methods
    print("\n2. Testing Machine Learning Anomaly Detection Methods...")
    print("-" * 50)
    
    # Generate multivariate test data
    data_multi, labels_multi = generate_synthetic_anomaly_data(
        n_samples=1000, anomaly_rate=0.08, data_type='multivariate'
    )
    
    # Split data for training/testing
    X_train, X_test, y_train, y_test = train_test_split(
        data_multi, labels_multi, test_size=0.3, random_state=42, stratify=labels_multi
    )
    
    # Use only normal data for training (unsupervised learning)
    X_train_normal = X_train[y_train == 0]
    
    ml_detector = ProductionMLDetector()
    
    # Test Isolation Forest
    print("Testing Isolation Forest...")
    iso_result = ml_detector.isolation_forest_detection(X_train_normal, X_test)
    precision_iso = precision_score(y_test, iso_result['anomalies'], zero_division=0)
    recall_iso = recall_score(y_test, iso_result['anomalies'], zero_division=0)
    print(f"Isolation Forest: Precision={precision_iso:.3f}, Recall={recall_iso:.3f}")
    print(f"Training time: {iso_result['performance']['fit_time']:.4f}s")
    
    # Test One-Class SVM
    print("Testing One-Class SVM...")
    svm_result = ml_detector.one_class_svm_detection(X_train_normal, X_test)
    precision_svm = precision_score(y_test, svm_result['anomalies'], zero_division=0)
    recall_svm = recall_score(y_test, svm_result['anomalies'], zero_division=0)
    print(f"One-Class SVM: Precision={precision_svm:.3f}, Recall={recall_svm:.3f}")
    print(f"Training time: {svm_result['performance']['fit_time']:.4f}s")
    
    # Test Autoencoder (if TensorFlow is available)
    if TF_AVAILABLE:
        print("Testing Autoencoder...")
        ae_result = ml_detector.autoencoder_detection(
            X_train_normal, X_test, epochs=20, encoding_dim=3
        )
        if 'error' not in ae_result:
            precision_ae = precision_score(y_test, ae_result['anomalies'], zero_division=0)
            recall_ae = recall_score(y_test, ae_result['anomalies'], zero_division=0)
            print(f"Autoencoder: Precision={precision_ae:.3f}, Recall={recall_ae:.3f}")
            print(f"Training time: {ae_result['performance']['fit_time']:.4f}s")
        else:
            print(f"Autoencoder failed: {ae_result['error']}")
    else:
        print("Autoencoder skipped (TensorFlow not available)")
    
    # Test 3: Time Series Methods
    print("\n3. Testing Time Series Anomaly Detection Methods...")
    print("-" * 50)
    
    # Generate time series data
    ts_data, ts_labels = generate_synthetic_anomaly_data(
        n_samples=500, anomaly_rate=0.06, data_type='timeseries'
    )
    
    # Create pandas Series with datetime index
    dates = pd.date_range(start='2023-01-01', periods=len(ts_data), freq='H')
    ts_series = pd.Series(ts_data.flatten(), index=dates)
    
    ts_detector = ProductionTimeSeriesDetector()
    
    # Test Statistical Process Control
    print("Testing Statistical Process Control...")
    spc_result = ts_detector.statistical_process_control(ts_series, window_size=24)
    if 'error' not in spc_result:
        precision_spc = precision_score(ts_labels, spc_result['anomalies'], zero_division=0)
        recall_spc = recall_score(ts_labels, spc_result['anomalies'], zero_division=0)
        print(f"SPC: Precision={precision_spc:.3f}, Recall={recall_spc:.3f}")
    else:
        print(f"SPC failed: {spc_result['error']}")
    
    # Test Seasonal Decomposition
    if STATSMODELS_AVAILABLE:
        print("Testing Seasonal Decomposition...")
        seasonal_result = ts_detector.seasonal_decomposition_detection(
            ts_series, period=24, model='additive'
        )
        if 'error' not in seasonal_result:
            precision_seas = precision_score(ts_labels, seasonal_result['anomalies'], zero_division=0)
            recall_seas = recall_score(ts_labels, seasonal_result['anomalies'], zero_division=0)
            print(f"Seasonal Decomposition: Precision={precision_seas:.3f}, Recall={recall_seas:.3f}")
        else:
            print(f"Seasonal Decomposition failed: {seasonal_result['error']}")
    else:
        print("Seasonal Decomposition skipped (Statsmodels not available)")
    
    # Test 4: Ensemble Methods
    print("\n4. Testing Ensemble Anomaly Detection...")
    print("-" * 50)
    
    ensemble_detector = ProductionEnsembleDetector()
    
    # Add detectors to ensemble
    ensemble_detector.add_detector('statistical', stat_detector, weight=0.3)
    ensemble_detector.add_detector('isolation_forest', ml_detector, weight=0.4)
    
    # Fit ensemble
    ensemble_detector.fit_ensemble(X_train_normal)
    
    # Test ensemble prediction
    ensemble_result = ensemble_detector.predict_ensemble(X_test)
    precision_ensemble = precision_score(y_test, ensemble_result['ensemble_predictions'], zero_division=0)
    recall_ensemble = recall_score(y_test, ensemble_result['ensemble_predictions'], zero_division=0)
    print(f"Ensemble: Precision={precision_ensemble:.3f}, Recall={recall_ensemble:.3f}")
    print(f"Confidence: {np.mean(ensemble_result['confidence']):.3f}")
    
    # Test 5: Real-time Pipeline
    print("\n5. Testing Real-time Anomaly Detection Pipeline...")
    print("-" * 50)
    
    pipeline = ProductionRealTimePipeline(buffer_size=200, update_frequency=50)
    pipeline.initialize_pipeline(X_train_normal)
    
    # Simulate real-time processing
    anomaly_count = 0
    for i, (data_point, true_label) in enumerate(zip(X_test[:100], y_test[:100])):
        result = pipeline.process_data_point(data_point, true_label=bool(true_label))
        if result['is_anomaly']:
            anomaly_count += 1
    
    print(f"Real-time pipeline detected {anomaly_count} anomalies in 100 samples")
    
    # Get pipeline status
    status = pipeline.get_pipeline_status()
    print(f"Pipeline status: {status['buffer_status']['buffer_utilization']:.2f} buffer utilization")
    
    if 'performance_metrics' in status:
        print(f"Pipeline accuracy: {status['performance_metrics']['accuracy']:.3f}")
    
    # Test concept drift detection
    drift_result = pipeline.detect_concept_drift()
    print(f"Concept drift detected: {drift_result['drift_detected']}")
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANOMALY DETECTION TEST COMPLETED!")
    print("=" * 80)
    
    return True

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    """
    Main execution demonstrating comprehensive anomaly detection system
    """
    
    print("Anomaly Detection - Production Implementation")
    print("=" * 60)
    
    # Check library availability
    print("\nLibrary Availability Check:")
    print(f"📊 Scikit-learn: ✅ Available")
    print(f"📈 Scipy/Numpy: ✅ Available")
    print(f"🧠 TensorFlow (Autoencoders): {'✅ Available' if TF_AVAILABLE else '❌ Not Available'}")
    print(f"📊 Statsmodels (Time Series): {'✅ Available' if STATSMODELS_AVAILABLE else '❌ Not Available'}")
    
    # Run comprehensive test
    try:
        success = comprehensive_anomaly_detection_test()
        
        if success:
            print("\n🎉 All tests completed successfully!")
            print("\nKey Takeaways:")
            print("• Statistical methods are fast and interpretable but assume data distributions")
            print("• ML methods handle complex patterns but require more computational resources")
            print("• Time series methods account for temporal dependencies and seasonality")
            print("• Ensemble methods combine strengths and provide robust performance")
            print("• Real-time pipelines enable streaming anomaly detection with adaptation")
            
            print("\nProduction Considerations:")
            print("• Choose methods based on data characteristics and business requirements")
            print("• Monitor performance and adapt to concept drift")
            print("• Balance false positives vs false negatives based on business impact")
            print("• Implement proper alerting and feedback mechanisms")
            print("• Consider computational constraints for real-time applications")
            
            print("\nNext Steps:")
            print("• Deploy selected methods in production environment")
            print("• Set up monitoring dashboards and alerting systems")
            print("• Implement feedback loops for continuous improvement")
            print("• Establish procedures for handling detected anomalies")
            
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)