"""
Day 28: Anomaly Detection - Statistical & ML-based Methods - Exercise

Business Scenario:
You're the Senior Data Scientist at SecureBank, a digital banking platform processing
millions of transactions daily. The fraud detection team has asked you to build a 
comprehensive anomaly detection system that can identify:

1. Fraudulent credit card transactions
2. Unusual account access patterns  
3. Suspicious money transfer behaviors
4. System performance anomalies

Your system needs to handle real-time processing, adapt to changing fraud patterns,
and minimize false positives to avoid blocking legitimate customers.

Requirements:
1. Implement multiple anomaly detection algorithms (statistical and ML-based)
2. Build an ensemble system that combines different approaches
3. Create real-time processing capabilities for streaming data
4. Develop evaluation metrics and validation framework
5. Handle concept drift and model adaptation

Success Criteria:
- Detect 95%+ of known fraud cases (high recall)
- Keep false positive rate below 2% (high precision)
- Process transactions in under 100ms (real-time requirement)
- Adapt to new fraud patterns within 24 hours
"""

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
import matplotlib.pyplot as plt
import seaborn as sns

# Statistical libraries
from scipy import stats
from scipy.stats import zscore

# Deep learning (optional - will handle gracefully if not available)
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("TensorFlow not available. Autoencoder methods will be skipped.")

# Time series libraries
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Statsmodels not available. Time series methods will be limited.")

# =============================================================================
# TASK 1: STATISTICAL ANOMALY DETECTION METHODS
# =============================================================================

class StatisticalAnomalyDetector:
    """
    Implement statistical methods for anomaly detection
    """
    
    def __init__(self):
        self.fitted_params = {}
        
    def zscore_detection(self, data: np.ndarray, threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement Z-score based anomaly detection
        """
        data = np.array(data)
        
        # Handle edge cases
        if len(data) == 0:
            return np.array([]), np.array([])
        
        # Calculate mean and std
        mean = np.mean(data)
        std = np.std(data)
        
        # Handle constant data
        if std == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        # Calculate Z-scores
        z_scores = np.abs((data - mean) / std)
        
        # Identify anomalies
        anomalies = z_scores > threshold
        
        return anomalies, z_scores
    
    def iqr_detection(self, data: np.ndarray, factor: float = 1.5) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Implement IQR (Interquartile Range) based anomaly detection
        """
        data = np.array(data)
        
        if len(data) == 0:
            return np.array([]), {}
        
        # Calculate quartiles
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        # Calculate bounds
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        
        # Identify anomalies
        anomalies = (data < lower_bound) | (data > upper_bound)
        
        bounds_info = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
        
        return anomalies, bounds_info
    
    def modified_zscore_detection(self, data: np.ndarray, threshold: float = 3.5) -> Tuple[np.ndarray, np.ndarray]:
        """
        Implement Modified Z-score using Median Absolute Deviation (MAD)
        """
        data = np.array(data)
        
        if len(data) == 0:
            return np.array([]), np.array([])
        
        # Calculate median
        median = np.median(data)
        
        # Calculate MAD
        mad = np.median(np.abs(data - median))
        
        # Handle case where MAD = 0
        if mad == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        # Calculate modified Z-scores
        modified_z_scores = 0.6745 * (data - median) / mad
        
        # Identify anomalies
        anomalies = np.abs(modified_z_scores) > threshold
        
        return anomalies, modified_z_scores
    
    def ensemble_statistical_detection(self, data: np.ndarray, 
                                     methods: List[str] = None,
                                     voting_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Combine multiple statistical methods using ensemble voting
        """
        if methods is None:
            methods = ['zscore', 'iqr', 'modified_zscore']
        
        results = {}
        votes = np.zeros(len(data))
        
        # Apply each method
        for method in methods:
            try:
                if method == 'zscore':
                    anomalies, scores = self.zscore_detection(data)
                    results[method] = {'anomalies': anomalies, 'scores': scores}
                elif method == 'iqr':
                    anomalies, bounds = self.iqr_detection(data)
                    results[method] = {'anomalies': anomalies, 'bounds': bounds}
                elif method == 'modified_zscore':
                    anomalies, scores = self.modified_zscore_detection(data)
                    results[method] = {'anomalies': anomalies, 'scores': scores}
                
                votes += anomalies.astype(int)
            except Exception as e:
                print(f"Method {method} failed: {e}")
                continue
        
        # Apply voting threshold
        ensemble_anomalies = votes >= (len(methods) * voting_threshold)
        
        return {
            'ensemble_anomalies': ensemble_anomalies,
            'vote_counts': votes,
            'individual_results': results,
            'methods_used': methods,
            'voting_threshold': voting_threshold
        }

# =============================================================================
# TASK 2: MACHINE LEARNING ANOMALY DETECTION
# =============================================================================

class MLAnomalyDetector:
    """
    Implement ML-based anomaly detection methods
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_fitted = {}
        
    def isolation_forest_detection(self, X_train: np.ndarray, X_test: np.ndarray = None,
                                 contamination: float = 0.1, 
                                 n_estimators: int = 100) -> Dict[str, Any]:
        """
        Implement Isolation Forest anomaly detection
        """
        if X_test is None:
            X_test = X_train
        
        # Initialize Isolation Forest
        model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42
        )
        
        # Fit on training data
        model.fit(X_train)
        
        # Generate predictions and scores
        predictions = model.predict(X_test)
        scores = model.decision_function(X_test)
        
        # Store model
        self.models['isolation_forest'] = model
        self.is_fitted['isolation_forest'] = True
        
        return {
            'predictions': predictions,
            'scores': scores,
            'anomalies': predictions == -1,
            'model_type': 'IsolationForest',
            'contamination': contamination,
            'n_estimators': n_estimators
        }
    
    def one_class_svm_detection(self, X_train: np.ndarray, X_test: np.ndarray = None,
                              kernel: str = 'rbf', nu: float = 0.1) -> Dict[str, Any]:
        """
        Implement One-Class SVM anomaly detection
        """
        if X_test is None:
            X_test = X_train
        
        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Initialize One-Class SVM
        model = OneClassSVM(kernel=kernel, nu=nu)
        
        # Fit on scaled training data
        model.fit(X_train_scaled)
        
        # Generate predictions and scores
        predictions = model.predict(X_test_scaled)
        scores = model.decision_function(X_test_scaled)
        
        # Store model and scaler
        self.models['one_class_svm'] = model
        self.scalers['one_class_svm'] = scaler
        self.is_fitted['one_class_svm'] = True
        
        return {
            'predictions': predictions,
            'scores': scores,
            'anomalies': predictions == -1,
            'model_type': 'OneClassSVM',
            'kernel': kernel,
            'nu': nu
        }
    
    def autoencoder_detection(self, X_train: np.ndarray, X_test: np.ndarray = None,
                            encoding_dim: int = 32, epochs: int = 100,
                            threshold_percentile: float = 95) -> Dict[str, Any]:
        """
        Implement Autoencoder-based anomaly detection
        """
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available for autoencoder'}
        
        if X_test is None:
            X_test = X_train
        
        input_dim = X_train.shape[1]
        
        # Build autoencoder
        input_layer = tf.keras.layers.Input(shape=(input_dim,))
        encoded = tf.keras.layers.Dense(64, activation='relu')(input_layer)
        encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(64, activation='relu')(encoded)
        decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(decoded)
        
        autoencoder = tf.keras.Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        # Train autoencoder
        autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=32, 
                       validation_split=0.2, verbose=0)
        
        # Calculate reconstruction errors
        reconstructed = autoencoder.predict(X_test, verbose=0)
        mse = np.mean(np.square(X_test - reconstructed), axis=1)
        
        # Set threshold based on percentile
        threshold = np.percentile(mse, threshold_percentile)
        anomalies = mse > threshold
        
        # Store model
        self.models['autoencoder'] = autoencoder
        self.is_fitted['autoencoder'] = True
        
        return {
            'predictions': anomalies.astype(int) * 2 - 1,  # Convert to -1/1 format
            'scores': mse,
            'anomalies': anomalies,
            'threshold': threshold,
            'model_type': 'Autoencoder',
            'encoding_dim': encoding_dim
        }

# =============================================================================
# TASK 3: TIME SERIES ANOMALY DETECTION
# =============================================================================

class TimeSeriesAnomalyDetector:
    """
    Implement time series specific anomaly detection methods
    """
    
    def __init__(self):
        self.models = {}
        
    def statistical_process_control(self, data: pd.Series, 
                                  window_size: int = 30,
                                  n_sigma: float = 3.0) -> Dict[str, Any]:
        """
        TODO: Implement Statistical Process Control (SPC) anomaly detection
        
        Args:
            data: Time series data with datetime index
            window_size: Rolling window size for control limits
            n_sigma: Number of standard deviations for control limits
            
        Returns:
            Dictionary with anomaly flags, control limits, and statistics
            
        Hints:
        - Calculate rolling mean and standard deviation
        - Compute upper and lower control limits
        - Flag points outside control limits
        - Handle initial window where statistics aren't available
        - Return control limits for visualization
        """
        # TODO: Calculate rolling statistics
        # TODO: Compute control limits
        # TODO: Identify anomalies outside control limits
        # TODO: Return comprehensive results
        pass
    
    def seasonal_decomposition_detection(self, data: pd.Series,
                                       period: int = 24,
                                       model: str = 'additive') -> Dict[str, Any]:
        """
        TODO: Implement seasonal decomposition-based anomaly detection
        
        Args:
            data: Time series data with datetime index
            period: Seasonal period (e.g., 24 for hourly data with daily seasonality)
            model: 'additive' or 'multiplicative' decomposition
            
        Returns:
            Dictionary with anomaly flags, decomposition components, and residual analysis
            
        Hints:
        - Use seasonal_decompose to separate trend, seasonal, and residual components
        - Apply anomaly detection to residual component
        - Use IQR or Z-score method on residuals
        - Handle missing values in decomposition
        - Return decomposition components for analysis
        """
        if not STATSMODELS_AVAILABLE:
            return {'error': 'Statsmodels not available for seasonal decomposition'}
            
        # TODO: Perform seasonal decomposition
        # TODO: Extract residual component
        # TODO: Apply anomaly detection to residuals
        # TODO: Map results back to original time series
        pass
    
    def sliding_window_detection(self, data: pd.Series,
                               window_size: int = 50,
                               step_size: int = 1,
                               method: str = 'isolation_forest') -> Dict[str, Any]:
        """
        TODO: Implement sliding window anomaly detection for streaming data
        
        Args:
            data: Time series data
            window_size: Size of sliding window
            step_size: Step size for window movement
            method: Anomaly detection method to apply in each window
            
        Returns:
            Dictionary with anomaly flags, window-wise results, and method info
            
        Hints:
        - Create sliding windows of specified size
        - Apply chosen anomaly detection method to each window
        - Aggregate results across windows
        - Handle overlapping windows appropriately
        - Consider computational efficiency for real-time processing
        """
        # TODO: Create sliding windows
        # TODO: Apply anomaly detection method to each window
        # TODO: Aggregate and combine results
        # TODO: Return time-aligned anomaly flags
        pass

# =============================================================================
# TASK 4: ENSEMBLE ANOMALY DETECTION SYSTEM
# =============================================================================

class EnsembleAnomalyDetector:
    """
    Combine multiple anomaly detection methods for robust detection
    """
    
    def __init__(self, combination_method: str = 'voting'):
        self.detectors = {}
        self.weights = {}
        self.combination_method = combination_method
        self.performance_history = {}
        
    def add_detector(self, name: str, detector: Any, weight: float = 1.0):
        """
        TODO: Add an anomaly detector to the ensemble
        
        Args:
            name: Unique name for the detector
            detector: Anomaly detector instance
            weight: Weight for ensemble combination
            
        Hints:
        - Store detector with its name and weight
        - Initialize performance tracking for this detector
        - Validate that detector has required methods
        """
        # TODO: Add detector to ensemble with weight
        # TODO: Initialize performance tracking
        pass
    
    def fit_ensemble(self, X_train: np.ndarray, y_train: np.ndarray = None) -> 'EnsembleAnomalyDetector':
        """
        TODO: Fit all detectors in the ensemble
        
        Args:
            X_train: Training data
            y_train: Training labels (optional, for supervised methods)
            
        Returns:
            Self for method chaining
            
        Hints:
        - Fit each detector in the ensemble
        - Handle different detector interfaces (some need labels, some don't)
        - Track fitting success/failure for each detector
        - Store fitted models for later use
        """
        # TODO: Fit each detector in the ensemble
        # TODO: Handle different detector requirements
        # TODO: Track fitting results
        pass
    
    def predict_ensemble(self, X_test: np.ndarray) -> Dict[str, Any]:
        """
        TODO: Generate ensemble predictions using all fitted detectors
        
        Args:
            X_test: Test data for anomaly detection
            
        Returns:
            Dictionary with ensemble predictions, individual results, and confidence scores
            
        Hints:
        - Get predictions from each fitted detector
        - Combine predictions using specified method (voting, weighted average, etc.)
        - Calculate confidence scores based on agreement between detectors
        - Handle cases where some detectors fail
        - Return both ensemble and individual results
        """
        # TODO: Get predictions from each detector
        # TODO: Combine predictions using ensemble method
        # TODO: Calculate confidence scores
        # TODO: Return comprehensive results
        pass
    
    def evaluate_ensemble(self, X_test: np.ndarray, y_true: np.ndarray) -> Dict[str, Any]:
        """
        TODO: Evaluate ensemble performance and individual detector performance
        
        Args:
            X_test: Test data
            y_true: True anomaly labels
            
        Returns:
            Dictionary with performance metrics for ensemble and individual detectors
            
        Hints:
        - Generate predictions from ensemble and individual detectors
        - Calculate standard metrics (precision, recall, F1, AUC)
        - Compare ensemble performance to individual detectors
        - Identify best and worst performing detectors
        - Update performance history for adaptive weighting
        """
        # TODO: Generate predictions from all detectors
        # TODO: Calculate performance metrics
        # TODO: Compare ensemble vs individual performance
        # TODO: Update performance history
        pass
    
    def update_weights(self, performance_metrics: Dict[str, float], 
                      adaptation_rate: float = 0.1):
        """
        TODO: Update detector weights based on recent performance
        
        Args:
            performance_metrics: Recent performance metrics for each detector
            adaptation_rate: Rate of weight adaptation
            
        Hints:
        - Adjust weights based on recent performance
        - Use exponential moving average for smooth adaptation
        - Ensure weights remain positive and sum to reasonable total
        - Consider different adaptation strategies (performance-based, error-based)
        """
        # TODO: Update weights based on performance
        # TODO: Apply adaptation rate for smooth changes
        # TODO: Normalize weights appropriately
        pass

# =============================================================================
# TASK 5: REAL-TIME ANOMALY DETECTION PIPELINE
# =============================================================================

class RealTimeAnomalyPipeline:
    """
    Production-ready real-time anomaly detection pipeline
    """
    
    def __init__(self, buffer_size: int = 1000, update_frequency: int = 100):
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        self.data_buffer = []
        self.anomaly_buffer = []
        self.models = {}
        self.update_counter = 0
        self.performance_metrics = {}
        
    def process_data_point(self, data_point: Dict[str, Any], 
                          timestamp: datetime = None) -> Dict[str, Any]:
        """
        TODO: Process a single data point in real-time
        
        Args:
            data_point: Dictionary containing feature values
            timestamp: Timestamp of the data point
            
        Returns:
            Dictionary with anomaly detection results and metadata
            
        Hints:
        - Add data point to buffer
        - Maintain sliding window of recent data
        - Apply anomaly detection to current point
        - Update models periodically
        - Return results with confidence scores and explanations
        """
        # TODO: Add data point to buffer
        # TODO: Maintain sliding window
        # TODO: Apply anomaly detection
        # TODO: Update models if needed
        # TODO: Return detection results
        pass
    
    def update_models(self, retrain: bool = False):
        """
        TODO: Update anomaly detection models with recent data
        
        Args:
            retrain: Whether to completely retrain models or just update parameters
            
        Hints:
        - Use recent data from buffer for model updates
        - Implement incremental learning where possible
        - Handle concept drift by adapting to new patterns
        - Validate model performance on recent data
        - Switch to backup models if performance degrades
        """
        # TODO: Extract recent normal data from buffer
        # TODO: Update or retrain models
        # TODO: Validate model performance
        # TODO: Handle concept drift
        pass
    
    def detect_concept_drift(self) -> Dict[str, Any]:
        """
        TODO: Detect if the data distribution has changed significantly
        
        Returns:
            Dictionary with drift detection results and recommended actions
            
        Hints:
        - Compare recent data distribution to historical baseline
        - Use statistical tests (KS test, chi-square test)
        - Monitor model performance over time
        - Detect sudden changes in anomaly rates
        - Recommend model retraining or parameter adjustment
        """
        # TODO: Compare recent vs historical data distributions
        # TODO: Apply statistical tests for drift detection
        # TODO: Monitor performance trends
        # TODO: Return drift analysis and recommendations
        pass
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        TODO: Get comprehensive status of the real-time pipeline
        
        Returns:
            Dictionary with pipeline health, performance, and operational metrics
            
        Hints:
        - Report buffer status and data flow rates
        - Include model performance metrics
        - Show recent anomaly detection rates
        - Report system health indicators
        - Include recommendations for optimization
        """
        # TODO: Collect buffer and data flow statistics
        # TODO: Aggregate model performance metrics
        # TODO: Calculate anomaly detection rates
        # TODO: Assess system health
        # TODO: Return comprehensive status report
        pass

# =============================================================================
# TASK 6: COMPREHENSIVE EVALUATION AND TESTING
# =============================================================================

def generate_synthetic_data(n_samples: int = 10000, 
                          anomaly_rate: float = 0.05,
                          data_type: str = 'multivariate') -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data with known anomalies for testing
    """
    np.random.seed(42)
    n_anomalies = int(n_samples * anomaly_rate)
    n_normal = n_samples - n_anomalies
    
    if data_type == 'univariate':
        # Generate 1D data
        normal_data = np.random.normal(0, 1, n_normal).reshape(-1, 1)
        anomaly_data = np.random.normal(4, 1, n_anomalies).reshape(-1, 1)  # Shifted mean
        
    elif data_type == 'multivariate':
        # Generate correlated multivariate data
        mean_normal = [0, 0, 0]
        cov_normal = [[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]]
        normal_data = np.random.multivariate_normal(mean_normal, cov_normal, n_normal)
        
        # Anomalies with different characteristics
        mean_anomaly = [3, -2, 4]
        cov_anomaly = [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
        anomaly_data = np.random.multivariate_normal(mean_anomaly, cov_anomaly, n_anomalies)
        
    elif data_type == 'timeseries':
        # Generate time series with trend and seasonality
        t = np.arange(n_samples)
        trend = 0.01 * t
        seasonal = 2 * np.sin(2 * np.pi * t / 50)
        noise = np.random.normal(0, 0.5, n_samples)
        
        normal_ts = trend + seasonal + noise
        
        # Inject anomalies at random positions
        anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
        normal_ts[anomaly_indices] += np.random.uniform(5, 10, n_anomalies)
        
        labels = np.zeros(n_samples, dtype=bool)
        labels[anomaly_indices] = True
        
        return normal_ts.reshape(-1, 1), labels
    
    else:
        raise ValueError(f"Unknown data_type: {data_type}")
    
    # Combine normal and anomaly data
    data = np.vstack([normal_data, anomaly_data])
    labels = np.hstack([np.zeros(n_normal, dtype=bool), np.ones(n_anomalies, dtype=bool)])
    
    # Shuffle the data
    indices = np.random.permutation(len(data))
    data = data[indices]
    labels = labels[indices]
    
    return data, labels

def comprehensive_evaluation(detectors: List[Any], 
                           X_test: np.ndarray, 
                           y_true: np.ndarray) -> pd.DataFrame:
    """
    TODO: Perform comprehensive evaluation of multiple anomaly detectors
    
    Args:
        detectors: List of fitted anomaly detectors
        X_test: Test data
        y_true: True anomaly labels
        
    Returns:
        DataFrame with performance metrics for each detector
        
    Hints:
    - Apply each detector to test data
    - Calculate multiple performance metrics
    - Include computational time measurements
    - Handle detectors that might fail
    - Create comparison table with rankings
    """
    # TODO: Apply each detector to test data
    # TODO: Calculate comprehensive metrics
    # TODO: Measure computational performance
    # TODO: Create comparison DataFrame
    pass

def plot_anomaly_results(data: np.ndarray, 
                        anomalies: np.ndarray, 
                        scores: np.ndarray = None,
                        title: str = "Anomaly Detection Results"):
    """
    Create comprehensive visualization of anomaly detection results
    """
    if data.shape[1] == 1:
        # 1D data visualization
        plt.figure(figsize=(12, 6))
        
        if scores is not None:
            # Plot with score-based coloring
            scatter = plt.scatter(range(len(data)), data.flatten(), 
                                c=scores, cmap='RdYlBu_r', alpha=0.7)
            plt.colorbar(scatter, label='Anomaly Score')
        else:
            # Plot with anomaly flags
            normal_mask = ~anomalies
            plt.scatter(np.where(normal_mask)[0], data[normal_mask].flatten(), 
                       c='blue', alpha=0.6, s=20, label='Normal')
            plt.scatter(np.where(anomalies)[0], data[anomalies].flatten(), 
                       c='red', alpha=0.8, s=30, label='Anomaly')
            plt.legend()
        
        plt.xlabel('Sample Index')
        plt.ylabel('Value')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        
    elif data.shape[1] >= 2:
        # 2D+ data visualization (use first two dimensions)
        plt.figure(figsize=(10, 8))
        
        if scores is not None:
            scatter = plt.scatter(data[:, 0], data[:, 1], c=scores, 
                                cmap='RdYlBu_r', alpha=0.7, s=30)
            plt.colorbar(scatter, label='Anomaly Score')
        else:
            normal_mask = ~anomalies
            plt.scatter(data[normal_mask, 0], data[normal_mask, 1], 
                       c='blue', alpha=0.6, s=20, label='Normal')
            plt.scatter(data[anomalies, 0], data[anomalies, 1], 
                       c='red', alpha=0.8, s=30, label='Anomaly')
            plt.legend()
        
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MAIN EXERCISE EXECUTION
# =============================================================================

def main():
    """
    Main function to execute the comprehensive anomaly detection exercise
    """
    print("=" * 80)
    print("DAY 28: ANOMALY DETECTION - COMPREHENSIVE EXERCISE")
    print("=" * 80)
    
    print("\n🏦 SecureBank Anomaly Detection System")
    print("Building comprehensive fraud detection and system monitoring...")
    
    # Task 1 - Test Statistical Methods
    print("\n" + "="*50)
    print("TASK 1: STATISTICAL ANOMALY DETECTION")
    print("="*50)
    
    # Generate test data
    print("Generating synthetic multivariate data...")
    X_multi, y_multi = generate_synthetic_data(1000, 0.1, 'multivariate')
    
    # Test statistical methods
    stat_detector = StatisticalAnomalyDetector()
    
    # Test on first feature for univariate methods
    feature_data = X_multi[:, 0]
    
    print("Testing Z-score method...")
    z_anomalies, z_scores = stat_detector.zscore_detection(feature_data)
    print(f"Z-score detected {np.sum(z_anomalies)} anomalies")
    
    print("Testing IQR method...")
    iqr_anomalies, iqr_bounds = stat_detector.iqr_detection(feature_data)
    print(f"IQR detected {np.sum(iqr_anomalies)} anomalies")
    
    print("Testing Modified Z-score method...")
    mad_anomalies, mad_scores = stat_detector.modified_zscore_detection(feature_data)
    print(f"Modified Z-score detected {np.sum(mad_anomalies)} anomalies")
    
    print("Testing ensemble statistical detection...")
    ensemble_result = stat_detector.ensemble_statistical_detection(feature_data)
    print(f"Ensemble detected {np.sum(ensemble_result['ensemble_anomalies'])} anomalies")
    
    # Visualize results
    plot_anomaly_results(feature_data.reshape(-1, 1), z_anomalies, z_scores, 
                        "Z-score Anomaly Detection")
    
    # Task 2 - Test ML Methods
    print("\n" + "="*50)
    print("TASK 2: MACHINE LEARNING ANOMALY DETECTION")
    print("="*50)
    
    # Prepare training data (normal samples only)
    normal_mask = ~y_multi
    X_train = X_multi[normal_mask]
    X_test = X_multi
    
    ml_detector = MLAnomalyDetector()
    
    print("Testing Isolation Forest...")
    iso_result = ml_detector.isolation_forest_detection(X_train, X_test)
    print(f"Isolation Forest detected {np.sum(iso_result['anomalies'])} anomalies")
    
    print("Testing One-Class SVM...")
    svm_result = ml_detector.one_class_svm_detection(X_train, X_test)
    print(f"One-Class SVM detected {np.sum(svm_result['anomalies'])} anomalies")
    
    if TF_AVAILABLE:
        print("Testing Autoencoder...")
        ae_result = ml_detector.autoencoder_detection(X_train, X_test, epochs=20)
        print(f"Autoencoder detected {np.sum(ae_result['anomalies'])} anomalies")
    
    # Visualize ML results
    plot_anomaly_results(X_test, iso_result['anomalies'], iso_result['scores'], 
                        "Isolation Forest Results")
    
    # Performance evaluation
    if len(np.unique(y_multi)) > 1:
        print("\n📊 Performance Evaluation:")
        
        methods_results = {
            'Isolation Forest': iso_result['anomalies'],
            'One-Class SVM': svm_result['anomalies']
        }
        
        for method_name, predictions in methods_results.items():
            try:
                precision = precision_score(y_multi, predictions)
                recall = recall_score(y_multi, predictions)
                f1 = f1_score(y_multi, predictions)
                print(f"{method_name:15} - Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
            except Exception as e:
                print(f"{method_name:15} - Evaluation failed: {e}")
    
    print("\n" + "="*80)
    print("EXERCISE COMPLETED!")
    print("="*80)
    
    print("\n📊 Key Insights:")
    print("• Statistical methods work well for simple, well-behaved data")
    print("• ML methods handle complex, high-dimensional patterns better")
    print("• Ensemble methods provide robust performance across scenarios")
    print("• Real-time processing requires careful optimization")
    print("• Domain expertise is crucial for threshold setting")
    
    print("\n🎯 Next Steps:")
    print("• Implement the solution with production-grade code")
    print("• Test on real financial transaction data")
    print("• Deploy real-time monitoring system")
    print("• Set up alerting and feedback mechanisms")

if __name__ == "__main__":
    main()