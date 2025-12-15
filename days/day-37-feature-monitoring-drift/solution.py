"""
Day 37: Feature Monitoring & Drift - Solution

Complete implementation of feature monitoring and drift detection system with
statistical methods, ML-based detection, real-time monitoring, and automated retraining.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import json
import logging
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PopulationStabilityIndex:
    """
    Population Stability Index (PSI) for detecting feature drift
    """
    
    def __init__(self, n_bins: int = 10, min_bin_size: float = 0.05):
        """Initialize PSI calculator with configuration parameters"""
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.reference_bins = {}
        self.reference_stats = {}
        self.feature_columns = []
    
    def fit(self, reference_data: pd.DataFrame, feature_columns: List[str]):
        """
        Fit PSI calculator on reference data
        """
        logger.info(f"Fitting PSI detector on {len(feature_columns)} features")
        self.feature_columns = feature_columns
        
        for column in feature_columns:
            if column not in reference_data.columns:
                logger.warning(f"Column {column} not found in reference data")
                continue
                
            data = reference_data[column].dropna()
            
            if pd.api.types.is_numeric_dtype(data):
                # Create quantile-based bins for numeric features
                bins = np.quantile(data, np.linspace(0, 1, self.n_bins + 1))
                bins = np.unique(bins)  # Remove duplicate bin edges
                
                if len(bins) < 2:
                    # Handle case where all values are the same
                    bins = [data.min() - 1, data.max() + 1]
                
                self.reference_bins[column] = bins
                
                # Calculate reference distribution
                bin_counts, _ = np.histogram(data, bins=bins)
                bin_proportions = bin_counts / len(data)
                
                # Ensure minimum bin size to avoid log(0)
                bin_proportions = np.maximum(bin_proportions, self.min_bin_size / self.n_bins)
                bin_proportions = bin_proportions / bin_proportions.sum()  # Renormalize
                
                self.reference_stats[column] = {
                    'type': 'numeric',
                    'bins': bins,
                    'proportions': bin_proportions,
                    'mean': data.mean(),
                    'std': data.std()
                }
                
            else:
                # Handle categorical features
                value_counts = data.value_counts(normalize=True)
                
                # Keep only top categories, group others as 'OTHER'
                if len(value_counts) > self.n_bins:
                    top_categories = value_counts.head(self.n_bins - 1)
                    other_proportion = value_counts.tail(len(value_counts) - self.n_bins + 1).sum()
                    
                    proportions = top_categories.to_dict()
                    proportions['OTHER'] = other_proportion
                else:
                    proportions = value_counts.to_dict()
                
                # Ensure minimum proportion
                for key in proportions:
                    proportions[key] = max(proportions[key], self.min_bin_size / len(proportions))
                
                # Renormalize
                total = sum(proportions.values())
                proportions = {k: v/total for k, v in proportions.items()}
                
                self.reference_stats[column] = {
                    'type': 'categorical',
                    'proportions': proportions,
                    'categories': list(proportions.keys())
                }
        
        logger.info(f"PSI detector fitted on {len(self.reference_stats)} features")
    
    def calculate_psi(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate PSI for current data against reference
        """
        psi_scores = {}
        
        for column in self.feature_columns:
            if column not in current_data.columns or column not in self.reference_stats:
                logger.warning(f"Skipping column {column} - not available")
                continue
                
            try:
                current_series = current_data[column].dropna()
                
                if len(current_series) == 0:
                    psi_scores[column] = float('inf')
                    continue
                
                ref_stats = self.reference_stats[column]
                
                if ref_stats['type'] == 'numeric':
                    # Calculate current distribution using reference bins
                    bins = ref_stats['bins']
                    current_counts, _ = np.histogram(current_series, bins=bins)
                    current_proportions = current_counts / len(current_series)
                    
                    # Ensure minimum proportion
                    current_proportions = np.maximum(current_proportions, self.min_bin_size / len(current_proportions))
                    current_proportions = current_proportions / current_proportions.sum()
                    
                    reference_proportions = ref_stats['proportions']
                    
                else:  # categorical
                    # Calculate current categorical distribution
                    current_counts = current_series.value_counts(normalize=True)
                    reference_categories = ref_stats['categories']
                    
                    current_proportions = {}
                    for category in reference_categories:
                        if category == 'OTHER':
                            # Sum all categories not in reference top categories
                            other_cats = set(current_counts.index) - set(reference_categories)
                            current_proportions[category] = sum(current_counts.get(cat, 0) for cat in other_cats)
                        else:
                            current_proportions[category] = current_counts.get(category, 0)
                    
                    # Ensure minimum proportion
                    for key in current_proportions:
                        current_proportions[key] = max(current_proportions[key], self.min_bin_size / len(current_proportions))
                    
                    # Renormalize
                    total = sum(current_proportions.values())
                    current_proportions = {k: v/total for k, v in current_proportions.items()}
                    
                    reference_proportions = ref_stats['proportions']
                    
                    # Convert to arrays for PSI calculation
                    current_proportions = np.array([current_proportions[cat] for cat in reference_categories])
                    reference_proportions = np.array([reference_proportions[cat] for cat in reference_categories])
                
                # Calculate PSI: sum((current - reference) * log(current/reference))
                psi = np.sum((current_proportions - reference_proportions) * 
                           np.log(current_proportions / reference_proportions))
                
                psi_scores[column] = psi
                
            except Exception as e:
                logger.error(f"Error calculating PSI for column {column}: {str(e)}")
                psi_scores[column] = float('inf')
        
        return psi_scores
    
    def interpret_psi(self, psi_score: float) -> str:
        """Interpret PSI score and return human-readable assessment"""
        if psi_score < 0.1:
            return "No significant change"
        elif psi_score < 0.2:
            return "Minor change"
        elif psi_score < 0.5:
            return "Major change"
        else:
            return "Severe change - investigate immediately"

class KolmogorovSmirnovDriftDetector:
    """
    Kolmogorov-Smirnov test for detecting distribution drift
    """
    
    def __init__(self, significance_level: float = 0.05):
        """Initialize KS drift detector"""
        self.significance_level = significance_level
        self.reference_distributions = {}
        self.feature_columns = []
    
    def fit(self, reference_data: pd.DataFrame, feature_columns: List[str]):
        """Fit detector on reference data"""
        logger.info(f"Fitting KS detector on {len(feature_columns)} features")
        self.feature_columns = feature_columns
        
        for column in feature_columns:
            if column in reference_data.columns:
                data = reference_data[column].dropna()
                
                if pd.api.types.is_numeric_dtype(data):
                    self.reference_distributions[column] = {
                        'type': 'numeric',
                        'data': data.values,
                        'mean': data.mean(),
                        'std': data.std()
                    }
                else:
                    # For categorical data, store value counts
                    value_counts = data.value_counts()
                    self.reference_distributions[column] = {
                        'type': 'categorical',
                        'value_counts': value_counts,
                        'categories': set(data.unique())
                    }
        
        logger.info(f"KS detector fitted on {len(self.reference_distributions)} features")
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Dict]:
        """Detect drift in current data using KS test"""
        drift_results = {}
        
        for column in self.feature_columns:
            if column not in current_data.columns or column not in self.reference_distributions:
                continue
                
            current_series = current_data[column].dropna()
            reference_info = self.reference_distributions[column]
            
            if reference_info['type'] == 'numeric':
                # KS test for numeric features
                statistic, p_value = ks_2samp(
                    reference_info['data'], 
                    current_series.values
                )
                
                drift_detected = p_value < self.significance_level
                
                # Calculate effect size (difference in means normalized by pooled std)
                ref_mean = reference_info['mean']
                ref_std = reference_info['std']
                curr_mean = current_series.mean()
                curr_std = current_series.std()
                
                pooled_std = np.sqrt((ref_std**2 + curr_std**2) / 2)
                effect_size = abs(ref_mean - curr_mean) / pooled_std if pooled_std > 0 else 0
                
                drift_results[column] = {
                    'test': 'kolmogorov_smirnov',
                    'statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': drift_detected,
                    'effect_size': effect_size,
                    'reference_mean': ref_mean,
                    'current_mean': curr_mean,
                    'mean_shift': curr_mean - ref_mean
                }
                
            else:
                # Chi-square test for categorical features
                reference_counts = reference_info['value_counts']
                current_counts = current_series.value_counts()
                
                # Align categories
                all_categories = set(reference_counts.index) | set(current_counts.index)
                
                ref_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [current_counts.get(cat, 0) for cat in all_categories]
                
                # Chi-square test
                try:
                    contingency_table = np.array([ref_aligned, curr_aligned])
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                    
                    drift_detected = p_value < self.significance_level
                    
                    # Calculate Cramér's V as effect size
                    n = contingency_table.sum()
                    cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))
                    
                    drift_results[column] = {
                        'test': 'chi_square',
                        'statistic': chi2_stat,
                        'p_value': p_value,
                        'drift_detected': drift_detected,
                        'effect_size': cramers_v,
                        'new_categories': set(current_counts.index) - reference_info['categories'],
                        'missing_categories': reference_info['categories'] - set(current_counts.index)
                    }
                    
                except ValueError as e:
                    # Handle case where chi-square test fails
                    drift_results[column] = {
                        'test': 'chi_square',
                        'error': str(e),
                        'drift_detected': True,  # Conservative approach
                        'effect_size': 1.0
                    }
        
        return drift_results

class MLBasedDriftDetector:
    """
    Machine Learning based drift detection using domain classifier approach
    """
    
    def __init__(self, model_type: str = 'isolation_forest', contamination: float = 0.1):
        """Initialize ML-based drift detector"""
        self.model_type = model_type
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.drift_model = None
        self.domain_classifier = None
        self.feature_columns = None
        self.reference_stats = None
        
    def fit(self, reference_data: pd.DataFrame, feature_columns: List[str]):
        """Fit drift detector on reference data"""
        logger.info(f"Fitting ML drift detector on {len(feature_columns)} features")
        self.feature_columns = feature_columns
        
        # Prepare reference data (numeric features only)
        X_ref = reference_data[feature_columns].select_dtypes(include=[np.number])
        X_ref = X_ref.fillna(X_ref.mean())  # Simple imputation
        
        # Fit scaler
        X_ref_scaled = self.scaler.fit_transform(X_ref)
        
        # Fit anomaly detection model
        if self.model_type == 'isolation_forest':
            self.drift_model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.model_type == 'one_class_svm':
            from sklearn.svm import OneClassSVM
            self.drift_model = OneClassSVM(nu=self.contamination)
        
        self.drift_model.fit(X_ref_scaled)
        
        # Store reference statistics
        self.reference_stats = {
            'mean': X_ref.mean(),
            'std': X_ref.std(),
            'min': X_ref.min(),
            'max': X_ref.max(),
            'feature_names': list(X_ref.columns)
        }
        
        logger.info("ML drift detector fitted successfully")
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Detect drift using trained model"""
        if self.drift_model is None:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Prepare current data
        X_curr = current_data[self.feature_columns].select_dtypes(include=[np.number])
        X_curr = X_curr.fillna(self.reference_stats['mean'])  # Use reference means for imputation
        
        # Scale current data
        X_curr_scaled = self.scaler.transform(X_curr)
        
        # Get anomaly scores
        anomaly_scores = self.drift_model.decision_function(X_curr_scaled)
        outlier_predictions = self.drift_model.predict(X_curr_scaled)
        
        # Calculate drift metrics
        outlier_ratio = (outlier_predictions == -1).mean()
        mean_anomaly_score = anomaly_scores.mean()
        
        # Statistical comparison with reference
        current_stats = {
            'mean': X_curr.mean(),
            'std': X_curr.std(),
            'min': X_curr.min(),
            'max': X_curr.max()
        }
        
        # Calculate feature-wise drift scores
        feature_drift_scores = {}
        for feature in X_curr.columns:
            ref_mean = self.reference_stats['mean'][feature]
            ref_std = self.reference_stats['std'][feature]
            curr_mean = current_stats['mean'][feature]
            curr_std = current_stats['std'][feature]
            
            # Normalized difference in means
            mean_drift = abs(curr_mean - ref_mean) / (ref_std + 1e-8)
            
            # Ratio of standard deviations
            std_ratio = curr_std / (ref_std + 1e-8)
            
            # Combined drift score
            feature_drift_scores[feature] = mean_drift + abs(np.log(std_ratio))
        
        return {
            'overall_drift_score': outlier_ratio,
            'mean_anomaly_score': mean_anomaly_score,
            'feature_drift_scores': feature_drift_scores,
            'outlier_ratio': outlier_ratio,
            'drift_detected': outlier_ratio > self.contamination * 2,  # Conservative threshold
            'current_stats': current_stats
        }
    
    def train_domain_classifier(self, reference_data: pd.DataFrame, 
                              current_data: pd.DataFrame, feature_columns: List[str]) -> float:
        """Train domain classifier to detect drift"""
        logger.info("Training domain classifier for drift detection")
        
        # Prepare data
        X_ref = reference_data[feature_columns].select_dtypes(include=[np.number])
        X_curr = current_data[feature_columns].select_dtypes(include=[np.number])
        
        # Fill missing values
        X_ref = X_ref.fillna(X_ref.mean())
        X_curr = X_curr.fillna(X_ref.mean())  # Use reference means
        
        # Create labels (0 = reference, 1 = current)
        y_ref = np.zeros(len(X_ref))
        y_curr = np.ones(len(X_curr))
        
        # Combine data
        X_combined = pd.concat([X_ref, X_curr], ignore_index=True)
        y_combined = np.concatenate([y_ref, y_curr])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_combined)
        
        # Train domain classifier
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_combined, test_size=0.3, random_state=42, stratify=y_combined
        )
        
        self.domain_classifier = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.domain_classifier.fit(X_train, y_train)
        
        # Calculate AUC
        y_pred_proba = self.domain_classifier.predict_proba(X_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"Domain classifier AUC: {auc_score:.3f}")
        return auc_score
class RealTimeDriftMonitor:
    """
    Real-time drift monitoring system using streaming data
    """
    
    def __init__(self, window_size: int = 1000):
        """Initialize real-time drift monitor"""
        self.window_size = window_size
        self.feature_windows = {}
        self.drift_detectors = {}
        self.alert_thresholds = {}
        self.alert_history = []
        
    def setup_feature_monitoring(self, feature_config: Dict[str, Dict]):
        """Setup monitoring for specific features"""
        logger.info(f"Setting up monitoring for {len(feature_config)} features")
        
        for feature_name, config in feature_config.items():
            # Initialize sliding window
            self.feature_windows[feature_name] = deque(maxlen=self.window_size)
            
            # Initialize appropriate drift detector
            if config['type'] == 'psi':
                detector = PopulationStabilityIndex(
                    n_bins=config.get('n_bins', 10)
                )
            elif config['type'] == 'ks':
                detector = KolmogorovSmirnovDriftDetector(
                    significance_level=config.get('significance_level', 0.05)
                )
            elif config['type'] == 'ml':
                detector = MLBasedDriftDetector(
                    contamination=config.get('contamination', 0.1)
                )
            
            # Fit detector on reference data if provided
            if 'reference_data' in config:
                detector.fit(config['reference_data'], [feature_name])
            
            self.drift_detectors[feature_name] = detector
            self.alert_thresholds[feature_name] = config.get('alert_threshold', 0.2)
        
        logger.info("Feature monitoring setup complete")
    
    def process_new_data(self, feature_data: Dict[str, float]) -> Dict[str, float]:
        """Process new incoming data point"""
        drift_scores = {}
        timestamp = datetime.now()
        
        # Add to sliding windows
        for feature_name in self.feature_windows:
            if feature_name in feature_data:
                self.feature_windows[feature_name].append({
                    'value': feature_data[feature_name],
                    'timestamp': timestamp
                })
                
                # Check for drift if window is full
                if len(self.feature_windows[feature_name]) >= self.window_size:
                    drift_score = self.detect_drift_in_window(feature_name)
                    
                    if drift_score is not None:
                        drift_scores[feature_name] = drift_score
                        
                        # Check alert threshold
                        threshold = self.alert_thresholds[feature_name]
                        if drift_score > threshold:
                            self._generate_alert(feature_name, drift_score, threshold, timestamp)
        
        return drift_scores
    
    def detect_drift_in_window(self, feature_name: str) -> Optional[float]:
        """Detect drift in current sliding window"""
        if feature_name not in self.feature_windows or feature_name not in self.drift_detectors:
            return None
        
        window = self.feature_windows[feature_name]
        if len(window) < self.window_size:
            return None
        
        # Extract values from window
        values = [item['value'] for item in window]
        current_data = pd.DataFrame({feature_name: values})
        
        # Detect drift using appropriate detector
        detector = self.drift_detectors[feature_name]
        
        try:
            if isinstance(detector, PopulationStabilityIndex):
                drift_scores = detector.calculate_psi(current_data)
                return drift_scores.get(feature_name, 0)
            elif isinstance(detector, KolmogorovSmirnovDriftDetector):
                drift_results = detector.detect_drift(current_data)
                return drift_results.get(feature_name, {}).get('statistic', 0)
            elif isinstance(detector, MLBasedDriftDetector):
                drift_results = detector.detect_drift(current_data)
                return drift_results.get('overall_drift_score', 0)
        except Exception as e:
            logger.error(f"Error detecting drift for {feature_name}: {str(e)}")
            return None
    
    def _generate_alert(self, feature_name: str, drift_score: float, 
                       threshold: float, timestamp: datetime):
        """Generate drift alert"""
        severity = self._calculate_severity(drift_score, threshold)
        
        alert = {
            'timestamp': timestamp.isoformat(),
            'feature_name': feature_name,
            'drift_score': drift_score,
            'threshold': threshold,
            'severity': severity,
            'message': f'Drift detected in {feature_name}: score {drift_score:.3f} > threshold {threshold:.3f}'
        }
        
        self.alert_history.append(alert)
        logger.warning(f"DRIFT ALERT: {alert['message']} (Severity: {severity})")
        
        return alert
    
    def _calculate_severity(self, drift_score: float, threshold: float) -> str:
        """Calculate alert severity based on drift score"""
        ratio = drift_score / threshold
        
        if ratio < 1.5:
            return 'low'
        elif ratio < 3.0:
            return 'medium'
        elif ratio < 5.0:
            return 'high'
        else:
            return 'critical'

class AutomatedRetrainingPipeline:
    """
    Automated model retraining pipeline triggered by drift detection
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize retraining pipeline"""
        self.config = config
        self.retraining_history = []
        
    def evaluate_retraining_triggers(self, drift_results: Dict[str, float], 
                                   model_performance: Dict[str, float]) -> Dict[str, bool]:
        """Evaluate whether retraining should be triggered"""
        triggers = {
            'data_drift': False,
            'performance_degradation': False,
            'time_based': False,
            'manual_trigger': False
        }
        
        # Check data drift triggers
        drift_config = self.config['triggers']['data_drift']
        drift_threshold = drift_config['threshold']
        critical_features = drift_config['critical_features']
        
        for feature, drift_score in drift_results.items():
            if feature in critical_features and drift_score > drift_threshold:
                triggers['data_drift'] = True
                logger.info(f"Data drift trigger activated for feature {feature}: {drift_score:.3f}")
                break
        
        # Check performance degradation
        perf_thresholds = self.config['triggers']['performance_degradation']
        
        for metric, threshold in perf_thresholds.items():
            if metric in model_performance:
                if model_performance[metric] < threshold:
                    triggers['performance_degradation'] = True
                    logger.info(f"Performance degradation trigger activated for {metric}: {model_performance[metric]:.3f}")
                    break
        
        # Check time-based trigger
        last_training_time = self._get_last_training_time()
        max_age_days = self.config['triggers']['time_based']['max_age_days']
        
        if last_training_time:
            age_days = (datetime.now() - last_training_time).days
            if age_days > max_age_days:
                triggers['time_based'] = True
                logger.info(f"Time-based trigger activated: {age_days} days since last training")
        
        return triggers
    
    def determine_retraining_strategy(self, triggers: Dict[str, bool], 
                                    drift_results: Dict[str, float]) -> Dict[str, Any]:
        """Determine appropriate retraining strategy"""
        max_drift = max(drift_results.values()) if drift_results else 0
        
        # Severe drift or performance issues -> full retrain
        if (triggers['performance_degradation'] or 
            max_drift > self.config['strategies']['full_retrain']['drift_threshold']):
            return {
                'type': 'full_retrain',
                'reason': 'severe_drift_or_performance_issues',
                'config': self.config['strategies']['full_retrain'],
                'max_drift': max_drift
            }
        
        # Moderate drift -> incremental update
        elif max_drift > self.config['strategies']['incremental_update']['drift_threshold']:
            return {
                'type': 'incremental_update',
                'reason': 'moderate_drift',
                'config': self.config['strategies']['incremental_update'],
                'max_drift': max_drift
            }
        
        # Time-based or minor drift -> feature selection retrain
        else:
            return {
                'type': 'feature_selection',
                'reason': 'time_based_or_minor_drift',
                'config': self.config['strategies']['feature_selection'],
                'max_drift': max_drift
            }
    
    def trigger_retraining(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retraining based on strategy"""
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Triggering retraining with strategy: {strategy['type']}")
        
        # Simulate retraining execution
        retraining_job = {
            'job_id': job_id,
            'strategy': strategy['type'],
            'reason': strategy['reason'],
            'timestamp': datetime.now().isoformat(),
            'status': 'in_progress'
        }
        
        try:
            # Simulate different retraining strategies
            if strategy['type'] == 'full_retrain':
                result = self._execute_full_retraining(retraining_job)
            elif strategy['type'] == 'incremental_update':
                result = self._execute_incremental_update(retraining_job)
            elif strategy['type'] == 'feature_selection':
                result = self._execute_feature_selection_retrain(retraining_job)
            
            # Record retraining history
            self.retraining_history.append(result)
            
            return result
            
        except Exception as e:
            error_result = {
                **retraining_job,
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            }
            
            self.retraining_history.append(error_result)
            return error_result
    
    def _execute_full_retraining(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute full model retraining"""
        logger.info(f"Executing full retraining for job {job_config['job_id']}")
        
        # Simulate full retraining process
        import time
        time.sleep(2)  # Simulate training time
        
        # Simulate success with high probability
        success = np.random.random() > 0.1
        
        if success:
            return {
                **job_config,
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'new_model_version': f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'performance_improvement': np.random.uniform(0.02, 0.08)
            }
        else:
            return {
                **job_config,
                'status': 'failed',
                'completion_time': datetime.now().isoformat(),
                'error': 'Training convergence failed'
            }
    
    def _execute_incremental_update(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute incremental model update"""
        logger.info(f"Executing incremental update for job {job_config['job_id']}")
        
        # Simulate incremental update
        import time
        time.sleep(1)  # Faster than full retraining
        
        success = np.random.random() > 0.05
        
        if success:
            return {
                **job_config,
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'update_type': 'incremental',
                'performance_improvement': np.random.uniform(0.01, 0.04)
            }
        else:
            return {
                **job_config,
                'status': 'failed',
                'completion_time': datetime.now().isoformat(),
                'error': 'Incremental update failed'
            }
    
    def _execute_feature_selection_retrain(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retraining with feature selection"""
        logger.info(f"Executing feature selection retraining for job {job_config['job_id']}")
        
        # Simulate feature selection retraining
        import time
        time.sleep(1.5)
        
        success = np.random.random() > 0.08
        
        if success:
            return {
                **job_config,
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'features_selected': np.random.randint(5, 15),
                'performance_improvement': np.random.uniform(0.005, 0.025)
            }
        else:
            return {
                **job_config,
                'status': 'failed',
                'completion_time': datetime.now().isoformat(),
                'error': 'Feature selection optimization failed'
            }
    
    def _get_last_training_time(self) -> Optional[datetime]:
        """Get timestamp of last model training"""
        if self.retraining_history:
            # Get last successful retraining
            successful_jobs = [job for job in self.retraining_history if job['status'] == 'completed']
            if successful_jobs:
                last_job = successful_jobs[-1]
                return datetime.fromisoformat(last_job['timestamp'])
        
        # Default to 30 days ago if no history
        return datetime.now() - timedelta(days=30)

class FeatureMonitoringDashboard:
    """
    Interactive dashboard for feature monitoring and drift detection
    """
    
    def __init__(self, drift_detectors: Dict[str, Any]):
        """Initialize monitoring dashboard"""
        self.drift_detectors = drift_detectors
        self.dashboard_data = {}
        
    def create_drift_summary(self, feature: str, drift_score: float) -> Dict[str, Any]:
        """Create drift summary for dashboard"""
        if drift_score < 0.1:
            status = "Normal"
            color = "green"
            recommendation = "No action required"
        elif drift_score < 0.2:
            status = "Minor Drift"
            color = "orange"
            recommendation = "Monitor closely"
        elif drift_score < 0.5:
            status = "Major Drift"
            color = "red"
            recommendation = "Consider retraining"
        else:
            status = "Severe Drift"
            color = "darkred"
            recommendation = "Immediate retraining required"
        
        return {
            'feature': feature,
            'drift_score': drift_score,
            'status': status,
            'color': color,
            'recommendation': recommendation,
            'timestamp': datetime.now().isoformat()
        }
    
    def create_distribution_chart(self, reference_data: pd.DataFrame, 
                                current_data: pd.DataFrame, feature: str) -> Dict[str, Any]:
        """Create distribution comparison chart data"""
        chart_data = {
            'feature': feature,
            'reference_stats': {
                'mean': reference_data[feature].mean(),
                'std': reference_data[feature].std(),
                'min': reference_data[feature].min(),
                'max': reference_data[feature].max(),
                'count': len(reference_data)
            },
            'current_stats': {
                'mean': current_data[feature].mean(),
                'std': current_data[feature].std(),
                'min': current_data[feature].min(),
                'max': current_data[feature].max(),
                'count': len(current_data)
            }
        }
        
        # Calculate distribution comparison metrics
        chart_data['mean_shift'] = chart_data['current_stats']['mean'] - chart_data['reference_stats']['mean']
        chart_data['std_ratio'] = chart_data['current_stats']['std'] / chart_data['reference_stats']['std']
        
        return chart_data
    
    def create_drift_timeline(self, drift_history: pd.DataFrame, feature: str) -> Dict[str, Any]:
        """Create drift score timeline chart data"""
        timeline_data = {
            'feature': feature,
            'timestamps': drift_history['timestamp'].tolist(),
            'drift_scores': drift_history['drift_score'].tolist(),
            'thresholds': {
                'minor': 0.1,
                'major': 0.2,
                'severe': 0.5
            }
        }
        
        # Add trend analysis
        if len(drift_history) > 1:
            recent_scores = drift_history['drift_score'].tail(10)
            timeline_data['trend'] = 'increasing' if recent_scores.iloc[-1] > recent_scores.iloc[0] else 'decreasing'
            timeline_data['trend_slope'] = (recent_scores.iloc[-1] - recent_scores.iloc[0]) / len(recent_scores)
        
        return timeline_data

def create_sample_ecommerce_data():
    """Create sample e-commerce recommendation data"""
    np.random.seed(42)
    
    n_samples = 10000
    
    # User behavior features
    data = {
        'user_age': np.random.randint(18, 70, n_samples),
        'session_duration': np.random.exponential(300, n_samples),  # seconds
        'pages_viewed': np.random.poisson(5, n_samples),
        'cart_value': np.random.lognormal(4, 1, n_samples),
        'days_since_last_purchase': np.random.exponential(30, n_samples),
        'user_category': np.random.choice(['new', 'returning', 'vip'], n_samples, p=[0.3, 0.6, 0.1])
    }
    
    # Product features
    data.update({
        'product_price': np.random.lognormal(3, 1, n_samples),
        'product_rating': np.random.beta(8, 2, n_samples) * 5,  # 0-5 scale
        'product_category': np.random.choice(['electronics', 'clothing', 'books', 'home', 'sports'], 
                                           n_samples, p=[0.25, 0.3, 0.15, 0.2, 0.1]),
        'inventory_level': np.random.exponential(100, n_samples)
    })
    
    # Seasonal/temporal features
    timestamps = pd.date_range('2024-01-01', periods=n_samples, freq='1H')
    data.update({
        'hour_of_day': timestamps.hour,
        'day_of_week': timestamps.dayofweek,
        'is_weekend': (timestamps.dayofweek >= 5).astype(int),
        'month': timestamps.month
    })
    
    # Target variable (click-through rate)
    ctr_base = 0.05
    ctr_noise = np.random.normal(0, 0.01, n_samples)
    data['ctr'] = np.clip(ctr_base + ctr_noise, 0, 1)
    
    logger.info(f"Created sample e-commerce dataset with {n_samples} samples and {len(data)} features")
    return pd.DataFrame(data)

def create_drifted_data(original_data: pd.DataFrame, drift_type: str = 'gradual') -> pd.DataFrame:
    """Create drifted version of original data"""
    drifted_data = original_data.copy()
    
    if drift_type == 'gradual':
        # Gradual shift in user behavior
        drifted_data['user_age'] = drifted_data['user_age'] + np.random.normal(5, 2, len(drifted_data))
        drifted_data['session_duration'] = drifted_data['session_duration'] * np.random.uniform(1.2, 1.5, len(drifted_data))
        drifted_data['cart_value'] = drifted_data['cart_value'] * np.random.uniform(0.8, 1.3, len(drifted_data))
        
    elif drift_type == 'sudden':
        # Sudden shift (e.g., new product launch, marketing campaign)
        mask = np.random.random(len(drifted_data)) < 0.3
        drifted_data.loc[mask, 'product_price'] *= 1.5
        drifted_data.loc[mask, 'product_rating'] = np.random.beta(6, 3, mask.sum()) * 5
        
    elif drift_type == 'seasonal':
        # Seasonal pattern (e.g., holiday shopping)
        seasonal_multiplier = 1 + 0.5 * np.sin(drifted_data['month'] * np.pi / 6)
        drifted_data['cart_value'] *= seasonal_multiplier
        drifted_data['pages_viewed'] = (drifted_data['pages_viewed'] * seasonal_multiplier).astype(int)
    
    # Ensure data constraints
    drifted_data['user_age'] = np.clip(drifted_data['user_age'], 18, 80)
    drifted_data['product_rating'] = np.clip(drifted_data['product_rating'], 0, 5)
    drifted_data['cart_value'] = np.maximum(drifted_data['cart_value'], 0)
    
    logger.info(f"Created drifted dataset with {drift_type} drift pattern")
    return drifted_data

def main():
    """
    Main solution implementation demonstrating comprehensive feature monitoring
    """
    print("=== Day 37: Feature Monitoring & Drift - Solution ===")
    
    # Create sample data
    print("\n1. Creating Sample E-commerce Data...")
    reference_data = create_sample_ecommerce_data()
    
    # Create different types of drifted data
    gradual_drift_data = create_drifted_data(reference_data, 'gradual')
    sudden_drift_data = create_drifted_data(reference_data, 'sudden')
    seasonal_drift_data = create_drifted_data(reference_data, 'seasonal')
    
    # Define features to monitor
    numeric_features = ['user_age', 'session_duration', 'pages_viewed', 'cart_value', 
                       'product_price', 'product_rating', 'inventory_level']
    categorical_features = ['user_category', 'product_category']
    all_features = numeric_features + categorical_features
    
    print(f"   Monitoring {len(all_features)} features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    
    # Scenario 1: PSI Drift Detection
    print("\n2. Implementing PSI Drift Detection...")
    psi_detector = PopulationStabilityIndex(n_bins=10)
    psi_detector.fit(reference_data, all_features)
    
    # Test with different drift types
    drift_datasets = {
        'gradual': gradual_drift_data,
        'sudden': sudden_drift_data,
        'seasonal': seasonal_drift_data
    }
    
    psi_results = {}
    for drift_type, drift_data in drift_datasets.items():
        psi_scores = psi_detector.calculate_psi(drift_data)
        psi_results[drift_type] = psi_scores
        
        print(f"   {drift_type.capitalize()} Drift PSI Scores:")
        for feature, score in psi_scores.items():
            interpretation = psi_detector.interpret_psi(score)
            print(f"     {feature}: {score:.3f} ({interpretation})")
    
    # Scenario 2: KS Test Drift Detection
    print("\n3. Implementing KS Test Drift Detection...")
    ks_detector = KolmogorovSmirnovDriftDetector(significance_level=0.05)
    ks_detector.fit(reference_data, numeric_features)  # KS test for numeric features only
    
    ks_results = {}
    for drift_type, drift_data in drift_datasets.items():
        drift_results = ks_detector.detect_drift(drift_data)
        ks_results[drift_type] = drift_results
        
        print(f"   {drift_type.capitalize()} Drift KS Test Results:")
        for feature, result in drift_results.items():
            if 'drift_detected' in result:
                status = "DRIFT DETECTED" if result['drift_detected'] else "No drift"
                print(f"     {feature}: {status} (p-value: {result.get('p_value', 'N/A'):.4f})")
    
    # Scenario 3: ML-based Drift Detection
    print("\n4. Implementing ML-based Drift Detection...")
    ml_detector = MLBasedDriftDetector(model_type='isolation_forest', contamination=0.1)
    ml_detector.fit(reference_data, numeric_features)
    
    ml_results = {}
    for drift_type, drift_data in drift_datasets.items():
        drift_result = ml_detector.detect_drift(drift_data)
        ml_results[drift_type] = drift_result
        
        print(f"   {drift_type.capitalize()} Drift ML Detection:")
        print(f"     Overall drift score: {drift_result['overall_drift_score']:.3f}")
        print(f"     Drift detected: {drift_result['drift_detected']}")
        
        # Show top drifted features
        feature_scores = drift_result['feature_drift_scores']
        top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"     Top drifted features: {', '.join([f'{f}({s:.2f})' for f, s in top_features])}")
    
    # Domain classifier test
    print("\n   Training Domain Classifier...")
    for drift_type, drift_data in drift_datasets.items():
        auc_score = ml_detector.train_domain_classifier(reference_data, drift_data, numeric_features)
        print(f"     {drift_type.capitalize()} drift AUC: {auc_score:.3f} (higher = more drift)")
    
    # Scenario 4: Real-time Monitoring
    print("\n5. Setting up Real-time Monitoring...")
    rt_monitor = RealTimeDriftMonitor(window_size=500)
    
    # Configure feature monitoring
    feature_config = {}
    for feature in numeric_features[:3]:  # Monitor top 3 features
        feature_config[feature] = {
            'type': 'psi',
            'reference_data': reference_data,
            'alert_threshold': 0.2,
            'n_bins': 10
        }
    
    rt_monitor.setup_feature_monitoring(feature_config)
    
    # Simulate streaming data
    print("   Simulating streaming data...")
    streaming_data = gradual_drift_data.sample(n=1000, random_state=42)
    
    drift_scores_timeline = []
    for i, (_, row) in enumerate(streaming_data.iterrows()):
        feature_data = row.to_dict()
        drift_scores = rt_monitor.process_new_data(feature_data)
        
        if drift_scores:
            drift_scores_timeline.append({
                'timestamp': datetime.now() - timedelta(seconds=1000-i),
                'drift_scores': drift_scores
            })
    
    print(f"   Processed {len(streaming_data)} data points")
    print(f"   Generated {len(rt_monitor.alert_history)} drift alerts")
    
    if rt_monitor.alert_history:
        print("   Recent alerts:")
        for alert in rt_monitor.alert_history[-3:]:
            print(f"     {alert['timestamp']}: {alert['message']} (Severity: {alert['severity']})")
    
    # Scenario 5: Automated Retraining Pipeline
    print("\n6. Building Automated Retraining Pipeline...")
    retraining_config = {
        'triggers': {
            'data_drift': {'threshold': 0.2, 'critical_features': ['user_age', 'cart_value']},
            'performance_degradation': {'ctr': 0.03, 'conversion_rate': 0.02},
            'time_based': {'max_age_days': 7}
        },
        'strategies': {
            'full_retrain': {'drift_threshold': 0.5},
            'incremental_update': {'drift_threshold': 0.3},
            'feature_selection': {'drift_threshold': 0.1}
        }
    }
    
    retraining_pipeline = AutomatedRetrainingPipeline(retraining_config)
    
    # Test retraining triggers
    test_scenarios = [
        {'name': 'Minor drift', 'drift_results': {'user_age': 0.15, 'cart_value': 0.12}, 'performance': {'ctr': 0.045}},
        {'name': 'Major drift', 'drift_results': {'user_age': 0.35, 'cart_value': 0.28}, 'performance': {'ctr': 0.042}},
        {'name': 'Severe drift', 'drift_results': {'user_age': 0.65, 'cart_value': 0.58}, 'performance': {'ctr': 0.025}}
    ]
    
    for scenario in test_scenarios:
        print(f"\n   Testing scenario: {scenario['name']}")
        
        # Evaluate triggers
        triggers = retraining_pipeline.evaluate_retraining_triggers(
            scenario['drift_results'], 
            scenario['performance']
        )
        
        activated_triggers = [k for k, v in triggers.items() if v]
        print(f"     Activated triggers: {activated_triggers}")
        
        if any(triggers.values()):
            # Determine strategy
            strategy = retraining_pipeline.determine_retraining_strategy(
                triggers, 
                scenario['drift_results']
            )
            
            print(f"     Retraining strategy: {strategy['type']} ({strategy['reason']})")
            
            # Execute retraining
            result = retraining_pipeline.trigger_retraining(strategy)
            print(f"     Retraining result: {result['status']}")
            
            if result['status'] == 'completed':
                improvement = result.get('performance_improvement', 0)
                print(f"     Performance improvement: {improvement:.3f}")
    
    # Scenario 6: Monitoring Dashboard
    print("\n7. Creating Monitoring Dashboard...")
    dashboard = FeatureMonitoringDashboard({
        'psi': psi_detector,
        'ks': ks_detector,
        'ml': ml_detector
    })
    
    # Create dashboard summaries for key features
    key_features = ['user_age', 'cart_value', 'session_duration']
    
    for feature in key_features:
        # Get drift score from gradual drift scenario
        if feature in psi_results['gradual']:
            drift_score = psi_results['gradual'][feature]
            
            # Create drift summary
            summary = dashboard.create_drift_summary(feature, drift_score)
            print(f"   {feature}: {summary['status']} (Score: {drift_score:.3f}) - {summary['recommendation']}")
            
            # Create distribution comparison
            dist_chart = dashboard.create_distribution_chart(
                reference_data, gradual_drift_data, feature
            )
            
            mean_shift = dist_chart['mean_shift']
            print(f"     Mean shift: {mean_shift:.2f} ({mean_shift/dist_chart['reference_stats']['mean']*100:.1f}%)")
    
    # Summary
    print("\n=== Feature Monitoring Summary ===")
    
    # Count drift detections across methods
    drift_detections = {
        'PSI': sum(1 for scores in psi_results.values() for score in scores.values() if score > 0.2),
        'KS Test': sum(1 for results in ks_results.values() for result in results.values() if result.get('drift_detected', False)),
        'ML-based': sum(1 for result in ml_results.values() if result['drift_detected'])
    }
    
    print(f"Drift detections by method:")
    for method, count in drift_detections.items():
        print(f"  {method}: {count} features with significant drift")
    
    print(f"\nReal-time monitoring:")
    print(f"  Alerts generated: {len(rt_monitor.alert_history)}")
    print(f"  Features monitored: {len(feature_config)}")
    
    print(f"\nAutomated retraining:")
    print(f"  Retraining jobs executed: {len(retraining_pipeline.retraining_history)}")
    successful_jobs = [job for job in retraining_pipeline.retraining_history if job['status'] == 'completed']
    print(f"  Successful retraining jobs: {len(successful_jobs)}")
    
    print("\n🎯 Feature monitoring system successfully implemented!")
    print("   • Statistical drift detection (PSI, KS test)")
    print("   • ML-based drift detection (domain classifier, anomaly detection)")
    print("   • Real-time monitoring with sliding windows")
    print("   • Automated retraining pipeline with multiple strategies")
    print("   • Interactive dashboard components")
    
    print("\n=== Solution Complete ===")
    print("Review the implementation to understand feature monitoring best practices!")

if __name__ == "__main__":
    main()