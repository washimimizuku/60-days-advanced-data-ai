"""
Day 37: Feature Monitoring & Drift - Exercise

Business Scenario:
You're the MLOps Engineer at E-commerce Recommendation Platform. The company serves 
millions of users daily with personalized product recommendations. Recently, the 
recommendation system's performance has been declining, and you suspect feature drift 
due to changing user behavior patterns, seasonal trends, and new product categories.

Your task is to build a comprehensive feature monitoring and drift detection system 
that can identify distribution changes in real-time, alert the team to significant 
drift, and automatically trigger model retraining when necessary.

Requirements:
1. Implement statistical drift detection (PSI, KS test)
2. Build ML-based drift detection using domain classifiers
3. Create real-time monitoring with Kafka streams
4. Design automated retraining pipeline with triggers
5. Build interactive dashboard for monitoring and visualization

Success Criteria:
- Detect drift with <5% false positive rate
- Real-time alerts within 30 seconds of drift detection
- Automated retraining triggers based on drift severity
- Interactive dashboard showing drift trends and statistics
- Comprehensive logging and audit trail
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
import logging
from scipy import stats
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

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
        
        Args:
            reference_data: Reference dataset
            feature_columns: List of feature columns to monitor
        """
        self.feature_columns = feature_columns
        
        for column in feature_columns:
            if column not in reference_data.columns:
                continue
                
            data = reference_data[column].dropna()
            
            if pd.api.types.is_numeric_dtype(data):
                # Create quantile-based bins for numeric features
                bins = np.quantile(data, np.linspace(0, 1, self.n_bins + 1))
                bins = np.unique(bins)
                
                if len(bins) < 2:
                    bins = [data.min() - 1, data.max() + 1]
                
                # Calculate reference distribution
                bin_counts, _ = np.histogram(data, bins=bins)
                bin_proportions = bin_counts / len(data)
                
                # Ensure minimum bin size
                bin_proportions = np.maximum(bin_proportions, self.min_bin_size / self.n_bins)
                bin_proportions = bin_proportions / bin_proportions.sum()
                
                self.reference_stats[column] = {
                    'type': 'numeric',
                    'bins': bins,
                    'proportions': bin_proportions
                }
            else:
                # Handle categorical features
                value_counts = data.value_counts(normalize=True)
                
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
                
                total = sum(proportions.values())
                proportions = {k: v/total for k, v in proportions.items()}
                
                self.reference_stats[column] = {
                    'type': 'categorical',
                    'proportions': proportions,
                    'categories': list(proportions.keys())
                }
    
    def calculate_psi(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate PSI for current data against reference
        
        Args:
            current_data: Current dataset to compare
            
        Returns:
            Dictionary of PSI scores by feature
        """
        psi_scores = {}
        
        for column in self.feature_columns:
            if column not in current_data.columns or column not in self.reference_stats:
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
                            other_cats = set(current_counts.index) - set(reference_categories)
                            current_proportions[category] = sum(current_counts.get(cat, 0) for cat in other_cats)
                        else:
                            current_proportions[category] = current_counts.get(category, 0)
                    
                    # Ensure minimum proportion
                    for key in current_proportions:
                        current_proportions[key] = max(current_proportions[key], self.min_bin_size / len(current_proportions))
                    
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
                logging.warning(f"Error calculating PSI for column {column}: {str(e)}")
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
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, Dict]:
        """
        Detect drift in current data using KS test
        
        Returns:
            Dictionary with drift test results for each feature
        """
        drift_results = {}
        
        for column in self.feature_columns:
            if column not in current_data.columns or column not in self.reference_distributions:
                continue
                
            current_series = current_data[column].dropna()
            reference_info = self.reference_distributions[column]
            
            if reference_info['type'] == 'numeric':
                # KS test for numeric features
                from scipy.stats import ks_2samp
                statistic, p_value = ks_2samp(
                    reference_info['data'], 
                    current_series.values
                )
                
                drift_detected = p_value < self.significance_level
                
                # Calculate effect size
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
                from scipy.stats import chi2_contingency
                reference_counts = reference_info['value_counts']
                current_counts = current_series.value_counts()
                
                # Align categories
                all_categories = set(reference_counts.index) | set(current_counts.index)
                
                ref_aligned = [reference_counts.get(cat, 0) for cat in all_categories]
                curr_aligned = [current_counts.get(cat, 0) for cat in all_categories]
                
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
                    drift_results[column] = {
                        'test': 'chi_square',
                        'error': str(e),
                        'drift_detected': True,
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
        self.feature_columns = feature_columns
        
        # Prepare reference data (numeric features only)
        X_ref = reference_data[feature_columns].select_dtypes(include=[np.number])
        X_ref = X_ref.fillna(X_ref.mean())
        
        # Fit scaler
        X_ref_scaled = self.scaler.fit_transform(X_ref)
        
        # Fit anomaly detection model
        if self.model_type == 'isolation_forest':
            self.drift_model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        
        self.drift_model.fit(X_ref_scaled)
        
        # Store reference statistics
        self.reference_stats = {
            'mean': X_ref.mean(),
            'std': X_ref.std(),
            'feature_names': list(X_ref.columns)
        }
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """Detect drift using trained model"""
        if self.drift_model is None:
            raise ValueError("Detector not fitted. Call fit() first.")
        
        # Prepare current data
        X_curr = current_data[self.feature_columns].select_dtypes(include=[np.number])
        X_curr = X_curr.fillna(self.reference_stats['mean'])
        
        # Scale current data
        X_curr_scaled = self.scaler.transform(X_curr)
        
        # Get anomaly scores
        anomaly_scores = self.drift_model.decision_function(X_curr_scaled)
        outlier_predictions = self.drift_model.predict(X_curr_scaled)
        
        # Calculate drift metrics
        outlier_ratio = (outlier_predictions == -1).mean()
        mean_anomaly_score = anomaly_scores.mean()
        
        # Calculate feature-wise drift scores
        current_stats = {
            'mean': X_curr.mean(),
            'std': X_curr.std()
        }
        
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
            'drift_detected': outlier_ratio > self.contamination * 2,
            'current_stats': current_stats
        }
    
    def train_domain_classifier(self, reference_data: pd.DataFrame, 
                              current_data: pd.DataFrame, feature_columns: List[str]) -> float:
        """Train domain classifier to detect drift"""
        # Prepare data
        X_ref = reference_data[feature_columns].select_dtypes(include=[np.number])
        X_curr = current_data[feature_columns].select_dtypes(include=[np.number])
        
        # Fill missing values
        X_ref = X_ref.fillna(X_ref.mean())
        X_curr = X_curr.fillna(X_ref.mean())
        
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
        for feature_name, config in feature_config.items():
            # Initialize sliding window
            from collections import deque
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
            
            # Fit detector on reference data if provided
            if 'reference_data' in config:
                detector.fit(config['reference_data'], [feature_name])
            
            self.drift_detectors[feature_name] = detector
            self.alert_thresholds[feature_name] = config.get('alert_threshold', 0.2)
    
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
        except Exception as e:
            logging.warning(f"Error detecting drift for {feature_name}: {str(e)}")
            return None
    
    def _generate_alert(self, feature_name: str, drift_score: float, 
                       threshold: float, timestamp: datetime):
        """Generate drift alert"""
        alert = {
            'timestamp': timestamp.isoformat(),
            'feature_name': feature_name,
            'drift_score': drift_score,
            'threshold': threshold,
            'severity': 'high' if drift_score > threshold * 2 else 'medium',
            'message': f'Drift detected in {feature_name}: score {drift_score:.3f} > threshold {threshold:.3f}'
        }
        
        self.alert_history.append(alert)
        print(f"ALERT: {alert['message']}")
        
        return alert

class AutomatedRetrainingPipeline:
    """
    Automated model retraining pipeline triggered by drift detection
    """
    
    def __init__(self, config: Dict[str, any]):
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
                break
        
        # Check performance degradation
        perf_thresholds = self.config['triggers']['performance_degradation']
        
        for metric, threshold in perf_thresholds.items():
            if metric in model_performance:
                if model_performance[metric] < threshold:
                    triggers['performance_degradation'] = True
                    break
        
        # Check time-based trigger (simplified)
        max_age_days = self.config['triggers']['time_based']['max_age_days']
        # For demo purposes, assume time trigger is not activated
        triggers['time_based'] = False
        
        return triggers
    
    def determine_retraining_strategy(self, triggers: Dict[str, bool], 
                                    drift_results: Dict[str, float]) -> Dict[str, any]:
        """Determine appropriate retraining strategy"""
        max_drift = max(drift_results.values()) if drift_results else 0
        
        # Severe drift or performance issues -> full retrain
        if (triggers['performance_degradation'] or 
            max_drift > self.config['strategies']['full_retrain']['drift_threshold']):
            return {
                'type': 'full_retrain',
                'reason': 'severe_drift_or_performance_issues',
                'max_drift': max_drift
            }
        
        # Moderate drift -> incremental update
        elif max_drift > self.config['strategies']['incremental_update']['drift_threshold']:
            return {
                'type': 'incremental_update',
                'reason': 'moderate_drift',
                'max_drift': max_drift
            }
        
        # Minor drift -> feature selection retrain
        else:
            return {
                'type': 'feature_selection',
                'reason': 'time_based_or_minor_drift',
                'max_drift': max_drift
            }
    
    def trigger_retraining(self, strategy: Dict[str, any]) -> Dict[str, any]:
        """Execute retraining based on strategy (simulated)"""
        job_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Simulate retraining execution
        retraining_job = {
            'job_id': job_id,
            'strategy': strategy['type'],
            'reason': strategy['reason'],
            'timestamp': datetime.now().isoformat(),
            'status': 'completed',  # Simulated success
            'performance_improvement': np.random.uniform(0.01, 0.05)
        }
        
        self.retraining_history.append(retraining_job)
        return retraining_job

class FeatureMonitoringDashboard:
    """
    Interactive dashboard for feature monitoring and drift detection
    """
    
    def __init__(self, drift_detectors: Dict[str, any]):
        """Initialize monitoring dashboard"""
        self.drift_detectors = drift_detectors
        self.dashboard_data = {}
    
    def create_drift_summary(self, feature: str, drift_score: float) -> Dict[str, any]:
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
                                current_data: pd.DataFrame, feature: str) -> Dict[str, any]:
        """Create distribution comparison chart data"""
        chart_data = {
            'feature': feature,
            'reference_stats': {
                'mean': reference_data[feature].mean(),
                'std': reference_data[feature].std(),
                'count': len(reference_data)
            },
            'current_stats': {
                'mean': current_data[feature].mean(),
                'std': current_data[feature].std(),
                'count': len(current_data)
            }
        }
        
        # Calculate comparison metrics
        chart_data['mean_shift'] = chart_data['current_stats']['mean'] - chart_data['reference_stats']['mean']
        chart_data['std_ratio'] = chart_data['current_stats']['std'] / chart_data['reference_stats']['std']
        
        return chart_data
    
    def create_drift_timeline(self, drift_history: pd.DataFrame, feature: str) -> Dict[str, any]:
        """Create drift score timeline chart data"""
        timeline_data = {
            'feature': feature,
            'timestamps': drift_history['timestamp'].tolist() if 'timestamp' in drift_history.columns else [],
            'drift_scores': drift_history['drift_score'].tolist() if 'drift_score' in drift_history.columns else [],
            'thresholds': {
                'minor': 0.1,
                'major': 0.2,
                'severe': 0.5
            }
        }
        
        return timeline_data

def create_sample_ecommerce_data():
    """
    Create sample e-commerce recommendation data with drift
    """
    np.random.seed(42)
    
    # TODO: Generate realistic e-commerce features
    # HINT: Include user behavior, product features, seasonal patterns
    
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
    
    return pd.DataFrame(data)

def create_drifted_data(original_data: pd.DataFrame, drift_type: str = 'gradual') -> pd.DataFrame:
    """
    Create drifted version of original data
    
    Args:
        original_data: Original dataset
        drift_type: Type of drift to apply ('gradual', 'sudden', 'seasonal')
        
    Returns:
        Drifted dataset
    """
    drifted_data = original_data.copy()
    
    if drift_type == 'gradual':
        # Gradual shift in user behavior
        drifted_data['user_age'] = drifted_data['user_age'] + np.random.normal(5, 2, len(drifted_data))
        drifted_data['session_duration'] = drifted_data['session_duration'] * np.random.uniform(1.2, 1.5, len(drifted_data))
        drifted_data['cart_value'] = drifted_data['cart_value'] * np.random.uniform(0.8, 1.3, len(drifted_data))
        
    elif drift_type == 'sudden':
        # Sudden shift (e.g., new product launch)
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
    
    return drifted_data

def main():
    """
    Main exercise implementation
    """
    print("=== Day 37: Feature Monitoring & Drift - Exercise ===")
    
    # Create sample data
    print("\n1. Creating Sample E-commerce Data...")
    reference_data = create_sample_ecommerce_data()
    print(f"   Created reference dataset with {len(reference_data)} samples")
    
    # Create drifted datasets
    gradual_drift_data = create_drifted_data(reference_data, 'gradual')
    sudden_drift_data = create_drifted_data(reference_data, 'sudden')
    
    # Define features to monitor
    numeric_features = ['user_age', 'session_duration', 'cart_value', 'product_price']
    categorical_features = ['user_category', 'product_category']
    all_features = numeric_features + categorical_features
    
    # Scenario 1 - PSI Drift Detection
    print("\n2. Implementing PSI Drift Detection...")
    psi_detector = PopulationStabilityIndex()
    psi_detector.fit(reference_data, all_features)
    
    # Test with gradual drift
    psi_scores = psi_detector.calculate_psi(gradual_drift_data)
    print("   PSI Scores (Gradual Drift):")
    for feature, score in psi_scores.items():
        interpretation = psi_detector.interpret_psi(score)
        print(f"     {feature}: {score:.3f} ({interpretation})")
    
    # Scenario 2 - KS Test Drift Detection
    print("\n3. Implementing KS Test Drift Detection...")
    ks_detector = KolmogorovSmirnovDriftDetector()
    ks_detector.fit(reference_data, numeric_features)  # KS test for numeric only
    
    # Test drift detection
    ks_results = ks_detector.detect_drift(gradual_drift_data)
    print("   KS Test Results (Gradual Drift):")
    for feature, result in ks_results.items():
        if 'drift_detected' in result:
            status = "DRIFT DETECTED" if result['drift_detected'] else "No drift"
            print(f"     {feature}: {status} (p-value: {result.get('p_value', 'N/A'):.4f})")
    
    # Scenario 3 - ML-based Drift Detection
    print("\n4. Implementing ML-based Drift Detection...")
    ml_detector = MLBasedDriftDetector()
    ml_detector.fit(reference_data, numeric_features)
    
    # Test ML-based detection
    ml_results = ml_detector.detect_drift(gradual_drift_data)
    print("   ML-based Detection Results:")
    print(f"     Overall drift score: {ml_results['overall_drift_score']:.3f}")
    print(f"     Drift detected: {ml_results['drift_detected']}")
    
    # Scenario 4 - Real-time Monitoring
    print("\n5. Setting up Real-time Monitoring...")
    rt_monitor = RealTimeDriftMonitor(window_size=200)
    
    # Configure feature monitoring
    feature_config = {
        'user_age': {
            'type': 'psi',
            'reference_data': reference_data,
            'alert_threshold': 0.2
        },
        'cart_value': {
            'type': 'psi',
            'reference_data': reference_data,
            'alert_threshold': 0.2
        }
    }
    
    rt_monitor.setup_feature_monitoring(feature_config)
    
    # Simulate streaming data
    print("   Simulating streaming data...")
    streaming_data = gradual_drift_data.sample(n=300, random_state=42)
    
    for i, (_, row) in enumerate(streaming_data.iterrows()):
        feature_data = {'user_age': row['user_age'], 'cart_value': row['cart_value']}
        drift_scores = rt_monitor.process_new_data(feature_data)
        
        if drift_scores and i % 50 == 0:
            print(f"     Step {i}: Drift scores = {drift_scores}")
    
    print(f"   Generated {len(rt_monitor.alert_history)} alerts")
    
    # Scenario 5 - Automated Retraining Pipeline
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
    test_drift_results = {'user_age': 0.35, 'cart_value': 0.28}
    test_performance = {'ctr': 0.025}
    
    triggers = retraining_pipeline.evaluate_retraining_triggers(test_drift_results, test_performance)
    print(f"   Activated triggers: {[k for k, v in triggers.items() if v]}")
    
    if any(triggers.values()):
        strategy = retraining_pipeline.determine_retraining_strategy(triggers, test_drift_results)
        print(f"   Recommended strategy: {strategy['type']} ({strategy['reason']})")
        
        # Execute retraining (simulated)
        result = retraining_pipeline.trigger_retraining(strategy)
        print(f"   Retraining result: {result['status']} (Job ID: {result['job_id']})")
    
    # Scenario 6 - Monitoring Dashboard
    print("\n7. Creating Monitoring Dashboard...")
    dashboard = FeatureMonitoringDashboard({
        'psi': psi_detector,
        'ks': ks_detector,
        'ml': ml_detector
    })
    
    # Create sample dashboard components
    for feature in ['user_age', 'cart_value']:
        if feature in psi_scores:
            summary = dashboard.create_drift_summary(feature, psi_scores[feature])
            print(f"   {feature}: {summary['status']} - {summary['recommendation']}")
            
            # Create distribution comparison
            dist_chart = dashboard.create_distribution_chart(
                reference_data, gradual_drift_data, feature
            )
            mean_shift = dist_chart['mean_shift']
            print(f"     Mean shift: {mean_shift:.2f}")
    
    print("\n=== Exercise Complete ===")
    print("\n🎯 Successfully implemented:")
    print("   • PSI drift detection with multiple features")
    print("   • KS test statistical significance testing")
    print("   • ML-based anomaly detection for drift")
    print("   • Real-time monitoring with sliding windows")
    print("   • Automated retraining pipeline with triggers")
    print("   • Dashboard components for visualization")
    
    # Summary statistics
    print("\n📊 Summary Statistics:")
    print(f"   Features monitored: {len(all_features)}")
    print(f"   PSI detections: {sum(1 for score in psi_scores.values() if score > 0.2)}")
    print(f"   KS detections: {sum(1 for result in ks_results.values() if result.get('drift_detected', False))}")
    print(f"   ML drift detected: {ml_results['drift_detected']}")
    print(f"   Real-time alerts: {len(rt_monitor.alert_history)}")
    print(f"   Retraining jobs: {len(retraining_pipeline.retraining_history)}")
    
    print("\nNext: Review solution.py for complete implementation and take the quiz!")

if __name__ == "__main__":
    main()