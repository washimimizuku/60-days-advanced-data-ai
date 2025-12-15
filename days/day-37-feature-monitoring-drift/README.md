# Day 37: Feature Monitoring & Drift - Data Quality, Model Performance

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Implement comprehensive feature monitoring** for production ML systems
- **Detect and respond to data drift** using statistical and ML-based methods
- **Build automated alerting systems** for feature quality degradation
- **Create drift detection pipelines** with real-time and batch processing
- **Design retraining triggers** based on drift severity and business impact

**Estimated Time**: 1 hour  
**Difficulty Level**: Advanced ⭐⭐⭐⭐

---

## 🎯 What is Feature Monitoring & Drift Detection?

Feature monitoring and drift detection are critical components of MLOps that ensure production ML models remain effective over time by continuously monitoring input data quality, distribution changes, and model performance degradation.

### Key Components of Feature Monitoring

**1. Data Drift Detection**
- Statistical tests for distribution changes
- Feature-level drift scoring and alerting
- Population stability index (PSI) monitoring
- Kolmogorov-Smirnov test implementation

**2. Concept Drift Detection**
- Model performance degradation monitoring
- Prediction accuracy tracking over time
- Business metric correlation analysis
- Adaptive threshold management

**3. Feature Quality Monitoring**
- Missing value rate tracking
- Data type validation and schema compliance
- Outlier detection and anomaly scoring
- Feature correlation change detection

**4. Real-time Alerting**
- Automated notification systems
- Severity-based escalation procedures
- Integration with incident management
- Dashboard visualization and reporting

---

## 🔧 Statistical Drift Detection Methods

### 1. **Population Stability Index (PSI)**

```python
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings

class PopulationStabilityIndex:
    """
    Population Stability Index (PSI) for detecting feature drift
    """
    
    def __init__(self, n_bins: int = 10, min_bin_size: float = 0.05):
        """
        Initialize PSI calculator
        
        Args:
            n_bins: Number of bins for discretization
            min_bin_size: Minimum bin size as fraction of total samples
        """
        self.n_bins = n_bins
        self.min_bin_size = min_bin_size
        self.reference_bins = {}
        self.reference_stats = {}
        
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
                # For numeric features, create quantile-based bins
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
                    'std': data.std(),
                    'min': data.min(),
                    'max': data.max()
                }
                
            else:
                # For categorical features
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
                
                # Calculate PSI
                psi = np.sum((current_proportions - reference_proportions) * 
                           np.log(current_proportions / reference_proportions))
                
                psi_scores[column] = psi
                
            except Exception as e:
                logging.warning(f"Error calculating PSI for column {column}: {str(e)}")
                psi_scores[column] = float('inf')
        
        return psi_scores
    
    def interpret_psi(self, psi_score: float) -> str:
        """
        Interpret PSI score
        
        Args:
            psi_score: PSI score to interpret
            
        Returns:
            Interpretation string
        """
        if psi_score < 0.1:
            return "No significant change"
        elif psi_score < 0.2:
            return "Minor change"
        elif psi_score < 0.5:
            return "Major change"
        else:
            return "Severe change - investigate immediately"
```
### 2. **Kolmogorov-Smirnov Test for Drift Detection**

```python
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

class KolmogorovSmirnovDriftDetector:
    """
    Kolmogorov-Smirnov test for detecting distribution drift
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize KS drift detector
        
        Args:
            significance_level: Statistical significance level for drift detection
        """
        self.significance_level = significance_level
        self.reference_distributions = {}
        
    def fit(self, reference_data: pd.DataFrame, feature_columns: List[str]):
        """
        Fit detector on reference data
        
        Args:
            reference_data: Reference dataset
            feature_columns: Features to monitor for drift
        """
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
        Detect drift in current data
        
        Args:
            current_data: Current dataset to test for drift
            
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
    
    def visualize_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame, 
                       feature: str, save_path: Optional[str] = None):
        """
        Visualize drift for a specific feature
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            feature: Feature to visualize
            save_path: Optional path to save the plot
        """
        if feature not in self.reference_distributions:
            raise ValueError(f"Feature {feature} not found in reference distributions")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ref_data = reference_data[feature].dropna()
        curr_data = current_data[feature].dropna()
        
        if self.reference_distributions[feature]['type'] == 'numeric':
            # Histogram comparison
            ax1.hist(ref_data, bins=30, alpha=0.7, label='Reference', density=True)
            ax1.hist(curr_data, bins=30, alpha=0.7, label='Current', density=True)
            ax1.set_xlabel(feature)
            ax1.set_ylabel('Density')
            ax1.set_title(f'{feature} - Distribution Comparison')
            ax1.legend()
            
            # Q-Q plot
            from scipy.stats import probplot
            probplot(curr_data, dist=stats.norm, plot=ax2)
            ax2.set_title(f'{feature} - Q-Q Plot (Current vs Normal)')
            
        else:
            # Bar plot for categorical
            ref_counts = ref_data.value_counts()
            curr_counts = curr_data.value_counts()
            
            all_cats = set(ref_counts.index) | set(curr_counts.index)
            
            ref_props = [ref_counts.get(cat, 0) / len(ref_data) for cat in all_cats]
            curr_props = [curr_counts.get(cat, 0) / len(curr_data) for cat in all_cats]
            
            x = np.arange(len(all_cats))
            width = 0.35
            
            ax1.bar(x - width/2, ref_props, width, label='Reference', alpha=0.7)
            ax1.bar(x + width/2, curr_props, width, label='Current', alpha=0.7)
            ax1.set_xlabel('Categories')
            ax1.set_ylabel('Proportion')
            ax1.set_title(f'{feature} - Category Distribution')
            ax1.set_xticks(x)
            ax1.set_xticklabels(list(all_cats), rotation=45)
            ax1.legend()
            
            # Difference plot
            diff = np.array(curr_props) - np.array(ref_props)
            colors = ['red' if d < 0 else 'green' for d in diff]
            ax2.bar(x, diff, color=colors, alpha=0.7)
            ax2.set_xlabel('Categories')
            ax2.set_ylabel('Proportion Difference')
            ax2.set_title(f'{feature} - Distribution Change')
            ax2.set_xticks(x)
            ax2.set_xticklabels(list(all_cats), rotation=45)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

### 3. **Advanced Drift Detection with Machine Learning**

```python
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

class MLBasedDriftDetector:
    """
    Machine Learning based drift detection using domain classifier approach
    """
    
    def __init__(self, model_type: str = 'isolation_forest', contamination: float = 0.1):
        """
        Initialize ML-based drift detector
        
        Args:
            model_type: Type of model to use ('isolation_forest', 'one_class_svm')
            contamination: Expected proportion of outliers in the data
        """
        self.model_type = model_type
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.drift_model = None
        self.domain_classifier = None
        self.feature_columns = None
        
    def fit(self, reference_data: pd.DataFrame, feature_columns: List[str]):
        """
        Fit drift detector on reference data
        
        Args:
            reference_data: Reference dataset
            feature_columns: Features to use for drift detection
        """
        self.feature_columns = feature_columns
        
        # Prepare reference data
        X_ref = reference_data[feature_columns].select_dtypes(include=[np.number])
        X_ref = X_ref.fillna(X_ref.mean())  # Simple imputation
        
        # Fit scaler
        X_ref_scaled = self.scaler.fit_transform(X_ref)
        
        # Fit anomaly detection model
        if self.model_type == 'isolation_forest':
            from sklearn.ensemble import IsolationForest
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
            'max': X_ref.max()
        }
    
    def detect_drift(self, current_data: pd.DataFrame) -> Dict[str, float]:
        """
        Detect drift using trained model
        
        Args:
            current_data: Current dataset to test
            
        Returns:
            Dictionary with drift scores and metrics
        """
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
        """
        Train domain classifier to detect drift
        
        Args:
            reference_data: Reference dataset
            current_data: Current dataset
            feature_columns: Features to use
            
        Returns:
            AUC score of domain classifier (higher = more drift)
        """
        from sklearn.ensemble import RandomForestClassifier
        
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
        X_scaled = self.scaler.fit_transform(X_combined)
        
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
```

---

## 🔍 Real-time Feature Monitoring System

### 1. **Streaming Drift Detection with Kafka**

```python
import json
from kafka import KafkaConsumer, KafkaProducer
from typing import Dict, Any
import threading
import time
from collections import deque
import numpy as np

class RealTimeDriftMonitor:
    """
    Real-time drift monitoring system using Kafka streams
    """
    
    def __init__(self, kafka_config: Dict[str, str], window_size: int = 1000):
        """
        Initialize real-time drift monitor
        
        Args:
            kafka_config: Kafka configuration
            window_size: Size of sliding window for drift detection
        """
        self.kafka_config = kafka_config
        self.window_size = window_size
        self.feature_windows = {}
        self.drift_detectors = {}
        self.alert_thresholds = {}
        self.running = False
        
        # Initialize Kafka producer for alerts
        self.producer = KafkaProducer(
            bootstrap_servers=kafka_config['bootstrap_servers'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
    def setup_feature_monitoring(self, feature_config: Dict[str, Dict]):
        """
        Setup monitoring for specific features
        
        Args:
            feature_config: Configuration for each feature to monitor
        """
        for feature_name, config in feature_config.items():
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
    
    def start_monitoring(self, input_topic: str, alert_topic: str):
        """
        Start real-time monitoring
        
        Args:
            input_topic: Kafka topic to consume feature data from
            alert_topic: Kafka topic to publish alerts to
        """
        self.running = True
        self.input_topic = input_topic
        self.alert_topic = alert_topic
        
        # Start consumer thread
        consumer_thread = threading.Thread(
            target=self._consume_features,
            args=(input_topic,)
        )
        consumer_thread.start()
        
        # Start drift detection thread
        detection_thread = threading.Thread(
            target=self._detect_drift_continuously
        )
        detection_thread.start()
        
        return consumer_thread, detection_thread
    
    def _consume_features(self, topic: str):
        """
        Consume feature data from Kafka topic
        
        Args:
            topic: Kafka topic to consume from
        """
        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.kafka_config['bootstrap_servers'],
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            auto_offset_reset='latest'
        )
        
        for message in consumer:
            if not self.running:
                break
                
            try:
                feature_data = message.value
                timestamp = feature_data.get('timestamp', time.time())
                
                # Add to sliding windows
                for feature_name in self.feature_windows:
                    if feature_name in feature_data:
                        self.feature_windows[feature_name].append({
                            'value': feature_data[feature_name],
                            'timestamp': timestamp
                        })
                        
            except Exception as e:
                logging.error(f"Error processing message: {str(e)}")
        
        consumer.close()
    
    def _detect_drift_continuously(self):
        """
        Continuously detect drift in sliding windows
        """
        while self.running:
            try:
                for feature_name, window in self.feature_windows.items():
                    if len(window) >= self.window_size:
                        # Extract values from window
                        values = [item['value'] for item in window]
                        current_data = pd.DataFrame({feature_name: values})
                        
                        # Detect drift
                        detector = self.drift_detectors[feature_name]
                        
                        if isinstance(detector, PopulationStabilityIndex):
                            drift_scores = detector.calculate_psi(current_data)
                            drift_score = drift_scores.get(feature_name, 0)
                        elif isinstance(detector, KolmogorovSmirnovDriftDetector):
                            drift_results = detector.detect_drift(current_data)
                            drift_score = drift_results.get(feature_name, {}).get('statistic', 0)
                        
                        # Check if alert threshold is exceeded
                        threshold = self.alert_thresholds[feature_name]
                        
                        if drift_score > threshold:
                            self._send_drift_alert(feature_name, drift_score, threshold)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"Error in drift detection: {str(e)}")
                time.sleep(60)  # Wait longer on error
    
    def _send_drift_alert(self, feature_name: str, drift_score: float, threshold: float):
        """
        Send drift alert to Kafka topic
        
        Args:
            feature_name: Name of feature with drift
            drift_score: Calculated drift score
            threshold: Alert threshold that was exceeded
        """
        alert = {
            'timestamp': datetime.now().isoformat(),
            'alert_type': 'feature_drift',
            'feature_name': feature_name,
            'drift_score': drift_score,
            'threshold': threshold,
            'severity': self._calculate_severity(drift_score, threshold),
            'message': f'Drift detected in feature {feature_name}: score {drift_score:.3f} > threshold {threshold:.3f}'
        }
        
        try:
            self.producer.send(self.alert_topic, value=alert)
            logging.warning(f"Drift alert sent for feature {feature_name}")
        except Exception as e:
            logging.error(f"Failed to send drift alert: {str(e)}")
    
    def _calculate_severity(self, drift_score: float, threshold: float) -> str:
        """
        Calculate alert severity based on drift score
        
        Args:
            drift_score: Calculated drift score
            threshold: Alert threshold
            
        Returns:
            Severity level string
        """
        ratio = drift_score / threshold
        
        if ratio < 1.5:
            return 'low'
        elif ratio < 3.0:
            return 'medium'
        elif ratio < 5.0:
            return 'high'
        else:
            return 'critical'
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.running = False
        self.producer.close()
```
### 2. **Automated Retraining Pipeline**

```python
import mlflow
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import subprocess
import yaml

class AutomatedRetrainingPipeline:
    """
    Automated model retraining pipeline triggered by drift detection
    """
    
    def __init__(self, config_path: str = "retraining_config.yaml"):
        """
        Initialize retraining pipeline
        
        Args:
            config_path: Path to retraining configuration file
        """
        self.config_path = config_path
        self.load_config()
        self.mlflow_client = mlflow.tracking.MlflowClient()
        
    def load_config(self):
        """Load retraining configuration"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def evaluate_retraining_triggers(self, drift_results: Dict[str, float], 
                                   model_performance: Dict[str, float]) -> Dict[str, bool]:
        """
        Evaluate whether retraining should be triggered
        
        Args:
            drift_results: Drift detection results by feature
            model_performance: Current model performance metrics
            
        Returns:
            Dictionary indicating which triggers are activated
        """
        triggers = {
            'data_drift': False,
            'performance_degradation': False,
            'time_based': False,
            'manual_trigger': False
        }
        
        # Check data drift triggers
        drift_threshold = self.config['triggers']['data_drift']['threshold']
        critical_features = self.config['triggers']['data_drift']['critical_features']
        
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
        
        # Check time-based trigger
        last_training_time = self._get_last_training_time()
        max_age_days = self.config['triggers']['time_based']['max_age_days']
        
        if last_training_time:
            age_days = (datetime.now() - last_training_time).days
            if age_days > max_age_days:
                triggers['time_based'] = True
        
        return triggers
    
    def trigger_retraining(self, triggers: Dict[str, bool], 
                          drift_results: Dict[str, float]) -> Dict[str, Any]:
        """
        Trigger model retraining based on activated triggers
        
        Args:
            triggers: Dictionary of activated triggers
            drift_results: Drift detection results
            
        Returns:
            Retraining job information
        """
        if not any(triggers.values()):
            return {'status': 'no_retraining_needed', 'triggers': triggers}
        
        # Determine retraining strategy based on triggers
        strategy = self._determine_retraining_strategy(triggers, drift_results)
        
        # Create retraining job
        job_config = {
            'job_id': f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'strategy': strategy,
            'triggers': triggers,
            'drift_results': drift_results,
            'timestamp': datetime.now().isoformat()
        }
        
        # Execute retraining
        if strategy['type'] == 'full_retrain':
            result = self._execute_full_retraining(job_config)
        elif strategy['type'] == 'incremental_update':
            result = self._execute_incremental_update(job_config)
        elif strategy['type'] == 'feature_selection':
            result = self._execute_feature_selection_retrain(job_config)
        
        return result
    
    def _determine_retraining_strategy(self, triggers: Dict[str, bool], 
                                     drift_results: Dict[str, float]) -> Dict[str, Any]:
        """
        Determine appropriate retraining strategy
        
        Args:
            triggers: Activated triggers
            drift_results: Drift detection results
            
        Returns:
            Retraining strategy configuration
        """
        # Severe drift or performance issues -> full retrain
        max_drift = max(drift_results.values()) if drift_results else 0
        
        if (triggers['performance_degradation'] or 
            max_drift > self.config['strategies']['full_retrain']['drift_threshold']):
            return {
                'type': 'full_retrain',
                'reason': 'severe_drift_or_performance_issues',
                'config': self.config['strategies']['full_retrain']
            }
        
        # Moderate drift -> incremental update
        elif max_drift > self.config['strategies']['incremental_update']['drift_threshold']:
            return {
                'type': 'incremental_update',
                'reason': 'moderate_drift',
                'config': self.config['strategies']['incremental_update']
            }
        
        # Time-based or minor drift -> feature selection retrain
        else:
            return {
                'type': 'feature_selection',
                'reason': 'time_based_or_minor_drift',
                'config': self.config['strategies']['feature_selection']
            }
    
    def _execute_full_retraining(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute full model retraining
        
        Args:
            job_config: Retraining job configuration
            
        Returns:
            Retraining results
        """
        try:
            # Start MLflow run
            with mlflow.start_run(run_name=f"retrain_{job_config['job_id']}") as run:
                # Log retraining metadata
                mlflow.log_params({
                    'retraining_strategy': 'full_retrain',
                    'triggers': str(job_config['triggers']),
                    'job_id': job_config['job_id']
                })
                
                # Execute training pipeline
                training_script = self.config['scripts']['full_retrain']
                result = subprocess.run(
                    ['python', training_script, '--job-id', job_config['job_id']],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    # Log success
                    mlflow.log_metric('retraining_success', 1)
                    
                    return {
                        'status': 'success',
                        'job_id': job_config['job_id'],
                        'run_id': run.info.run_id,
                        'strategy': 'full_retrain',
                        'output': result.stdout
                    }
                else:
                    # Log failure
                    mlflow.log_metric('retraining_success', 0)
                    mlflow.log_text(result.stderr, 'error_log.txt')
                    
                    return {
                        'status': 'failed',
                        'job_id': job_config['job_id'],
                        'run_id': run.info.run_id,
                        'error': result.stderr
                    }
                    
        except Exception as e:
            return {
                'status': 'failed',
                'job_id': job_config['job_id'],
                'error': str(e)
            }
    
    def _execute_incremental_update(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute incremental model update
        
        Args:
            job_config: Retraining job configuration
            
        Returns:
            Update results
        """
        try:
            # Load current model
            current_model_uri = self._get_current_production_model()
            
            with mlflow.start_run(run_name=f"incremental_{job_config['job_id']}") as run:
                mlflow.log_params({
                    'retraining_strategy': 'incremental_update',
                    'base_model_uri': current_model_uri,
                    'job_id': job_config['job_id']
                })
                
                # Execute incremental update script
                update_script = self.config['scripts']['incremental_update']
                result = subprocess.run(
                    ['python', update_script, 
                     '--base-model', current_model_uri,
                     '--job-id', job_config['job_id']],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    mlflow.log_metric('update_success', 1)
                    
                    return {
                        'status': 'success',
                        'job_id': job_config['job_id'],
                        'run_id': run.info.run_id,
                        'strategy': 'incremental_update',
                        'base_model': current_model_uri
                    }
                else:
                    mlflow.log_metric('update_success', 0)
                    mlflow.log_text(result.stderr, 'error_log.txt')
                    
                    return {
                        'status': 'failed',
                        'job_id': job_config['job_id'],
                        'error': result.stderr
                    }
                    
        except Exception as e:
            return {
                'status': 'failed',
                'job_id': job_config['job_id'],
                'error': str(e)
            }
    
    def _execute_feature_selection_retrain(self, job_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute retraining with feature selection
        
        Args:
            job_config: Retraining job configuration
            
        Returns:
            Retraining results
        """
        try:
            with mlflow.start_run(run_name=f"feature_select_{job_config['job_id']}") as run:
                mlflow.log_params({
                    'retraining_strategy': 'feature_selection',
                    'drift_results': str(job_config['drift_results']),
                    'job_id': job_config['job_id']
                })
                
                # Execute feature selection retraining
                script = self.config['scripts']['feature_selection']
                result = subprocess.run(
                    ['python', script, 
                     '--drift-results', json.dumps(job_config['drift_results']),
                     '--job-id', job_config['job_id']],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    mlflow.log_metric('retraining_success', 1)
                    
                    return {
                        'status': 'success',
                        'job_id': job_config['job_id'],
                        'run_id': run.info.run_id,
                        'strategy': 'feature_selection'
                    }
                else:
                    mlflow.log_metric('retraining_success', 0)
                    mlflow.log_text(result.stderr, 'error_log.txt')
                    
                    return {
                        'status': 'failed',
                        'job_id': job_config['job_id'],
                        'error': result.stderr
                    }
                    
        except Exception as e:
            return {
                'status': 'failed',
                'job_id': job_config['job_id'],
                'error': str(e)
            }
    
    def _get_last_training_time(self) -> Optional[datetime]:
        """Get timestamp of last model training"""
        try:
            # Get latest model version from MLflow
            model_name = self.config['model']['name']
            latest_versions = self.mlflow_client.get_latest_versions(
                model_name, stages=["Production"]
            )
            
            if latest_versions:
                latest_version = latest_versions[0]
                run = self.mlflow_client.get_run(latest_version.run_id)
                return datetime.fromtimestamp(run.info.start_time / 1000)
            
            return None
            
        except Exception as e:
            logging.error(f"Error getting last training time: {str(e)}")
            return None
    
    def _get_current_production_model(self) -> str:
        """Get URI of current production model"""
        model_name = self.config['model']['name']
        latest_versions = self.mlflow_client.get_latest_versions(
            model_name, stages=["Production"]
        )
        
        if latest_versions:
            return f"models:/{model_name}/{latest_versions[0].version}"
        
        raise ValueError("No production model found")

### 3. **Comprehensive Monitoring Dashboard**

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import pandas as pd

class FeatureMonitoringDashboard:
    """
    Interactive dashboard for feature monitoring and drift detection
    """
    
    def __init__(self, drift_detector, port: int = 8050):
        """
        Initialize monitoring dashboard
        
        Args:
            drift_detector: Configured drift detector
            port: Port to run dashboard on
        """
        self.drift_detector = drift_detector
        self.port = port
        self.app = Dash(__name__)
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Setup dashboard layout"""
        self.app.layout = html.Div([
            html.H1("Feature Monitoring & Drift Detection Dashboard", 
                   style={'textAlign': 'center', 'marginBottom': 30}),
            
            # Control panel
            html.Div([
                html.Div([
                    html.Label("Select Feature:"),
                    dcc.Dropdown(
                        id='feature-dropdown',
                        options=[{'label': f, 'value': f} for f in self.drift_detector.feature_columns],
                        value=self.drift_detector.feature_columns[0] if self.drift_detector.feature_columns else None
                    )
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                html.Div([
                    html.Label("Time Range:"),
                    dcc.Dropdown(
                        id='time-range-dropdown',
                        options=[
                            {'label': 'Last Hour', 'value': '1h'},
                            {'label': 'Last 6 Hours', 'value': '6h'},
                            {'label': 'Last Day', 'value': '1d'},
                            {'label': 'Last Week', 'value': '7d'}
                        ],
                        value='1d'
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),
                
                html.Div([
                    html.Button('Refresh Data', id='refresh-button', n_clicks=0,
                               style={'backgroundColor': '#007bff', 'color': 'white', 'border': 'none',
                                     'padding': '10px 20px', 'borderRadius': '5px'})
                ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%', 'textAlign': 'right'})
            ], style={'marginBottom': 30}),
            
            # Drift score summary
            html.Div(id='drift-summary', style={'marginBottom': 30}),
            
            # Main charts
            html.Div([
                # Distribution comparison
                html.Div([
                    dcc.Graph(id='distribution-chart')
                ], style={'width': '50%', 'display': 'inline-block'}),
                
                # Drift score over time
                html.Div([
                    dcc.Graph(id='drift-timeline-chart')
                ], style={'width': '50%', 'display': 'inline-block'})
            ]),
            
            # Feature statistics table
            html.Div([
                html.H3("Feature Statistics"),
                html.Div(id='feature-stats-table')
            ], style={'marginTop': 30}),
            
            # Alert log
            html.Div([
                html.H3("Recent Alerts"),
                html.Div(id='alert-log')
            ], style={'marginTop': 30})
        ])
    
    def setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('drift-summary', 'children'),
             Output('distribution-chart', 'figure'),
             Output('drift-timeline-chart', 'figure'),
             Output('feature-stats-table', 'children'),
             Output('alert-log', 'children')],
            [Input('feature-dropdown', 'value'),
             Input('time-range-dropdown', 'value'),
             Input('refresh-button', 'n_clicks')]
        )
        def update_dashboard(selected_feature, time_range, n_clicks):
            if not selected_feature:
                return "No feature selected", {}, {}, "No data", "No alerts"
            
            # Generate sample data for demonstration
            reference_data, current_data, drift_history = self._generate_sample_data(selected_feature, time_range)
            
            # Calculate drift
            drift_results = self.drift_detector.detect_drift(current_data)
            drift_score = drift_results.get(selected_feature, {}).get('statistic', 0)
            
            # Create drift summary
            drift_summary = self._create_drift_summary(selected_feature, drift_score)
            
            # Create distribution chart
            dist_chart = self._create_distribution_chart(reference_data, current_data, selected_feature)
            
            # Create drift timeline chart
            timeline_chart = self._create_drift_timeline_chart(drift_history, selected_feature)
            
            # Create feature statistics table
            stats_table = self._create_feature_stats_table(reference_data, current_data, selected_feature)
            
            # Create alert log
            alert_log = self._create_alert_log(selected_feature, drift_score)
            
            return drift_summary, dist_chart, timeline_chart, stats_table, alert_log
    
    def _generate_sample_data(self, feature: str, time_range: str):
        """Generate sample data for demonstration"""
        np.random.seed(42)
        
        # Reference data
        ref_data = pd.DataFrame({
            feature: np.random.normal(100, 15, 1000)
        })
        
        # Current data with some drift
        drift_factor = 1.2 if time_range in ['1d', '7d'] else 1.0
        curr_data = pd.DataFrame({
            feature: np.random.normal(100 * drift_factor, 15 * drift_factor, 500)
        })
        
        # Drift history
        timestamps = pd.date_range(end=datetime.now(), periods=100, freq='1H')
        drift_scores = np.random.exponential(0.1, 100) * (1 + 0.5 * np.sin(np.arange(100) * 0.1))
        
        drift_history = pd.DataFrame({
            'timestamp': timestamps,
            'drift_score': drift_scores
        })
        
        return ref_data, curr_data, drift_history
    
    def _create_drift_summary(self, feature: str, drift_score: float):
        """Create drift summary component"""
        if drift_score < 0.1:
            status = "Normal"
            color = "green"
        elif drift_score < 0.2:
            status = "Minor Drift"
            color = "orange"
        else:
            status = "Significant Drift"
            color = "red"
        
        return html.Div([
            html.H3(f"Drift Status for {feature}"),
            html.Div([
                html.Span(f"Status: ", style={'fontWeight': 'bold'}),
                html.Span(status, style={'color': color, 'fontWeight': 'bold', 'fontSize': '18px'}),
                html.Span(f" (Score: {drift_score:.3f})", style={'marginLeft': '10px'})
            ])
        ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'})
    
    def _create_distribution_chart(self, ref_data: pd.DataFrame, curr_data: pd.DataFrame, feature: str):
        """Create distribution comparison chart"""
        fig = go.Figure()
        
        # Reference distribution
        fig.add_trace(go.Histogram(
            x=ref_data[feature],
            name='Reference',
            opacity=0.7,
            nbinsx=30,
            histnorm='probability density'
        ))
        
        # Current distribution
        fig.add_trace(go.Histogram(
            x=curr_data[feature],
            name='Current',
            opacity=0.7,
            nbinsx=30,
            histnorm='probability density'
        ))
        
        fig.update_layout(
            title=f'Distribution Comparison - {feature}',
            xaxis_title=feature,
            yaxis_title='Density',
            barmode='overlay'
        )
        
        return fig
    
    def _create_drift_timeline_chart(self, drift_history: pd.DataFrame, feature: str):
        """Create drift score timeline chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drift_history['timestamp'],
            y=drift_history['drift_score'],
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='blue')
        ))
        
        # Add threshold lines
        fig.add_hline(y=0.1, line_dash="dash", line_color="orange", 
                     annotation_text="Minor Drift Threshold")
        fig.add_hline(y=0.2, line_dash="dash", line_color="red", 
                     annotation_text="Major Drift Threshold")
        
        fig.update_layout(
            title=f'Drift Score Timeline - {feature}',
            xaxis_title='Time',
            yaxis_title='Drift Score'
        )
        
        return fig
    
    def _create_feature_stats_table(self, ref_data: pd.DataFrame, curr_data: pd.DataFrame, feature: str):
        """Create feature statistics table"""
        ref_stats = {
            'Mean': ref_data[feature].mean(),
            'Std': ref_data[feature].std(),
            'Min': ref_data[feature].min(),
            'Max': ref_data[feature].max(),
            'Missing %': ref_data[feature].isnull().mean() * 100
        }
        
        curr_stats = {
            'Mean': curr_data[feature].mean(),
            'Std': curr_data[feature].std(),
            'Min': curr_data[feature].min(),
            'Max': curr_data[feature].max(),
            'Missing %': curr_data[feature].isnull().mean() * 100
        }
        
        stats_df = pd.DataFrame({
            'Metric': list(ref_stats.keys()),
            'Reference': [f"{v:.2f}" for v in ref_stats.values()],
            'Current': [f"{v:.2f}" for v in curr_stats.values()],
            'Change %': [f"{((curr_stats[k] - ref_stats[k]) / ref_stats[k] * 100):.1f}%" 
                        if ref_stats[k] != 0 else "N/A" for k in ref_stats.keys()]
        })
        
        return html.Table([
            html.Thead([
                html.Tr([html.Th(col) for col in stats_df.columns])
            ]),
            html.Tbody([
                html.Tr([html.Td(stats_df.iloc[i][col]) for col in stats_df.columns])
                for i in range(len(stats_df))
            ])
        ], style={'width': '100%', 'textAlign': 'center'})
    
    def _create_alert_log(self, feature: str, drift_score: float):
        """Create alert log component"""
        # Sample alerts
        alerts = [
            {
                'timestamp': datetime.now() - timedelta(hours=2),
                'feature': feature,
                'severity': 'Medium' if drift_score > 0.1 else 'Low',
                'message': f'Drift detected in {feature} (score: {drift_score:.3f})'
            },
            {
                'timestamp': datetime.now() - timedelta(hours=6),
                'feature': 'other_feature',
                'severity': 'Low',
                'message': 'Minor distribution change detected'
            }
        ]
        
        alert_items = []
        for alert in alerts:
            color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}.get(alert['severity'], 'gray')
            
            alert_items.append(
                html.Div([
                    html.Span(alert['timestamp'].strftime('%Y-%m-%d %H:%M'), 
                             style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Span(f"[{alert['severity']}]", 
                             style={'color': color, 'fontWeight': 'bold', 'marginRight': '10px'}),
                    html.Span(alert['message'])
                ], style={'padding': '5px', 'borderBottom': '1px solid #eee'})
            )
        
        return html.Div(alert_items)
    
    def run(self, debug: bool = False):
        """Run the dashboard"""
        self.app.run_server(debug=debug, port=self.port)
```

---

## 🔧 Hands-On Exercise

You'll build a comprehensive feature monitoring and drift detection system for a production ML environment:

### Exercise Scenario
**Company**: E-commerce Recommendation Platform  
**Challenge**: Monitor feature drift in real-time recommendation system
- **Data Sources**: User behavior, product catalog, seasonal trends
- **Drift Detection**: Statistical and ML-based methods
- **Real-time Monitoring**: Kafka-based streaming pipeline
- **Automated Response**: Retraining triggers and alerting
- **Dashboard**: Interactive monitoring and visualization

### Requirements
1. **Statistical Drift Detection**: PSI and KS test implementations
2. **ML-based Detection**: Domain classifier and anomaly detection
3. **Real-time Monitoring**: Kafka streaming with sliding windows
4. **Automated Retraining**: Trigger-based pipeline orchestration
5. **Interactive Dashboard**: Plotly/Dash visualization system

---

## 📚 Key Takeaways

- **Monitor continuously** - drift detection should be part of your production ML pipeline
- **Use multiple methods** - combine statistical tests with ML-based approaches for robust detection
- **Set appropriate thresholds** - balance sensitivity with false positive rates
- **Automate responses** - implement trigger-based retraining and alerting systems
- **Visualize trends** - dashboards help identify patterns and root causes
- **Consider business impact** - not all drift requires immediate action
- **Test your detectors** - validate drift detection methods on historical data
- **Document decisions** - maintain clear records of drift events and responses

---

## 🔄 What's Next?

Tomorrow, we'll explore **AutoML** where you'll learn how to:
- Automate feature engineering and selection
- Implement automated model selection and hyperparameter tuning
- Build end-to-end AutoML pipelines
- Compare AutoML frameworks and tools