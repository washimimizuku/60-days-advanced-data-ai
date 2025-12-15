# Day 28: Anomaly Detection - Statistical & ML-based Methods

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Implement** statistical anomaly detection methods (Z-score, IQR, Isolation Forest)
- **Build** machine learning-based anomaly detection systems using autoencoders and clustering
- **Design** real-time anomaly detection pipelines for production environments
- **Apply** ensemble methods for robust anomaly detection across different data types
- **Evaluate** anomaly detection performance using appropriate metrics and validation techniques

⭐ **Difficulty Level**: Advanced  
🕒 **Estimated Time**: 60 minutes  
🛠️ **Prerequisites**: Machine learning fundamentals, time series analysis, feature engineering

---

## 🎯 What is Anomaly Detection?

Anomaly detection is the identification of rare items, events, or observations that raise suspicions by differing significantly from the majority of the data. In production systems, anomaly detection serves as an early warning system for:

- **Fraud Detection**: Identifying suspicious transactions or user behavior
- **System Monitoring**: Detecting infrastructure failures or performance degradation  
- **Quality Control**: Finding defective products or process deviations
- **Security**: Identifying cyber attacks or unauthorized access
- **Business Intelligence**: Discovering unusual patterns in customer behavior or market trends

### Types of Anomalies

**Point Anomalies**: Individual data points that are anomalous with respect to the rest of the data
- Example: A single fraudulent credit card transaction

**Contextual Anomalies**: Data points that are anomalous in a specific context but not otherwise
- Example: High temperature in winter vs. summer

**Collective Anomalies**: A collection of related data instances that are anomalous
- Example: A sequence of failed login attempts indicating a brute force attack

---

## 🔬 Statistical Anomaly Detection Methods

### 1. Z-Score Method

The Z-score measures how many standard deviations a data point is from the mean. Points with |Z-score| > threshold (typically 2-3) are considered anomalies.

```python
import numpy as np
from scipy import stats

def zscore_anomaly_detection(data, threshold=3):
    """
    Detect anomalies using Z-score method
    
    Args:
        data: Input data array
        threshold: Z-score threshold for anomaly detection
    
    Returns:
        Boolean array indicating anomalies
    """
    z_scores = np.abs(stats.zscore(data))
    return z_scores > threshold

# Example usage
data = np.random.normal(100, 15, 1000)
data[50] = 200  # Inject anomaly
anomalies = zscore_anomaly_detection(data)
print(f"Detected {np.sum(anomalies)} anomalies")
```

**Advantages**: Simple, fast, interpretable  
**Limitations**: Assumes normal distribution, sensitive to outliers in mean/std calculation

### 2. Interquartile Range (IQR) Method

IQR method identifies outliers as points that fall below Q1 - 1.5×IQR or above Q3 + 1.5×IQR.

```python
def iqr_anomaly_detection(data, factor=1.5):
    """
    Detect anomalies using IQR method
    
    Args:
        data: Input data array
        factor: IQR multiplication factor
    
    Returns:
        Boolean array indicating anomalies
    """
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return (data < lower_bound) | (data > upper_bound)
```

**Advantages**: Robust to non-normal distributions, less sensitive to extreme outliers  
**Limitations**: May not work well with skewed distributions

### 3. Modified Z-Score (Median Absolute Deviation)

Uses median and MAD instead of mean and standard deviation, making it more robust to outliers.

```python
def modified_zscore_anomaly_detection(data, threshold=3.5):
    """
    Detect anomalies using Modified Z-score with MAD
    
    Args:
        data: Input data array
        threshold: Modified Z-score threshold
    
    Returns:
        Boolean array indicating anomalies
    """
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    modified_z_scores = 0.6745 * (data - median) / mad
    return np.abs(modified_z_scores) > threshold
```

---

## 🤖 Machine Learning-Based Methods

### 1. Isolation Forest

Isolation Forest isolates anomalies by randomly selecting features and split values. Anomalies are easier to isolate and require fewer splits.

```python
from sklearn.ensemble import IsolationForest

# Initialize and fit Isolation Forest
iso_forest = IsolationForest(
    contamination=0.1,  # Expected proportion of anomalies
    random_state=42,
    n_estimators=100
)

# Fit and predict (-1 for anomalies, 1 for normal)
predictions = iso_forest.fit_predict(data.reshape(-1, 1))
anomalies = predictions == -1
```

**Advantages**: No assumptions about data distribution, handles high-dimensional data well  
**Limitations**: Performance depends on contamination parameter, may struggle with very high dimensions

### 2. One-Class SVM

One-Class SVM learns a decision function for novelty detection, mapping data to a high-dimensional space.

```python
from sklearn.svm import OneClassSVM

# Initialize One-Class SVM
oc_svm = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.1  # Upper bound on fraction of training errors
)

# Fit and predict
predictions = oc_svm.fit_predict(data.reshape(-1, 1))
anomalies = predictions == -1
```

### 3. Autoencoders for Anomaly Detection

Neural network autoencoders learn to compress and reconstruct normal data. High reconstruction error indicates anomalies.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

class AnomalyAutoencoder:
    def __init__(self, input_dim, encoding_dim=32):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.model = self._build_model()
        
    def _build_model(self):
        # Encoder
        input_layer = layers.Input(shape=(self.input_dim,))
        encoded = layers.Dense(64, activation='relu')(input_layer)
        encoded = layers.Dense(self.encoding_dim, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(64, activation='relu')(encoded)
        decoded = layers.Dense(self.input_dim, activation='sigmoid')(decoded)
        
        # Autoencoder model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def fit(self, normal_data, epochs=100, validation_split=0.2):
        """Train autoencoder on normal data only"""
        return self.model.fit(
            normal_data, normal_data,
            epochs=epochs,
            validation_split=validation_split,
            verbose=0
        )
    
    def detect_anomalies(self, data, threshold_percentile=95):
        """Detect anomalies based on reconstruction error"""
        reconstructed = self.model.predict(data)
        mse = np.mean(np.square(data - reconstructed), axis=1)
        threshold = np.percentile(mse, threshold_percentile)
        return mse > threshold, mse
```

---

## 📊 Time Series Anomaly Detection

### 1. Statistical Process Control (SPC)

Uses control charts to monitor process stability and detect when a process goes out of statistical control.

```python
def spc_anomaly_detection(data, window_size=30, n_sigma=3):
    """
    Statistical Process Control anomaly detection
    
    Args:
        data: Time series data
        window_size: Rolling window for calculating control limits
        n_sigma: Number of standard deviations for control limits
    
    Returns:
        Boolean array indicating anomalies
    """
    rolling_mean = pd.Series(data).rolling(window=window_size).mean()
    rolling_std = pd.Series(data).rolling(window=window_size).std()
    
    upper_limit = rolling_mean + n_sigma * rolling_std
    lower_limit = rolling_mean - n_sigma * rolling_std
    
    anomalies = (data > upper_limit) | (data < lower_limit)
    return anomalies.fillna(False).values
```

### 2. Seasonal Decomposition-based Anomaly Detection

Decomposes time series into trend, seasonal, and residual components, then detects anomalies in residuals.

```python
from statsmodels.tsa.seasonal import seasonal_decompose

def seasonal_anomaly_detection(data, period=24, method='multiplicative'):
    """
    Seasonal decomposition-based anomaly detection
    
    Args:
        data: Time series data with datetime index
        period: Seasonal period
        method: 'additive' or 'multiplicative'
    
    Returns:
        Boolean array indicating anomalies
    """
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(data, model=method, period=period)
    
    # Detect anomalies in residuals using IQR method
    residuals = decomposition.resid.dropna()
    anomalies = iqr_anomaly_detection(residuals.values)
    
    # Map back to original data index
    result = np.zeros(len(data), dtype=bool)
    result[residuals.index] = anomalies
    
    return result
```

---

## 🎯 Ensemble Anomaly Detection

Combining multiple anomaly detection methods often provides more robust results than any single method.

### Voting-based Ensemble

```python
class EnsembleAnomalyDetector:
    def __init__(self, methods=None, voting_threshold=0.5):
        self.methods = methods or []
        self.voting_threshold = voting_threshold
        
    def add_method(self, method, weight=1.0):
        """Add anomaly detection method to ensemble"""
        self.methods.append({'method': method, 'weight': weight})
        
    def detect_anomalies(self, data):
        """Detect anomalies using ensemble voting"""
        votes = np.zeros(len(data))
        total_weight = sum(m['weight'] for m in self.methods)
        
        for method_info in self.methods:
            method = method_info['method']
            weight = method_info['weight']
            
            # Get anomaly predictions from method
            anomalies = method(data)
            votes += anomalies.astype(float) * weight
        
        # Normalize votes and apply threshold
        normalized_votes = votes / total_weight
        return normalized_votes > self.voting_threshold, normalized_votes
```

---

## 📈 Evaluation Metrics for Anomaly Detection

### 1. Classification Metrics

When ground truth labels are available:

```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

def evaluate_anomaly_detection(y_true, y_pred, y_scores=None):
    """
    Evaluate anomaly detection performance
    
    Args:
        y_true: True binary labels (1 for anomaly, 0 for normal)
        y_pred: Predicted binary labels
        y_scores: Anomaly scores (optional)
    
    Returns:
        Dictionary of evaluation metrics
    """
    metrics = {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred)
    }
    
    if y_scores is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_scores)
        
    return metrics
```

### 2. Unsupervised Evaluation

When no ground truth is available:

- **Silhouette Score**: Measures how similar anomalies are to their own cluster vs. other clusters
- **Calinski-Harabasz Index**: Ratio of between-cluster to within-cluster variance
- **Davies-Bouldin Index**: Average similarity between clusters

---

## 🏭 Production Considerations

### 1. Real-time Processing

```python
class StreamingAnomalyDetector:
    def __init__(self, window_size=1000, update_frequency=100):
        self.window_size = window_size
        self.update_frequency = update_frequency
        self.data_buffer = []
        self.model = None
        self.update_counter = 0
        
    def process_point(self, data_point):
        """Process single data point in streaming fashion"""
        self.data_buffer.append(data_point)
        
        # Maintain sliding window
        if len(self.data_buffer) > self.window_size:
            self.data_buffer.pop(0)
            
        # Update model periodically
        self.update_counter += 1
        if self.update_counter >= self.update_frequency:
            self._update_model()
            self.update_counter = 0
            
        # Detect anomaly
        return self._detect_anomaly(data_point)
        
    def _update_model(self):
        """Update anomaly detection model with recent data"""
        if len(self.data_buffer) >= self.window_size:
            # Retrain model on recent normal data
            # Implementation depends on chosen method
            pass
            
    def _detect_anomaly(self, data_point):
        """Detect if current point is anomalous"""
        # Implementation depends on chosen method
        pass
```

### 2. Handling Concept Drift

Anomaly patterns can change over time, requiring adaptive models:

```python
class AdaptiveAnomalyDetector:
    def __init__(self, base_detector, adaptation_rate=0.01):
        self.base_detector = base_detector
        self.adaptation_rate = adaptation_rate
        self.performance_history = []
        
    def adapt_threshold(self, recent_performance):
        """Adapt detection threshold based on recent performance"""
        if len(self.performance_history) > 10:
            # Calculate performance trend
            recent_avg = np.mean(self.performance_history[-5:])
            historical_avg = np.mean(self.performance_history[:-5])
            
            # Adjust threshold if performance is declining
            if recent_avg < historical_avg * 0.9:
                self.base_detector.threshold *= (1 + self.adaptation_rate)
                
        self.performance_history.append(recent_performance)
```

---

## 🎯 Industry Applications

### 1. Financial Fraud Detection

- **Credit Card Transactions**: Detect unusual spending patterns, location anomalies
- **Insurance Claims**: Identify potentially fraudulent claims
- **Trading**: Detect market manipulation or insider trading

### 2. Cybersecurity

- **Network Intrusion**: Identify unusual network traffic patterns
- **User Behavior**: Detect compromised accounts or insider threats
- **Malware Detection**: Identify suspicious file or process behavior

### 3. Industrial IoT

- **Predictive Maintenance**: Detect equipment failures before they occur
- **Quality Control**: Identify defective products in manufacturing
- **Energy Management**: Detect unusual consumption patterns

### 4. Healthcare

- **Medical Diagnosis**: Identify unusual patient symptoms or test results
- **Drug Discovery**: Detect unusual molecular properties
- **Epidemic Detection**: Identify disease outbreak patterns

---

## 🔧 Best Practices

### 1. Data Preprocessing

- **Handle Missing Values**: Use appropriate imputation strategies
- **Normalize Features**: Ensure all features are on similar scales
- **Feature Engineering**: Create domain-specific features that capture anomalous behavior

### 2. Model Selection

- **Understand Your Data**: Choose methods appropriate for your data distribution
- **Consider Interpretability**: Balance accuracy with explainability requirements
- **Validate Thoroughly**: Use appropriate cross-validation for time series data

### 3. Threshold Selection

- **Business Context**: Set thresholds based on business impact of false positives/negatives
- **Adaptive Thresholds**: Allow thresholds to adapt to changing conditions
- **Multiple Thresholds**: Use different severity levels (warning, critical, emergency)

### 4. Monitoring and Maintenance

- **Performance Tracking**: Monitor detection accuracy over time
- **Feedback Loops**: Incorporate human feedback to improve model performance
- **Regular Retraining**: Update models with new data and patterns

---

## 🚀 Advanced Techniques

### 1. Deep Learning Approaches

- **Variational Autoencoders (VAE)**: Probabilistic approach to anomaly detection
- **Generative Adversarial Networks (GANs)**: Use generator-discriminator framework
- **LSTM Autoencoders**: For sequential/time series anomaly detection

### 2. Graph-based Anomaly Detection

- **Community Detection**: Identify anomalous nodes in networks
- **Graph Neural Networks**: Learn representations for graph anomaly detection

### 3. Multi-modal Anomaly Detection

- **Sensor Fusion**: Combine multiple data sources for robust detection
- **Cross-modal Learning**: Detect anomalies across different data modalities

---

## 🏗️ Infrastructure Setup

### Quick Start (5 minutes)

```bash
# 1. Navigate to day 28
cd days/day-28-anomaly-detection

# 2. Start the complete infrastructure
./setup.sh

# 3. Run interactive demo
python demo.py
```

### Infrastructure Components

**Anomaly Detection Stack**:
- **PostgreSQL**: Metadata and anomaly storage
- **InfluxDB**: High-performance time series anomaly storage
- **Redis**: Real-time caching and statistics
- **Kafka**: Streaming data and anomaly alerts
- **MLflow**: Model tracking and registry

**Detection Services**:
- **FastAPI Server**: Production anomaly detection endpoints
- **Stream Detector**: Real-time Kafka-based anomaly detection
- **Data Generator**: Realistic datasets with controlled anomaly injection
- **Jupyter Notebook**: Interactive analysis environment

**Monitoring & Alerting**:
- **Prometheus**: Metrics collection and anomaly rate monitoring
- **Alertmanager**: Anomaly alert routing and notification
- **Grafana**: Anomaly visualization and dashboards
- **Health Checks**: Service availability monitoring

### Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Anomaly Detection API | http://localhost:8000 | - |
| Grafana Dashboard | http://localhost:3000 | admin/anomaly123 |
| Jupyter Notebook | http://localhost:8888 | token: anomaly123 |
| Prometheus | http://localhost:9090 | - |
| Alertmanager | http://localhost:9093 | - |
| MLflow Tracking | http://localhost:5000 | - |

### API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# List available datasets
curl http://localhost:8000/datasets

# Detect anomalies in data
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"data": [{"amount": 1000, "hour": 14}], "method": "isolation_forest"}'

# Batch detection on stored dataset
curl -X POST http://localhost:8000/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"dataset_name": "financial_transactions", "method": "ensemble"}'

# Get dataset statistics
curl http://localhost:8000/datasets/financial_transactions/stats
```

### Generated Datasets

The infrastructure provides realistic datasets with controlled anomaly injection:

- **Financial Transactions**: Credit card fraud with amount, time, and location anomalies
- **Network Traffic**: DDoS attacks, port scans, and data exfiltration patterns
- **IoT Sensor Data**: Equipment failures with temperature, pressure, and vibration anomalies
- **User Behavior**: Bot activity, brute force attacks, and suspicious access patterns

---

## 📚 Key Takeaways

- **Statistical methods** (Z-score, IQR, MAD) are simple and interpretable but make distributional assumptions
- **Machine learning methods** (Isolation Forest, One-Class SVM, Autoencoders) are more flexible but less interpretable
- **Ensemble approaches** combine multiple methods for improved robustness and accuracy
- **Time series anomaly detection** requires specialized techniques that account for temporal dependencies
- **Production systems** need real-time processing, concept drift handling, and adaptive thresholds
- **Evaluation** should consider both statistical metrics and business impact
- **Domain expertise** is crucial for feature engineering and threshold setting
- **Continuous monitoring** and model updates are essential for maintaining performance
- **Real-time streaming** enables immediate anomaly detection and alerting
- **Comprehensive infrastructure** supports production-grade anomaly detection systems

---

## 🔄 What's Next?

Tomorrow, we'll explore **Recommendation Systems** where you'll learn to build collaborative filtering and content-based recommendation engines. We'll cover matrix factorization, deep learning approaches, and how to handle the cold start problem in production recommendation systems.

The anomaly detection techniques you've learned today will be valuable for detecting unusual user behavior and improving recommendation quality by filtering out anomalous interactions.