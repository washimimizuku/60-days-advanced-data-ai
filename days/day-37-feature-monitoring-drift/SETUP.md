# Day 37: Feature Monitoring & Drift - Setup Guide

## Overview
This lesson covers comprehensive feature monitoring and drift detection for production ML systems, including statistical methods, ML-based detection, real-time monitoring, and automated retraining pipelines.

## Prerequisites
- Completed Days 1-36 (especially Days 35-36 on MLOps)
- Python 3.11+
- Docker and Docker Compose
- 8GB+ RAM (for ML models and Kafka)
- Basic understanding of statistical testing

## Quick Start (5 minutes)

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 2. Run Basic Exercise
```bash
# Run the main exercise
python exercise.py

# Run the complete solution
python solution.py
```

### 3. Run Tests
```bash
# Run test suite
pytest test_feature_monitoring.py -v

# Run with coverage
pytest test_feature_monitoring.py --cov=solution --cov-report=html
```

## Full Setup (15 minutes)

### 1. Environment Setup
```bash
# Clone repository (if not already done)
git clone <repository-url>
cd days/day-37-feature-monitoring-drift

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Infrastructure Setup
```bash
# Start infrastructure services (Kafka, MLflow)
docker-compose up -d

# Wait for services to be ready (30-60 seconds)
docker-compose logs -f

# Verify services are running
curl http://localhost:5000  # MLflow
```

### 3. Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration (optional)
nano retraining_config.yaml
```

### 4. Run Complete System
```bash
# Run feature monitoring system
python solution.py

# In another terminal, run dashboard (if implemented)
python dashboard_app.py

# Access dashboard at http://localhost:8050
```

## Project Structure
```
day-37-feature-monitoring-drift/
├── README.md                    # Lesson content
├── exercise.py                  # Exercise with TODOs
├── solution.py                  # Complete solution
├── quiz.md                      # Knowledge check
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── docker-compose.yml           # Infrastructure services
├── retraining_config.yaml       # Retraining configuration
├── test_feature_monitoring.py   # Test suite
├── SETUP.md                     # This file
├── .env.example                 # Environment template
└── data/                        # Generated data files
    ├── reference_data.csv
    ├── drifted_data.csv
    └── drift_history.csv
```

## Key Components

### 1. Statistical Drift Detection
- **Population Stability Index (PSI)**: Quantile-based distribution comparison
- **Kolmogorov-Smirnov Test**: Statistical significance testing
- **Chi-square Test**: Categorical feature drift detection

### 2. ML-based Drift Detection
- **Domain Classifier**: Binary classification approach
- **Isolation Forest**: Anomaly detection for drift
- **Feature-wise Analysis**: Individual feature drift scoring

### 3. Real-time Monitoring
- **Sliding Windows**: Continuous data processing
- **Kafka Integration**: Streaming data ingestion
- **Alert System**: Configurable thresholds and notifications

### 4. Automated Retraining
- **Trigger Evaluation**: Data drift, performance, time-based
- **Strategy Selection**: Full retrain, incremental, feature selection
- **MLflow Integration**: Experiment tracking and model registry

## Configuration Options

### Drift Detection Thresholds
```yaml
# PSI thresholds
psi_thresholds:
  minor: 0.1
  major: 0.2
  severe: 0.5

# KS test significance
ks_significance: 0.05

# ML-based contamination
ml_contamination: 0.1
```

### Retraining Triggers
```yaml
triggers:
  data_drift:
    threshold: 0.2
    critical_features: ["user_age", "cart_value"]
  
  performance_degradation:
    ctr: 0.03
    conversion_rate: 0.02
  
  time_based:
    max_age_days: 7
```

## Running Scenarios

### Scenario 1: Basic Drift Detection
```bash
# Run PSI and KS tests
python -c "
from solution import *
data = create_sample_ecommerce_data()
drifted = create_drifted_data(data, 'gradual')

psi = PopulationStabilityIndex()
psi.fit(data, ['user_age', 'cart_value'])
scores = psi.calculate_psi(drifted)
print('PSI Scores:', scores)
"
```

### Scenario 2: Real-time Monitoring
```bash
# Start Kafka services
docker-compose up -d kafka zookeeper

# Run real-time monitor
python -c "
from solution import RealTimeDriftMonitor
monitor = RealTimeDriftMonitor(window_size=100)
# Configure and run monitoring...
"
```

### Scenario 3: Automated Retraining
```bash
# Test retraining pipeline
python -c "
from solution import AutomatedRetrainingPipeline
import yaml

with open('retraining_config.yaml') as f:
    config = yaml.safe_load(f)

pipeline = AutomatedRetrainingPipeline(config)
# Test triggers and strategies...
"
```

## Troubleshooting

### Common Issues

1. **Kafka Connection Errors**
   ```bash
   # Check Kafka status
   docker-compose ps
   
   # Restart Kafka services
   docker-compose restart kafka zookeeper
   ```

2. **Memory Issues**
   ```bash
   # Reduce data size for testing
   export TEST_DATA_SIZE=1000
   
   # Use smaller ML models
   export ML_MODEL_SIZE=small
   ```

3. **MLflow Connection Issues**
   ```bash
   # Check MLflow status
   curl http://localhost:5000
   
   # Restart MLflow
   docker-compose restart mlflow
   ```

4. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   
   # Check Python path
   export PYTHONPATH=$PWD:$PYTHONPATH
   ```

### Performance Optimization

1. **Reduce Data Size for Testing**
   ```python
   # In solution.py, modify data generation
   n_samples = 1000  # Instead of 10000
   ```

2. **Use Smaller Windows**
   ```python
   # Reduce window size for faster testing
   window_size = 100  # Instead of 1000
   ```

3. **Disable Heavy Components**
   ```python
   # Skip ML-based detection for faster runs
   skip_ml_detection = True
   ```

## Validation

### Test Data Quality
```bash
# Verify data generation
python -c "
from solution import create_sample_ecommerce_data
data = create_sample_ecommerce_data()
print(f'Data shape: {data.shape}')
print(f'Columns: {list(data.columns)}')
print(data.describe())
"
```

### Test Drift Detection
```bash
# Verify drift detection works
python -c "
from solution import *
ref_data = create_sample_ecommerce_data()
drift_data = create_drifted_data(ref_data, 'gradual')

psi = PopulationStabilityIndex()
psi.fit(ref_data, ['user_age'])
score = psi.calculate_psi(drift_data)['user_age']
print(f'PSI Score: {score:.3f}')
assert score > 0.1, 'Drift should be detected'
print('✓ Drift detection working')
"
```

### Test Infrastructure
```bash
# Test Kafka connectivity
python -c "
from kafka import KafkaProducer, KafkaConsumer
producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
producer.send('test-topic', b'test-message')
print('✓ Kafka working')
"

# Test MLflow connectivity
python -c "
import mlflow
mlflow.set_tracking_uri('http://localhost:5000')
print('✓ MLflow working')
"
```

## Next Steps

1. **Complete the Exercise**: Work through `exercise.py` step by step
2. **Run the Solution**: Execute `solution.py` to see complete implementation
3. **Take the Quiz**: Test your understanding with `quiz.md`
4. **Experiment**: Try different drift types and detection methods
5. **Extend**: Add custom drift detection methods or dashboard features

## Additional Resources

- [Evidently AI Documentation](https://docs.evidentlyai.com/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Kafka Python Client](https://kafka-python.readthedocs.io/)
- [Statistical Drift Detection Papers](https://arxiv.org/search/?query=drift+detection)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the test suite for examples
3. Examine the solution code for reference implementations
4. Ensure all dependencies are properly installed