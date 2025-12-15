# Day 38: AutoML - Setup Guide

## Overview
This lesson covers comprehensive AutoML (Automated Machine Learning) implementation including automated feature engineering, model selection, hyperparameter optimization, and ensemble methods for production-ready ML systems.

## Prerequisites
- Completed Days 1-37 (especially Days 35-37 on MLOps)
- Python 3.11+
- Java 11+ (for H2O.ai)
- 8GB+ RAM (for AutoML frameworks)
- Basic understanding of machine learning concepts

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
pytest test_automl.py -v

# Run with coverage
pytest test_automl.py --cov=solution --cov-report=html
```

## Full Setup (15 minutes)

### 1. Environment Setup
```bash
# Clone repository (if not already done)
git clone <repository-url>
cd days/day-38-automl

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

### 2. Java Setup (for H2O.ai)
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install openjdk-11-jdk

# macOS
brew install openjdk@11

# Windows
# Download and install Java 11 from Oracle or OpenJDK

# Set JAVA_HOME
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  # Linux
export JAVA_HOME=/usr/local/opt/openjdk@11  # macOS
```

### 3. Verify Installation
```bash
# Test H2O installation
python -c "import h2o; h2o.init(); print('H2O working')"

# Test AutoGluon installation
python -c "from autogluon.tabular import TabularPredictor; print('AutoGluon working')"

# Test Optuna installation
python -c "import optuna; print('Optuna working')"
```

### 4. Configuration
```bash
# Copy configuration template
cp automl_config.yaml my_automl_config.yaml

# Edit configuration (optional)
nano my_automl_config.yaml
```

## Project Structure
```
day-38-automl/
├── README.md                    # Lesson content
├── exercise.py                  # Exercise with TODOs
├── solution.py                  # Complete solution
├── quiz.md                      # Knowledge check
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── automl_config.yaml           # AutoML configuration
├── test_automl.py              # Test suite
├── SETUP.md                     # This file
└── models/                      # Generated models
    ├── automl_pipeline.pkl
    └── model_reports/
```

## Key Components

### 1. Automated Feature Engineering
- **Medical Domain Features**: BMI categories, age groups, BP classifications
- **Statistical Features**: Polynomial interactions, aggregations
- **Feature Selection**: Univariate and mutual information methods
- **Preprocessing**: Missing value handling, outlier detection

### 2. Automated Model Selection
- **Algorithms**: Random Forest, Gradient Boosting, XGBoost, LightGBM
- **Hyperparameter Optimization**: Optuna with TPE sampler
- **Cross-Validation**: Stratified K-fold for robust evaluation
- **Time Management**: Budget allocation across models

### 3. Ensemble Methods
- **Voting Classifiers**: Soft voting with probability averaging
- **Model Diversity**: Selection based on performance and diversity
- **Ensemble Validation**: Cross-validation for ensemble performance

### 4. AutoML Frameworks
- **H2O.ai**: Enterprise-grade AutoML with leaderboard
- **AutoGluon**: Easy-to-use tabular AutoML
- **Custom Pipeline**: Full control with modular components

## Configuration Options

### Feature Engineering Settings
```yaml
feature_engineering:
  max_features: 1000
  feature_selection_ratio: 0.8
  enable_interactions: true
  polynomial_degree: 2
```

### Model Selection Settings
```yaml
model_selection:
  algorithms:
    - random_forest
    - gradient_boosting
    - xgboost
  hyperparameter_optimization:
    n_trials: 100
    timeout_per_model: 600
```

### Ensemble Configuration
```yaml
ensemble:
  enable: true
  max_ensemble_size: 5
  ensemble_method: "voting"
  voting_type: "soft"
```

## Running Scenarios

### Scenario 1: Basic AutoML Pipeline
```bash
# Run automated feature engineering
python -c "
from solution import AutomatedFeatureEngineer, create_medical_diagnosis_dataset
import pandas as pd

data = create_medical_diagnosis_dataset()
X = data.drop('diagnosis', axis=1)
y = data['diagnosis']

fe = AutomatedFeatureEngineer()
X_transformed = fe.fit_transform(X, y)
print(f'Features: {X.shape[1]} -> {X_transformed.shape[1]}')
"
```

### Scenario 2: Model Optimization
```bash
# Run hyperparameter optimization
python -c "
from solution import AutomatedModelSelector, create_medical_diagnosis_dataset

data = create_medical_diagnosis_dataset()
X = data.drop('diagnosis', axis=1).head(1000)  # Subset for speed
y = data['diagnosis'].head(1000)

selector = AutomatedModelSelector(n_trials=10, time_budget=300)
results = selector.optimize_models(X, y)
print(f'Best model: {results[\"best_overall\"][\"model_name\"]}')
"
```

### Scenario 3: Complete AutoML
```bash
# Run complete AutoML pipeline
python -c "
from solution import ComprehensiveAutoMLPipeline, create_medical_diagnosis_dataset

data = create_medical_diagnosis_dataset()
X = data.drop('diagnosis', axis=1).head(500)  # Subset for demo
y = data['diagnosis'].head(500)

pipeline = ComprehensiveAutoMLPipeline(time_budget=300)
results = pipeline.fit(X, y)
print(f'Final score: {results[\"final_score\"]:.4f}')
"
```

## Troubleshooting

### Common Issues

1. **H2O Installation Problems**
   ```bash
   # Check Java installation
   java -version
   
   # Reinstall H2O
   pip uninstall h2o
   pip install h2o
   
   # Set JAVA_HOME explicitly
   export JAVA_HOME=/path/to/java
   ```

2. **AutoGluon Installation Issues**
   ```bash
   # Install with specific version
   pip install autogluon.tabular==0.8.0
   
   # For Apple Silicon Macs
   pip install autogluon.tabular --no-deps
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory Issues**
   ```bash
   # Reduce dataset size
   export AUTOML_SAMPLE_SIZE=1000
   
   # Reduce time budget
   export AUTOML_TIME_BUDGET=300
   
   # Disable ensemble
   export AUTOML_ENABLE_ENSEMBLE=false
   ```

4. **Optuna Optimization Errors**
   ```bash
   # Clear Optuna cache
   rm -rf ~/.optuna/
   
   # Reduce trial count
   export OPTUNA_N_TRIALS=10
   ```

### Performance Optimization

1. **Speed Up Training**
   ```python
   # Reduce dataset size
   data_sample = data.sample(n=1000, random_state=42)
   
   # Reduce time budget
   pipeline = ComprehensiveAutoMLPipeline(time_budget=300)
   
   # Disable expensive features
   fe = AutomatedFeatureEngineer(enable_interactions=False)
   ```

2. **Memory Optimization**
   ```python
   # Limit feature count
   fe = AutomatedFeatureEngineer(max_features=100)
   
   # Use smaller ensemble
   ensemble = EnsembleModelBuilder(ensemble_size=3)
   ```

3. **Parallel Processing**
   ```python
   # Set number of jobs
   import os
   os.environ['OMP_NUM_THREADS'] = '4'
   
   # Configure in YAML
   general:
     n_jobs: 4
   ```

## Validation

### Test AutoML Components
```bash
# Test feature engineering
python -c "
from solution import AutomatedFeatureEngineer
import pandas as pd
import numpy as np

X = pd.DataFrame(np.random.randn(100, 5))
y = np.random.choice([0, 1], 100)

fe = AutomatedFeatureEngineer()
X_transformed = fe.fit_transform(X, y)
print(f'✓ Feature engineering: {X.shape[1]} -> {X_transformed.shape[1]}')
"
```

### Test Model Selection
```bash
# Test hyperparameter optimization
python -c "
from solution import AutomatedModelSelector
import pandas as pd
import numpy as np

X = pd.DataFrame(np.random.randn(200, 10))
y = np.random.choice([0, 1], 200)

selector = AutomatedModelSelector(n_trials=5, time_budget=60)
results = selector.optimize_models(X, y)
print(f'✓ Model selection: {len(results)-1} models optimized')
"
```

### Test Complete Pipeline
```bash
# Test end-to-end pipeline
python -c "
from solution import ComprehensiveAutoMLPipeline, create_medical_diagnosis_dataset

data = create_medical_diagnosis_dataset()
X = data.drop('diagnosis', axis=1).head(200)
y = data['diagnosis'].head(200)

pipeline = ComprehensiveAutoMLPipeline(time_budget=120)
results = pipeline.fit(X, y)
predictions = pipeline.predict(X.head(10))
print(f'✓ Complete pipeline: score={results[\"final_score\"]:.3f}')
"
```

## Docker Usage

### Build and Run Container
```bash
# Build Docker image
docker build -t automl-day38 .

# Run container
docker run -it --rm \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  automl-day38

# Run with custom configuration
docker run -it --rm \
  -v $(pwd)/my_automl_config.yaml:/app/automl_config.yaml \
  -v $(pwd)/models:/app/models \
  automl-day38
```

## Next Steps

1. **Complete the Exercise**: Work through `exercise.py` step by step
2. **Run the Solution**: Execute `solution.py` to see complete implementation
3. **Take the Quiz**: Test your understanding with `quiz.md`
4. **Experiment**: Try different AutoML configurations and datasets
5. **Extend**: Add custom feature engineering or model types

## Additional Resources

- [H2O.ai Documentation](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html)
- [AutoGluon Documentation](https://auto.gluon.ai/stable/tutorials/tabular_prediction/index.html)
- [Optuna Documentation](https://optuna.readthedocs.io/)
- [AutoML Survey Paper](https://arxiv.org/abs/1908.00709)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the test suite for examples
3. Examine the solution code for reference implementations
4. Ensure all dependencies are properly installed
5. Verify Java installation for H2O.ai