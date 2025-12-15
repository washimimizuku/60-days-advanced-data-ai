# Day 35: Model Versioning with DVC - MLflow, Model Registry

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Implement version control** for ML models, datasets, and experiments using DVC
- **Build reproducible ML pipelines** with data lineage and dependency tracking
- **Create model registries** with MLflow for production model management
- **Handle model rollbacks** and deployment strategies safely
- **Automate artifact management** with CI/CD integration for ML workflows

**Estimated Time**: 1 hour  
**Difficulty Level**: Advanced ⭐⭐⭐⭐

---

## 🎯 What is Model Versioning?

Model versioning is the practice of systematically tracking, storing, and managing different versions of machine learning models, datasets, and experiments throughout the ML lifecycle. It ensures reproducibility, enables collaboration, and provides the foundation for safe production deployments.

### Key Components of ML Versioning

**1. Data Versioning**
- Track changes in training and validation datasets
- Maintain data lineage and provenance
- Handle large files efficiently with content-based addressing

**2. Model Versioning**
- Version trained models with metadata and performance metrics
- Track model architecture, hyperparameters, and training configuration
- Enable model comparison and rollback capabilities

**3. Experiment Versioning**
- Record experiment configurations and results
- Track hyperparameter tuning and model selection
- Maintain reproducible experiment environments

**4. Pipeline Versioning**
- Version entire ML workflows and dependencies
- Track code, data, and model interactions
- Enable pipeline reproduction and debugging

---

## 🔧 DVC (Data Version Control) Fundamentals

### 1. **DVC Architecture and Concepts**

```bash
# Initialize DVC in your project
dvc init

# Add data files to DVC tracking
dvc add data/raw/dataset.csv
dvc add models/trained_model.pkl

# Create DVC pipeline stages
dvc stage add -n prepare \
    -d data/raw/dataset.csv \
    -o data/processed/clean_data.csv \
    python src/prepare_data.py

dvc stage add -n train \
    -d data/processed/clean_data.csv \
    -d src/train_model.py \
    -o models/model.pkl \
    -M metrics/train_metrics.json \
    python src/train_model.py

# Run the pipeline
dvc repro

# Track changes with Git
git add dvc.yaml dvc.lock .gitignore
git commit -m "Add ML pipeline"
```

### 2. **DVC Pipeline Configuration**

```yaml
# dvc.yaml - Pipeline definition
stages:
  prepare_data:
    cmd: python src/prepare_data.py
    deps:
    - data/raw/dataset.csv
    - src/prepare_data.py
    outs:
    - data/processed/clean_data.csv
    params:
    - prepare.test_size
    - prepare.random_state
    
  feature_engineering:
    cmd: python src/feature_engineering.py
    deps:
    - data/processed/clean_data.csv
    - src/feature_engineering.py
    outs:
    - data/features/features.csv
    params:
    - features.n_components
    - features.scaler_type
    
  train_model:
    cmd: python src/train_model.py
    deps:
    - data/features/features.csv
    - src/train_model.py
    outs:
    - models/model.pkl
    metrics:
    - metrics/train_metrics.json
    params:
    - train.algorithm
    - train.hyperparameters
    
  evaluate_model:
    cmd: python src/evaluate_model.py
    deps:
    - models/model.pkl
    - data/features/features.csv
    - src/evaluate_model.py
    metrics:
    - metrics/eval_metrics.json
    plots:
    - plots/confusion_matrix.png
    - plots/roc_curve.png
```

### 3. **Parameters and Metrics Management**

```yaml
# params.yaml - Centralized parameter configuration
prepare:
  test_size: 0.2
  random_state: 42
  
features:
  n_components: 10
  scaler_type: "standard"
  
train:
  algorithm: "random_forest"
  hyperparameters:
    n_estimators: 100
    max_depth: 10
    min_samples_split: 5
    
evaluate:
  metrics:
    - accuracy
    - precision
    - recall
    - f1_score
```

```python
# src/train_model.py - Parameter-driven training
import yaml
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params

def train_model():
    """Train model with versioned parameters"""
    params = load_params()
    
    # Load data
    data = pd.read_csv('data/features/features.csv')
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['prepare']['test_size'],
        random_state=params['prepare']['random_state']
    )
    
    # Train model
    model_params = params['train']['hyperparameters']
    model = RandomForestClassifier(**model_params)
    model.fit(X_train, y_train)
    
    # Save model
    with open('models/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    # Calculate and save metrics
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    metrics = {
        'train_accuracy': train_score,
        'test_accuracy': test_score,
        'model_params': model_params
    }
    
    with open('metrics/train_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == "__main__":
    train_model()
```

---

## 🏗️ MLflow Model Registry

### 1. **MLflow Tracking and Experiments**

```python
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class MLflowModelManager:
    def __init__(self, experiment_name="ml_pipeline", tracking_uri="sqlite:///mlflow.db"):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()
        
    def train_and_log_model(self, data_path, model_params, run_name=None):
        """Train model and log to MLflow"""
        
        with mlflow.start_run(run_name=run_name) as run:
            # Load and prepare data
            data = pd.read_csv(data_path)
            X = data.drop('target', axis=1)
            y = data['target']
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Log parameters
            mlflow.log_params(model_params)
            mlflow.log_param("data_path", data_path)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Train model
            model = RandomForestClassifier(**model_params)
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted')
            }
            
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name="fraud_detection_model"
            )
            
            # Log artifacts
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv("feature_importance.csv", index=False)
            mlflow.log_artifact("feature_importance.csv")
            
            return run.info.run_id, metrics
    
    def register_model(self, run_id, model_name, stage="Staging"):
        """Register model in MLflow Model Registry"""
        
        model_uri = f"runs:/{run_id}/model"
        
        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        
        # Transition to specified stage
        self.client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage=stage
        )
        
        return model_version
    
    def promote_model(self, model_name, version, stage):
        """Promote model to production stage"""
        
        self.client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        # Add description
        self.client.update_model_version(
            name=model_name,
            version=version,
            description=f"Promoted to {stage} on {pd.Timestamp.now()}"
        )
    
    def load_production_model(self, model_name):
        """Load the current production model"""
        
        model_version = self.client.get_latest_versions(
            model_name, 
            stages=["Production"]
        )[0]
        
        model_uri = f"models:/{model_name}/{model_version.version}"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model, model_version
    
    def compare_models(self, model_name, versions):
        """Compare multiple model versions"""
        
        comparison_data = []
        
        for version in versions:
            model_version = self.client.get_model_version(model_name, version)
            run = self.client.get_run(model_version.run_id)
            
            comparison_data.append({
                'version': version,
                'stage': model_version.current_stage,
                'accuracy': run.data.metrics.get('accuracy', 0),
                'f1_score': run.data.metrics.get('f1_score', 0),
                'created_at': model_version.creation_timestamp
            })
        
        return pd.DataFrame(comparison_data)
```

### 2. **Model Deployment and Rollback Strategies**

```python
class ModelDeploymentManager:
    def __init__(self, mlflow_manager):
        self.mlflow_manager = mlflow_manager
        self.deployment_history = []
        
    def deploy_model(self, model_name, version, deployment_config):
        """Deploy model with blue-green deployment strategy"""
        
        deployment_id = f"deploy_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Load model
            model_uri = f"models:/{model_name}/{version}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Validate model
            validation_result = self._validate_model(model, deployment_config)
            
            if not validation_result['is_valid']:
                raise ValueError(f"Model validation failed: {validation_result['errors']}")
            
            # Deploy to staging environment first
            staging_deployment = self._deploy_to_staging(model, deployment_config)
            
            # Run smoke tests
            smoke_test_result = self._run_smoke_tests(staging_deployment)
            
            if not smoke_test_result['passed']:
                raise ValueError(f"Smoke tests failed: {smoke_test_result['errors']}")
            
            # Deploy to production (blue-green)
            production_deployment = self._deploy_to_production(model, deployment_config)
            
            # Record deployment
            deployment_record = {
                'deployment_id': deployment_id,
                'model_name': model_name,
                'model_version': version,
                'timestamp': pd.Timestamp.now(),
                'status': 'success',
                'config': deployment_config
            }
            
            self.deployment_history.append(deployment_record)
            
            # Promote model to Production stage
            self.mlflow_manager.promote_model(model_name, version, "Production")
            
            return deployment_record
            
        except Exception as e:
            # Record failed deployment
            deployment_record = {
                'deployment_id': deployment_id,
                'model_name': model_name,
                'model_version': version,
                'timestamp': pd.Timestamp.now(),
                'status': 'failed',
                'error': str(e),
                'config': deployment_config
            }
            
            self.deployment_history.append(deployment_record)
            raise
    
    def rollback_deployment(self, model_name, target_version=None):
        """Rollback to previous model version"""
        
        if target_version is None:
            # Find last successful deployment
            successful_deployments = [
                d for d in self.deployment_history 
                if d['model_name'] == model_name and d['status'] == 'success'
            ]
            
            if len(successful_deployments) < 2:
                raise ValueError("No previous successful deployment found for rollback")
            
            target_version = successful_deployments[-2]['model_version']
        
        # Deploy previous version
        rollback_config = {
            'deployment_type': 'rollback',
            'reason': 'Manual rollback requested'
        }
        
        return self.deploy_model(model_name, target_version, rollback_config)
    
    def _validate_model(self, model, config):
        """Validate model before deployment"""
        
        errors = []
        
        # Check model attributes
        if not hasattr(model, 'predict'):
            errors.append("Model missing predict method")
        
        if not hasattr(model, 'predict_proba'):
            errors.append("Model missing predict_proba method")
        
        # Check model size
        model_size = len(pickle.dumps(model))
        max_size = config.get('max_model_size_mb', 100) * 1024 * 1024
        
        if model_size > max_size:
            errors.append(f"Model size {model_size/1024/1024:.1f}MB exceeds limit")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors
        }
    
    def _deploy_to_staging(self, model, config):
        """Deploy model to staging environment"""
        
        # Simulate staging deployment
        staging_endpoint = f"staging-{config.get('service_name', 'ml-service')}"
        
        return {
            'endpoint': staging_endpoint,
            'status': 'deployed',
            'timestamp': pd.Timestamp.now()
        }
    
    def _run_smoke_tests(self, deployment):
        """Run smoke tests on deployed model"""
        
        # Simulate smoke tests
        test_results = {
            'latency_test': True,
            'prediction_test': True,
            'health_check': True
        }
        
        passed = all(test_results.values())
        
        return {
            'passed': passed,
            'results': test_results,
            'errors': [] if passed else ['Smoke test simulation']
        }
    
    def _deploy_to_production(self, model, config):
        """Deploy model to production with blue-green strategy"""
        
        # Simulate production deployment
        production_endpoint = f"prod-{config.get('service_name', 'ml-service')}"
        
        return {
            'endpoint': production_endpoint,
            'status': 'deployed',
            'timestamp': pd.Timestamp.now()
        }
```

---

## 🔄 Reproducible ML Pipelines

### 1. **Pipeline Orchestration with DVC and MLflow**

```python
import subprocess
import yaml
import json
from pathlib import Path

class MLPipelineOrchestrator:
    def __init__(self, pipeline_config_path="pipeline_config.yaml"):
        self.config_path = pipeline_config_path
        self.load_config()
        
    def load_config(self):
        """Load pipeline configuration"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run_pipeline(self, stages=None, force=False):
        """Run DVC pipeline with MLflow tracking"""
        
        if stages is None:
            stages = list(self.config['stages'].keys())
        
        results = {}
        
        for stage in stages:
            print(f"Running stage: {stage}")
            
            try:
                # Run DVC stage
                cmd = ['dvc', 'repro', stage]
                if force:
                    cmd.append('--force')
                
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode != 0:
                    raise RuntimeError(f"Stage {stage} failed: {result.stderr}")
                
                results[stage] = {
                    'status': 'success',
                    'output': result.stdout
                }
                
            except Exception as e:
                results[stage] = {
                    'status': 'failed',
                    'error': str(e)
                }
                
                if self.config.get('fail_fast', True):
                    break
        
        return results
    
    def validate_pipeline(self):
        """Validate pipeline configuration and dependencies"""
        
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if all required files exist
        for stage_name, stage_config in self.config['stages'].items():
            # Check dependencies
            for dep in stage_config.get('deps', []):
                if not Path(dep).exists():
                    validation_results['errors'].append(
                        f"Stage {stage_name}: dependency {dep} not found"
                    )
            
            # Check command
            cmd = stage_config.get('cmd')
            if not cmd:
                validation_results['errors'].append(
                    f"Stage {stage_name}: no command specified"
                )
        
        validation_results['is_valid'] = len(validation_results['errors']) == 0
        
        return validation_results
    
    def generate_pipeline_report(self):
        """Generate comprehensive pipeline report"""
        
        # Get DVC pipeline status
        result = subprocess.run(['dvc', 'status'], capture_output=True, text=True)
        dvc_status = result.stdout
        
        # Get metrics
        metrics = {}
        metrics_files = Path('metrics').glob('*.json')
        
        for metrics_file in metrics_files:
            with open(metrics_file, 'r') as f:
                stage_metrics = json.load(f)
                metrics[metrics_file.stem] = stage_metrics
        
        # Get parameters
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        
        report = {
            'pipeline_status': dvc_status,
            'metrics': metrics,
            'parameters': params,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return report
```

### 2. **Data Lineage and Dependency Tracking**

```python
class DataLineageTracker:
    def __init__(self):
        self.lineage_graph = {}
        
    def track_data_transformation(self, input_data, output_data, transformation_info):
        """Track data transformation for lineage"""
        
        lineage_record = {
            'input_data': input_data,
            'output_data': output_data,
            'transformation': transformation_info,
            'timestamp': pd.Timestamp.now().isoformat(),
            'hash_input': self._calculate_data_hash(input_data),
            'hash_output': self._calculate_data_hash(output_data)
        }
        
        self.lineage_graph[output_data] = lineage_record
        
    def get_data_lineage(self, data_path):
        """Get complete lineage for a data artifact"""
        
        lineage = []
        current_path = data_path
        
        while current_path in self.lineage_graph:
            record = self.lineage_graph[current_path]
            lineage.append(record)
            current_path = record['input_data']
        
        return lineage
    
    def visualize_lineage(self, data_path):
        """Create visualization of data lineage"""
        
        lineage = self.get_data_lineage(data_path)
        
        # Create simple text-based visualization
        visualization = f"Data Lineage for {data_path}:\n"
        visualization += "=" * 50 + "\n"
        
        for i, record in enumerate(reversed(lineage)):
            indent = "  " * i
            visualization += f"{indent}{record['input_data']}\n"
            visualization += f"{indent}  └─ {record['transformation']['name']}\n"
            visualization += f"{indent}     └─ {record['output_data']}\n"
        
        return visualization
    
    def _calculate_data_hash(self, data_path):
        """Calculate hash of data file for change detection"""
        
        if isinstance(data_path, str) and Path(data_path).exists():
            import hashlib
            
            with open(data_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            
            return file_hash
        
        return None
```

---

## 🚀 CI/CD Integration for ML

### 1. **Automated Model Training Pipeline**

```yaml
# .github/workflows/ml_pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  data-validation:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.8
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Validate data schema
      run: |
        python scripts/validate_data_schema.py
    
    - name: Run data quality checks
      run: |
        python scripts/data_quality_checks.py

  model-training:
    needs: data-validation
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up DVC
      uses: iterative/setup-dvc@v1
    
    - name: Configure DVC remote
      run: |
        dvc remote add -d storage s3://ml-artifacts-bucket
        dvc remote modify storage access_key_id ${{ secrets.AWS_ACCESS_KEY_ID }}
        dvc remote modify storage secret_access_key ${{ secrets.AWS_SECRET_ACCESS_KEY }}
    
    - name: Pull data
      run: dvc pull
    
    - name: Run ML pipeline
      run: dvc repro
    
    - name: Push artifacts
      run: dvc push
    
    - name: Upload metrics
      uses: actions/upload-artifact@v2
      with:
        name: metrics
        path: metrics/

  model-validation:
    needs: model-training
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    
    - name: Download metrics
      uses: actions/download-artifact@v2
      with:
        name: metrics
        path: metrics/
    
    - name: Validate model performance
      run: |
        python scripts/validate_model_performance.py
    
    - name: Run model tests
      run: |
        python -m pytest tests/test_model.py

  deploy-staging:
    needs: model-validation
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    steps:
    - name: Deploy to staging
      run: |
        python scripts/deploy_model.py --environment staging
```

### 2. **Model Performance Monitoring**

```python
class ModelPerformanceMonitor:
    def __init__(self, model_name, mlflow_manager):
        self.model_name = model_name
        self.mlflow_manager = mlflow_manager
        self.performance_history = []
        
    def monitor_model_drift(self, new_data, reference_data):
        """Monitor for data drift in production"""
        
        from scipy.stats import ks_2samp
        
        drift_results = {}
        
        for column in new_data.columns:
            if column in reference_data.columns:
                # Kolmogorov-Smirnov test for distribution drift
                statistic, p_value = ks_2samp(
                    reference_data[column].dropna(),
                    new_data[column].dropna()
                )
                
                drift_results[column] = {
                    'ks_statistic': statistic,
                    'p_value': p_value,
                    'drift_detected': p_value < 0.05
                }
        
        return drift_results
    
    def monitor_performance_degradation(self, predictions, actuals):
        """Monitor for model performance degradation"""
        
        from sklearn.metrics import accuracy_score, f1_score
        
        current_performance = {
            'accuracy': accuracy_score(actuals, predictions),
            'f1_score': f1_score(actuals, predictions, average='weighted'),
            'timestamp': pd.Timestamp.now()
        }
        
        self.performance_history.append(current_performance)
        
        # Check for degradation
        if len(self.performance_history) > 1:
            previous_performance = self.performance_history[-2]
            
            accuracy_drop = (
                previous_performance['accuracy'] - current_performance['accuracy']
            )
            
            f1_drop = (
                previous_performance['f1_score'] - current_performance['f1_score']
            )
            
            degradation_detected = accuracy_drop > 0.05 or f1_drop > 0.05
            
            return {
                'degradation_detected': degradation_detected,
                'accuracy_drop': accuracy_drop,
                'f1_drop': f1_drop,
                'current_performance': current_performance
            }
        
        return {'degradation_detected': False}
    
    def generate_monitoring_report(self):
        """Generate comprehensive monitoring report"""
        
        if not self.performance_history:
            return {'error': 'No performance history available'}
        
        df = pd.DataFrame(self.performance_history)
        
        report = {
            'model_name': self.model_name,
            'monitoring_period': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'performance_summary': {
                'avg_accuracy': df['accuracy'].mean(),
                'min_accuracy': df['accuracy'].min(),
                'max_accuracy': df['accuracy'].max(),
                'accuracy_trend': 'improving' if df['accuracy'].iloc[-1] > df['accuracy'].iloc[0] else 'declining'
            },
            'alerts': self._generate_alerts(df)
        }
        
        return report
    
    def _generate_alerts(self, performance_df):
        """Generate performance alerts"""
        
        alerts = []
        
        # Check for consistent degradation
        if len(performance_df) >= 5:
            recent_trend = performance_df['accuracy'].tail(5).diff().mean()
            if recent_trend < -0.01:
                alerts.append({
                    'type': 'performance_degradation',
                    'message': 'Consistent accuracy decline detected over last 5 measurements',
                    'severity': 'high'
                })
        
        # Check for sudden drops
        if len(performance_df) >= 2:
            latest_accuracy = performance_df['accuracy'].iloc[-1]
            previous_accuracy = performance_df['accuracy'].iloc[-2]
            
            if previous_accuracy - latest_accuracy > 0.1:
                alerts.append({
                    'type': 'sudden_drop',
                    'message': f'Sudden accuracy drop: {previous_accuracy:.3f} → {latest_accuracy:.3f}',
                    'severity': 'critical'
                })
        
        return alerts
```

---

## 🔧 Hands-On Exercise

You'll build a complete model versioning system that integrates DVC, MLflow, and automated deployment:

### Exercise Scenario
**Company**: HealthTech Analytics  
**Challenge**: Build a reproducible ML pipeline for medical diagnosis prediction
- **Data Versioning**: Track medical datasets with patient privacy
- **Model Registry**: Manage multiple diagnostic models with approval workflows
- **Deployment Pipeline**: Safe rollout with A/B testing integration
- **Monitoring**: Track model performance and data drift in production

### Requirements
1. **DVC Pipeline**: Reproducible data processing and model training
2. **MLflow Registry**: Model versioning with stage transitions
3. **Deployment Automation**: Blue-green deployment with rollback
4. **Performance Monitoring**: Drift detection and alerting
5. **CI/CD Integration**: Automated testing and deployment

---

## 📚 Key Takeaways

- **Version everything** - data, models, code, and configurations for complete reproducibility
- **Use content-based addressing** for efficient storage and deduplication of large artifacts
- **Implement proper model registry** with stage transitions and approval workflows
- **Design for rollback** - always have a path back to previous working versions
- **Monitor continuously** - track both technical metrics and business impact
- **Automate safely** - use CI/CD with proper validation gates and testing
- **Document lineage** - maintain clear data and model provenance for compliance
- **Plan for scale** - design versioning systems that work with large teams and datasets

---

## 🔄 What's Next?

Tomorrow, we'll explore **CI/CD for ML** where you'll learn how to:
- Build automated ML pipelines with testing and validation
- Implement infrastructure as code for ML systems
- Create deployment strategies for ML models at scale
- Handle model monitoring and automated retraining

The versioning foundation you build today will enable sophisticated automation and deployment strategies.

---

## 📖 Additional Resources

### DVC and Data Versioning
- [DVC Documentation](https://dvc.org/doc)
- [DVC Best Practices](https://dvc.org/doc/user-guide/best-practices)
- [Data Versioning Patterns](https://dvc.org/doc/use-cases/versioning-data-and-model-files)

### MLflow Model Registry
- [MLflow Model Registry Guide](https://mlflow.org/docs/latest/model-registry.html)
- [MLflow Production Deployment](https://mlflow.org/docs/latest/models.html#deploy-mlflow-models)
- [Model Lifecycle Management](https://mlflow.org/docs/latest/model-registry.html#model-registry-workflows)

### ML Pipeline Orchestration
- [Reproducible ML Pipelines](https://papers.nips.cc/paper/2019/file/5878a7ab84fb43402106c575658472fa-Paper.pdf)
- [ML Pipeline Design Patterns](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)
- [Versioning for ML Systems](https://blog.ml.cmu.edu/2020/12/31/versioning-machine-learning/)

### Production ML Systems
- [Hidden Technical Debt in ML Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf)
- [ML Model Management](https://www.usenix.org/system/files/conference/opml19/opml19-papers-zaharia.pdf)
- [Production ML Monitoring](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/)