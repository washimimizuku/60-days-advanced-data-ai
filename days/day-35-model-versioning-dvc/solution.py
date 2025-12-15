"""
Day 35: Model Versioning with DVC - Complete Solution

Production-ready model versioning system for HealthTech Analytics demonstrating
comprehensive MLOps practices including DVC pipelines, MLflow model registry,
automated deployments, and regulatory compliance for healthcare AI applications.

This solution implements:
1. DVC pipeline management with reproducible workflows
2. MLflow model registry with stage transitions and approval workflows
3. Data lineage tracking for complete audit trails
4. Blue-green deployment strategy with automated rollback
5. Performance monitoring with drift detection and alerting
6. Regulatory compliance reporting for healthcare applications
7. CI/CD integration with automated testing and validation

Architecture Components:
- DVC Pipeline: Reproducible data processing and model training
- MLflow Registry: Model versioning, staging, and production management
- Deployment Manager: Blue-green deployments with validation and rollback
- Performance Monitor: Real-time monitoring with drift detection
- Compliance System: Audit trails and regulatory reporting
"""

import os
import json
import yaml
import pickle
import hashlib
import subprocess
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict, field
from abc import ABC, abstractmethod
import logging
import uuid
import time

# ML libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# MLflow (mock implementation if not available)
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available, using mock implementation")

# Statistical libraries for drift detection
try:
    from scipy.stats import ks_2samp, chi2_contingency
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, using simplified drift detection")

# =============================================================================
# CONFIGURATION AND DATA STRUCTURES
# =============================================================================

@dataclass
class DVCPipelineConfig:
    """Configuration for DVC pipeline stages"""
    stages: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=list)
    output_artifacts: List[str] = field(default_factory=list)
    
@dataclass
class ModelMetadata:
    """Metadata for model versions"""
    model_name: str
    version: str
    stage: str
    created_at: datetime
    performance_metrics: Dict[str, float]
    training_data_hash: str
    model_hash: str
    parameters: Dict[str, Any]
    
@dataclass
class DeploymentRecord:
    """Record of model deployment"""
    deployment_id: str
    model_name: str
    model_version: str
    environment: str
    timestamp: datetime
    status: str
    validation_results: Dict[str, Any]
    rollback_version: Optional[str] = None

# =============================================================================
# DVC PIPELINE MANAGEMENT
# =============================================================================

class DVCPipelineManager:
    """Manages DVC pipeline execution and configuration"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.dvc_file = self.project_root / "dvc.yaml"
        self.params_file = self.project_root / "params.yaml"
        self.pipeline_config = None
        self.logger = logging.getLogger(__name__)
        
    def initialize_dvc_project(self):
        """Initialize DVC in the project directory"""
        
        try:
            # Check if DVC is already initialized
            if (self.project_root / ".dvc").exists():
                print("DVC already initialized")
                return True
            
            # Initialize DVC (simulate since we might not have DVC installed)
            dvc_dir = self.project_root / ".dvc"
            dvc_dir.mkdir(exist_ok=True)
            
            # Create basic DVC config
            config_dir = dvc_dir / "config"
            config_dir.touch()
            
            # Create .dvcignore
            dvcignore = self.project_root / ".dvcignore"
            dvcignore.write_text("# Add patterns of files dvc should ignore\n")
            
            print("✅ DVC project initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize DVC: {e}")
            return False
    
    def create_pipeline_stage(self, stage_name: str, command: str, 
                            dependencies: List[str], outputs: List[str],
                            parameters: List[str] = None, metrics: List[str] = None):
        """Create a DVC pipeline stage"""
        
        if self.pipeline_config is None:
            self.pipeline_config = {'stages': {}}
        
        stage_config = {
            'cmd': command,
            'deps': dependencies,
            'outs': outputs
        }
        
        if parameters:
            stage_config['params'] = parameters
            
        if metrics:
            stage_config['metrics'] = metrics
        
        self.pipeline_config['stages'][stage_name] = stage_config
        
        # Save to dvc.yaml
        self._save_pipeline_config()
        
        print(f"✅ Created pipeline stage: {stage_name}")
    
    def _save_pipeline_config(self):
        """Save pipeline configuration to dvc.yaml"""
        
        if self.pipeline_config:
            with open(self.dvc_file, 'w') as f:
                yaml.dump(self.pipeline_config, f, default_flow_style=False)
    
    def run_pipeline(self, stages: List[str] = None, force: bool = False) -> Dict[str, Any]:
        """Run DVC pipeline"""
        
        results = {}
        
        if not self.pipeline_config:
            return {'error': 'No pipeline configuration found'}
        
        stages_to_run = stages or list(self.pipeline_config['stages'].keys())
        
        for stage in stages_to_run:
            if stage not in self.pipeline_config['stages']:
                results[stage] = {'status': 'failed', 'error': f'Stage {stage} not found'}
                continue
            
            try:
                print(f"Running stage: {stage}")
                
                stage_config = self.pipeline_config['stages'][stage]
                command = stage_config['cmd']
                
                # Simulate running the command
                # In real implementation, this would execute the actual command
                result = self._simulate_stage_execution(stage, stage_config)
                
                results[stage] = {
                    'status': 'success',
                    'command': command,
                    'execution_time': result.get('execution_time', 0),
                    'outputs': stage_config.get('outs', [])
                }
                
            except Exception as e:
                results[stage] = {
                    'status': 'failed',
                    'error': str(e)
                }
                
                if not force:
                    break
        
        return results
    
    def _simulate_stage_execution(self, stage_name: str, stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate stage execution for demonstration"""
        
        # Create output directories
        for output in stage_config.get('outs', []):
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create dummy output file
            if not output_path.exists():
                if output_path.suffix == '.csv':
                    pd.DataFrame({'dummy': [1, 2, 3]}).to_csv(output_path, index=False)
                elif output_path.suffix == '.json':
                    with open(output_path, 'w') as f:
                        json.dump({'stage': stage_name, 'timestamp': datetime.now().isoformat()}, f)
                else:
                    output_path.touch()
        
        return {'execution_time': np.random.uniform(1, 5)}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        
        if not self.dvc_file.exists():
            return {'status': 'no_pipeline', 'message': 'No DVC pipeline found'}
        
        # Simulate DVC status check
        status = {
            'pipeline_file': str(self.dvc_file),
            'stages': {},
            'last_run': datetime.now().isoformat()
        }
        
        if self.pipeline_config:
            for stage_name, stage_config in self.pipeline_config['stages'].items():
                # Check if outputs exist
                outputs_exist = all(
                    Path(output).exists() 
                    for output in stage_config.get('outs', [])
                )
                
                status['stages'][stage_name] = {
                    'outputs_exist': outputs_exist,
                    'status': 'up_to_date' if outputs_exist else 'needs_update'
                }
        
        return status
    
    def add_data_file(self, file_path: str):
        """Add data file to DVC tracking"""
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")
        
        # Create .dvc file
        dvc_file = file_path.with_suffix(file_path.suffix + '.dvc')
        
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Create DVC metadata
        dvc_metadata = {
            'outs': [{
                'path': str(file_path),
                'md5': file_hash,
                'size': file_path.stat().st_size
            }]
        }
        
        with open(dvc_file, 'w') as f:
            yaml.dump(dvc_metadata, f)
        
        # Add to .gitignore
        gitignore_path = self.project_root / ".gitignore"
        
        if gitignore_path.exists():
            gitignore_content = gitignore_path.read_text()
        else:
            gitignore_content = ""
        
        if str(file_path) not in gitignore_content:
            with open(gitignore_path, 'a') as f:
                f.write(f"\n{file_path}\n")
        
        print(f"✅ Added {file_path} to DVC tracking")
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate MD5 hash of file"""
        
        hash_md5 = hashlib.md5()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        
        return hash_md5.hexdigest()
# =============================================================================
# MLFLOW MODEL MANAGEMENT
# =============================================================================

class MLflowModelManager:
    """Manages MLflow experiments and model registry"""
    
    def __init__(self, experiment_name: str = "healthtech_diagnostics", 
                 tracking_uri: str = "sqlite:///mlflow.db"):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.current_run = None
        self.model_registry = {}  # Mock registry if MLflow not available
        
        if MLFLOW_AVAILABLE:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.client = MlflowClient()
        else:
            self.client = None
            print("Using mock MLflow implementation")
    
    def start_experiment_run(self, run_name: str = None) -> str:
        """Start a new MLflow experiment run"""
        
        if MLFLOW_AVAILABLE:
            self.current_run = mlflow.start_run(run_name=run_name)
            return self.current_run.info.run_id
        else:
            # Mock implementation
            run_id = str(uuid.uuid4())
            self.current_run = {
                'run_id': run_id,
                'run_name': run_name,
                'start_time': datetime.now(),
                'params': {},
                'metrics': {},
                'artifacts': []
            }
            return run_id
    
    def log_parameters(self, params: Dict[str, Any]):
        """Log parameters to current run"""
        
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.log_params(params)
        elif self.current_run:
            self.current_run['params'].update(params)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to current run"""
        
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.log_metrics(metrics)
        elif self.current_run:
            self.current_run['metrics'].update(metrics)
    
    def log_model(self, model, model_name: str, registered_model_name: str = None):
        """Log model to MLflow"""
        
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.sklearn.log_model(
                model, 
                model_name,
                registered_model_name=registered_model_name
            )
        elif self.current_run:
            # Mock model logging
            model_path = f"models/{model_name}.pkl"
            os.makedirs("models", exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            self.current_run['artifacts'].append({
                'name': model_name,
                'path': model_path,
                'type': 'model'
            })
            
            if registered_model_name:
                self._register_model_mock(model_path, registered_model_name)
    
    def _register_model_mock(self, model_path: str, model_name: str):
        """Mock model registration"""
        
        if model_name not in self.model_registry:
            self.model_registry[model_name] = {'versions': []}
        
        version = len(self.model_registry[model_name]['versions']) + 1
        
        self.model_registry[model_name]['versions'].append({
            'version': str(version),
            'path': model_path,
            'stage': 'None',
            'created_at': datetime.now(),
            'run_id': self.current_run['run_id']
        })
    
    def register_model(self, run_id: str, model_name: str, stage: str = "Staging"):
        """Register model in MLflow Model Registry"""
        
        if MLFLOW_AVAILABLE:
            model_uri = f"runs:/{run_id}/model"
            model_version = mlflow.register_model(model_uri, model_name)
            
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage=stage
            )
            
            return model_version
        else:
            # Mock registration
            if model_name in self.model_registry:
                latest_version = self.model_registry[model_name]['versions'][-1]
                latest_version['stage'] = stage
                return latest_version
    
    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to different stage"""
        
        if MLFLOW_AVAILABLE:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
        else:
            # Mock promotion
            if model_name in self.model_registry:
                for model_version in self.model_registry[model_name]['versions']:
                    if model_version['version'] == version:
                        model_version['stage'] = stage
                        break
    
    def load_model(self, model_name: str, stage: str = "Production"):
        """Load model from registry"""
        
        if MLFLOW_AVAILABLE:
            model_uri = f"models:/{model_name}/{stage}"
            return mlflow.sklearn.load_model(model_uri)
        else:
            # Mock model loading
            if model_name in self.model_registry:
                for model_version in self.model_registry[model_name]['versions']:
                    if model_version['stage'] == stage:
                        with open(model_version['path'], 'rb') as f:
                            return pickle.load(f)
            
            raise ValueError(f"No model found for {model_name} in stage {stage}")
    
    def get_model_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """Get all versions of a model"""
        
        if MLFLOW_AVAILABLE:
            return self.client.search_model_versions(f"name='{model_name}'")
        else:
            if model_name in self.model_registry:
                return self.model_registry[model_name]['versions']
            return []
    
    def end_run(self):
        """End current MLflow run"""
        
        if MLFLOW_AVAILABLE and self.current_run:
            mlflow.end_run()
        
        self.current_run = None

# =============================================================================
# DATA LINEAGE TRACKING
# =============================================================================

class DataLineageTracker:
    """Tracks data lineage and transformations"""
    
    def __init__(self):
        self.lineage_graph = {}
        self.transformations = []
        
    def track_transformation(self, input_data: str, output_data: str, 
                           transformation_info: Dict[str, Any]):
        """Track a data transformation"""
        
        transformation_record = {
            'id': str(uuid.uuid4()),
            'input_data': input_data,
            'output_data': output_data,
            'transformation_info': transformation_info,
            'timestamp': datetime.now(),
            'input_hash': self.calculate_data_hash(input_data),
            'output_hash': self.calculate_data_hash(output_data)
        }
        
        self.transformations.append(transformation_record)
        self.lineage_graph[output_data] = transformation_record
        
        print(f"✅ Tracked transformation: {input_data} → {output_data}")
    
    def get_lineage(self, data_identifier: str) -> List[Dict[str, Any]]:
        """Get complete lineage for a data artifact"""
        
        lineage = []
        current_data = data_identifier
        
        while current_data in self.lineage_graph:
            record = self.lineage_graph[current_data]
            lineage.append(record)
            current_data = record['input_data']
        
        return lineage
    
    def visualize_lineage(self, data_identifier: str) -> str:
        """Create text visualization of data lineage"""
        
        lineage = self.get_lineage(data_identifier)
        
        if not lineage:
            return f"No lineage found for {data_identifier}"
        
        visualization = f"Data Lineage for {data_identifier}:\n"
        visualization += "=" * 50 + "\n"
        
        for i, record in enumerate(reversed(lineage)):
            indent = "  " * i
            transformation_name = record['transformation_info'].get('name', 'Unknown')
            
            visualization += f"{indent}{record['input_data']}\n"
            visualization += f"{indent}  └─ [{transformation_name}]\n"
            visualization += f"{indent}     └─ {record['output_data']}\n"
        
        return visualization
    
    def calculate_data_hash(self, file_path: str) -> Optional[str]:
        """Calculate hash of data file"""
        
        if not isinstance(file_path, str):
            return None
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            return None
        
        hash_md5 = hashlib.md5()
        
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            
            return hash_md5.hexdigest()
        except Exception:
            return None
    
    def export_lineage(self, output_path: str):
        """Export lineage graph to JSON"""
        
        lineage_data = {
            'transformations': [],
            'export_timestamp': datetime.now().isoformat()
        }
        
        for transformation in self.transformations:
            # Convert datetime to string for JSON serialization
            transformation_copy = transformation.copy()
            transformation_copy['timestamp'] = transformation['timestamp'].isoformat()
            lineage_data['transformations'].append(transformation_copy)
        
        with open(output_path, 'w') as f:
            json.dump(lineage_data, f, indent=2)
        
        print(f"✅ Exported lineage to {output_path}")

# =============================================================================
# MODEL DEPLOYMENT MANAGEMENT
# =============================================================================

class ModelDeploymentManager:
    """Manages model deployments with blue-green strategy"""
    
    def __init__(self, mlflow_manager: MLflowModelManager):
        self.mlflow_manager = mlflow_manager
        self.deployment_history = []
        self.current_deployments = {}
        
    def deploy_model(self, model_name: str, version: str, 
                    environment: str = "staging") -> Dict[str, Any]:
        """Deploy model using blue-green deployment strategy"""
        
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            print(f"Starting deployment {deployment_id}")
            
            # Load model
            if environment == "production":
                model = self.mlflow_manager.load_model(model_name, "Production")
            else:
                model = self.mlflow_manager.load_model(model_name, "Staging")
            
            # Validate model
            validation_results = self.validate_model(model)
            
            if not validation_results['is_valid']:
                raise ValueError(f"Model validation failed: {validation_results['errors']}")
            
            # Deploy to environment
            deployment_info = self._deploy_to_environment(model, environment, deployment_id)
            
            # Run smoke tests
            smoke_test_results = self.run_smoke_tests(deployment_info)
            
            if not smoke_test_results['passed']:
                raise ValueError(f"Smoke tests failed: {smoke_test_results['errors']}")
            
            # Record successful deployment
            deployment_record = DeploymentRecord(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=version,
                environment=environment,
                timestamp=datetime.now(),
                status='success',
                validation_results=validation_results
            )
            
            self.deployment_history.append(deployment_record)
            self.current_deployments[f"{model_name}_{environment}"] = deployment_record
            
            print(f"✅ Successfully deployed {model_name} v{version} to {environment}")
            
            return asdict(deployment_record)
            
        except Exception as e:
            # Record failed deployment
            deployment_record = DeploymentRecord(
                deployment_id=deployment_id,
                model_name=model_name,
                model_version=version,
                environment=environment,
                timestamp=datetime.now(),
                status='failed',
                validation_results={'error': str(e)}
            )
            
            self.deployment_history.append(deployment_record)
            
            print(f"❌ Deployment failed: {e}")
            raise
    
    def rollback_deployment(self, model_name: str, environment: str = "production"):
        """Rollback to previous model version"""
        
        # Find previous successful deployment
        successful_deployments = [
            d for d in self.deployment_history 
            if (d.model_name == model_name and 
                d.environment == environment and 
                d.status == 'success')
        ]
        
        if len(successful_deployments) < 2:
            raise ValueError("No previous successful deployment found for rollback")
        
        previous_deployment = successful_deployments[-2]
        
        print(f"Rolling back {model_name} to version {previous_deployment.model_version}")
        
        # Deploy previous version
        return self.deploy_model(
            model_name, 
            previous_deployment.model_version, 
            environment
        )
    
    def validate_model(self, model) -> Dict[str, Any]:
        """Validate model before deployment"""
        
        errors = []
        warnings = []
        
        # Check model has required methods
        if not hasattr(model, 'predict'):
            errors.append("Model missing predict method")
        
        if not hasattr(model, 'predict_proba'):
            warnings.append("Model missing predict_proba method")
        
        # Check model size (simplified)
        try:
            model_size = len(pickle.dumps(model))
            max_size = 100 * 1024 * 1024  # 100MB
            
            if model_size > max_size:
                errors.append(f"Model size {model_size/1024/1024:.1f}MB exceeds 100MB limit")
        except Exception as e:
            warnings.append(f"Could not determine model size: {e}")
        
        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def _deploy_to_environment(self, model, environment: str, deployment_id: str) -> Dict[str, Any]:
        """Deploy model to specified environment"""
        
        # Simulate deployment process
        deployment_path = f"deployments/{environment}/{deployment_id}"
        os.makedirs(deployment_path, exist_ok=True)
        
        # Save model
        model_path = f"{deployment_path}/model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Create deployment metadata
        metadata = {
            'deployment_id': deployment_id,
            'environment': environment,
            'model_path': model_path,
            'endpoint': f"http://{environment}-api.healthtech.com/predict",
            'deployed_at': datetime.now().isoformat()
        }
        
        with open(f"{deployment_path}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata
    
    def run_smoke_tests(self, deployment_info: Dict[str, Any]) -> Dict[str, Any]:
        """Run smoke tests on deployed model"""
        
        test_results = {}
        errors = []
        
        try:
            # Load deployed model
            model_path = deployment_info['model_path']
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            # Test 1: Basic prediction
            test_data = np.random.random((5, 10))  # 5 samples, 10 features
            predictions = model.predict(test_data)
            test_results['prediction_test'] = len(predictions) == 5
            
            # Test 2: Prediction consistency
            predictions2 = model.predict(test_data)
            test_results['consistency_test'] = np.array_equal(predictions, predictions2)
            
            # Test 3: Performance test (latency)
            start_time = time.time()
            for _ in range(100):
                model.predict(test_data[:1])
            avg_latency = (time.time() - start_time) / 100
            
            test_results['latency_test'] = avg_latency < 0.1  # Less than 100ms
            test_results['avg_latency_ms'] = avg_latency * 1000
            
        except Exception as e:
            errors.append(f"Smoke test error: {e}")
            test_results['error'] = str(e)
        
        passed = len(errors) == 0 and all(test_results.get(test, False) for test in ['prediction_test', 'consistency_test', 'latency_test'])
        
        return {
            'passed': passed,
            'results': test_results,
            'errors': errors
        }
    
    def get_deployment_history(self, model_name: str = None) -> List[Dict[str, Any]]:
        """Get deployment history"""
        
        if model_name:
            history = [d for d in self.deployment_history if d.model_name == model_name]
        else:
            history = self.deployment_history
        
        return [asdict(d) for d in history]
# =============================================================================
# PERFORMANCE MONITORING
# =============================================================================

class ModelPerformanceMonitor:
    """Monitors model performance and detects drift"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.performance_history = []
        self.drift_history = []
        self.alerts = []
        
    def monitor_performance(self, predictions: np.array, actuals: np.array) -> Dict[str, Any]:
        """Monitor model performance"""
        
        # Calculate performance metrics
        metrics = {
            'accuracy': accuracy_score(actuals, predictions),
            'precision': precision_score(actuals, predictions, average='weighted', zero_division=0),
            'recall': recall_score(actuals, predictions, average='weighted', zero_division=0),
            'f1_score': f1_score(actuals, predictions, average='weighted', zero_division=0)
        }
        
        # Add timestamp
        performance_record = {
            'timestamp': datetime.now(),
            'metrics': metrics,
            'sample_size': len(predictions)
        }
        
        self.performance_history.append(performance_record)
        
        # Check for performance degradation
        degradation_alert = self._check_performance_degradation(metrics)
        
        if degradation_alert:
            self.generate_alert('performance_degradation', degradation_alert, 'high')
        
        return {
            'current_metrics': metrics,
            'degradation_detected': degradation_alert is not None,
            'alert': degradation_alert
        }
    
    def detect_data_drift(self, reference_data: pd.DataFrame, 
                         current_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect data drift between reference and current data"""
        
        drift_results = {}
        
        # Check each feature for drift
        for column in reference_data.columns:
            if column in current_data.columns:
                drift_result = self._detect_feature_drift(
                    reference_data[column], 
                    current_data[column],
                    column
                )
                drift_results[column] = drift_result
        
        # Overall drift assessment
        drift_detected = any(result['drift_detected'] for result in drift_results.values())
        
        drift_record = {
            'timestamp': datetime.now(),
            'drift_detected': drift_detected,
            'feature_results': drift_results
        }
        
        self.drift_history.append(drift_record)
        
        if drift_detected:
            drift_features = [col for col, result in drift_results.items() if result['drift_detected']]
            self.generate_alert(
                'data_drift', 
                f"Data drift detected in features: {', '.join(drift_features)}", 
                'medium'
            )
        
        return drift_record
    
    def _detect_feature_drift(self, reference_data: pd.Series, current_data: pd.Series, 
                            feature_name: str) -> Dict[str, Any]:
        """Detect drift for a single feature"""
        
        # Remove NaN values
        ref_clean = reference_data.dropna()
        curr_clean = current_data.dropna()
        
        if len(ref_clean) == 0 or len(curr_clean) == 0:
            return {
                'drift_detected': False,
                'test': 'insufficient_data',
                'p_value': None,
                'statistic': None
            }
        
        # Determine if feature is numerical or categorical
        if pd.api.types.is_numeric_dtype(reference_data):
            return self._detect_numerical_drift(ref_clean, curr_clean)
        else:
            return self._detect_categorical_drift(ref_clean, curr_clean)
    
    def _detect_numerical_drift(self, reference_data: pd.Series, current_data: pd.Series) -> Dict[str, Any]:
        """Detect drift in numerical features using KS test"""
        
        if SCIPY_AVAILABLE:
            statistic, p_value = ks_2samp(reference_data, current_data)
            
            return {
                'drift_detected': p_value < 0.05,
                'test': 'kolmogorov_smirnov',
                'p_value': p_value,
                'statistic': statistic
            }
        else:
            # Simplified drift detection without scipy
            ref_mean = reference_data.mean()
            curr_mean = current_data.mean()
            ref_std = reference_data.std()
            
            # Simple threshold-based detection
            drift_detected = abs(curr_mean - ref_mean) > 2 * ref_std
            
            return {
                'drift_detected': drift_detected,
                'test': 'mean_shift',
                'reference_mean': ref_mean,
                'current_mean': curr_mean,
                'threshold': 2 * ref_std
            }
    
    def _detect_categorical_drift(self, reference_data: pd.Series, current_data: pd.Series) -> Dict[str, Any]:
        """Detect drift in categorical features"""
        
        # Get value counts
        ref_counts = reference_data.value_counts(normalize=True)
        curr_counts = current_data.value_counts(normalize=True)
        
        # Align indices
        all_categories = set(ref_counts.index) | set(curr_counts.index)
        ref_aligned = ref_counts.reindex(all_categories, fill_value=0)
        curr_aligned = curr_counts.reindex(all_categories, fill_value=0)
        
        if SCIPY_AVAILABLE and len(all_categories) > 1:
            # Chi-square test
            observed = np.array([curr_aligned.values, ref_aligned.values])
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(observed)
                
                return {
                    'drift_detected': p_value < 0.05,
                    'test': 'chi_square',
                    'p_value': p_value,
                    'statistic': chi2
                }
            except ValueError:
                # Fallback for edge cases
                pass
        
        # Simple distribution comparison
        max_diff = max(abs(curr_aligned - ref_aligned))
        drift_detected = max_diff > 0.1  # 10% threshold
        
        return {
            'drift_detected': drift_detected,
            'test': 'distribution_comparison',
            'max_difference': max_diff,
            'threshold': 0.1
        }
    
    def _check_performance_degradation(self, current_metrics: Dict[str, float]) -> Optional[str]:
        """Check for performance degradation"""
        
        if len(self.performance_history) < 2:
            return None
        
        # Get recent performance
        recent_performance = [record['metrics'] for record in self.performance_history[-5:]]
        
        # Calculate average recent performance
        avg_recent = {}
        for metric in current_metrics.keys():
            values = [perf[metric] for perf in recent_performance if metric in perf]
            if values:
                avg_recent[metric] = np.mean(values)
        
        # Compare with baseline (first few records)
        baseline_performance = [record['metrics'] for record in self.performance_history[:5]]
        
        if not baseline_performance:
            return None
        
        avg_baseline = {}
        for metric in current_metrics.keys():
            values = [perf[metric] for perf in baseline_performance if metric in perf]
            if values:
                avg_baseline[metric] = np.mean(values)
        
        # Check for significant degradation
        degradation_threshold = 0.05  # 5% degradation threshold
        
        for metric in ['accuracy', 'f1_score']:
            if metric in avg_recent and metric in avg_baseline:
                degradation = avg_baseline[metric] - avg_recent[metric]
                
                if degradation > degradation_threshold:
                    return f"{metric} degraded by {degradation:.3f} ({degradation/avg_baseline[metric]*100:.1f}%)"
        
        return None
    
    def generate_alert(self, alert_type: str, message: str, severity: str = "medium"):
        """Generate performance alert"""
        
        alert = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(),
            'model_name': self.model_name,
            'alert_type': alert_type,
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        
        print(f"🚨 ALERT [{severity.upper()}]: {message}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not self.performance_history:
            return {'error': 'No performance history available'}
        
        # Convert to DataFrame for analysis
        metrics_data = []
        for record in self.performance_history:
            row = {'timestamp': record['timestamp']}
            row.update(record['metrics'])
            metrics_data.append(row)
        
        df = pd.DataFrame(metrics_data)
        
        # Calculate summary statistics
        summary = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            if metric in df.columns:
                summary[metric] = {
                    'mean': df[metric].mean(),
                    'std': df[metric].std(),
                    'min': df[metric].min(),
                    'max': df[metric].max(),
                    'latest': df[metric].iloc[-1] if len(df) > 0 else None
                }
        
        # Recent alerts
        recent_alerts = [
            alert for alert in self.alerts 
            if alert['timestamp'] > datetime.now() - timedelta(days=7)
        ]
        
        return {
            'model_name': self.model_name,
            'monitoring_period': {
                'start': df['timestamp'].min().isoformat() if len(df) > 0 else None,
                'end': df['timestamp'].max().isoformat() if len(df) > 0 else None,
                'total_records': len(df)
            },
            'performance_summary': summary,
            'recent_alerts': len(recent_alerts),
            'drift_detections': len(self.drift_history),
            'alerts': recent_alerts[-5:]  # Last 5 alerts
        }

# =============================================================================
# HEALTHCARE ML PIPELINE
# =============================================================================

class HealthTechMLPipeline:
    """Complete ML pipeline for healthcare diagnostics"""
    
    def __init__(self, project_root: str = "./healthtech_ml"):
        self.project_root = Path(project_root)
        self.project_root.mkdir(exist_ok=True)
        
        # Initialize components
        self.dvc_manager = DVCPipelineManager(str(self.project_root))
        self.mlflow_manager = MLflowModelManager()
        self.lineage_tracker = DataLineageTracker()
        self.deployment_manager = ModelDeploymentManager(self.mlflow_manager)
        self.performance_monitor = ModelPerformanceMonitor("diagnostic_model")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_project(self):
        """Set up the ML project structure"""
        
        print("Setting up HealthTech ML project structure...")
        
        # Create directory structure
        directories = [
            "data/raw",
            "data/processed", 
            "data/features",
            "models",
            "metrics",
            "plots",
            "src",
            "tests",
            "deployments/staging",
            "deployments/production",
            "configs"
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize DVC
        os.chdir(self.project_root)
        self.dvc_manager.initialize_dvc_project()
        
        # Create parameter file
        params = {
            'data': {
                'n_samples': 10000,
                'test_size': 0.2,
                'random_state': 42
            },
            'features': {
                'n_features': 20,
                'scale_features': True
            },
            'model': {
                'algorithm': 'random_forest',
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5
            },
            'evaluation': {
                'cv_folds': 5,
                'metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            }
        }
        
        with open(self.project_root / "params.yaml", 'w') as f:
            yaml.dump(params, f, default_flow_style=False)
        
        print("✅ Project structure created")
    
    def create_pipeline(self):
        """Create the complete ML pipeline"""
        
        print("Creating DVC pipeline...")
        
        # Stage 1: Data generation
        self.dvc_manager.create_pipeline_stage(
            stage_name="generate_data",
            command="python src/generate_data.py",
            dependencies=["src/generate_data.py"],
            outputs=["data/raw/healthcare_data.csv"],
            parameters=["data.n_samples", "data.random_state"]
        )
        
        # Stage 2: Data preprocessing
        self.dvc_manager.create_pipeline_stage(
            stage_name="preprocess_data",
            command="python src/preprocess_data.py",
            dependencies=["src/preprocess_data.py", "data/raw/healthcare_data.csv"],
            outputs=["data/processed/clean_data.csv"],
            parameters=["data.test_size"]
        )
        
        # Stage 3: Feature engineering
        self.dvc_manager.create_pipeline_stage(
            stage_name="feature_engineering",
            command="python src/feature_engineering.py",
            dependencies=["src/feature_engineering.py", "data/processed/clean_data.csv"],
            outputs=["data/features/features.csv"],
            parameters=["features.n_features", "features.scale_features"]
        )
        
        # Stage 4: Model training
        self.dvc_manager.create_pipeline_stage(
            stage_name="train_model",
            command="python src/train_model.py",
            dependencies=["src/train_model.py", "data/features/features.csv"],
            outputs=["models/model.pkl"],
            metrics=["metrics/train_metrics.json"],
            parameters=["model.algorithm", "model.n_estimators", "model.max_depth"]
        )
        
        # Stage 5: Model evaluation
        self.dvc_manager.create_pipeline_stage(
            stage_name="evaluate_model",
            command="python src/evaluate_model.py",
            dependencies=["src/evaluate_model.py", "models/model.pkl", "data/features/features.csv"],
            metrics=["metrics/eval_metrics.json"],
            outputs=["plots/confusion_matrix.png", "plots/roc_curve.png"]
        )
        
        print("✅ DVC pipeline created")
    
    def generate_synthetic_data(self, n_samples: int = 10000):
        """Generate synthetic healthcare data for demonstration"""
        
        print(f"Generating {n_samples} synthetic healthcare records...")
        
        # Create realistic healthcare features
        np.random.seed(42)
        
        # Patient demographics
        age = np.random.normal(50, 15, n_samples).clip(18, 90)
        gender = np.random.choice([0, 1], n_samples)  # 0: Female, 1: Male
        
        # Vital signs
        systolic_bp = np.random.normal(120, 20, n_samples).clip(80, 200)
        diastolic_bp = np.random.normal(80, 15, n_samples).clip(50, 120)
        heart_rate = np.random.normal(70, 15, n_samples).clip(40, 150)
        temperature = np.random.normal(98.6, 1.5, n_samples).clip(95, 105)
        
        # Lab values
        glucose = np.random.lognormal(4.5, 0.3, n_samples).clip(70, 400)
        cholesterol = np.random.normal(200, 40, n_samples).clip(100, 400)
        hemoglobin = np.random.normal(14, 2, n_samples).clip(8, 20)
        
        # Symptoms (binary)
        chest_pain = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        shortness_breath = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        fatigue = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])
        
        # Risk factors
        smoking = np.random.choice([0, 1], n_samples, p=[0.75, 0.25])
        diabetes = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        family_history = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
        
        # Generate target variable (cardiovascular disease risk)
        # Higher risk with age, high BP, symptoms, risk factors
        risk_score = (
            0.02 * age +
            0.01 * systolic_bp +
            0.005 * glucose +
            0.3 * chest_pain +
            0.2 * shortness_breath +
            0.15 * smoking +
            0.25 * diabetes +
            0.1 * family_history +
            np.random.normal(0, 0.5, n_samples)
        )
        
        # Convert to binary classification
        target = (risk_score > np.percentile(risk_score, 70)).astype(int)  # Top 30% as high risk
        
        # Create DataFrame
        data = pd.DataFrame({
            'patient_id': range(n_samples),
            'age': age,
            'gender': gender,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'heart_rate': heart_rate,
            'temperature': temperature,
            'glucose': glucose,
            'cholesterol': cholesterol,
            'hemoglobin': hemoglobin,
            'chest_pain': chest_pain,
            'shortness_breath': shortness_breath,
            'fatigue': fatigue,
            'smoking': smoking,
            'diabetes': diabetes,
            'family_history': family_history,
            'cardiovascular_risk': target
        })
        
        # Save data
        data_path = self.project_root / "data/raw/healthcare_data.csv"
        data.to_csv(data_path, index=False)
        
        # Track in DVC
        self.dvc_manager.add_data_file(str(data_path))
        
        # Track lineage
        self.lineage_tracker.track_transformation(
            input_data="synthetic_generation",
            output_data=str(data_path),
            transformation_info={
                'name': 'synthetic_data_generation',
                'n_samples': n_samples,
                'features': list(data.columns),
                'target_distribution': data['cardiovascular_risk'].value_counts().to_dict()
            }
        )
        
        print(f"✅ Generated healthcare data: {data.shape}")
        print(f"   Target distribution: {data['cardiovascular_risk'].value_counts().to_dict()}")
        
        return data
    def run_training_pipeline(self) -> Dict[str, Any]:
        """Run the complete training pipeline"""
        
        print("Running ML training pipeline...")
        
        # Start MLflow run
        run_id = self.mlflow_manager.start_experiment_run("healthcare_pipeline_run")
        
        try:
            # Load parameters
            with open(self.project_root / "params.yaml", 'r') as f:
                params = yaml.safe_load(f)
            
            # Log parameters to MLflow
            self.mlflow_manager.log_parameters(params)
            
            # Run DVC pipeline
            pipeline_results = self.dvc_manager.run_pipeline()
            
            # Load and evaluate trained model
            model_path = self.project_root / "models/model.pkl"
            
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Load test data for evaluation
                data_path = self.project_root / "data/features/features.csv"
                if data_path.exists():
                    data = pd.read_csv(data_path)
                    X = data.drop('cardiovascular_risk', axis=1)
                    y = data['cardiovascular_risk']
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    metrics = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted'),
                        'recall': recall_score(y_test, y_pred, average='weighted'),
                        'f1_score': f1_score(y_test, y_pred, average='weighted')
                    }
                    
                    # Log metrics to MLflow
                    self.mlflow_manager.log_metrics(metrics)
                    
                    # Log model to MLflow
                    self.mlflow_manager.log_model(model, "model", "healthcare_diagnostic_model")
                    
                    print(f"✅ Model training completed:")
                    for metric, value in metrics.items():
                        print(f"   {metric}: {value:.4f}")
                    
                    return {
                        'status': 'success',
                        'run_id': run_id,
                        'metrics': metrics,
                        'pipeline_results': pipeline_results
                    }
            
            return {
                'status': 'failed',
                'error': 'Model file not found after pipeline execution'
            }
            
        except Exception as e:
            print(f"❌ Pipeline execution failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
        finally:
            self.mlflow_manager.end_run()
    
    def deploy_model_to_production(self, model_name: str, version: str) -> Dict[str, Any]:
        """Deploy model to production with full validation"""
        
        print(f"Deploying {model_name} v{version} to production...")
        
        try:
            # First deploy to staging
            staging_result = self.deployment_manager.deploy_model(
                model_name, version, "staging"
            )
            
            if staging_result['status'] != 'success':
                raise ValueError(f"Staging deployment failed: {staging_result}")
            
            # Run additional validation tests
            validation_results = self._run_production_validation(model_name)
            
            if not validation_results['passed']:
                raise ValueError(f"Production validation failed: {validation_results['errors']}")
            
            # Promote model to production stage in MLflow
            self.mlflow_manager.promote_model(model_name, version, "Production")
            
            # Deploy to production
            production_result = self.deployment_manager.deploy_model(
                model_name, version, "production"
            )
            
            print(f"✅ Successfully deployed {model_name} v{version} to production")
            
            return {
                'status': 'success',
                'staging_deployment': staging_result,
                'production_deployment': production_result,
                'validation_results': validation_results
            }
            
        except Exception as e:
            print(f"❌ Production deployment failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _run_production_validation(self, model_name: str) -> Dict[str, Any]:
        """Run comprehensive validation for production deployment"""
        
        validation_results = {
            'passed': True,
            'tests': {},
            'errors': []
        }
        
        try:
            # Load model from staging
            model = self.mlflow_manager.load_model(model_name, "Staging")
            
            # Test 1: Model performance validation
            test_data_path = self.project_root / "data/features/features.csv"
            if test_data_path.exists():
                data = pd.read_csv(test_data_path)
                X = data.drop('cardiovascular_risk', axis=1)
                y = data['cardiovascular_risk']
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                validation_results['tests']['performance_test'] = {
                    'passed': accuracy > 0.7,  # Minimum 70% accuracy
                    'accuracy': accuracy
                }
                
                if accuracy <= 0.7:
                    validation_results['errors'].append(f"Model accuracy {accuracy:.3f} below threshold 0.7")
            
            # Test 2: Prediction consistency
            test_input = np.random.random((10, X.shape[1]))
            pred1 = model.predict(test_input)
            pred2 = model.predict(test_input)
            
            consistency_passed = np.array_equal(pred1, pred2)
            validation_results['tests']['consistency_test'] = {
                'passed': consistency_passed
            }
            
            if not consistency_passed:
                validation_results['errors'].append("Model predictions are not consistent")
            
            # Test 3: Latency test
            start_time = time.time()
            for _ in range(100):
                model.predict(test_input[:1])
            avg_latency = (time.time() - start_time) / 100
            
            latency_passed = avg_latency < 0.1  # Less than 100ms
            validation_results['tests']['latency_test'] = {
                'passed': latency_passed,
                'avg_latency_ms': avg_latency * 1000
            }
            
            if not latency_passed:
                validation_results['errors'].append(f"Model latency {avg_latency*1000:.1f}ms exceeds 100ms threshold")
            
            # Overall validation result
            validation_results['passed'] = len(validation_results['errors']) == 0
            
        except Exception as e:
            validation_results['passed'] = False
            validation_results['errors'].append(f"Validation error: {e}")
        
        return validation_results
    
    def simulate_production_monitoring(self, days: int = 30):
        """Simulate production model monitoring"""
        
        print(f"Simulating {days} days of production monitoring...")
        
        # Generate reference data (training data distribution)
        reference_data = pd.read_csv(self.project_root / "data/features/features.csv")
        X_reference = reference_data.drop('cardiovascular_risk', axis=1)
        y_reference = reference_data['cardiovascular_risk']
        
        monitoring_results = []
        
        for day in range(days):
            print(f"Day {day + 1}/{days}")
            
            # Simulate daily production data with potential drift
            n_daily_samples = np.random.randint(100, 500)
            
            # Add gradual drift over time
            drift_factor = day / days * 0.2  # Up to 20% drift by end
            
            # Generate production data with drift
            production_data = self._generate_production_data(
                X_reference, n_daily_samples, drift_factor
            )
            
            # Simulate model predictions
            try:
                model = self.mlflow_manager.load_model("healthcare_diagnostic_model", "Production")
                predictions = model.predict(production_data)
                
                # Simulate actual outcomes (with some delay/noise)
                actual_outcomes = self._simulate_actual_outcomes(production_data, predictions)
                
                # Monitor performance
                performance_result = self.performance_monitor.monitor_performance(
                    predictions, actual_outcomes
                )
                
                # Detect data drift
                drift_result = self.performance_monitor.detect_data_drift(
                    X_reference, production_data
                )
                
                monitoring_results.append({
                    'day': day + 1,
                    'samples': n_daily_samples,
                    'performance': performance_result,
                    'drift': drift_result
                })
                
                # Simulate alerts
                if performance_result['degradation_detected']:
                    print(f"   🚨 Performance degradation detected on day {day + 1}")
                
                if drift_result['drift_detected']:
                    print(f"   🚨 Data drift detected on day {day + 1}")
                
            except Exception as e:
                print(f"   ❌ Monitoring error on day {day + 1}: {e}")
        
        print("✅ Production monitoring simulation completed")
        
        return monitoring_results
    
    def _generate_production_data(self, reference_data: pd.DataFrame, 
                                n_samples: int, drift_factor: float) -> pd.DataFrame:
        """Generate production data with potential drift"""
        
        production_data = reference_data.sample(n_samples, replace=True).copy()
        
        # Add drift to some features
        drift_features = ['age', 'systolic_bp', 'glucose']
        
        for feature in drift_features:
            if feature in production_data.columns:
                # Add systematic shift
                shift = drift_factor * production_data[feature].std()
                production_data[feature] += shift
                
                # Add noise
                noise = np.random.normal(0, 0.1 * production_data[feature].std(), n_samples)
                production_data[feature] += noise
        
        return production_data
    
    def _simulate_actual_outcomes(self, production_data: pd.DataFrame, 
                                predictions: np.array) -> np.array:
        """Simulate actual outcomes for monitoring"""
        
        # Simulate outcomes based on predictions with some noise
        # In reality, these would come from follow-up medical records
        
        actual_outcomes = predictions.copy()
        
        # Add some noise to simulate real-world variability
        flip_probability = 0.1  # 10% of predictions are "wrong"
        flip_mask = np.random.random(len(predictions)) < flip_probability
        actual_outcomes[flip_mask] = 1 - actual_outcomes[flip_mask]
        
        return actual_outcomes
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate regulatory compliance report"""
        
        print("Generating regulatory compliance report...")
        
        report = {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'model_name': 'healthcare_diagnostic_model',
            'compliance_framework': 'FDA_AI_ML_Guidance',
            'sections': {}
        }
        
        # Section 1: Data Lineage
        lineage_data = []
        for transformation in self.lineage_tracker.transformations:
            lineage_data.append({
                'transformation_id': transformation['id'],
                'input_data': transformation['input_data'],
                'output_data': transformation['output_data'],
                'transformation_name': transformation['transformation_info'].get('name'),
                'timestamp': transformation['timestamp'].isoformat()
            })
        
        report['sections']['data_lineage'] = {
            'description': 'Complete data transformation history',
            'transformations': lineage_data
        }
        
        # Section 2: Model Versions
        model_versions = self.mlflow_manager.get_model_versions('healthcare_diagnostic_model')
        
        report['sections']['model_versions'] = {
            'description': 'All model versions and their stages',
            'versions': [
                {
                    'version': v.get('version', 'unknown'),
                    'stage': v.get('stage', 'unknown'),
                    'created_at': v.get('created_at', datetime.now()).isoformat() if hasattr(v.get('created_at', datetime.now()), 'isoformat') else str(v.get('created_at'))
                } for v in model_versions
            ]
        }
        
        # Section 3: Deployment History
        deployment_history = self.deployment_manager.get_deployment_history('healthcare_diagnostic_model')
        
        report['sections']['deployment_history'] = {
            'description': 'Model deployment and rollback history',
            'deployments': deployment_history
        }
        
        # Section 4: Performance Monitoring
        performance_report = self.performance_monitor.get_performance_report()
        
        report['sections']['performance_monitoring'] = {
            'description': 'Model performance monitoring and alerts',
            'summary': performance_report
        }
        
        # Section 5: Risk Assessment
        report['sections']['risk_assessment'] = {
            'description': 'AI/ML risk assessment for healthcare application',
            'risk_factors': [
                {
                    'factor': 'Data Quality',
                    'risk_level': 'Medium',
                    'mitigation': 'Automated data validation and drift detection'
                },
                {
                    'factor': 'Model Performance',
                    'risk_level': 'Low',
                    'mitigation': 'Continuous monitoring and automated alerts'
                },
                {
                    'factor': 'Regulatory Compliance',
                    'risk_level': 'Low',
                    'mitigation': 'Complete audit trail and documentation'
                }
            ]
        }
        
        # Save report
        report_path = self.project_root / f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"✅ Compliance report generated: {report_path}")
        
        return report

# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def main():
    """
    Main function demonstrating model versioning with DVC and MLflow
    """
    print("Day 35: Model Versioning with DVC - Complete Solution")
    print("=" * 70)
    print("HealthTech Analytics - Healthcare AI Model Versioning System")
    print("=" * 70)
    
    # Initialize HealthTech ML Pipeline
    print("\n1. INITIALIZING HEALTHTECH ML PIPELINE")
    print("-" * 40)
    
    pipeline = HealthTechMLPipeline()
    print("✅ ML pipeline initialized")
    
    # Set up project structure
    print("\n2. SETTING UP PROJECT STRUCTURE")
    print("-" * 40)
    
    pipeline.setup_project()
    
    # Generate synthetic healthcare data
    print("\n3. GENERATING SYNTHETIC HEALTHCARE DATA")
    print("-" * 40)
    
    data = pipeline.generate_synthetic_data(n_samples=5000)
    
    # Create DVC pipeline
    print("\n4. CREATING DVC PIPELINE")
    print("-" * 40)
    
    pipeline.create_pipeline()
    
    # Run training pipeline
    print("\n5. RUNNING TRAINING PIPELINE")
    print("-" * 40)
    
    training_results = pipeline.run_training_pipeline()
    
    if training_results['status'] == 'success':
        print("✅ Training pipeline completed successfully")
        
        # Register model in MLflow
        print("\n6. REGISTERING MODEL IN MLFLOW")
        print("-" * 40)
        
        run_id = training_results['run_id']
        model_version = pipeline.mlflow_manager.register_model(
            run_id, "healthcare_diagnostic_model", "Staging"
        )
        
        print(f"✅ Model registered: version {model_version.get('version', '1')}")
        
        # Deploy model to staging
        print("\n7. DEPLOYING MODEL TO STAGING")
        print("-" * 40)
        
        staging_deployment = pipeline.deployment_manager.deploy_model(
            "healthcare_diagnostic_model", 
            model_version.get('version', '1'), 
            "staging"
        )
        
        # Deploy to production
        print("\n8. DEPLOYING TO PRODUCTION")
        print("-" * 40)
        
        production_deployment = pipeline.deploy_model_to_production(
            "healthcare_diagnostic_model", 
            model_version.get('version', '1')
        )
        
        if production_deployment['status'] == 'success':
            print("✅ Production deployment successful")
            
            # Simulate production monitoring
            print("\n9. SIMULATING PRODUCTION MONITORING")
            print("-" * 40)
            
            monitoring_results = pipeline.simulate_production_monitoring(days=7)
            
            # Generate compliance report
            print("\n10. GENERATING COMPLIANCE REPORT")
            print("-" * 40)
            
            compliance_report = pipeline.generate_compliance_report()
            
            # Final summary
            print("\n" + "=" * 70)
            print("MODEL VERSIONING SYSTEM DEPLOYMENT COMPLETE!")
            print("=" * 70)
            
            print("\n🎯 SYSTEM CAPABILITIES:")
            print("   ✅ Reproducible ML pipelines with DVC")
            print("   ✅ Model registry with MLflow and stage transitions")
            print("   ✅ Complete data lineage tracking")
            print("   ✅ Blue-green deployment with automated validation")
            print("   ✅ Real-time performance monitoring and drift detection")
            print("   ✅ Regulatory compliance reporting")
            print("   ✅ Automated rollback capabilities")
            
            print("\n📊 TRAINING RESULTS:")
            metrics = training_results.get('metrics', {})
            for metric, value in metrics.items():
                print(f"   • {metric}: {value:.4f}")
            
            print("\n🚀 DEPLOYMENT STATUS:")
            print(f"   • Staging: {staging_deployment.get('status', 'unknown')}")
            print(f"   • Production: {production_deployment.get('status', 'unknown')}")
            
            print("\n📈 MONITORING SUMMARY:")
            alerts_count = len(pipeline.performance_monitor.alerts)
            drift_count = len(pipeline.performance_monitor.drift_history)
            print(f"   • Performance alerts: {alerts_count}")
            print(f"   • Drift detections: {drift_count}")
            print(f"   • Monitoring days: {len(monitoring_results)}")
            
            print("\n📋 COMPLIANCE:")
            print(f"   • Report ID: {compliance_report['report_id']}")
            print(f"   • Framework: {compliance_report['compliance_framework']}")
            print(f"   • Sections: {len(compliance_report['sections'])}")
            
            return {
                'training_results': training_results,
                'deployment_results': {
                    'staging': staging_deployment,
                    'production': production_deployment
                },
                'monitoring_results': monitoring_results,
                'compliance_report': compliance_report
            }
        
        else:
            print("❌ Production deployment failed")
            return {'error': 'Production deployment failed'}
    
    else:
        print("❌ Training pipeline failed")
        return {'error': 'Training pipeline failed'}

if __name__ == "__main__":
    # Run the complete demonstration
    results = main()
    
    print(f"\n🎉 Day 35 Complete! Model versioning system successfully implemented.")
    print(f"📚 Key learnings: DVC pipelines, MLflow registry, deployment automation, compliance")
    print(f"🔄 Next: Day 36 - CI/CD for ML with automated testing and infrastructure as code")