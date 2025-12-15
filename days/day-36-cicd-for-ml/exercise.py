"""
Day 36: CI/CD for ML - Exercise

Business Scenario:
You're the MLOps Engineer at FinTech Solutions Inc. The company processes millions of 
financial transactions daily and needs a robust CI/CD pipeline for their fraud detection 
models. The current manual deployment process is error-prone and doesn't scale with the 
business needs.

Your task is to build a comprehensive CI/CD pipeline that automates model training, 
testing, deployment, and monitoring while ensuring zero-downtime deployments and 
automatic rollback capabilities.

Requirements:
1. Create automated ML pipeline with GitHub Actions
2. Implement comprehensive testing strategy (unit, integration, performance)
3. Build blue-green deployment system with health checks
4. Set up monitoring and alerting for model performance
5. Create infrastructure as code with Terraform

Success Criteria:
- Pipeline automatically triggers on code/data changes
- All tests pass before deployment
- Zero-downtime deployments with rollback capability
- Real-time monitoring with automated alerts
- Infrastructure is reproducible and version-controlled
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import boto3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

class MLCICDPipeline:
    """
    Complete CI/CD pipeline for ML models
    """
    
    def __init__(self, config_path="config/pipeline_config.json"):
        self.config_path = Path(config_path)
        self.load_configuration()
        
        # Initialize AWS clients (mock for exercise)
        self.ecs_client = None  # boto3.client('ecs') in production
        self.elbv2_client = None  # boto3.client('elbv2') in production
        self.cloudwatch = None  # boto3.client('cloudwatch') in production
    
    def load_configuration(self):
        """Load pipeline configuration"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default configuration
            self.config = {
                "aws_region": "us-west-2",
                "model_performance_thresholds": {
                    "min_accuracy": 0.85,
                    "min_precision": 0.80,
                    "min_recall": 0.80,
                    "min_f1_score": 0.80
                },
                "data_quality_thresholds": {
                    "max_missing_percentage": 0.05,
                    "max_duplicate_percentage": 0.02
                }
            }
    
    def validate_data_quality(self, data_path):
        """
        Validate incoming data quality
        
        Args:
            data_path: Path to data file
            
        Returns:
            dict: Validation results with pass/fail status
        """
        try:
            # Load data
            data = pd.read_csv(data_path)
            
            validation_results = {
                "status": "passed",
                "checks": {},
                "errors": [],
                "warnings": []
            }
            
            # Check 1: Data not empty
            if data.empty:
                validation_results["errors"].append("Data is empty")
                validation_results["status"] = "failed"
            else:
                validation_results["checks"]["data_not_empty"] = True
            
            # Check 2: Required columns present
            required_columns = ['amount', 'merchant_category_encoded', 'hour', 'user_age', 'account_balance', 'is_fraud']
            missing_columns = set(required_columns) - set(data.columns)
            
            if missing_columns:
                validation_results["errors"].append(f"Missing required columns: {missing_columns}")
                validation_results["status"] = "failed"
            else:
                validation_results["checks"]["required_columns_present"] = True
            
            # Check 3: Missing values threshold
            max_missing = self.config["data_quality_thresholds"]["max_missing_percentage"]
            
            for column in data.columns:
                missing_percentage = data[column].isnull().sum() / len(data)
                
                if missing_percentage > max_missing:
                    validation_results["errors"].append(
                        f"Column {column} has {missing_percentage:.2%} missing values (threshold: {max_missing:.2%})"
                    )
                    validation_results["status"] = "failed"
            
            # Check 4: Duplicate records
            max_duplicates = self.config["data_quality_thresholds"]["max_duplicate_percentage"]
            duplicate_percentage = data.duplicated().sum() / len(data)
            
            if duplicate_percentage > max_duplicates:
                validation_results["warnings"].append(
                    f"Data has {duplicate_percentage:.2%} duplicates (threshold: {max_duplicates:.2%})"
                )
            
            return validation_results
            
        except Exception as e:
            return {
                "status": "failed",
                "checks": {},
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    def run_model_tests(self, model_path, test_data_path):
        """
        Run comprehensive model tests
        
        Args:
            model_path: Path to trained model
            test_data_path: Path to test dataset
            
        Returns:
            dict: Test results with pass/fail status
        """
        try:
            # Load model and test data
            model = joblib.load(model_path)
            test_data = pd.read_csv(test_data_path)
            
            X_test = test_data.drop('is_fraud', axis=1)
            y_test = test_data['is_fraud']
            
            test_results = {
                "status": "passed",
                "tests": {},
                "errors": [],
                "metrics": {}
            }
            
            # Test 1: Model loading
            if model is None:
                test_results["errors"].append("Model failed to load")
                test_results["status"] = "failed"
            else:
                test_results["tests"]["model_loading"] = True
            
            # Test 2: Required methods
            required_methods = ['predict', 'predict_proba']
            for method in required_methods:
                if not hasattr(model, method):
                    test_results["errors"].append(f"Model missing {method} method")
                    test_results["status"] = "failed"
            
            # Test 3: Prediction functionality
            try:
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)
                
                # Check shapes
                if len(predictions) != len(X_test):
                    test_results["errors"].append("Prediction length mismatch")
                    test_results["status"] = "failed"
                
                # Check probability ranges
                if not np.all((probabilities >= 0) & (probabilities <= 1)):
                    test_results["errors"].append("Probabilities outside [0,1] range")
                    test_results["status"] = "failed"
                
            except Exception as e:
                test_results["errors"].append(f"Prediction test failed: {str(e)}")
                test_results["status"] = "failed"
            
            # Test 4: Performance thresholds
            if test_results["status"] != "failed":
                try:
                    predictions = model.predict(X_test)
                    
                    accuracy = accuracy_score(y_test, predictions)
                    precision = precision_score(y_test, predictions, average='weighted')
                    recall = recall_score(y_test, predictions, average='weighted')
                    f1 = f1_score(y_test, predictions, average='weighted')
                    
                    test_results["metrics"] = {
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "f1_score": f1
                    }
                    
                    thresholds = self.config["model_performance_thresholds"]
                    
                    if accuracy < thresholds["min_accuracy"]:
                        test_results["errors"].append(f"Accuracy {accuracy:.3f} below threshold {thresholds['min_accuracy']}")
                        test_results["status"] = "failed"
                    
                except Exception as e:
                    test_results["errors"].append(f"Performance test failed: {str(e)}")
                    test_results["status"] = "failed"
            
            return test_results
            
        except Exception as e:
            return {
                "status": "failed",
                "tests": {},
                "errors": [f"Test error: {str(e)}"],
                "metrics": {}
            }
    
    def deploy_model_blue_green(self, model_path, deployment_config):
        """
        Deploy model using blue-green strategy
        
        Args:
            model_path: Path to model to deploy
            deployment_config: Deployment configuration
            
        Returns:
            dict: Deployment results
        """
        deployment_id = f"deploy-{int(datetime.now().timestamp())}"
        
        try:
            deployment_result = {
                "deployment_id": deployment_id,
                "status": "in_progress",
                "steps": {},
                "errors": []
            }
            
            # Step 1: Create green service (simulated)
            print(f"Creating green service for deployment {deployment_id}...")
            deployment_result["steps"]["green_service_created"] = True
            
            # Step 2: Health checks (simulated)
            print("Running health checks on green service...")
            health_check_passed = True  # Simulate successful health check
            
            if not health_check_passed:
                deployment_result["errors"].append("Health checks failed")
                deployment_result["status"] = "failed"
                return deployment_result
            
            deployment_result["steps"]["health_checks_passed"] = True
            
            # Step 3: Switch traffic (simulated)
            print("Switching traffic to green service...")
            deployment_result["steps"]["traffic_switched"] = True
            
            # Step 4: Monitor performance (simulated)
            print("Monitoring green service performance...")
            performance_ok = True  # Simulate successful monitoring
            
            if not performance_ok:
                print("Performance issues detected, rolling back...")
                deployment_result["errors"].append("Performance monitoring failed")
                deployment_result["status"] = "rolled_back"
                return deployment_result
            
            deployment_result["steps"]["performance_monitoring_passed"] = True
            
            # Step 5: Cleanup old service
            print("Cleaning up old blue service...")
            deployment_result["steps"]["old_service_cleaned"] = True
            
            deployment_result["status"] = "success"
            deployment_result["timestamp"] = datetime.now().isoformat()
            
            return deployment_result
            
        except Exception as e:
            return {
                "deployment_id": deployment_id,
                "status": "failed",
                "steps": deployment_result.get("steps", {}),
                "errors": [f"Deployment error: {str(e)}"],
                "timestamp": datetime.now().isoformat()
            }
    
    def monitor_model_performance(self, model_endpoint, duration_minutes=30):
        """
        Monitor deployed model performance
        
        Args:
            model_endpoint: Model service endpoint
            duration_minutes: Monitoring duration
            
        Returns:
            dict: Performance metrics
        """
        try:
            print(f"Monitoring model performance for {duration_minutes} minutes...")
            
            # Simulate performance monitoring
            metrics = {
                'total_requests': duration_minutes * 10,  # 10 requests per minute
                'successful_requests': int(duration_minutes * 10 * 0.98),  # 98% success rate
                'failed_requests': int(duration_minutes * 10 * 0.02),  # 2% failure rate
                'avg_response_time': 0.45,  # 450ms average
                'p95_response_time': 0.8,   # 800ms 95th percentile
                'error_rate': 0.02,         # 2% error rate
                'monitoring_duration': duration_minutes,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Performance monitoring completed:")
            print(f"  - Error rate: {metrics['error_rate']:.2%}")
            print(f"  - Avg response time: {metrics['avg_response_time']:.3f}s")
            print(f"  - Total requests: {metrics['total_requests']}")
            
            return metrics
            
        except Exception as e:
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def setup_infrastructure(self, environment="staging"):
        """
        Set up infrastructure using Terraform
        
        Args:
            environment: Target environment (staging/production)
            
        Returns:
            dict: Infrastructure setup results
        """
        try:
            print(f"Setting up infrastructure for environment: {environment}")
            
            # Simulate Terraform operations
            print("Generating Terraform configuration...")
            print("Initializing Terraform...")
            print("Planning infrastructure changes...")
            print("Applying infrastructure...")
            
            # Simulate successful infrastructure creation
            result = {
                'status': 'success',
                'environment': environment,
                'resources_created': [
                    'VPC',
                    'Subnets',
                    'Internet Gateway',
                    'ECS Cluster',
                    'S3 Buckets',
                    'Security Groups'
                ],
                'outputs': {
                    'vpc_id': f'vpc-{environment}-12345678',
                    'ecs_cluster_name': f'fraud-detection-{environment}-cluster',
                    's3_artifacts_bucket': f'fraud-detection-{environment}-artifacts-abcd1234'
                },
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Infrastructure setup completed successfully")
            print(f"  - VPC ID: {result['outputs']['vpc_id']}")
            print(f"  - ECS Cluster: {result['outputs']['ecs_cluster_name']}")
            print(f"  - S3 Bucket: {result['outputs']['s3_artifacts_bucket']}")
            
            return result
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def create_monitoring_dashboard(self, pipeline_name):
        """
        Create CloudWatch monitoring dashboard
        
        Args:
            pipeline_name: Name of the ML pipeline
            
        Returns:
            dict: Dashboard creation results
        """
        try:
            print(f"Creating monitoring dashboard for: {pipeline_name}")
            
            # Simulate dashboard creation
            dashboard_name = f'{pipeline_name}-MLPipeline-Dashboard'
            
            # Simulate alarm creation
            alarms_created = [
                f'{pipeline_name}-ModelAccuracy-Low',
                f'{pipeline_name}-PredictionLatency-High',
                f'{pipeline_name}-ErrorRate-High',
                f'{pipeline_name}-DataDrift-High'
            ]
            
            result = {
                'status': 'success',
                'dashboard_name': dashboard_name,
                'dashboard_url': f'https://console.aws.amazon.com/cloudwatch/home?region=us-west-2#dashboards:name={dashboard_name}',
                'alarms_created': alarms_created,
                'metrics_configured': [
                    'ModelAccuracy',
                    'PredictionLatency', 
                    'ErrorRate',
                    'DataDriftScore',
                    'ThroughputRPS'
                ],
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"Monitoring dashboard created successfully")
            print(f"  - Dashboard: {dashboard_name}")
            print(f"  - Alarms created: {len(alarms_created)}")
            print(f"  - Metrics configured: {len(result['metrics_configured'])}")
            
            return result
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

class ModelTestSuite:
    """
    Comprehensive test suite for ML models
    """
    
    def __init__(self, model_path, test_data_path, config_path=None):
        self.model_path = model_path
        self.test_data_path = test_data_path
        
        # Load model and test data
        self.model = joblib.load(model_path)
        self.test_data = pd.read_csv(test_data_path)
        
        # Default configuration if not provided
        self.config = {
            "min_accuracy": 0.8,
            "min_precision": 0.8,
            "min_recall": 0.8,
            "min_f1_score": 0.8
        }
    
    def test_model_loading(self):
        """Test that model loads correctly and has required methods"""
        assert self.model is not None, "Model failed to load"
        assert hasattr(self.model, 'predict'), "Model missing predict method"
        assert hasattr(self.model, 'predict_proba'), "Model missing predict_proba method"
        return True
    
    def test_prediction_functionality(self):
        """Test model prediction functionality"""
        X_test = self.test_data.drop('is_fraud', axis=1)
        
        # Test predictions
        predictions = self.model.predict(X_test)
        probabilities = self.model.predict_proba(X_test)
        
        # Check shapes
        assert len(predictions) == len(X_test), "Prediction length mismatch"
        assert probabilities.shape[0] == len(X_test), "Probability shape mismatch"
        
        # Check data types
        assert isinstance(predictions, np.ndarray), "Predictions not numpy array"
        assert isinstance(probabilities, np.ndarray), "Probabilities not numpy array"
        
        # Check probability ranges
        assert np.all((probabilities >= 0) & (probabilities <= 1)), "Probabilities outside [0,1] range"
        
        return True
    
    def test_performance_thresholds(self):
        """Test model meets minimum performance requirements"""
        X_test = self.test_data.drop('is_fraud', axis=1)
        y_test = self.test_data['is_fraud']
        
        predictions = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average='weighted')
        recall = recall_score(y_test, predictions, average='weighted')
        f1 = f1_score(y_test, predictions, average='weighted')
        
        assert accuracy >= self.config["min_accuracy"], f"Accuracy {accuracy:.3f} below threshold"
        assert precision >= self.config["min_precision"], f"Precision {precision:.3f} below threshold"
        assert recall >= self.config["min_recall"], f"Recall {recall:.3f} below threshold"
        assert f1 >= self.config["min_f1_score"], f"F1-score {f1:.3f} below threshold"
        
        return True
    
    def test_model_robustness(self):
        """Test model robustness to input variations"""
        X_test = self.test_data.drop('is_fraud', axis=1)
        sample_input = X_test.iloc[:10]
        
        # Test consistency
        predictions_1 = self.model.predict(sample_input)
        predictions_2 = self.model.predict(sample_input)
        
        assert np.array_equal(predictions_1, predictions_2), "Model predictions not consistent"
        
        # Test with small noise
        noisy_input = sample_input.copy()
        noise = np.random.normal(0, 0.01, noisy_input.shape)
        noisy_input += noise
        
        noisy_predictions = self.model.predict(noisy_input)
        
        # Most predictions should remain the same with small noise
        stability_ratio = np.mean(predictions_1 == noisy_predictions)
        assert stability_ratio >= 0.7, f"Model not robust to noise: {stability_ratio:.2f}"
        
        return True

class DeploymentManager:
    """
    Handles model deployment strategies
    """
    
    def __init__(self, aws_region="us-west-2"):
        # TODO: Initialize AWS clients
        # HINT: Use boto3 for ECS, ELB, and CloudWatch
        pass
    
    def create_green_service(self, service_config):
        """
        TODO: Create new green service for blue-green deployment
        - ECS service creation
        - Task definition setup
        - Load balancer configuration
        """
        pass
    
    def validate_service_health(self, service_endpoint):
        """
        TODO: Validate service health before traffic switch
        - Health endpoint checks
        - Prediction functionality tests
        - Performance validation
        """
        pass
    
    def switch_traffic(self, target_group_arn, green_service_targets):
        """
        TODO: Switch traffic from blue to green service
        - Register new targets
        - Deregister old targets
        - Monitor traffic switch
        """
        pass
    
    def rollback_deployment(self, blue_service_targets, target_group_arn):
        """
        TODO: Rollback to previous version on failure
        - Switch traffic back to blue
        - Cleanup failed green service
        - Log rollback reason
        """
        pass

class InfrastructureManager:
    """
    Manages infrastructure as code with Terraform
    """
    
    def __init__(self, terraform_dir="infrastructure"):
        # TODO: Initialize Terraform configuration directory
        pass
    
    def generate_terraform_config(self, environment, config):
        """
        TODO: Generate Terraform configuration files
        - VPC and networking
        - ECS cluster
        - Load balancer
        - S3 buckets
        - RDS instance
        """
        pass
    
    def apply_infrastructure(self, environment):
        """
        TODO: Apply Terraform configuration
        - terraform init
        - terraform plan
        - terraform apply
        """
        pass
    
    def destroy_infrastructure(self, environment):
        """
        TODO: Destroy infrastructure when no longer needed
        - terraform destroy
        - Cleanup state files
        """
        pass

def create_sample_data():
    """Create sample fraud detection dataset"""
    np.random.seed(42)
    
    n_samples = 1000
    
    # Generate features
    data = {
        'amount': np.random.lognormal(3, 1, n_samples),
        'merchant_category_encoded': np.random.randint(0, 5, n_samples),
        'hour': np.random.randint(0, 24, n_samples),
        'user_age': np.random.randint(18, 80, n_samples),
        'account_balance': np.random.lognormal(8, 1, n_samples)
    }
    
    # Generate target (fraud labels)
    fraud_probability = (
        (data['amount'] > 1000).astype(int) * 0.3 +
        (np.array(data['merchant_category_encoded']) > 3).astype(int) * 0.2 +
        np.random.random(n_samples) * 0.1
    )
    
    data['is_fraud'] = (fraud_probability > 0.4).astype(int)
    
    return pd.DataFrame(data)

def train_sample_model():
    """Train a sample fraud detection model"""
    print("Training sample fraud detection model...")
    
    # Create sample data
    data = create_sample_data()
    
    # Split data
    X = data.drop('is_fraud', axis=1)
    y = data['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save model and test data
    Path("models").mkdir(exist_ok=True)
    Path("data").mkdir(exist_ok=True)
    
    joblib.dump(model, "models/fraud_detection_model.pkl")
    
    # Save test data
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv("data/test_data.csv", index=False)
    
    # Save full dataset
    data.to_csv("data/fraud_detection_data.csv", index=False)
    
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"Model trained with accuracy: {accuracy:.3f}")
    
    return {
        'model_path': "models/fraud_detection_model.pkl",
        'test_data_path': "data/test_data.csv",
        'accuracy': accuracy
    }

def main():
    """
    Main exercise implementation
    """
    print("=== Day 36: CI/CD for ML - Exercise ===")
    
    # Initialize pipeline
    pipeline = MLCICDPipeline()
    
    # Create sample model and data
    print("\n0. Setting up sample model and data...")
    model_info = train_sample_model()
    
    # Scenario 1: Data Quality Validation
    print("\n1. Validating Data Quality...")
    validation_result = pipeline.validate_data_quality("data/fraud_detection_data.csv")
    
    print(f"   Status: {validation_result['status']}")
    print(f"   Checks passed: {sum(validation_result['checks'].values())}/{len(validation_result['checks'])}")
    
    if validation_result['errors']:
        print(f"   Errors: {validation_result['errors']}")
    
    # Scenario 2: Model Testing
    print("\n2. Running Model Tests...")
    test_result = pipeline.run_model_tests(
        model_info['model_path'],
        model_info['test_data_path']
    )
    
    print(f"   Status: {test_result['status']}")
    print(f"   Tests passed: {sum(test_result['tests'].values())}/{len(test_result['tests'])}")
    
    if test_result['metrics']:
        metrics = test_result['metrics']
        print(f"   Model Performance:")
        print(f"     - Accuracy: {metrics['accuracy']:.3f}")
        print(f"     - Precision: {metrics['precision']:.3f}")
        print(f"     - Recall: {metrics['recall']:.3f}")
        print(f"     - F1-Score: {metrics['f1_score']:.3f}")
    
    # Scenario 3: Blue-Green Deployment
    print("\n3. Deploying Model (Blue-Green)...")
    deployment_config = {
        'cluster_name': 'fraud-detection-cluster',
        'service_name': 'fraud-detection-service',
        'task_definition_arn': 'arn:aws:ecs:us-west-2:123456789012:task-definition/fraud-detection:1'
    }
    
    deployment_result = pipeline.deploy_model_blue_green(
        model_info['model_path'],
        deployment_config
    )
    
    print(f"   Status: {deployment_result['status']}")
    print(f"   Deployment ID: {deployment_result['deployment_id']}")
    
    if deployment_result['status'] == 'success':
        steps_completed = sum(deployment_result['steps'].values())
        total_steps = len(deployment_result['steps'])
        print(f"   Steps completed: {steps_completed}/{total_steps}")
    
    # Scenario 4: Performance Monitoring
    print("\n4. Monitoring Model Performance...")
    if deployment_result['status'] == 'success':
        performance_metrics = pipeline.monitor_model_performance(
            "http://fraud-detection-service.example.com",
            duration_minutes=2  # Short duration for demo
        )
        
        if 'error' not in performance_metrics:
            print(f"   Total Requests: {performance_metrics['total_requests']}")
            print(f"   Error Rate: {performance_metrics['error_rate']:.2%}")
            print(f"   Avg Response Time: {performance_metrics['avg_response_time']:.3f}s")
    
    # Scenario 5: Infrastructure Setup
    print("\n5. Setting Up Infrastructure...")
    infrastructure_result = pipeline.setup_infrastructure("staging")
    
    print(f"   Status: {infrastructure_result['status']}")
    
    if infrastructure_result['status'] == 'success':
        outputs = infrastructure_result.get('outputs', {})
        print(f"   VPC ID: {outputs.get('vpc_id', 'N/A')}")
        print(f"   ECS Cluster: {outputs.get('ecs_cluster_name', 'N/A')}")
        print(f"   S3 Bucket: {outputs.get('s3_artifacts_bucket', 'N/A')}")
    
    # Scenario 6: Monitoring Dashboard
    print("\n6. Creating Monitoring Dashboard...")
    dashboard_result = pipeline.create_monitoring_dashboard("fraud-detection-pipeline")
    
    print(f"   Status: {dashboard_result['status']}")
    
    if dashboard_result['status'] == 'success':
        print(f"   Dashboard: {dashboard_result['dashboard_name']}")
        print(f"   Alarms Created: {len(dashboard_result['alarms_created'])}")
    
    # Summary
    print("\n=== CI/CD Pipeline Summary ===")
    
    pipeline_status = {
        'Data Validation': validation_result['status'],
        'Model Testing': test_result['status'],
        'Blue-Green Deployment': deployment_result['status'],
        'Infrastructure Setup': infrastructure_result['status'],
        'Monitoring Dashboard': dashboard_result['status']
    }
    
    for component, status in pipeline_status.items():
        status_icon = "✅" if status in ['success', 'passed'] else "❌"
        print(f"   {status_icon} {component}: {status}")
    
    print("\n=== Exercise Complete ===")
    print("🎯 You've successfully implemented a complete CI/CD pipeline for ML!")
    print("📚 Review the solution.py for the complete implementation.")
    print("🧪 Take the quiz to test your understanding!")

if __name__ == "__main__":
    main()