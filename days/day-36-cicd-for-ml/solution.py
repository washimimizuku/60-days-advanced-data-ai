"""
Day 36: CI/CD for ML - Solution

Complete implementation of CI/CD pipeline for ML models with automated testing,
blue-green deployment, and comprehensive monitoring.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import boto3
import subprocess
import time
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import logging
from typing import Dict, List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLCICDPipeline:
    """
    Complete CI/CD pipeline for ML models
    """
    
    def __init__(self, config_path="config/pipeline_config.json"):
        """Initialize pipeline with configuration"""
        self.config_path = Path(config_path)
        self.load_configuration()
        
        # Initialize AWS clients
        self.ecs_client = boto3.client('ecs', region_name=self.config['aws_region'])
        self.elbv2_client = boto3.client('elbv2', region_name=self.config['aws_region'])
        self.cloudwatch = boto3.client('cloudwatch', region_name=self.config['aws_region'])
        self.s3_client = boto3.client('s3', region_name=self.config['aws_region'])
        
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
                },
                "deployment_config": {
                    "health_check_timeout": 300,
                    "monitoring_duration": 600
                }
            }
    
    def validate_data_quality(self, data_path):
        """
        Validate incoming data quality with comprehensive checks
        """
        logger.info(f"Validating data quality for: {data_path}")
        
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
            required_columns = ['amount', 'merchant_category', 'transaction_time', 'is_fraud']
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
            
            validation_results["checks"]["missing_values_acceptable"] = validation_results["status"] != "failed"
            
            # Check 4: Duplicate records
            max_duplicates = self.config["data_quality_thresholds"]["max_duplicate_percentage"]
            duplicate_percentage = data.duplicated().sum() / len(data)
            
            if duplicate_percentage > max_duplicates:
                validation_results["warnings"].append(
                    f"Data has {duplicate_percentage:.2%} duplicates (threshold: {max_duplicates:.2%})"
                )
            
            validation_results["checks"]["duplicate_records_acceptable"] = duplicate_percentage <= max_duplicates
            
            # Check 5: Data types validation
            numeric_columns = ['amount']
            for column in numeric_columns:
                if column in data.columns:
                    if not pd.api.types.is_numeric_dtype(data[column]):
                        validation_results["errors"].append(f"Column {column} should be numeric")
                        validation_results["status"] = "failed"
            
            validation_results["checks"]["data_types_correct"] = validation_results["status"] != "failed"
            
            # Check 6: Value ranges
            if 'amount' in data.columns:
                if (data['amount'] < 0).any():
                    validation_results["errors"].append("Negative transaction amounts found")
                    validation_results["status"] = "failed"
                
                if (data['amount'] > 100000).any():
                    validation_results["warnings"].append("Very large transaction amounts found")
            
            validation_results["checks"]["value_ranges_reasonable"] = validation_results["status"] != "failed"
            
            logger.info(f"Data validation completed: {validation_results['status']}")
            return validation_results
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return {
                "status": "failed",
                "checks": {},
                "errors": [f"Validation error: {str(e)}"],
                "warnings": []
            }
    
    def run_model_tests(self, model_path, test_data_path):
        """
        Run comprehensive model tests
        """
        logger.info(f"Running model tests for: {model_path}")
        
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
            
            test_results["tests"]["required_methods"] = len(test_results["errors"]) == 0
            
            # Test 3: Prediction functionality
            try:
                predictions = model.predict(X_test)
                probabilities = model.predict_proba(X_test)
                
                # Check shapes
                if len(predictions) != len(X_test):
                    test_results["errors"].append("Prediction length mismatch")
                    test_results["status"] = "failed"
                
                if probabilities.shape[0] != len(X_test):
                    test_results["errors"].append("Probability shape mismatch")
                    test_results["status"] = "failed"
                
                # Check probability ranges
                if not np.all((probabilities >= 0) & (probabilities <= 1)):
                    test_results["errors"].append("Probabilities outside [0,1] range")
                    test_results["status"] = "failed"
                
                test_results["tests"]["prediction_functionality"] = len(test_results["errors"]) == 0
                
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
                    
                    if precision < thresholds["min_precision"]:
                        test_results["errors"].append(f"Precision {precision:.3f} below threshold {thresholds['min_precision']}")
                        test_results["status"] = "failed"
                    
                    if recall < thresholds["min_recall"]:
                        test_results["errors"].append(f"Recall {recall:.3f} below threshold {thresholds['min_recall']}")
                        test_results["status"] = "failed"
                    
                    if f1 < thresholds["min_f1_score"]:
                        test_results["errors"].append(f"F1-score {f1:.3f} below threshold {thresholds['min_f1_score']}")
                        test_results["status"] = "failed"
                    
                    test_results["tests"]["performance_thresholds"] = test_results["status"] != "failed"
                    
                except Exception as e:
                    test_results["errors"].append(f"Performance test failed: {str(e)}")
                    test_results["status"] = "failed"
            
            # Test 5: Model stability
            if test_results["status"] != "failed":
                try:
                    sample_input = X_test.iloc[:10]
                    
                    predictions_1 = model.predict(sample_input)
                    predictions_2 = model.predict(sample_input)
                    
                    if not np.array_equal(predictions_1, predictions_2):
                        test_results["errors"].append("Model predictions not stable")
                        test_results["status"] = "failed"
                    
                    test_results["tests"]["model_stability"] = test_results["status"] != "failed"
                    
                except Exception as e:
                    test_results["errors"].append(f"Stability test failed: {str(e)}")
                    test_results["status"] = "failed"
            
            logger.info(f"Model tests completed: {test_results['status']}")
            return test_results
            
        except Exception as e:
            logger.error(f"Model testing failed: {str(e)}")
            return {
                "status": "failed",
                "tests": {},
                "errors": [f"Test error: {str(e)}"],
                "metrics": {}
            }
    
    def deploy_model_blue_green(self, model_path, deployment_config):
        """
        Deploy model using blue-green strategy
        """
        logger.info("Starting blue-green deployment")
        
        deployment_id = f"deploy-{int(datetime.now().timestamp())}"
        
        try:
            deployment_result = {
                "deployment_id": deployment_id,
                "status": "in_progress",
                "steps": {},
                "errors": []
            }
            
            # Step 1: Create green service
            logger.info("Creating green service...")
            green_service_name = f"{deployment_config['service_name']}-green-{deployment_id}"
            
            green_service_result = self._create_green_service(
                deployment_config['cluster_name'],
                green_service_name,
                deployment_config['task_definition_arn']
            )
            
            if not green_service_result['success']:
                deployment_result["errors"].append("Failed to create green service")
                deployment_result["status"] = "failed"
                return deployment_result
            
            deployment_result["steps"]["green_service_created"] = True
            
            # Step 2: Wait for service to be healthy
            logger.info("Waiting for green service to be healthy...")
            if not self._wait_for_service_healthy(
                deployment_config['cluster_name'], 
                green_service_name,
                timeout=self.config["deployment_config"]["health_check_timeout"]
            ):
                deployment_result["errors"].append("Green service failed to become healthy")
                deployment_result["status"] = "failed"
                return deployment_result
            
            deployment_result["steps"]["green_service_healthy"] = True
            
            # Step 3: Run health checks
            logger.info("Running health checks...")
            green_endpoint = self._get_service_endpoint(deployment_config['cluster_name'], green_service_name)
            
            health_check_result = self._run_comprehensive_health_checks(green_endpoint)
            
            if not health_check_result['success']:
                deployment_result["errors"].extend(health_check_result['errors'])
                deployment_result["status"] = "failed"
                return deployment_result
            
            deployment_result["steps"]["health_checks_passed"] = True
            
            # Step 4: Switch traffic
            logger.info("Switching traffic to green service...")
            traffic_switch_result = self._switch_traffic_to_green(
                deployment_config['target_group_arn'],
                deployment_config['cluster_name'],
                green_service_name
            )
            
            if not traffic_switch_result['success']:
                deployment_result["errors"].append("Failed to switch traffic")
                deployment_result["status"] = "failed"
                return deployment_result
            
            deployment_result["steps"]["traffic_switched"] = True
            
            # Step 5: Monitor performance
            logger.info("Monitoring green service performance...")
            monitoring_result = self._monitor_deployment_performance(
                green_endpoint,
                duration=self.config["deployment_config"]["monitoring_duration"]
            )
            
            if not monitoring_result['success']:
                # Rollback on performance issues
                logger.warning("Performance issues detected, rolling back...")
                self._rollback_to_blue_service(
                    deployment_config['target_group_arn'],
                    deployment_config['cluster_name'],
                    deployment_config['service_name']
                )
                
                deployment_result["errors"].append(f"Performance issues: {monitoring_result['reason']}")
                deployment_result["status"] = "rolled_back"
                return deployment_result
            
            deployment_result["steps"]["performance_monitoring_passed"] = True
            
            # Step 6: Cleanup old service
            logger.info("Cleaning up old blue service...")
            self._cleanup_old_blue_service(
                deployment_config['cluster_name'],
                deployment_config['service_name']
            )
            
            deployment_result["steps"]["old_service_cleaned"] = True
            
            # Step 7: Promote green to blue
            self._promote_green_to_blue(
                deployment_config['cluster_name'],
                green_service_name,
                deployment_config['service_name']
            )
            
            deployment_result["status"] = "success"
            deployment_result["green_service_name"] = green_service_name
            deployment_result["timestamp"] = datetime.now().isoformat()
            
            logger.info("Blue-green deployment completed successfully")
            return deployment_result
            
        except Exception as e:
            logger.error(f"Deployment failed: {str(e)}")
            
            # Cleanup on failure
            try:
                self._cleanup_failed_deployment(deployment_config['cluster_name'], green_service_name)
            except:
                pass
            
            return {
                "deployment_id": deployment_id,
                "status": "failed",
                "steps": deployment_result.get("steps", {}),
                "errors": [f"Deployment error: {str(e)}"],
                "timestamp": datetime.now().isoformat()
            }
    
    def _create_green_service(self, cluster_name, service_name, task_definition_arn):
        """Create new green service"""
        try:
            # Simulate service creation (in real implementation, use ECS API)
            logger.info(f"Creating ECS service: {service_name}")
            
            # In real implementation:
            # response = self.ecs_client.create_service(...)
            
            return {"success": True, "service_arn": f"arn:aws:ecs:us-west-2:123456789012:service/{service_name}"}
            
        except Exception as e:
            logger.error(f"Failed to create green service: {str(e)}")
            return {"success": False, "error": str(e)}
    
    def _wait_for_service_healthy(self, cluster_name, service_name, timeout=600):
        """Wait for service to become healthy"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            # Simulate health check (in real implementation, check ECS service status)
            logger.info(f"Checking service health: {service_name}")
            
            # In real implementation:
            # response = self.ecs_client.describe_services(...)
            # Check running_count == desired_count and status == 'ACTIVE'
            
            # Simulate successful health check after 30 seconds
            if time.time() - start_time > 30:
                return True
            
            time.sleep(10)
        
        return False
    
    def _get_service_endpoint(self, cluster_name, service_name):
        """Get service endpoint URL"""
        # In real implementation, get ALB DNS name or service discovery endpoint
        return f"http://{service_name}.example.com"
    
    def _run_comprehensive_health_checks(self, endpoint):
        """Run comprehensive health checks on service"""
        try:
            health_checks = [
                self._check_health_endpoint,
                self._check_prediction_endpoint,
                self._check_response_time
            ]
            
            for check in health_checks:
                result = check(endpoint)
                if not result['success']:
                    return {
                        'success': False,
                        'errors': [result['error']]
                    }
            
            return {'success': True, 'errors': []}
            
        except Exception as e:
            return {
                'success': False,
                'errors': [f"Health check error: {str(e)}"]
            }
    
    def _check_health_endpoint(self, endpoint):
        """Check health endpoint"""
        try:
            # Simulate health check
            logger.info(f"Checking health endpoint: {endpoint}/health")
            
            # In real implementation:
            # response = requests.get(f"{endpoint}/health", timeout=10)
            # return {'success': response.status_code == 200}
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': f"Health endpoint failed: {str(e)}"}
    
    def _check_prediction_endpoint(self, endpoint):
        """Check prediction endpoint functionality"""
        try:
            # Simulate prediction check
            logger.info(f"Checking prediction endpoint: {endpoint}/predict")
            
            # In real implementation:
            # sample_data = {"amount": 100.0, "merchant_category": "grocery"}
            # response = requests.post(f"{endpoint}/predict", json=sample_data, timeout=10)
            # return {'success': response.status_code == 200 and 'prediction' in response.json()}
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': f"Prediction endpoint failed: {str(e)}"}
    
    def _check_response_time(self, endpoint):
        """Check response time requirements"""
        try:
            # Simulate response time check
            logger.info(f"Checking response time: {endpoint}")
            
            # In real implementation:
            # start_time = time.time()
            # response = requests.get(f"{endpoint}/health", timeout=5)
            # response_time = time.time() - start_time
            # return {'success': response.status_code == 200 and response_time < 2.0}
            
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': f"Response time check failed: {str(e)}"}
    
    def _switch_traffic_to_green(self, target_group_arn, cluster_name, green_service_name):
        """Switch ALB traffic to green service"""
        try:
            logger.info("Switching ALB traffic to green service")
            
            # In real implementation:
            # 1. Get green service target IPs
            # 2. Register new targets in ALB target group
            # 3. Wait for targets to be healthy
            # 4. Deregister old targets
            
            return {'success': True}
            
        except Exception as e:
            logger.error(f"Failed to switch traffic: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _monitor_deployment_performance(self, endpoint, duration=600):
        """Monitor deployment performance"""
        try:
            logger.info(f"Monitoring performance for {duration} seconds")
            
            start_time = time.time()
            error_count = 0
            total_requests = 0
            
            while time.time() - start_time < duration:
                # Simulate performance monitoring
                total_requests += 1
                
                # In real implementation:
                # response = requests.get(f"{endpoint}/health", timeout=5)
                # if response.status_code != 200:
                #     error_count += 1
                
                # Simulate 99% success rate
                if np.random.random() < 0.01:
                    error_count += 1
                
                error_rate = error_count / total_requests if total_requests > 0 else 0
                
                # Check error rate threshold (5%)
                if error_rate > 0.05:
                    return {
                        'success': False,
                        'reason': f'Error rate {error_rate:.2%} exceeds threshold'
                    }
                
                time.sleep(10)
            
            return {'success': True}
            
        except Exception as e:
            return {
                'success': False,
                'reason': f'Monitoring error: {str(e)}'
            }
    
    def _rollback_to_blue_service(self, target_group_arn, cluster_name, blue_service_name):
        """Rollback traffic to blue service"""
        logger.info("Rolling back to blue service")
        
        # In real implementation:
        # 1. Get blue service target IPs
        # 2. Register blue targets in ALB target group
        # 3. Deregister green targets
        
        return True
    
    def _cleanup_old_blue_service(self, cluster_name, service_name):
        """Cleanup old blue service"""
        logger.info(f"Cleaning up old service: {service_name}")
        
        # In real implementation:
        # 1. Scale service to 0 desired count
        # 2. Wait for tasks to stop
        # 3. Delete service
        
        return True
    
    def _promote_green_to_blue(self, cluster_name, green_service_name, blue_service_name):
        """Promote green service to blue"""
        logger.info(f"Promoting {green_service_name} to {blue_service_name}")
        
        # In real implementation:
        # Update service discovery or DNS records
        
        return True
    
    def _cleanup_failed_deployment(self, cluster_name, service_name):
        """Cleanup failed deployment"""
        try:
            self._cleanup_old_blue_service(cluster_name, service_name)
        except:
            pass
    
    def monitor_model_performance(self, model_endpoint, duration_minutes=30):
        """
        Monitor deployed model performance
        """
        logger.info(f"Monitoring model performance for {duration_minutes} minutes")
        
        try:
            duration_seconds = duration_minutes * 60
            start_time = time.time()
            
            metrics = {
                'total_requests': 0,
                'successful_requests': 0,
                'failed_requests': 0,
                'response_times': [],
                'error_rate': 0.0,
                'avg_response_time': 0.0,
                'p95_response_time': 0.0
            }
            
            while time.time() - start_time < duration_seconds:
                # Simulate request monitoring
                request_start = time.time()
                
                # In real implementation:
                # response = requests.get(f"{model_endpoint}/health", timeout=5)
                
                # Simulate response
                success = np.random.random() > 0.02  # 98% success rate
                response_time = np.random.normal(0.5, 0.1)  # 500ms avg response time
                
                metrics['total_requests'] += 1
                
                if success:
                    metrics['successful_requests'] += 1
                    metrics['response_times'].append(response_time)
                else:
                    metrics['failed_requests'] += 1
                
                # Calculate running metrics
                if metrics['total_requests'] > 0:
                    metrics['error_rate'] = metrics['failed_requests'] / metrics['total_requests']
                
                if metrics['response_times']:
                    metrics['avg_response_time'] = np.mean(metrics['response_times'])
                    metrics['p95_response_time'] = np.percentile(metrics['response_times'], 95)
                
                time.sleep(5)  # Monitor every 5 seconds
            
            # Final calculations
            metrics['monitoring_duration'] = duration_minutes
            metrics['timestamp'] = datetime.now().isoformat()
            
            logger.info(f"Performance monitoring completed: {metrics['error_rate']:.2%} error rate, {metrics['avg_response_time']:.3f}s avg response time")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Performance monitoring failed: {str(e)}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def setup_infrastructure(self, environment="staging"):
        """
        Set up infrastructure using Terraform
        """
        logger.info(f"Setting up infrastructure for environment: {environment}")
        
        try:
            terraform_dir = Path("infrastructure")
            terraform_dir.mkdir(exist_ok=True)
            
            # Generate Terraform configuration
            terraform_config = self._generate_terraform_config(environment)
            
            # Write main.tf
            with open(terraform_dir / "main.tf", "w") as f:
                f.write(terraform_config)
            
            # Write variables.tf
            variables_config = self._generate_variables_config()
            with open(terraform_dir / "variables.tf", "w") as f:
                f.write(variables_config)
            
            # Write terraform.tfvars
            tfvars_config = self._generate_tfvars_config(environment)
            with open(terraform_dir / f"{environment}.tfvars", "w") as f:
                f.write(tfvars_config)
            
            # Initialize and apply Terraform
            result = self._apply_terraform(terraform_dir, environment)
            
            logger.info(f"Infrastructure setup completed: {result['status']}")
            return result
            
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_terraform_config(self, environment):
        """Generate Terraform main configuration"""
        return f'''
terraform {{
  required_version = ">= 1.0"
  
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
  
  backend "s3" {{
    bucket = "ml-terraform-state-{environment}"
    key    = "ml-infrastructure/terraform.tfstate"
    region = "us-west-2"
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

# VPC
resource "aws_vpc" "ml_vpc" {{
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {{
    Name        = "${{var.project_name}}-${{var.environment}}-vpc"
    Environment = var.environment
  }}
}}

# Subnets
resource "aws_subnet" "ml_subnet" {{
  count = 2
  
  vpc_id            = aws_vpc.ml_vpc.id
  cidr_block        = "10.0.${{count.index + 1}}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {{
    Name        = "${{var.project_name}}-${{var.environment}}-subnet-${{count.index + 1}}"
    Environment = var.environment
  }}
}}

# Internet Gateway
resource "aws_internet_gateway" "ml_igw" {{
  vpc_id = aws_vpc.ml_vpc.id
  
  tags = {{
    Name        = "${{var.project_name}}-${{var.environment}}-igw"
    Environment = var.environment
  }}
}}

# ECS Cluster
resource "aws_ecs_cluster" "ml_cluster" {{
  name = "${{var.project_name}}-${{var.environment}}-cluster"
  
  setting {{
    name  = "containerInsights"
    value = "enabled"
  }}
  
  tags = {{
    Name        = "${{var.project_name}}-${{var.environment}}-cluster"
    Environment = var.environment
  }}
}}

# S3 Buckets
resource "aws_s3_bucket" "ml_artifacts" {{
  bucket = "${{var.project_name}}-${{var.environment}}-artifacts-${{random_id.bucket_suffix.hex}}"
  
  tags = {{
    Name        = "${{var.project_name}}-${{var.environment}}-artifacts"
    Environment = var.environment
  }}
}}

resource "random_id" "bucket_suffix" {{
  byte_length = 4
}}

# Outputs
output "vpc_id" {{
  description = "VPC ID"
  value       = aws_vpc.ml_vpc.id
}}

output "ecs_cluster_name" {{
  description = "ECS Cluster Name"
  value       = aws_ecs_cluster.ml_cluster.name
}}

output "s3_artifacts_bucket" {{
  description = "S3 Artifacts Bucket Name"
  value       = aws_s3_bucket.ml_artifacts.bucket
}}
'''
    
    def _generate_variables_config(self):
        """Generate Terraform variables configuration"""
        return '''
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-west-2"
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "ml-pipeline"
}

data "aws_availability_zones" "available" {
  state = "available"
}
'''
    
    def _generate_tfvars_config(self, environment):
        """Generate Terraform tfvars configuration"""
        return f'''
aws_region = "us-west-2"
environment = "{environment}"
project_name = "fraud-detection-ml"
'''
    
    def _apply_terraform(self, terraform_dir, environment):
        """Apply Terraform configuration"""
        try:
            # Change to terraform directory
            original_dir = Path.cwd()
            
            # Initialize Terraform
            logger.info("Initializing Terraform...")
            init_result = subprocess.run(
                ["terraform", "init"],
                cwd=terraform_dir,
                capture_output=True,
                text=True
            )
            
            if init_result.returncode != 0:
                return {
                    'status': 'failed',
                    'step': 'init',
                    'error': init_result.stderr
                }
            
            # Plan Terraform
            logger.info("Planning Terraform...")
            plan_result = subprocess.run(
                ["terraform", "plan", f"-var-file={environment}.tfvars"],
                cwd=terraform_dir,
                capture_output=True,
                text=True
            )
            
            if plan_result.returncode != 0:
                return {
                    'status': 'failed',
                    'step': 'plan',
                    'error': plan_result.stderr
                }
            
            # Apply Terraform (in real implementation, you might want approval)
            logger.info("Applying Terraform...")
            # apply_result = subprocess.run(
            #     ["terraform", "apply", "-auto-approve", f"-var-file={environment}.tfvars"],
            #     cwd=terraform_dir,
            #     capture_output=True,
            #     text=True
            # )
            
            # Simulate successful apply
            return {
                'status': 'success',
                'step': 'apply',
                'outputs': {
                    'vpc_id': 'vpc-12345678',
                    'ecs_cluster_name': f'fraud-detection-ml-{environment}-cluster',
                    's3_artifacts_bucket': f'fraud-detection-ml-{environment}-artifacts-abcd1234'
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'step': 'execution',
                'error': str(e)
            }
    
    def create_monitoring_dashboard(self, pipeline_name):
        """
        Create CloudWatch monitoring dashboard
        """
        logger.info(f"Creating monitoring dashboard for: {pipeline_name}")
        
        try:
            # Create custom metrics namespace
            namespace = f'MLPipeline/{pipeline_name}'
            
            # Define dashboard widgets
            dashboard_body = {
                "widgets": [
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [namespace, "ModelAccuracy"],
                                [".", "ModelPrecision"],
                                [".", "ModelRecall"],
                                [".", "ModelF1Score"]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config['aws_region'],
                            "title": "Model Performance Metrics",
                            "yAxis": {
                                "left": {
                                    "min": 0,
                                    "max": 1
                                }
                            }
                        }
                    },
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 0,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [namespace, "PredictionLatency"],
                                [".", "ThroughputRPS"]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config['aws_region'],
                            "title": "System Performance"
                        }
                    },
                    {
                        "type": "metric",
                        "x": 0,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [namespace, "DataDriftScore"],
                                [".", "ModelDriftScore"]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config['aws_region'],
                            "title": "Drift Detection"
                        }
                    },
                    {
                        "type": "metric",
                        "x": 12,
                        "y": 6,
                        "width": 12,
                        "height": 6,
                        "properties": {
                            "metrics": [
                                [namespace, "ErrorRate"],
                                [".", "SuccessRate"]
                            ],
                            "period": 300,
                            "stat": "Average",
                            "region": self.config['aws_region'],
                            "title": "Error Rates"
                        }
                    }
                ]
            }
            
            # Create dashboard
            dashboard_name = f'{pipeline_name}-MLPipeline-Dashboard'
            
            # In real implementation:
            # self.cloudwatch.put_dashboard(
            #     DashboardName=dashboard_name,
            #     DashboardBody=json.dumps(dashboard_body)
            # )
            
            # Create alarms
            alarms_created = self._create_cloudwatch_alarms(pipeline_name, namespace)
            
            # Publish initial metrics
            self._publish_initial_metrics(namespace)
            
            result = {
                'status': 'success',
                'dashboard_name': dashboard_name,
                'dashboard_url': f'https://console.aws.amazon.com/cloudwatch/home?region={self.config["aws_region"]}#dashboards:name={dashboard_name}',
                'alarms_created': alarms_created,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Monitoring dashboard created successfully")
            return result
            
        except Exception as e:
            logger.error(f"Failed to create monitoring dashboard: {str(e)}")
            return {
                'status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _create_cloudwatch_alarms(self, pipeline_name, namespace):
        """Create CloudWatch alarms"""
        alarms = [
            {
                'name': f'{pipeline_name}-ModelAccuracy-Low',
                'metric': 'ModelAccuracy',
                'threshold': 0.8,
                'comparison': 'LessThanThreshold',
                'description': 'Model accuracy dropped below 80%'
            },
            {
                'name': f'{pipeline_name}-PredictionLatency-High',
                'metric': 'PredictionLatency',
                'threshold': 2000,
                'comparison': 'GreaterThanThreshold',
                'description': 'Prediction latency exceeded 2 seconds'
            },
            {
                'name': f'{pipeline_name}-ErrorRate-High',
                'metric': 'ErrorRate',
                'threshold': 0.05,
                'comparison': 'GreaterThanThreshold',
                'description': 'Error rate exceeded 5%'
            },
            {
                'name': f'{pipeline_name}-DataDrift-High',
                'metric': 'DataDriftScore',
                'threshold': 0.3,
                'comparison': 'GreaterThanThreshold',
                'description': 'Data drift score exceeded threshold'
            }
        ]
        
        created_alarms = []
        
        for alarm in alarms:
            try:
                # In real implementation:
                # self.cloudwatch.put_metric_alarm(
                #     AlarmName=alarm['name'],
                #     ComparisonOperator=alarm['comparison'],
                #     EvaluationPeriods=2,
                #     MetricName=alarm['metric'],
                #     Namespace=namespace,
                #     Period=300,
                #     Statistic='Average',
                #     Threshold=alarm['threshold'],
                #     ActionsEnabled=True,
                #     AlarmDescription=alarm['description'],
                #     Unit='None'
                # )
                
                created_alarms.append(alarm['name'])
                logger.info(f"Created alarm: {alarm['name']}")
                
            except Exception as e:
                logger.error(f"Failed to create alarm {alarm['name']}: {str(e)}")
        
        return created_alarms
    
    def _publish_initial_metrics(self, namespace):
        """Publish initial metrics to CloudWatch"""
        initial_metrics = [
            {'MetricName': 'ModelAccuracy', 'Value': 0.85, 'Unit': 'None'},
            {'MetricName': 'ModelPrecision', 'Value': 0.82, 'Unit': 'None'},
            {'MetricName': 'ModelRecall', 'Value': 0.88, 'Unit': 'None'},
            {'MetricName': 'ModelF1Score', 'Value': 0.85, 'Unit': 'None'},
            {'MetricName': 'PredictionLatency', 'Value': 500, 'Unit': 'Milliseconds'},
            {'MetricName': 'ThroughputRPS', 'Value': 100, 'Unit': 'Count/Second'},
            {'MetricName': 'ErrorRate', 'Value': 0.02, 'Unit': 'Percent'},
            {'MetricName': 'SuccessRate', 'Value': 0.98, 'Unit': 'Percent'},
            {'MetricName': 'DataDriftScore', 'Value': 0.1, 'Unit': 'None'},
            {'MetricName': 'ModelDriftScore', 'Value': 0.05, 'Unit': 'None'}
        ]
        
        for metric in initial_metrics:
            metric['Timestamp'] = datetime.utcnow()
        
        # In real implementation:
        # self.cloudwatch.put_metric_data(
        #     Namespace=namespace,
        #     MetricData=initial_metrics
        # )
        
        logger.info(f"Published {len(initial_metrics)} initial metrics")

def create_sample_data():
    """Create sample fraud detection dataset"""
    np.random.seed(42)
    
    n_samples = 1000
    
    # Generate features
    data = {
        'amount': np.random.lognormal(3, 1, n_samples),
        'merchant_category': np.random.choice(['grocery', 'gas', 'restaurant', 'retail', 'online'], n_samples),
        'transaction_time': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'user_age': np.random.randint(18, 80, n_samples),
        'account_balance': np.random.lognormal(8, 1, n_samples)
    }
    
    # Generate target (fraud labels)
    # Higher amounts and certain categories more likely to be fraud
    fraud_probability = (
        (data['amount'] > 1000).astype(int) * 0.3 +
        (pd.Series(data['merchant_category']).isin(['online', 'retail'])).astype(int) * 0.2 +
        np.random.random(n_samples) * 0.1
    )
    
    data['is_fraud'] = (fraud_probability > 0.4).astype(int)
    
    df = pd.DataFrame(data)
    
    # Encode categorical variables
    df['merchant_category_encoded'] = pd.Categorical(df['merchant_category']).codes
    df['hour'] = df['transaction_time'].dt.hour
    
    # Select features for model
    feature_columns = ['amount', 'merchant_category_encoded', 'hour', 'user_age', 'account_balance']
    model_df = df[feature_columns + ['is_fraud']]
    
    return model_df

def train_sample_model():
    """Train a sample fraud detection model"""
    logger.info("Training sample fraud detection model...")
    
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
    Path("data/test").mkdir(parents=True, exist_ok=True)
    Path("config").mkdir(exist_ok=True)
    
    joblib.dump(model, "models/fraud_detection_model.pkl")
    
    # Save test data
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv("data/test/test_data.csv", index=False)
    
    # Save full dataset
    data.to_csv("data/fraud_detection_data.csv", index=False)
    
    # Create model config
    config = {
        "model_type": "RandomForestClassifier",
        "target_column": "is_fraud",
        "feature_columns": list(X.columns),
        "min_accuracy": 0.85,
        "min_precision": 0.80,
        "min_recall": 0.80,
        "min_f1_score": 0.80
    }
    
    with open("config/model_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Evaluate model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='weighted')
    recall = recall_score(y_test, predictions, average='weighted')
    f1 = f1_score(y_test, predictions, average='weighted')
    
    logger.info(f"Model trained - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    return {
        'model_path': "models/fraud_detection_model.pkl",
        'test_data_path': "data/test/test_data.csv",
        'config_path': "config/model_config.json",
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    }

def main():
    """
    Main solution implementation demonstrating complete CI/CD pipeline
    """
    print("=== Day 36: CI/CD for ML - Solution ===")
    
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
    
    if validation_result['warnings']:
        print(f"   Warnings: {validation_result['warnings']}")
    
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
    
    if test_result['errors']:
        print(f"   Errors: {test_result['errors']}")
    
    # Scenario 3: Blue-Green Deployment
    print("\n3. Deploying Model (Blue-Green)...")
    deployment_config = {
        'cluster_name': 'fraud-detection-cluster',
        'service_name': 'fraud-detection-service',
        'task_definition_arn': 'arn:aws:ecs:us-west-2:123456789012:task-definition/fraud-detection:1',
        'target_group_arn': 'arn:aws:elasticloadbalancing:us-west-2:123456789012:targetgroup/fraud-detection/1234567890123456'
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
    
    if deployment_result['errors']:
        print(f"   Errors: {deployment_result['errors']}")
    
    # Scenario 4: Performance Monitoring
    print("\n4. Monitoring Model Performance...")
    if deployment_result['status'] == 'success':
        performance_metrics = pipeline.monitor_model_performance(
            "http://fraud-detection-service.example.com",
            duration_minutes=1  # Short duration for demo
        )
        
        if 'error' not in performance_metrics:
            print(f"   Total Requests: {performance_metrics['total_requests']}")
            print(f"   Error Rate: {performance_metrics['error_rate']:.2%}")
            print(f"   Avg Response Time: {performance_metrics['avg_response_time']:.3f}s")
            print(f"   P95 Response Time: {performance_metrics['p95_response_time']:.3f}s")
        else:
            print(f"   Monitoring Error: {performance_metrics['error']}")
    else:
        print("   Skipping performance monitoring due to deployment failure")
    
    # Scenario 5: Infrastructure Setup
    print("\n5. Setting Up Infrastructure...")
    infrastructure_result = pipeline.setup_infrastructure("staging")
    
    print(f"   Status: {infrastructure_result['status']}")
    
    if infrastructure_result['status'] == 'success':
        outputs = infrastructure_result.get('outputs', {})
        print(f"   VPC ID: {outputs.get('vpc_id', 'N/A')}")
        print(f"   ECS Cluster: {outputs.get('ecs_cluster_name', 'N/A')}")
        print(f"   S3 Bucket: {outputs.get('s3_artifacts_bucket', 'N/A')}")
    
    if 'error' in infrastructure_result:
        print(f"   Error: {infrastructure_result['error']}")
    
    # Scenario 6: Monitoring Dashboard
    print("\n6. Creating Monitoring Dashboard...")
    dashboard_result = pipeline.create_monitoring_dashboard("fraud-detection-pipeline")
    
    print(f"   Status: {dashboard_result['status']}")
    
    if dashboard_result['status'] == 'success':
        print(f"   Dashboard: {dashboard_result['dashboard_name']}")
        print(f"   Alarms Created: {len(dashboard_result['alarms_created'])}")
        print(f"   Dashboard URL: {dashboard_result['dashboard_url']}")
    
    if 'error' in dashboard_result:
        print(f"   Error: {dashboard_result['error']}")
    
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
        status_icon = "✅" if status == "success" else "❌" if status == "failed" else "⚠️"
        print(f"   {status_icon} {component}: {status}")
    
    overall_success = all(status in ['success', 'passed'] for status in pipeline_status.values())
    
    print(f"\n🎯 Overall Pipeline Status: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")
    
    if overall_success:
        print("\n🚀 Your fraud detection model is now deployed with:")
        print("   • Automated data quality validation")
        print("   • Comprehensive model testing")
        print("   • Zero-downtime blue-green deployment")
        print("   • Real-time performance monitoring")
        print("   • Infrastructure as code")
        print("   • CloudWatch dashboards and alerting")
    
    print("\n=== Solution Complete ===")
    print("Review the implementation to understand CI/CD best practices for ML!")

if __name__ == "__main__":
    main()