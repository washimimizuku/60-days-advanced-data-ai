"""
Day 56: Kubernetes for ML & Data Workloads - Comprehensive Test Suite
Tests for Kubernetes ML orchestration, scaling, and management
"""

import pytest
import yaml
import json
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import the classes from exercise and solution files
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solution import (
    MLTrainingManager, KubernetesClientManager, 
    KubernetesResource, MockKubernetesAPI
)


class TestKubernetesClientManager:
    """Test Kubernetes client management functionality"""
    
    def test_client_manager_initialization(self):
        """Test Kubernetes client manager initialization"""
        manager = KubernetesClientManager("test-cluster")
        assert manager.cluster_name == "test-cluster"
        assert "ml-training" in manager.namespaces
        assert "ml-serving" in manager.namespaces
    
    def test_mock_client_setup(self):
        """Test mock client setup when Kubernetes is not available"""
        manager = KubernetesClientManager("test-cluster")
        manager._setup_mock_client()
        
        assert manager.api_client is not None
        assert manager.apps_v1 is not None
        assert manager.core_v1 is not None
        assert manager.batch_v1 is not None
    
    def test_create_resource_from_yaml(self):
        """Test creating Kubernetes resource from YAML"""
        manager = KubernetesClientManager("test-cluster")
        
        yaml_content = """
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test-container
    image: nginx:latest
"""
        
        resource_id = manager.create_resource_from_yaml(yaml_content, "default")
        assert "default/Pod/test-pod" in resource_id
        assert resource_id in manager.resources
    
    def test_resource_storage(self):
        """Test resource storage and retrieval"""
        manager = KubernetesClientManager("test-cluster")
        
        resource = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "test-service"},
            "spec": {"selector": {"app": "test"}}
        }
        
        resource_id = manager._create_resource(resource, "default")
        stored_resource = manager.resources[resource_id]
        
        assert stored_resource["resource"]["kind"] == "Service"
        assert stored_resource["status"] == "Creating"
        assert "created_at" in stored_resource


class TestMLTrainingManager:
    """Test ML training job management functionality"""
    
    def setup_method(self):
        """Set up test environment"""
        self.k8s_client = KubernetesClientManager("test-cluster")
        self.training_manager = MLTrainingManager(self.k8s_client)
    
    def test_training_manager_initialization(self):
        """Test ML training manager initialization"""
        assert self.training_manager.k8s_client == self.k8s_client
        assert self.training_manager.training_jobs == {}
    
    def test_create_training_job(self):
        """Test creating a basic training job"""
        training_config = {
            "command": ["python"],
            "args": ["train.py"],
            "environment": {
                "EPOCHS": "100",
                "BATCH_SIZE": "32"
            },
            "resources": {
                "cpu": "4",
                "memory": "8Gi",
                "gpu": 1
            },
            "framework": "tensorflow"
        }
        
        job_id = self.training_manager.create_training_job(
            "test-training-job",
            "tensorflow/tensorflow:2.13.0-gpu",
            training_config
        )
        
        assert job_id is not None
        assert "test-training-job" in self.training_manager.training_jobs
        
        job_info = self.training_manager.training_jobs["test-training-job"]
        assert job_info["status"] == "Created"
        assert job_info["config"] == training_config
    
    def test_create_distributed_training_job(self):
        """Test creating distributed training job"""
        training_config = {
            "image": "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime",
            "command": ["python"],
            "args": ["train_distributed.py"],
            "resources": {
                "cpu": "8",
                "memory": "16Gi",
                "gpu": 2
            }
        }
        
        job_id = self.training_manager.create_distributed_training_job(
            "distributed-training",
            worker_count=3,
            training_config=training_config
        )
        
        assert job_id is not None
        assert "distributed-training" in self.training_manager.training_jobs
        
        job_info = self.training_manager.training_jobs["distributed-training"]
        assert job_info["type"] == "distributed"
        assert job_info["worker_count"] == 3
        assert len(job_info["worker_job_ids"]) == 3
    
    def test_create_hyperparameter_tuning_job(self):
        """Test creating hyperparameter tuning jobs"""
        hyperparameter_space = {
            "learning_rate": [0.001, 0.01],
            "batch_size": [16, 32],
            "dropout_rate": [0.1, 0.2]
        }
        
        job_ids = self.training_manager.create_hyperparameter_tuning_job(
            "hp-tuning",
            hyperparameter_space
        )
        
        # Should create 2 * 2 * 2 = 8 jobs
        assert len(job_ids) == 8
        
        # Check that jobs were created with different parameter combinations
        created_jobs = [job for job in self.training_manager.training_jobs.keys() 
                       if job.startswith("hp-tuning-hp-")]
        assert len(created_jobs) == 8
    
    def test_monitor_training_progress(self):
        """Test monitoring training job progress"""
        # First create a job
        training_config = {
            "resources": {"cpu": "4", "memory": "8Gi", "gpu": 1}
        }
        
        self.training_manager.create_training_job(
            "monitor-test-job",
            "tensorflow/tensorflow:2.13.0-gpu",
            training_config
        )
        
        # Monitor the job
        progress = self.training_manager.monitor_training_progress("monitor-test-job")
        
        assert progress["job_name"] == "monitor-test-job"
        assert "status" in progress
        assert "progress_percent" in progress
        assert "resource_usage" in progress
        assert "training_metrics" in progress
        
        # Test monitoring non-existent job
        error_result = self.training_manager.monitor_training_progress("non-existent-job")
        assert "error" in error_result
    
    def test_setup_gpu_training_job(self):
        """Test setting up GPU-enabled training job"""
        job_id = self.training_manager.setup_gpu_training_job(
            "gpu-training-job",
            gpu_count=2,
            gpu_type="nvidia-tesla-v100"
        )
        
        assert job_id is not None
        assert "gpu-training-job" in self.training_manager.training_jobs
        
        job_info = self.training_manager.training_jobs["gpu-training-job"]
        assert job_info["config"]["resources"]["gpu"] == 2
        assert job_info["config"]["gpu_type"] == "nvidia-tesla-v100"
    
    def test_distributed_job_spec_creation(self):
        """Test creation of distributed job specifications"""
        config = {
            "image": "pytorch/pytorch:latest",
            "command": ["python"],
            "args": ["train.py"],
            "environment": {
                "WORLD_SIZE": "4",
                "RANK": "0"
            },
            "resources": {
                "cpu": "8",
                "memory": "16Gi",
                "gpu": 1
            }
        }
        
        job_spec = self.training_manager._create_distributed_job_spec(
            "test-master-job",
            "pytorch/pytorch:latest",
            config,
            "master"
        )
        
        assert job_spec["kind"] == "Job"
        assert job_spec["metadata"]["name"] == "test-master-job"
        assert job_spec["metadata"]["labels"]["role"] == "master"
        
        container = job_spec["spec"]["template"]["spec"]["containers"][0]
        assert container["image"] == "pytorch/pytorch:latest"
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"


class TestKubernetesResources:
    """Test Kubernetes resource management"""
    
    def test_kubernetes_resource_creation(self):
        """Test KubernetesResource dataclass"""
        resource = KubernetesResource(
            kind="Job",
            name="test-job",
            namespace="ml-training",
            spec={"replicas": 1}
        )
        
        assert resource.kind == "Job"
        assert resource.name == "test-job"
        assert resource.namespace == "ml-training"
        assert resource.status == "Pending"
        assert resource.created_at is not None
    
    def test_resource_yaml_generation(self):
        """Test generating YAML from resource specifications"""
        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": "test-job",
                "namespace": "ml-training"
            },
            "spec": {
                "template": {
                    "spec": {
                        "containers": [{
                            "name": "trainer",
                            "image": "tensorflow/tensorflow:2.13.0-gpu"
                        }],
                        "restartPolicy": "Never"
                    }
                }
            }
        }
        
        yaml_content = yaml.dump(job_spec)
        parsed_yaml = yaml.safe_load(yaml_content)
        
        assert parsed_yaml["kind"] == "Job"
        assert parsed_yaml["metadata"]["name"] == "test-job"
        assert parsed_yaml["spec"]["template"]["spec"]["restartPolicy"] == "Never"


class TestMLWorkloadPatterns:
    """Test ML workload patterns and configurations"""
    
    def setup_method(self):
        """Set up test environment"""
        self.k8s_client = KubernetesClientManager("test-cluster")
        self.training_manager = MLTrainingManager(self.k8s_client)
    
    def test_single_node_training_pattern(self):
        """Test single-node training job pattern"""
        config = {
            "image": "tensorflow/tensorflow:2.13.0-gpu",
            "command": ["python"],
            "args": ["train.py", "--epochs=100"],
            "resources": {
                "cpu": "8",
                "memory": "32Gi",
                "gpu": 2
            },
            "environment": {
                "TF_CONFIG": json.dumps({
                    "cluster": {"worker": ["localhost:12345"]},
                    "task": {"type": "worker", "index": 0}
                })
            }
        }
        
        job_id = self.training_manager.create_training_job(
            "single-node-training",
            config["image"],
            config
        )
        
        assert job_id is not None
        job_info = self.training_manager.training_jobs["single-node-training"]
        assert job_info["config"]["resources"]["gpu"] == 2
    
    def test_multi_gpu_training_pattern(self):
        """Test multi-GPU training configuration"""
        job_id = self.training_manager.setup_gpu_training_job(
            "multi-gpu-training",
            gpu_count=4,
            gpu_type="nvidia-tesla-a100"
        )
        
        assert job_id is not None
        job_info = self.training_manager.training_jobs["multi-gpu-training"]
        
        # Check GPU configuration
        assert job_info["config"]["resources"]["gpu"] == 4
        assert job_info["config"]["gpu_type"] == "nvidia-tesla-a100"
        
        # Check CPU and memory scaling
        assert job_info["config"]["resources"]["cpu"] == "16"  # 4 CPUs per GPU
        assert job_info["config"]["resources"]["memory"] == "64Gi"  # 16GB per GPU
    
    def test_batch_inference_pattern(self):
        """Test batch inference job pattern"""
        inference_config = {
            "image": "tensorflow/serving:2.13.0-gpu",
            "command": ["python"],
            "args": ["batch_inference.py"],
            "environment": {
                "MODEL_PATH": "/models/text-classifier",
                "INPUT_PATH": "/data/inference-batch",
                "OUTPUT_PATH": "/results/predictions"
            },
            "resources": {
                "cpu": "4",
                "memory": "16Gi",
                "gpu": 1
            },
            "timeout": 3600  # 1 hour timeout
        }
        
        job_id = self.training_manager.create_training_job(
            "batch-inference-job",
            inference_config["image"],
            inference_config
        )
        
        assert job_id is not None
        job_info = self.training_manager.training_jobs["batch-inference-job"]
        assert job_info["config"]["timeout"] == 3600
    
    def test_hyperparameter_optimization_pattern(self):
        """Test hyperparameter optimization with multiple jobs"""
        # Define search space
        search_space = {
            "learning_rate": [1e-4, 1e-3, 1e-2],
            "batch_size": [16, 32, 64],
            "model_size": ["small", "medium", "large"]
        }
        
        job_ids = self.training_manager.create_hyperparameter_tuning_job(
            "hyperopt-experiment",
            search_space
        )
        
        # Should create 3 * 3 * 3 = 27 jobs
        assert len(job_ids) == 27
        
        # Verify all jobs have unique parameter combinations
        created_jobs = [job for job in self.training_manager.training_jobs.keys() 
                       if job.startswith("hyperopt-experiment-hp-")]
        assert len(created_jobs) == 27
        assert len(set(created_jobs)) == 27  # All unique


class TestResourceManagement:
    """Test Kubernetes resource management and optimization"""
    
    def setup_method(self):
        """Set up test environment"""
        self.k8s_client = KubernetesClientManager("test-cluster")
        self.training_manager = MLTrainingManager(self.k8s_client)
    
    def test_resource_quota_compliance(self):
        """Test resource quota compliance"""
        # Create job with specific resource requirements
        config = {
            "resources": {
                "cpu": "16",
                "memory": "64Gi",
                "gpu": 4
            }
        }
        
        job_id = self.training_manager.create_training_job(
            "resource-test-job",
            "tensorflow/tensorflow:2.13.0-gpu",
            config
        )
        
        assert job_id is not None
        
        # Verify resource configuration in stored job
        job_info = self.training_manager.training_jobs["resource-test-job"]
        assert job_info["config"]["resources"]["cpu"] == "16"
        assert job_info["config"]["resources"]["memory"] == "64Gi"
        assert job_info["config"]["resources"]["gpu"] == 4
    
    def test_node_selector_configuration(self):
        """Test node selector configuration for GPU jobs"""
        config = {
            "resources": {"gpu": 2},
            "gpu_type": "nvidia-tesla-v100"
        }
        
        job_id = self.training_manager.create_training_job(
            "node-selector-test",
            "pytorch/pytorch:latest",
            config
        )
        
        assert job_id is not None
        job_info = self.training_manager.training_jobs["node-selector-test"]
        assert job_info["config"]["gpu_type"] == "nvidia-tesla-v100"
    
    def test_volume_mount_configuration(self):
        """Test volume mount configuration for data access"""
        config = {
            "resources": {"cpu": "4", "memory": "8Gi"},
            "volumes": {
                "training_data": "/data",
                "model_output": "/models",
                "checkpoints": "/checkpoints"
            }
        }
        
        job_id = self.training_manager.create_training_job(
            "volume-test-job",
            "tensorflow/tensorflow:2.13.0-gpu",
            config
        )
        
        assert job_id is not None
        # Volume configuration is handled in the YAML generation
        assert "volume-test-job" in self.training_manager.training_jobs


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def setup_method(self):
        """Set up test environment"""
        self.k8s_client = KubernetesClientManager("test-cluster")
        self.training_manager = MLTrainingManager(self.k8s_client)
    
    def test_invalid_job_configuration(self):
        """Test handling of invalid job configurations"""
        # Test with missing required fields
        invalid_config = {}
        
        try:
            job_id = self.training_manager.create_training_job(
                "invalid-job",
                "tensorflow/tensorflow:2.13.0-gpu",
                invalid_config
            )
            # Should still create job with defaults
            assert job_id is not None
        except Exception as e:
            # Or handle gracefully with error
            assert isinstance(e, (ValueError, KeyError))
    
    def test_monitoring_nonexistent_job(self):
        """Test monitoring non-existent job"""
        result = self.training_manager.monitor_training_progress("nonexistent-job")
        
        assert "error" in result
        assert "not found" in result["error"].lower()
    
    def test_empty_hyperparameter_space(self):
        """Test hyperparameter tuning with empty space"""
        empty_space = {}
        
        job_ids = self.training_manager.create_hyperparameter_tuning_job(
            "empty-hp-test",
            empty_space
        )
        
        # Should return empty list for empty space
        assert len(job_ids) == 0
    
    def test_large_hyperparameter_space(self):
        """Test handling of large hyperparameter spaces"""
        large_space = {
            "param1": list(range(10)),
            "param2": list(range(5)),
            "param3": ["a", "b", "c"]
        }
        
        # This would create 10 * 5 * 3 = 150 jobs
        job_ids = self.training_manager.create_hyperparameter_tuning_job(
            "large-hp-test",
            large_space
        )
        
        assert len(job_ids) == 150


class TestIntegrationScenarios:
    """Test integration scenarios and workflows"""
    
    def setup_method(self):
        """Set up test environment"""
        self.k8s_client = KubernetesClientManager("test-cluster")
        self.training_manager = MLTrainingManager(self.k8s_client)
    
    def test_end_to_end_training_workflow(self):
        """Test complete training workflow"""
        # 1. Create training job
        config = {
            "image": "tensorflow/tensorflow:2.13.0-gpu",
            "resources": {"cpu": "8", "memory": "32Gi", "gpu": 2},
            "environment": {"EXPERIMENT_NAME": "e2e-test"}
        }
        
        job_id = self.training_manager.create_training_job(
            "e2e-training",
            config["image"],
            config
        )
        
        assert job_id is not None
        
        # 2. Monitor job progress
        progress = self.training_manager.monitor_training_progress("e2e-training")
        assert progress["job_name"] == "e2e-training"
        
        # 3. Verify job is tracked
        assert "e2e-training" in self.training_manager.training_jobs
    
    def test_multi_experiment_management(self):
        """Test managing multiple experiments simultaneously"""
        experiments = [
            {"name": "exp-1", "lr": 0.001},
            {"name": "exp-2", "lr": 0.01},
            {"name": "exp-3", "lr": 0.1}
        ]
        
        job_ids = []
        for exp in experiments:
            config = {
                "resources": {"cpu": "4", "memory": "16Gi"},
                "environment": {"LEARNING_RATE": str(exp["lr"])}
            }
            
            job_id = self.training_manager.create_training_job(
                exp["name"],
                "tensorflow/tensorflow:2.13.0-gpu",
                config
            )
            job_ids.append(job_id)
        
        assert len(job_ids) == 3
        assert len(self.training_manager.training_jobs) == 3
        
        # Monitor all experiments
        for exp in experiments:
            progress = self.training_manager.monitor_training_progress(exp["name"])
            assert progress["job_name"] == exp["name"]
    
    def test_distributed_training_coordination(self):
        """Test distributed training with proper coordination"""
        config = {
            "image": "pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime",
            "resources": {"cpu": "16", "memory": "64Gi", "gpu": 4}
        }
        
        job_id = self.training_manager.create_distributed_training_job(
            "distributed-coordination-test",
            worker_count=4,
            training_config=config
        )
        
        assert job_id is not None
        
        job_info = self.training_manager.training_jobs["distributed-coordination-test"]
        assert job_info["type"] == "distributed"
        assert job_info["worker_count"] == 4
        assert job_info["master_job_id"] is not None
        assert len(job_info["worker_job_ids"]) == 4


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])