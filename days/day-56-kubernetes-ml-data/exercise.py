"""
Day 56: Kubernetes for ML & Data Workloads - Orchestration, Scaling & Management
Exercises for deploying and managing ML workloads on Kubernetes
"""

import yaml
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class KubernetesResource:
    """Kubernetes resource representation"""
    kind: str
    name: str
    namespace: str
    spec: Dict[str, Any]
    status: str = "Pending"


class MockKubernetesClient:
    """Mock Kubernetes client for exercises"""
    
    def __init__(self, cluster_name: str = "ml-cluster"):
        self.cluster_name = cluster_name
        self.resources = {}
        self.namespaces = {"default", "ml-training", "ml-serving", "data-processing"}
    
    def create_resource(self, resource: Dict[str, Any]) -> str:
        """Simulate resource creation"""
        kind = resource.get("kind", "Unknown")
        name = resource.get("metadata", {}).get("name", "unnamed")
        namespace = resource.get("metadata", {}).get("namespace", "default")
        
        resource_id = f"{namespace}/{kind}/{name}"
        self.resources[resource_id] = {
            "resource": resource,
            "status": "Creating",
            "created_at": datetime.now()
        }
        
        logger.info(f"Created {kind}: {name} in namespace {namespace}")
        return resource_id
    
    def create_resource_from_yaml(self, yaml_content: str, namespace: str = "default") -> str:
        """Create Kubernetes resource from YAML"""
        try:
            resource = yaml.safe_load(yaml_content)
            # Set namespace if not specified
            if "namespace" not in resource.get("metadata", {}):
                resource["metadata"]["namespace"] = namespace
            return self.create_resource(resource)
        except Exception as e:
            logger.error(f"Failed to create resource from YAML: {e}")
            raise
    
    def get_resource(self, resource_id: str) -> Dict:
        """Get resource by ID"""
        return self.resources.get(resource_id, {})
    
    def list_resources(self, kind: Optional[str] = None, namespace: Optional[str] = None) -> List[Dict]:
        """List resources with optional filtering"""
        resources = []
        for resource_id, resource_data in self.resources.items():
            ns, res_kind, name = resource_id.split("/")
            
            if kind and res_kind != kind:
                continue
            if namespace and ns != namespace:
                continue
            
            resources.append(resource_data)
        
        return resources


# Exercise 1: ML Training Jobs
def exercise_1_ml_training_jobs():
    """
    Exercise 1: Create and manage ML training jobs on Kubernetes
    
    TODO: Complete the MLTrainingManager class
    """
    print("=== Exercise 1: ML Training Jobs ===")
    
    class MLTrainingManager:
        def __init__(self, k8s_client: MockKubernetesClient):
            self.k8s_client = k8s_client
        
        def create_training_job(self, job_name: str, image: str, 
                              training_config: Dict[str, Any]) -> str:
            """Create Kubernetes Job for ML training"""
            # Prepare environment variables
            env_vars = []
            for key, value in training_config.get("environment", {}).items():
                env_vars.append({"name": key, "value": str(value)})
            
            # Configure resources
            resources = training_config.get("resources", {})
            resource_requests = {
                "cpu": resources.get("cpu", "4"),
                "memory": resources.get("memory", "8Gi")
            }
            
            if resources.get("gpu", 0) > 0:
                resource_requests["nvidia.com/gpu"] = str(resources["gpu"])
            
            # Create Job specification
            job_spec = {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {
                    "name": job_name,
                    "labels": {"app": "ml-training", "job-type": "training"}
                },
                "spec": {
                    "backoffLimit": 3,
                    "template": {
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [{
                                "name": "trainer",
                                "image": image,
                                "command": training_config.get("command", ["python"]),
                                "args": training_config.get("args", ["train.py"]),
                                "env": env_vars,
                                "resources": {"requests": resource_requests, "limits": resource_requests}
                            }]
                        }
                    }
                }
            }
            
            job_yaml = yaml.dump(job_spec)
            return self.k8s_client.create_resource_from_yaml(job_yaml, "ml-training")
        
        def create_distributed_training_job(self, job_name: str, 
                                          worker_count: int, 
                                          training_config: Dict[str, Any]) -> str:
            """Create distributed training job with multiple workers"""
            # Create master service for coordination
            service_spec = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": f"{job_name}-master"},
                "spec": {
                    "selector": {"job": job_name, "role": "master"},
                    "ports": [{"port": 29500, "targetPort": 29500}],
                    "clusterIP": "None"
                }
            }
            
            # Create distributed job with parallelism
            job_spec = {
                "apiVersion": "batch/v1",
                "kind": "Job",
                "metadata": {"name": job_name},
                "spec": {
                    "parallelism": worker_count,
                    "completions": worker_count,
                    "template": {
                        "metadata": {"labels": {"job": job_name}},
                        "spec": {
                            "restartPolicy": "Never",
                            "containers": [{
                                "name": "worker",
                                "image": training_config.get("image", "pytorch/pytorch:latest"),
                                "env": [
                                    {"name": "WORLD_SIZE", "value": str(worker_count)},
                                    {"name": "MASTER_ADDR", "value": f"{job_name}-master"},
                                    {"name": "MASTER_PORT", "value": "29500"}
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": training_config.get("resources", {}).get("cpu", "8"),
                                        "memory": training_config.get("resources", {}).get("memory", "16Gi")
                                    }
                                }
                            }]
                        }
                    }
                }
            }
            
            # Create service first, then job
            service_yaml = yaml.dump(service_spec)
            self.k8s_client.create_resource_from_yaml(service_yaml, "ml-training")
            
            job_yaml = yaml.dump(job_spec)
            return self.k8s_client.create_resource_from_yaml(job_yaml, "ml-training")
        
        def create_hyperparameter_tuning_job(self, base_job_name: str,
                                           hyperparameter_space: Dict[str, List]) -> List[str]:
            """Create multiple jobs for hyperparameter tuning"""
            import itertools
            import hashlib
            
            # Generate parameter combinations
            param_names = list(hyperparameter_space.keys())
            param_values = list(hyperparameter_space.values())
            param_combinations = list(itertools.product(*param_values))
            
            job_ids = []
            for i, combination in enumerate(param_combinations):
                params = dict(zip(param_names, combination))
                param_hash = hashlib.md5(str(params).encode()).hexdigest()[:8]
                job_name = f"{base_job_name}-hp-{param_hash}"
                
                # Create training config with hyperparameters
                training_config = {
                    "command": ["python"],
                    "args": ["train.py"] + [f"--{k}={v}" for k, v in params.items()],
                    "environment": {f"HP_{k.upper()}": str(v) for k, v in params.items()},
                    "resources": {"cpu": "4", "memory": "8Gi", "gpu": 1}
                }
                
                job_id = self.create_training_job(
                    job_name, 
                    "tensorflow/tensorflow:2.13.0-gpu", 
                    training_config
                )
                job_ids.append(job_id)
                logger.info(f"Created HP job {i+1}/{len(param_combinations)}: {job_name}")
            
            return job_ids
        
        def monitor_training_progress(self, job_name: str) -> Dict[str, Any]:
            """Monitor training job progress and metrics"""
            try:
                # Simulate job monitoring (in production, query K8s API)
                resource_id = f"ml-training/Job/{job_name}"
                if resource_id not in self.k8s_client.resources:
                    return {"error": f"Job {job_name} not found"}
                
                job_data = self.k8s_client.resources[resource_id]
                created_at = job_data["created_at"]
                elapsed_time = datetime.now() - created_at
                
                # Simulate progress based on elapsed time
                if elapsed_time.total_seconds() < 60:
                    status = "Running"
                    progress = min(elapsed_time.total_seconds() / 60 * 100, 100)
                else:
                    status = "Completed"
                    progress = 100
                
                return {
                    "job_name": job_name,
                    "status": status,
                    "progress_percent": progress,
                    "elapsed_time_seconds": elapsed_time.total_seconds(),
                    "resource_usage": {
                        "cpu_usage_percent": 75.5,
                        "memory_usage_gb": 12.3,
                        "gpu_usage_percent": 89.2
                    },
                    "training_metrics": {
                        "current_epoch": min(int(progress / 10), 10),
                        "loss": 0.234,
                        "accuracy": 0.876
                    }
                }
            except Exception as e:
                return {"error": str(e)}
        
        def setup_gpu_training_job(self, job_name: str, gpu_count: int,
                                 gpu_type: str = "nvidia-tesla-v100") -> str:
            """Create GPU-enabled training job"""
            training_config = {
                "command": ["python"],
                "args": ["train_gpu.py"],
                "environment": {
                    "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(gpu_count)),
                    "NVIDIA_VISIBLE_DEVICES": "all",
                    "TF_FORCE_GPU_ALLOW_GROWTH": "true"
                },
                "resources": {
                    "cpu": str(gpu_count * 4),  # 4 CPUs per GPU
                    "memory": f"{gpu_count * 16}Gi",  # 16GB RAM per GPU
                    "gpu": gpu_count
                },
                "node_selector": {"accelerator": gpu_type},
                "tolerations": [{
                    "key": "nvidia.com/gpu",
                    "operator": "Exists",
                    "effect": "NoSchedule"
                }]
            }
            
            return self.create_training_job(
                job_name, 
                "tensorflow/tensorflow:2.13.0-gpu", 
                training_config
            )
    
    # Test ML training jobs
    k8s_client = MockKubernetesClient()
    training_manager = MLTrainingManager(k8s_client)
    
    print("Testing ML Training Jobs...")
    print("\n--- Your implementation should create training jobs ---")
    
    # Example training configuration
    training_config = {
        "dataset_path": "/data/training-set",
        "model_output_path": "/models/output",
        "epochs": 100,
        "batch_size": 32,
        "learning_rate": 0.001,
        "resources": {
            "cpu": "8",
            "memory": "16Gi",
            "gpu": 2
        }
    }
    
    # job_id = training_manager.create_training_job(
    #     job_name="text-classifier-training",
    #     image="tensorflow/tensorflow:2.13.0-gpu",
    #     training_config=training_config
    # )
    # print(f"Training job created: {job_id}")


# Exercise 2: Model Serving Deployments
def exercise_2_model_serving():
    """
    Exercise 2: Deploy ML models for serving with auto-scaling
    
    TODO: Complete the ModelServingManager class
    """
    print("\n=== Exercise 2: Model Serving Deployments ===")
    
    class ModelServingManager:
        def __init__(self, k8s_client: MockKubernetesClient):
            self.k8s_client = k8s_client
        
        def create_model_deployment(self, model_name: str, model_image: str,
                                  serving_config: Dict[str, Any]) -> str:
            """Create Deployment for model serving"""
            deployment_spec = {
                "apiVersion": "apps/v1",
                "kind": "Deployment",
                "metadata": {
                    "name": f"{model_name}-deployment",
                    "labels": {"app": model_name, "type": "model-serving"}
                },
                "spec": {
                    "replicas": serving_config.get("replicas", 3),
                    "selector": {"matchLabels": {"app": model_name}},
                    "template": {
                        "metadata": {"labels": {"app": model_name}},
                        "spec": {
                            "containers": [{
                                "name": "model-server",
                                "image": model_image,
                                "ports": [{"containerPort": 8080}],
                                "env": [
                                    {"name": "MODEL_NAME", "value": model_name},
                                    {"name": "MODEL_PATH", "value": serving_config.get("model_path", "/models")}
                                ],
                                "resources": {
                                    "requests": {
                                        "cpu": serving_config.get("resources", {}).get("cpu", "2"),
                                        "memory": serving_config.get("resources", {}).get("memory", "4Gi")
                                    },
                                    "limits": {
                                        "cpu": serving_config.get("resources", {}).get("cpu", "4"),
                                        "memory": serving_config.get("resources", {}).get("memory", "8Gi")
                                    }
                                },
                                "readinessProbe": {
                                    "httpGet": {"path": "/health", "port": 8080},
                                    "initialDelaySeconds": 30,
                                    "periodSeconds": 10
                                },
                                "livenessProbe": {
                                    "httpGet": {"path": "/health", "port": 8080},
                                    "initialDelaySeconds": 60,
                                    "periodSeconds": 30
                                }
                            }]
                        }
                    }
                }
            }
            
            deployment_yaml = yaml.dump(deployment_spec)
            return self.k8s_client.create_resource_from_yaml(deployment_yaml, "ml-serving")
        
        def create_model_service(self, model_name: str, port: int = 8080) -> str:
            """Create Service to expose model deployment"""
            service_spec = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {
                    "name": f"{model_name}-service",
                    "labels": {"app": model_name},
                    "annotations": {
                        "prometheus.io/scrape": "true",
                        "prometheus.io/port": "9090",
                        "prometheus.io/path": "/metrics"
                    }
                },
                "spec": {
                    "selector": {"app": model_name},
                    "ports": [
                        {"port": port, "targetPort": port, "name": "http"},
                        {"port": 9090, "targetPort": 9090, "name": "metrics"}
                    ],
                    "type": "ClusterIP"
                }
            }
            
            service_yaml = yaml.dump(service_spec)
            return self.k8s_client.create_resource_from_yaml(service_yaml, "ml-serving")
        
        def setup_auto_scaling(self, deployment_name: str, 
                             min_replicas: int = 2, max_replicas: int = 20,
                             target_cpu: int = 70) -> str:
            """Configure Horizontal Pod Autoscaler"""
            hpa_spec = {
                "apiVersion": "autoscaling/v2",
                "kind": "HorizontalPodAutoscaler",
                "metadata": {
                    "name": f"{deployment_name}-hpa",
                    "labels": {"app": deployment_name}
                },
                "spec": {
                    "scaleTargetRef": {
                        "apiVersion": "apps/v1",
                        "kind": "Deployment",
                        "name": deployment_name
                    },
                    "minReplicas": min_replicas,
                    "maxReplicas": max_replicas,
                    "metrics": [
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "cpu",
                                "target": {"type": "Utilization", "averageUtilization": target_cpu}
                            }
                        },
                        {
                            "type": "Resource",
                            "resource": {
                                "name": "memory",
                                "target": {"type": "Utilization", "averageUtilization": 80}
                            }
                        }
                    ],
                    "behavior": {
                        "scaleDown": {
                            "stabilizationWindowSeconds": 300,
                            "policies": [{"type": "Percent", "value": 10, "periodSeconds": 60}]
                        },
                        "scaleUp": {
                            "stabilizationWindowSeconds": 60,
                            "policies": [{"type": "Percent", "value": 50, "periodSeconds": 60}]
                        }
                    }
                }
            }
            
            hpa_yaml = yaml.dump(hpa_spec)
            return self.k8s_client.create_resource_from_yaml(hpa_yaml, "ml-serving")
        
        def create_canary_deployment(self, model_name: str, 
                                   stable_version: str, canary_version: str,
                                   traffic_split: float = 0.1) -> Dict[str, str]:
            """Set up canary deployment for A/B testing"""
            # Create stable deployment
            stable_config = {
                "replicas": int(10 * (1 - traffic_split)),
                "model_path": f"/models/{model_name}/{stable_version}",
                "resources": {"cpu": "2", "memory": "4Gi"}
            }
            stable_id = self.create_model_deployment(
                f"{model_name}-stable",
                f"model-server:{stable_version}",
                stable_config
            )
            
            # Create canary deployment
            canary_config = {
                "replicas": int(10 * traffic_split),
                "model_path": f"/models/{model_name}/{canary_version}",
                "resources": {"cpu": "2", "memory": "4Gi"}
            }
            canary_id = self.create_model_deployment(
                f"{model_name}-canary",
                f"model-server:{canary_version}",
                canary_config
            )
            
            # Create unified service for both deployments
            service_spec = {
                "apiVersion": "v1",
                "kind": "Service",
                "metadata": {"name": f"{model_name}-canary-service"},
                "spec": {
                    "selector": {"app": model_name},  # Matches both deployments
                    "ports": [{"port": 8080, "targetPort": 8080}]
                }
            }
            
            service_yaml = yaml.dump(service_spec)
            service_id = self.k8s_client.create_resource_from_yaml(service_yaml, "ml-serving")
            
            return {
                "stable_deployment": stable_id,
                "canary_deployment": canary_id,
                "service": service_id,
                "traffic_split": str(traffic_split)
            }
        
        def setup_model_monitoring(self, model_name: str) -> str:
            """Set up monitoring for model serving"""
            # Create ServiceMonitor for Prometheus
            service_monitor_spec = {
                "apiVersion": "monitoring.coreos.com/v1",
                "kind": "ServiceMonitor",
                "metadata": {
                    "name": f"{model_name}-metrics",
                    "labels": {"app": model_name}
                },
                "spec": {
                    "selector": {"matchLabels": {"app": model_name}},
                    "endpoints": [{
                        "port": "metrics",
                        "interval": "30s",
                        "path": "/metrics"
                    }]
                }
            }
            
            # Create PrometheusRule for alerts
            alert_rule_spec = {
                "apiVersion": "monitoring.coreos.com/v1",
                "kind": "PrometheusRule",
                "metadata": {"name": f"{model_name}-alerts"},
                "spec": {
                    "groups": [{
                        "name": f"{model_name}-serving",
                        "rules": [
                            {
                                "alert": "HighModelLatency",
                                "expr": f"histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket{{model=\"{model_name}\"}}[5m])) > 1.0",
                                "for": "2m",
                                "labels": {"severity": "warning"},
                                "annotations": {"summary": "High model inference latency"}
                            },
                            {
                                "alert": "ModelAccuracyDrop",
                                "expr": f"model_accuracy{{model=\"{model_name}\"}} < 0.85",
                                "for": "5m",
                                "labels": {"severity": "critical"},
                                "annotations": {"summary": "Model accuracy dropped below threshold"}
                            }
                        ]
                    }]
                }
            }
            
            # Create resources
            monitor_yaml = yaml.dump(service_monitor_spec)
            monitor_id = self.k8s_client.create_resource_from_yaml(monitor_yaml, "monitoring")
            
            alert_yaml = yaml.dump(alert_rule_spec)
            self.k8s_client.create_resource_from_yaml(alert_yaml, "monitoring")
            
            return monitor_id
    
    # Test model serving
    serving_manager = ModelServingManager(k8s_client)
    
    print("Testing Model Serving...")
    print("\n--- Your implementation should create serving infrastructure ---")
    
    serving_config = {
        "model_path": "/models/text-classifier/v1",
        "batch_size": 32,
        "max_batch_delay": "100ms",
        "resources": {
            "cpu": "4",
            "memory": "8Gi",
            "gpu": 1
        }
    }
    
    # deployment_id = serving_manager.create_model_deployment(
    #     model_name="text-classifier",
    #     model_image="tensorflow/serving:2.13.0-gpu",
    #     serving_config=serving_config
    # )
    # print(f"Model deployment created: {deployment_id}")


# Exercise 3: Kubeflow Pipelines
def exercise_3_kubeflow_pipelines():
    """
    Exercise 3: Create ML pipelines using Kubeflow
    
    TODO: Complete the KubeflowPipelineManager class
    """
    print("\n=== Exercise 3: Kubeflow ML Pipelines ===")
    
    class KubeflowPipelineManager:
        def __init__(self, k8s_client: MockKubernetesClient):
            self.k8s_client = k8s_client
        
        def create_pipeline_workflow(self, pipeline_name: str, 
                                   pipeline_steps: List[Dict]) -> str:
            """Create Argo Workflow for ML pipeline"""
            # Build workflow templates from steps
            templates = []
            dag_tasks = []
            
            for step in pipeline_steps:
                template = {
                    "name": step["name"],
                    "container": {
                        "image": step["image"],
                        "command": ["python"],
                        "args": [step["script"]],
                        "resources": {
                            "requests": {"cpu": "2", "memory": "4Gi"}
                        }
                    }
                }
                templates.append(template)
                
                # Create DAG task
                task = {"name": step["name"], "template": step["name"]}
                if "depends_on" in step:
                    task["dependencies"] = step["depends_on"]
                dag_tasks.append(task)
            
            # Create main DAG template
            dag_template = {
                "name": "ml-pipeline",
                "dag": {"tasks": dag_tasks}
            }
            templates.append(dag_template)
            
            # Create workflow specification
            workflow_spec = {
                "apiVersion": "argoproj.io/v1alpha1",
                "kind": "Workflow",
                "metadata": {
                    "name": pipeline_name,
                    "labels": {"pipeline": "ml-training"}
                },
                "spec": {
                    "entrypoint": "ml-pipeline",
                    "templates": templates,
                    "ttlStrategy": {"secondsAfterCompletion": 86400}  # 24 hours
                }
            }
            
            workflow_yaml = yaml.dump(workflow_spec)
            return self.k8s_client.create_resource_from_yaml(workflow_yaml, "argo")
        
        def create_data_preprocessing_step(self, step_config: Dict) -> Dict:
            """Create data preprocessing pipeline step"""
            return {
                "name": "data-preprocessing",
                "container": {
                    "image": step_config.get("image", "python:3.9"),
                    "command": ["python"],
                    "args": [step_config.get("script", "preprocess.py")],
                    "resources": {
                        "requests": {"cpu": "2", "memory": "4Gi"}
                    },
                    "env": [
                        {"name": "INPUT_PATH", "value": step_config.get("input_path", "/data/raw")},
                        {"name": "OUTPUT_PATH", "value": step_config.get("output_path", "/data/processed")}
                    ]
                },
                "outputs": {
                    "artifacts": [{
                        "name": "processed-data",
                        "path": "/data/processed"
                    }]
                }
            }
        
        def create_training_step(self, training_config: Dict) -> Dict:
            """Create model training pipeline step"""
            return {
                "name": "model-training",
                "container": {
                    "image": training_config.get("image", "tensorflow/tensorflow:2.13.0-gpu"),
                    "command": ["python"],
                    "args": [training_config.get("script", "train.py")],
                    "resources": {
                        "requests": {
                            "cpu": "8",
                            "memory": "32Gi",
                            "nvidia.com/gpu": str(training_config.get("gpu_count", 2))
                        }
                    },
                    "env": [
                        {"name": "EPOCHS", "value": str(training_config.get("epochs", 100))},
                        {"name": "BATCH_SIZE", "value": str(training_config.get("batch_size", 32))}
                    ]
                },
                "inputs": {
                    "artifacts": [{
                        "name": "training-data",
                        "path": "/data/training"
                    }]
                },
                "outputs": {
                    "artifacts": [{
                        "name": "trained-model",
                        "path": "/models/output"
                    }]
                }
            }
        
        def create_evaluation_step(self, evaluation_config: Dict) -> Dict:
            logger.info("Method implementation needed")
        
        def create_deployment_step(self, deployment_config: Dict) -> Dict:
            logger.info("Method implementation needed")
        
        def monitor_pipeline_execution(self, pipeline_name: str) -> Dict:
            logger.info("Method implementation needed")
    
    # Test Kubeflow pipelines
    pipeline_manager = KubeflowPipelineManager(k8s_client)
    
    print("Testing Kubeflow Pipelines...")
    print("\n--- Your implementation should create ML pipelines ---")
    
    pipeline_steps = [
        {
            "name": "data-preprocessing",
            "type": "preprocessing",
            "image": "python:3.9",
            "script": "preprocess.py"
        },
        {
            "name": "model-training",
            "type": "training",
            "image": "tensorflow/tensorflow:2.13.0-gpu",
            "script": "train.py",
            "depends_on": ["data-preprocessing"]
        },
        {
            "name": "model-evaluation",
            "type": "evaluation",
            "image": "python:3.9",
            "script": "evaluate.py",
            "depends_on": ["model-training"]
        }
    ]
    
    # pipeline_id = pipeline_manager.create_pipeline_workflow(
    #     pipeline_name="ml-training-pipeline",
    #     pipeline_steps=pipeline_steps
    # )
    # print(f"Pipeline created: {pipeline_id}")


# Exercise 4: Data Processing with Spark
def exercise_4_spark_on_kubernetes():
    """
    Exercise 4: Deploy Spark applications for data processing
    
    TODO: Complete the SparkOnKubernetesManager class
    """
    print("\n=== Exercise 4: Spark on Kubernetes ===")
    
    class SparkOnKubernetesManager:
        def __init__(self, k8s_client: MockKubernetesClient):
            self.k8s_client = k8s_client
        
        def create_spark_application(self, app_name: str, 
                                   spark_config: Dict[str, Any]) -> str:
            """Create SparkApplication custom resource"""
            spark_app_spec = {
                "apiVersion": "sparkoperator.k8s.io/v1beta2",
                "kind": "SparkApplication",
                "metadata": {
                    "name": app_name,
                    "namespace": "data-processing"
                },
                "spec": {
                    "type": "Python",
                    "pythonVersion": "3",
                    "mode": "cluster",
                    "image": "apache/spark-py:v3.5.0",
                    "mainApplicationFile": spark_config.get("main_file", "s3a://spark-jobs/app.py"),
                    "sparkVersion": "3.5.0",
                    "driver": {
                        "cores": spark_config.get("driver", {}).get("cores", 2),
                        "memory": spark_config.get("driver", {}).get("memory", "4g"),
                        "serviceAccount": "spark-driver"
                    },
                    "executor": {
                        "cores": spark_config.get("executor", {}).get("cores", 4),
                        "instances": spark_config.get("executor", {}).get("instances", 3),
                        "memory": spark_config.get("executor", {}).get("memory", "8g")
                    }
                }
            }
            
            spark_yaml = yaml.dump(spark_app_spec)
            return self.k8s_client.create_resource_from_yaml(spark_yaml, "data-processing")
        
        def create_spark_cluster(self, cluster_name: str, 
                               worker_count: int = 3) -> str:
            logger.info("Method implementation needed")
        
        def submit_data_processing_job(self, job_config: Dict) -> str:
            logger.info("Method implementation needed")
        
        def setup_spark_streaming_job(self, streaming_config: Dict) -> str:
            logger.info("Method implementation needed")
        
        def monitor_spark_applications(self, namespace: str = "data-processing") -> List[Dict]:
            logger.info("Method implementation needed")
    
    # Test Spark on Kubernetes
    spark_manager = SparkOnKubernetesManager(k8s_client)
    
    print("Testing Spark on Kubernetes...")
    print("\n--- Your implementation should create Spark applications ---")
    
    spark_config = {
        "main_application_file": "s3a://spark-jobs/data_processing.py",
        "driver": {
            "cores": 2,
            "memory": "8g"
        },
        "executor": {
            "cores": 4,
            "instances": 5,
            "memory": "16g"
        },
        "dependencies": [
            "s3a://spark-deps/hadoop-aws-3.3.4.jar"
        ]
    }
    
    # app_id = spark_manager.create_spark_application(
    #     app_name="data-processing-job",
    #     spark_config=spark_config
    # )
    # print(f"Spark application created: {app_id}")


# Exercise 5: GPU Resource Management
def exercise_5_gpu_management():
    """
    Exercise 5: Manage GPU resources for ML workloads
    
    TODO: Complete the GPUResourceManager class
    """
    print("\n=== Exercise 5: GPU Resource Management ===")
    
    class GPUResourceManager:
        def __init__(self, k8s_client: MockKubernetesClient):
            self.k8s_client = k8s_client
        
        def create_gpu_node_pool(self, pool_name: str, gpu_type: str, 
                               node_count: int) -> str:
            logger.info("Method implementation needed")
        
        def setup_gpu_resource_quota(self, namespace: str, 
                                   max_gpus: int) -> str:
            """Set up GPU resource quotas"""
            quota_spec = {
                "apiVersion": "v1",
                "kind": "ResourceQuota",
                "metadata": {
                    "name": "gpu-quota",
                    "namespace": namespace
                },
                "spec": {
                    "hard": {
                        "requests.nvidia.com/gpu": str(max_gpus),
                        "limits.nvidia.com/gpu": str(max_gpus)
                    }
                }
            }
            
            quota_yaml = yaml.dump(quota_spec)
            return self.k8s_client.create_resource_from_yaml(quota_yaml, namespace)
        
        def create_multi_gpu_training_job(self, job_name: str, 
                                        gpu_count: int, 
                                        training_config: Dict) -> str:
            logger.info("Method implementation needed")
        
        def setup_gpu_sharing(self, sharing_config: Dict) -> str:
            logger.info("Method implementation needed")
        
        def monitor_gpu_utilization(self, namespace: str = None) -> Dict:
            logger.info("Method implementation needed")
    
    # Test GPU management
    gpu_manager = GPUResourceManager(k8s_client)
    
    print("Testing GPU Resource Management...")
    print("\n--- Your implementation should manage GPU resources ---")
    
    gpu_config = {
        "gpu_type": "nvidia-tesla-v100",
        "memory_per_gpu": "32Gi",
        "sharing_strategy": "time-slicing",
        "max_shared_clients": 4
    }
    
    # quota_id = gpu_manager.setup_gpu_resource_quota(
    #     namespace="ml-training",
    #     max_gpus=8
    # )
    # print(f"GPU quota created: {quota_id}")


# Exercise 6: Monitoring and Observability
def exercise_6_monitoring_setup():
    """
    Exercise 6: Set up monitoring for ML workloads
    
    TODO: Complete the MLMonitoringManager class
    """
    print("\n=== Exercise 6: ML Workload Monitoring ===")
    
    class MLMonitoringManager:
        def __init__(self, k8s_client: MockKubernetesClient):
            self.k8s_client = k8s_client
        
        def setup_prometheus_monitoring(self, monitoring_config: Dict) -> str:
            """Set up Prometheus for ML metrics collection"""
            service_monitor_spec = {
                "apiVersion": "monitoring.coreos.com/v1",
                "kind": "ServiceMonitor",
                "metadata": {
                    "name": "ml-metrics",
                    "namespace": "monitoring"
                },
                "spec": {
                    "selector": {
                        "matchLabels": {"app": "ml-workload"}
                    },
                    "endpoints": [{
                        "port": "metrics",
                        "interval": monitoring_config.get("interval", "30s"),
                        "path": "/metrics"
                    }]
                }
            }
            
            monitor_yaml = yaml.dump(service_monitor_spec)
            return self.k8s_client.create_resource_from_yaml(monitor_yaml, "monitoring")
        
        def create_ml_dashboards(self, dashboard_configs: List[Dict]) -> List[str]:
            logger.info("Method implementation needed")
        
        def setup_ml_alerting(self, alert_configs: List[Dict]) -> str:
            logger.info("Method implementation needed")
        
        def setup_distributed_tracing(self, tracing_config: Dict) -> str:
            logger.info("Method implementation needed")
        
        def create_ml_metrics_exporter(self, service_name: str, 
                                     metrics_config: Dict) -> str:
            logger.info("Method implementation needed")
    
    # Test monitoring setup
    monitoring_manager = MLMonitoringManager(k8s_client)
    
    print("Testing ML Monitoring Setup...")
    print("\n--- Your implementation should set up comprehensive monitoring ---")
    
    monitoring_config = {
        "metrics_interval": "30s",
        "retention_period": "30d",
        "alert_thresholds": {
            "model_latency_p95": 1.0,
            "model_accuracy": 0.85,
            "gpu_utilization": 90
        }
    }
    
    # monitoring_id = monitoring_manager.setup_prometheus_monitoring(monitoring_config)
    # print(f"Monitoring setup created: {monitoring_id}")


# Exercise 7: Complete ML Platform Deployment
def exercise_7_ml_platform():
    """
    Exercise 7: Deploy complete ML platform on Kubernetes
    
    TODO: Complete the MLPlatformManager class
    """
    print("\n=== Exercise 7: Complete ML Platform Deployment ===")
    
    class MLPlatformManager:
        def __init__(self, k8s_client: MockKubernetesClient):
            self.k8s_client = k8s_client
            self.components = {}
        
        def deploy_kubeflow_platform(self, platform_config: Dict) -> Dict[str, str]:
            """Deploy complete Kubeflow platform"""
            deployment_result = {
                "kubeflow_namespace": "kubeflow",
                "pipelines": "deployed",
                "notebooks": "deployed" if platform_config.get("enable_notebooks") else "disabled",
                "katib": "deployed" if platform_config.get("enable_katib") else "disabled",
                "kfserving": "deployed" if platform_config.get("enable_kfserving") else "disabled",
                "status": "deployed"
            }
            
            logger.info(f"Kubeflow platform deployed with config: {platform_config}")
            return deployment_result
        
        def setup_ml_data_pipeline(self, pipeline_config: Dict) -> str:
            logger.info("Method implementation needed")
        
        def deploy_model_registry(self, registry_config: Dict) -> str:
            logger.info("Method implementation needed")
        
        def setup_feature_store(self, feature_store_config: Dict) -> str:
            logger.info("Method implementation needed")
        
        def configure_ml_security(self, security_config: Dict) -> Dict[str, str]:
            logger.info("Method implementation needed")
        
        def validate_platform_deployment(self) -> Dict[str, Any]:
            logger.info("Method implementation needed")
    
    # Test complete ML platform
    platform_manager = MLPlatformManager(k8s_client)
    
    print("Testing Complete ML Platform...")
    print("\n--- Your implementation should deploy full ML platform ---")
    
    platform_config = {
        "kubeflow_version": "1.7.0",
        "enable_notebooks": True,
        "enable_pipelines": True,
        "enable_katib": True,
        "enable_kfserving": True,
        "storage_class": "fast-ssd",
        "ingress_domain": "ml-platform.company.com"
    }
    
    # deployment_result = platform_manager.deploy_kubeflow_platform(platform_config)
    # print(f"ML Platform deployment: {deployment_result}")


def main():
    """Run all Kubernetes ML exercises"""
    print("⚙️ Day 56: Kubernetes for ML & Data Workloads - Orchestration, Scaling & Management")
    print("=" * 90)
    
    exercises = [
        exercise_1_ml_training_jobs,
        exercise_2_model_serving,
        exercise_3_kubeflow_pipelines,
        exercise_4_spark_on_kubernetes,
        exercise_5_gpu_management,
        exercise_6_monitoring_setup,
        exercise_7_ml_platform
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n📋 Starting Exercise {i}")
        try:
            exercise()
            print(f"✅ Exercise {i} setup complete")
        except Exception as e:
            print(f"❌ Exercise {i} error: {e}")
        
        if i < len(exercises):
            input("\nPress Enter to continue to the next exercise...")
    
    print("\n🎉 All exercises completed!")
    print("\nNext steps:")
    print("1. Implement the TODO sections in each exercise")
    print("2. Test your implementations with a real Kubernetes cluster")
    print("3. Deploy actual ML workloads and monitor their performance")
    print("4. Review the solution file for complete implementations")
    print("5. Experiment with different ML frameworks and operators")
    
    print("\n🚀 Production Deployment Checklist:")
    print("• Set up proper RBAC and security policies")
    print("• Configure resource quotas and limits")
    print("• Implement comprehensive monitoring and alerting")
    print("• Set up backup and disaster recovery")
    print("• Configure auto-scaling policies")
    print("• Implement CI/CD pipelines for ML workloads")
    print("• Set up cost monitoring and optimization")


if __name__ == "__main__":
    main()
