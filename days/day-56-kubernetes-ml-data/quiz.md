# Day 56 Quiz: Kubernetes for ML & Data Workloads - Orchestration, Scaling & Management

## Instructions
Answer all questions. Each question has one correct answer.

---

1. **What is the primary advantage of using Kubernetes Jobs for ML training workloads?**
   - A) They provide persistent storage for models
   - B) They ensure completion of batch workloads with automatic retry and failure handling
   - C) They only work with GPU resources
   - D) They replace the need for container images

2. **Which Kubernetes resource is most appropriate for serving ML models in production?**
   - A) Jobs for all serving workloads
   - B) Deployments with Services and Horizontal Pod Autoscaler for scalable serving
   - C) DaemonSets for model distribution
   - D) StatefulSets for all ML workloads

3. **What is the main benefit of using Kubeflow for ML workflows?**
   - A) It only works with TensorFlow models
   - B) It provides a complete ML platform with pipelines, training operators, and serving capabilities
   - C) It eliminates the need for Kubernetes
   - D) It only supports CPU-based workloads

4. **How should GPU resources be managed in Kubernetes for ML workloads?**
   - A) GPUs cannot be used in Kubernetes
   - B) Use resource requests/limits with nvidia.com/gpu and appropriate node selectors and tolerations
   - C) Only use CPU resources for ML workloads
   - D) GPUs are automatically managed without configuration

5. **What is the purpose of Seldon Core in ML deployments?**
   - A) To replace Kubernetes entirely
   - B) To provide advanced model serving capabilities with A/B testing, canary deployments, and explainability
   - C) To only handle data preprocessing
   - D) To manage cluster networking

6. **Which auto-scaling strategy is most effective for ML model serving workloads?**
   - A) Only manual scaling
   - B) Horizontal Pod Autoscaler (HPA) based on CPU, memory, and custom metrics like requests per second
   - C) Vertical scaling only
   - D) No scaling is needed for ML workloads

7. **What is the recommended approach for handling distributed ML training in Kubernetes?**
   - A) Use single-node training only
   - B) Use Jobs with parallelism, proper inter-pod communication, and shared storage
   - C) Avoid distributed training in Kubernetes
   - D) Use only StatefulSets for all training

8. **How should sensitive data like model credentials be managed in Kubernetes ML workloads?**
   - A) Store them in container images
   - B) Use Kubernetes Secrets with proper RBAC and encryption at rest
   - C) Include them in YAML files
   - D) Use environment variables in plain text

9. **What is the benefit of using Argo Workflows for ML pipelines?**
   - A) It only works with specific ML frameworks
   - B) It provides workflow orchestration with DAG support, conditional execution, and artifact management
   - C) It replaces the need for containers
   - D) It only handles data storage

10. **Which monitoring approach is most comprehensive for ML workloads on Kubernetes?**
    - A) Only check pod status
    - B) Use Prometheus for metrics collection, Grafana for visualization, and custom ML metrics like model accuracy and inference latency
    - C) Monitor only CPU usage
    - D) No monitoring is needed for ML workloads

---

## Answer Key

**1. B** - Kubernetes Jobs ensure completion of batch workloads with automatic retry and failure handling. They are designed for workloads that run to completion, making them ideal for ML training tasks that need to finish successfully and handle failures gracefully.

**2. B** - Deployments with Services and Horizontal Pod Autoscaler provide the best solution for scalable ML model serving. Deployments ensure high availability and rolling updates, Services provide stable networking, and HPA enables automatic scaling based on demand.

**3. B** - Kubeflow provides a complete ML platform with pipelines, training operators, and serving capabilities. It offers end-to-end ML workflow management, including experiment tracking, hyperparameter tuning, and multi-framework support, all integrated with Kubernetes.

**4. B** - GPU resources should be managed using resource requests/limits with nvidia.com/gpu and appropriate node selectors and tolerations. This ensures proper GPU allocation, prevents resource conflicts, and enables scheduling on GPU-enabled nodes.

**5. B** - Seldon Core provides advanced model serving capabilities with A/B testing, canary deployments, and explainability. It extends Kubernetes with ML-specific features like multi-armed bandits, model explanations, and advanced deployment strategies.

**6. B** - Horizontal Pod Autoscaler (HPA) based on CPU, memory, and custom metrics like requests per second is most effective for ML serving workloads. This approach scales based on actual demand and can incorporate ML-specific metrics for optimal performance.

**7. B** - Distributed ML training should use Jobs with parallelism, proper inter-pod communication, and shared storage. This approach enables coordination between training workers, handles failures gracefully, and provides access to shared datasets and model checkpoints.

**8. B** - Sensitive data should be managed using Kubernetes Secrets with proper RBAC and encryption at rest. This provides secure storage, access control, and integration with pod security contexts while keeping credentials separate from application code.

**9. B** - Argo Workflows provides workflow orchestration with DAG support, conditional execution, and artifact management. It enables complex ML pipelines with dependencies, parallel execution, and proper handling of intermediate results and model artifacts.

**10. B** - Comprehensive monitoring should use Prometheus for metrics collection, Grafana for visualization, and custom ML metrics like model accuracy and inference latency. This approach provides both infrastructure and application-level insights specific to ML workloads.
