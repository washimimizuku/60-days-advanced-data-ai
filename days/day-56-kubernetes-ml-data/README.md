# Day 56: Kubernetes for ML & Data Workloads - Orchestration, Scaling & Management

## Learning Objectives
By the end of this session, you will be able to:
- Deploy and manage ML workloads on Kubernetes with proper resource allocation
- Implement Kubernetes operators for ML frameworks (Kubeflow, MLflow, Seldon)
- Design scalable data processing pipelines using Kubernetes jobs and workflows
- Configure GPU scheduling and resource management for ML training and inference
- Build production-ready ML serving infrastructure with auto-scaling and monitoring

## Theory (15 minutes)

### Kubernetes for ML & Data Workloads

Kubernetes has become the de facto standard for orchestrating containerized applications, including machine learning and data processing workloads. Its declarative approach, auto-scaling capabilities, and rich ecosystem make it ideal for production ML systems.

### Core Kubernetes Concepts for ML

#### 1. Pods and Containers for ML Workloads

**ML Training Pod**
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: ml-training-pod
  labels:
    app: ml-training
    workload-type: training
spec:
  containers:
  - name: trainer
    image: tensorflow/tensorflow:2.13.0-gpu
    resources:
      requests:
        memory: "8Gi"
        cpu: "4"
        nvidia.com/gpu: 1
      limits:
        memory: "16Gi"
        cpu: "8"
        nvidia.com/gpu: 1
    env:
    - name: CUDA_VISIBLE_DEVICES
      value: "0"
    volumeMounts:
    - name: training-data
      mountPath: /data
    - name: model-output
      mountPath: /models
  volumes:
  - name: training-data
    persistentVolumeClaim:
      claimName: training-data-pvc
  - name: model-output
    persistentVolumeClaim:
      claimName: model-output-pvc
  nodeSelector:
    accelerator: nvidia-tesla-v100
  tolerations:
  - key: nvidia.com/gpu
    operator: Exists
    effect: NoSchedule
```

#### 2. Jobs for Batch ML Workloads

**Distributed Training Job**
```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: distributed-training-job
spec:
  parallelism: 4
  completions: 4
  template:
    metadata:
      labels:
        app: distributed-training
    spec:
      containers:
      - name: worker
        image: pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime
        command: ["python", "train_distributed.py"]
        args: 
        - "--world-size=4"
        - "--rank=$(JOB_COMPLETION_INDEX)"
        - "--master-addr=training-master-service"
        - "--master-port=29500"
        resources:
          requests:
            nvidia.com/gpu: 1
            memory: "16Gi"
            cpu: "8"
        env:
        - name: JOB_COMPLETION_INDEX
          valueFrom:
            fieldRef:
              fieldPath: metadata.annotations['batch.kubernetes.io/job-completion-index']
      restartPolicy: Never
  backoffLimit: 3
```

#### 3. Services for ML Model Serving

**Model Serving Service**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-serving-service
  labels:
    app: model-server
spec:
  selector:
    app: model-server
  ports:
  - port: 8080
    targetPort: 8080
    name: http
  - port: 8081
    targetPort: 8081
    name: grpc
  type: ClusterIP
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-server-deployment
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-server
  template:
    metadata:
      labels:
        app: model-server
    spec:
      containers:
      - name: model-server
        image: tensorflow/serving:2.13.0-gpu
        ports:
        - containerPort: 8080
        - containerPort: 8081
        env:
        - name: MODEL_NAME
          value: "text_classifier"
        - name: MODEL_BASE_PATH
          value: "/models"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        volumeMounts:
        - name: model-storage
          mountPath: /models
        readinessProbe:
          httpGet:
            path: /v1/models/text_classifier
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /v1/models/text_classifier
            port: 8080
          initialDelaySeconds: 60
          periodSeconds: 30
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-storage-pvc
```

### Kubernetes Operators for ML

#### 1. Kubeflow - Complete ML Platform

**Kubeflow Pipeline**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: ml-pipeline-workflow
spec:
  entrypoint: ml-pipeline
  templates:
  - name: ml-pipeline
    dag:
      tasks:
      - name: data-preprocessing
        template: preprocess-data
      - name: model-training
        template: train-model
        dependencies: [data-preprocessing]
      - name: model-evaluation
        template: evaluate-model
        dependencies: [model-training]
      - name: model-deployment
        template: deploy-model
        dependencies: [model-evaluation]
  
  - name: preprocess-data
    container:
      image: python:3.9
      command: [python]
      args: ["preprocess.py"]
      resources:
        requests:
          memory: "4Gi"
          cpu: "2"
  
  - name: train-model
    container:
      image: tensorflow/tensorflow:2.13.0-gpu
      command: [python]
      args: ["train.py"]
      resources:
        requests:
          memory: "16Gi"
          cpu: "8"
          nvidia.com/gpu: 2
  
  - name: evaluate-model
    container:
      image: python:3.9
      command: [python]
      args: ["evaluate.py"]
      resources:
        requests:
          memory: "8Gi"
          cpu: "4"
  
  - name: deploy-model
    container:
      image: seldon/seldon-core-operator:1.15.0
      command: [kubectl]
      args: ["apply", "-f", "model-deployment.yaml"]
```

#### 2. Seldon Core for Model Serving

**Seldon Deployment**
```yaml
apiVersion: machinelearning.seldon.io/v1
kind: SeldonDeployment
metadata:
  name: text-classifier-model
spec:
  name: text-classifier
  predictors:
  - name: default
    replicas: 3
    graph:
      name: classifier
      implementation: TENSORFLOW_SERVER
      modelUri: gs://ml-models/text-classifier/v1
      envSecretRefName: model-credentials
    componentSpecs:
    - spec:
        containers:
        - name: classifier
          resources:
            requests:
              memory: "4Gi"
              cpu: "2"
              nvidia.com/gpu: 1
            limits:
              memory: "8Gi"
              cpu: "4"
              nvidia.com/gpu: 1
          env:
          - name: CUDA_VISIBLE_DEVICES
            value: "0"
    traffic: 100
    explainer:
      type: AnchorTabular
      modelUri: gs://ml-models/text-classifier-explainer/v1
```

#### 3. Argo Workflows for ML Pipelines

**ML Training Workflow**
```yaml
apiVersion: argoproj.io/v1alpha1
kind: Workflow
metadata:
  name: ml-training-pipeline
spec:
  entrypoint: training-pipeline
  arguments:
    parameters:
    - name: dataset-path
      value: "s3://ml-datasets/training-data"
    - name: model-version
      value: "v1.0.0"
  
  templates:
  - name: training-pipeline
    steps:
    - - name: validate-data
        template: data-validation
        arguments:
          parameters:
          - name: data-path
            value: "{{workflow.parameters.dataset-path}}"
    
    - - name: train-model
        template: model-training
        arguments:
          parameters:
          - name: data-path
            value: "{{workflow.parameters.dataset-path}}"
          - name: version
            value: "{{workflow.parameters.model-version}}"
    
    - - name: evaluate-model
        template: model-evaluation
        arguments:
          parameters:
          - name: model-path
            value: "{{steps.train-model.outputs.parameters.model-path}}"
    
    - - name: deploy-model
        template: model-deployment
        arguments:
          parameters:
          - name: model-path
            value: "{{steps.train-model.outputs.parameters.model-path}}"
        when: "{{steps.evaluate-model.outputs.parameters.accuracy}} > 0.85"
  
  - name: data-validation
    inputs:
      parameters:
      - name: data-path
    container:
      image: python:3.9
      command: [python]
      args: ["validate_data.py", "--data-path", "{{inputs.parameters.data-path}}"]
  
  - name: model-training
    inputs:
      parameters:
      - name: data-path
      - name: version
    outputs:
      parameters:
      - name: model-path
        valueFrom:
          path: /tmp/model-path.txt
    container:
      image: tensorflow/tensorflow:2.13.0-gpu
      command: [python]
      args: 
      - "train.py"
      - "--data-path={{inputs.parameters.data-path}}"
      - "--version={{inputs.parameters.version}}"
      resources:
        requests:
          nvidia.com/gpu: 2
          memory: "32Gi"
          cpu: "16"
```

### GPU Management and Scheduling

#### 1. GPU Node Configuration

**GPU Node Pool**
```yaml
apiVersion: v1
kind: Node
metadata:
  name: gpu-node-1
  labels:
    accelerator: nvidia-tesla-v100
    gpu-count: "4"
    node-type: gpu-worker
spec:
  capacity:
    nvidia.com/gpu: "4"
    memory: "128Gi"
    cpu: "32"
  allocatable:
    nvidia.com/gpu: "4"
    memory: "120Gi"
    cpu: "30"
```

#### 2. GPU Resource Quotas

**GPU Resource Quota**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-training
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
    requests.memory: "256Gi"
    requests.cpu: "64"
---
apiVersion: v1
kind: LimitRange
metadata:
  name: gpu-limits
  namespace: ml-training
spec:
  limits:
  - type: Container
    default:
      nvidia.com/gpu: "1"
      memory: "8Gi"
      cpu: "4"
    defaultRequest:
      nvidia.com/gpu: "1"
      memory: "4Gi"
      cpu: "2"
    max:
      nvidia.com/gpu: "4"
      memory: "64Gi"
      cpu: "32"
```

### Data Processing Pipelines

#### 1. Spark on Kubernetes

**Spark Application**
```yaml
apiVersion: sparkoperator.k8s.io/v1beta2
kind: SparkApplication
metadata:
  name: data-processing-job
  namespace: data-processing
spec:
  type: Python
  pythonVersion: "3"
  mode: cluster
  image: "apache/spark-py:v3.4.0"
  imagePullPolicy: Always
  mainApplicationFile: "s3a://spark-jobs/data_processing.py"
  sparkVersion: "3.4.0"
  restartPolicy:
    type: OnFailure
    onFailureRetries: 3
    onFailureRetryInterval: 10
    onSubmissionFailureRetries: 5
    onSubmissionFailureRetryInterval: 20
  driver:
    cores: 2
    coreLimit: "2000m"
    memory: "8g"
    labels:
      version: 3.4.0
    serviceAccount: spark-driver
    env:
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: aws-credentials
          key: access-key-id
  executor:
    cores: 4
    instances: 10
    memory: "16g"
    labels:
      version: 3.4.0
    env:
    - name: AWS_ACCESS_KEY_ID
      valueFrom:
        secretKeyRef:
          name: aws-credentials
          key: access-key-id
  deps:
    jars:
    - "s3a://spark-dependencies/hadoop-aws-3.3.4.jar"
    - "s3a://spark-dependencies/aws-java-sdk-bundle-1.12.262.jar"
```

#### 2. Dask on Kubernetes

**Dask Cluster**
```yaml
apiVersion: kubernetes.dask.org/v1
kind: DaskCluster
metadata:
  name: dask-cluster
spec:
  worker:
    replicas: 5
    spec:
      containers:
      - name: worker
        image: daskdev/dask:2023.5.0
        resources:
          requests:
            memory: "16Gi"
            cpu: "8"
          limits:
            memory: "32Gi"
            cpu: "16"
        env:
        - name: EXTRA_PIP_PACKAGES
          value: "scikit-learn pandas numpy"
  scheduler:
    spec:
      containers:
      - name: scheduler
        image: daskdev/dask:2023.5.0
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
    service:
      type: LoadBalancer
      ports:
      - name: tcp-comm
        port: 8786
        protocol: TCP
        targetPort: 8786
      - name: http-dashboard
        port: 8787
        protocol: TCP
        targetPort: 8787
```

### Auto-scaling and Resource Management

#### 1. Horizontal Pod Autoscaler (HPA)

**ML Model Serving HPA**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server-deployment
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "100"
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
      - type: Pods
        value: 2
        periodSeconds: 60
      selectPolicy: Max
```

#### 2. Vertical Pod Autoscaler (VPA)

**Training Job VPA**
```yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: training-job-vpa
spec:
  targetRef:
    apiVersion: batch/v1
    kind: Job
    name: ml-training-job
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: trainer
      maxAllowed:
        cpu: "32"
        memory: "128Gi"
        nvidia.com/gpu: "4"
      minAllowed:
        cpu: "2"
        memory: "8Gi"
        nvidia.com/gpu: "1"
      controlledResources: ["cpu", "memory", "nvidia.com/gpu"]
```

### Monitoring and Observability

#### 1. Prometheus Monitoring

**ML Metrics ServiceMonitor**
```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ml-model-metrics
  labels:
    app: model-server
spec:
  selector:
    matchLabels:
      app: model-server
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
---
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: ml-model-alerts
spec:
  groups:
  - name: ml-model-serving
    rules:
    - alert: HighModelLatency
      expr: histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m])) > 1.0
      for: 2m
      labels:
        severity: warning
      annotations:
        summary: "High model inference latency"
        description: "95th percentile latency is {{ $value }}s"
    
    - alert: ModelAccuracyDrop
      expr: model_accuracy < 0.85
      for: 5m
      labels:
        severity: critical
      annotations:
        summary: "Model accuracy dropped below threshold"
        description: "Model accuracy is {{ $value }}"
```

#### 2. Grafana Dashboards

**ML Dashboard ConfigMap**
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ml-dashboard
  labels:
    grafana_dashboard: "1"
data:
  ml-dashboard.json: |
    {
      "dashboard": {
        "title": "ML Model Serving Dashboard",
        "panels": [
          {
            "title": "Request Rate",
            "type": "graph",
            "targets": [
              {
                "expr": "rate(model_requests_total[5m])",
                "legendFormat": "{{model_name}}"
              }
            ]
          },
          {
            "title": "Inference Latency",
            "type": "graph",
            "targets": [
              {
                "expr": "histogram_quantile(0.95, rate(model_inference_duration_seconds_bucket[5m]))",
                "legendFormat": "95th percentile"
              }
            ]
          },
          {
            "title": "GPU Utilization",
            "type": "graph",
            "targets": [
              {
                "expr": "nvidia_gpu_utilization_gpu",
                "legendFormat": "GPU {{gpu}}"
              }
            ]
          }
        ]
      }
    }
```

### Security and Compliance

#### 1. Network Policies

**ML Workload Network Policy**
```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ml-workload-policy
  namespace: ml-production
spec:
  podSelector:
    matchLabels:
      workload-type: ml-serving
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: api-gateway
    - podSelector:
        matchLabels:
          app: load-balancer
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: data-storage
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
  - to: []
    ports:
    - protocol: TCP
      port: 443   # HTTPS
    - protocol: TCP
      port: 53    # DNS
    - protocol: UDP
      port: 53    # DNS
```

#### 2. Pod Security Standards

**Pod Security Policy**
```yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: ml-workload-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
  readOnlyRootFilesystem: true
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    fsGroup: 2000
```

### Why Kubernetes for ML & Data Matters

1. **Scalability**: Automatic scaling based on workload demands
2. **Resource Efficiency**: Optimal resource utilization across the cluster
3. **Portability**: Consistent deployment across different environments
4. **Reliability**: Self-healing and fault tolerance capabilities
5. **Ecosystem**: Rich ecosystem of ML-specific operators and tools
6. **Cost Optimization**: Efficient resource sharing and spot instance support

### Real-world Use Cases

- **Netflix**: Uses Kubernetes for ML model training and serving at scale
- **Spotify**: Deploys recommendation models using Kubernetes operators
- **Uber**: Runs ML pipelines and feature stores on Kubernetes
- **Airbnb**: Manages data processing workflows with Kubernetes jobs
- **Pinterest**: Serves ML models with auto-scaling on Kubernetes

## Exercise (40 minutes)
Complete the hands-on exercises in `exercise.py` to build production-ready Kubernetes deployments for ML workloads, including training jobs, model serving, and data processing pipelines.

## Resources
- [Kubernetes Documentation](https://kubernetes.io/docs/)
- [Kubeflow Documentation](https://www.kubeflow.org/docs/)
- [Seldon Core Documentation](https://docs.seldon.io/)
- [Argo Workflows](https://argoproj.github.io/argo-workflows/)
- [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/overview.html)
- [Kubernetes ML Operators](https://github.com/kubeflow/community/blob/master/proposals/kubeflow-operators.md)

## Next Steps
- Complete the Kubernetes ML exercises
- Review container orchestration patterns
- Take the quiz to test your understanding
- Move to Day 57: Terraform & Infrastructure as Code
