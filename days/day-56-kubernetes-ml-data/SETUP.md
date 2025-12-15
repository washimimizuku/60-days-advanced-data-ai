# Day 56: Kubernetes for ML & Data Workloads - Setup Guide

## Overview
This guide helps you set up the environment for Kubernetes ML workloads exercises, including cluster setup, GPU configuration, and ML operator installation.

## Prerequisites

### Required Software
- Python 3.9+ with pip
- Docker Desktop with Kubernetes enabled
- kubectl CLI tool
- Helm 3.x
- Git
- Code editor (VS Code recommended)

### Kubernetes Cluster Requirements
- Kubernetes 1.25+ cluster
- At least 8GB RAM and 4 CPU cores available
- GPU nodes (optional but recommended for GPU exercises)
- Storage class for persistent volumes

## Installation Steps

### 1. Install Dependencies

```bash
# Navigate to day 56 directory
cd days/day-56-kubernetes-ml-data

# Install Python dependencies
pip install -r requirements.txt

# Verify installations
kubectl version --client
helm version
docker --version
```

### 2. Kubernetes Cluster Setup

#### Option A: Docker Desktop (Recommended for Development)
```bash
# Enable Kubernetes in Docker Desktop
# Go to Docker Desktop Settings > Kubernetes > Enable Kubernetes

# Verify cluster is running
kubectl cluster-info
kubectl get nodes
```

#### Option B: Minikube
```bash
# Install minikube
# macOS: brew install minikube
# Linux: curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64

# Start minikube with sufficient resources
minikube start --memory=8192 --cpus=4 --disk-size=50g

# Enable addons
minikube addons enable ingress
minikube addons enable metrics-server
```

#### Option C: Kind (Kubernetes in Docker)
```bash
# Install kind
# macOS: brew install kind
# Linux: curl -Lo ./kind https://kind.sigs.k8s.io/dl/v0.20.0/kind-linux-amd64

# Create cluster with config
cat > kind-config.yaml << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  kubeadmConfigPatches:
  - |
    kind: InitConfiguration
    nodeRegistration:
      kubeletExtraArgs:
        node-labels: "ingress-ready=true"
  extraPortMappings:
  - containerPort: 80
    hostPort: 80
    protocol: TCP
  - containerPort: 443
    hostPort: 443
    protocol: TCP
- role: worker
- role: worker
EOF

kind create cluster --config kind-config.yaml --name ml-cluster
```

### 3. Create Namespaces

```bash
# Create namespaces for ML workloads
kubectl create namespace ml-training
kubectl create namespace ml-serving
kubectl create namespace data-processing
kubectl create namespace monitoring

# Verify namespaces
kubectl get namespaces
```

### 4. Set Up Storage

#### Create Storage Classes and PVCs
```bash
# Create storage class (if not exists)
cat > storage-class.yaml << EOF
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: fast-ssd
provisioner: kubernetes.io/host-path
parameters:
  type: DirectoryOrCreate
allowVolumeExpansion: true
volumeBindingMode: WaitForFirstConsumer
EOF

kubectl apply -f storage-class.yaml

# Create persistent volume claims
cat > ml-pvcs.yaml << EOF
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: training-data-pvc
  namespace: ml-training
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-output-pvc
  namespace: ml-training
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 20Gi
  storageClassName: fast-ssd
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-storage-pvc
  namespace: ml-serving
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: fast-ssd
EOF

kubectl apply -f ml-pvcs.yaml
```

### 5. GPU Setup (Optional)

#### Install NVIDIA GPU Operator
```bash
# Add NVIDIA Helm repository
helm repo add nvidia https://nvidia.github.io/gpu-operator
helm repo update

# Install GPU operator
helm install --wait --generate-name \
  -n gpu-operator --create-namespace \
  nvidia/gpu-operator

# Verify GPU nodes
kubectl get nodes -l nvidia.com/gpu.present=true
```

#### Create GPU Resource Quota
```bash
cat > gpu-quota.yaml << EOF
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: ml-training
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
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
    defaultRequest:
      nvidia.com/gpu: "1"
    max:
      nvidia.com/gpu: "4"
EOF

kubectl apply -f gpu-quota.yaml
```

### 6. Install ML Operators

#### Install Kubeflow Training Operator
```bash
# Install training operator
kubectl apply -k "github.com/kubeflow/training-operator/manifests/overlays/standalone?ref=v1.7.0"

# Verify installation
kubectl get pods -n kubeflow
```

#### Install Seldon Core
```bash
# Add Seldon Helm repository
helm repo add seldonio https://storage.googleapis.com/seldon-charts
helm repo update

# Install Seldon Core
helm install seldon-core seldonio/seldon-core-operator \
  --namespace seldon-system --create-namespace \
  --set usageMetrics.enabled=true

# Verify installation
kubectl get pods -n seldon-system
```

#### Install Argo Workflows
```bash
# Install Argo Workflows
kubectl create namespace argo
kubectl apply -n argo -f https://github.com/argoproj/argo-workflows/releases/download/v3.5.2/install.yaml

# Create service account for workflows
kubectl create rolebinding default-admin --clusterrole=admin --serviceaccount=argo:default -n argo

# Verify installation
kubectl get pods -n argo
```

### 7. Install Monitoring Stack

#### Install Prometheus and Grafana
```bash
# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install kube-prometheus-stack
helm install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set grafana.adminPassword=admin123

# Verify installation
kubectl get pods -n monitoring
```

#### Access Grafana Dashboard
```bash
# Port forward to access Grafana
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80

# Access at http://localhost:3000
# Username: admin, Password: admin123
```

### 8. Install Spark Operator (Optional)

```bash
# Add Spark Operator Helm repository
helm repo add spark-operator https://googlecloudplatform.github.io/spark-on-k8s-operator
helm repo update

# Install Spark Operator
helm install spark-operator spark-operator/spark-operator \
  --namespace spark-operator --create-namespace \
  --set webhook.enable=true

# Verify installation
kubectl get pods -n spark-operator
```

### 9. Environment Configuration

Create a `.env` file with your cluster configuration:

```bash
# Create environment file
cat > .env << EOF
# Kubernetes Configuration
KUBECONFIG=${HOME}/.kube/config
KUBERNETES_NAMESPACE=ml-training
CLUSTER_NAME=ml-cluster

# Container Registry
CONTAINER_REGISTRY=docker.io
REGISTRY_USERNAME=your-username
REGISTRY_PASSWORD=your-password

# ML Configuration
DEFAULT_CPU_REQUEST=2
DEFAULT_MEMORY_REQUEST=4Gi
DEFAULT_GPU_REQUEST=1
DEFAULT_STORAGE_CLASS=fast-ssd

# Monitoring Configuration
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
GRAFANA_USERNAME=admin
GRAFANA_PASSWORD=admin123

# Workflow Configuration
ARGO_SERVER_URL=http://localhost:2746
KUBEFLOW_NAMESPACE=kubeflow

# Data Configuration
DATA_BUCKET=ml-training-data
MODEL_REGISTRY_URL=http://localhost:5000
FEATURE_STORE_URL=http://localhost:6379

# Security Configuration
ENABLE_RBAC=true
ENABLE_NETWORK_POLICIES=true
ENABLE_POD_SECURITY_POLICIES=true
EOF
```

### 10. Test Setup

```bash
# Run the setup test
python test_setup.py

# Expected output:
# ✅ Kubernetes cluster accessible
# ✅ Required namespaces exist
# ✅ Storage classes configured
# ✅ ML operators installed
# ✅ Monitoring stack running
# ✅ All dependencies available
```

## Sample Workload Deployment

### Deploy a Test Training Job

```bash
# Create a simple training job
cat > test-training-job.yaml << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: test-training-job
  namespace: ml-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: tensorflow/tensorflow:2.13.0
        command: ["python", "-c"]
        args: ["import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('Training completed successfully!')"]
        resources:
          requests:
            cpu: "1"
            memory: "2Gi"
          limits:
            cpu: "2"
            memory: "4Gi"
      restartPolicy: Never
  backoffLimit: 3
EOF

kubectl apply -f test-training-job.yaml

# Check job status
kubectl get jobs -n ml-training
kubectl logs -n ml-training job/test-training-job
```

### Deploy a Test Model Serving

```bash
# Create a simple model server
cat > test-model-server.yaml << EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test-model-server
  namespace: ml-serving
spec:
  replicas: 2
  selector:
    matchLabels:
      app: test-model-server
  template:
    metadata:
      labels:
        app: test-model-server
    spec:
      containers:
      - name: model-server
        image: tensorflow/serving:2.13.0
        ports:
        - containerPort: 8501
        env:
        - name: MODEL_NAME
          value: "test_model"
        resources:
          requests:
            cpu: "500m"
            memory: "1Gi"
          limits:
            cpu: "1"
            memory: "2Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: test-model-service
  namespace: ml-serving
spec:
  selector:
    app: test-model-server
  ports:
  - port: 8501
    targetPort: 8501
  type: ClusterIP
EOF

kubectl apply -f test-model-server.yaml

# Check deployment status
kubectl get deployments -n ml-serving
kubectl get services -n ml-serving
```

## Troubleshooting

### Common Issues

1. **Insufficient Resources**
   ```bash
   # Check node resources
   kubectl top nodes
   kubectl describe nodes
   
   # Check resource quotas
   kubectl get resourcequota -A
   ```

2. **Storage Issues**
   ```bash
   # Check storage classes
   kubectl get storageclass
   
   # Check PVC status
   kubectl get pvc -A
   kubectl describe pvc <pvc-name> -n <namespace>
   ```

3. **GPU Not Available**
   ```bash
   # Check GPU operator status
   kubectl get pods -n gpu-operator
   
   # Check node labels
   kubectl get nodes -o yaml | grep nvidia
   ```

4. **Network Issues**
   ```bash
   # Check network policies
   kubectl get networkpolicies -A
   
   # Test pod connectivity
   kubectl run test-pod --image=busybox --rm -it -- /bin/sh
   ```

### Verification Commands

```bash
# Check cluster status
kubectl cluster-info
kubectl get nodes -o wide

# Check all ML namespaces
kubectl get all -n ml-training
kubectl get all -n ml-serving
kubectl get all -n data-processing

# Check operators
kubectl get pods -n kubeflow
kubectl get pods -n seldon-system
kubectl get pods -n argo

# Check monitoring
kubectl get pods -n monitoring
kubectl get servicemonitors -A
```

## Performance Optimization

### Resource Limits and Requests
```bash
# Set default resource limits
cat > default-limits.yaml << EOF
apiVersion: v1
kind: LimitRange
metadata:
  name: default-limits
  namespace: ml-training
spec:
  limits:
  - default:
      cpu: "2"
      memory: "4Gi"
    defaultRequest:
      cpu: "1"
      memory: "2Gi"
    type: Container
EOF

kubectl apply -f default-limits.yaml
```

### Node Affinity for ML Workloads
```bash
# Label nodes for ML workloads
kubectl label nodes <node-name> workload-type=ml-training
kubectl label nodes <node-name> accelerator=nvidia-tesla-v100
```

## Security Configuration

### RBAC Setup
```bash
# Create service account for ML workloads
kubectl create serviceaccount ml-training-sa -n ml-training

# Create role and role binding
cat > ml-rbac.yaml << EOF
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  namespace: ml-training
  name: ml-training-role
rules:
- apiGroups: [""]
  resources: ["pods", "services", "configmaps", "secrets"]
  verbs: ["get", "list", "create", "update", "patch", "delete"]
- apiGroups: ["batch"]
  resources: ["jobs"]
  verbs: ["get", "list", "create", "update", "patch", "delete"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: ml-training-binding
  namespace: ml-training
subjects:
- kind: ServiceAccount
  name: ml-training-sa
  namespace: ml-training
roleRef:
  kind: Role
  name: ml-training-role
  apiGroup: rbac.authorization.k8s.io
EOF

kubectl apply -f ml-rbac.yaml
```

## Next Steps

1. Complete the setup verification
2. Review the exercise.py file
3. Start with Exercise 1: ML Training Jobs
4. Progress through all 7 exercises
5. Review the solution.py for complete implementations
6. Take the quiz to test your understanding

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review Kubernetes and operator documentation
3. Ensure you have sufficient cluster resources
4. Verify all prerequisites are installed
5. Check cluster and pod logs for error messages

Remember to clean up resources after completing the exercises to avoid resource exhaustion!