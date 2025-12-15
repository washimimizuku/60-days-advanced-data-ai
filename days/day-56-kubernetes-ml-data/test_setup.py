#!/usr/bin/env python3
"""
Day 56: Kubernetes for ML & Data Workloads - Setup Verification Script
Verifies that all required dependencies and Kubernetes resources are properly configured
"""

import sys
import os
import subprocess
import json
from typing import Dict, List, Tuple
import importlib.util

def check_python_version() -> Tuple[bool, str]:
    """Check if Python version is 3.9 or higher"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        return True, f"✅ Python {version.major}.{version.minor}.{version.micro}"
    else:
        return False, f"❌ Python {version.major}.{version.minor}.{version.micro} (requires 3.9+)"

def check_package_installation(packages: List[str]) -> Dict[str, Tuple[bool, str]]:
    """Check if required Python packages are installed"""
    results = {}
    
    for package in packages:
        try:
            spec = importlib.util.find_spec(package)
            if spec is not None:
                module = importlib.import_module(package)
                version = getattr(module, '__version__', 'unknown')
                results[package] = (True, f"✅ {package} {version}")
            else:
                results[package] = (False, f"❌ {package} not found")
        except ImportError as e:
            results[package] = (False, f"❌ {package} import error: {e}")
        except Exception as e:
            results[package] = (False, f"❌ {package} error: {e}")
    
    return results

def check_kubectl() -> Tuple[bool, str]:
    """Check if kubectl is installed and configured"""
    try:
        # Check kubectl installation
        result = subprocess.run(['kubectl', 'version', '--client'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "❌ kubectl not installed"
        
        version = result.stdout.strip()
        
        # Check cluster connectivity
        result = subprocess.run(['kubectl', 'cluster-info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, f"❌ kubectl not connected to cluster: {result.stderr.strip()}"
        
        cluster_info = result.stdout.strip()
        
        return True, f"✅ kubectl installed and connected\n   {cluster_info.split(chr(10))[0]}"
        
    except subprocess.TimeoutExpired:
        return False, "❌ kubectl command timeout"
    except FileNotFoundError:
        return False, "❌ kubectl not found in PATH"
    except Exception as e:
        return False, f"❌ kubectl error: {e}"

def check_docker() -> Tuple[bool, str]:
    """Check if Docker is installed and running"""
    try:
        # Check Docker installation
        result = subprocess.run(['docker', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "❌ Docker not installed"
        
        version = result.stdout.strip()
        
        # Check if Docker daemon is running
        result = subprocess.run(['docker', 'info'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, f"❌ Docker daemon not running: {result.stderr.strip()}"
        
        return True, f"✅ {version} (daemon running)"
        
    except subprocess.TimeoutExpired:
        return False, "❌ Docker command timeout"
    except FileNotFoundError:
        return False, "❌ Docker not found in PATH"
    except Exception as e:
        return False, f"❌ Docker error: {e}"

def check_kubernetes_namespaces() -> Dict[str, Tuple[bool, str]]:
    """Check if required Kubernetes namespaces exist"""
    results = {}
    required_namespaces = [
        'ml-training',
        'ml-serving', 
        'data-processing',
        'monitoring'
    ]
    
    for namespace in required_namespaces:
        try:
            result = subprocess.run([
                'kubectl', 'get', 'namespace', namespace
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                results[f'Namespace: {namespace}'] = (True, f"✅ {namespace} exists")
            else:
                results[f'Namespace: {namespace}'] = (False, f"❌ {namespace} not found")
                
        except Exception as e:
            results[f'Namespace: {namespace}'] = (False, f"❌ Error: {e}")
    
    return results

def check_kubernetes_operators() -> Dict[str, Tuple[bool, str]]:
    """Check if ML operators are installed"""
    results = {}
    
    # Check for common ML operators
    operators = [
        ('Kubeflow Training Operator', 'kubeflow', 'training-operator'),
        ('Seldon Core', 'seldon-system', 'seldon-controller-manager'),
        ('Argo Workflows', 'argo', 'workflow-controller'),
        ('Spark Operator', 'spark-operator', 'spark-operator')
    ]
    
    for operator_name, namespace, deployment in operators:
        try:
            result = subprocess.run([
                'kubectl', 'get', 'deployment', deployment, '-n', namespace
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                results[operator_name] = (True, f"✅ {operator_name} installed")
            else:
                results[operator_name] = (False, f"❌ {operator_name} not found")
                
        except Exception as e:
            results[operator_name] = (False, f"❌ {operator_name} error: {e}")
    
    return results

def check_storage_classes() -> Tuple[bool, str]:
    """Check if storage classes are available"""
    try:
        result = subprocess.run(['kubectl', 'get', 'storageclass'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            storage_classes = len([line for line in result.stdout.split('\n') if line.strip() and not line.startswith('NAME')])
            return True, f"✅ {storage_classes} storage classes available"
        else:
            return False, f"❌ Storage classes not accessible"
    except Exception as e:
        return False, f"❌ Storage class error: {e}"

def check_environment_file() -> Tuple[bool, str]:
    """Check if .env file exists and has required variables"""
    env_file = '.env'
    
    if not os.path.exists(env_file):
        return False, "❌ .env file not found (copy from .env.example)"
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        required_vars = [
            'KUBECONFIG',
            'KUBERNETES_NAMESPACE',
            'CLUSTER_NAME',
            'DEFAULT_CPU_REQUEST',
            'DEFAULT_MEMORY_REQUEST'
        ]
        
        missing_vars = []
        for var in required_vars:
            if var not in content or f'{var}=' not in content:
                missing_vars.append(var)
        
        if missing_vars:
            return False, f"❌ Missing variables: {', '.join(missing_vars)}"
        
        return True, f"✅ .env file configured with {len(required_vars)} variables"
        
    except Exception as e:
        return False, f"❌ Error reading .env file: {e}"

def run_basic_tests() -> Dict[str, Tuple[bool, str]]:
    """Run basic functionality tests"""
    results = {}
    
    # Test importing main modules
    try:
        from exercise import MLTrainingManager, MockKubernetesClient
        results['Exercise imports'] = (True, "✅ All exercise classes importable")
    except Exception as e:
        results['Exercise imports'] = (False, f"❌ Import error: {e}")
    
    # Test creating mock client
    try:
        from exercise import MockKubernetesClient
        client = MockKubernetesClient('test-cluster')
        test_yaml = """
apiVersion: v1
kind: Pod
metadata:
  name: test-pod
spec:
  containers:
  - name: test
    image: nginx
"""
        resource_id = client.create_resource_from_yaml(test_yaml, 'default')
        results['Mock Kubernetes client'] = (True, "✅ Mock Kubernetes client working")
    except Exception as e:
        results['Mock Kubernetes client'] = (False, f"❌ Mock client error: {e}")
    
    return results

def check_gpu_support() -> Tuple[bool, str]:
    """Check if GPU support is available (optional)"""
    try:
        result = subprocess.run([
            'kubectl', 'get', 'nodes', '-o', 'jsonpath={.items[*].status.capacity.nvidia\.com/gpu}'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and result.stdout.strip():
            gpu_counts = [count for count in result.stdout.strip().split() if count and count != '0']
            if gpu_counts:
                total_gpus = sum(int(count) for count in gpu_counts)
                return True, f"✅ GPU support available ({total_gpus} GPUs total)"
            else:
                return False, "⚠️ No GPU nodes found (optional for CPU-only exercises)"
        else:
            return False, "⚠️ GPU support not detected (optional for CPU-only exercises)"
    except Exception as e:
        return False, f"⚠️ GPU check error: {e} (optional)"

def main():
    """Run all setup verification checks"""
    print("🔍 Day 56: Kubernetes for ML & Data Workloads - Setup Verification")
    print("=" * 70)
    
    all_passed = True
    
    # Check Python version
    print("\n📋 Python Environment")
    print("-" * 20)
    passed, message = check_python_version()
    print(message)
    if not passed:
        all_passed = False
    
    # Check required packages
    print("\n📦 Python Packages")
    print("-" * 18)
    required_packages = [
        'kubernetes', 'yaml', 'pandas', 'numpy', 'scikit-learn',
        'torch', 'tensorflow', 'pytest', 'docker', 'requests'
    ]
    
    package_results = check_package_installation(required_packages)
    for package, (passed, message) in package_results.items():
        print(message)
        if not passed:
            all_passed = False
    
    # Check kubectl
    print("\n⚙️ Kubernetes CLI")
    print("-" * 16)
    passed, message = check_kubectl()
    print(message)
    if not passed:
        all_passed = False
    
    # Check Docker
    print("\n🐳 Docker")
    print("-" * 8)
    passed, message = check_docker()
    print(message)
    if not passed:
        all_passed = False
    
    # Check environment file
    print("\n⚙️ Environment Configuration")
    print("-" * 28)
    passed, message = check_environment_file()
    print(message)
    if not passed:
        all_passed = False
    
    # Check Kubernetes namespaces
    print("\n🏷️ Kubernetes Namespaces")
    print("-" * 24)
    namespace_results = check_kubernetes_namespaces()
    for namespace, (passed, message) in namespace_results.items():
        print(message)
        if not passed:
            all_passed = False
    
    # Check storage classes
    print("\n💾 Storage Classes")
    print("-" * 17)
    passed, message = check_storage_classes()
    print(message)
    if not passed:
        all_passed = False
    
    # Check ML operators (optional)
    print("\n🤖 ML Operators (Optional)")
    print("-" * 26)
    operator_results = check_kubernetes_operators()
    for operator, (passed, message) in operator_results.items():
        print(message)
        # Don't fail overall setup for missing operators
    
    # Check GPU support (optional)
    print("\n🎮 GPU Support (Optional)")
    print("-" * 24)
    passed, message = check_gpu_support()
    print(message)
    
    # Run basic tests
    print("\n🧪 Basic Tests")
    print("-" * 13)
    test_results = run_basic_tests()
    for test, (passed, message) in test_results.items():
        print(message)
        if not passed:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 All essential checks passed! You're ready to start the exercises.")
        print("\nNext steps:")
        print("1. Run: python exercise.py")
        print("2. Complete all 7 exercises")
        print("3. Review solution.py for complete implementations")
        print("4. Take the quiz in quiz.md")
        return 0
    else:
        print("❌ Some essential checks failed. Please fix the issues above.")
        print("\nCommon fixes:")
        print("• Install missing packages: pip install -r requirements.txt")
        print("• Start Kubernetes cluster (Docker Desktop/minikube)")
        print("• Create required namespaces: kubectl create namespace ml-training")
        print("• Copy .env.example to .env and configure")
        print("• Install kubectl and configure cluster access")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)