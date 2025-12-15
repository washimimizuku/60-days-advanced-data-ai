#!/usr/bin/env python3
"""
Day 55: AWS Deep Dive - Setup Verification Script
Verifies that all required dependencies and AWS resources are properly configured
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
                # Try to import to check if it's working
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

def check_aws_cli() -> Tuple[bool, str]:
    """Check if AWS CLI is installed and configured"""
    try:
        # Check AWS CLI installation
        result = subprocess.run(['aws', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, "❌ AWS CLI not installed"
        
        version = result.stdout.strip()
        
        # Check AWS CLI configuration
        result = subprocess.run(['aws', 'sts', 'get-caller-identity'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            return False, f"❌ AWS CLI not configured: {result.stderr.strip()}"
        
        identity = json.loads(result.stdout)
        account_id = identity.get('Account', 'unknown')
        user_arn = identity.get('Arn', 'unknown')
        
        return True, f"✅ AWS CLI {version}\n   Account: {account_id}\n   User: {user_arn}"
        
    except subprocess.TimeoutExpired:
        return False, "❌ AWS CLI command timeout"
    except FileNotFoundError:
        return False, "❌ AWS CLI not found in PATH"
    except json.JSONDecodeError:
        return False, "❌ AWS CLI returned invalid JSON"
    except Exception as e:
        return False, f"❌ AWS CLI error: {e}"

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

def check_aws_resources() -> Dict[str, Tuple[bool, str]]:
    """Check if required AWS resources exist"""
    results = {}
    
    # Check IAM roles
    iam_roles = [
        'SageMakerExecutionRole',
        'ECSTaskExecutionRole', 
        'LambdaExecutionRole'
    ]
    
    for role_name in iam_roles:
        try:
            result = subprocess.run([
                'aws', 'iam', 'get-role', 
                '--role-name', role_name
            ], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                role_data = json.loads(result.stdout)
                role_arn = role_data['Role']['Arn']
                results[f'IAM Role: {role_name}'] = (True, f"✅ {role_arn}")
            else:
                results[f'IAM Role: {role_name}'] = (False, f"❌ Role not found")
                
        except Exception as e:
            results[f'IAM Role: {role_name}'] = (False, f"❌ Error: {e}")
    
    # Check S3 buckets (optional - they might not exist yet)
    try:
        result = subprocess.run(['aws', 's3', 'ls'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            bucket_count = len([line for line in result.stdout.split('\n') if line.strip()])
            results['S3 Access'] = (True, f"✅ S3 accessible ({bucket_count} buckets)")
        else:
            results['S3 Access'] = (False, f"❌ S3 access denied")
    except Exception as e:
        results['S3 Access'] = (False, f"❌ S3 error: {e}")
    
    return results

def check_environment_file() -> Tuple[bool, str]:
    """Check if .env file exists and has required variables"""
    env_file = '.env'
    
    if not os.path.exists(env_file):
        return False, "❌ .env file not found (copy from .env.example)"
    
    try:
        with open(env_file, 'r') as f:
            content = f.read()
        
        required_vars = [
            'AWS_REGION',
            'AWS_ACCOUNT_ID',
            'SAGEMAKER_ROLE_ARN',
            'ECS_TASK_ROLE_ARN',
            'LAMBDA_ROLE_ARN'
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
        from exercise import SageMakerPipeline, ECSDeployment, LambdaPipeline
        results['Exercise imports'] = (True, "✅ All exercise classes importable")
    except Exception as e:
        results['Exercise imports'] = (False, f"❌ Import error: {e}")
    
    # Test creating mock clients
    try:
        from exercise import MockAWSClient
        client = MockAWSClient('test', 'us-west-2')
        response = client.test_method()
        results['Mock AWS client'] = (True, "✅ Mock AWS client working")
    except Exception as e:
        results['Mock AWS client'] = (False, f"❌ Mock client error: {e}")
    
    return results

def main():
    """Run all setup verification checks"""
    print("🔍 Day 55: AWS Deep Dive - Setup Verification")
    print("=" * 60)
    
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
        'boto3', 'botocore', 'pandas', 'numpy', 'scikit-learn',
        'fastapi', 'uvicorn', 'pydantic', 'pytest', 'docker',
        'yaml', 'requests', 'click'
    ]
    
    package_results = check_package_installation(required_packages)
    for package, (passed, message) in package_results.items():
        print(message)
        if not passed:
            all_passed = False
    
    # Check AWS CLI
    print("\n☁️ AWS Configuration")
    print("-" * 19)
    passed, message = check_aws_cli()
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
    
    # Check AWS resources
    print("\n🏗️ AWS Resources")
    print("-" * 15)
    aws_results = check_aws_resources()
    for resource, (passed, message) in aws_results.items():
        print(message)
        if not passed:
            all_passed = False
    
    # Run basic tests
    print("\n🧪 Basic Tests")
    print("-" * 13)
    test_results = run_basic_tests()
    for test, (passed, message) in test_results.items():
        print(message)
        if not passed:
            all_passed = False
    
    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All checks passed! You're ready to start the exercises.")
        print("\nNext steps:")
        print("1. Run: python exercise.py")
        print("2. Complete all 7 exercises")
        print("3. Review solution.py for complete implementations")
        print("4. Take the quiz in quiz.md")
        return 0
    else:
        print("❌ Some checks failed. Please fix the issues above before proceeding.")
        print("\nCommon fixes:")
        print("• Install missing packages: pip install -r requirements.txt")
        print("• Configure AWS CLI: aws configure")
        print("• Start Docker Desktop")
        print("• Create IAM roles using SETUP.md instructions")
        print("• Copy .env.example to .env and configure")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)