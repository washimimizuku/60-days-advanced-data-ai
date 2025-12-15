#!/usr/bin/env python3
"""
Test Airflow setup and validate installation
"""

import os
import sys
import subprocess
import requests
import time
from pathlib import Path

def test_python_packages():
    """Test required Python packages"""
    print("Testing Python packages...")
    
    required_packages = [
        'airflow',
        'pandas',
        'requests',
        'psycopg2'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package}")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    return True

def test_docker_services():
    """Test Docker services are running"""
    print("\nTesting Docker services...")
    
    try:
        result = subprocess.run(['docker-compose', 'ps'], 
                              capture_output=True, text=True, check=True)
        
        services = ['postgres', 'webserver', 'scheduler']
        running_services = []
        
        for service in services:
            if service in result.stdout and 'Up' in result.stdout:
                running_services.append(service)
                print(f"✅ {service}")
            else:
                print(f"❌ {service}")
        
        return len(running_services) == len(services)
        
    except subprocess.CalledProcessError:
        print("❌ Docker Compose not available or services not running")
        return False

def test_airflow_webserver():
    """Test Airflow webserver is accessible"""
    print("\nTesting Airflow webserver...")
    
    try:
        response = requests.get('http://localhost:8080/health', timeout=10)
        if response.status_code == 200:
            print("✅ Airflow webserver is accessible")
            return True
        else:
            print(f"❌ Webserver returned status {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to webserver: {e}")
        return False

def test_airflow_cli():
    """Test Airflow CLI commands"""
    print("\nTesting Airflow CLI...")
    
    try:
        # Test airflow version
        result = subprocess.run(['docker-compose', 'exec', '-T', 'airflow-webserver', 
                               'airflow', 'version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ Airflow version: {result.stdout.strip()}")
        
        # Test DAG list
        result = subprocess.run(['docker-compose', 'exec', '-T', 'airflow-webserver',
                               'airflow', 'dags', 'list'], 
                              capture_output=True, text=True, check=True)
        
        if 'example_etl_dag' in result.stdout:
            print("✅ Sample DAGs loaded")
        else:
            print("⚠️  Sample DAGs not found")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Airflow CLI test failed: {e}")
        return False

def test_dag_files():
    """Test DAG files are valid"""
    print("\nTesting DAG files...")
    
    dag_dir = Path('./dags')
    if not dag_dir.exists():
        print("❌ DAGs directory not found")
        return False
    
    dag_files = list(dag_dir.glob('*.py'))
    if not dag_files:
        print("❌ No DAG files found")
        return False
    
    valid_dags = 0
    for dag_file in dag_files:
        try:
            # Basic syntax check
            with open(dag_file, 'r') as f:
                compile(f.read(), dag_file, 'exec')
            print(f"✅ {dag_file.name}")
            valid_dags += 1
        except SyntaxError as e:
            print(f"❌ {dag_file.name}: {e}")
    
    return valid_dags == len(dag_files)

def test_directories():
    """Test required directories exist"""
    print("\nTesting directory structure...")
    
    required_dirs = ['dags', 'logs', 'plugins', 'config', 'data', 'scripts']
    
    all_exist = True
    for dir_name in required_dirs:
        if Path(dir_name).exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("=== Day 12: Airflow Setup Test ===\n")
    
    tests = [
        ("Python Packages", test_python_packages),
        ("Directory Structure", test_directories),
        ("DAG Files", test_dag_files),
        ("Docker Services", test_docker_services),
        ("Airflow Webserver", test_airflow_webserver),
        ("Airflow CLI", test_airflow_cli),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append(False)
        print()
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Airflow setup is complete.")
        print("\nNext steps:")
        print("1. Open http://localhost:8080 (admin/admin)")
        print("2. Explore the sample DAGs")
        print("3. Run: python exercise.py")
        print("4. Complete the TODO exercises")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        print("\nTroubleshooting:")
        print("1. Check Docker services: docker-compose ps")
        print("2. View logs: docker-compose logs")
        print("3. Restart services: docker-compose restart")
        return 1

if __name__ == "__main__":
    sys.exit(main())