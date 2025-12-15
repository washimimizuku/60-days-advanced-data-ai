#!/usr/bin/env python3
"""
Setup verification script for 60 Days Advanced Data and AI bootcamp.
Run this to verify your environment is ready.
"""

import sys
import subprocess

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"❌ Python version: {version.major}.{version.minor}.{version.micro}")
        print("   Required: Python 3.11 or higher")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} installed")
        return True
    except ImportError:
        print(f"❌ {package_name} not installed")
        return False

def check_docker():
    """Check if Docker is running"""
    try:
        result = subprocess.run(
            ['docker', 'ps'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("✅ Docker is running")
            return True
        else:
            print("❌ Docker is not running")
            print("   Start Docker Desktop and try again")
            return False
    except FileNotFoundError:
        print("❌ Docker not installed")
        print("   Install Docker Desktop from docker.com")
        return False
    except subprocess.TimeoutExpired:
        print("⚠️  Docker check timed out")
        return False

def main():
    print("=" * 60)
    print("60 DAYS ADVANCED DATA AND AI - SETUP VERIFICATION")
    print("=" * 60)
    print()
    
    checks = []
    
    # Check Python version
    checks.append(check_python_version())
    print()
    
    # Check Docker
    checks.append(check_docker())
    print()
    
    # Check core packages
    print("Checking core packages...")
    core_packages = [
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('matplotlib', 'matplotlib'),
        ('requests', 'requests'),
    ]
    
    for package, import_name in core_packages:
        checks.append(check_package(package, import_name))
    
    print()
    
    # Check data engineering packages
    print("Checking data engineering packages...")
    de_packages = [
        ('psycopg2', 'psycopg2'),
        ('pymongo', 'pymongo'),
        ('redis', 'redis'),
    ]
    
    for package, import_name in de_packages:
        checks.append(check_package(package, import_name))
    
    print()
    
    # Check ML packages
    print("Checking ML packages...")
    ml_packages = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
    ]
    
    for package, import_name in ml_packages:
        checks.append(check_package(package, import_name))
    
    print()
    
    # Summary
    print("=" * 60)
    passed = sum(checks)
    total = len(checks)
    
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print()
        print("You're ready to start Day 1!")
        print("Navigate to: days/day-01-postgresql-advanced/")
        print()
        return 0
    else:
        print(f"⚠️  SOME CHECKS FAILED ({passed}/{total} passed)")
        print()
        print("Please install missing packages:")
        print("  pip install -r requirements.txt")
        print()
        print("For help, see:")
        print("  - docs/SETUP.md")
        print("  - docs/TROUBLESHOOTING.md")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
