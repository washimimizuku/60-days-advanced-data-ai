#!/usr/bin/env python3
"""
Test dbt setup and validate installation
"""

import os
import sys
import subprocess
from pathlib import Path

def test_dbt_installation():
    """Test dbt is installed and accessible"""
    print("Testing dbt installation...")
    
    try:
        result = subprocess.run(['dbt', '--version'], 
                              capture_output=True, text=True, check=True)
        print(f"✅ dbt version: {result.stdout.strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ dbt not found or not working")
        return False

def test_database_connection():
    """Test database connection"""
    print("Testing database connection...")
    
    try:
        result = subprocess.run(['dbt', 'debug'], 
                              capture_output=True, text=True, check=True)
        if "All checks passed!" in result.stdout:
            print("✅ Database connection successful")
            return True
        else:
            print("❌ Database connection failed")
            print(result.stdout)
            return False
    except subprocess.CalledProcessError as e:
        print(f"❌ dbt debug failed: {e}")
        return False

def test_project_structure():
    """Test dbt project structure"""
    print("Testing project structure...")
    
    required_files = [
        'dbt_project.yml',
        'models/sources.yml',
        'models/staging/stg_customers.sql',
        'macros/calculate_profit_margin.sql',
        'seeds/customers.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    return True

def test_dbt_commands():
    """Test basic dbt commands"""
    print("Testing dbt commands...")
    
    commands = [
        (['dbt', 'deps'], "Install packages"),
        (['dbt', 'compile'], "Compile models"),
        (['dbt', 'seed'], "Load seed data"),
    ]
    
    for cmd, description in commands:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {description}")
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed: {e}")
            return False
    
    return True

def main():
    """Run all dbt tests"""
    print("=== Day 13: dbt Setup Test ===\n")
    
    # Change to dbt project directory
    os.chdir('ecommerce_analytics')
    
    tests = [
        ("dbt Installation", test_dbt_installation),
        ("Project Structure", test_project_structure),
        ("Database Connection", test_database_connection),
        ("dbt Commands", test_dbt_commands),
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
        print("🎉 All tests passed! dbt setup is complete.")
        print("\nNext steps:")
        print("1. Run: dbt run")
        print("2. Run: dbt test")
        print("3. Run: dbt docs generate && dbt docs serve")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())