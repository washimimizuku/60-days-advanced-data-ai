#!/usr/bin/env python3
"""
Day 9: Data Lineage Tracking - Setup Test Script
Test all database connections and verify environment setup
"""

import os
import sys
import time
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_postgres_connection() -> Dict[str, Any]:
    """Test PostgreSQL connection"""
    try:
        import psycopg2
        
        conn_params = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': os.getenv('POSTGRES_PORT', '5432'),
            'database': os.getenv('POSTGRES_DB', 'lineage_db'),
            'user': os.getenv('POSTGRES_USER', 'lineage_user'),
            'password': os.getenv('POSTGRES_PASSWORD', 'secure_password')
        }
        
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        return {
            'status': 'success',
            'version': version[:50] + '...',
            'connection_params': {k: v for k, v in conn_params.items() if k != 'password'}
        }
        
    except Exception as e:
        return {
            'status': 'failed',
            'error': str(e),
            'suggestion': 'Check PostgreSQL service and credentials'
        }

def test_python_dependencies() -> Dict[str, Any]:
    """Test required Python dependencies"""
    required_packages = [
        'sqlparse', 'networkx', 'pandas', 'numpy'
    ]
    
    optional_packages = [
        'psycopg2', 'pymongo', 'redis', 'neo4j'
    ]
    
    results = {}
    missing_required = []
    missing_optional = []
    
    for package in required_packages:
        try:
            __import__(package)
            results[package] = 'installed'
        except ImportError:
            results[package] = 'missing'
            missing_required.append(package)
    
    for package in optional_packages:
        try:
            __import__(package)
            results[package] = 'installed'
        except ImportError:
            results[package] = 'missing'
            missing_optional.append(package)
    
    status = 'success' if not missing_required else 'partial' if not missing_optional else 'failed'
    
    return {
        'status': status,
        'results': results,
        'missing_required': missing_required,
        'missing_optional': missing_optional,
        'suggestion': f'Install: pip install {" ".join(missing_required + missing_optional)}' if missing_required or missing_optional else 'All packages installed'
    }

def run_comprehensive_test() -> None:
    """Run comprehensive setup test"""
    print("🚀 Day 9: Data Lineage Tracking - Setup Test")
    print("=" * 60)
    
    # Test Python dependencies first
    print("\n--- Python Dependencies ---")
    deps_result = test_python_dependencies()
    
    if deps_result['status'] == 'success':
        print("✅ Python Dependencies: ALL INSTALLED")
    elif deps_result['status'] == 'partial':
        print("⚠️  Python Dependencies: CORE INSTALLED")
        if deps_result['missing_optional']:
            print(f"   Optional missing: {', '.join(deps_result['missing_optional'])}")
    else:
        print("❌ Python Dependencies: MISSING REQUIRED")
        print(f"   Required missing: {', '.join(deps_result['missing_required'])}")
        print(f"   Install with: pip install -r requirements.txt")
        return False
    
    # Test database connections if packages are available
    database_tests = []
    
    if 'psycopg2' in deps_result['results'] and deps_result['results']['psycopg2'] == 'installed':
        database_tests.append(("PostgreSQL", test_postgres_connection))
    
    for test_name, test_func in database_tests:
        print(f"\n--- {test_name} Connection ---")
        try:
            result = test_func()
            
            if result['status'] == 'success':
                print(f"✅ {test_name}: CONNECTED")
            else:
                print(f"❌ {test_name}: FAILED")
                print(f"   Error: {result.get('error', 'Unknown error')}")
                print(f"   Suggestion: {result.get('suggestion', 'Check service')}")
                
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {str(e)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SETUP STATUS")
    print("=" * 60)
    
    if deps_result['status'] in ['success', 'partial']:
        print("🎉 Core setup complete! You can run the exercises.")
        print("\nNext steps:")
        print("1. Run exercises: python exercise.py")
        print("2. Review solution: python solution_complete.py")
        print("3. Take the quiz: quiz.md")
        
        if deps_result['missing_optional']:
            print(f"\n💡 Optional: Install database drivers for full functionality:")
            print(f"   pip install {' '.join(deps_result['missing_optional'])}")
    else:
        print("⚠️  Setup incomplete. Install required dependencies first.")
        print("   pip install -r requirements.txt")
    
    return True

if __name__ == "__main__":
    run_comprehensive_test()