#!/usr/bin/env python3
"""
Test setup for Day 11 access control systems
"""

import os
import sys
import psycopg2
import redis
from dotenv import load_dotenv

def test_environment():
    """Test environment configuration"""
    print("Testing environment configuration...")
    
    load_dotenv()
    
    required_vars = [
        'DATABASE_URL',
        'REDIS_URL', 
        'SECRET_KEY',
        'JWT_SECRET_KEY',
        'ENCRYPTION_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        return False
    
    print("✅ Environment configuration OK")
    return True

def test_database():
    """Test database connection"""
    print("Testing database connection...")
    
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cursor = conn.cursor()
        
        # Test basic query
        cursor.execute("SELECT version();")
        version = cursor.fetchone()[0]
        print(f"✅ Database connected: {version[:50]}...")
        
        # Test tables exist
        cursor.execute("""
            SELECT table_name FROM information_schema.tables 
            WHERE table_schema = 'public' AND table_name IN 
            ('users', 'tenants', 'customer_data', 'financial_data', 'employee_data')
        """)
        tables = [row[0] for row in cursor.fetchall()]
        
        expected_tables = ['users', 'tenants', 'customer_data', 'financial_data', 'employee_data']
        missing_tables = set(expected_tables) - set(tables)
        
        if missing_tables:
            print(f"❌ Missing tables: {', '.join(missing_tables)}")
            return False
        
        print("✅ All required tables exist")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False

def test_redis():
    """Test Redis connection"""
    print("Testing Redis connection...")
    
    try:
        r = redis.from_url(os.getenv('REDIS_URL'))
        r.ping()
        
        # Test basic operations
        r.set('test_key', 'test_value', ex=10)
        value = r.get('test_key')
        
        if value.decode() == 'test_value':
            print("✅ Redis connection and operations OK")
            r.delete('test_key')
            return True
        else:
            print("❌ Redis operations failed")
            return False
            
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False

def test_imports():
    """Test required Python imports"""
    print("Testing Python imports...")
    
    required_modules = [
        'cryptography',
        'pyjwt',
        'passlib',
        'bcrypt',
        'psycopg2',
        'redis',
        'structlog',
        'pydantic'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"❌ Missing Python modules: {', '.join(missing_modules)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✅ All required Python modules available")
    return True

def test_access_control_classes():
    """Test access control class imports"""
    print("Testing access control classes...")
    
    try:
        # Add parent directory to path to import solution
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from solution import (
            RBACSystem, 
            RowLevelSecurityManager, 
            ABACEngine,
            MultiTenantSecurityManager,
            AuditLogger,
            Permission
        )
        
        # Test basic instantiation
        rbac = RBACSystem()
        rls = RowLevelSecurityManager()
        abac = ABACEngine()
        mt_security = MultiTenantSecurityManager()
        audit = AuditLogger()
        
        print("✅ Access control classes imported and instantiated successfully")
        return True
        
    except Exception as e:
        print(f"❌ Access control class import failed: {e}")
        return False

def main():
    """Run all setup tests"""
    print("=== Day 11: Access Control Setup Test ===\n")
    
    tests = [
        test_environment,
        test_imports,
        test_database,
        test_redis,
        test_access_control_classes
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}\n")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    
    print("=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Run: python exercise.py")
        print("2. Complete the TODO exercises")
        print("3. Check your work with: python solution.py")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())