#!/usr/bin/env python3
"""
Setup demo data and test scenarios for Day 11
"""

import os
import sys
import psycopg2
import redis
from dotenv import load_dotenv
from datetime import datetime

def setup_database_context():
    """Set up database session context for testing"""
    print("Setting up database context...")
    
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cursor = conn.cursor()
        
        # Set session variables for testing
        test_contexts = [
            ("app.current_tenant_id", "550e8400-e29b-41d4-a716-446655440001"),  # Acme Corp
            ("app.user_regions", "US,CA"),
            ("app.user_clearance_level", "2"),
            ("app.user_department", "Analytics"),
            ("app.user_employee_id", "1001")
        ]
        
        for key, value in test_contexts:
            cursor.execute(f"SELECT set_config('{key}', '{value}', false);")
        
        conn.commit()
        print("✅ Database context configured")
        
        # Test RLS policies
        cursor.execute("SELECT COUNT(*) FROM customer_data;")
        count = cursor.fetchone()[0]
        print(f"✅ Can access {count} customer records with current context")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Database context setup failed: {e}")
        return False

def setup_redis_cache():
    """Set up Redis cache with sample data"""
    print("Setting up Redis cache...")
    
    try:
        r = redis.from_url(os.getenv('REDIS_URL'))
        
        # Sample cache entries
        cache_data = {
            "user_permissions_alice": '["read_customer_data", "read_financial_data"]',
            "user_permissions_bob": '["read_customer_data", "create_models"]',
            "tenant_limits_acme": '{"max_users": 100, "max_storage_gb": 1000}',
            "policy_cache_customer_data": '{"applicable_policies": ["tenant_isolation", "regional_access"]}'
        }
        
        for key, value in cache_data.items():
            r.setex(key, 900, value)  # 15 minute TTL
        
        print(f"✅ Cached {len(cache_data)} sample entries")
        return True
        
    except Exception as e:
        print(f"❌ Redis cache setup failed: {e}")
        return False

def create_sample_audit_logs():
    """Create sample audit log entries"""
    print("Creating sample audit logs...")
    
    try:
        conn = psycopg2.connect(os.getenv('DATABASE_URL'))
        cursor = conn.cursor()
        
        # Sample audit events
        audit_events = [
            ("authentication", "alice", "success", "192.168.1.100", "login"),
            ("data_access", "alice", "success", "192.168.1.100", "select"),
            ("authentication", "bob", "success", "192.168.1.101", "login"),
            ("data_access", "bob", "success", "192.168.1.101", "select"),
            ("permission_change", "admin", "success", "192.168.1.1", "grant_permission"),
            ("authentication", "suspicious_user", "failure", "10.0.0.1", "login_attempt"),
            ("authentication", "suspicious_user", "failure", "10.0.0.1", "login_attempt"),
            ("authentication", "suspicious_user", "failure", "10.0.0.1", "login_attempt")
        ]
        
        for event_type, user_id, result, ip, action in audit_events:
            cursor.execute("""
                INSERT INTO audit_log (
                    event_id, event_type, user_id, tenant_id, resource, 
                    action, result, ip_address, user_agent, session_id, additional_data
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}",
                event_type,
                user_id,
                "550e8400-e29b-41d4-a716-446655440001",
                "system" if event_type == "authentication" else "customer_data",
                action,
                result,
                ip,
                "Mozilla/5.0 (Demo Browser)",
                f"sess_{user_id}_{datetime.now().strftime('%H%M%S')}",
                '{"demo": true, "timestamp": "' + datetime.now().isoformat() + '"}'
            ))
        
        conn.commit()
        print(f"✅ Created {len(audit_events)} sample audit log entries")
        
        cursor.close()
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Audit log setup failed: {e}")
        return False

def test_access_control_scenarios():
    """Test various access control scenarios"""
    print("Testing access control scenarios...")
    
    try:
        # Add parent directory to path
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from solution import RBACSystem, Permission
        
        # Test RBAC
        rbac = RBACSystem()
        
        # Create test roles
        rbac.create_role("test_analyst", {Permission.READ_CUSTOMER_DATA})
        rbac.create_role("test_scientist", {Permission.CREATE_MODELS}, ["test_analyst"])
        
        # Assign roles
        rbac.assign_role_to_user("test_alice", "test_analyst")
        rbac.assign_role_to_user("test_bob", "test_scientist")
        
        # Test permissions
        alice_can_read = rbac.check_permission("test_alice", Permission.READ_CUSTOMER_DATA)
        bob_can_read = rbac.check_permission("test_bob", Permission.READ_CUSTOMER_DATA)  # Inherited
        bob_can_create = rbac.check_permission("test_bob", Permission.CREATE_MODELS)
        
        if alice_can_read and bob_can_read and bob_can_create:
            print("✅ RBAC system working correctly")
        else:
            print("❌ RBAC system test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Access control scenario test failed: {e}")
        return False

def display_demo_info():
    """Display information about the demo setup"""
    print("\n=== Demo Setup Complete ===")
    print("\n📊 Sample Data Available:")
    print("• 3 tenants: Acme Corp (premium), Beta Inc (basic), Gamma Ltd (enterprise)")
    print("• 5 users: alice, bob, charlie, diana, admin")
    print("• Customer data with different sensitivity levels")
    print("• Financial data with high security requirements")
    print("• Employee data for hierarchical access testing")
    print("• Audit logs with various event types")
    
    print("\n🔐 Access Control Features:")
    print("• RBAC with role inheritance")
    print("• Row-Level Security policies")
    print("• ABAC with dynamic attributes")
    print("• Multi-tenant data isolation")
    print("• Comprehensive audit logging")
    
    print("\n🧪 Test Scenarios:")
    print("• Regional data access restrictions")
    print("• Sensitivity-based filtering")
    print("• Time-based access controls")
    print("• Hierarchical permissions")
    print("• Suspicious activity detection")
    
    print("\n🚀 Next Steps:")
    print("1. Run exercises: python exercise.py")
    print("2. Test with different user contexts")
    print("3. Explore RLS policies in PostgreSQL")
    print("4. Check audit logs and security alerts")
    print("5. Review production implementation: python solution.py")

def main():
    """Set up demo environment"""
    print("=== Day 11: Access Control Demo Setup ===\n")
    
    load_dotenv()
    
    setup_tasks = [
        setup_database_context,
        setup_redis_cache,
        create_sample_audit_logs,
        test_access_control_scenarios
    ]
    
    results = []
    for task in setup_tasks:
        try:
            result = task()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Task {task.__name__} failed: {e}\n")
            results.append(False)
    
    if all(results):
        display_demo_info()
        return 0
    else:
        print("❌ Demo setup incomplete. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())