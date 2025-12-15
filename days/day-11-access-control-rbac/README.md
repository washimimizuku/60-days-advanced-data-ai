# Day 11: Access Control - RBAC, Row-Level Security

## 📖 Learning Objectives

By the end of today, you will:
- Understand comprehensive access control models (RBAC, ABAC, MAC, DAC)
- Implement role-based access control systems with hierarchical permissions
- Build row-level security policies for fine-grained data access
- Apply attribute-based access control for dynamic authorization
- Create multi-tenant security architectures with data isolation
- Deploy production-ready access control systems with audit logging

---

## Theory

### The Access Control Landscape

Access control is the foundation of data security, determining who can access what data under which circumstances. Modern data systems require sophisticated access control mechanisms that can scale to thousands of users while maintaining security and performance.

**Core Access Control Models**:
- **DAC** (Discretionary Access Control) - Owner-controlled permissions
- **MAC** (Mandatory Access Control) - System-enforced security labels
- **RBAC** (Role-Based Access Control) - Permission assignment through roles
- **ABAC** (Attribute-Based Access Control) - Policy-based dynamic authorization

### Why Access Control Matters

#### The Security Breach Reality

```
Without Proper Access Control:
💰 Average data breach cost: $4.45 million (2023)
📊 83% of breaches involve internal actors
⏱️ Average time to identify breach: 277 days
🎯 Insider threats account for 34% of breaches
📈 Regulatory fines increasing 50% year-over-year
```

```
With Comprehensive Access Control:
✅ 60% reduction in security incidents
🛡️ 80% faster threat detection and response
📉 90% reduction in unauthorized data access
⚡ Automated compliance reporting
🔍 Complete audit trails for investigations
```

#### Real-World Access Control Failures

**Major Breaches Due to Access Control Issues**:
- **Capital One (2019)**: Misconfigured firewall rules exposed 100M+ records
- **Equifax (2017)**: Unpatched systems with excessive privileges
- **Marriott (2018)**: Inadequate access controls in acquired systems
- **SolarWinds (2020)**: Compromised privileged accounts

**Common Access Control Vulnerabilities**:
- Over-privileged accounts (principle of least privilege violations)
- Stale permissions (accounts not deprovisioned)
- Shared accounts and credentials
- Inadequate segregation of duties
- Missing audit trails and monitoring

### Role-Based Access Control (RBAC)

#### RBAC Fundamentals

RBAC is the most widely adopted access control model, organizing permissions around roles that reflect organizational functions and responsibilities.

**Core RBAC Components**:
```python
rbac_model = {
    "users": ["alice", "bob", "charlie", "diana"],
    "roles": ["data_analyst", "data_scientist", "data_engineer", "admin"],
    "permissions": [
        "read_customer_data", "write_customer_data",
        "read_financial_data", "write_financial_data",
        "create_models", "deploy_models",
        "manage_users", "manage_roles"
    ],
    "user_role_assignments": {
        "alice": ["data_analyst"],
        "bob": ["data_scientist"],
        "charlie": ["data_engineer"],
        "diana": ["admin"]
    },
    "role_permission_assignments": {
        "data_analyst": ["read_customer_data", "read_financial_data"],
        "data_scientist": ["read_customer_data", "create_models"],
        "data_engineer": ["read_customer_data", "write_customer_data"],
        "admin": ["manage_users", "manage_roles"]
    }
}
```

#### Hierarchical RBAC

Real organizations require role hierarchies that reflect reporting structures and responsibility levels:

```python
role_hierarchy = {
    "senior_data_scientist": {
        "inherits_from": ["data_scientist"],
        "additional_permissions": ["deploy_models", "review_models"]
    },
    "lead_data_engineer": {
        "inherits_from": ["data_engineer"],
        "additional_permissions": ["manage_pipelines", "approve_deployments"]
    },
    "data_team_manager": {
        "inherits_from": ["senior_data_scientist", "lead_data_engineer"],
        "additional_permissions": ["manage_team", "budget_approval"]
    }
}
```

#### RBAC Implementation Patterns

**Database-Level RBAC**:
```sql
-- PostgreSQL RBAC implementation
-- Create roles
CREATE ROLE data_analyst;
CREATE ROLE data_scientist;
CREATE ROLE data_engineer;

-- Grant permissions to roles
GRANT SELECT ON customer_data TO data_analyst;
GRANT SELECT ON financial_data TO data_analyst;

GRANT SELECT ON customer_data TO data_scientist;
GRANT SELECT, INSERT, UPDATE ON ml_models TO data_scientist;

GRANT SELECT, INSERT, UPDATE, DELETE ON customer_data TO data_engineer;
GRANT ALL PRIVILEGES ON etl_jobs TO data_engineer;

-- Assign roles to users
GRANT data_analyst TO alice;
GRANT data_scientist TO bob;
GRANT data_engineer TO charlie;

-- Role hierarchy
GRANT data_analyst TO data_scientist;  -- Scientists inherit analyst permissions
GRANT data_scientist TO data_engineer; -- Engineers inherit scientist permissions
```

**Application-Level RBAC**:
```python
from enum import Enum
from typing import Dict, List, Set
from dataclasses import dataclass

class Permission(Enum):
    READ_CUSTOMER_DATA = "read_customer_data"
    WRITE_CUSTOMER_DATA = "write_customer_data"
    READ_FINANCIAL_DATA = "read_financial_data"
    WRITE_FINANCIAL_DATA = "write_financial_data"
    CREATE_MODELS = "create_models"
    DEPLOY_MODELS = "deploy_models"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"

@dataclass
class Role:
    name: str
    permissions: Set[Permission]
    inherits_from: List[str] = None

@dataclass
class User:
    username: str
    roles: Set[str]
    attributes: Dict[str, str] = None

class RBACSystem:
    """Production RBAC implementation"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.role_hierarchy: Dict[str, List[str]] = {}
    
    def create_role(self, name: str, permissions: Set[Permission], 
                   inherits_from: List[str] = None):
        """Create a new role with permissions"""
        self.roles[name] = Role(name, permissions, inherits_from)
        
        if inherits_from:
            self.role_hierarchy[name] = inherits_from
    
    def assign_role_to_user(self, username: str, role_name: str):
        """Assign role to user"""
        if username not in self.users:
            self.users[username] = User(username, set())
        
        self.users[username].roles.add(role_name)
    
    def get_user_permissions(self, username: str) -> Set[Permission]:
        """Get all permissions for a user (including inherited)"""
        if username not in self.users:
            return set()
        
        all_permissions = set()
        user_roles = self.users[username].roles
        
        # Get permissions from all assigned roles
        for role_name in user_roles:
            all_permissions.update(self._get_role_permissions(role_name))
        
        return all_permissions
    
    def _get_role_permissions(self, role_name: str) -> Set[Permission]:
        """Get all permissions for a role (including inherited)"""
        if role_name not in self.roles:
            return set()
        
        role = self.roles[role_name]
        permissions = role.permissions.copy()
        
        # Add inherited permissions
        if role.inherits_from:
            for parent_role in role.inherits_from:
                permissions.update(self._get_role_permissions(parent_role))
        
        return permissions
    
    def check_permission(self, username: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.get_user_permissions(username)
        return permission in user_permissions
    
    def get_users_with_permission(self, permission: Permission) -> List[str]:
        """Find all users with a specific permission"""
        users_with_permission = []
        
        for username in self.users:
            if self.check_permission(username, permission):
                users_with_permission.append(username)
        
        return users_with_permission

# Example usage
rbac = RBACSystem()

# Create roles
rbac.create_role("data_analyst", {
    Permission.READ_CUSTOMER_DATA,
    Permission.READ_FINANCIAL_DATA
})

rbac.create_role("data_scientist", {
    Permission.CREATE_MODELS
}, inherits_from=["data_analyst"])

rbac.create_role("data_engineer", {
    Permission.WRITE_CUSTOMER_DATA,
    Permission.DEPLOY_MODELS
}, inherits_from=["data_scientist"])

# Assign roles
rbac.assign_role_to_user("alice", "data_analyst")
rbac.assign_role_to_user("bob", "data_scientist")
rbac.assign_role_to_user("charlie", "data_engineer")

# Check permissions
print(f"Alice can read customer data: {rbac.check_permission('alice', Permission.READ_CUSTOMER_DATA)}")
print(f"Bob can create models: {rbac.check_permission('bob', Permission.CREATE_MODELS)}")
print(f"Charlie can deploy models: {rbac.check_permission('charlie', Permission.DEPLOY_MODELS)}")
```

### Row-Level Security (RLS)

Row-Level Security provides fine-grained access control at the data row level, ensuring users only see data they're authorized to access.

#### RLS Fundamentals

**Why Row-Level Security**:
- **Multi-tenancy**: Isolate data between customers/organizations
- **Geographic restrictions**: Limit data access by region
- **Hierarchical access**: Managers see subordinate data
- **Privacy compliance**: Restrict PII access based on roles
- **Data classification**: Control access based on sensitivity levels

#### PostgreSQL RLS Implementation

**Basic RLS Setup**:
```sql
-- Enable RLS on table
ALTER TABLE customer_data ENABLE ROW LEVEL SECURITY;

-- Create policy for data analysts (only see non-sensitive data)
CREATE POLICY analyst_policy ON customer_data
    FOR SELECT
    TO data_analyst
    USING (sensitivity_level != 'confidential');

-- Create policy for managers (see all data in their region)
CREATE POLICY manager_policy ON customer_data
    FOR ALL
    TO data_manager
    USING (region = current_setting('app.user_region'));

-- Create policy for customer service (only see assigned customers)
CREATE POLICY customer_service_policy ON customer_data
    FOR SELECT
    TO customer_service
    USING (assigned_rep = current_user);
```

**Advanced RLS with Functions**:
```sql
-- Create function to get user's accessible regions
CREATE OR REPLACE FUNCTION get_user_regions(username TEXT)
RETURNS TEXT[] AS $$
BEGIN
    -- In production, this would query user permissions table
    CASE username
        WHEN 'alice' THEN RETURN ARRAY['US', 'CA'];
        WHEN 'bob' THEN RETURN ARRAY['EU'];
        WHEN 'charlie' THEN RETURN ARRAY['US', 'EU', 'CA'];
        ELSE RETURN ARRAY[]::TEXT[];
    END CASE;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Policy using function
CREATE POLICY regional_access_policy ON customer_data
    FOR ALL
    TO data_user
    USING (region = ANY(get_user_regions(current_user)));

-- Time-based access policy
CREATE POLICY business_hours_policy ON sensitive_data
    FOR ALL
    TO business_user
    USING (
        EXTRACT(hour FROM now()) BETWEEN 9 AND 17
        AND EXTRACT(dow FROM now()) BETWEEN 1 AND 5
    );

-- Hierarchical access policy
CREATE POLICY hierarchy_policy ON employee_data
    FOR SELECT
    TO manager
    USING (
        manager_id = (
            SELECT employee_id 
            FROM employees 
            WHERE username = current_user
        )
        OR employee_id = (
            SELECT employee_id 
            FROM employees 
            WHERE username = current_user
        )
    );
```

#### Application-Level RLS

```python
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class AccessLevel(Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"

@dataclass
class RowLevelPolicy:
    name: str
    table: str
    role: str
    access_level: AccessLevel
    condition: str  # SQL-like condition
    
class RowLevelSecurityManager:
    """Application-level row-level security"""
    
    def __init__(self):
        self.policies: List[RowLevelPolicy] = []
        self.user_context: Dict[str, Any] = {}
    
    def add_policy(self, policy: RowLevelPolicy):
        """Add a row-level security policy"""
        self.policies.append(policy)
    
    def set_user_context(self, username: str, context: Dict[str, Any]):
        """Set context for current user session"""
        self.user_context[username] = context
    
    def apply_rls_filter(self, username: str, table: str, 
                        access_level: AccessLevel, 
                        base_query: str) -> str:
        """Apply RLS policies to a query"""
        
        user_roles = self.user_context.get(username, {}).get('roles', [])
        applicable_policies = []
        
        # Find applicable policies
        for policy in self.policies:
            if (policy.table == table and 
                policy.role in user_roles and 
                policy.access_level == access_level):
                applicable_policies.append(policy)
        
        if not applicable_policies:
            return base_query
        
        # Build WHERE clause from policies
        conditions = []
        for policy in applicable_policies:
            condition = self._substitute_context(
                policy.condition, 
                self.user_context.get(username, {})
            )
            conditions.append(f"({condition})")
        
        # Combine conditions with OR (user needs to satisfy any policy)
        combined_condition = " OR ".join(conditions)
        
        # Add to query
        if "WHERE" in base_query.upper():
            return f"{base_query} AND ({combined_condition})"
        else:
            return f"{base_query} WHERE ({combined_condition})"
    
    def _substitute_context(self, condition: str, context: Dict[str, Any]) -> str:
        """Substitute context variables in condition"""
        
        # Simple substitution - in production, use proper SQL parameter binding
        for key, value in context.items():
            placeholder = f"${key}"
            if isinstance(value, str):
                condition = condition.replace(placeholder, f"'{value}'")
            elif isinstance(value, list):
                value_list = "', '".join(str(v) for v in value)
                condition = condition.replace(placeholder, f"('{value_list}')")
            else:
                condition = condition.replace(placeholder, str(value))
        
        return condition

# Example usage
rls_manager = RowLevelSecurityManager()

# Add policies
rls_manager.add_policy(RowLevelPolicy(
    name="regional_access",
    table="customer_data",
    role="data_analyst",
    access_level=AccessLevel.READ,
    condition="region IN $user_regions"
))

rls_manager.add_policy(RowLevelPolicy(
    name="sensitivity_filter",
    table="customer_data", 
    role="data_analyst",
    access_level=AccessLevel.READ,
    condition="sensitivity_level != 'confidential'"
))

rls_manager.add_policy(RowLevelPolicy(
    name="manager_access",
    table="employee_data",
    role="manager",
    access_level=AccessLevel.READ,
    condition="department = '$user_department' OR employee_id = $user_id"
))

# Set user context
rls_manager.set_user_context("alice", {
    "roles": ["data_analyst"],
    "user_regions": ["US", "CA"],
    "user_id": 123,
    "user_department": "Engineering"
})

# Apply RLS to query
base_query = "SELECT * FROM customer_data"
filtered_query = rls_manager.apply_rls_filter(
    "alice", "customer_data", AccessLevel.READ, base_query
)

print(f"Original: {base_query}")
print(f"Filtered: {filtered_query}")
```

### Attribute-Based Access Control (ABAC)

ABAC provides the most flexible access control model, making authorization decisions based on attributes of users, resources, actions, and environment.

#### ABAC Architecture

**ABAC Components**:
```python
from typing import Dict, Any, List
from dataclasses import dataclass
from datetime import datetime, time
import json

@dataclass
class Subject:
    """User or system requesting access"""
    id: str
    attributes: Dict[str, Any]  # role, department, clearance_level, etc.

@dataclass 
class Resource:
    """Data or system being accessed"""
    id: str
    attributes: Dict[str, Any]  # classification, owner, sensitivity, etc.

@dataclass
class Action:
    """Operation being performed"""
    name: str
    attributes: Dict[str, Any]  # operation_type, risk_level, etc.

@dataclass
class Environment:
    """Context of the access request"""
    attributes: Dict[str, Any]  # time, location, network, device, etc.

@dataclass
class Policy:
    """ABAC policy rule"""
    id: str
    name: str
    description: str
    target: str  # When this policy applies
    condition: str  # Access control logic
    effect: str  # "permit" or "deny"
    priority: int = 0

class ABACEngine:
    """Attribute-Based Access Control engine"""
    
    def __init__(self):
        self.policies: List[Policy] = []
    
    def add_policy(self, policy: Policy):
        """Add an ABAC policy"""
        self.policies.append(policy)
        # Sort by priority (higher priority first)
        self.policies.sort(key=lambda p: p.priority, reverse=True)
    
    def evaluate(self, subject: Subject, resource: Resource, 
                action: Action, environment: Environment) -> Dict[str, Any]:
        """Evaluate access request against policies"""
        
        context = {
            "subject": subject.attributes,
            "resource": resource.attributes,
            "action": action.attributes,
            "environment": environment.attributes
        }
        
        # Default deny
        decision = "deny"
        applicable_policies = []
        
        for policy in self.policies:
            # Check if policy applies to this request
            if self._evaluate_target(policy.target, context):
                applicable_policies.append(policy)
                
                # Evaluate policy condition
                if self._evaluate_condition(policy.condition, context):
                    decision = policy.effect
                    
                    # First applicable policy wins (due to priority sorting)
                    break
        
        return {
            "decision": decision,
            "applicable_policies": [p.id for p in applicable_policies],
            "context": context
        }
    
    def _evaluate_target(self, target: str, context: Dict[str, Any]) -> bool:
        """Evaluate if policy target matches request"""
        try:
            # Simple expression evaluation
            # In production, use a proper policy language parser
            return eval(target, {"__builtins__": {}}, context)
        except:
            return False
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate policy condition"""
        try:
            # Add helper functions to context
            evaluation_context = context.copy()
            evaluation_context.update({
                "current_time": datetime.now().time(),
                "current_date": datetime.now().date(),
                "is_business_hours": self._is_business_hours(),
                "is_weekend": datetime.now().weekday() >= 5
            })
            
            return eval(condition, {"__builtins__": {}}, evaluation_context)
        except Exception as e:
            print(f"Error evaluating condition: {e}")
            return False
    
    def _is_business_hours(self) -> bool:
        """Check if current time is business hours"""
        current_time = datetime.now().time()
        return time(9, 0) <= current_time <= time(17, 0)

# Example ABAC policies
abac_engine = ABACEngine()

# Policy 1: Data analysts can read customer data during business hours
abac_engine.add_policy(Policy(
    id="analyst_customer_read",
    name="Analyst Customer Data Read",
    description="Data analysts can read customer data during business hours",
    target="subject['role'] == 'data_analyst' and resource['type'] == 'customer_data'",
    condition="action['name'] == 'read' and is_business_hours and not is_weekend",
    effect="permit",
    priority=100
))

# Policy 2: Managers can access all data in their department
abac_engine.add_policy(Policy(
    id="manager_department_access",
    name="Manager Department Access",
    description="Managers can access all data in their department",
    target="subject['role'] == 'manager'",
    condition="resource['department'] == subject['department']",
    effect="permit",
    priority=200
))

# Policy 3: High sensitivity data requires high clearance
abac_engine.add_policy(Policy(
    id="high_sensitivity_clearance",
    name="High Sensitivity Clearance Required",
    description="High sensitivity data requires high clearance level",
    target="resource['sensitivity'] == 'high'",
    condition="subject['clearance_level'] >= 3",
    effect="permit",
    priority=300
))

# Policy 4: Deny access from untrusted networks
abac_engine.add_policy(Policy(
    id="untrusted_network_deny",
    name="Untrusted Network Deny",
    description="Deny access from untrusted networks",
    target="True",  # Applies to all requests
    condition="environment['network_trust'] == 'untrusted'",
    effect="deny",
    priority=1000  # High priority deny policy
))

# Example access request
subject = Subject("alice", {
    "role": "data_analyst",
    "department": "marketing",
    "clearance_level": 2
})

resource = Resource("customer_profiles", {
    "type": "customer_data",
    "department": "marketing",
    "sensitivity": "medium",
    "classification": "internal"
})

action = Action("read", {
    "name": "read",
    "operation_type": "query"
})

environment = Environment({
    "network_trust": "trusted",
    "location": "office",
    "device_type": "laptop"
})

# Evaluate access request
result = abac_engine.evaluate(subject, resource, action, environment)
print(f"Access decision: {result['decision']}")
print(f"Applicable policies: {result['applicable_policies']}")
```

### Multi-Tenant Security Architecture

Multi-tenancy requires careful design to ensure complete data isolation between tenants while maintaining performance and scalability.

#### Tenant Isolation Strategies

**1. Database-Level Isolation**:
```sql
-- Separate database per tenant
CREATE DATABASE tenant_123;
CREATE DATABASE tenant_456;

-- Pros: Complete isolation, easy backup/restore
-- Cons: Resource overhead, complex management at scale
```

**2. Schema-Level Isolation**:
```sql
-- Separate schema per tenant
CREATE SCHEMA tenant_123;
CREATE SCHEMA tenant_456;

CREATE TABLE tenant_123.customers (...);
CREATE TABLE tenant_456.customers (...);

-- Pros: Good isolation, shared resources
-- Cons: Schema proliferation, complex queries
```

**3. Row-Level Isolation**:
```sql
-- Shared tables with tenant_id column
CREATE TABLE customers (
    id SERIAL PRIMARY KEY,
    tenant_id UUID NOT NULL,
    name VARCHAR(255),
    email VARCHAR(255),
    -- Add tenant_id to all indexes
    CONSTRAINT fk_tenant FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- RLS policy for tenant isolation
CREATE POLICY tenant_isolation ON customers
    FOR ALL
    TO application_role
    USING (tenant_id = current_setting('app.current_tenant_id')::UUID);

-- Pros: Efficient resource usage, simple management
-- Cons: Risk of data leakage, complex application logic
```

#### Production Multi-Tenant Implementation

```python
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from contextlib import contextmanager
import uuid
import threading

@dataclass
class Tenant:
    id: str
    name: str
    plan: str  # basic, premium, enterprise
    settings: Dict[str, Any]
    created_at: str
    status: str  # active, suspended, deleted

class TenantContext:
    """Thread-local tenant context"""
    
    def __init__(self):
        self._local = threading.local()
    
    def set_tenant(self, tenant_id: str):
        """Set current tenant for this thread"""
        self._local.tenant_id = tenant_id
    
    def get_tenant(self) -> Optional[str]:
        """Get current tenant for this thread"""
        return getattr(self._local, 'tenant_id', None)
    
    def clear_tenant(self):
        """Clear tenant context"""
        if hasattr(self._local, 'tenant_id'):
            delattr(self._local, 'tenant_id')

class MultiTenantSecurityManager:
    """Multi-tenant security and isolation manager"""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_context = TenantContext()
        self.tenant_permissions: Dict[str, Dict[str, List[str]]] = {}
    
    def register_tenant(self, tenant: Tenant):
        """Register a new tenant"""
        self.tenants[tenant.id] = tenant
        self.tenant_permissions[tenant.id] = {}
    
    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get tenant by ID"""
        return self.tenants.get(tenant_id)
    
    @contextmanager
    def tenant_context_manager(self, tenant_id: str):
        """Context manager for tenant operations"""
        if tenant_id not in self.tenants:
            raise ValueError(f"Unknown tenant: {tenant_id}")
        
        # Check tenant status
        tenant = self.tenants[tenant_id]
        if tenant.status != 'active':
            raise ValueError(f"Tenant {tenant_id} is not active")
        
        # Set tenant context
        self.tenant_context.set_tenant(tenant_id)
        try:
            yield tenant
        finally:
            self.tenant_context.clear_tenant()
    
    def add_tenant_permission(self, tenant_id: str, user_id: str, permissions: List[str]):
        """Add permissions for user within tenant"""
        if tenant_id not in self.tenant_permissions:
            self.tenant_permissions[tenant_id] = {}
        
        self.tenant_permissions[tenant_id][user_id] = permissions
    
    def check_tenant_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has permission in current tenant"""
        current_tenant = self.tenant_context.get_tenant()
        if not current_tenant:
            return False
        
        user_permissions = self.tenant_permissions.get(current_tenant, {}).get(user_id, [])
        return permission in user_permissions
    
    def get_tenant_filtered_query(self, base_query: str, table_name: str) -> str:
        """Add tenant filtering to SQL query"""
        current_tenant = self.tenant_context.get_tenant()
        if not current_tenant:
            raise ValueError("No tenant context set")
        
        # Add tenant_id filter
        tenant_filter = f"tenant_id = '{current_tenant}'"
        
        if "WHERE" in base_query.upper():
            return f"{base_query} AND {tenant_filter}"
        else:
            return f"{base_query} WHERE {tenant_filter}"
    
    def validate_tenant_data_access(self, data: Dict[str, Any]) -> bool:
        """Validate that data belongs to current tenant"""
        current_tenant = self.tenant_context.get_tenant()
        if not current_tenant:
            return False
        
        return data.get('tenant_id') == current_tenant
    
    def get_tenant_resource_limits(self, tenant_id: str) -> Dict[str, Any]:
        """Get resource limits for tenant based on plan"""
        tenant = self.get_tenant(tenant_id)
        if not tenant:
            return {}
        
        # Define limits based on plan
        limits = {
            "basic": {
                "max_users": 10,
                "max_storage_gb": 100,
                "max_api_calls_per_hour": 1000,
                "max_concurrent_queries": 5
            },
            "premium": {
                "max_users": 100,
                "max_storage_gb": 1000,
                "max_api_calls_per_hour": 10000,
                "max_concurrent_queries": 20
            },
            "enterprise": {
                "max_users": -1,  # unlimited
                "max_storage_gb": -1,
                "max_api_calls_per_hour": -1,
                "max_concurrent_queries": 100
            }
        }
        
        return limits.get(tenant.plan, limits["basic"])

# Example usage
security_manager = MultiTenantSecurityManager()

# Register tenants
tenant1 = Tenant(
    id="tenant_123",
    name="Acme Corp",
    plan="premium",
    settings={"region": "us-east-1"},
    created_at="2024-01-01",
    status="active"
)

tenant2 = Tenant(
    id="tenant_456", 
    name="Beta Inc",
    plan="basic",
    settings={"region": "eu-west-1"},
    created_at="2024-01-15",
    status="active"
)

security_manager.register_tenant(tenant1)
security_manager.register_tenant(tenant2)

# Add permissions
security_manager.add_tenant_permission("tenant_123", "alice", ["read_data", "write_data"])
security_manager.add_tenant_permission("tenant_456", "bob", ["read_data"])

# Use tenant context
with security_manager.tenant_context_manager("tenant_123") as tenant:
    print(f"Operating in tenant: {tenant.name}")
    
    # Check permissions
    can_write = security_manager.check_tenant_permission("alice", "write_data")
    print(f"Alice can write data: {can_write}")
    
    # Get filtered query
    base_query = "SELECT * FROM customers"
    filtered_query = security_manager.get_tenant_filtered_query(base_query, "customers")
    print(f"Filtered query: {filtered_query}")
    
    # Check resource limits
    limits = security_manager.get_tenant_resource_limits("tenant_123")
    print(f"Resource limits: {limits}")
```

### Audit Logging and Compliance

Comprehensive audit logging is essential for security monitoring, compliance, and forensic analysis.

#### Audit Log Requirements

**What to Log**:
```python
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, Optional
import json

@dataclass
class AuditEvent:
    timestamp: datetime
    event_type: str  # login, logout, data_access, permission_change
    user_id: str
    tenant_id: Optional[str]
    resource: str
    action: str
    result: str  # success, failure, denied
    ip_address: str
    user_agent: str
    session_id: str
    additional_data: Dict[str, Any]

class AuditLogger:
    """Comprehensive audit logging system"""
    
    def __init__(self):
        self.events: List[AuditEvent] = []
    
    def log_authentication(self, user_id: str, result: str, 
                          ip_address: str, user_agent: str, **kwargs):
        """Log authentication events"""
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="authentication",
            user_id=user_id,
            tenant_id=kwargs.get('tenant_id'),
            resource="auth_system",
            action="login" if result == "success" else "login_attempt",
            result=result,
            ip_address=ip_address,
            user_agent=user_agent,
            session_id=kwargs.get('session_id', ''),
            additional_data=kwargs
        )
        self._store_event(event)
    
    def log_data_access(self, user_id: str, tenant_id: str, resource: str,
                       action: str, result: str, **kwargs):
        """Log data access events"""
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="data_access",
            user_id=user_id,
            tenant_id=tenant_id,
            resource=resource,
            action=action,
            result=result,
            ip_address=kwargs.get('ip_address', ''),
            user_agent=kwargs.get('user_agent', ''),
            session_id=kwargs.get('session_id', ''),
            additional_data=kwargs
        )
        self._store_event(event)
    
    def log_permission_change(self, admin_user: str, target_user: str,
                             permission: str, action: str, **kwargs):
        """Log permission changes"""
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="permission_change",
            user_id=admin_user,
            tenant_id=kwargs.get('tenant_id'),
            resource=f"user:{target_user}",
            action=f"{action}_permission",
            result="success",
            ip_address=kwargs.get('ip_address', ''),
            user_agent=kwargs.get('user_agent', ''),
            session_id=kwargs.get('session_id', ''),
            additional_data={
                "target_user": target_user,
                "permission": permission,
                **kwargs
            }
        )
        self._store_event(event)
    
    def _store_event(self, event: AuditEvent):
        """Store audit event (in production, send to SIEM/log aggregation)"""
        self.events.append(event)
        
        # In production, send to:
        # - Elasticsearch/Splunk for analysis
        # - SIEM system for security monitoring
        # - Compliance database for regulatory reporting
        
        print(f"AUDIT: {event.timestamp} - {event.user_id} {event.action} {event.resource} -> {event.result}")
    
    def get_user_activity(self, user_id: str, start_date: datetime, 
                         end_date: datetime) -> List[AuditEvent]:
        """Get user activity for time period"""
        return [
            event for event in self.events
            if event.user_id == user_id and start_date <= event.timestamp <= end_date
        ]
    
    def get_failed_access_attempts(self, hours: int = 24) -> List[AuditEvent]:
        """Get failed access attempts in last N hours"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.events
            if event.timestamp >= cutoff and event.result in ["failure", "denied"]
        ]
    
    def generate_compliance_report(self, tenant_id: str, 
                                  start_date: datetime, 
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate compliance report for tenant"""
        tenant_events = [
            event for event in self.events
            if (event.tenant_id == tenant_id and 
                start_date <= event.timestamp <= end_date)
        ]
        
        return {
            "tenant_id": tenant_id,
            "period": f"{start_date} to {end_date}",
            "total_events": len(tenant_events),
            "event_types": {
                event_type: len([e for e in tenant_events if e.event_type == event_type])
                for event_type in set(e.event_type for e in tenant_events)
            },
            "unique_users": len(set(e.user_id for e in tenant_events)),
            "failed_attempts": len([e for e in tenant_events if e.result in ["failure", "denied"]]),
            "data_access_events": len([e for e in tenant_events if e.event_type == "data_access"])
        }

# Example usage
audit_logger = AuditLogger()

# Log authentication
audit_logger.log_authentication(
    user_id="alice",
    result="success",
    ip_address="192.168.1.100",
    user_agent="Mozilla/5.0...",
    tenant_id="tenant_123",
    session_id="sess_abc123"
)

# Log data access
audit_logger.log_data_access(
    user_id="alice",
    tenant_id="tenant_123",
    resource="customer_data",
    action="select",
    result="success",
    ip_address="192.168.1.100",
    rows_accessed=150,
    query="SELECT * FROM customers WHERE region = 'US'"
)

# Log permission change
audit_logger.log_permission_change(
    admin_user="admin",
    target_user="bob",
    permission="read_financial_data",
    action="grant",
    tenant_id="tenant_123"
)
```

---

## 💻 Hands-On Exercise

See `exercise.py` for hands-on practice with access control implementation.

**What you'll build**:
1. Implement hierarchical RBAC system with role inheritance
2. Create row-level security policies for multi-tenant data
3. Build attribute-based access control engine
4. Design multi-tenant security architecture
5. Implement comprehensive audit logging
6. Create access control monitoring and alerting

**Expected time**: 45 minutes

---

## 📚 Resources

- [NIST RBAC Standard](https://csrc.nist.gov/projects/role-based-access-control)
- [PostgreSQL Row Level Security](https://www.postgresql.org/docs/current/ddl-rowsecurity.html)
- [XACML ABAC Standard](http://docs.oasis-open.org/xacml/3.0/xacml-3.0-core-spec-os-en.html)
- [OWASP Access Control Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Access_Control_Cheat_Sheet.html)
- [Multi-Tenant Architecture Patterns](https://docs.microsoft.com/en-us/azure/architecture/guide/multitenant/overview)
- [Zero Trust Security Model](https://www.nist.gov/publications/zero-trust-architecture)

---

## 🎯 Key Takeaways

- **RBAC provides scalable permission management** through role-based organization
- **Row-level security enables fine-grained data access control** at the database level
- **ABAC offers the most flexible authorization** based on dynamic attributes
- **Multi-tenant architectures require careful isolation design** to prevent data leakage
- **Comprehensive audit logging is essential** for security monitoring and compliance
- **Access control must be implemented at multiple layers** for defense in depth
- **Regular access reviews and cleanup** prevent privilege creep and stale permissions
- **Performance considerations are critical** when implementing fine-grained access control

---

## 🔄 What's Next?

Tomorrow we'll explore **Apache Airflow Basics** where you'll learn to build and orchestrate data pipelines with workflow management. This builds on today's access control concepts by adding the orchestration layer that coordinates secure data processing workflows.

**Preview of Day 12**:
- Airflow architecture and core concepts
- DAG design patterns and best practices
- Task dependencies and scheduling
- Monitoring and alerting
- Integration with data systems
- Production deployment patterns
