"""
Day 11: Access Control - RBAC, Row-Level Security - Solution
Production-ready access control systems
"""

from enum import Enum
from typing import Dict, List, Set, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
import threading
from contextlib import contextmanager
import re
import json
import hashlib
import uuid

# Solution 1 - Production Hierarchical RBAC System
class Permission(Enum):
    READ_CUSTOMER_DATA = "read_customer_data"
    WRITE_CUSTOMER_DATA = "write_customer_data"
    READ_FINANCIAL_DATA = "read_financial_data"
    WRITE_FINANCIAL_DATA = "write_financial_data"
    CREATE_MODELS = "create_models"
    DEPLOY_MODELS = "deploy_models"
    MANAGE_USERS = "manage_users"
    MANAGE_ROLES = "manage_roles"
    DELETE_DATA = "delete_data"
    ADMIN_ACCESS = "admin_access"

@dataclass
class Role:
    name: str
    permissions: Set[Permission]
    inherits_from: List[str] = field(default_factory=list)
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class User:
    username: str
    roles: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True

class RBACSystem:
    """Production hierarchical RBAC implementation"""
    
    def __init__(self):
        self.roles: Dict[str, Role] = {}
        self.users: Dict[str, User] = {}
        self.role_hierarchy: Dict[str, List[str]] = {}
        self._permission_cache: Dict[str, Set[Permission]] = {}
        self._cache_ttl: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=15)
    
    def create_role(self, name: str, permissions: Set[Permission], 
                   inherits_from: List[str] = None, description: str = ""):
        """Create role with inheritance and validation"""
        
        if name in self.roles:
            raise ValueError(f"Role '{name}' already exists")
        
        inherits_from = inherits_from or []
        
        # Validate parent roles exist
        for parent_role in inherits_from:
            if parent_role not in self.roles:
                raise ValueError(f"Parent role '{parent_role}' does not exist")
        
        # Check for circular inheritance
        if self._would_create_cycle(name, inherits_from):
            raise ValueError(f"Creating role '{name}' would create circular inheritance")
        
        # Create role
        role = Role(
            name=name,
            permissions=permissions,
            inherits_from=inherits_from,
            description=description
        )
        
        self.roles[name] = role
        self.role_hierarchy[name] = inherits_from
        
        # Clear permission cache
        self._clear_permission_cache()
        
        print(f"Created role '{name}' with {len(permissions)} permissions")
        if inherits_from:
            print(f"  Inherits from: {', '.join(inherits_from)}")
    
    def _would_create_cycle(self, new_role: str, parents: List[str]) -> bool:
        """Check if adding inheritance would create a cycle"""
        
        def has_path_to(from_role: str, to_role: str, visited: Set[str]) -> bool:
            if from_role == to_role:
                return True
            
            if from_role in visited:
                return False
            
            visited.add(from_role)
            
            for parent in self.role_hierarchy.get(from_role, []):
                if has_path_to(parent, to_role, visited):
                    return True
            
            return False
        
        # Check if any parent has a path back to new_role
        for parent in parents:
            if has_path_to(parent, new_role, set()):
                return True
        
        return False
    
    def assign_role_to_user(self, username: str, role_name: str):
        """Assign role to user with validation"""
        
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")
        
        if username not in self.users:
            self.users[username] = User(username=username)
        
        self.users[username].roles.add(role_name)
        
        # Clear user's permission cache
        self._clear_user_permission_cache(username)
        
        print(f"Assigned role '{role_name}' to user '{username}'")
    
    def remove_role_from_user(self, username: str, role_name: str):
        """Remove role from user"""
        
        if username not in self.users:
            raise ValueError(f"User '{username}' does not exist")
        
        if role_name not in self.users[username].roles:
            raise ValueError(f"User '{username}' does not have role '{role_name}'")
        
        self.users[username].roles.remove(role_name)
        self._clear_user_permission_cache(username)
        
        print(f"Removed role '{role_name}' from user '{username}'")
    
    def get_user_permissions(self, username: str) -> Set[Permission]:
        """Get all permissions for user with caching"""
        
        if username not in self.users:
            return set()
        
        # Check cache
        cache_key = f"user_permissions_{username}"
        if (cache_key in self._permission_cache and 
            cache_key in self._cache_ttl and
            datetime.now() < self._cache_ttl[cache_key]):
            return self._permission_cache[cache_key]
        
        # Calculate permissions
        all_permissions = set()
        user_roles = self.users[username].roles
        
        for role_name in user_roles:
            all_permissions.update(self._get_role_permissions(role_name))
        
        # Cache result
        self._permission_cache[cache_key] = all_permissions
        self._cache_ttl[cache_key] = datetime.now() + self.cache_duration
        
        return all_permissions
    
    def _get_role_permissions(self, role_name: str, visited: Set[str] = None) -> Set[Permission]:
        """Get all permissions for role including inherited"""
        
        if role_name not in self.roles:
            return set()
        
        if visited is None:
            visited = set()
        
        if role_name in visited:
            # Circular reference detected
            return set()
        
        visited.add(role_name)
        
        role = self.roles[role_name]
        permissions = role.permissions.copy()
        
        # Add inherited permissions
        for parent_role in role.inherits_from:
            permissions.update(self._get_role_permissions(parent_role, visited.copy()))
        
        return permissions
    
    def check_permission(self, username: str, permission: Permission) -> bool:
        """Check if user has specific permission"""
        user_permissions = self.get_user_permissions(username)
        return permission in user_permissions
    
    def get_users_with_permission(self, permission: Permission) -> List[str]:
        """Find all users with specific permission"""
        users_with_permission = []
        
        for username in self.users:
            if self.check_permission(username, permission):
                users_with_permission.append(username)
        
        return users_with_permission
    
    def delete_role(self, role_name: str):
        """Delete role and update all references"""
        
        if role_name not in self.roles:
            raise ValueError(f"Role '{role_name}' does not exist")
        
        # Remove role from all users
        for user in self.users.values():
            user.roles.discard(role_name)
        
        # Remove from role hierarchy
        del self.role_hierarchy[role_name]
        
        # Remove role from inheritance of other roles
        for role in self.roles.values():
            if role_name in role.inherits_from:
                role.inherits_from.remove(role_name)
        
        # Delete role
        del self.roles[role_name]
        
        # Clear cache
        self._clear_permission_cache()
        
        print(f"Deleted role '{role_name}'")
    
    def _clear_permission_cache(self):
        """Clear all permission caches"""
        self._permission_cache.clear()
        self._cache_ttl.clear()
    
    def _clear_user_permission_cache(self, username: str):
        """Clear permission cache for specific user"""
        cache_key = f"user_permissions_{username}"
        self._permission_cache.pop(cache_key, None)
        self._cache_ttl.pop(cache_key, None)
    
    def get_role_hierarchy_tree(self) -> Dict[str, Any]:
        """Get role hierarchy as tree structure"""
        
        def build_tree(role_name: str, visited: Set[str]) -> Dict[str, Any]:
            if role_name in visited:
                return {"name": role_name, "circular": True}
            
            visited.add(role_name)
            
            children = []
            for child_role, parents in self.role_hierarchy.items():
                if role_name in parents:
                    children.append(build_tree(child_role, visited.copy()))
            
            return {
                "name": role_name,
                "permissions": len(self._get_role_permissions(role_name)),
                "children": children
            }
        
        # Find root roles (no parents)
        root_roles = [
            role_name for role_name, parents in self.role_hierarchy.items()
            if not parents
        ]
        
        return {
            "roots": [build_tree(role, set()) for role in root_roles]
        }

# Solution 2 - Production Row-Level Security System
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
    condition: str
    priority: int = 0
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

class RowLevelSecurityManager:
    """Production row-level security implementation"""
    
    def __init__(self):
        self.policies: List[RowLevelPolicy] = []
        self.user_context: Dict[str, Dict[str, Any]] = {}
        self.policy_cache: Dict[str, List[RowLevelPolicy]] = {}
    
    def add_policy(self, policy: RowLevelPolicy):
        """Add RLS policy with validation"""
        
        # Validate policy
        self._validate_policy(policy)
        
        # Check for conflicts
        conflicts = self._check_policy_conflicts(policy)
        if conflicts:
            print(f"Warning: Policy conflicts detected with: {conflicts}")
        
        self.policies.append(policy)
        
        # Sort by priority (higher first)
        self.policies.sort(key=lambda p: p.priority, reverse=True)
        
        # Clear cache
        self.policy_cache.clear()
        
        print(f"Added RLS policy '{policy.name}' for {policy.table}.{policy.role}")
    
    def _validate_policy(self, policy: RowLevelPolicy):
        """Validate policy syntax and parameters"""
        
        if not policy.name or not policy.table or not policy.role:
            raise ValueError("Policy name, table, and role are required")
        
        if not policy.condition:
            raise ValueError("Policy condition cannot be empty")
        
        # Basic SQL injection protection
        dangerous_keywords = ['DROP', 'DELETE', 'INSERT', 'UPDATE', 'CREATE', 'ALTER']
        condition_upper = policy.condition.upper()
        
        for keyword in dangerous_keywords:
            if keyword in condition_upper:
                raise ValueError(f"Dangerous keyword '{keyword}' not allowed in policy condition")
    
    def _check_policy_conflicts(self, new_policy: RowLevelPolicy) -> List[str]:
        """Check for policy conflicts"""
        
        conflicts = []
        
        for existing_policy in self.policies:
            if (existing_policy.table == new_policy.table and
                existing_policy.role == new_policy.role and
                existing_policy.access_level == new_policy.access_level and
                existing_policy.is_active):
                conflicts.append(existing_policy.name)
        
        return conflicts
    
    def set_user_context(self, username: str, context: Dict[str, Any]):
        """Set context variables for user session"""
        
        # Validate context
        if not isinstance(context, dict):
            raise ValueError("Context must be a dictionary")
        
        self.user_context[username] = context.copy()
        print(f"Set context for user '{username}': {list(context.keys())}")
    
    def apply_rls_filter(self, username: str, table: str, 
                        access_level: AccessLevel, 
                        base_query: str) -> str:
        """Apply RLS policies to SQL query"""
        
        if username not in self.user_context:
            raise ValueError(f"No context set for user '{username}'")
        
        user_context = self.user_context[username]
        user_roles = user_context.get('roles', [])
        
        # Get applicable policies
        applicable_policies = self.get_applicable_policies(user_roles, table, access_level)
        
        if not applicable_policies:
            # No policies apply - allow access (or deny based on default policy)
            return base_query
        
        # Build conditions from policies
        conditions = []
        
        for policy in applicable_policies:
            try:
                condition = self._substitute_context(policy.condition, user_context)
                conditions.append(f"({condition})")
            except Exception as e:
                print(f"Error processing policy '{policy.name}': {e}")
                continue
        
        if not conditions:
            return base_query
        
        # Combine conditions with OR (user needs to satisfy any policy)
        combined_condition = " OR ".join(conditions)
        
        # Add to query
        if "WHERE" in base_query.upper():
            return f"{base_query} AND ({combined_condition})"
        else:
            return f"{base_query} WHERE ({combined_condition})"
    
    def _substitute_context(self, condition: str, context: Dict[str, Any]) -> str:
        """Safely substitute context variables in conditions"""
        
        result = condition
        
        # Find all context variable references ($variable)
        pattern = r'\$(\w+)'
        matches = re.findall(pattern, condition)
        
        for var_name in matches:
            if var_name in context:
                value = context[var_name]
                placeholder = f"${var_name}"
                
                if isinstance(value, str):
                    # Escape single quotes and wrap in quotes
                    escaped_value = value.replace("'", "''")
                    result = result.replace(placeholder, f"'{escaped_value}'")
                elif isinstance(value, list):
                    # Convert list to SQL IN clause format
                    if all(isinstance(v, str) for v in value):
                        value_list = "', '".join(v.replace("'", "''") for v in value)
                        result = result.replace(placeholder, f"('{value_list}')")
                    else:
                        value_list = ", ".join(str(v) for v in value)
                        result = result.replace(placeholder, f"({value_list})")
                else:
                    result = result.replace(placeholder, str(value))
            else:
                raise ValueError(f"Context variable '${var_name}' not found")
        
        return result
    
    def get_applicable_policies(self, user_roles: List[str], table: str, 
                              access_level: AccessLevel) -> List[RowLevelPolicy]:
        """Get policies applicable to this access request"""
        
        cache_key = f"{','.join(sorted(user_roles))}:{table}:{access_level.value}"
        
        if cache_key in self.policy_cache:
            return self.policy_cache[cache_key]
        
        applicable_policies = []
        
        for policy in self.policies:
            if (policy.table == table and
                policy.role in user_roles and
                policy.access_level == access_level and
                policy.is_active):
                applicable_policies.append(policy)
        
        # Cache result
        self.policy_cache[cache_key] = applicable_policies
        
        return applicable_policies
    
    def validate_data_access(self, username: str, table: str, 
                           access_level: AccessLevel, 
                           data_row: Dict[str, Any]) -> bool:
        """Validate if user can access specific data row"""
        
        if username not in self.user_context:
            return False
        
        user_context = self.user_context[username]
        user_roles = user_context.get('roles', [])
        
        applicable_policies = self.get_applicable_policies(user_roles, table, access_level)
        
        if not applicable_policies:
            return True  # No policies = allow (or change to False for deny-by-default)
        
        # Check if data row satisfies any policy
        for policy in applicable_policies:
            try:
                # Simple condition evaluation for data row
                if self._evaluate_condition_on_data(policy.condition, data_row, user_context):
                    return True
            except Exception as e:
                print(f"Error evaluating policy '{policy.name}' on data: {e}")
                continue
        
        return False
    
    def _evaluate_condition_on_data(self, condition: str, data_row: Dict[str, Any], 
                                   user_context: Dict[str, Any]) -> bool:
        """Evaluate policy condition against data row"""
        
        # Substitute context variables
        condition = self._substitute_context(condition, user_context)
        
        # Simple evaluation - in production, use proper SQL evaluation
        # This is a simplified version for demonstration
        
        # Handle common patterns
        if "=" in condition:
            parts = condition.split("=")
            if len(parts) == 2:
                field = parts[0].strip()
                value = parts[1].strip().strip("'\"")
                return str(data_row.get(field, "")) == value
        
        return True  # Default allow for complex conditions

# Solution 3 - Production ABAC Engine
@dataclass
class Subject:
    id: str
    attributes: Dict[str, Any]

@dataclass 
class Resource:
    id: str
    attributes: Dict[str, Any]

@dataclass
class Action:
    name: str
    attributes: Dict[str, Any]

@dataclass
class Environment:
    attributes: Dict[str, Any]

@dataclass
class ABACPolicy:
    id: str
    name: str
    description: str
    target: str
    condition: str
    effect: str  # "permit" or "deny"
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    is_active: bool = True

class ABACEngine:
    """Production Attribute-Based Access Control engine"""
    
    def __init__(self):
        self.policies: List[ABACPolicy] = []
        self.evaluation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        self.cache_duration = timedelta(minutes=5)
    
    def add_policy(self, policy: ABACPolicy):
        """Add ABAC policy with validation"""
        
        # Validate policy
        self._validate_policy(policy)
        
        self.policies.append(policy)
        
        # Sort by priority (higher first)
        self.policies.sort(key=lambda p: p.priority, reverse=True)
        
        # Clear cache
        self.evaluation_cache.clear()
        self.cache_ttl.clear()
        
        print(f"Added ABAC policy '{policy.name}' with priority {policy.priority}")
    
    def _validate_policy(self, policy: ABACPolicy):
        """Validate policy syntax"""
        
        if not policy.id or not policy.name:
            raise ValueError("Policy ID and name are required")
        
        if policy.effect not in ["permit", "deny"]:
            raise ValueError("Policy effect must be 'permit' or 'deny'")
        
        if not policy.target or not policy.condition:
            raise ValueError("Policy target and condition are required")
        
        # Check for existing policy with same ID
        if any(p.id == policy.id for p in self.policies):
            raise ValueError(f"Policy with ID '{policy.id}' already exists")
    
    def evaluate(self, subject: Subject, resource: Resource, 
                action: Action, environment: Environment) -> Dict[str, Any]:
        """Evaluate access request against policies"""
        
        # Create cache key
        cache_key = self._create_cache_key(subject, resource, action, environment)
        
        # Check cache
        if (cache_key in self.evaluation_cache and 
            cache_key in self.cache_ttl and
            datetime.now() < self.cache_ttl[cache_key]):
            return self.evaluation_cache[cache_key]
        
        # Build evaluation context
        context = self._build_context(subject, resource, action, environment)
        
        # Default deny
        decision = "deny"
        applicable_policies = []
        decision_reason = "No applicable policies found"
        
        # Evaluate policies in priority order
        for policy in self.policies:
            if not policy.is_active:
                continue
            
            try:
                # Check if policy applies
                if self._evaluate_expression(policy.target, context):
                    applicable_policies.append(policy.id)
                    
                    # Evaluate policy condition
                    if self._evaluate_expression(policy.condition, context):
                        decision = policy.effect
                        decision_reason = f"Policy '{policy.name}' ({policy.id}) applied"
                        
                        # First matching policy wins (due to priority sorting)
                        break
            except Exception as e:
                print(f"Error evaluating policy '{policy.name}': {e}")
                continue
        
        result = {
            "decision": decision,
            "reason": decision_reason,
            "applicable_policies": applicable_policies,
            "evaluation_time": datetime.now().isoformat(),
            "context": {
                "subject_id": subject.id,
                "resource_id": resource.id,
                "action": action.name
            }
        }
        
        # Cache result
        self.evaluation_cache[cache_key] = result
        self.cache_ttl[cache_key] = datetime.now() + self.cache_duration
        
        return result
    
    def _create_cache_key(self, subject: Subject, resource: Resource, 
                         action: Action, environment: Environment) -> str:
        """Create cache key for evaluation result"""
        
        key_data = {
            "subject": subject.id,
            "subject_attrs": sorted(subject.attributes.items()),
            "resource": resource.id,
            "resource_attrs": sorted(resource.attributes.items()),
            "action": action.name,
            "action_attrs": sorted(action.attributes.items()),
            "env_attrs": sorted(environment.attributes.items())
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _build_context(self, subject: Subject, resource: Resource, 
                      action: Action, environment: Environment) -> Dict[str, Any]:
        """Build evaluation context from request components"""
        
        context = {
            "subject": subject.attributes,
            "resource": resource.attributes,
            "action": action.attributes,
            "environment": environment.attributes
        }
        
        # Add helper functions
        context.update({
            "current_time": datetime.now().time(),
            "current_date": datetime.now().date(),
            "current_hour": datetime.now().hour,
            "current_weekday": datetime.now().weekday(),
            "is_business_hours": self._is_business_hours(),
            "is_weekend": datetime.now().weekday() >= 5
        })
        
        return context
    
    def _evaluate_expression(self, expression: str, context: Dict[str, Any]) -> bool:
        """Safely evaluate policy expressions"""
        
        try:
            # Create safe evaluation environment
            safe_dict = {
                "__builtins__": {},
                # Add safe functions
                "len": len,
                "str": str,
                "int": int,
                "float": float,
                "bool": bool,
                "min": min,
                "max": max,
                "abs": abs,
                "round": round
            }
            
            # Add context
            safe_dict.update(context)
            
            # Evaluate expression
            result = eval(expression, safe_dict)
            return bool(result)
            
        except Exception as e:
            print(f"Error evaluating expression '{expression}': {e}")
            return False
    
    def _is_business_hours(self) -> bool:
        """Check if current time is business hours"""
        current_time = datetime.now().time()
        return time(9, 0) <= current_time <= time(17, 0)
    
    def get_policy_coverage(self) -> Dict[str, Any]:
        """Analyze policy coverage and gaps"""
        
        coverage = {
            "total_policies": len(self.policies),
            "active_policies": len([p for p in self.policies if p.is_active]),
            "permit_policies": len([p for p in self.policies if p.effect == "permit"]),
            "deny_policies": len([p for p in self.policies if p.effect == "deny"]),
            "priority_distribution": {}
        }
        
        # Analyze priority distribution
        for policy in self.policies:
            priority_range = f"{policy.priority//100*100}-{policy.priority//100*100+99}"
            coverage["priority_distribution"][priority_range] = \
                coverage["priority_distribution"].get(priority_range, 0) + 1
        
        return coverage

# Solution 4 - Production Multi-Tenant Security
@dataclass
class Tenant:
    id: str
    name: str
    plan: str
    settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"

class TenantContext:
    """Thread-local tenant context management"""
    
    def __init__(self):
        self._local = threading.local()
    
    def set_tenant(self, tenant_id: str):
        """Set current tenant for thread"""
        self._local.tenant_id = tenant_id
    
    def get_tenant(self) -> Optional[str]:
        """Get current tenant for thread"""
        return getattr(self._local, 'tenant_id', None)
    
    def clear_tenant(self):
        """Clear tenant context"""
        if hasattr(self._local, 'tenant_id'):
            delattr(self._local, 'tenant_id')

class MultiTenantSecurityManager:
    """Production multi-tenant security manager"""
    
    def __init__(self):
        self.tenants: Dict[str, Tenant] = {}
        self.tenant_context = TenantContext()
        self.tenant_permissions: Dict[str, Dict[str, List[str]]] = {}
        self.tenant_policies: Dict[str, List[RowLevelPolicy]] = {}
        self.resource_usage: Dict[str, Dict[str, Any]] = {}
    
    def register_tenant(self, tenant: Tenant):
        """Register new tenant with security setup"""
        
        if tenant.id in self.tenants:
            raise ValueError(f"Tenant '{tenant.id}' already exists")
        
        # Validate tenant configuration
        self._validate_tenant_config(tenant)
        
        # Register tenant
        self.tenants[tenant.id] = tenant
        self.tenant_permissions[tenant.id] = {}
        self.tenant_policies[tenant.id] = []
        self.resource_usage[tenant.id] = {
            "users": 0,
            "storage_gb": 0,
            "api_calls_today": 0,
            "concurrent_queries": 0
        }
        
        # Set up default security policies
        self._setup_default_policies(tenant.id)
        
        print(f"Registered tenant '{tenant.name}' ({tenant.id}) with plan '{tenant.plan}'")
    
    def _validate_tenant_config(self, tenant: Tenant):
        """Validate tenant configuration"""
        
        if not tenant.id or not tenant.name:
            raise ValueError("Tenant ID and name are required")
        
        valid_plans = ["basic", "premium", "enterprise"]
        if tenant.plan not in valid_plans:
            raise ValueError(f"Invalid plan '{tenant.plan}'. Must be one of: {valid_plans}")
        
        valid_statuses = ["active", "suspended", "deleted"]
        if tenant.status not in valid_statuses:
            raise ValueError(f"Invalid status '{tenant.status}'. Must be one of: {valid_statuses}")
    
    def _setup_default_policies(self, tenant_id: str):
        """Set up default security policies for tenant"""
        
        # Default tenant isolation policy
        isolation_policy = RowLevelPolicy(
            name=f"tenant_isolation_{tenant_id}",
            table="*",  # Apply to all tables
            role="*",   # Apply to all roles
            access_level=AccessLevel.READ,
            condition=f"tenant_id = '{tenant_id}'",
            priority=1000,
            description="Default tenant isolation policy"
        )
        
        self.tenant_policies[tenant_id].append(isolation_policy)
    
    @contextmanager
    def tenant_context_manager(self, tenant_id: str):
        """Context manager for tenant operations"""
        
        if tenant_id not in self.tenants:
            raise ValueError(f"Unknown tenant: {tenant_id}")
        
        tenant = self.tenants[tenant_id]
        
        if tenant.status != 'active':
            raise ValueError(f"Tenant {tenant_id} is not active (status: {tenant.status})")
        
        # Set tenant context
        previous_tenant = self.tenant_context.get_tenant()
        self.tenant_context.set_tenant(tenant_id)
        
        try:
            yield tenant
        finally:
            # Restore previous context
            if previous_tenant:
                self.tenant_context.set_tenant(previous_tenant)
            else:
                self.tenant_context.clear_tenant()
    
    def add_tenant_permission(self, tenant_id: str, user_id: str, permissions: List[str]):
        """Add permissions for user within tenant"""
        
        if tenant_id not in self.tenants:
            raise ValueError(f"Unknown tenant: {tenant_id}")
        
        if tenant_id not in self.tenant_permissions:
            self.tenant_permissions[tenant_id] = {}
        
        self.tenant_permissions[tenant_id][user_id] = permissions
        print(f"Added {len(permissions)} permissions for user '{user_id}' in tenant '{tenant_id}'")
    
    def check_tenant_permission(self, user_id: str, permission: str) -> bool:
        """Check user permission within current tenant context"""
        
        current_tenant = self.tenant_context.get_tenant()
        if not current_tenant:
            return False
        
        user_permissions = self.tenant_permissions.get(current_tenant, {}).get(user_id, [])
        return permission in user_permissions
    
    def get_tenant_filtered_query(self, base_query: str, table_name: str) -> str:
        """Add tenant isolation to SQL queries"""
        
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
        """Get resource limits based on tenant plan"""
        
        tenant = self.tenants.get(tenant_id)
        if not tenant:
            return {}
        
        limits = {
            "basic": {
                "max_users": 10,
                "max_storage_gb": 100,
                "max_api_calls_per_hour": 1000,
                "max_concurrent_queries": 5,
                "features": ["basic_analytics", "standard_support"]
            },
            "premium": {
                "max_users": 100,
                "max_storage_gb": 1000,
                "max_api_calls_per_hour": 10000,
                "max_concurrent_queries": 20,
                "features": ["advanced_analytics", "priority_support", "custom_reports"]
            },
            "enterprise": {
                "max_users": -1,  # unlimited
                "max_storage_gb": -1,
                "max_api_calls_per_hour": -1,
                "max_concurrent_queries": 100,
                "features": ["all_features", "dedicated_support", "custom_integrations", "sla_guarantee"]
            }
        }
        
        return limits.get(tenant.plan, limits["basic"])
    
    def check_resource_limits(self, tenant_id: str, resource_type: str, 
                            requested_amount: int = 1) -> Dict[str, Any]:
        """Check if resource usage would exceed limits"""
        
        limits = self.get_tenant_resource_limits(tenant_id)
        current_usage = self.resource_usage.get(tenant_id, {})
        
        limit_key = f"max_{resource_type}"
        usage_key = resource_type
        
        max_allowed = limits.get(limit_key, 0)
        current_used = current_usage.get(usage_key, 0)
        
        if max_allowed == -1:  # Unlimited
            return {"allowed": True, "reason": "unlimited"}
        
        if current_used + requested_amount > max_allowed:
            return {
                "allowed": False,
                "reason": f"Would exceed limit: {current_used + requested_amount} > {max_allowed}",
                "current_usage": current_used,
                "limit": max_allowed
            }
        
        return {
            "allowed": True,
            "reason": "within limits",
            "current_usage": current_used,
            "limit": max_allowed
        }
    
    def update_resource_usage(self, tenant_id: str, resource_type: str, amount: int):
        """Update resource usage for tenant"""
        
        if tenant_id not in self.resource_usage:
            self.resource_usage[tenant_id] = {}
        
        current = self.resource_usage[tenant_id].get(resource_type, 0)
        self.resource_usage[tenant_id][resource_type] = max(0, current + amount)

# Solution 5 - Production Audit Logging System
@dataclass
class AuditEvent:
    timestamp: datetime
    event_type: str
    user_id: str
    tenant_id: Optional[str]
    resource: str
    action: str
    result: str
    ip_address: str
    user_agent: str
    session_id: str
    additional_data: Dict[str, Any] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

class AuditLogger:
    """Production audit logging system"""
    
    def __init__(self):
        self.events: List[AuditEvent] = []
        self.event_handlers: Dict[str, List[Callable]] = {}
        self.suspicious_patterns: List[Dict[str, Any]] = []
        self._setup_suspicious_patterns()
    
    def _setup_suspicious_patterns(self):
        """Set up patterns for suspicious activity detection"""
        
        self.suspicious_patterns = [
            {
                "name": "multiple_failed_logins",
                "description": "Multiple failed login attempts",
                "condition": lambda events: len([
                    e for e in events[-50:] 
                    if e.event_type == "authentication" and e.result == "failure"
                ]) >= 5,
                "severity": "high"
            },
            {
                "name": "off_hours_access",
                "description": "Data access outside business hours",
                "condition": lambda events: any(
                    e.event_type == "data_access" and 
                    (e.timestamp.hour < 9 or e.timestamp.hour > 17)
                    for e in events[-10:]
                ),
                "severity": "medium"
            },
            {
                "name": "bulk_data_access",
                "description": "Large volume data access",
                "condition": lambda events: any(
                    e.event_type == "data_access" and 
                    e.additional_data.get("rows_accessed", 0) > 10000
                    for e in events[-10:]
                ),
                "severity": "high"
            },
            {
                "name": "privilege_escalation",
                "description": "Permission changes followed by data access",
                "condition": lambda events: any(
                    e.event_type == "permission_change" and
                    any(e2.event_type == "data_access" and 
                        e2.timestamp > e.timestamp and
                        e2.timestamp < e.timestamp + timedelta(minutes=30)
                        for e2 in events)
                    for e in events[-20:]
                ),
                "severity": "critical"
            }
        ]
    
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
            additional_data={
                "auth_method": kwargs.get('auth_method', 'password'),
                "mfa_used": kwargs.get('mfa_used', False),
                "failure_reason": kwargs.get('failure_reason', ''),
                **{k: v for k, v in kwargs.items() if k not in ['tenant_id', 'session_id']}
            }
        )
        
        self._store_event(event)
        
        # Check for suspicious activity
        if result == "failure":
            self._check_suspicious_activity()
    
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
            additional_data={
                "query": kwargs.get('query', ''),
                "rows_accessed": kwargs.get('rows_accessed', 0),
                "execution_time_ms": kwargs.get('execution_time_ms', 0),
                "data_classification": kwargs.get('data_classification', ''),
                **{k: v for k, v in kwargs.items() 
                   if k not in ['ip_address', 'user_agent', 'session_id']}
            }
        )
        
        self._store_event(event)
        
        # Check for bulk access
        if kwargs.get('rows_accessed', 0) > 10000:
            self._check_suspicious_activity()
    
    def log_permission_change(self, admin_user: str, target_user: str,
                             permission: str, action: str, **kwargs):
        """Log permission and role changes"""
        
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
                "previous_permissions": kwargs.get('previous_permissions', []),
                "new_permissions": kwargs.get('new_permissions', []),
                **{k: v for k, v in kwargs.items() 
                   if k not in ['tenant_id', 'ip_address', 'user_agent', 'session_id']}
            }
        )
        
        self._store_event(event)
    
    def log_policy_change(self, admin_user: str, policy_id: str,
                         change_type: str, **kwargs):
        """Log security policy changes"""
        
        event = AuditEvent(
            timestamp=datetime.now(),
            event_type="policy_change",
            user_id=admin_user,
            tenant_id=kwargs.get('tenant_id'),
            resource=f"policy:{policy_id}",
            action=change_type,
            result="success",
            ip_address=kwargs.get('ip_address', ''),
            user_agent=kwargs.get('user_agent', ''),
            session_id=kwargs.get('session_id', ''),
            additional_data={
                "policy_id": policy_id,
                "policy_type": kwargs.get('policy_type', ''),
                "previous_config": kwargs.get('previous_config', {}),
                "new_config": kwargs.get('new_config', {}),
                **{k: v for k, v in kwargs.items() 
                   if k not in ['tenant_id', 'ip_address', 'user_agent', 'session_id']}
            }
        )
        
        self._store_event(event)
    
    def _store_event(self, event: AuditEvent):
        """Store audit event with indexing"""
        
        self.events.append(event)
        
        # In production, send to:
        # - SIEM system (Splunk, ELK stack)
        # - Database with proper indexing
        # - Log aggregation service
        # - Real-time alerting system
        
        # Trigger event handlers
        for handler in self.event_handlers.get(event.event_type, []):
            try:
                handler(event)
            except Exception as e:
                print(f"Error in event handler: {e}")
        
        # Keep only recent events in memory (for demo)
        if len(self.events) > 10000:
            self.events = self.events[-5000:]
        
        print(f"AUDIT: {event.timestamp.strftime('%H:%M:%S')} - {event.user_id} {event.action} {event.resource} -> {event.result}")
    
    def get_user_activity(self, user_id: str, start_date: datetime, 
                         end_date: datetime) -> List[AuditEvent]:
        """Get user activity for time period"""
        
        return [
            event for event in self.events
            if (event.user_id == user_id and 
                start_date <= event.timestamp <= end_date)
        ]
    
    def get_failed_access_attempts(self, hours: int = 24) -> List[AuditEvent]:
        """Get failed access attempts for security monitoring"""
        
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            event for event in self.events
            if (event.timestamp >= cutoff and 
                event.result in ["failure", "denied"])
        ]
    
    def detect_suspicious_activity(self) -> List[Dict[str, Any]]:
        """Detect suspicious access patterns"""
        
        alerts = []
        
        for pattern in self.suspicious_patterns:
            try:
                if pattern["condition"](self.events):
                    alerts.append({
                        "pattern": pattern["name"],
                        "description": pattern["description"],
                        "severity": pattern["severity"],
                        "detected_at": datetime.now().isoformat(),
                        "recent_events": len(self.events[-50:])
                    })
            except Exception as e:
                print(f"Error checking pattern '{pattern['name']}': {e}")
        
        return alerts
    
    def _check_suspicious_activity(self):
        """Check for suspicious activity and alert"""
        
        alerts = self.detect_suspicious_activity()
        
        for alert in alerts:
            if alert["severity"] in ["high", "critical"]:
                print(f"SECURITY ALERT: {alert['description']} (Severity: {alert['severity']})")
    
    def generate_compliance_report(self, tenant_id: str, 
                                  start_date: datetime, 
                                  end_date: datetime) -> Dict[str, Any]:
        """Generate compliance audit report"""
        
        tenant_events = [
            event for event in self.events
            if (event.tenant_id == tenant_id and 
                start_date <= event.timestamp <= end_date)
        ]
        
        # Analyze events
        event_types = {}
        users = set()
        failed_attempts = 0
        data_access_events = 0
        
        for event in tenant_events:
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            users.add(event.user_id)
            
            if event.result in ["failure", "denied"]:
                failed_attempts += 1
            
            if event.event_type == "data_access":
                data_access_events += 1
        
        return {
            "tenant_id": tenant_id,
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "summary": {
                "total_events": len(tenant_events),
                "unique_users": len(users),
                "failed_attempts": failed_attempts,
                "data_access_events": data_access_events,
                "success_rate": (len(tenant_events) - failed_attempts) / max(len(tenant_events), 1)
            },
            "event_breakdown": event_types,
            "top_users": self._get_top_users(tenant_events),
            "security_incidents": self.detect_suspicious_activity()
        }
    
    def _get_top_users(self, events: List[AuditEvent]) -> List[Dict[str, Any]]:
        """Get top users by activity"""
        
        user_activity = {}
        
        for event in events:
            if event.user_id not in user_activity:
                user_activity[event.user_id] = {
                    "user_id": event.user_id,
                    "total_events": 0,
                    "data_access_events": 0,
                    "failed_attempts": 0
                }
            
            user_activity[event.user_id]["total_events"] += 1
            
            if event.event_type == "data_access":
                user_activity[event.user_id]["data_access_events"] += 1
            
            if event.result in ["failure", "denied"]:
                user_activity[event.user_id]["failed_attempts"] += 1
        
        # Sort by total events
        return sorted(
            user_activity.values(),
            key=lambda x: x["total_events"],
            reverse=True
        )[:10]

def main():
    """Demonstrate all access control solutions"""
    
    print("=== Day 11: Access Control - RBAC, Row-Level Security - Solutions ===\n")
    
    # Solution 1: Hierarchical RBAC System
    print("=== Solution 1: Hierarchical RBAC System ===")
    rbac = RBACSystem()
    
    # Create roles with inheritance
    rbac.create_role("data_analyst", {
        Permission.READ_CUSTOMER_DATA,
        Permission.READ_FINANCIAL_DATA
    }, description="Basic data analysis role")
    
    rbac.create_role("data_scientist", {
        Permission.CREATE_MODELS
    }, inherits_from=["data_analyst"], description="Data science role")
    
    rbac.create_role("senior_data_scientist", {
        Permission.DEPLOY_MODELS
    }, inherits_from=["data_scientist"], description="Senior data science role")
    
    rbac.create_role("data_engineer", {
        Permission.WRITE_CUSTOMER_DATA,
        Permission.DELETE_DATA
    }, inherits_from=["data_scientist"], description="Data engineering role")
    
    rbac.create_role("admin", {
        Permission.MANAGE_USERS,
        Permission.MANAGE_ROLES,
        Permission.ADMIN_ACCESS
    }, description="Administrative role")
    
    # Assign roles to users
    rbac.assign_role_to_user("alice", "data_analyst")
    rbac.assign_role_to_user("bob", "data_scientist")
    rbac.assign_role_to_user("charlie", "senior_data_scientist")
    rbac.assign_role_to_user("diana", "data_engineer")
    rbac.assign_role_to_user("admin", "admin")
    
    # Test permissions
    print(f"\nPermission Tests:")
    print(f"Alice can read customer data: {rbac.check_permission('alice', Permission.READ_CUSTOMER_DATA)}")
    print(f"Bob can create models: {rbac.check_permission('bob', Permission.CREATE_MODELS)}")
    print(f"Bob can read customer data (inherited): {rbac.check_permission('bob', Permission.READ_CUSTOMER_DATA)}")
    print(f"Charlie can deploy models: {rbac.check_permission('charlie', Permission.DEPLOY_MODELS)}")
    print(f"Diana can write customer data: {rbac.check_permission('diana', Permission.WRITE_CUSTOMER_DATA)}")
    print(f"Diana can create models (inherited): {rbac.check_permission('diana', Permission.CREATE_MODELS)}")
    
    # Show role hierarchy
    hierarchy = rbac.get_role_hierarchy_tree()
    print(f"\nRole hierarchy: {len(hierarchy['roots'])} root roles")
    
    # Solution 2: Row-Level Security
    print("\n=== Solution 2: Row-Level Security ===")
    rls_manager = RowLevelSecurityManager()
    
    # Add RLS policies
    rls_manager.add_policy(RowLevelPolicy(
        name="regional_access",
        table="customer_data",
        role="data_analyst",
        access_level=AccessLevel.READ,
        condition="region IN $user_regions",
        priority=100,
        description="Analysts can only see data from their regions"
    ))
    
    rls_manager.add_policy(RowLevelPolicy(
        name="sensitivity_filter",
        table="customer_data",
        role="data_analyst", 
        access_level=AccessLevel.READ,
        condition="sensitivity_level != 'confidential'",
        priority=200,
        description="Analysts cannot see confidential data"
    ))
    
    rls_manager.add_policy(RowLevelPolicy(
        name="manager_access",
        table="employee_data",
        role="manager",
        access_level=AccessLevel.READ,
        condition="department = '$user_department' OR employee_id = $user_id",
        priority=150,
        description="Managers see their department data"
    ))
    
    # Set user contexts
    rls_manager.set_user_context("alice", {
        "roles": ["data_analyst"],
        "user_regions": ["US", "CA"],
        "user_id": 123,
        "user_department": "Engineering"
    })
    
    rls_manager.set_user_context("manager1", {
        "roles": ["manager"],
        "user_department": "Sales",
        "user_id": 456
    })
    
    # Test RLS filtering
    base_query = "SELECT * FROM customer_data"
    filtered_query = rls_manager.apply_rls_filter(
        "alice", "customer_data", AccessLevel.READ, base_query
    )
    
    print(f"Original query: {base_query}")
    print(f"Filtered query: {filtered_query}")
    
    # Test data validation
    test_data = {"tenant_id": "tenant_123", "region": "US", "sensitivity_level": "public"}
    can_access = rls_manager.validate_data_access(
        "alice", "customer_data", AccessLevel.READ, test_data
    )
    print(f"Alice can access test data: {can_access}")
    
    # Solution 3: ABAC Engine
    print("\n=== Solution 3: Attribute-Based Access Control ===")
    abac_engine = ABACEngine()
    
    # Add ABAC policies
    abac_engine.add_policy(ABACPolicy(
        id="business_hours_policy",
        name="Business Hours Access",
        description="Allow data access only during business hours",
        target="subject['role'] == 'data_analyst' and resource['type'] == 'customer_data'",
        condition="is_business_hours and not is_weekend",
        effect="permit",
        priority=100
    ))
    
    abac_engine.add_policy(ABACPolicy(
        id="high_clearance_policy",
        name="High Clearance Required",
        description="High sensitivity data requires high clearance",
        target="resource['sensitivity'] == 'high'",
        condition="subject['clearance_level'] >= 3",
        effect="permit",
        priority=200
    ))
    
    abac_engine.add_policy(ABACPolicy(
        id="untrusted_network_deny",
        name="Untrusted Network Deny",
        description="Deny access from untrusted networks",
        target="True",
        condition="environment['network_trust'] == 'untrusted'",
        effect="deny",
        priority=1000
    ))
    
    # Test ABAC evaluation
    subject = Subject("alice", {
        "role": "data_analyst",
        "department": "marketing",
        "clearance_level": 2
    })
    
    resource = Resource("customer_profiles", {
        "type": "customer_data",
        "sensitivity": "medium",
        "classification": "internal"
    })
    
    action = Action("read", {
        "operation_type": "query"
    })
    
    environment = Environment({
        "network_trust": "trusted",
        "location": "office"
    })
    
    result = abac_engine.evaluate(subject, resource, action, environment)
    print(f"ABAC Decision: {result['decision']}")
    print(f"Reason: {result['reason']}")
    print(f"Applicable policies: {result['applicable_policies']}")
    
    # Solution 4: Multi-Tenant Security
    print("\n=== Solution 4: Multi-Tenant Security ===")
    mt_security = MultiTenantSecurityManager()
    
    # Register tenants
    tenant1 = Tenant(
        id="tenant_123",
        name="Acme Corp",
        plan="premium",
        settings={"region": "us-east-1"}
    )
    
    tenant2 = Tenant(
        id="tenant_456",
        name="Beta Inc", 
        plan="basic",
        settings={"region": "eu-west-1"}
    )
    
    mt_security.register_tenant(tenant1)
    mt_security.register_tenant(tenant2)
    
    # Add tenant permissions
    mt_security.add_tenant_permission("tenant_123", "alice", ["read_data", "write_data"])
    mt_security.add_tenant_permission("tenant_456", "bob", ["read_data"])
    
    # Test tenant context
    with mt_security.tenant_context_manager("tenant_123") as tenant:
        print(f"Operating in tenant: {tenant.name}")
        
        # Check permissions
        can_write = mt_security.check_tenant_permission("alice", "write_data")
        print(f"Alice can write data: {can_write}")
        
        # Get filtered query
        base_query = "SELECT * FROM customers"
        filtered_query = mt_security.get_tenant_filtered_query(base_query, "customers")
        print(f"Tenant-filtered query: {filtered_query}")
        
        # Check resource limits
        limits = mt_security.get_tenant_resource_limits("tenant_123")
        print(f"Resource limits: max_users={limits['max_users']}, max_storage_gb={limits['max_storage_gb']}")
        
        # Test resource limit checking
        limit_check = mt_security.check_resource_limits("tenant_123", "users", 5)
        print(f"Can add 5 users: {limit_check['allowed']}")
    
    # Solution 5: Comprehensive Audit Logging
    print("\n=== Solution 5: Comprehensive Audit Logging ===")
    audit_logger = AuditLogger()
    
    # Log various events
    audit_logger.log_authentication(
        user_id="alice",
        result="success",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0...",
        tenant_id="tenant_123",
        session_id="sess_abc123",
        auth_method="password",
        mfa_used=True
    )
    
    audit_logger.log_data_access(
        user_id="alice",
        tenant_id="tenant_123",
        resource="customer_data",
        action="select",
        result="success",
        ip_address="192.168.1.100",
        rows_accessed=150,
        query="SELECT * FROM customers WHERE region = 'US'",
        execution_time_ms=250
    )
    
    audit_logger.log_permission_change(
        admin_user="admin",
        target_user="bob",
        permission="read_financial_data",
        action="grant",
        tenant_id="tenant_123",
        previous_permissions=["read_customer_data"],
        new_permissions=["read_customer_data", "read_financial_data"]
    )
    
    # Simulate suspicious activity
    for i in range(6):
        audit_logger.log_authentication(
            user_id="suspicious_user",
            result="failure",
            ip_address="10.0.0.1",
            user_agent="curl/7.68.0",
            failure_reason="invalid_password"
        )
    
    # Check for suspicious activity
    alerts = audit_logger.detect_suspicious_activity()
    print(f"Security alerts detected: {len(alerts)}")
    for alert in alerts:
        print(f"  - {alert['description']} (Severity: {alert['severity']})")
    
    # Generate compliance report
    start_date = datetime.now() - timedelta(days=1)
    end_date = datetime.now()
    
    report = audit_logger.generate_compliance_report("tenant_123", start_date, end_date)
    print(f"\nCompliance Report for tenant_123:")
    print(f"  Total events: {report['summary']['total_events']}")
    print(f"  Unique users: {report['summary']['unique_users']}")
    print(f"  Success rate: {report['summary']['success_rate']:.2%}")
    print(f"  Failed attempts: {report['summary']['failed_attempts']}")
    
    print("\n=== All Solutions Demonstrated Successfully ===")
    print("Production-ready access control systems implemented!")
    print("Ready for integration with real databases and identity providers.")

if __name__ == "__main__":
    main()
