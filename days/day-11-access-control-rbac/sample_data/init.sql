-- Day 11: Access Control Database Initialization
-- Creates tables and sample data for access control testing

-- Enable Row Level Security extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table with multi-tenant support
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    tenant_id UUID,
    department VARCHAR(100),
    region VARCHAR(50),
    clearance_level INTEGER DEFAULT 1,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP
);

-- Tenants table
CREATE TABLE IF NOT EXISTS tenants (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    plan VARCHAR(50) NOT NULL DEFAULT 'basic',
    status VARCHAR(50) NOT NULL DEFAULT 'active',
    settings JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Customer data with sensitivity levels
CREATE TABLE IF NOT EXISTS customer_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    customer_name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    region VARCHAR(50) NOT NULL,
    sensitivity_level VARCHAR(50) NOT NULL DEFAULT 'public',
    department VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Financial data with high security
CREATE TABLE IF NOT EXISTS financial_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    customer_id UUID NOT NULL,
    account_balance DECIMAL(15,2),
    credit_score INTEGER,
    annual_income DECIMAL(15,2),
    sensitivity_level VARCHAR(50) NOT NULL DEFAULT 'confidential',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id),
    FOREIGN KEY (customer_id) REFERENCES customer_data(id)
);

-- Employee data for hierarchical access testing
CREATE TABLE IF NOT EXISTS employee_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id UUID NOT NULL,
    employee_id INTEGER UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    department VARCHAR(100) NOT NULL,
    manager_id INTEGER,
    salary DECIMAL(10,2),
    hire_date DATE,
    sensitivity_level VARCHAR(50) NOT NULL DEFAULT 'internal',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (tenant_id) REFERENCES tenants(id)
);

-- Audit log table
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_id VARCHAR(255) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    tenant_id UUID,
    resource VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    result VARCHAR(50) NOT NULL,
    ip_address INET,
    user_agent TEXT,
    session_id VARCHAR(255),
    additional_data JSONB DEFAULT '{}'
);

-- Insert sample tenants
INSERT INTO tenants (id, name, plan, status, settings) VALUES
    ('550e8400-e29b-41d4-a716-446655440001', 'Acme Corp', 'premium', 'active', '{"region": "us-east-1", "features": ["advanced_analytics"]}'),
    ('550e8400-e29b-41d4-a716-446655440002', 'Beta Inc', 'basic', 'active', '{"region": "eu-west-1", "features": ["basic_analytics"]}'),
    ('550e8400-e29b-41d4-a716-446655440003', 'Gamma Ltd', 'enterprise', 'active', '{"region": "ap-south-1", "features": ["all_features"]}')
ON CONFLICT (id) DO NOTHING;

-- Insert sample users
INSERT INTO users (id, username, email, password_hash, tenant_id, department, region, clearance_level) VALUES
    ('660e8400-e29b-41d4-a716-446655440001', 'alice', 'alice@acme.com', '$2b$12$hash1', '550e8400-e29b-41d4-a716-446655440001', 'Analytics', 'US', 2),
    ('660e8400-e29b-41d4-a716-446655440002', 'bob', 'bob@acme.com', '$2b$12$hash2', '550e8400-e29b-41d4-a716-446655440001', 'Data Science', 'US', 3),
    ('660e8400-e29b-41d4-a716-446655440003', 'charlie', 'charlie@beta.com', '$2b$12$hash3', '550e8400-e29b-41d4-a716-446655440002', 'Engineering', 'EU', 2),
    ('660e8400-e29b-41d4-a716-446655440004', 'diana', 'diana@gamma.com', '$2b$12$hash4', '550e8400-e29b-41d4-a716-446655440003', 'Operations', 'AP', 4),
    ('660e8400-e29b-41d4-a716-446655440005', 'admin', 'admin@system.com', '$2b$12$hash5', NULL, 'IT', 'GLOBAL', 5)
ON CONFLICT (id) DO NOTHING;

-- Insert sample customer data
INSERT INTO customer_data (tenant_id, customer_name, email, phone, region, sensitivity_level, department) VALUES
    ('550e8400-e29b-41d4-a716-446655440001', 'John Doe', 'john@example.com', '+1-555-0101', 'US', 'public', 'Sales'),
    ('550e8400-e29b-41d4-a716-446655440001', 'Jane Smith', 'jane@example.com', '+1-555-0102', 'US', 'internal', 'Marketing'),
    ('550e8400-e29b-41d4-a716-446655440001', 'Bob Johnson', 'bob@example.com', '+1-555-0103', 'CA', 'confidential', 'Finance'),
    ('550e8400-e29b-41d4-a716-446655440002', 'Alice Brown', 'alice@example.eu', '+44-20-7946-0958', 'EU', 'public', 'Sales'),
    ('550e8400-e29b-41d4-a716-446655440002', 'Charlie Wilson', 'charlie@example.eu', '+44-20-7946-0959', 'EU', 'internal', 'Support'),
    ('550e8400-e29b-41d4-a716-446655440003', 'Diana Lee', 'diana@example.in', '+91-11-2345-6789', 'AP', 'public', 'Sales')
ON CONFLICT DO NOTHING;

-- Insert sample employee data
INSERT INTO employee_data (tenant_id, employee_id, name, department, manager_id, salary, hire_date, sensitivity_level) VALUES
    ('550e8400-e29b-41d4-a716-446655440001', 1001, 'Alice Manager', 'Analytics', NULL, 120000.00, '2020-01-15', 'confidential'),
    ('550e8400-e29b-41d4-a716-446655440001', 1002, 'Bob Analyst', 'Analytics', 1001, 85000.00, '2021-03-10', 'internal'),
    ('550e8400-e29b-41d4-a716-446655440001', 1003, 'Carol Engineer', 'Engineering', 1001, 95000.00, '2021-06-01', 'internal'),
    ('550e8400-e29b-41d4-a716-446655440002', 2001, 'David Director', 'Engineering', NULL, 150000.00, '2019-08-20', 'confidential'),
    ('550e8400-e29b-41d4-a716-446655440002', 2002, 'Eve Developer', 'Engineering', 2001, 90000.00, '2022-01-15', 'internal')
ON CONFLICT (employee_id) DO NOTHING;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_users_tenant_id ON users(tenant_id);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_customer_data_tenant_id ON customer_data(tenant_id);
CREATE INDEX IF NOT EXISTS idx_customer_data_region ON customer_data(region);
CREATE INDEX IF NOT EXISTS idx_customer_data_sensitivity ON customer_data(sensitivity_level);
CREATE INDEX IF NOT EXISTS idx_financial_data_tenant_id ON financial_data(tenant_id);
CREATE INDEX IF NOT EXISTS idx_employee_data_tenant_id ON employee_data(tenant_id);
CREATE INDEX IF NOT EXISTS idx_employee_data_department ON employee_data(department);
CREATE INDEX IF NOT EXISTS idx_audit_log_timestamp ON audit_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_log_user_id ON audit_log(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_tenant_id ON audit_log(tenant_id);

-- Grant permissions for application user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO access_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO access_user;