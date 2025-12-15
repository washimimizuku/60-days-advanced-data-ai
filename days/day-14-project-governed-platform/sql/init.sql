-- DataCorp Governed Platform - Database Initialization
-- Creates schemas, tables, and initial data for the platform

-- Create schemas
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS marts;
CREATE SCHEMA IF NOT EXISTS compliance;

-- Create raw data table
CREATE TABLE IF NOT EXISTS raw_customers (
    customer_id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    phone VARCHAR(20),
    city VARCHAR(100),
    state VARCHAR(50),
    country VARCHAR(50) DEFAULT 'USA',
    consent_status VARCHAR(20) DEFAULT 'granted',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create audit log table
CREATE TABLE IF NOT EXISTS audit_logs (
    log_id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    record_id INTEGER,
    user_id VARCHAR(100),
    event_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    event_details JSONB,
    ip_address INET,
    user_agent TEXT
);

-- Create data quality metrics table
CREATE TABLE IF NOT EXISTS data_quality_metrics (
    metric_id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value NUMERIC,
    threshold_value NUMERIC,
    status VARCHAR(20),
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO raw_customers (email, first_name, last_name, phone, city, state, country, consent_status, created_at)
VALUES 
    ('john.doe@datacorp.com', 'John', 'Doe', '555-123-4567', 'New York', 'NY', 'USA', 'granted', '2024-01-15 10:30:00'),
    ('jane.smith@datacorp.com', 'Jane', 'Smith', '555-234-5678', 'Los Angeles', 'CA', 'USA', 'granted', '2024-01-16 14:22:00'),
    ('bob.johnson@datacorp.com', 'Bob', 'Johnson', '555-345-6789', 'Chicago', 'IL', 'USA', 'granted', '2024-01-17 09:15:00'),
    ('alice.brown@datacorp.com', 'Alice', 'Brown', '555-456-7890', 'Houston', 'TX', 'USA', 'granted', '2024-01-18 16:45:00'),
    ('charlie.davis@datacorp.com', 'Charlie', 'Davis', '555-567-8901', 'Phoenix', 'AZ', 'USA', 'granted', '2024-01-19 11:20:00')
ON CONFLICT (customer_id) DO NOTHING;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_customers_email ON raw_customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_created_at ON raw_customers(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(event_timestamp);
CREATE INDEX IF NOT EXISTS idx_audit_logs_table ON audit_logs(table_name);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO platform_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA staging TO platform_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA marts TO platform_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA compliance TO platform_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO platform_user;