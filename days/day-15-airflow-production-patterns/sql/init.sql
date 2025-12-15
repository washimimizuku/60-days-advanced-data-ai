-- TechCorp Data Platform - Database Initialization
-- Creates schemas, tables, and sample data for testing Airflow production patterns

-- Create schemas
CREATE SCHEMA IF NOT EXISTS techcorp_raw;
CREATE SCHEMA IF NOT EXISTS techcorp_staging;
CREATE SCHEMA IF NOT EXISTS techcorp_marts;

-- Create raw data tables
CREATE TABLE IF NOT EXISTS techcorp_raw.customers (
    customer_id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone VARCHAR(20),
    address TEXT,
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS techcorp_raw.transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES techcorp_raw.customers(customer_id),
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) DEFAULT 'USD',
    transaction_type VARCHAR(50),
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS techcorp_raw.products (
    product_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    price DECIMAL(10,2),
    cost DECIMAL(10,2),
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS techcorp_raw.user_events (
    event_id VARCHAR(50) PRIMARY KEY,
    user_id INTEGER,
    event_type VARCHAR(50),
    timestamp TIMESTAMP,
    page_url TEXT,
    user_agent TEXT,
    ip_address INET,
    session_id UUID,
    properties JSONB
);

CREATE TABLE IF NOT EXISTS techcorp_raw.partner_integrations (
    id SERIAL PRIMARY KEY,
    partner_id VARCHAR(50),
    integration_type VARCHAR(50),
    transaction_id UUID,
    timestamp TIMESTAMP,
    status VARCHAR(20),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create staging tables (processed data)
CREATE TABLE IF NOT EXISTS techcorp_staging.customers_clean (
    customer_id INTEGER PRIMARY KEY,
    email_hash VARCHAR(64),  -- Hashed for privacy
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    phone_masked VARCHAR(20),  -- Masked for privacy
    city VARCHAR(100),
    state VARCHAR(50),
    zip_code VARCHAR(10),
    customer_segment VARCHAR(20),
    created_at TIMESTAMP,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS techcorp_staging.transactions_clean (
    transaction_id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount DECIMAL(10,2),
    currency VARCHAR(3),
    transaction_type VARCHAR(50),
    status VARCHAR(20),
    fraud_score DECIMAL(3,2),
    category VARCHAR(50),
    created_at TIMESTAMP,
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create data quality tracking table
CREATE TABLE IF NOT EXISTS techcorp_staging.data_quality_metrics (
    id SERIAL PRIMARY KEY,
    table_name VARCHAR(100),
    metric_name VARCHAR(100),
    metric_value DECIMAL(10,4),
    threshold_value DECIMAL(10,4),
    status VARCHAR(20),
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dag_run_id VARCHAR(100),
    task_id VARCHAR(100)
);

-- Create pipeline execution tracking
CREATE TABLE IF NOT EXISTS techcorp_staging.pipeline_executions (
    id SERIAL PRIMARY KEY,
    dag_id VARCHAR(100),
    run_id VARCHAR(100),
    execution_date TIMESTAMP,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    status VARCHAR(20),
    records_processed INTEGER,
    data_size_mb DECIMAL(10,2),
    processing_strategy VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data for testing
INSERT INTO techcorp_raw.customers (email, first_name, last_name, phone, city, state, status)
VALUES 
    ('john.doe@techcorp.com', 'John', 'Doe', '555-123-4567', 'New York', 'NY', 'active'),
    ('jane.smith@techcorp.com', 'Jane', 'Smith', '555-234-5678', 'Los Angeles', 'CA', 'active'),
    ('bob.johnson@techcorp.com', 'Bob', 'Johnson', '555-345-6789', 'Chicago', 'IL', 'active'),
    ('alice.brown@techcorp.com', 'Alice', 'Brown', '555-456-7890', 'Houston', 'TX', 'inactive'),
    ('charlie.davis@techcorp.com', 'Charlie', 'Davis', '555-567-8901', 'Phoenix', 'AZ', 'active')
ON CONFLICT (customer_id) DO NOTHING;

-- Insert sample transactions
INSERT INTO techcorp_raw.transactions (customer_id, amount, transaction_type, status)
SELECT 
    (random() * 4 + 1)::integer,
    (random() * 1000 + 10)::decimal(10,2),
    CASE (random() * 3)::integer
        WHEN 0 THEN 'purchase'
        WHEN 1 THEN 'refund'
        ELSE 'adjustment'
    END,
    CASE (random() * 2)::integer
        WHEN 0 THEN 'completed'
        ELSE 'pending'
    END
FROM generate_series(1, 100);

-- Insert sample products
INSERT INTO techcorp_raw.products (product_id, name, category, price, cost, status)
VALUES 
    ('PROD-0001', 'Wireless Headphones', 'Electronics', 99.99, 45.00, 'active'),
    ('PROD-0002', 'Cotton T-Shirt', 'Clothing', 24.99, 12.00, 'active'),
    ('PROD-0003', 'Programming Book', 'Books', 49.99, 25.00, 'active'),
    ('PROD-0004', 'Garden Hose', 'Home & Garden', 34.99, 18.00, 'inactive'),
    ('PROD-0005', 'Basketball', 'Sports', 29.99, 15.00, 'active')
ON CONFLICT (product_id) DO NOTHING;

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_customers_email ON techcorp_raw.customers(email);
CREATE INDEX IF NOT EXISTS idx_customers_created_at ON techcorp_raw.customers(created_at);
CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON techcorp_raw.transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON techcorp_raw.transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_user_events_timestamp ON techcorp_raw.user_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_user_events_user_id ON techcorp_raw.user_events(user_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_executions_dag_id ON techcorp_staging.pipeline_executions(dag_id);
CREATE INDEX IF NOT EXISTS idx_data_quality_table_name ON techcorp_staging.data_quality_metrics(table_name);

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA techcorp_raw TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA techcorp_staging TO airflow;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA techcorp_marts TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA techcorp_raw TO airflow;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA techcorp_staging TO airflow;
GRANT USAGE ON SCHEMA techcorp_raw TO airflow;
GRANT USAGE ON SCHEMA techcorp_staging TO airflow;
GRANT USAGE ON SCHEMA techcorp_marts TO airflow;