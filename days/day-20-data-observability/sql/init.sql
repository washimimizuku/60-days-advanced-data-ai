-- Initialize observability database schema

-- Create database user and permissions
CREATE USER IF NOT EXISTS obs_user WITH PASSWORD 'obs_password';
GRANT ALL PRIVILEGES ON DATABASE observability_db TO obs_user;

-- Create schemas
CREATE SCHEMA IF NOT EXISTS data_sources;
CREATE SCHEMA IF NOT EXISTS monitoring;
CREATE SCHEMA IF NOT EXISTS alerts;

-- Grant schema permissions
GRANT ALL ON SCHEMA data_sources TO obs_user;
GRANT ALL ON SCHEMA monitoring TO obs_user;
GRANT ALL ON SCHEMA alerts TO obs_user;

-- Create data source tables
CREATE TABLE IF NOT EXISTS data_sources.customer_transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id VARCHAR(50) NOT NULL,
    email VARCHAR(255) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    quantity INTEGER NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL,
    category VARCHAR(50) NOT NULL,
    region VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    subscription_status VARCHAR(20) DEFAULT 'active'
);

-- Create monitoring tables
CREATE TABLE IF NOT EXISTS monitoring.data_quality_metrics (
    metric_id SERIAL PRIMARY KEY,
    table_name VARCHAR(100) NOT NULL,
    metric_type VARCHAR(50) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    threshold_value DECIMAL(10,4),
    is_anomaly BOOLEAN DEFAULT FALSE,
    severity VARCHAR(20) DEFAULT 'low',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create alert tables
CREATE TABLE IF NOT EXISTS alerts.alert_history (
    alert_id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    table_name VARCHAR(100),
    current_value DECIMAL(10,4) NOT NULL,
    threshold_value DECIMAL(10,4) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    status VARCHAR(20) DEFAULT 'open',
    message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Grant permissions on all tables
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA data_sources TO obs_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO obs_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA alerts TO obs_user;