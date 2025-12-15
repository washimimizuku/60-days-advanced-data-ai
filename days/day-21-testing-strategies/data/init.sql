-- Day 21: Testing Strategies - Database Initialization
-- Initialize PostgreSQL database for testing

-- Create testing database schema
CREATE SCHEMA IF NOT EXISTS testing;

-- Create sample transactions table
CREATE TABLE IF NOT EXISTS testing.transactions (
    transaction_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    merchant_name VARCHAR(100),
    transaction_date TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create customer metrics table
CREATE TABLE IF NOT EXISTS testing.customer_metrics (
    customer_id VARCHAR(20) PRIMARY KEY,
    total_spent DECIMAL(12,2),
    avg_amount DECIMAL(10,2),
    transaction_count INTEGER,
    first_transaction TIMESTAMP,
    last_transaction TIMESTAMP,
    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create fraud detection results table
CREATE TABLE IF NOT EXISTS testing.fraud_results (
    transaction_id VARCHAR(20) PRIMARY KEY,
    is_suspicious BOOLEAN DEFAULT FALSE,
    fraud_score DECIMAL(5,3),
    fraud_reasons TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create test results tracking table
CREATE TABLE IF NOT EXISTS testing.test_results (
    test_id SERIAL PRIMARY KEY,
    test_name VARCHAR(100) NOT NULL,
    test_type VARCHAR(50) NOT NULL,
    status VARCHAR(20) NOT NULL,
    execution_time DECIMAL(10,3),
    memory_usage_mb DECIMAL(10,2),
    throughput DECIMAL(10,2),
    error_message TEXT,
    executed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample transaction data
INSERT INTO testing.transactions (transaction_id, customer_id, amount, currency, merchant_name, transaction_date) VALUES
('TXN000001', 'CUST001', 125.50, 'USD', 'Amazon', '2024-01-01 10:30:00'),
('TXN000002', 'CUST001', 89.99, 'USD', 'Walmart', '2024-01-01 14:15:00'),
('TXN000003', 'CUST002', 250.00, 'EUR', 'MediaMarkt', '2024-01-01 16:45:00'),
('TXN000004', 'CUST003', 45.75, 'GBP', 'Tesco', '2024-01-02 09:20:00'),
('TXN000005', 'CUST001', 15000.00, 'USD', 'Luxury Cars Inc', '2024-01-02 11:00:00'),
('TXN000006', 'CUST002', 75.25, 'EUR', 'Carrefour', '2024-01-02 13:30:00'),
('TXN000007', 'CUST004', 199.99, 'USD', 'Best Buy', '2024-01-02 15:45:00'),
('TXN000008', 'CUST003', 32.50, 'GBP', 'Sainsbury', '2024-01-03 08:15:00'),
('TXN000009', 'CUST005', 500.00, 'USD', 'Apple Store', '2024-01-03 12:00:00'),
('TXN000010', 'CUST001', 25.99, 'USD', 'Starbucks', '2024-01-03 16:30:00');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transactions_customer_id ON testing.transactions(customer_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON testing.transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_transactions_amount ON testing.transactions(amount);
CREATE INDEX IF NOT EXISTS idx_test_results_name ON testing.test_results(test_name);
CREATE INDEX IF NOT EXISTS idx_test_results_type ON testing.test_results(test_type);

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA testing TO testuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA testing TO testuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA testing TO testuser;