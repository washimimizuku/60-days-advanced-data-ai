-- Sample data for QualityFirst Corp data quality testing

-- Create schema
CREATE SCHEMA IF NOT EXISTS quality_test;
SET search_path TO quality_test;

-- Customer data table
CREATE TABLE IF NOT EXISTS customer_data (
    customer_id VARCHAR(20) PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    age INTEGER,
    subscription_status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Sample customer data
INSERT INTO customer_data VALUES
('CUST_00000001', 'john.doe@email.com', 'John', 'Doe', 28, 'active', '2023-01-15 10:30:00', '2024-01-15 10:30:00'),
('CUST_00000002', 'jane.smith@email.com', 'Jane', 'Smith', 34, 'active', '2023-02-20 14:15:00', '2024-01-20 14:15:00'),
('CUST_00000003', 'bob.wilson@email.com', 'Bob', 'Wilson', 45, 'inactive', '2023-03-10 09:45:00', '2024-01-10 09:45:00'),
('CUST_00000004', 'alice.brown@email.com', 'Alice', 'Brown', 29, 'trial', '2024-01-01 16:20:00', '2024-01-01 16:20:00'),
('CUST_00000005', 'charlie.davis@email.com', 'Charlie', 'Davis', 52, 'suspended', '2023-06-15 11:10:00', '2024-01-15 11:10:00'),
('CUST_00000006', 'diana.miller@email.com', 'Diana', 'Miller', 31, 'active', '2023-08-22 13:25:00', '2024-01-22 13:25:00'),
('CUST_00000007', 'frank.garcia@email.com', 'Frank', 'Garcia', 38, 'cancelled', '2023-04-18 15:40:00', '2023-12-18 15:40:00'),
('CUST_00000008', 'grace.lee@email.com', 'Grace', 'Lee', 26, 'active', '2023-09-05 12:55:00', '2024-01-05 12:55:00');

-- Transaction data table
CREATE TABLE IF NOT EXISTS transaction_data (
    transaction_id VARCHAR(20) PRIMARY KEY,
    customer_id VARCHAR(20) NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    currency VARCHAR(3) NOT NULL,
    status VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customer_data(customer_id)
);

-- Sample transaction data
INSERT INTO transaction_data VALUES
('TXN_000000000001', 'CUST_00000001', 99.99, 'USD', 'completed', '2024-01-15 11:00:00'),
('TXN_000000000002', 'CUST_00000002', 149.50, 'USD', 'completed', '2024-01-20 15:30:00'),
('TXN_000000000003', 'CUST_00000001', 29.99, 'USD', 'completed', '2024-01-22 09:15:00'),
('TXN_000000000004', 'CUST_00000004', 199.00, 'USD', 'pending', '2024-01-25 14:45:00'),
('TXN_000000000005', 'CUST_00000006', 79.99, 'USD', 'completed', '2024-01-23 16:20:00'),
('TXN_000000000006', 'CUST_00000002', 299.99, 'USD', 'failed', '2024-01-24 10:10:00'),
('TXN_000000000007', 'CUST_00000008', 49.99, 'USD', 'completed', '2024-01-25 13:35:00'),
('TXN_000000000008', 'CUST_00000001', 19.99, 'USD', 'refunded', '2024-01-26 11:50:00');

-- Product catalog table
CREATE TABLE IF NOT EXISTS product_catalog (
    product_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) NOT NULL,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL
);

-- Sample product data
INSERT INTO product_catalog VALUES
('PROD_001', 'Premium Subscription', 'subscription', 99.99, 'active', '2023-01-01 00:00:00', '2024-01-01 00:00:00'),
('PROD_002', 'Basic Plan', 'subscription', 29.99, 'active', '2023-01-01 00:00:00', '2024-01-01 00:00:00'),
('PROD_003', 'Enterprise Package', 'subscription', 299.99, 'active', '2023-01-01 00:00:00', '2024-01-01 00:00:00'),
('PROD_004', 'Add-on Feature', 'feature', 19.99, 'active', '2023-06-01 00:00:00', '2024-01-01 00:00:00'),
('PROD_005', 'Legacy Plan', 'subscription', 49.99, 'discontinued', '2022-01-01 00:00:00', '2023-12-31 23:59:59');

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_customer_email ON customer_data(email);
CREATE INDEX IF NOT EXISTS idx_customer_status ON customer_data(subscription_status);
CREATE INDEX IF NOT EXISTS idx_transaction_customer ON transaction_data(customer_id);
CREATE INDEX IF NOT EXISTS idx_transaction_timestamp ON transaction_data(timestamp);
CREATE INDEX IF NOT EXISTS idx_product_category ON product_catalog(category);
CREATE INDEX IF NOT EXISTS idx_product_status ON product_catalog(status);