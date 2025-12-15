-- Create additional databases
CREATE DATABASE dataflow_warehouse;

-- Connect to dataflow_warehouse and create sample tables
\c dataflow_warehouse;

CREATE TABLE IF NOT EXISTS customers (
    customer_id SERIAL PRIMARY KEY,
    email VARCHAR(255),
    first_name VARCHAR(100),
    customer_segment VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id SERIAL PRIMARY KEY,
    customer_id INTEGER,
    amount DECIMAL(10,2),
    transaction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert sample data
INSERT INTO customers (email, first_name, customer_segment) VALUES
('user1@example.com', 'John', 'premium'),
('user2@example.com', 'Jane', 'standard'),
('user3@example.com', 'Bob', 'basic');

INSERT INTO transactions (customer_id, amount) VALUES
(1, 150.00),
(2, 75.50),
(3, 200.00);