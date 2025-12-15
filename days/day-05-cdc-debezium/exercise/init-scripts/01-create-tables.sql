-- Day 5: CDC with Debezium - Database Initialization
-- Create sample e-commerce database schema

-- Enable logical replication (already set in docker-compose)
-- This is handled by the postgres command in docker-compose.yml

-- Create replication user for Debezium
CREATE USER debezium WITH REPLICATION LOGIN PASSWORD 'debezium';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO debezium;
GRANT USAGE ON SCHEMA public TO debezium;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO debezium;

-- Create publication for all tables
CREATE PUBLICATION dbz_publication FOR ALL TABLES;

-- ============================================================================
-- E-COMMERCE SCHEMA
-- ============================================================================

-- Customers table
CREATE TABLE customers (
    customer_id SERIAL PRIMARY KEY,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    phone VARCHAR(20),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    country VARCHAR(50) DEFAULT 'USA',
    customer_segment VARCHAR(20) DEFAULT 'Standard',
    registration_date DATE DEFAULT CURRENT_DATE,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255) NOT NULL,
    description TEXT,
    category VARCHAR(100),
    subcategory VARCHAR(100),
    brand VARCHAR(100),
    unit_price DECIMAL(10,2) NOT NULL,
    unit_cost DECIMAL(10,2),
    weight_kg DECIMAL(8,3),
    dimensions_cm VARCHAR(50),
    color VARCHAR(50),
    size VARCHAR(20),
    inventory_quantity INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    customer_id INTEGER REFERENCES customers(customer_id),
    order_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    order_status VARCHAR(20) DEFAULT 'pending',
    total_amount DECIMAL(12,2) NOT NULL,
    tax_amount DECIMAL(10,2) DEFAULT 0,
    shipping_amount DECIMAL(8,2) DEFAULT 0,
    discount_amount DECIMAL(8,2) DEFAULT 0,
    payment_method VARCHAR(50),
    shipping_address TEXT,
    billing_address TEXT,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Order items table
CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    total_price DECIMAL(12,2) NOT NULL,
    discount_amount DECIMAL(8,2) DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inventory movements table (for tracking stock changes)
CREATE TABLE inventory_movements (
    movement_id SERIAL PRIMARY KEY,
    product_id INTEGER REFERENCES products(product_id),
    movement_type VARCHAR(20) NOT NULL, -- 'IN', 'OUT', 'ADJUSTMENT'
    quantity INTEGER NOT NULL,
    reference_type VARCHAR(50), -- 'ORDER', 'RETURN', 'ADJUSTMENT'
    reference_id INTEGER,
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User events table (for behavioral tracking)
CREATE TABLE user_events (
    event_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    session_id VARCHAR(100),
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB,
    page_url VARCHAR(500),
    user_agent TEXT,
    ip_address INET,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES FOR PERFORMANCE
-- ============================================================================

-- Customer indexes
CREATE INDEX idx_customers_email ON customers(email);
CREATE INDEX idx_customers_segment ON customers(customer_segment);
CREATE INDEX idx_customers_active ON customers(is_active);

-- Product indexes
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_brand ON products(brand);
CREATE INDEX idx_products_active ON products(is_active);

-- Order indexes
CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_orders_date ON orders(order_date);
CREATE INDEX idx_orders_status ON orders(order_status);

-- Order items indexes
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);

-- Inventory indexes
CREATE INDEX idx_inventory_product_id ON inventory_movements(product_id);
CREATE INDEX idx_inventory_type ON inventory_movements(movement_type);
CREATE INDEX idx_inventory_created_at ON inventory_movements(created_at);

-- User events indexes
CREATE INDEX idx_user_events_user_id ON user_events(user_id);
CREATE INDEX idx_user_events_type ON user_events(event_type);
CREATE INDEX idx_user_events_created_at ON user_events(created_at);
CREATE INDEX idx_user_events_session_id ON user_events(session_id);

-- ============================================================================
-- TRIGGERS FOR UPDATED_AT TIMESTAMPS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to tables with updated_at column
CREATE TRIGGER update_customers_updated_at 
    BEFORE UPDATE ON customers 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at 
    BEFORE UPDATE ON products 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at 
    BEFORE UPDATE ON orders 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SAMPLE DATA
-- ============================================================================

-- Insert sample customers
INSERT INTO customers (first_name, last_name, email, phone, address_line1, city, state, postal_code, customer_segment) VALUES
('John', 'Doe', 'john.doe@email.com', '+1-555-0101', '123 Main St', 'New York', 'NY', '10001', 'Premium'),
('Jane', 'Smith', 'jane.smith@email.com', '+1-555-0102', '456 Oak Ave', 'Los Angeles', 'CA', '90210', 'Standard'),
('Bob', 'Johnson', 'bob.johnson@email.com', '+1-555-0103', '789 Pine St', 'Chicago', 'IL', '60601', 'Premium'),
('Alice', 'Brown', 'alice.brown@email.com', '+1-555-0104', '321 Elm Dr', 'Houston', 'TX', '77001', 'Standard'),
('Charlie', 'Wilson', 'charlie.wilson@email.com', '+1-555-0105', '654 Maple Ln', 'Phoenix', 'AZ', '85001', 'Basic'),
('Diana', 'Davis', 'diana.davis@email.com', '+1-555-0106', '987 Cedar Ave', 'Philadelphia', 'PA', '19101', 'Premium'),
('Eve', 'Miller', 'eve.miller@email.com', '+1-555-0107', '147 Birch Rd', 'San Antonio', 'TX', '78201', 'Standard'),
('Frank', 'Garcia', 'frank.garcia@email.com', '+1-555-0108', '258 Spruce St', 'San Diego', 'CA', '92101', 'Basic'),
('Grace', 'Martinez', 'grace.martinez@email.com', '+1-555-0109', '369 Willow Way', 'Dallas', 'TX', '75201', 'Premium'),
('Henry', 'Anderson', 'henry.anderson@email.com', '+1-555-0110', '741 Poplar Pl', 'San Jose', 'CA', '95101', 'Standard');

-- Insert sample products
INSERT INTO products (product_name, description, category, subcategory, brand, unit_price, unit_cost, inventory_quantity) VALUES
('Wireless Headphones', 'Premium noise-canceling wireless headphones', 'Electronics', 'Audio', 'TechBrand', 299.99, 150.00, 50),
('Smartphone Case', 'Protective case for smartphones', 'Electronics', 'Accessories', 'ProtectCo', 24.99, 8.00, 200),
('Running Shoes', 'Lightweight running shoes for athletes', 'Footwear', 'Athletic', 'SportMax', 129.99, 65.00, 75),
('Coffee Maker', 'Automatic drip coffee maker', 'Home & Kitchen', 'Appliances', 'BrewMaster', 89.99, 45.00, 30),
('Yoga Mat', 'Non-slip yoga and exercise mat', 'Sports', 'Fitness', 'FlexFit', 39.99, 15.00, 100),
('Laptop Stand', 'Adjustable aluminum laptop stand', 'Electronics', 'Accessories', 'TechBrand', 79.99, 35.00, 40),
('Water Bottle', 'Insulated stainless steel water bottle', 'Sports', 'Accessories', 'HydroMax', 34.99, 12.00, 150),
('Desk Lamp', 'LED desk lamp with adjustable brightness', 'Home & Kitchen', 'Lighting', 'BrightLight', 59.99, 25.00, 60),
('Bluetooth Speaker', 'Portable wireless Bluetooth speaker', 'Electronics', 'Audio', 'SoundWave', 149.99, 70.00, 80),
('Backpack', 'Durable travel and work backpack', 'Accessories', 'Bags', 'CarryAll', 89.99, 40.00, 90);

-- Insert sample orders
INSERT INTO orders (customer_id, order_status, total_amount, tax_amount, shipping_amount, payment_method) VALUES
(1, 'completed', 324.98, 24.99, 9.99, 'Credit Card'),
(2, 'shipped', 154.98, 11.99, 7.99, 'PayPal'),
(3, 'processing', 219.97, 16.99, 12.99, 'Credit Card'),
(1, 'completed', 89.99, 6.99, 0.00, 'Credit Card'),
(4, 'cancelled', 39.99, 3.09, 5.99, 'Debit Card'),
(5, 'pending', 179.98, 13.99, 8.99, 'Credit Card'),
(2, 'completed', 299.99, 23.99, 0.00, 'PayPal'),
(6, 'shipped', 124.98, 9.99, 6.99, 'Credit Card'),
(3, 'completed', 59.99, 4.99, 5.99, 'Debit Card'),
(7, 'processing', 269.97, 20.99, 11.99, 'Credit Card');

-- Insert sample order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price, total_price) VALUES
(1, 1, 1, 299.99, 299.99),
(1, 2, 1, 24.99, 24.99),
(2, 3, 1, 129.99, 129.99),
(2, 2, 1, 24.99, 24.99),
(3, 1, 1, 299.99, 299.99),
(4, 4, 1, 89.99, 89.99),
(5, 5, 1, 39.99, 39.99),
(6, 6, 1, 79.99, 79.99),
(6, 7, 1, 34.99, 34.99),
(6, 8, 1, 59.99, 59.99),
(7, 1, 1, 299.99, 299.99),
(8, 3, 1, 129.99, 129.99),
(9, 8, 1, 59.99, 59.99),
(10, 9, 1, 149.99, 149.99),
(10, 10, 1, 89.99, 89.99);

-- Insert sample inventory movements
INSERT INTO inventory_movements (product_id, movement_type, quantity, reference_type, reference_id) VALUES
(1, 'OUT', -1, 'ORDER', 1),
(2, 'OUT', -2, 'ORDER', 1),
(3, 'OUT', -2, 'ORDER', 2),
(4, 'OUT', -1, 'ORDER', 4),
(5, 'OUT', -1, 'ORDER', 5),
(1, 'IN', 10, 'RESTOCK', NULL),
(3, 'IN', 5, 'RESTOCK', NULL);

-- Insert sample user events
INSERT INTO user_events (user_id, session_id, event_type, event_data, page_url) VALUES
(1, 'sess_abc123', 'page_view', '{"referrer": "google.com"}', '/products/1'),
(1, 'sess_abc123', 'add_to_cart', '{"product_id": 1, "quantity": 1}', '/products/1'),
(1, 'sess_abc123', 'checkout_start', '{"cart_total": 324.98}', '/checkout'),
(1, 'sess_abc123', 'purchase', '{"order_id": 1, "total": 324.98}', '/checkout/success'),
(2, 'sess_def456', 'page_view', '{"referrer": "direct"}', '/'),
(2, 'sess_def456', 'search', '{"query": "running shoes", "results": 5}', '/search'),
(2, 'sess_def456', 'page_view', '{}', '/products/3'),
(2, 'sess_def456', 'add_to_cart', '{"product_id": 3, "quantity": 1}', '/products/3'),
(3, 'sess_ghi789', 'page_view', '{"referrer": "facebook.com"}', '/products/1'),
(3, 'sess_ghi789', 'add_to_cart', '{"product_id": 1, "quantity": 1}', '/products/1');

-- Grant permissions to debezium user for new tables
GRANT SELECT ON ALL TABLES IN SCHEMA public TO debezium;

-- Show table counts
SELECT 'customers' as table_name, COUNT(*) as row_count FROM customers
UNION ALL
SELECT 'products', COUNT(*) FROM products
UNION ALL
SELECT 'orders', COUNT(*) FROM orders
UNION ALL
SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL
SELECT 'inventory_movements', COUNT(*) FROM inventory_movements
UNION ALL
SELECT 'user_events', COUNT(*) FROM user_events;

-- Show replication status
SELECT * FROM pg_publication;
SELECT * FROM pg_replication_slots;