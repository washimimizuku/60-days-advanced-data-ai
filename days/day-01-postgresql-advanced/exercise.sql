-- Day 1: PostgreSQL Advanced - Indexing and Query Optimization
-- Exercise: Optimize an e-commerce database

-- ============================================================================
-- SETUP: Create sample tables and data
-- ============================================================================

-- Create users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    user_id INTEGER,
    total_amount DECIMAL(10, 2),
    status VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create products table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    category VARCHAR(100),
    price DECIMAL(10, 2),
    metadata JSONB
);

-- Insert sample data (1000 users, 5000 orders, 500 products)
INSERT INTO users (email, first_name, last_name, status)
SELECT 
    'user' || i || '@example.com',
    'First' || i,
    'Last' || i,
    CASE WHEN i % 3 = 0 THEN 'inactive' ELSE 'active' END
FROM generate_series(1, 1000) AS i;

INSERT INTO orders (user_id, total_amount, status)
SELECT 
    (random() * 999 + 1)::INTEGER,
    (random() * 1000)::DECIMAL(10, 2),
    CASE WHEN random() < 0.8 THEN 'completed' ELSE 'pending' END
FROM generate_series(1, 5000);

INSERT INTO products (name, category, price, metadata)
SELECT 
    'Product ' || i,
    CASE (i % 5) 
        WHEN 0 THEN 'Electronics'
        WHEN 1 THEN 'Books'
        WHEN 2 THEN 'Clothing'
        WHEN 3 THEN 'Home'
        ELSE 'Sports'
    END,
    (random() * 500 + 10)::DECIMAL(10, 2),
    jsonb_build_object(
        'featured', random() < 0.2,
        'rating', (random() * 5)::NUMERIC(2,1)
    )
FROM generate_series(1, 500) AS i;

-- ============================================================================
-- EXERCISE 1: Analyze slow queries
-- ============================================================================

-- TODO: Run EXPLAIN ANALYZE on this query and note the execution time
-- Query: Find user by email
EXPLAIN ANALYZE
SELECT * FROM users WHERE email = 'user500@example.com';

-- Question: Is it using a Seq Scan or Index Scan?
-- Answer: _______________

-- TODO: Run EXPLAIN ANALYZE on this query
-- Query: Find all orders for a specific user
EXPLAIN ANALYZE
SELECT * FROM orders WHERE user_id = 500;

-- Question: What is the execution time?
-- Answer: _______________

-- ============================================================================
-- EXERCISE 2: Create appropriate indexes
-- ============================================================================

-- TODO: Create an index on users.email
-- Hint: Use CREATE INDEX

-- TODO: Create an index on orders.user_id
-- Hint: Consider what type of queries will use this

-- TODO: Create a composite index on orders(user_id, created_at)
-- Hint: Most selective column first

-- TODO: Create a partial index on users for active users only
-- Hint: Use WHERE clause in index definition

-- TODO: Create a GIN index on products.metadata
-- Hint: Use USING GIN

-- ============================================================================
-- EXERCISE 3: Verify improvements
-- ============================================================================

-- TODO: Re-run the queries from Exercise 1 and compare execution times

-- Query 1: Find user by email (should now use index)
EXPLAIN ANALYZE
SELECT * FROM users WHERE email = 'user500@example.com';

-- Query 2: Find orders for user (should now use index)
EXPLAIN ANALYZE
SELECT * FROM orders WHERE user_id = 500;

-- Question: How much faster are the queries now?
-- Answer: _______________

-- ============================================================================
-- EXERCISE 4: Optimize complex queries
-- ============================================================================

-- TODO: Analyze this query and create appropriate indexes
EXPLAIN ANALYZE
SELECT 
    u.email,
    COUNT(o.order_id) as order_count,
    SUM(o.total_amount) as total_spent
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
WHERE u.status = 'active'
  AND o.created_at > CURRENT_DATE - INTERVAL '30 days'
GROUP BY u.email
ORDER BY total_spent DESC
LIMIT 10;

-- TODO: What indexes would help this query?
-- Answer: _______________

-- ============================================================================
-- EXERCISE 5: Find unused indexes
-- ============================================================================

-- TODO: Query to find indexes that are never used
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
ORDER BY pg_relation_size(indexrelid) DESC;

-- ============================================================================
-- BONUS: Advanced optimization
-- ============================================================================

-- TODO: Create an index on a function (e.g., LOWER(email))
-- Hint: CREATE INDEX idx_name ON table(LOWER(column))

-- TODO: Find products with specific JSONB properties
EXPLAIN ANALYZE
SELECT * FROM products 
WHERE metadata @> '{"featured": true}';

-- TODO: Optimize this query with appropriate index

-- ============================================================================
-- CLEANUP (optional)
-- ============================================================================

-- DROP TABLE IF EXISTS users CASCADE;
-- DROP TABLE IF EXISTS orders CASCADE;
-- DROP TABLE IF EXISTS products CASCADE;
