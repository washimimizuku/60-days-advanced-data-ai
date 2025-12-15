-- Day 1: PostgreSQL Advanced - Indexing and Query Optimization
-- SOLUTION: Optimize an e-commerce database

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

-- Run EXPLAIN ANALYZE on this query and note the execution time
-- Query: Find user by email
EXPLAIN ANALYZE
SELECT * FROM users WHERE email = 'user500@example.com';

-- Answer: Using Seq Scan (Sequential Scan)
-- Execution time: ~0.5-2ms for 1000 rows (will be much slower with millions)

-- Run EXPLAIN ANALYZE on this query
-- Query: Find all orders for a specific user
EXPLAIN ANALYZE
SELECT * FROM orders WHERE user_id = 500;

-- Answer: Using Seq Scan
-- Execution time: ~1-3ms for 5000 rows

-- ============================================================================
-- EXERCISE 2: Create appropriate indexes
-- ============================================================================

-- Solution 1: Create an index on users.email
-- Email is frequently used for lookups (login, search)
CREATE INDEX idx_users_email ON users(email);

-- Solution 2: Create an index on orders.user_id
-- Foreign key lookups are very common
CREATE INDEX idx_orders_user_id ON orders(user_id);

-- Solution 3: Create a composite index on orders(user_id, created_at)
-- Useful for queries filtering by user and date range
-- Most selective column (user_id) comes first
CREATE INDEX idx_orders_user_created ON orders(user_id, created_at);

-- Solution 4: Create a partial index on users for active users only
-- Only index active users since they're queried most often
-- This makes the index smaller and faster
CREATE INDEX idx_users_active_email ON users(email) WHERE status = 'active';

-- Solution 5: Create a GIN index on products.metadata
-- GIN indexes are perfect for JSONB containment queries
CREATE INDEX idx_products_metadata_gin ON products USING GIN(metadata);

-- ============================================================================
-- EXERCISE 3: Verify improvements
-- ============================================================================

-- Query 1: Find user by email (should now use Index Scan)
EXPLAIN ANALYZE
SELECT * FROM users WHERE email = 'user500@example.com';

-- Result: Now uses "Index Scan using idx_users_email"
-- Execution time: ~0.05-0.2ms (10-20x faster!)

-- Query 2: Find orders for user (should now use Index Scan)
EXPLAIN ANALYZE
SELECT * FROM orders WHERE user_id = 500;

-- Result: Now uses "Index Scan using idx_orders_user_id"
-- Execution time: ~0.1-0.3ms (10-30x faster!)

-- Answer: Queries are typically 10-30x faster with proper indexes
-- The improvement is even more dramatic with larger datasets

-- ============================================================================
-- EXERCISE 4: Optimize complex queries
-- ============================================================================

-- Analyze this query and create appropriate indexes
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

-- Solution: Indexes that would help this query:
-- 1. idx_users_active_email (already created) - filters active users
-- 2. idx_orders_user_created (already created) - join + date filter
-- 3. Additional index on orders.created_at for date filtering
CREATE INDEX idx_orders_created_at ON orders(created_at);

-- 4. Index on users.status for filtering
CREATE INDEX idx_users_status ON users(status);

-- Re-run the query to see improvements
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

-- ============================================================================
-- EXERCISE 5: Find unused indexes
-- ============================================================================

-- Query to find indexes that are never used
-- This helps identify indexes that can be dropped to improve write performance
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelname NOT LIKE '%_pkey'  -- Exclude primary keys
ORDER BY pg_relation_size(indexrelid) DESC;

-- Note: In a fresh database, all indexes will show 0 scans
-- In production, run this query after the system has been running for a while

-- ============================================================================
-- BONUS: Advanced optimization
-- ============================================================================

-- Solution 1: Create an index on a function (e.g., LOWER(email))
-- Useful for case-insensitive searches
CREATE INDEX idx_users_email_lower ON users(LOWER(email));

-- Now this query can use the index:
EXPLAIN ANALYZE
SELECT * FROM users WHERE LOWER(email) = 'user500@example.com';

-- Solution 2: Find products with specific JSONB properties
EXPLAIN ANALYZE
SELECT * FROM products 
WHERE metadata @> '{"featured": true}';

-- This query already uses idx_products_metadata_gin (created earlier)
-- GIN indexes are perfect for JSONB containment queries (@>)

-- Additional JSONB index examples:
-- Index for existence queries
CREATE INDEX idx_products_metadata_rating ON products 
USING GIN((metadata -> 'rating'));

-- Query that uses it:
EXPLAIN ANALYZE
SELECT * FROM products WHERE metadata ? 'rating';

-- ============================================================================
-- ADDITIONAL SOLUTIONS: Index Maintenance
-- ============================================================================

-- Check index sizes
SELECT 
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;

-- Check table sizes
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS total_size,
    pg_size_pretty(pg_relation_size(schemaname||'.'||tablename)) AS table_size,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename) - 
                   pg_relation_size(schemaname||'.'||tablename)) AS indexes_size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Update statistics for query planner
ANALYZE users;
ANALYZE orders;
ANALYZE products;

-- Or analyze all tables
ANALYZE;

-- Rebuild an index to remove bloat (use CONCURRENTLY in production)
-- REINDEX INDEX CONCURRENTLY idx_users_email;

-- Check for bloated indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS index_size,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;

-- ============================================================================
-- PERFORMANCE COMPARISON SUMMARY
-- ============================================================================

/*
BEFORE INDEXES:
- User lookup by email: Seq Scan, ~1-2ms
- Orders by user_id: Seq Scan, ~2-3ms
- Complex join query: Multiple Seq Scans, ~10-20ms

AFTER INDEXES:
- User lookup by email: Index Scan, ~0.05-0.2ms (10-20x faster)
- Orders by user_id: Index Scan, ~0.1-0.3ms (10-30x faster)
- Complex join query: Index Scans, ~1-3ms (5-10x faster)

KEY LEARNINGS:
1. Indexes dramatically improve read performance
2. B-tree indexes work for most use cases
3. Composite indexes must be ordered correctly (most selective first)
4. Partial indexes save space and improve performance
5. GIN indexes are essential for JSONB queries
6. Function-based indexes enable optimized function queries
7. Regular maintenance (ANALYZE, REINDEX) is important
8. Monitor index usage to identify unused indexes
9. Balance read performance vs write overhead
10. EXPLAIN ANALYZE is your best friend for optimization
*/

-- ============================================================================
-- CLEANUP (optional)
-- ============================================================================

-- Uncomment to drop tables and start fresh
-- DROP TABLE IF EXISTS users CASCADE;
-- DROP TABLE IF EXISTS orders CASCADE;
-- DROP TABLE IF EXISTS products CASCADE;
