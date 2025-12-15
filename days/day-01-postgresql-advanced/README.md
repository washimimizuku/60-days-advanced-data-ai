# Day 1: PostgreSQL Advanced - Indexing and Query Optimization

## 📖 Learning Objectives

By the end of today, you will:
- Understand PostgreSQL indexing strategies and types
- Analyze query performance using EXPLAIN and EXPLAIN ANALYZE
- Optimize slow queries for production workloads
- Apply best practices for index design and maintenance

---

## Theory

### Why PostgreSQL Indexing Matters

In production systems, query performance can make or break your application. A well-indexed database can handle millions of rows efficiently, while a poorly indexed one struggles with thousands.

**Real-world impact**:
- Queries go from 30 seconds to 50 milliseconds
- Database CPU usage drops from 90% to 10%
- Application response times improve dramatically
- Infrastructure costs decrease

### Index Types in PostgreSQL

#### 1. B-tree Index (Default)

The most common index type, perfect for equality and range queries.

```sql
-- Create a B-tree index
CREATE INDEX idx_users_email ON users(email);

-- Good for:
SELECT * FROM users WHERE email = 'user@example.com';
SELECT * FROM users WHERE created_at > '2024-01-01';
SELECT * FROM users WHERE age BETWEEN 25 AND 35;
```

**When to use**: Equality comparisons, range queries, sorting

**How it works**: B-tree (Balanced Tree) organizes data in a sorted tree structure. Each node contains keys and pointers to child nodes. This allows PostgreSQL to quickly navigate to the desired data without scanning the entire table.

#### 2. Hash Index

Optimized for equality comparisons only.

```sql
CREATE INDEX idx_users_id_hash ON users USING HASH(user_id);

-- Good for:
SELECT * FROM users WHERE user_id = 12345;

-- NOT good for:
SELECT * FROM users WHERE user_id > 12345;  -- Won't use hash index
```

**When to use**: Only equality comparisons, faster for exact matches

**Limitation**: Cannot be used for range queries or sorting

#### 3. GIN (Generalized Inverted Index)

Perfect for full-text search and JSONB columns.

```sql
-- For JSONB
CREATE INDEX idx_users_metadata_gin ON users USING GIN(metadata);
SELECT * FROM users WHERE metadata @> '{"premium": true}';

-- For arrays
CREATE INDEX idx_tags_gin ON articles USING GIN(tags);
SELECT * FROM articles WHERE tags @> ARRAY['postgresql', 'database'];

-- For full-text search
CREATE INDEX idx_content_fts ON articles USING GIN(to_tsvector('english', content));
SELECT * FROM articles WHERE to_tsvector('english', content) @@ to_tsquery('postgresql');
```

**When to use**: Full-text search, JSONB queries, array containment

**Trade-off**: Slower to build and update, but very fast for searches

#### 4. GiST (Generalized Search Tree)

Used for geometric data and full-text search.

```sql
-- For geometric data
CREATE INDEX idx_locations_gist ON stores USING GIST(location);
SELECT * FROM stores WHERE location <-> point '(40.7128, -74.0060)' < 10;
```

**When to use**: Geometric data, spatial queries, custom data types

### Composite Indexes

Indexes on multiple columns, order matters!

```sql
-- Create composite index
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);

-- Can use index (filters on leftmost column)
SELECT * FROM orders WHERE user_id = 123;
SELECT * FROM orders WHERE user_id = 123 AND created_at > '2024-01-01';

-- CANNOT use index (doesn't filter on leftmost column)
SELECT * FROM orders WHERE created_at > '2024-01-01';
```

**Rule**: Put the most selective column first (the one that filters out the most rows)

### Partial Indexes

Index only a subset of rows.

```sql
-- Only index active users
CREATE INDEX idx_active_users ON users(email) WHERE status = 'active';

-- This query uses the index
SELECT * FROM users WHERE email = 'user@example.com' AND status = 'active';

-- This query does NOT use the index
SELECT * FROM users WHERE email = 'user@example.com' AND status = 'inactive';
```

**Benefits**:
- Smaller index size
- Faster index scans
- Lower maintenance overhead
- Better for skewed data distributions

### Expression Indexes

Index the result of a function or expression.

```sql
-- Index for case-insensitive searches
CREATE INDEX idx_email_lower ON users(LOWER(email));

-- Now this query can use the index
SELECT * FROM users WHERE LOWER(email) = 'user@example.com';

-- Index for date extraction
CREATE INDEX idx_orders_year ON orders(EXTRACT(YEAR FROM created_at));
SELECT * FROM orders WHERE EXTRACT(YEAR FROM created_at) = 2024;
```

### Query Analysis with EXPLAIN

```sql
-- Show query plan (estimated)
EXPLAIN SELECT * FROM users WHERE email = 'user@example.com';

-- Execute and show actual performance
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'user@example.com';

-- More detailed output
EXPLAIN (ANALYZE, BUFFERS, VERBOSE) 
SELECT * FROM users WHERE email = 'user@example.com';
```

**Key metrics to understand**:

```
Seq Scan on users  (cost=0.00..18.50 rows=1 width=100) (actual time=0.015..0.234 rows=1 loops=1)
  Filter: (email = 'user@example.com')
  Rows Removed by Filter: 999
Planning Time: 0.123 ms
Execution Time: 0.267 ms
```

- **Seq Scan**: Full table scan (reads every row) - SLOW for large tables
- **Index Scan**: Uses an index - FAST
- **cost**: Estimated cost (startup..total)
- **rows**: Estimated number of rows
- **actual time**: Real execution time (startup..total)
- **loops**: Number of times the node was executed

**Common scan types**:
- **Seq Scan**: Full table scan
- **Index Scan**: Uses index, fetches rows from table
- **Index Only Scan**: Gets all data from index (fastest)
- **Bitmap Index Scan**: Uses multiple indexes
- **Nested Loop**: Join method for small datasets
- **Hash Join**: Join method for large datasets
- **Merge Join**: Join method for sorted data

### Real-World Example

```sql
-- Before optimization
EXPLAIN ANALYZE
SELECT u.email, COUNT(o.order_id) as order_count
FROM users u
JOIN orders o ON u.user_id = o.user_id
WHERE u.status = 'active'
  AND o.created_at > '2024-01-01'
GROUP BY u.email;

-- Result: Seq Scan on both tables, 250ms execution time

-- Add indexes
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_orders_user_date ON orders(user_id, created_at);

-- After optimization
-- Result: Index Scans, 15ms execution time (16x faster!)
```

### Index Maintenance

Indexes need regular maintenance to stay efficient.

```sql
-- Update statistics for query planner
ANALYZE users;
ANALYZE;  -- All tables

-- Rebuild an index (removes bloat)
REINDEX INDEX idx_users_email;
REINDEX TABLE users;

-- In production, use CONCURRENTLY to avoid blocking
REINDEX INDEX CONCURRENTLY idx_users_email;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan;

-- Find unused indexes
SELECT 
    schemaname,
    tablename,
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
WHERE idx_scan = 0
  AND indexrelname NOT LIKE '%_pkey';
```

### Best Practices

#### 1. Index Selectively

Don't index everything - indexes have costs:
- Disk space
- Slower writes (INSERT, UPDATE, DELETE)
- Maintenance overhead

**Index when**:
- Column is frequently in WHERE clauses
- Column is used in JOINs
- Column is used in ORDER BY
- Table is large (> 10,000 rows)

**Don't index when**:
- Table is small (< 1,000 rows)
- Column has low cardinality (few unique values)
- Column is rarely queried
- High write frequency

#### 2. Monitor Performance

```sql
-- Slow query log
ALTER DATABASE mydb SET log_min_duration_statement = 1000;  -- Log queries > 1s

-- Check table sizes
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Check index sizes
SELECT 
    indexname,
    pg_size_pretty(pg_relation_size(indexrelid)) AS size
FROM pg_stat_user_indexes
ORDER BY pg_relation_size(indexrelid) DESC;
```

#### 3. Composite Index Order

Most selective column first:

```sql
-- Good: user_id is very selective (many unique values)
CREATE INDEX idx_orders_user_status ON orders(user_id, status);

-- Bad: status has only 3 values (pending, completed, cancelled)
CREATE INDEX idx_orders_status_user ON orders(status, user_id);
```

#### 4. Use Partial Indexes for Skewed Data

```sql
-- If 95% of orders are 'completed', only index the 5%
CREATE INDEX idx_pending_orders ON orders(user_id) WHERE status = 'pending';
```

#### 5. Regular Maintenance

```sql
-- Weekly: Update statistics
ANALYZE;

-- Monthly: Check for bloat and reindex if needed
REINDEX INDEX CONCURRENTLY idx_users_email;

-- Quarterly: Review index usage and drop unused indexes
```

### Common Pitfalls

#### 1. Functions on Indexed Columns

```sql
-- BAD: Function prevents index usage
SELECT * FROM users WHERE LOWER(email) = 'user@example.com';

-- GOOD: Create expression index
CREATE INDEX idx_email_lower ON users(LOWER(email));
```

#### 2. Implicit Type Conversions

```sql
-- BAD: user_id is INTEGER, but comparing to string
SELECT * FROM users WHERE user_id = '123';  -- Prevents index usage

-- GOOD: Use correct type
SELECT * FROM users WHERE user_id = 123;
```

#### 3. Leading Wildcards

```sql
-- BAD: Cannot use B-tree index
SELECT * FROM users WHERE email LIKE '%@example.com';

-- GOOD: Can use index
SELECT * FROM users WHERE email LIKE 'user%';

-- For full-text search, use GIN index
CREATE INDEX idx_email_gin ON users USING GIN(email gin_trgm_ops);
```

### Performance Tuning Checklist

- [ ] Run EXPLAIN ANALYZE on slow queries
- [ ] Check for Seq Scans on large tables
- [ ] Create indexes on frequently filtered columns
- [ ] Use composite indexes for multi-column filters
- [ ] Consider partial indexes for skewed data
- [ ] Create expression indexes for function calls
- [ ] Monitor index usage with pg_stat_user_indexes
- [ ] Drop unused indexes
- [ ] Run ANALYZE regularly
- [ ] REINDEX to remove bloat

---

## 💻 Hands-On Exercise

See `exercise.sql` for hands-on practice with indexing and optimization.

**What you'll do**:
1. Analyze slow queries using EXPLAIN ANALYZE
2. Create appropriate indexes (B-tree, GIN, composite, partial)
3. Verify performance improvements
4. Optimize complex JOIN queries
5. Find and remove unused indexes

**Expected time**: 45 minutes

---

## 📚 Resources

- [PostgreSQL Indexes Documentation](https://www.postgresql.org/docs/current/indexes.html)
- [EXPLAIN Guide](https://www.postgresql.org/docs/current/using-explain.html)
- [Use The Index, Luke](https://use-the-index-luke.com/) - Excellent guide to database indexing
- [PostgreSQL Index Types](https://www.postgresql.org/docs/current/indexes-types.html)
- [pg_stat_user_indexes](https://www.postgresql.org/docs/current/monitoring-stats.html)

---

## 🎯 Key Takeaways

- **Indexes dramatically improve read performance** but have write costs
- **B-tree is the default** and works for most use cases (equality, ranges, sorting)
- **GIN for JSONB and arrays**, GiST for geometric data
- **EXPLAIN ANALYZE is your best friend** for understanding query performance
- **Composite indexes work left-to-right** - most selective column first
- **Partial indexes save space** and improve performance for skewed data
- **Expression indexes enable function optimization** (e.g., LOWER(email))
- **Monitor and maintain indexes regularly** - ANALYZE, REINDEX, drop unused
- **Don't over-index** - balance read performance vs write overhead
- **Seq Scan isn't always bad** - for small tables, it's often faster than index scans

---

## 🚀 What's Next?

Tomorrow (Day 2), you'll learn **NoSQL with MongoDB** - understanding document databases, when to use them over relational databases, and MongoDB-specific patterns for data modeling and querying.

**How today's knowledge applies**: The indexing principles you learned today (B-tree, composite, partial indexes) also apply to MongoDB, but with document-specific considerations:
- MongoDB uses similar B-tree indexes for queries
- Compound indexes work left-to-right (same as PostgreSQL)
- MongoDB adds specialized indexes (GIN-like for arrays, geospatial)
- Query optimization with explain() works similarly

**Preview**: MongoDB uses a flexible document model (JSON-like), which is great for:
- Rapidly evolving schemas
- Hierarchical data structures  
- High write throughput
- Horizontal scaling

---

## ✅ Before Moving On

- [ ] Understand different PostgreSQL index types (B-tree, Hash, GIN, GiST)
- [ ] Can use EXPLAIN and EXPLAIN ANALYZE to analyze queries
- [ ] Know when to use composite vs single-column indexes
- [ ] Understand partial and expression indexes
- [ ] Can identify and fix slow queries
- [ ] Complete the exercise in `exercise.sql`
- [ ] Review the solution in `solution.sql`
- [ ] Take the quiz in `quiz.md`

**Time**: ~1 hour | **Difficulty**: ⭐⭐⭐ (Intermediate)

Ready to optimize your databases! 🚀
