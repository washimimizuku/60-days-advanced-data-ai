# Day 1 Quiz: PostgreSQL Advanced - Indexing and Query Optimization

## Questions

### 1. What is the default index type in PostgreSQL?
- a) Hash
- b) B-tree
- c) GIN
- d) GiST

### 2. Which command shows the actual execution time of a query?
- a) EXPLAIN
- b) ANALYZE
- c) EXPLAIN ANALYZE
- d) SHOW QUERY

### 3. What does "Seq Scan" in EXPLAIN output indicate?
- a) The query is using an index
- b) The query is performing a full table scan
- c) The query has a syntax error
- d) The query is optimized

### 4. Which index type is best for JSONB columns?
- a) B-tree
- b) Hash
- c) GIN
- d) GiST

### 5. In a composite index on (user_id, created_at), which query can use the index?
- a) WHERE created_at > '2024-01-01'
- b) WHERE user_id = 123
- c) WHERE user_id = 123 OR created_at > '2024-01-01'
- d) WHERE created_at > '2024-01-01' AND status = 'active'

### 6. What is a partial index?
- a) An index that is not fully built
- b) An index on only part of a column
- c) An index on a subset of rows matching a condition
- d) An index that only works sometimes

### 7. Why might you avoid creating too many indexes?
- a) They take up disk space
- b) They slow down INSERT, UPDATE, and DELETE operations
- c) They need to be maintained
- d) All of the above

### 8. Which query will NOT use an index on the email column?
- a) WHERE email = 'user@example.com'
- b) WHERE LOWER(email) = 'user@example.com'
- c) WHERE email IN ('user1@example.com', 'user2@example.com')
- d) WHERE email LIKE 'user@%'

### 9. What does REINDEX do?
- a) Removes an index
- b) Rebuilds an index to reduce bloat
- c) Creates a new index
- d) Analyzes index usage

### 10. Which PostgreSQL command updates statistics for the query planner?
- a) UPDATE STATS
- b) REFRESH
- c) ANALYZE
- d) OPTIMIZE

---

## Answers

### 1. What is the default index type in PostgreSQL?
**Answer: b) B-tree**

**Explanation:** B-tree is the default index type in PostgreSQL and works well for most use cases including equality comparisons, range queries, and sorting. When you create an index without specifying a type (CREATE INDEX idx_name ON table(column)), PostgreSQL creates a B-tree index.

---

### 2. Which command shows the actual execution time of a query?
**Answer: c) EXPLAIN ANALYZE**

**Explanation:** EXPLAIN ANALYZE actually executes the query and shows real execution times, whereas EXPLAIN only shows the estimated query plan without executing it. EXPLAIN ANALYZE is essential for understanding actual query performance in production.

---

### 3. What does "Seq Scan" in EXPLAIN output indicate?
**Answer: b) The query is performing a full table scan**

**Explanation:** Seq Scan (Sequential Scan) means PostgreSQL is reading every row in the table to find matching records. This is slow for large tables and usually indicates a missing index. For small tables (< 1000 rows), Seq Scan might actually be faster than using an index.

---

### 4. Which index type is best for JSONB columns?
**Answer: c) GIN**

**Explanation:** GIN (Generalized Inverted Index) is specifically designed for JSONB columns and supports containment queries (@>), existence queries (?), and other JSONB operators. B-tree indexes don't work well with JSONB data.

---

### 5. In a composite index on (user_id, created_at), which query can use the index?
**Answer: b) WHERE user_id = 123**

**Explanation:** Composite indexes work left-to-right. A query can use the index if it filters on the leftmost column(s). "WHERE user_id = 123" uses the first column, so it can use the index. "WHERE created_at > '2024-01-01'" cannot use this index because it doesn't filter on user_id first.

---

### 6. What is a partial index?
**Answer: c) An index on a subset of rows matching a condition**

**Explanation:** A partial index includes only rows that match a WHERE condition (e.g., CREATE INDEX idx_active_users ON users(email) WHERE status = 'active'). This makes the index smaller, faster, and more efficient for queries that filter on the same condition.

---

### 7. Why might you avoid creating too many indexes?
**Answer: d) All of the above**

**Explanation:** While indexes speed up reads, they have costs: they consume disk space, slow down writes (INSERT/UPDATE/DELETE must update all indexes), and require maintenance (VACUUM, ANALYZE, REINDEX). The key is to index strategically based on actual query patterns.

---

### 8. Which query will NOT use an index on the email column?
**Answer: b) WHERE LOWER(email) = 'user@example.com'**

**Explanation:** Applying a function (LOWER) to an indexed column prevents index usage because the index stores the original values, not the function results. To fix this, create an expression index: CREATE INDEX idx_email_lower ON users(LOWER(email)).

---

### 9. What does REINDEX do?
**Answer: b) Rebuilds an index to reduce bloat**

**Explanation:** REINDEX rebuilds an index from scratch, which removes bloat (wasted space from updates/deletes) and can improve performance. Use REINDEX CONCURRENTLY in production to avoid blocking queries. Regular REINDEX is part of database maintenance.

---

### 10. Which PostgreSQL command updates statistics for the query planner?
**Answer: c) ANALYZE**

**Explanation:** ANALYZE collects statistics about table contents (row counts, value distributions, etc.) that the query planner uses to choose optimal execution plans. Run ANALYZE after significant data changes or as part of regular maintenance. VACUUM ANALYZE does both cleanup and statistics update.

---

## Score Interpretation

- **9-10 correct**: Excellent! You have a strong understanding of PostgreSQL indexing
- **7-8 correct**: Good job! Review the questions you missed
- **5-6 correct**: You're on the right track. Review the theory section
- **Below 5**: Take time to review the material and try the exercise again

---

## Key Concepts to Remember

1. **B-tree is the default** and works for most cases
2. **EXPLAIN ANALYZE** shows actual performance
3. **Seq Scan = full table scan** (usually slow)
4. **GIN for JSONB**, GiST for spatial data
5. **Composite indexes** work left-to-right
6. **Partial indexes** save space and improve performance
7. **Too many indexes** slow down writes
8. **Functions on columns** prevent index usage
9. **REINDEX** reduces bloat
10. **ANALYZE** updates query planner statistics

Ready to move on to Day 2! 🚀
