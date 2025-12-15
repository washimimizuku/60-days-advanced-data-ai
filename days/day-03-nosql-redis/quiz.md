# Day 3 Quiz: NoSQL Databases - Redis for Caching

## Questions

### 1. What is the primary storage location for Redis data?
- a) Hard disk
- b) SSD storage
- c) Memory (RAM)
- d) Network storage

### 2. Which Redis data type is best for implementing a leaderboard?
- a) String
- b) Hash
- c) List
- d) Sorted Set (ZSet)

### 3. In the cache-aside pattern, what happens on a cache miss?
- a) Return null to the application
- b) Fetch data from database and store in cache
- c) Wait for cache to be populated
- d) Throw an exception

### 4. Which Redis command is used to set a key with an expiration time?
- a) SET key value TTL seconds
- b) SETEX key seconds value
- c) EXPIRE key value seconds
- d) TTL key seconds value

### 5. What is the main advantage of using Redis pipelines?
- a) Automatic data compression
- b) Reduced network round trips
- c) Better data persistence
- d) Automatic failover

### 6. Which Redis data structure is most memory-efficient for storing user profiles?
- a) Individual string keys for each field
- b) Hash for all user fields
- c) List of user attributes
- d) Set of user properties

### 7. What does the Redis ZADD command do?
- a) Adds elements to a list
- b) Adds members to a set
- c) Adds members with scores to a sorted set
- d) Adds fields to a hash

### 8. In Redis rate limiting, what is a "sliding window"?
- a) A fixed time period that resets at intervals
- b) A moving time window that continuously updates
- c) A window that slides between Redis instances
- d) A GUI window for monitoring Redis

### 9. Which Redis persistence option provides better durability?
- a) RDB (snapshots)
- b) AOF (append-only file)
- c) Memory-only mode
- d) Cluster mode

### 10. What is the purpose of Redis HyperLogLog?
- a) High-performance logging
- b) Approximate counting of unique elements
- c) Hyperfast data retrieval
- d) Log file compression

---

## Answers

### 1. What is the primary storage location for Redis data?
**Answer: c) Memory (RAM)**

**Explanation:** Redis is an in-memory data structure store, meaning all data is stored in RAM for ultra-fast access. This is why Redis can achieve sub-millisecond response times. While Redis offers persistence options (RDB snapshots and AOF logs) to save data to disk, the primary working dataset resides in memory. This makes Redis extremely fast but also means you need sufficient RAM to hold your entire dataset.

---

### 2. Which Redis data type is best for implementing a leaderboard?
**Answer: d) Sorted Set (ZSet)**

**Explanation:** Sorted Sets (ZSets) are perfect for leaderboards because they store unique members with associated scores and automatically maintain sorted order. You can easily get top players with ZREVRANGE, find a player's rank with ZREVRANK, and update scores with ZINCRBY. The sorted nature makes range queries and ranking operations very efficient, which is exactly what leaderboards need.

---

### 3. In the cache-aside pattern, what happens on a cache miss?
**Answer: b) Fetch data from database and store in cache**

**Explanation:** Cache-aside (also called lazy loading) is a caching pattern where the application manages the cache. On a cache miss, the application fetches data from the primary data store (database), stores it in the cache for future requests, and returns the data. This ensures the cache only contains data that's actually requested and keeps the cache and database loosely coupled.

---

### 4. Which Redis command is used to set a key with an expiration time?
**Answer: b) SETEX key seconds value**

**Explanation:** SETEX atomically sets a key with a value and expiration time in seconds. For example, `SETEX session:123 3600 "user_data"` sets the key with a 1-hour expiration. You can also use `SET key value EX seconds` or `SET key value PX milliseconds` for the same effect. Setting expiration prevents memory leaks and ensures stale data is automatically removed.

---

### 5. What is the main advantage of using Redis pipelines?
**Answer: b) Reduced network round trips**

**Explanation:** Redis pipelines allow you to send multiple commands in a single network round trip instead of waiting for each command's response. This dramatically reduces network latency overhead. For example, instead of 100 individual SET commands requiring 100 network round trips, a pipeline can send all 100 commands at once and receive all responses together, often resulting in 10x+ performance improvements.

---

### 6. Which Redis data structure is most memory-efficient for storing user profiles?
**Answer: b) Hash for all user fields**

**Explanation:** Redis hashes are specifically designed for storing objects with multiple fields and are much more memory-efficient than individual string keys. Storing user data as `HSET user:123 name "John" email "john@example.com" age "30"` uses significantly less memory than separate keys like `user:123:name`, `user:123:email`, `user:123:age`. Hashes also provide atomic operations on individual fields.

---

### 7. What does the Redis ZADD command do?
**Answer: c) Adds members with scores to a sorted set**

**Explanation:** ZADD adds one or more members with their associated scores to a sorted set. For example, `ZADD leaderboard 1500 "player1" 2000 "player2"` adds two players with their scores. The sorted set automatically maintains order by score, making it perfect for leaderboards, priority queues, and time-series data where you need both uniqueness and ordering.

---

### 8. In Redis rate limiting, what is a "sliding window"?
**Answer: b) A moving time window that continuously updates**

**Explanation:** A sliding window rate limiter uses a continuously moving time window rather than fixed intervals. For example, with a 1-hour sliding window, it counts requests from exactly 60 minutes ago to now, updating continuously. This provides smoother rate limiting compared to fixed windows, which can allow burst traffic at window boundaries. Redis sorted sets are commonly used to implement sliding window rate limiting.

---

### 9. Which Redis persistence option provides better durability?
**Answer: b) AOF (append-only file)**

**Explanation:** AOF (Append-Only File) provides better durability because it logs every write operation, potentially losing only the last second of data (with `appendfsync everysec`). RDB creates point-in-time snapshots, which can lose more data between snapshots. AOF files are also human-readable and can be manually edited if needed. However, AOF files are larger and slower to load than RDB files, so many production setups use both (hybrid persistence).

---

### 10. What is the purpose of Redis HyperLogLog?
**Answer: b) Approximate counting of unique elements**

**Explanation:** HyperLogLog is a probabilistic data structure that estimates the cardinality (number of unique elements) in a set using very little memory (only 12KB regardless of set size). It's perfect for counting unique visitors, unique page views, or unique events where exact counts aren't critical but memory efficiency is important. The error rate is typically less than 1%, making it suitable for analytics and monitoring use cases.

---

## Score Interpretation

- **9-10 correct**: Excellent! You have a strong understanding of Redis and caching patterns
- **7-8 correct**: Good job! Review the questions you missed and practice more with Redis data types
- **5-6 correct**: You're on the right track. Focus on understanding different Redis data structures and their use cases
- **Below 5**: Review the theory section and practice with the exercises

---

## Key Concepts to Remember

1. **Redis stores data in memory** for ultra-fast access
2. **Sorted Sets (ZSets)** are perfect for leaderboards and rankings
3. **Cache-aside pattern** - application manages cache, fetch on miss
4. **SETEX** sets key with expiration atomically
5. **Pipelines reduce network round trips** for better performance
6. **Hashes are memory-efficient** for storing objects
7. **ZADD** adds scored members to sorted sets
8. **Sliding window** provides smooth rate limiting
9. **AOF provides better durability** than RDB snapshots
10. **HyperLogLog** efficiently counts unique elements

---

## Redis Best Practices

- **Set appropriate TTLs** to prevent memory leaks
- **Use connection pooling** for better performance
- **Choose right data structure** for your use case
- **Use pipelines** for bulk operations
- **Monitor memory usage** and set limits
- **Implement proper error handling** for Redis failures
- **Use consistent key naming** conventions
- **Consider persistence options** based on durability needs
- **Plan for high availability** with replication/clustering
- **Test failover scenarios** in production-like environments

---

## Common Redis Use Cases

- **Caching** - Database query results, API responses, computed values
- **Session storage** - User sessions in web applications
- **Real-time analytics** - Counters, metrics, unique visitors
- **Rate limiting** - API throttling, abuse prevention
- **Leaderboards** - Gaming scores, rankings, top lists
- **Message queues** - Task queues, pub/sub messaging
- **Geospatial** - Location-based services, proximity searches
- **Time-series data** - Metrics, logs, sensor data

Ready to move on to Day 4! 🚀