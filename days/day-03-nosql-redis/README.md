# Day 3: NoSQL Databases - Redis for Caching

## 📖 Learning Objectives

**Estimated Time**: 60 minutes

By the end of today, you will:
- Understand Redis architecture and use cases
- Implement caching strategies with Redis
- Use Redis data structures for real-time applications
- Design session management and rate limiting systems
- Apply Redis patterns for high-performance applications

---

## Theory

### What is Redis?

Redis (Remote Dictionary Server) is an in-memory data structure store used as a database, cache, message broker, and streaming engine. It's known for exceptional performance and versatility.

**Key characteristics**:
- **In-memory storage**: All data stored in RAM for ultra-fast access
- **Persistent**: Optional disk persistence for durability
- **Data structures**: Rich set of data types beyond key-value
- **Atomic operations**: All operations are atomic
- **High availability**: Replication and clustering support
- **Sub-millisecond latency**: Typical response times < 1ms

**Common use cases**:
- **Caching**: Database query results, API responses, session data
- **Session storage**: User sessions in web applications
- **Real-time analytics**: Counters, leaderboards, metrics
- **Rate limiting**: API throttling and abuse prevention
- **Message queues**: Pub/Sub messaging and task queues
- **Geospatial**: Location-based services and mapping

### Redis vs Other Solutions

| Feature | Redis | Memcached | Database |
|---------|-------|-----------|----------|
| Data Types | Rich (strings, lists, sets, etc.) | Key-value only | Tables/Documents |
| Persistence | Optional | No | Yes |
| Performance | Sub-ms | Sub-ms | 10-100ms |
| Memory Usage | Efficient | Very efficient | High |
| Clustering | Yes | Limited | Yes |
| Use Case | Cache + Data structures | Simple caching | Primary storage |

### Redis Data Types

#### 1. Strings

The most basic Redis data type - can store text, numbers, or binary data up to 512MB.

```bash
# Basic string operations
SET user:1000:name "John Doe"
GET user:1000:name
# Returns: "John Doe"

# Numeric operations
SET page:views 1000
INCR page:views        # Increment by 1
INCRBY page:views 5    # Increment by 5
DECR page:views        # Decrement by 1

# Expiration
SET session:abc123 "user_data" EX 3600  # Expires in 1 hour
SETEX cache:key 300 "cached_data"       # Set with 5-minute expiration

# Multiple operations
MSET user:1:name "John" user:1:email "john@example.com"
MGET user:1:name user:1:email
```

**Use cases**: Caching, counters, flags, configuration values

#### 2. Hashes

Perfect for representing objects - like a mini key-value store within a key.

```bash
# Hash operations
HSET user:1000 name "John Doe" email "john@example.com" age 30
HGET user:1000 name
# Returns: "John Doe"

HGETALL user:1000
# Returns: 1) "name" 2) "John Doe" 3) "email" 4) "john@example.com" 5) "age" 6) "30"

HMGET user:1000 name email  # Get multiple fields
HINCRBY user:1000 age 1     # Increment age by 1
HEXISTS user:1000 phone     # Check if field exists
HDEL user:1000 age          # Delete field
```

**Use cases**: User profiles, product details, configuration objects

#### 3. Lists

Ordered collections of strings - can act as stacks, queues, or arrays.

```bash
# List operations
LPUSH notifications "New message from Alice"
LPUSH notifications "System maintenance scheduled"
RPUSH notifications "Welcome to our platform!"

LRANGE notifications 0 -1   # Get all items
LPOP notifications          # Remove and return first item
RPOP notifications          # Remove and return last item

# Queue operations
LPUSH queue:tasks "process_payment"
BRPOP queue:tasks 10        # Blocking pop with 10-second timeout

# Recent items (keep only last 100)
LPUSH recent:views "product:123"
LTRIM recent:views 0 99
```

**Use cases**: Activity feeds, task queues, recent items, message queues

#### 4. Sets

Unordered collections of unique strings - perfect for membership testing.

```bash
# Set operations
SADD tags:post:123 "redis" "database" "caching"
SMEMBERS tags:post:123      # Get all members
SISMEMBER tags:post:123 "redis"  # Check membership (returns 1)

# Set operations between multiple sets
SADD users:online "user1" "user2" "user3"
SADD users:premium "user2" "user4"
SINTER users:online users:premium    # Intersection (user2)
SUNION users:online users:premium    # Union (user1, user2, user3, user4)
SDIFF users:online users:premium     # Difference (user1, user3)

# Random operations
SRANDMEMBER tags:post:123   # Get random member
SPOP tags:post:123          # Remove and return random member
```

**Use cases**: Tags, unique visitors, permissions, recommendations

#### 5. Sorted Sets (ZSets)

Sets ordered by score - combine the uniqueness of sets with ordering.

```bash
# Sorted set operations
ZADD leaderboard 1000 "player1" 1500 "player2" 800 "player3"
ZRANGE leaderboard 0 -1 WITHSCORES  # Get all with scores
ZREVRANGE leaderboard 0 2           # Top 3 players (highest scores)

# Score operations
ZINCRBY leaderboard 100 "player1"   # Increase player1's score by 100
ZSCORE leaderboard "player1"        # Get player1's score
ZRANK leaderboard "player1"         # Get player1's rank (0-based)

# Range queries
ZRANGEBYSCORE leaderboard 1000 2000 # Players with scores 1000-2000
ZCOUNT leaderboard 1000 2000        # Count players in score range

# Time-based sorted sets
ZADD recent:purchases 1640995200 "order:123" 1640995300 "order:124"
ZREMRANGEBYSCORE recent:purchases 0 1640991600  # Remove old entries
```

**Use cases**: Leaderboards, priority queues, time-series data, rate limiting

### Caching Strategies

#### 1. Cache-Aside (Lazy Loading)

Application manages the cache - most common pattern.

```python
def get_user(user_id):
    # Try cache first
    cache_key = f"user:{user_id}"
    user_data = redis.get(cache_key)
    
    if user_data:
        return json.loads(user_data)  # Cache hit
    
    # Cache miss - fetch from database
    user = database.get_user(user_id)
    
    # Store in cache for future requests
    redis.setex(cache_key, 3600, json.dumps(user))  # 1 hour TTL
    return user

def update_user(user_id, data):
    # Update database
    database.update_user(user_id, data)
    
    # Invalidate cache
    redis.delete(f"user:{user_id}")
```

**Pros**: Simple, cache only what's needed
**Cons**: Cache miss penalty, potential stale data

#### 2. Write-Through

Write to cache and database simultaneously.

```python
def update_user(user_id, data):
    # Update database
    database.update_user(user_id, data)
    
    # Update cache
    cache_key = f"user:{user_id}"
    redis.setex(cache_key, 3600, json.dumps(data))
```

**Pros**: Cache always fresh, no cache miss penalty
**Cons**: Write latency, cache may store unused data

#### 3. Write-Behind (Write-Back)

Write to cache immediately, database asynchronously.

```python
def update_user(user_id, data):
    # Update cache immediately
    cache_key = f"user:{user_id}"
    redis.setex(cache_key, 3600, json.dumps(data))
    
    # Queue database update
    redis.lpush("db_updates", json.dumps({
        "operation": "update_user",
        "user_id": user_id,
        "data": data
    }))

# Background worker processes db_updates queue
```

**Pros**: Fast writes, reduced database load
**Cons**: Risk of data loss, complexity

#### 4. Cache Patterns for Different Data

```python
# User sessions
def store_session(session_id, user_data):
    redis.setex(f"session:{session_id}", 1800, json.dumps(user_data))  # 30 min

# API response caching
def cache_api_response(endpoint, params, response):
    cache_key = f"api:{endpoint}:{hash(str(params))}"
    redis.setex(cache_key, 300, json.dumps(response))  # 5 min

# Database query result caching
def cache_query_result(sql, params, result):
    cache_key = f"query:{hash(sql + str(params))}"
    redis.setex(cache_key, 600, json.dumps(result))  # 10 min

# Computed values caching
def cache_expensive_calculation(input_data, result):
    cache_key = f"calc:{hash(str(input_data))}"
    redis.setex(cache_key, 3600, json.dumps(result))  # 1 hour
```

### Real-Time Applications with Redis

#### 1. Real-Time Analytics

```bash
# Page view counters
INCR page:views:/products
HINCRBY page:views:daily "2024-01-15" 1

# Unique visitors (using HyperLogLog for memory efficiency)
PFADD unique:visitors:2024-01-15 "user123" "user456"
PFCOUNT unique:visitors:2024-01-15  # Approximate unique count

# Real-time metrics
ZADD response_times 150 "api:users:get:2024-01-15:10:30:00"
ZREMRANGEBYSCORE response_times 0 (CURRENT_TIME - 3600)  # Keep last hour
```

#### 2. Leaderboards

```bash
# Gaming leaderboard
ZADD game:leaderboard 1500 "player1" 2000 "player2" 1200 "player3"

# Get top 10 players
ZREVRANGE game:leaderboard 0 9 WITHSCORES

# Get player's rank and score
ZREVRANK game:leaderboard "player1"  # Rank (0-based)
ZSCORE game:leaderboard "player1"    # Score

# Weekly leaderboard (with expiration)
ZADD weekly:leaderboard 1000 "player1"
EXPIRE weekly:leaderboard 604800  # 7 days
```

#### 3. Rate Limiting

```python
def rate_limit_user(user_id, limit=100, window=3600):
    """Allow 100 requests per hour per user"""
    key = f"rate_limit:{user_id}:{int(time.time() // window)}"
    
    current = redis.incr(key)
    if current == 1:
        redis.expire(key, window)
    
    return current <= limit

def sliding_window_rate_limit(user_id, limit=100, window=3600):
    """Sliding window rate limiting"""
    key = f"rate_limit:{user_id}"
    now = time.time()
    
    # Remove old entries
    redis.zremrangebyscore(key, 0, now - window)
    
    # Count current requests
    current = redis.zcard(key)
    
    if current < limit:
        # Add current request
        redis.zadd(key, {str(uuid.uuid4()): now})
        redis.expire(key, window)
        return True
    
    return False
```

#### 4. Session Management

```python
class RedisSessionManager:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.session_timeout = 1800  # 30 minutes
    
    def create_session(self, user_id, user_data):
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": time.time(),
            **user_data
        }
        
        key = f"session:{session_id}"
        self.redis.setex(key, self.session_timeout, json.dumps(session_data))
        return session_id
    
    def get_session(self, session_id):
        key = f"session:{session_id}"
        data = self.redis.get(key)
        
        if data:
            # Extend session on access
            self.redis.expire(key, self.session_timeout)
            return json.loads(data)
        return None
    
    def destroy_session(self, session_id):
        self.redis.delete(f"session:{session_id}")
```

### Redis Persistence

#### 1. RDB (Redis Database Backup)

Point-in-time snapshots of your dataset.

```bash
# Configuration
save 900 1      # Save if at least 1 key changed in 900 seconds
save 300 10     # Save if at least 10 keys changed in 300 seconds
save 60 10000   # Save if at least 10000 keys changed in 60 seconds

# Manual snapshot
BGSAVE          # Background save
SAVE            # Blocking save (avoid in production)
```

**Pros**: Compact files, fast restarts, good for backups
**Cons**: Data loss between snapshots, CPU intensive

#### 2. AOF (Append Only File)

Logs every write operation.

```bash
# Configuration
appendonly yes
appendfsync everysec    # Sync every second (recommended)
# appendfsync always    # Sync every command (slow but safe)
# appendfsync no        # Let OS decide (fast but risky)

# AOF rewrite (compact the log)
BGREWRITEAOF
```

**Pros**: Better durability, readable format
**Cons**: Larger files, slower restarts

#### 3. Hybrid Persistence

Combine RDB and AOF for best of both worlds.

```bash
# Use RDB for snapshots and AOF for recent changes
save 900 1
appendonly yes
aof-use-rdb-preamble yes
```

### Redis Performance Optimization

#### 1. Memory Optimization

```bash
# Memory usage analysis
MEMORY USAGE key_name
INFO memory

# Memory-efficient data structures
# Use hashes for small objects (< 512 fields)
HSET user:1000 name "John" email "john@example.com"

# Use sets for unique collections
SADD user:1000:tags "developer" "redis"

# Compress strings
SET large_json "compressed_json_data"
```

#### 2. Connection Pooling

```python
import redis.connection

# Connection pool for better performance
pool = redis.ConnectionPool(
    host='localhost',
    port=6379,
    max_connections=20,
    retry_on_timeout=True
)

redis_client = redis.Redis(connection_pool=pool)
```

#### 3. Pipeline Operations

```python
# Batch multiple operations
pipe = redis.pipeline()
pipe.set("key1", "value1")
pipe.set("key2", "value2")
pipe.incr("counter")
results = pipe.execute()  # Execute all at once
```

#### 4. Lua Scripts

```lua
-- Atomic rate limiting script
local key = KEYS[1]
local limit = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local current_time = tonumber(ARGV[3])

-- Remove old entries
redis.call('ZREMRANGEBYSCORE', key, 0, current_time - window)

-- Count current requests
local current = redis.call('ZCARD', key)

if current < limit then
    -- Add current request
    redis.call('ZADD', key, current_time, current_time)
    redis.call('EXPIRE', key, window)
    return 1
else
    return 0
end
```

```python
# Use the Lua script
rate_limit_script = redis.register_script(lua_script)
allowed = rate_limit_script(keys=['user:123'], args=[100, 3600, time.time()])
```

### Best Practices

#### 1. Key Naming Conventions

```bash
# Use consistent, hierarchical naming
user:1000:profile
user:1000:sessions
product:456:details
cache:api:users:list
queue:email:notifications

# Include version in keys for schema changes
user:v2:1000:profile
```

#### 2. Expiration Strategies

```python
# Set appropriate TTLs
redis.setex("cache:user:1000", 3600, user_data)      # 1 hour
redis.setex("session:abc123", 1800, session_data)    # 30 minutes
redis.setex("rate_limit:user:1000", 60, "1")         # 1 minute

# Use EXPIRE for existing keys
redis.expire("existing_key", 300)  # 5 minutes from now
```

#### 3. Memory Management

```python
# Monitor memory usage
info = redis.info('memory')
used_memory = info['used_memory_human']

# Set memory policies
# maxmemory 2gb
# maxmemory-policy allkeys-lru  # Evict least recently used keys
```

#### 4. Error Handling

```python
import redis
from redis.exceptions import ConnectionError, TimeoutError

def safe_redis_operation(operation):
    try:
        return operation()
    except ConnectionError:
        # Handle connection issues
        logger.error("Redis connection failed")
        return None
    except TimeoutError:
        # Handle timeouts
        logger.error("Redis operation timed out")
        return None
    except Exception as e:
        logger.error(f"Redis error: {e}")
        return None
```

### Redis in Production

#### 1. High Availability

```bash
# Master-Slave replication
# On slave: SLAVEOF master_host master_port
SLAVEOF 192.168.1.100 6379

# Redis Sentinel for automatic failover
sentinel monitor mymaster 192.168.1.100 6379 2
sentinel down-after-milliseconds mymaster 5000
sentinel failover-timeout mymaster 10000
```

#### 2. Clustering

```bash
# Redis Cluster for horizontal scaling
redis-cli --cluster create \
  192.168.1.100:7000 192.168.1.100:7001 \
  192.168.1.101:7000 192.168.1.101:7001 \
  192.168.1.102:7000 192.168.1.102:7001 \
  --cluster-replicas 1
```

#### 3. Monitoring

```bash
# Key metrics to monitor
INFO stats          # Operations per second
INFO memory         # Memory usage
INFO replication    # Replication status
INFO persistence    # RDB/AOF status

# Slow query log
SLOWLOG GET 10      # Get last 10 slow queries
CONFIG SET slowlog-log-slower-than 10000  # Log queries > 10ms
```

---

## 💻 Hands-On Exercise

Build a complete caching and real-time system with Redis.

**What you'll create**:
1. Multi-level caching system
2. User session management
3. Real-time leaderboard
4. Rate limiting system
5. Real-time analytics dashboard

**Skills practiced**:
- Different Redis data types
- Caching strategies
- Session management
- Real-time features
- Performance optimization

See `exercise.py` for hands-on practice.

---

## 📚 Resources

- [Redis Documentation](https://redis.io/documentation)
- [Redis Commands Reference](https://redis.io/commands)
- [Redis Best Practices](https://redis.io/topics/memory-optimization)
- [Redis University](https://university.redis.com/) - Free courses
- [Redis Patterns](https://redis.io/topics/data-types-intro)

---

## 🎯 Key Takeaways

- **Redis is in-memory** - ultra-fast but requires memory management
- **Rich data structures** - strings, hashes, lists, sets, sorted sets
- **Perfect for caching** - implement cache-aside, write-through, or write-behind
- **Real-time applications** - leaderboards, analytics, rate limiting
- **Session management** - fast, scalable user sessions
- **Atomic operations** - all Redis operations are atomic
- **Persistence options** - RDB snapshots, AOF logs, or hybrid
- **Production ready** - clustering, replication, monitoring
- **Key naming matters** - use consistent, hierarchical conventions
- **Set appropriate TTLs** - prevent memory leaks and stale data

---

## 🚀 What's Next?

Tomorrow (Day 4), you'll learn **Snowflake Data Warehouse** - a cloud-native data warehouse that separates compute and storage, perfect for analytics workloads and data science.

**Preview**: Snowflake complements Redis perfectly - use Redis for real-time, low-latency operations and Snowflake for complex analytics, data warehousing, and business intelligence on large datasets.

---

## ✅ Before Moving On

- [ ] Understand Redis data types and when to use each
- [ ] Can implement different caching strategies
- [ ] Know how to build real-time features with Redis
- [ ] Understand session management and rate limiting
- [ ] Can optimize Redis for performance and memory
- [ ] Complete the exercise in `exercise.py`
- [ ] Review the solution in `solution.py`
- [ ] Take the quiz in `quiz.md`

**Time**: ~60 minutes | **Difficulty**: ⭐⭐⭐ (Intermediate)

Ready to build lightning-fast applications! ⚡