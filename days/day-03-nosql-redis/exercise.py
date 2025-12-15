# Day 3: NoSQL Databases - Redis for Caching
# Exercise: Build a Complete Caching and Real-Time System

import redis
import json
import time
import uuid
import hashlib
from datetime import datetime, timedelta

# ============================================================================
# SETUP: Connect to Redis
# ============================================================================

# Connect to Redis (assumes Redis is running locally on default port 6379)
# In production, use connection pooling and proper configuration
try:
    r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    r.ping()  # Test connection
    print("✅ Connected to Redis successfully!")
    
    # Display Redis info for verification
    info = r.info()
    print(f"📊 Redis Info:")
    print(f"   Version: {info.get('redis_version', 'Unknown')}")
    print(f"   Memory used: {info.get('used_memory_human', 'Unknown')}")
    print(f"   Connected clients: {info.get('connected_clients', 'Unknown')}")
    
except redis.ConnectionError:
    print("❌ Could not connect to Redis. Make sure Redis is running.")
    print("   Install Redis: https://redis.io/download")
    print("   Start Redis: redis-server")
    print("   Or using Docker: docker run -d -p 6379:6379 redis:latest")
    exit(1)
except Exception as e:
    print(f"❌ Redis connection error: {e}")
    exit(1)

# ============================================================================
# EXERCISE 1: Basic Redis operations with different data types
# ============================================================================

def exercise_1_basic_operations():
    """Practice basic Redis operations with different data types"""
    print("\n=== Exercise 1: Basic Redis Operations ===")
    
    # TODO: String operations
    # 1. Store a user's name and retrieve it
    #    Hint: r.set("user:1000:name", "John Doe") and r.get("user:1000:name")
    # 2. Implement a page view counter that increments
    #    Hint: r.incr("page:views:/home")
    # 3. Store a JSON object as a string with expiration
    #    Hint: r.setex("cache:user:1000", 300, json.dumps(data))
    
    # TODO: Hash operations  
    # 1. Store user profile data (name, email, age) in a hash
    #    Hint: r.hset("user:1000:profile", mapping={"name": "John", "email": "john@example.com"})
    # 2. Update specific fields in the hash
    #    Hint: r.hset("user:1000:profile", "age", "31")
    # 3. Get all fields from the hash
    #    Hint: r.hgetall("user:1000:profile")
    
    # TODO: List operations
    # 1. Create a notification queue using lists
    #    Hint: r.lpush("notifications:user:1000", "Welcome message")
    # 2. Add notifications to the queue
    #    Hint: r.lpush("notifications:user:1000", "New notification")
    # 3. Process notifications from the queue
    #    Hint: r.rpop("notifications:user:1000") for FIFO processing
    
    # TODO: Set operations
    # 1. Store user tags in a set
    #    Hint: r.sadd("user:1000:tags", "developer", "python", "redis")
    # 2. Add and remove tags
    #    Hint: r.sadd("user:1000:tags", "new_tag") and r.srem("user:1000:tags", "old_tag")
    # 3. Check if a tag exists
    #    Hint: r.sismember("user:1000:tags", "developer")
    # 4. Find common tags between two users
    #    Hint: r.sinter("user:1000:tags", "user:2000:tags")
    
    # TODO: Sorted Set operations
    # 1. Create a simple leaderboard with player scores
    #    Hint: r.zadd("leaderboard:game1", {"player1": 1000, "player2": 1500})
    # 2. Add players with scores
    #    Hint: r.zadd("leaderboard:game1", {"player3": 800})
    # 3. Get top 3 players
    #    Hint: r.zrevrange("leaderboard:game1", 0, 2, withscores=True)
    # 4. Get a player's rank and score
    #    Hint: r.zrevrank("leaderboard:game1", "player1") and r.zscore("leaderboard:game1", "player1")
    
    pass  # Remove this and implement the above

# ============================================================================
# EXERCISE 2: Implement caching strategies
# ============================================================================

class DatabaseSimulator:
    """Simulates a slow database for caching exercises"""
    
    def __init__(self):
        self.users = {
            1: {"name": "John Doe", "email": "john@example.com", "age": 30},
            2: {"name": "Jane Smith", "email": "jane@example.com", "age": 25},
            3: {"name": "Bob Johnson", "email": "bob@example.com", "age": 35}
        }
        self.call_count = 0
    
    def get_user(self, user_id):
        """Simulate slow database query"""
        self.call_count += 1
        time.sleep(0.1)  # Simulate 100ms database query
        return self.users.get(user_id)
    
    def update_user(self, user_id, data):
        """Simulate database update"""
        time.sleep(0.05)  # Simulate 50ms database update
        if user_id in self.users:
            self.users[user_id].update(data)
            return True
        return False

# Initialize database simulator
db = DatabaseSimulator()

def exercise_2_cache_aside():
    """Implement cache-aside (lazy loading) pattern"""
    print("\n=== Exercise 2: Cache-Aside Pattern ===")
    
    def get_user_cached(user_id):
        """TODO: Implement cache-aside pattern
        1. Check Redis cache first
        2. If cache miss, fetch from database
        3. Store result in cache with TTL
        4. Return user data
        """
        # Your implementation here
        pass
    
    def update_user_cached(user_id, data):
        """TODO: Implement cache invalidation
        1. Update database
        2. Invalidate cache entry
        """
        # Your implementation here
        pass
    
    # TODO: Test the caching implementation
    # 1. Call get_user_cached multiple times for same user
    # 2. Measure performance difference
    # 3. Update user and verify cache invalidation
    
    pass  # Remove this and implement the above

def exercise_2_write_through():
    """Implement write-through caching pattern"""
    print("\n=== Exercise 2b: Write-Through Pattern ===")
    
    def update_user_write_through(user_id, data):
        """TODO: Implement write-through pattern
        1. Update database
        2. Update cache simultaneously
        """
        # Your implementation here
        pass
    
    # TODO: Test write-through implementation
    pass

# ============================================================================
# EXERCISE 3: Session management system
# ============================================================================

class SessionManager:
    """Redis-based session management"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.session_timeout = 1800  # 30 minutes
    
    def create_session(self, user_id, user_data=None):
        """TODO: Create a new session
        1. Generate unique session ID
        #    Hint: session_id = str(uuid.uuid4())
        2. Store session data in Redis with expiration
        #    Hint: r.setex(f"session:{session_id}", 1800, json.dumps(data))
        3. Return session ID
        """
        # Your implementation here
        pass
    
    def get_session(self, session_id):
        """TODO: Retrieve session data
        1. Get session data from Redis
        2. Extend session expiration on access
        3. Return session data or None if expired
        """
        # Your implementation here
        pass
    
    def update_session(self, session_id, data):
        """TODO: Update session data
        1. Update existing session data
        2. Maintain expiration time
        """
        # Your implementation here
        pass
    
    def destroy_session(self, session_id):
        """TODO: Destroy session
        1. Remove session from Redis
        """
        # Your implementation here
        pass
    
    def get_active_sessions_count(self):
        """TODO: Get count of active sessions
        1. Count all session keys
        """
        # Your implementation here
        pass

def exercise_3_session_management():
    """Test session management system"""
    print("\n=== Exercise 3: Session Management ===")
    
    session_manager = SessionManager(r)
    
    # TODO: Test session management
    # 1. Create sessions for multiple users
    # 2. Retrieve and update session data
    # 3. Test session expiration
    # 4. Count active sessions
    # 5. Destroy sessions
    
    pass

# ============================================================================
# EXERCISE 4: Real-time leaderboard system
# ============================================================================

class Leaderboard:
    """Redis-based leaderboard using sorted sets"""
    
    def __init__(self, redis_client, name):
        self.redis = redis_client
        self.key = f"leaderboard:{name}"
    
    def add_score(self, player, score):
        """TODO: Add or update player score
        1. Use ZADD to add player with score
        #    Hint: self.redis.zadd(self.key, {player: score})
        2. Handle score updates automatically
        #    Note: ZADD automatically updates existing members
        """
        # Your implementation here
        pass
    
    def increment_score(self, player, increment):
        """TODO: Increment player's score
        1. Use ZINCRBY to increment score
        """
        # Your implementation here
        pass
    
    def get_top_players(self, count=10):
        """TODO: Get top N players
        1. Use ZREVRANGE to get highest scores
        2. Include scores in result
        """
        # Your implementation here
        pass
    
    def get_player_rank(self, player):
        """TODO: Get player's rank (1-based)
        1. Use ZREVRANK to get rank
        2. Convert to 1-based ranking
        """
        # Your implementation here
        pass
    
    def get_player_score(self, player):
        """TODO: Get player's current score
        1. Use ZSCORE to get score
        """
        # Your implementation here
        pass
    
    def get_players_in_range(self, start_rank, end_rank):
        """TODO: Get players in rank range
        1. Use ZREVRANGE with start and end positions
        """
        # Your implementation here
        pass

def exercise_4_leaderboard():
    """Test leaderboard system"""
    print("\n=== Exercise 4: Real-time Leaderboard ===")
    
    leaderboard = Leaderboard(r, "game_scores")
    
    # TODO: Test leaderboard functionality
    # 1. Add multiple players with scores
    # 2. Update scores and verify rankings
    # 3. Get top players
    # 4. Get specific player's rank and score
    # 5. Simulate real-time score updates
    
    pass

# ============================================================================
# EXERCISE 5: Rate limiting system
# ============================================================================

class RateLimiter:
    """Redis-based rate limiting"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def fixed_window_limit(self, key, limit, window_seconds):
        """TODO: Implement fixed window rate limiting
        1. Create time-based key (e.g., user:123:2024-01-15:10)
        #    Hint: window_key = f"rate_limit:{key}:{int(time.time() // window_seconds)}"
        2. Increment counter
        #    Hint: current_count = self.redis.incr(window_key)
        3. Set expiration on first increment
        #    Hint: if current_count == 1: self.redis.expire(window_key, window_seconds)
        4. Check if limit exceeded
        #    Hint: return current_count <= limit
        """
        # Your implementation here
        pass
    
    def sliding_window_limit(self, key, limit, window_seconds):
        """TODO: Implement sliding window rate limiting
        1. Use sorted set with timestamps as scores
        2. Remove old entries outside window
        3. Count current entries
        4. Add new entry if under limit
        """
        # Your implementation here
        pass
    
    def token_bucket_limit(self, key, capacity, refill_rate, refill_period):
        """TODO: Implement token bucket rate limiting
        1. Store current tokens and last refill time
        2. Calculate tokens to add based on time elapsed
        3. Check if enough tokens available
        4. Consume token if available
        """
        # Your implementation here
        pass

def exercise_5_rate_limiting():
    """Test rate limiting systems"""
    print("\n=== Exercise 5: Rate Limiting ===")
    
    rate_limiter = RateLimiter(r)
    
    # TODO: Test different rate limiting strategies
    # 1. Test fixed window limiting
    # 2. Test sliding window limiting  
    # 3. Test token bucket limiting
    # 4. Simulate burst traffic and verify limits
    
    pass

# ============================================================================
# EXERCISE 6: Real-time analytics
# ============================================================================

class Analytics:
    """Redis-based real-time analytics"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def track_page_view(self, page, user_id=None):
        """TODO: Track page views
        1. Increment total page views
        2. Increment daily page views
        3. Track unique visitors using HyperLogLog
        4. Store recent page views in a list
        """
        # Your implementation here
        pass
    
    def track_user_action(self, user_id, action, metadata=None):
        """TODO: Track user actions
        1. Store action in user's activity stream
        2. Increment action counters
        3. Update user's last activity time
        """
        # Your implementation here
        pass
    
    def get_page_stats(self, page):
        """TODO: Get page statistics
        1. Get total views
        2. Get today's views
        3. Get unique visitors count
        4. Get recent views
        """
        # Your implementation here
        pass
    
    def get_top_pages(self, count=10):
        """TODO: Get most viewed pages
        1. Use sorted set to track page view counts
        2. Return top N pages
        """
        # Your implementation here
        pass
    
    def get_user_activity(self, user_id, count=10):
        """TODO: Get user's recent activity
        1. Get recent actions from user's activity stream
        """
        # Your implementation here
        pass

def exercise_6_analytics():
    """Test real-time analytics system"""
    print("\n=== Exercise 6: Real-time Analytics ===")
    
    analytics = Analytics(r)
    
    # TODO: Test analytics functionality
    # 1. Track page views for different pages
    # 2. Track user actions
    # 3. Get page statistics
    # 4. Get top pages
    # 5. Get user activity
    
    pass

# ============================================================================
# EXERCISE 7: Advanced Redis patterns
# ============================================================================

def exercise_7_lua_scripts():
    """Use Lua scripts for atomic operations"""
    print("\n=== Exercise 7: Lua Scripts ===")
    
    # TODO: Implement atomic rate limiting with Lua script
    rate_limit_script = """
    -- TODO: Write Lua script for atomic rate limiting
    -- 1. Remove old entries from sorted set
    -- 2. Count current entries
    -- 3. Add new entry if under limit
    -- 4. Return 1 if allowed, 0 if rate limited
    """
    
    # TODO: Register and use the Lua script
    pass

def exercise_7_pub_sub():
    """Implement pub/sub messaging"""
    print("\n=== Exercise 7b: Pub/Sub Messaging ===")
    
    # TODO: Implement pub/sub system
    # 1. Create publisher function
    # 2. Create subscriber function
    # 3. Test message publishing and receiving
    
    pass

def exercise_7_distributed_lock():
    """Implement distributed locking"""
    print("\n=== Exercise 7c: Distributed Lock ===")
    
    def acquire_lock(lock_name, timeout=10):
        """TODO: Acquire distributed lock
        1. Use SET with NX and EX options
        2. Return lock identifier if successful
        """
        # Your implementation here
        pass
    
    def release_lock(lock_name, lock_id):
        """TODO: Release distributed lock
        1. Use Lua script to check lock ownership
        2. Delete lock only if owned by caller
        """
        # Your implementation here
        pass
    
    # TODO: Test distributed locking
    pass

# ============================================================================
# EXERCISE 8: Performance optimization
# ============================================================================

def exercise_8_performance():
    """Test Redis performance optimization techniques"""
    print("\n=== Exercise 8: Performance Optimization ===")
    
    def test_pipeline_performance():
        """TODO: Compare single operations vs pipeline
        1. Time 1000 individual SET operations
        2. Time 1000 SET operations in pipeline
        3. Compare performance
        """
        # Your implementation here
        pass
    
    def test_memory_usage():
        """TODO: Analyze memory usage
        1. Check memory usage of different data structures
        2. Compare hash vs individual keys
        """
        # Your implementation here
        pass
    
    def test_connection_pooling():
        """TODO: Test connection pooling benefits
        1. Create connection pool
        2. Compare performance with/without pooling
        """
        # Your implementation here
        pass
    
    # TODO: Run performance tests
    pass

# ============================================================================
# EXERCISE 9: Production patterns
# ============================================================================

def exercise_9_production_patterns():
    """Implement production-ready patterns"""
    print("\n=== Exercise 9: Production Patterns ===")
    
    def implement_circuit_breaker():
        """TODO: Implement circuit breaker pattern
        1. Track Redis operation failures
        2. Open circuit after threshold failures
        3. Periodically test if Redis is back
        """
        # Your implementation here
        pass
    
    def implement_fallback_cache():
        """TODO: Implement fallback caching
        1. Try Redis first
        2. Fall back to in-memory cache if Redis fails
        3. Fall back to database if both fail
        """
        # Your implementation here
        pass
    
    def implement_cache_warming():
        """TODO: Implement cache warming
        1. Pre-populate cache with frequently accessed data
        2. Refresh cache before expiration
        """
        # Your implementation here
        pass
    
    # TODO: Test production patterns
    pass

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all exercises"""
    print("🚀 Starting Redis Exercises")
    print("=" * 50)
    
    # Clear Redis for clean start (be careful in production!)
    r.flushdb()
    print("🧹 Cleared Redis database for clean start")
    
    try:
        # Run exercises
        exercise_1_basic_operations()
        exercise_2_cache_aside()
        exercise_2_write_through()
        exercise_3_session_management()
        exercise_4_leaderboard()
        exercise_5_rate_limiting()
        exercise_6_analytics()
        exercise_7_lua_scripts()
        exercise_7_pub_sub()
        exercise_7_distributed_lock()
        exercise_8_performance()
        exercise_9_production_patterns()
        
        print("\n🎉 All exercises completed!")
        
    except Exception as e:
        print(f"\n❌ Error during exercises: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

# ============================================================================
# BONUS CHALLENGES
# ============================================================================

"""
BONUS CHALLENGES (try these after completing main exercises):

1. Build a Redis-based job queue system
   - Add jobs to queue
   - Process jobs with workers
   - Handle job failures and retries

2. Implement a Redis-based chat system
   - Store chat messages
   - Track online users
   - Implement chat rooms

3. Create a Redis-based recommendation engine
   - Track user preferences
   - Calculate similarity scores
   - Generate recommendations

4. Build a Redis-based inventory system
   - Track product quantities
   - Handle concurrent updates
   - Implement low-stock alerts

5. Implement Redis-based A/B testing
   - Assign users to test groups
   - Track experiment results
   - Calculate statistical significance
"""