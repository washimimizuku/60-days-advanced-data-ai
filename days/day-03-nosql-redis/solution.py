# Day 3: NoSQL Databases - Redis for Caching
# SOLUTION: Build a Complete Caching and Real-Time System

import redis
import json
import time
import uuid
import hashlib
from datetime import datetime, timedelta

# ============================================================================
# SETUP: Connect to Redis
# ============================================================================

# Connect to Redis with connection pooling for better performance
pool = redis.ConnectionPool(host='localhost', port=6379, db=0, max_connections=20)
r = redis.Redis(connection_pool=pool, decode_responses=True)

try:
    r.ping()
    print("✅ Connected to Redis successfully!")
except redis.ConnectionError:
    print("❌ Could not connect to Redis. Make sure Redis is running.")
    exit(1)

# ============================================================================
# EXERCISE 1: Basic Redis operations with different data types
# ============================================================================

def exercise_1_basic_operations():
    """Practice basic Redis operations with different data types"""
    print("\n=== Exercise 1: Basic Redis Operations ===")
    
    # Solution 1: String operations
    print("1. String Operations:")
    
    # Store and retrieve user name
    r.set("user:1000:name", "John Doe")
    name = r.get("user:1000:name")
    print(f"   User name: {name}")
    
    # Page view counter
    r.set("page:views:/home", 0)
    r.incr("page:views:/home")
    r.incr("page:views:/home")
    views = r.get("page:views:/home")
    print(f"   Page views: {views}")
    
    # JSON object with expiration
    user_data = {"name": "John", "email": "john@example.com", "age": 30}
    r.setex("cache:user:1000", 300, json.dumps(user_data))  # 5 minutes
    cached_data = json.loads(r.get("cache:user:1000"))
    print(f"   Cached user: {cached_data}")
    
    # Solution 2: Hash operations
    print("\n2. Hash Operations:")
    
    # Store user profile in hash
    r.hset("user:1000:profile", mapping={
        "name": "John Doe",
        "email": "john@example.com", 
        "age": "30"
    })
    
    # Update specific field
    r.hset("user:1000:profile", "age", "31")
    
    # Get all fields
    profile = r.hgetall("user:1000:profile")
    print(f"   User profile: {profile}")
    
    # Get specific fields
    name_email = r.hmget("user:1000:profile", "name", "email")
    print(f"   Name and email: {name_email}")
    
    # Solution 3: List operations
    print("\n3. List Operations:")
    
    # Notification queue
    r.lpush("notifications:user:1000", "Welcome to our platform!")
    r.lpush("notifications:user:1000", "New message from Alice")
    r.lpush("notifications:user:1000", "System maintenance scheduled")
    
    # Process notifications (FIFO queue)
    notification = r.rpop("notifications:user:1000")
    print(f"   Processed notification: {notification}")
    
    # Get all remaining notifications
    remaining = r.lrange("notifications:user:1000", 0, -1)
    print(f"   Remaining notifications: {remaining}")
    
    # Solution 4: Set operations
    print("\n4. Set Operations:")
    
    # User tags
    r.sadd("user:1000:tags", "developer", "python", "redis")
    r.sadd("user:2000:tags", "developer", "javascript", "nodejs")
    
    # Check membership
    is_developer = r.sismember("user:1000:tags", "developer")
    print(f"   User 1000 is developer: {is_developer}")
    
    # Common tags between users
    common_tags = r.sinter("user:1000:tags", "user:2000:tags")
    print(f"   Common tags: {list(common_tags)}")
    
    # All tags from both users
    all_tags = r.sunion("user:1000:tags", "user:2000:tags")
    print(f"   All tags: {list(all_tags)}")
    
    # Solution 5: Sorted Set operations
    print("\n5. Sorted Set Operations:")
    
    # Simple leaderboard
    r.zadd("leaderboard:game1", {
        "player1": 1000,
        "player2": 1500,
        "player3": 800,
        "player4": 2000
    })
    
    # Get top 3 players
    top_players = r.zrevrange("leaderboard:game1", 0, 2, withscores=True)
    print(f"   Top 3 players: {top_players}")
    
    # Get player's rank (0-based, so add 1 for 1-based)
    rank = r.zrevrank("leaderboard:game1", "player2")
    print(f"   Player2's rank: {rank + 1 if rank is not None else 'Not found'}")
    
    # Get player's score
    score = r.zscore("leaderboard:game1", "player2")
    print(f"   Player2's score: {score}")

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
        """Solution: Cache-aside pattern implementation"""
        cache_key = f"user:{user_id}"
        
        # Try cache first
        cached_data = r.get(cache_key)
        if cached_data:
            print(f"   Cache HIT for user {user_id}")
            return json.loads(cached_data)
        
        # Cache miss - fetch from database
        print(f"   Cache MISS for user {user_id} - fetching from database")
        user_data = db.get_user(user_id)
        
        if user_data:
            # Store in cache with 1 hour TTL
            r.setex(cache_key, 3600, json.dumps(user_data))
        
        return user_data
    
    def update_user_cached(user_id, data):
        """Solution: Cache invalidation on update"""
        # Update database
        success = db.update_user(user_id, data)
        
        if success:
            # Invalidate cache
            cache_key = f"user:{user_id}"
            r.delete(cache_key)
            print(f"   Cache invalidated for user {user_id}")
        
        return success
    
    # Test the caching implementation
    print("Testing cache-aside pattern:")
    
    # First call - cache miss
    start_time = time.time()
    user1 = get_user_cached(1)
    first_call_time = time.time() - start_time
    print(f"   First call took {first_call_time:.3f}s: {user1}")
    
    # Second call - cache hit
    start_time = time.time()
    user1_cached = get_user_cached(1)
    second_call_time = time.time() - start_time
    print(f"   Second call took {second_call_time:.3f}s: {user1_cached}")
    
    print(f"   Performance improvement: {first_call_time/second_call_time:.1f}x faster")
    
    # Test cache invalidation
    print("\nTesting cache invalidation:")
    update_user_cached(1, {"age": 31})
    
    # This should be a cache miss again
    start_time = time.time()
    user1_updated = get_user_cached(1)
    third_call_time = time.time() - start_time
    print(f"   After update call took {third_call_time:.3f}s: {user1_updated}")
    
    print(f"   Database calls made: {db.call_count}")

def exercise_2_write_through():
    """Implement write-through caching pattern"""
    print("\n=== Exercise 2b: Write-Through Pattern ===")
    
    def update_user_write_through(user_id, data):
        """Solution: Write-through pattern implementation"""
        # Update database
        success = db.update_user(user_id, data)
        
        if success:
            # Update cache simultaneously
            cache_key = f"user:{user_id}"
            updated_user = db.users[user_id]  # Get updated data
            r.setex(cache_key, 3600, json.dumps(updated_user))
            print(f"   Database and cache updated for user {user_id}")
        
        return success
    
    # Test write-through implementation
    print("Testing write-through pattern:")
    
    # Update user with write-through
    update_user_write_through(2, {"age": 26})
    
    # Verify cache has updated data
    cached_data = r.get("user:2")
    if cached_data:
        user_data = json.loads(cached_data)
        print(f"   Cache contains updated data: {user_data}")
    else:
        print("   No data in cache")

# ============================================================================
# EXERCISE 3: Session management system
# ============================================================================

class SessionManager:
    """Redis-based session management"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.session_timeout = 1800  # 30 minutes
    
    def create_session(self, user_id, user_data=None):
        """Solution: Create a new session"""
        session_id = str(uuid.uuid4())
        session_data = {
            "user_id": user_id,
            "created_at": time.time(),
            "last_accessed": time.time()
        }
        
        if user_data:
            session_data.update(user_data)
        
        session_key = f"session:{session_id}"
        self.redis.setex(session_key, self.session_timeout, json.dumps(session_data))
        
        return session_id
    
    def get_session(self, session_id):
        """Solution: Retrieve session data"""
        session_key = f"session:{session_id}"
        session_data = self.redis.get(session_key)
        
        if session_data:
            data = json.loads(session_data)
            # Update last accessed time
            data["last_accessed"] = time.time()
            
            # Extend session expiration
            self.redis.setex(session_key, self.session_timeout, json.dumps(data))
            
            return data
        
        return None
    
    def update_session(self, session_id, data):
        """Solution: Update session data"""
        session_key = f"session:{session_id}"
        existing_data = self.redis.get(session_key)
        
        if existing_data:
            session_data = json.loads(existing_data)
            session_data.update(data)
            session_data["last_accessed"] = time.time()
            
            # Get remaining TTL and maintain it
            ttl = self.redis.ttl(session_key)
            if ttl > 0:
                self.redis.setex(session_key, ttl, json.dumps(session_data))
                return True
        
        return False
    
    def destroy_session(self, session_id):
        """Solution: Destroy session"""
        session_key = f"session:{session_id}"
        return self.redis.delete(session_key) > 0
    
    def get_active_sessions_count(self):
        """Solution: Get count of active sessions"""
        session_keys = self.redis.keys("session:*")
        return len(session_keys)

def exercise_3_session_management():
    """Test session management system"""
    print("\n=== Exercise 3: Session Management ===")
    
    session_manager = SessionManager(r)
    
    # Test session creation
    print("1. Creating sessions:")
    session1 = session_manager.create_session(1, {"username": "john", "role": "admin"})
    session2 = session_manager.create_session(2, {"username": "jane", "role": "user"})
    print(f"   Created session 1: {session1[:8]}...")
    print(f"   Created session 2: {session2[:8]}...")
    
    # Test session retrieval
    print("\n2. Retrieving sessions:")
    data1 = session_manager.get_session(session1)
    print(f"   Session 1 data: {data1}")
    
    # Test session update
    print("\n3. Updating session:")
    session_manager.update_session(session1, {"last_page": "/dashboard"})
    updated_data = session_manager.get_session(session1)
    print(f"   Updated session 1: {updated_data}")
    
    # Test active sessions count
    print("\n4. Active sessions:")
    count = session_manager.get_active_sessions_count()
    print(f"   Active sessions count: {count}")
    
    # Test session destruction
    print("\n5. Destroying session:")
    destroyed = session_manager.destroy_session(session2)
    print(f"   Session 2 destroyed: {destroyed}")
    
    final_count = session_manager.get_active_sessions_count()
    print(f"   Active sessions after destruction: {final_count}")
    
    # Test session expiration (simulate)
    print("\n6. Testing session expiration:")
    # Create short-lived session for testing
    short_session = session_manager.create_session(3, {"test": True})
    r.expire(f"session:{short_session}", 2)  # Expire in 2 seconds
    
    print("   Waiting for session to expire...")
    time.sleep(3)
    
    expired_data = session_manager.get_session(short_session)
    print(f"   Expired session data: {expired_data}")

# ============================================================================
# EXERCISE 4: Real-time leaderboard system
# ============================================================================

class Leaderboard:
    """Redis-based leaderboard using sorted sets"""
    
    def __init__(self, redis_client, name):
        self.redis = redis_client
        self.key = f"leaderboard:{name}"
    
    def add_score(self, player, score):
        """Solution: Add or update player score"""
        return self.redis.zadd(self.key, {player: score})
    
    def increment_score(self, player, increment):
        """Solution: Increment player's score"""
        return self.redis.zincrby(self.key, increment, player)
    
    def get_top_players(self, count=10):
        """Solution: Get top N players"""
        return self.redis.zrevrange(self.key, 0, count-1, withscores=True)
    
    def get_player_rank(self, player):
        """Solution: Get player's rank (1-based)"""
        rank = self.redis.zrevrank(self.key, player)
        return rank + 1 if rank is not None else None
    
    def get_player_score(self, player):
        """Solution: Get player's current score"""
        return self.redis.zscore(self.key, player)
    
    def get_players_in_range(self, start_rank, end_rank):
        """Solution: Get players in rank range"""
        # Convert to 0-based indexing
        return self.redis.zrevrange(self.key, start_rank-1, end_rank-1, withscores=True)
    
    def get_total_players(self):
        """Get total number of players"""
        return self.redis.zcard(self.key)

def exercise_4_leaderboard():
    """Test leaderboard system"""
    print("\n=== Exercise 4: Real-time Leaderboard ===")
    
    leaderboard = Leaderboard(r, "game_scores")
    
    # Clear existing leaderboard
    r.delete("leaderboard:game_scores")
    
    # Test adding players with scores
    print("1. Adding players to leaderboard:")
    players_scores = [
        ("Alice", 1500),
        ("Bob", 1200),
        ("Charlie", 1800),
        ("Diana", 1350),
        ("Eve", 1600)
    ]
    
    for player, score in players_scores:
        leaderboard.add_score(player, score)
        print(f"   Added {player} with score {score}")
    
    # Test getting top players
    print("\n2. Top 3 players:")
    top_players = leaderboard.get_top_players(3)
    for i, (player, score) in enumerate(top_players, 1):
        print(f"   {i}. {player}: {int(score)}")
    
    # Test getting player rank and score
    print("\n3. Player stats:")
    for player in ["Alice", "Bob", "Charlie"]:
        rank = leaderboard.get_player_rank(player)
        score = leaderboard.get_player_score(player)
        print(f"   {player}: Rank {rank}, Score {int(score) if score else 'N/A'}")
    
    # Test score updates
    print("\n4. Updating scores:")
    leaderboard.increment_score("Bob", 400)  # Bob gets 400 more points
    print("   Bob gained 400 points!")
    
    new_rank = leaderboard.get_player_rank("Bob")
    new_score = leaderboard.get_player_score("Bob")
    print(f"   Bob's new rank: {new_rank}, new score: {int(new_score)}")
    
    # Test range queries
    print("\n5. Players ranked 2-4:")
    middle_players = leaderboard.get_players_in_range(2, 4)
    for i, (player, score) in enumerate(middle_players, 2):
        print(f"   {i}. {player}: {int(score)}")
    
    # Simulate real-time updates
    print("\n6. Simulating real-time score updates:")
    import random
    
    for _ in range(5):
        player = random.choice(["Alice", "Bob", "Charlie", "Diana", "Eve"])
        increment = random.randint(10, 100)
        new_score = leaderboard.increment_score(player, increment)
        rank = leaderboard.get_player_rank(player)
        print(f"   {player} +{increment} points → Rank {rank}, Score {int(new_score)}")
        time.sleep(0.5)
    
    print("\n   Final leaderboard:")
    final_top = leaderboard.get_top_players(5)
    for i, (player, score) in enumerate(final_top, 1):
        print(f"   {i}. {player}: {int(score)}")

# ============================================================================
# EXERCISE 5: Rate limiting system
# ============================================================================

class RateLimiter:
    """Redis-based rate limiting"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def fixed_window_limit(self, key, limit, window_seconds):
        """Solution: Fixed window rate limiting"""
        # Create time-based key
        current_window = int(time.time() // window_seconds)
        window_key = f"rate_limit:{key}:{current_window}"
        
        # Increment counter
        current_count = self.redis.incr(window_key)
        
        # Set expiration on first increment
        if current_count == 1:
            self.redis.expire(window_key, window_seconds)
        
        return current_count <= limit
    
    def sliding_window_limit(self, key, limit, window_seconds):
        """Solution: Sliding window rate limiting"""
        now = time.time()
        window_start = now - window_seconds
        
        # Remove old entries
        self.redis.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
        
        # Count current entries
        current_count = self.redis.zcard(f"rate_limit:{key}")
        
        if current_count < limit:
            # Add new entry
            self.redis.zadd(f"rate_limit:{key}", {str(uuid.uuid4()): now})
            # Set expiration
            self.redis.expire(f"rate_limit:{key}", window_seconds)
            return True
        
        return False
    
    def token_bucket_limit(self, key, capacity, refill_rate, refill_period):
        """Solution: Token bucket rate limiting"""
        bucket_key = f"token_bucket:{key}"
        
        # Get current bucket state
        bucket_data = self.redis.hmget(bucket_key, "tokens", "last_refill")
        
        current_tokens = float(bucket_data[0]) if bucket_data[0] else capacity
        last_refill = float(bucket_data[1]) if bucket_data[1] else time.time()
        
        # Calculate tokens to add
        now = time.time()
        time_passed = now - last_refill
        tokens_to_add = (time_passed / refill_period) * refill_rate
        
        # Update token count (don't exceed capacity)
        current_tokens = min(capacity, current_tokens + tokens_to_add)
        
        if current_tokens >= 1:
            # Consume one token
            current_tokens -= 1
            
            # Update bucket state
            self.redis.hmset(bucket_key, {
                "tokens": current_tokens,
                "last_refill": now
            })
            self.redis.expire(bucket_key, refill_period * 2)  # Keep bucket alive
            
            return True
        else:
            # Update last refill time even if no token consumed
            self.redis.hmset(bucket_key, {
                "tokens": current_tokens,
                "last_refill": now
            })
            self.redis.expire(bucket_key, refill_period * 2)
            
            return False

def exercise_5_rate_limiting():
    """Test rate limiting systems"""
    print("\n=== Exercise 5: Rate Limiting ===")
    
    rate_limiter = RateLimiter(r)
    
    # Test fixed window limiting
    print("1. Testing fixed window rate limiting (5 requests per 10 seconds):")
    
    for i in range(8):
        allowed = rate_limiter.fixed_window_limit("user:123", 5, 10)
        status = "✅ ALLOWED" if allowed else "❌ RATE LIMITED"
        print(f"   Request {i+1}: {status}")
        time.sleep(0.5)
    
    # Test sliding window limiting
    print("\n2. Testing sliding window rate limiting (3 requests per 5 seconds):")
    
    # Clear any existing data
    r.delete("rate_limit:user:456")
    
    for i in range(6):
        allowed = rate_limiter.sliding_window_limit("user:456", 3, 5)
        status = "✅ ALLOWED" if allowed else "❌ RATE LIMITED"
        print(f"   Request {i+1}: {status}")
        time.sleep(1)
    
    # Test token bucket limiting
    print("\n3. Testing token bucket rate limiting (capacity=3, refill=1 token/2 seconds):")
    
    # Clear any existing bucket
    r.delete("token_bucket:user:789")
    
    # Initial burst (should allow 3 requests)
    print("   Initial burst:")
    for i in range(5):
        allowed = rate_limiter.token_bucket_limit("user:789", 3, 1, 2)
        status = "✅ ALLOWED" if allowed else "❌ RATE LIMITED"
        print(f"     Request {i+1}: {status}")
    
    # Wait for token refill
    print("   Waiting 4 seconds for token refill...")
    time.sleep(4)
    
    # Should allow 2 more requests (2 tokens refilled)
    print("   After refill:")
    for i in range(3):
        allowed = rate_limiter.token_bucket_limit("user:789", 3, 1, 2)
        status = "✅ ALLOWED" if allowed else "❌ RATE LIMITED"
        print(f"     Request {i+1}: {status}")

# ============================================================================
# EXERCISE 6: Real-time analytics
# ============================================================================

class Analytics:
    """Redis-based real-time analytics"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
    
    def track_page_view(self, page, user_id=None):
        """Solution: Track page views"""
        now = datetime.now()
        today = now.strftime("%Y-%m-%d")
        
        # Increment total page views
        self.redis.incr(f"page_views:{page}")
        
        # Increment daily page views
        self.redis.hincrby(f"page_views:daily:{page}", today, 1)
        
        # Track unique visitors using HyperLogLog
        if user_id:
            self.redis.pfadd(f"unique_visitors:{page}:{today}", user_id)
        
        # Store recent page views (keep last 100)
        view_data = {
            "timestamp": now.isoformat(),
            "user_id": user_id,
            "page": page
        }
        self.redis.lpush(f"recent_views:{page}", json.dumps(view_data))
        self.redis.ltrim(f"recent_views:{page}", 0, 99)  # Keep only last 100
        
        # Update top pages leaderboard
        self.redis.zincrby("top_pages", 1, page)
    
    def track_user_action(self, user_id, action, metadata=None):
        """Solution: Track user actions"""
        now = datetime.now()
        
        # Store action in user's activity stream
        action_data = {
            "action": action,
            "timestamp": now.isoformat(),
            "metadata": metadata or {}
        }
        
        self.redis.lpush(f"user_activity:{user_id}", json.dumps(action_data))
        self.redis.ltrim(f"user_activity:{user_id}", 0, 49)  # Keep last 50 actions
        
        # Increment action counters
        self.redis.hincrby("action_counts", action, 1)
        
        # Update user's last activity time
        self.redis.hset(f"user_stats:{user_id}", "last_activity", now.isoformat())
    
    def get_page_stats(self, page):
        """Solution: Get page statistics"""
        today = datetime.now().strftime("%Y-%m-%d")
        
        total_views = self.redis.get(f"page_views:{page}") or 0
        today_views = self.redis.hget(f"page_views:daily:{page}", today) or 0
        unique_visitors = self.redis.pfcount(f"unique_visitors:{page}:{today}")
        
        # Get recent views
        recent_views_raw = self.redis.lrange(f"recent_views:{page}", 0, 9)  # Last 10
        recent_views = [json.loads(view) for view in recent_views_raw]
        
        return {
            "page": page,
            "total_views": int(total_views),
            "today_views": int(today_views),
            "unique_visitors_today": unique_visitors,
            "recent_views": recent_views
        }
    
    def get_top_pages(self, count=10):
        """Solution: Get most viewed pages"""
        return self.redis.zrevrange("top_pages", 0, count-1, withscores=True)
    
    def get_user_activity(self, user_id, count=10):
        """Solution: Get user's recent activity"""
        activity_raw = self.redis.lrange(f"user_activity:{user_id}", 0, count-1)
        return [json.loads(activity) for activity in activity_raw]

def exercise_6_analytics():
    """Test real-time analytics system"""
    print("\n=== Exercise 6: Real-time Analytics ===")
    
    analytics = Analytics(r)
    
    # Clear existing analytics data
    r.delete("top_pages")
    
    # Test tracking page views
    print("1. Tracking page views:")
    pages = ["/home", "/products", "/about", "/contact", "/home", "/products", "/home"]
    users = ["user1", "user2", "user3", "user1", "user2", "user1", "user3"]
    
    for page, user in zip(pages, users):
        analytics.track_page_view(page, user)
        print(f"   Tracked view: {page} by {user}")
        time.sleep(0.1)
    
    # Test tracking user actions
    print("\n2. Tracking user actions:")
    actions = [
        ("user1", "login", {"ip": "192.168.1.1"}),
        ("user1", "view_product", {"product_id": "123"}),
        ("user1", "add_to_cart", {"product_id": "123", "quantity": 1}),
        ("user2", "login", {"ip": "192.168.1.2"}),
        ("user2", "search", {"query": "redis tutorial"})
    ]
    
    for user_id, action, metadata in actions:
        analytics.track_user_action(user_id, action, metadata)
        print(f"   Tracked action: {user_id} -> {action}")
    
    # Test getting page statistics
    print("\n3. Page statistics:")
    for page in ["/home", "/products", "/about"]:
        stats = analytics.get_page_stats(page)
        print(f"   {page}:")
        print(f"     Total views: {stats['total_views']}")
        print(f"     Today's views: {stats['today_views']}")
        print(f"     Unique visitors today: {stats['unique_visitors_today']}")
        print(f"     Recent views: {len(stats['recent_views'])}")
    
    # Test getting top pages
    print("\n4. Top pages:")
    top_pages = analytics.get_top_pages(5)
    for i, (page, views) in enumerate(top_pages, 1):
        print(f"   {i}. {page}: {int(views)} views")
    
    # Test getting user activity
    print("\n5. User activity:")
    for user_id in ["user1", "user2"]:
        activity = analytics.get_user_activity(user_id, 5)
        print(f"   {user_id} recent activity:")
        for action in activity:
            print(f"     {action['timestamp'][:19]}: {action['action']}")

# ============================================================================
# EXERCISE 7: Advanced Redis patterns
# ============================================================================

def exercise_7_lua_scripts():
    """Use Lua scripts for atomic operations"""
    print("\n=== Exercise 7: Lua Scripts ===")
    
    # Solution: Atomic rate limiting with Lua script
    rate_limit_script = """
    local key = KEYS[1]
    local limit = tonumber(ARGV[1])
    local window = tonumber(ARGV[2])
    local current_time = tonumber(ARGV[3])
    
    -- Remove old entries
    redis.call('ZREMRANGEBYSCORE', key, 0, current_time - window)
    
    -- Count current entries
    local current = redis.call('ZCARD', key)
    
    if current < limit then
        -- Add current request
        redis.call('ZADD', key, current_time, current_time .. ':' .. math.random())
        redis.call('EXPIRE', key, window)
        return 1
    else
        return 0
    end
    """
    
    # Register the Lua script
    rate_limit_lua = r.register_script(rate_limit_script)
    
    print("Testing atomic rate limiting with Lua script (3 requests per 5 seconds):")
    
    # Clear existing data
    r.delete("lua_rate_limit:user:123")
    
    for i in range(6):
        allowed = rate_limit_lua(keys=['lua_rate_limit:user:123'], 
                                args=[3, 5, time.time()])
        status = "✅ ALLOWED" if allowed else "❌ RATE LIMITED"
        print(f"   Request {i+1}: {status}")
        time.sleep(0.5)

def exercise_7_pub_sub():
    """Implement pub/sub messaging"""
    print("\n=== Exercise 7b: Pub/Sub Messaging ===")
    
    def publisher():
        """Solution: Publisher function"""
        messages = [
            "Hello from publisher!",
            "This is message 2",
            "Final message"
        ]
        
        for i, message in enumerate(messages, 1):
            r.publish("notifications", f"Message {i}: {message}")
            print(f"   Published: Message {i}")
            time.sleep(1)
    
    def subscriber_demo():
        """Solution: Subscriber demonstration"""
        # Note: In a real application, this would run in a separate process/thread
        pubsub = r.pubsub()
        pubsub.subscribe("notifications")
        
        print("   Subscriber listening for messages...")
        
        # Listen for a few messages (in real app, this would be a continuous loop)
        for i in range(4):  # Listen for 4 messages (including subscribe confirmation)
            message = pubsub.get_message(timeout=2)
            if message and message['type'] == 'message':
                print(f"   Received: {message['data']}")
        
        pubsub.unsubscribe("notifications")
        pubsub.close()
    
    print("Pub/Sub messaging demo:")
    print("   (In production, subscriber would run in separate process)")
    
    # Simulate pub/sub (simplified for demo)
    import threading
    
    # Start subscriber in background
    subscriber_thread = threading.Thread(target=subscriber_demo)
    subscriber_thread.start()
    
    time.sleep(0.5)  # Give subscriber time to start
    
    # Publish messages
    publisher()
    
    # Wait for subscriber to finish
    subscriber_thread.join()

def exercise_7_distributed_lock():
    """Implement distributed locking"""
    print("\n=== Exercise 7c: Distributed Lock ===")
    
    def acquire_lock(lock_name, timeout=10):
        """Solution: Acquire distributed lock"""
        lock_key = f"lock:{lock_name}"
        lock_id = str(uuid.uuid4())
        
        # Try to acquire lock with SET NX EX
        acquired = r.set(lock_key, lock_id, nx=True, ex=timeout)
        
        if acquired:
            return lock_id
        return None
    
    def release_lock(lock_name, lock_id):
        """Solution: Release distributed lock"""
        lock_key = f"lock:{lock_name}"
        
        # Lua script to check ownership and delete atomically
        release_script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        else
            return 0
        end
        """
        
        release_lua = r.register_script(release_script)
        return release_lua(keys=[lock_key], args=[lock_id]) == 1
    
    # Test distributed locking
    print("Testing distributed lock:")
    
    # Acquire lock
    lock_id = acquire_lock("resource:123", 5)
    if lock_id:
        print(f"   ✅ Lock acquired: {lock_id[:8]}...")
        
        # Try to acquire same lock (should fail)
        lock_id2 = acquire_lock("resource:123", 5)
        if lock_id2:
            print("   ❌ Second lock acquired (this shouldn't happen!)")
        else:
            print("   ✅ Second lock attempt failed (correct behavior)")
        
        # Release lock
        released = release_lock("resource:123", lock_id)
        if released:
            print("   ✅ Lock released successfully")
        else:
            print("   ❌ Failed to release lock")
        
        # Try to acquire lock again (should succeed now)
        lock_id3 = acquire_lock("resource:123", 5)
        if lock_id3:
            print(f"   ✅ Lock re-acquired: {lock_id3[:8]}...")
            release_lock("resource:123", lock_id3)
        else:
            print("   ❌ Failed to re-acquire lock")
    else:
        print("   ❌ Failed to acquire initial lock")

# ============================================================================
# EXERCISE 8: Performance optimization
# ============================================================================

def exercise_8_performance():
    """Test Redis performance optimization techniques"""
    print("\n=== Exercise 8: Performance Optimization ===")
    
    def test_pipeline_performance():
        """Solution: Compare single operations vs pipeline"""
        print("1. Testing pipeline performance:")
        
        # Test individual operations
        start_time = time.time()
        for i in range(100):
            r.set(f"test:individual:{i}", f"value_{i}")
        individual_time = time.time() - start_time
        
        # Test pipeline operations
        start_time = time.time()
        pipe = r.pipeline()
        for i in range(100):
            pipe.set(f"test:pipeline:{i}", f"value_{i}")
        pipe.execute()
        pipeline_time = time.time() - start_time
        
        print(f"   Individual operations (100 SETs): {individual_time:.3f}s")
        print(f"   Pipeline operations (100 SETs): {pipeline_time:.3f}s")
        print(f"   Pipeline is {individual_time/pipeline_time:.1f}x faster")
        
        # Cleanup
        pipe = r.pipeline()
        for i in range(100):
            pipe.delete(f"test:individual:{i}")
            pipe.delete(f"test:pipeline:{i}")
        pipe.execute()
    
    def test_memory_usage():
        """Solution: Analyze memory usage"""
        print("\n2. Testing memory usage patterns:")
        
        # Test individual keys vs hash
        # Individual keys
        for i in range(100):
            r.set(f"user:{i}:name", f"User {i}")
            r.set(f"user:{i}:email", f"user{i}@example.com")
            r.set(f"user:{i}:age", str(20 + i % 50))
        
        # Hash approach
        for i in range(100):
            r.hset(f"user_hash:{i}", mapping={
                "name": f"User {i}",
                "email": f"user{i}@example.com",
                "age": str(20 + i % 50)
            })
        
        # Check memory usage (approximate)
        individual_keys = len(r.keys("user:*:*"))
        hash_keys = len(r.keys("user_hash:*"))
        
        print(f"   Individual keys approach: {individual_keys} keys")
        print(f"   Hash approach: {hash_keys} keys")
        print(f"   Hash approach uses {individual_keys/hash_keys:.1f}x fewer keys")
        
        # Cleanup
        for pattern in ["user:*", "user_hash:*"]:
            keys = r.keys(pattern)
            if keys:
                r.delete(*keys)
    
    def test_connection_pooling():
        """Solution: Connection pooling demonstration"""
        print("\n3. Connection pooling benefits:")
        print("   Using connection pool for better performance")
        print("   Pool configuration:")
        print(f"     Max connections: {pool.max_connections}")
        print(f"     Current connections: {len(pool._available_connections)}")
        
        # Demonstrate pool usage
        start_time = time.time()
        for i in range(50):
            r.ping()
        pooled_time = time.time() - start_time
        
        print(f"   50 operations with connection pool: {pooled_time:.3f}s")
    
    # Run performance tests
    test_pipeline_performance()
    test_memory_usage()
    test_connection_pooling()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all exercises"""
    print("🚀 Starting Redis Solutions")
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
        
        print("\n🎉 All exercises completed successfully!")
        
        # Show final Redis stats
        info = r.info()
        print(f"\n📊 Final Redis Stats:")
        print(f"   Connected clients: {info.get('connected_clients', 'N/A')}")
        print(f"   Used memory: {info.get('used_memory_human', 'N/A')}")
        print(f"   Total commands processed: {info.get('total_commands_processed', 'N/A')}")
        print(f"   Keyspace hits: {info.get('keyspace_hits', 'N/A')}")
        print(f"   Keyspace misses: {info.get('keyspace_misses', 'N/A')}")
        
    except Exception as e:
        print(f"\n❌ Error during exercises: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()