"""
Day 2: NoSQL Databases - MongoDB Patterns
SOLUTION: Build an E-commerce System with PyMongo

Prerequisites:
- pip install pymongo
- MongoDB running locally on default port (27017)
"""

from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime, timedelta
from decimal import Decimal
import json

# ============================================================================
# SETUP: Connect to MongoDB and create sample data
# ============================================================================

def connect_to_mongodb():
    """Connect to MongoDB database"""
    try:
        client = MongoClient('mongodb://localhost:27017/')
        db = client['ecommerce_db']
        # Test connection
        client.admin.command('ping')
        print("Connected to MongoDB successfully")
        return db
    except Exception as e:
        print(f"Failed to connect to MongoDB: {e}")
        return None

def setup_collections(db):
    """Create and populate collections with sample data"""
    # Drop existing collections to start fresh
    db.users.drop()
    db.products.drop() 
    db.orders.drop()
    print("Dropped existing collections")

# ============================================================================
# EXERCISE 1: Design and create user documents
# ============================================================================

def create_user_documents(db):
    """Create user collection with embedded addresses"""
    
    users = [
        {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "+1-555-0123",
            "date_of_birth": datetime(1990, 5, 15),
            "addresses": [
                {
                    "type": "home",
                    "street": "123 Main St",
                    "city": "New York",
                    "state": "NY", 
                    "zipcode": "10001",
                    "country": "USA",
                    "is_default": True
                },
                {
                    "type": "work",
                    "street": "456 Business Ave",
                    "city": "New York",
                    "state": "NY",
                    "zipcode": "10002", 
                    "country": "USA",
                    "is_default": False
                }
            ],
            "preferences": {
                "notifications": ["email", "sms"],
                "currency": "USD",
                "language": "en",
                "newsletter": True
            },
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow() - timedelta(days=1),
            "status": "active"
        },
        {
            "name": "Alice Johnson",
            "email": "alice@example.com",
            "phone": "+1-555-0124",
            "addresses": [
                {
                    "type": "home",
                    "street": "789 Oak Ave",
                    "city": "San Francisco",
                    "state": "CA",
                    "zipcode": "94102",
                    "country": "USA",
                    "is_default": True
                }
            ],
            "preferences": {
                "notifications": ["email"],
                "currency": "USD",
                "language": "en"
            },
            "created_at": datetime.utcnow() - timedelta(days=30),
            "status": "active"
        },
        {
            "name": "Bob Smith",
            "email": "bob@example.com",
            "phone": "+1-555-0125",
            "addresses": [
                {
                    "type": "home",
                    "street": "321 Pine St",
                    "city": "Seattle",
                    "state": "WA",
                    "zipcode": "98101",
                    "country": "USA",
                    "is_default": True
                }
            ],
            "preferences": {
                "notifications": ["email", "sms"],
                "currency": "USD",
                "language": "en"
            },
            "created_at": datetime.utcnow() - timedelta(days=60),
            "status": "active"
        },
        {
            "name": "Carol Davis",
            "email": "carol@example.com",
            "phone": "+1-555-0126",
            "addresses": [
                {
                    "type": "home",
                    "street": "987 Elm Dr",
                    "city": "Austin",
                    "state": "TX",
                    "zipcode": "73301",
                    "country": "USA",
                    "is_default": True
                }
            ],
            "preferences": {
                "notifications": ["email"],
                "currency": "USD",
                "language": "en"
            },
            "created_at": datetime.utcnow() - timedelta(days=90),
            "status": "inactive"
        },
        {
            "name": "David Wilson",
            "email": "david@example.com",
            "phone": "+1-555-0127",
            "addresses": [
                {
                    "type": "home",
                    "street": "147 Maple Ln",
                    "city": "Denver",
                    "state": "CO",
                    "zipcode": "80201",
                    "country": "USA",
                    "is_default": True
                }
            ],
            "preferences": {
                "notifications": ["sms"],
                "currency": "USD",
                "language": "en"
            },
            "created_at": datetime.utcnow() - timedelta(days=120),
            "status": "active"
        }
    ]
    
    result = db.users.insert_many(users)
    print(f"Inserted {len(result.inserted_ids)} users")
    return result.inserted_ids

# ============================================================================
# EXERCISE 2: Design and create product documents
# ============================================================================

def create_product_documents(db):
    """Create products collection with embedded reviews"""
    
    products = [
        # Electronics
        {
            "name": "MacBook Pro 16\"",
            "description": "Powerful laptop for professionals with M3 Pro chip",
            "price": Decimal("2499.99"),
            "category": "Electronics",
            "subcategory": "Laptops",
            "brand": "Apple",
            "sku": "MBP16-M3-512",
            "images": [
                "https://example.com/images/mbp16-1.jpg",
                "https://example.com/images/mbp16-2.jpg"
            ],
            "specifications": {
                "processor": "Apple M3 Pro",
                "memory": "18GB",
                "storage": "512GB SSD",
                "display": "16-inch Liquid Retina XDR",
                "weight": "2.1 kg",
                "color": "Space Gray"
            },
            "inventory": {
                "quantity": 50,
                "reserved": 5,
                "available": 45,
                "warehouse_location": "NYC-01"
            },
            "reviews": [
                {
                    "_id": ObjectId(),
                    "user_name": "Alice Johnson",
                    "rating": 5,
                    "title": "Excellent laptop!",
                    "comment": "Fast, reliable, great for development work.",
                    "date": datetime.utcnow() - timedelta(days=10),
                    "verified_purchase": True,
                    "helpful_votes": 12
                }
            ],
            "rating_summary": {
                "average": 4.8,
                "count": 156,
                "distribution": {"5": 120, "4": 25, "3": 8, "2": 2, "1": 1}
            },
            "tags": ["laptop", "apple", "professional", "m3"],
            "created_at": datetime.utcnow() - timedelta(days=30),
            "updated_at": datetime.utcnow(),
            "status": "active",
            "view_count": 1250
        },
        {
            "name": "iPhone 15 Pro",
            "description": "Latest iPhone with titanium design",
            "price": Decimal("999.99"),
            "category": "Electronics",
            "subcategory": "Smartphones",
            "brand": "Apple",
            "sku": "IP15P-128-TIT",
            "inventory": {"quantity": 100, "available": 95},
            "rating_summary": {"average": 4.7, "count": 89},
            "created_at": datetime.utcnow() - timedelta(days=60),
            "status": "active",
            "view_count": 890
        },
        {
            "name": "Sony WH-1000XM5",
            "description": "Premium noise-canceling headphones",
            "price": Decimal("399.99"),
            "category": "Electronics",
            "subcategory": "Audio",
            "brand": "Sony",
            "inventory": {"quantity": 60, "available": 58},
            "rating_summary": {"average": 4.8, "count": 234},
            "created_at": datetime.utcnow() - timedelta(days=45),
            "status": "active"
        },
        # Books
        {
            "name": "The Pragmatic Programmer",
            "description": "Classic programming book",
            "price": Decimal("49.99"),
            "category": "Books",
            "subcategory": "Technology",
            "brand": "Addison-Wesley",
            "inventory": {"quantity": 200, "available": 195},
            "rating_summary": {"average": 4.9, "count": 1250},
            "created_at": datetime.utcnow() - timedelta(days=90),
            "status": "active"
        },
        {
            "name": "Clean Code",
            "description": "A handbook of agile software craftsmanship",
            "price": Decimal("44.99"),
            "category": "Books",
            "subcategory": "Technology", 
            "brand": "Prentice Hall",
            "inventory": {"quantity": 150, "available": 148},
            "rating_summary": {"average": 4.6, "count": 890},
            "created_at": datetime.utcnow() - timedelta(days=75),
            "status": "active"
        },
        # Clothing
        {
            "name": "Levi's 501 Jeans",
            "description": "Classic straight-leg jeans",
            "price": Decimal("89.99"),
            "category": "Clothing",
            "subcategory": "Jeans",
            "brand": "Levi's",
            "inventory": {"quantity": 120, "available": 115},
            "rating_summary": {"average": 4.4, "count": 456},
            "created_at": datetime.utcnow() - timedelta(days=120),
            "status": "active"
        },
        {
            "name": "Nike Air Max 90",
            "description": "Iconic running shoes",
            "price": Decimal("119.99"),
            "category": "Clothing",
            "subcategory": "Shoes",
            "brand": "Nike",
            "inventory": {"quantity": 90, "available": 87},
            "rating_summary": {"average": 4.3, "count": 678},
            "created_at": datetime.utcnow() - timedelta(days=100),
            "status": "active"
        },
        # Home
        {
            "name": "Instant Pot Duo 7-in-1",
            "description": "Multi-use pressure cooker",
            "price": Decimal("79.99"),
            "category": "Home",
            "subcategory": "Kitchen",
            "brand": "Instant Pot",
            "inventory": {"quantity": 75, "available": 72},
            "rating_summary": {"average": 4.7, "count": 2340},
            "created_at": datetime.utcnow() - timedelta(days=80),
            "status": "active"
        },
        {
            "name": "Dyson V15 Detect",
            "description": "Cordless vacuum cleaner",
            "price": Decimal("749.99"),
            "category": "Home",
            "subcategory": "Cleaning",
            "brand": "Dyson",
            "inventory": {"quantity": 40, "available": 38},
            "rating_summary": {"average": 4.6, "count": 567},
            "created_at": datetime.utcnow() - timedelta(days=50),
            "status": "active"
        },
        # Sports
        {
            "name": "Peloton Bike+",
            "description": "Interactive exercise bike",
            "price": Decimal("2495.00"),
            "category": "Sports",
            "subcategory": "Fitness",
            "brand": "Peloton",
            "inventory": {"quantity": 25, "available": 23},
            "rating_summary": {"average": 4.5, "count": 123},
            "created_at": datetime.utcnow() - timedelta(days=40),
            "status": "active"
        }
    ]
    
    result = db.products.insert_many(products)
    print(f"Inserted {len(result.inserted_ids)} products")
    return result.inserted_ids

# ============================================================================
# EXERCISE 3: Design and create order documents
# ============================================================================

def create_order_documents(db, user_ids, product_ids):
    """Create orders collection with embedded order items"""
    
    orders = [
        {
            "order_number": "ORD-2024-001",
            "user_id": user_ids[0],  # John Doe
            "status": "completed",
            "items": [
                {
                    "product_id": product_ids[0],  # MacBook Pro
                    "product_name": "MacBook Pro 16\"",
                    "sku": "MBP16-M3-512",
                    "price": Decimal("2499.99"),
                    "quantity": 1,
                    "subtotal": Decimal("2499.99")
                }
            ],
            "pricing": {
                "subtotal": Decimal("2499.99"),
                "tax": Decimal("199.99"),
                "shipping": Decimal("0.00"),
                "discount": Decimal("0.00"),
                "total": Decimal("2699.98")
            },
            "shipping_address": {
                "name": "John Doe",
                "street": "123 Main St",
                "city": "New York",
                "state": "NY",
                "zipcode": "10001",
                "country": "USA"
            },
            "payment": {
                "method": "credit_card",
                "card_last_four": "1234",
                "transaction_id": "txn_abc123",
                "status": "completed"
            },
            "status_history": [
                {
                    "status": "pending",
                    "timestamp": datetime.utcnow() - timedelta(days=5),
                    "note": "Order created"
                },
                {
                    "status": "processing",
                    "timestamp": datetime.utcnow() - timedelta(days=4),
                    "note": "Payment confirmed"
                },
                {
                    "status": "shipped",
                    "timestamp": datetime.utcnow() - timedelta(days=3),
                    "note": "Order shipped"
                },
                {
                    "status": "completed",
                    "timestamp": datetime.utcnow() - timedelta(days=1),
                    "note": "Order delivered"
                }
            ],
            "created_at": datetime.utcnow() - timedelta(days=5),
            "updated_at": datetime.utcnow() - timedelta(days=1)
        },
        {
            "order_number": "ORD-2024-002",
            "user_id": user_ids[1],  # Alice Johnson
            "status": "shipped",
            "items": [
                {
                    "product_id": product_ids[1],  # iPhone 15 Pro
                    "product_name": "iPhone 15 Pro",
                    "price": Decimal("999.99"),
                    "quantity": 1,
                    "subtotal": Decimal("999.99")
                },
                {
                    "product_id": product_ids[2],  # Sony Headphones
                    "product_name": "Sony WH-1000XM5",
                    "price": Decimal("399.99"),
                    "quantity": 1,
                    "subtotal": Decimal("399.99")
                }
            ],
            "pricing": {
                "subtotal": Decimal("1399.98"),
                "tax": Decimal("111.99"),
                "shipping": Decimal("9.99"),
                "total": Decimal("1521.96")
            },
            "status": "shipped",
            "created_at": datetime.utcnow() - timedelta(days=3),
            "updated_at": datetime.utcnow() - timedelta(days=1)
        },
        {
            "order_number": "ORD-2024-003",
            "user_id": user_ids[2],  # Bob Smith
            "status": "pending",
            "items": [
                {
                    "product_id": product_ids[3],  # Pragmatic Programmer
                    "product_name": "The Pragmatic Programmer",
                    "price": Decimal("49.99"),
                    "quantity": 2,
                    "subtotal": Decimal("99.98")
                }
            ],
            "pricing": {
                "subtotal": Decimal("99.98"),
                "tax": Decimal("7.99"),
                "shipping": Decimal("5.99"),
                "total": Decimal("113.96")
            },
            "status": "pending",
            "created_at": datetime.utcnow() - timedelta(days=1),
            "updated_at": datetime.utcnow() - timedelta(days=1)
        }
    ]
    
    result = db.orders.insert_many(orders)
    print(f"Inserted {len(result.inserted_ids)} orders")
    return result.inserted_ids

# ============================================================================
# EXERCISE 4: Basic queries
# ============================================================================

def basic_queries(db):
    """Practice basic MongoDB queries"""
    
    print("=== Basic Queries ===")
    
    # 1. Find all users from a specific city
    print("1. Users from New York:")
    users_ny = db.users.find({"addresses.city": "New York"})
    for user in users_ny:
        print(f"  - {user['name']} ({user['email']})")
    
    # 2. Find products in a price range
    print("\\n2. Products between $100-$500:")
    products = db.products.find({
        "price": {"$gte": Decimal("100.00"), "$lte": Decimal("500.00")}
    })
    for product in products:
        print(f"  - {product['name']}: ${product['price']}")
    
    # 3. Find products by category with high ratings
    print("\\n3. High-rated Electronics:")
    electronics = db.products.find({
        "category": "Electronics",
        "rating_summary.average": {"$gte": 4.0}
    })
    for product in electronics:
        rating = product.get('rating_summary', {}).get('average', 0)
        print(f"  - {product['name']}: {rating}/5.0")
    
    # 4. Find orders by status
    print("\\n4. Shipped orders:")
    shipped_orders = db.orders.find({"status": "shipped"})
    for order in shipped_orders:
        print(f"  - {order['order_number']}: ${order['pricing']['total']}")
    
    # 5. Find user's order history
    print("\\n5. John's order history:")
    john = db.users.find_one({"email": "john@example.com"})
    if john:
        orders = db.orders.find({"user_id": john["_id"]})
        for order in orders:
            print(f"  - {order['order_number']}: {order['status']}")

# ============================================================================
# EXERCISE 5: Complex queries with arrays and nested documents
# ============================================================================

def complex_queries(db):
    """Practice complex MongoDB queries"""
    
    print("=== Complex Queries ===")
    
    # 1. Find users with specific preferences
    print("1. Users with email notifications:")
    users = db.users.find({"preferences.notifications": "email"})
    for user in users:
        print(f"  - {user['name']}")
    
    # 2. Find products with specific features
    print("\\n2. Products with Apple processors:")
    products = db.products.find({"specifications.processor": {"$regex": "Apple", "$options": "i"}})
    for product in products:
        processor = product.get('specifications', {}).get('processor', 'N/A')
        print(f"  - {product['name']}: {processor}")
    
    # 3. Find orders containing specific products
    print("\\n3. Orders containing MacBook:")
    orders = db.orders.find({"items.product_name": {"$regex": "MacBook", "$options": "i"}})
    for order in orders:
        print(f"  - {order['order_number']}: {order['status']}")
    
    # 4. Text search on products
    print("\\n4. Text search results:")
    try:
        db.products.create_index([("name", "text"), ("description", "text")])
        results = db.products.find({"$text": {"$search": "laptop computer"}})
        for product in results:
            print(f"  - {product['name']}")
    except Exception as e:
        print(f"  Text search error: {e}")

# ============================================================================
# EXERCISE 6: Aggregation pipeline queries
# ============================================================================

def aggregation_queries(db):
    """Practice MongoDB aggregation pipelines"""
    
    print("=== Aggregation Queries ===")
    
    # 1. Calculate total revenue by category
    print("1. Revenue by category:")
    pipeline1 = [
        {"$match": {"status": "completed"}},
        {"$unwind": "$items"},
        {"$lookup": {
            "from": "products",
            "localField": "items.product_id", 
            "foreignField": "_id",
            "as": "product_info"
        }},
        {"$unwind": "$product_info"},
        {"$group": {
            "_id": "$product_info.category",
            "total_revenue": {"$sum": "$items.subtotal"},
            "total_quantity": {"$sum": "$items.quantity"}
        }},
        {"$sort": {"total_revenue": -1}}
    ]
    results = list(db.orders.aggregate(pipeline1))
    for result in results:
        print(f"  - {result['_id']}: ${result['total_revenue']}")
    
    # 2. Find top customers by total spending
    print("\\n2. Top customers:")
    pipeline2 = [
        {"$match": {"status": "completed"}},
        {"$group": {
            "_id": "$user_id",
            "total_spent": {"$sum": "$pricing.total"},
            "order_count": {"$sum": 1}
        }},
        {"$sort": {"total_spent": -1}},
        {"$limit": 3},
        {"$lookup": {
            "from": "users",
            "localField": "_id",
            "foreignField": "_id",
            "as": "user_info"
        }},
        {"$unwind": "$user_info"},
        {"$project": {
            "user_name": "$user_info.name",
            "user_email": "$user_info.email",
            "total_spent": 1,
            "order_count": 1
        }}
    ]
    results = list(db.orders.aggregate(pipeline2))
    for result in results:
        print(f"  - {result['user_name']}: ${result['total_spent']} ({result['order_count']} orders)")
    
    # 3. Calculate average order value by month
    print("\\n3. Average order value by month:")
    pipeline3 = [
        {"$match": {"status": "completed"}},
        {"$group": {
            "_id": {
                "year": {"$year": "$created_at"},
                "month": {"$month": "$created_at"}
            },
            "avg_order_value": {"$avg": "$pricing.total"},
            "total_orders": {"$sum": 1}
        }},
        {"$sort": {"_id.year": 1, "_id.month": 1}}
    ]
    results = list(db.orders.aggregate(pipeline3))
    for result in results:
        print(f"  - {result['_id']['year']}-{result['_id']['month']:02d}: ${result['avg_order_value']:.2f}")
    
    # 4. Product performance analysis
    print("\\n4. Product performance:")
    pipeline4 = [
        {"$match": {"status": "completed"}},
        {"$unwind": "$items"},
        {"$group": {
            "_id": "$items.product_name",
            "total_quantity_sold": {"$sum": "$items.quantity"},
            "total_revenue": {"$sum": "$items.subtotal"},
            "avg_price": {"$avg": "$items.price"}
        }},
        {"$sort": {"total_revenue": -1}},
        {"$limit": 5}
    ]
    results = list(db.orders.aggregate(pipeline4))
    for result in results:
        print(f"  - {result['_id']}: {result['total_quantity_sold']} sold, ${result['total_revenue']} revenue")

# ============================================================================
# EXERCISE 7: Create appropriate indexes
# ============================================================================

def create_indexes(db):
    """Create indexes for common queries"""
    
    print("=== Creating Indexes ===")
    
    # Basic indexes
    db.users.create_index("email")
    print("Created index on users.email")
    
    db.users.create_index("status")
    print("Created index on users.status")
    
    db.products.create_index("category")
    print("Created index on products.category")
    
    db.products.create_index("price")
    print("Created index on products.price")
    
    db.orders.create_index("user_id")
    print("Created index on orders.user_id")
    
    db.orders.create_index("status")
    print("Created index on orders.status")
    
    # Compound indexes
    db.products.create_index([("category", 1), ("price", 1)])
    print("Created compound index on products(category, price)")
    
    db.orders.create_index([("user_id", 1), ("created_at", -1)])
    print("Created compound index on orders(user_id, created_at)")
    
    # Text index
    try:
        db.products.create_index([("name", "text"), ("description", "text")])
        print("Created text index on products(name, description)")
    except Exception as e:
        print(f"Text index already exists: {e}")
    
    # Partial index
    db.products.create_index(
        [("category", 1), ("rating_summary.average", -1)],
        partialFilterExpression={"status": "active"}
    )
    print("Created partial index on active products")

# ============================================================================
# EXERCISE 8: Update operations
# ============================================================================

def update_operations(db):
    """Practice MongoDB update operations"""
    
    print("=== Update Operations ===")
    
    # 1. Update user preferences
    print("1. Adding user preference:")
    result = db.users.update_one(
        {"email": "john@example.com"},
        {"$addToSet": {"preferences.notifications": "push"}}
    )
    print(f"  Modified {result.modified_count} user(s)")
    
    # 2. Add a review to a product
    print("\\n2. Adding product review:")
    new_review = {
        "_id": ObjectId(),
        "user_name": "Bob Smith",
        "rating": 5,
        "title": "Great laptop!",
        "comment": "Perfect for my development work.",
        "date": datetime.utcnow(),
        "verified_purchase": True,
        "helpful_votes": 0
    }
    result = db.products.update_one(
        {"name": "MacBook Pro 16\\\""},
        {
            "$push": {"reviews": new_review},
            "$inc": {"rating_summary.count": 1}
        }
    )
    print(f"  Modified {result.modified_count} product(s)")
    
    # 3. Update order status
    print("\\n3. Updating order status:")
    result = db.orders.update_one(
        {"order_number": "ORD-2024-003"},
        {
            "$set": {
                "status": "processing",
                "updated_at": datetime.utcnow()
            },
            "$push": {
                "status_history": {
                    "status": "processing",
                    "timestamp": datetime.utcnow(),
                    "note": "Payment confirmed"
                }
            }
        }
    )
    print(f"  Modified {result.modified_count} order(s)")
    
    # 4. Increment product view count
    print("\\n4. Incrementing view count:")
    result = db.products.update_one(
        {"name": "MacBook Pro 16\\\""},
        {"$inc": {"view_count": 1}}
    )
    print(f"  Modified {result.modified_count} product(s)")
    
    # 5. Update inventory after purchase
    print("\\n5. Updating inventory:")
    result = db.products.update_one(
        {"name": "MacBook Pro 16\\\""},
        {
            "$inc": {
                "inventory.available": -1,
                "inventory.reserved": 1
            }
        }
    )
    print(f"  Modified {result.modified_count} product(s)")

# ============================================================================
# EXERCISE 9: Performance analysis
# ============================================================================

def performance_analysis(db):
    """Analyze query performance"""
    
    print("=== Performance Analysis ===")
    
    # 1. Analyze query performance using explain()
    print("1. Query performance analysis:")
    explain_result = db.users.find({"email": "john@example.com"}).explain()
    execution_stats = explain_result.get('executionStats', {})
    print(f"  Execution time: {execution_stats.get('executionTimeMillis', 'N/A')}ms")
    print(f"  Documents examined: {execution_stats.get('totalDocsExamined', 'N/A')}")
    print(f"  Index used: {explain_result.get('queryPlanner', {}).get('winningPlan', {}).get('inputStage', {}).get('indexName', 'None')}")
    
    # 2. Check index usage statistics
    print("\\n2. Index usage statistics:")
    try:
        # Get index stats (requires MongoDB 3.2+)
        stats = db.users.aggregate([{"$indexStats": {}}])
        for stat in stats:
            print(f"  - {stat['name']}: {stat['accesses']['ops']} operations")
    except Exception as e:
        print(f"  Index stats not available: {e}")
    
    # 3. Compare query performance
    print("\\n3. Performance comparison:")
    
    # Query without index hint
    import time
    start_time = time.time()
    list(db.products.find({"category": "Electronics"}))
    no_hint_time = time.time() - start_time
    
    # Query with index hint
    start_time = time.time()
    list(db.products.find({"category": "Electronics"}).hint([("category", 1)]))
    with_hint_time = time.time() - start_time
    
    print(f"  Without hint: {no_hint_time:.4f}s")
    print(f"  With hint: {with_hint_time:.4f}s")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run all exercises"""
    print("Day 2: NoSQL Databases - MongoDB Patterns")
    print("=========================================")
    
    # Connect to MongoDB
    db = connect_to_mongodb()
    if not db:
        print("Failed to connect to MongoDB")
        return
    
    # Setup collections
    setup_collections(db)
    
    # Exercise 1: Create user documents
    user_ids = create_user_documents(db)
    
    # Exercise 2: Create product documents  
    product_ids = create_product_documents(db)
    
    # Exercise 3: Create order documents
    order_ids = create_order_documents(db, user_ids, product_ids)
    
    # Exercise 4: Basic queries
    basic_queries(db)
    
    # Exercise 5: Complex queries
    complex_queries(db)
    
    # Exercise 6: Aggregation queries
    aggregation_queries(db)
    
    # Exercise 7: Create indexes
    create_indexes(db)
    
    # Exercise 8: Update operations
    update_operations(db)
    
    # Exercise 9: Performance analysis
    performance_analysis(db)
    
    print("\\n=== Exercise Completed Successfully! ===")
    print("\\nKey learnings:")
    print("- Document modeling with embedding vs referencing")
    print("- Complex queries with arrays and nested documents")
    print("- Aggregation pipelines for data analysis")
    print("- Index creation and performance optimization")
    print("- Update operations with atomic modifiers")

if __name__ == "__main__":
    main()

# ============================================================================
# ANSWERS TO QUESTIONS
# ============================================================================

"""
1. When would you embed vs reference data in this e-commerce system?
   Answer: 
   - EMBED: User addresses (limited, accessed together), order items (fixed at order time), 
     recent product reviews (limited to 10-20), product specifications
   - REFERENCE: User orders (unlimited growth), product catalog in orders (changes frequently),
     all product reviews (unlimited), user profiles in orders

2. How would you handle a product with thousands of reviews?
   Answer: 
   - Store only recent/featured reviews (10-20) embedded in product document
   - Store all reviews in separate reviews collection with product_id reference
   - Use aggregation to calculate rating summaries
   - Implement pagination for review display

3. What indexes would you create for a high-traffic e-commerce site?
   Answer:
   - users: email (login), status
   - products: category, price, rating, text search on name/description
   - orders: user_id, status, created_at
   - Compound: products(category, price), orders(user_id, created_at)
   - Partial: active products only

4. How would you implement real-time inventory updates?
   Answer:
   - Use atomic operations ($inc) for inventory changes
   - Implement optimistic concurrency control
   - Use change streams for real-time notifications
   - Consider separate inventory service with MongoDB as backing store
   - Use transactions for multi-document inventory operations

5. What are the trade-offs of denormalizing order data?
   Answer:
   PROS: Fast queries, no joins needed, order data frozen at purchase time
   CONS: Data duplication, potential inconsistency, larger documents, 
         complex updates when product info changes
"""