"""
Day 2: NoSQL Databases - MongoDB Patterns
Exercise: Build an E-commerce System with PyMongo

Prerequisites:
- pip install pymongo
- MongoDB running locally on default port (27017)
"""

from pymongo import MongoClient
from datetime import datetime, timedelta
from decimal import Decimal
import json

# ============================================================================
# SETUP: Connect to MongoDB and create sample data
# ============================================================================

def connect_to_mongodb():
    """Connect to MongoDB database"""
    # TODO: Create MongoDB client connection
    # client = MongoClient('mongodb://localhost:27017/')
    # db = client['ecommerce_db']
    # return db
    pass

def setup_collections(db):
    """Create and populate collections with sample data"""
    # TODO: Drop existing collections to start fresh
    # db.users.drop()
    # db.products.drop() 
    # db.orders.drop()
    pass

# ============================================================================
# EXERCISE 1: Design and create user documents
# ============================================================================

def create_user_documents(db):
    """Create user collection with embedded addresses"""
    
    # TODO: Insert sample users with embedded addresses
    # Requirements:
    # - Store user profile information
    # - Embed addresses (home, work, etc.)
    # - Track user preferences and settings
    # - Include registration and last login dates
    
    sample_user = {
        # TODO: Complete this document structure
        "name": "John Doe",
        "email": "john@example.com",
        "phone": "+1-555-0123",
        "addresses": [
            {
                "type": "home",
                "street": "123 Main St",
                "city": "New York",
                "state": "NY", 
                "zipcode": "10001",
                "country": "USA",
                "is_default": True
            }
            # TODO: Add more address types
        ],
        "preferences": {
            # TODO: Add user preferences
        },
        "created_at": datetime.utcnow(),
        "status": "active"
    }
    
    # TODO: Insert the user document
    # result = db.users.insert_one(sample_user)
    # print(f"Inserted user with ID: {result.inserted_id}")
    
    # TODO: Insert 4 more users with different profiles
    pass

# ============================================================================
# EXERCISE 2: Design and create product documents
# ============================================================================

def create_product_documents(db):
    """Create products collection with embedded reviews"""
    
    # TODO: Insert sample products with embedded reviews
    # Requirements:
    # - Product information (name, description, price, category)
    # - Embed product reviews (limited to recent 10 reviews)
    # - Include inventory information
    # - Store product images and specifications
    
    sample_product = {
        # TODO: Complete this document structure
        "name": "MacBook Pro 16\"",
        "description": "Powerful laptop for professionals",
        "price": Decimal("2499.99"),
        "category": "Electronics",
        "subcategory": "Laptops",
        "brand": "Apple",
        "sku": "MBP16-M3-512",
        "images": [
            "https://example.com/images/mbp16-1.jpg"
        ],
        "specifications": {
            # TODO: Add product specifications
        },
        "inventory": {
            # TODO: Add inventory information
        },
        "reviews": [
            # TODO: Add sample reviews
        ],
        "rating_summary": {
            # TODO: Add rating summary
        },
        "created_at": datetime.utcnow(),
        "status": "active"
    }
    
    # TODO: Insert the product document
    # TODO: Insert 19 more products across different categories
    pass

# ============================================================================
# EXERCISE 3: Design and create order documents
# ============================================================================

def create_order_documents(db):
    """Create orders collection with embedded order items"""
    
    # TODO: Insert sample orders with embedded items
    # Requirements:
    # - Reference to user (user_id)
    # - Embed order items with product details
    # - Order status and tracking information
    # - Shipping and billing addresses
    # - Payment information
    
    # First get a user ID for the order
    # user = db.users.find_one({"email": "john@example.com"})
    # if not user:
    #     print("No user found. Create users first.")
    #     return
    
    sample_order = {
        # TODO: Complete this document structure
        "order_number": "ORD-2024-001",
        "user_id": "user_object_id_here",  # TODO: Use actual user ObjectId
        "status": "pending",
        "items": [
            {
                # TODO: Add order items with product details
            }
        ],
        "pricing": {
            # TODO: Add pricing breakdown
        },
        "shipping_address": {
            # TODO: Add shipping address
        },
        "created_at": datetime.utcnow()
    }
    
    # TODO: Insert the order document
    # TODO: Insert 14 more orders with various statuses
    pass

# ============================================================================
# EXERCISE 4: Basic queries
# ============================================================================

def basic_queries(db):
    """Practice basic MongoDB queries"""
    
    print("=== Basic Queries ===")
    
    # TODO: Find all users from a specific city
    # Query: Find users with addresses in "New York"
    print("1. Users from New York:")
    # users_ny = db.users.find({"addresses.city": "New York"})
    # for user in users_ny:
    #     print(f"  - {user['name']} ({user['email']})")
    
    # TODO: Find products in a price range
    # Query: Find products between $100 and $500
    print("\n2. Products between $100-$500:")
    # products = db.products.find({
    #     "price": {"$gte": Decimal("100.00"), "$lte": Decimal("500.00")}
    # })
    
    # TODO: Find products by category with high ratings
    # Query: Find Electronics products with average rating >= 4.0
    print("\n3. High-rated Electronics:")
    
    # TODO: Find orders by status
    # Query: Find all "shipped" orders
    print("\n4. Shipped orders:")
    
    # TODO: Find user's order history
    # Query: Find all orders for a specific user
    print("\n5. User's order history:")

# ============================================================================
# EXERCISE 5: Complex queries with arrays and nested documents
# ============================================================================

def complex_queries(db):
    """Practice complex MongoDB queries"""
    
    print("=== Complex Queries ===")
    
    # TODO: Find users with specific preferences
    # Query: Find users who prefer "email" notifications
    print("1. Users with email notifications:")
    
    # TODO: Find products with specific features
    # Query: Find products that have "wireless" in specifications
    print("\n2. Products with wireless features:")
    
    # TODO: Find orders containing specific products
    # Query: Find orders that contain "MacBook" products
    print("\n3. Orders containing MacBook:")
    
    # TODO: Text search on products
    # First create a text index, then search for products
    print("\n4. Text search results:")
    # db.products.create_index([("name", "text"), ("description", "text")])
    # results = db.products.find({"$text": {"$search": "laptop computer"}})

# ============================================================================
# EXERCISE 6: Aggregation pipeline queries
# ============================================================================

def aggregation_queries(db):
    """Practice MongoDB aggregation pipelines"""
    
    print("=== Aggregation Queries ===")
    
    # TODO: Calculate total revenue by category
    print("1. Revenue by category:")
    pipeline1 = [
        # TODO: Complete this aggregation pipeline
        # {"$match": {"status": "completed"}},
        # {"$unwind": "$items"},
        # {"$lookup": {
        #     "from": "products",
        #     "localField": "items.product_id", 
        #     "foreignField": "_id",
        #     "as": "product_info"
        # }},
        # {"$unwind": "$product_info"},
        # {"$group": {
        #     "_id": "$product_info.category",
        #     "total_revenue": {"$sum": "$items.subtotal"}
        # }},
        # {"$sort": {"total_revenue": -1}}
    ]
    # results = db.orders.aggregate(pipeline1)
    
    # TODO: Find top customers by total spending
    print("\n2. Top customers:")
    pipeline2 = [
        # TODO: Complete this aggregation pipeline
    ]
    
    # TODO: Calculate average order value by month
    print("\n3. Average order value by month:")
    pipeline3 = [
        # TODO: Complete this aggregation pipeline
    ]
    
    # TODO: Product performance analysis
    print("\n4. Product performance:")
    pipeline4 = [
        # TODO: Complete this aggregation pipeline
    ]

# ============================================================================
# EXERCISE 7: Create appropriate indexes
# ============================================================================

def create_indexes(db):
    """Create indexes for common queries"""
    
    print("=== Creating Indexes ===")
    
    # TODO: Create indexes for common queries
    # Consider these query patterns:
    # - Find user by email (login)
    # - Find products by category
    # - Find products by price range
    # - Find orders by user_id
    # - Find orders by status
    # - Text search on products
    # - Find orders by date range
    
    # Example:
    # db.users.create_index("email")
    # print("Created index on users.email")
    
    # TODO: Create compound indexes for complex queries
    
    # TODO: Create a partial index
    # Example: Index only active users or available products
    
    # TODO: Create a TTL index
    # Example: Expire user sessions or temporary data

# ============================================================================
# EXERCISE 8: Update operations
# ============================================================================

def update_operations(db):
    """Practice MongoDB update operations"""
    
    print("=== Update Operations ===")
    
    # TODO: Update user preferences
    # Add a new preference to a user's preferences array
    print("1. Adding user preference:")
    
    # TODO: Add a review to a product
    # Use $push to add a new review to a product's reviews array
    print("\n2. Adding product review:")
    
    # TODO: Update order status
    # Change an order status from "pending" to "processing"
    print("\n3. Updating order status:")
    
    # TODO: Increment product view count
    # Use $inc to increment a product's view_count field
    print("\n4. Incrementing view count:")
    
    # TODO: Update inventory after purchase
    # Decrease product inventory count when an order is placed
    print("\n5. Updating inventory:")

# ============================================================================
# EXERCISE 9: Performance analysis
# ============================================================================

def performance_analysis(db):
    """Analyze query performance"""
    
    print("=== Performance Analysis ===")
    
    # TODO: Analyze query performance using explain()
    print("1. Query performance analysis:")
    
    # Example:
    # explain_result = db.users.find({"email": "john@example.com"}).explain()
    # print(f"Execution stats: {explain_result['executionStats']}")
    
    # TODO: Compare performance before and after adding indexes
    print("\n2. Performance comparison:")
    
    # TODO: Check index usage statistics
    print("\n3. Index usage statistics:")

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function to run all exercises"""
    print("Day 2: NoSQL Databases - MongoDB Patterns")
    print("=========================================")
    
    # TODO: Uncomment and complete each section
    
    # Connect to MongoDB
    # db = connect_to_mongodb()
    # if not db:
    #     print("Failed to connect to MongoDB")
    #     return
    
    # Setup collections
    # setup_collections(db)
    
    # Exercise 1: Create user documents
    # create_user_documents(db)
    
    # Exercise 2: Create product documents  
    # create_product_documents(db)
    
    # Exercise 3: Create order documents
    # create_order_documents(db)
    
    # Exercise 4: Basic queries
    # basic_queries(db)
    
    # Exercise 5: Complex queries
    # complex_queries(db)
    
    # Exercise 6: Aggregation queries
    # aggregation_queries(db)
    
    # Exercise 7: Create indexes
    # create_indexes(db)
    
    # Exercise 8: Update operations
    # update_operations(db)
    
    # Exercise 9: Performance analysis
    # performance_analysis(db)
    
    print("\nExercise completed! Check the solution.py for complete implementations.")

if __name__ == "__main__":
    main()

# ============================================================================
# QUESTIONS TO CONSIDER
# ============================================================================

"""
1. When would you embed vs reference data in this e-commerce system?
   Answer: _______________

2. How would you handle a product with thousands of reviews?
   Answer: _______________

3. What indexes would you create for a high-traffic e-commerce site?
   Answer: _______________

4. How would you implement real-time inventory updates?
   Answer: _______________

5. What are the trade-offs of denormalizing order data?
   Answer: _______________
"""