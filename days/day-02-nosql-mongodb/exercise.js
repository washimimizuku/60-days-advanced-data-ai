// Day 2: NoSQL Databases - MongoDB Patterns
// Exercise: Build an E-commerce System

// ============================================================================
// SETUP: Connect to MongoDB and create sample data
// ============================================================================

// Connect to MongoDB (assumes MongoDB is running locally)
// In production, use connection string: mongodb://username:password@host:port/database
use('ecommerce_db');

// ============================================================================
// EXERCISE 1: Design and create collections
// ============================================================================

// TODO: Create a users collection with embedded addresses
// Requirements:
// - Store user profile information
// - Embed addresses (home, work, etc.)
// - Track user preferences and settings
// - Include registration and last login dates

// Sample user document structure:
db.users.insertOne({
  // TODO: Complete this document structure
  name: "John Doe",
  email: "john@example.com",
  // Add more fields...
});

// TODO: Create a products collection with embedded reviews
// Requirements:
// - Product information (name, description, price, category)
// - Embed product reviews (limited to recent 10 reviews)
// - Include inventory information
// - Store product images and specifications

// Sample product document:
db.products.insertOne({
  // TODO: Complete this document structure
  name: "MacBook Pro 16\"",
  price: 2499.99,
  // Add more fields...
});

// TODO: Create an orders collection with embedded order items
// Requirements:
// - Reference to user (user_id)
// - Embed order items with product details
// - Order status and tracking information
// - Shipping and billing addresses
// - Payment information

// Sample order document:
db.orders.insertOne({
  // TODO: Complete this document structure
  user_id: ObjectId("..."),
  status: "pending",
  // Add more fields...
});

// ============================================================================
// EXERCISE 2: Insert sample data
// ============================================================================

// TODO: Insert 5 users with different profiles
// Include users with multiple addresses, different preferences

// TODO: Insert 20 products across different categories
// Categories: Electronics, Books, Clothing, Home, Sports
// Include products with different price ranges and ratings

// TODO: Insert 15 orders with various statuses
// Statuses: pending, processing, shipped, delivered, cancelled
// Include orders with multiple items

// ============================================================================
// EXERCISE 3: Basic queries
// ============================================================================

// TODO: Find all users from a specific city
// Query: Find users with addresses in "New York"

// TODO: Find products in a price range
// Query: Find products between $100 and $500

// TODO: Find products by category with high ratings
// Query: Find Electronics products with average rating >= 4.0

// TODO: Find orders by status
// Query: Find all "shipped" orders

// TODO: Find user's order history
// Query: Find all orders for a specific user (use a real user_id from your data)

// ============================================================================
// EXERCISE 4: Complex queries with arrays and nested documents
// ============================================================================

// TODO: Find users with specific preferences
// Query: Find users who prefer "email" notifications

// TODO: Find products with specific features
// Query: Find products that have "wireless" in specifications

// TODO: Find orders containing specific products
// Query: Find orders that contain "MacBook" products

// TODO: Text search on products
// First create a text index, then search for products
// Create index: db.products.createIndex({ name: "text", description: "text" })
// Query: Search for products containing "laptop computer"

// ============================================================================
// EXERCISE 5: Aggregation pipeline queries
// ============================================================================

// TODO: Calculate total revenue by category
// Use aggregation to:
// 1. Match completed orders
// 2. Unwind order items
// 3. Group by product category
// 4. Calculate total revenue per category
// 5. Sort by revenue (highest first)

db.orders.aggregate([
  // TODO: Complete this aggregation pipeline
]);

// TODO: Find top customers by total spending
// Use aggregation to:
// 1. Match completed orders
// 2. Group by user_id
// 3. Calculate total spent per user
// 4. Sort by total spent (highest first)
// 5. Limit to top 5 customers
// 6. Lookup user details from users collection

db.orders.aggregate([
  // TODO: Complete this aggregation pipeline
]);

// TODO: Calculate average order value by month
// Use aggregation to:
// 1. Match orders from 2024
// 2. Group by year and month
// 3. Calculate average order value per month
// 4. Sort by month

db.orders.aggregate([
  // TODO: Complete this aggregation pipeline
]);

// TODO: Product performance analysis
// Use aggregation to:
// 1. Unwind order items
// 2. Group by product name
// 3. Calculate total quantity sold, total revenue, average price
// 4. Sort by total revenue
// 5. Limit to top 10 products

db.orders.aggregate([
  // TODO: Complete this aggregation pipeline
]);

// ============================================================================
// EXERCISE 6: Create appropriate indexes
// ============================================================================

// TODO: Create indexes for common queries
// Consider these query patterns:
// - Find user by email (login)
// - Find products by category
// - Find products by price range
// - Find orders by user_id
// - Find orders by status
// - Text search on products
// - Find orders by date range

// Example:
// db.users.createIndex({ email: 1 });

// TODO: Create compound indexes for complex queries
// Consider queries that filter on multiple fields

// TODO: Create a partial index
// Example: Index only active users or available products

// TODO: Create a TTL index
// Example: Expire user sessions or temporary data

// ============================================================================
// EXERCISE 7: Update operations
// ============================================================================

// TODO: Update user preferences
// Add a new preference to a user's preferences array

// TODO: Add a review to a product
// Use $push to add a new review to a product's reviews array
// Include review text, rating, reviewer name, and date

// TODO: Update order status
// Change an order status from "pending" to "processing"
// Also update the status_history array with timestamp

// TODO: Increment product view count
// Use $inc to increment a product's view_count field

// TODO: Update inventory after purchase
// Decrease product inventory count when an order is placed

// ============================================================================
// EXERCISE 8: Advanced operations
// ============================================================================

// TODO: Implement shopping cart functionality
// Create a separate carts collection or embed cart in user document
// Include methods to:
// - Add item to cart
// - Remove item from cart
// - Update item quantity
// - Calculate cart total

// TODO: Implement product recommendations
// Find products that are:
// - In the same category as user's previous purchases
// - Frequently bought together with items in user's cart
// - Highly rated products the user hasn't purchased

// TODO: Create a data validation schema
// Use MongoDB schema validation to ensure:
// - Email format is valid
// - Price is a positive number
// - Required fields are present

// Example schema validation:
db.createCollection("validated_users", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["name", "email"],
      properties: {
        name: {
          bsonType: "string",
          description: "must be a string and is required"
        },
        email: {
          bsonType: "string",
          pattern: "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
          description: "must be a valid email address"
        }
      }
    }
  }
});

// ============================================================================
// EXERCISE 9: Performance analysis
// ============================================================================

// TODO: Analyze query performance
// Use explain() to analyze the performance of your queries
// Example: db.users.find({ email: "john@example.com" }).explain("executionStats")

// TODO: Identify slow queries
// Run some queries and note their execution times
// Identify which queries would benefit from indexes

// TODO: Compare performance before and after adding indexes
// Run the same query before and after creating an index
// Note the difference in execution time and explain output

// ============================================================================
// EXERCISE 10: Real-world scenarios
// ============================================================================

// TODO: Handle inventory management
// Create a function that:
// 1. Checks if product is in stock
// 2. Reserves inventory when order is placed
// 3. Releases inventory if order is cancelled
// 4. Updates inventory when order is completed

// TODO: Implement order processing workflow
// Create operations for:
// 1. Create order (validate inventory, calculate totals)
// 2. Process payment (update order status)
// 3. Ship order (update status, add tracking info)
// 4. Complete order (final status update)

// TODO: Generate business reports
// Create aggregation queries for:
// 1. Daily sales report
// 2. Top-selling products by category
// 3. Customer lifetime value
// 4. Inventory levels by category
// 5. Revenue trends over time

// ============================================================================
// BONUS: Advanced patterns
// ============================================================================

// TODO: Implement the Subset Pattern
// Store frequently accessed product data in order items
// Keep full product details in products collection

// TODO: Implement the Bucket Pattern
// Group time-series data (like page views or sales) by time periods

// TODO: Implement the Polymorphic Pattern
// Store different types of events in a single events collection

// ============================================================================
// QUESTIONS TO CONSIDER
// ============================================================================

/*
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
*/

// ============================================================================
// CLEANUP (optional)
// ============================================================================

// Uncomment to drop collections and start fresh
// db.users.drop();
// db.products.drop();
// db.orders.drop();
// db.carts.drop();