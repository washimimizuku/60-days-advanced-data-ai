// Day 2: NoSQL Databases - MongoDB Patterns
// SOLUTION: Build an E-commerce System

// ============================================================================
// SETUP: Connect to MongoDB and create sample data
// ============================================================================

// Connect to MongoDB
use('ecommerce_db');

// ============================================================================
// EXERCISE 1: Design and create collections
// ============================================================================

// Solution 1: Users collection with embedded addresses
db.users.insertOne({
  _id: ObjectId(),
  name: "John Doe",
  email: "john@example.com",
  password_hash: "<hashed_password>",
  phone: "+1-555-0123",
  date_of_birth: ISODate("1990-05-15"),
  addresses: [
    {
      type: "home",
      street: "123 Main St",
      city: "New York",
      state: "NY",
      zipcode: "10001",
      country: "USA",
      is_default: true
    },
    {
      type: "work", 
      street: "456 Business Ave",
      city: "New York",
      state: "NY",
      zipcode: "10002",
      country: "USA",
      is_default: false
    }
  ],
  preferences: {
    notifications: ["email", "sms"],
    currency: "USD",
    language: "en",
    newsletter: true
  },
  created_at: ISODate("2024-01-01T00:00:00Z"),
  last_login: ISODate("2024-01-15T10:30:00Z"),
  status: "active"
});

// Solution 2: Products collection with embedded reviews
db.products.insertOne({
  _id: ObjectId(),
  name: "MacBook Pro 16\"",
  description: "Powerful laptop for professionals with M3 Pro chip",
  price: NumberDecimal("2499.99"),
  category: "Electronics",
  subcategory: "Laptops",
  brand: "Apple",
  sku: "MBP16-M3-512",
  images: [
    "https://example.com/images/mbp16-1.jpg",
    "https://example.com/images/mbp16-2.jpg"
  ],
  specifications: {
    processor: "Apple M3 Pro",
    memory: "18GB",
    storage: "512GB SSD",
    display: "16-inch Liquid Retina XDR",
    weight: "2.1 kg",
    color: "Space Gray"
  },
  inventory: {
    quantity: 50,
    reserved: 5,
    available: 45,
    warehouse_location: "NYC-01"
  },
  reviews: [
    {
      _id: ObjectId(),
      user_id: ObjectId(),
      user_name: "Alice Johnson",
      rating: 5,
      title: "Excellent laptop!",
      comment: "Fast, reliable, great for development work.",
      date: ISODate("2024-01-10T00:00:00Z"),
      verified_purchase: true,
      helpful_votes: 12
    }
  ],
  rating_summary: {
    average: 4.8,
    count: 156,
    distribution: {
      "5": 120,
      "4": 25,
      "3": 8,
      "2": 2,
      "1": 1
    }
  },
  tags: ["laptop", "apple", "professional", "m3"],
  created_at: ISODate("2024-01-01T00:00:00Z"),
  updated_at: ISODate("2024-01-15T00:00:00Z"),
  status: "active",
  view_count: 1250
});

// Solution 3: Orders collection with embedded order items
db.orders.insertOne({
  _id: ObjectId(),
  order_number: "ORD-2024-001",
  user_id: ObjectId("user_id_here"),
  status: "pending",
  items: [
    {
      product_id: ObjectId("product_id_here"),
      product_name: "MacBook Pro 16\"",
      sku: "MBP16-M3-512",
      price: NumberDecimal("2499.99"),
      quantity: 1,
      subtotal: NumberDecimal("2499.99")
    },
    {
      product_id: ObjectId("another_product_id"),
      product_name: "Magic Mouse",
      sku: "MM-WHITE",
      price: NumberDecimal("79.99"),
      quantity: 1,
      subtotal: NumberDecimal("79.99")
    }
  ],
  pricing: {
    subtotal: NumberDecimal("2579.98"),
    tax: NumberDecimal("206.40"),
    shipping: NumberDecimal("0.00"),
    discount: NumberDecimal("0.00"),
    total: NumberDecimal("2786.38")
  },
  shipping_address: {
    name: "John Doe",
    street: "123 Main St",
    city: "New York",
    state: "NY",
    zipcode: "10001",
    country: "USA"
  },
  billing_address: {
    name: "John Doe",
    street: "123 Main St", 
    city: "New York",
    state: "NY",
    zipcode: "10001",
    country: "USA"
  },
  payment: {
    method: "credit_card",
    card_last_four: "1234",
    transaction_id: "txn_abc123",
    status: "authorized"
  },
  status_history: [
    {
      status: "pending",
      timestamp: ISODate("2024-01-15T10:00:00Z"),
      note: "Order created"
    }
  ],
  created_at: ISODate("2024-01-15T10:00:00Z"),
  updated_at: ISODate("2024-01-15T10:00:00Z"),
  estimated_delivery: ISODate("2024-01-20T00:00:00Z")
});
// ============================================================================
// EXERCISE 2: Insert sample data
// ============================================================================

// Solution: Insert 5 users with different profiles
db.users.insertMany([
  {
    name: "Alice Johnson",
    email: "alice@example.com",
    phone: "+1-555-0124",
    addresses: [
      {
        type: "home",
        street: "789 Oak Ave",
        city: "San Francisco",
        state: "CA",
        zipcode: "94102",
        country: "USA",
        is_default: true
      }
    ],
    preferences: {
      notifications: ["email"],
      currency: "USD",
      language: "en"
    },
    created_at: ISODate("2024-01-02T00:00:00Z"),
    status: "active"
  },
  {
    name: "Bob Smith",
    email: "bob@example.com",
    phone: "+1-555-0125",
    addresses: [
      {
        type: "home",
        street: "321 Pine St",
        city: "Seattle",
        state: "WA",
        zipcode: "98101",
        country: "USA",
        is_default: true
      },
      {
        type: "work",
        street: "654 Tech Blvd",
        city: "Seattle", 
        state: "WA",
        zipcode: "98109",
        country: "USA",
        is_default: false
      }
    ],
    preferences: {
      notifications: ["email", "sms"],
      currency: "USD",
      language: "en"
    },
    created_at: ISODate("2024-01-03T00:00:00Z"),
    status: "active"
  },
  {
    name: "Carol Davis",
    email: "carol@example.com",
    phone: "+1-555-0126",
    addresses: [
      {
        type: "home",
        street: "987 Elm Dr",
        city: "Austin",
        state: "TX",
        zipcode: "73301",
        country: "USA",
        is_default: true
      }
    ],
    preferences: {
      notifications: ["email"],
      currency: "USD",
      language: "en"
    },
    created_at: ISODate("2024-01-04T00:00:00Z"),
    status: "inactive"
  },
  {
    name: "David Wilson",
    email: "david@example.com",
    phone: "+1-555-0127",
    addresses: [
      {
        type: "home",
        street: "147 Maple Ln",
        city: "Denver",
        state: "CO",
        zipcode: "80201",
        country: "USA",
        is_default: true
      }
    ],
    preferences: {
      notifications: ["sms"],
      currency: "USD",
      language: "en"
    },
    created_at: ISODate("2024-01-05T00:00:00Z"),
    status: "active"
  },
  {
    name: "Eva Brown",
    email: "eva@example.com",
    phone: "+1-555-0128",
    addresses: [
      {
        type: "home",
        street: "258 Cedar Ave",
        city: "Miami",
        state: "FL",
        zipcode: "33101",
        country: "USA",
        is_default: true
      }
    ],
    preferences: {
      notifications: ["email", "push"],
      currency: "USD",
      language: "es"
    },
    created_at: ISODate("2024-01-06T00:00:00Z"),
    status: "active"
  }
]);

// Solution: Insert 20 products across different categories
db.products.insertMany([
  // Electronics
  {
    name: "iPhone 15 Pro",
    description: "Latest iPhone with titanium design",
    price: NumberDecimal("999.99"),
    category: "Electronics",
    subcategory: "Smartphones",
    brand: "Apple",
    inventory: { quantity: 100, available: 95 },
    rating_summary: { average: 4.7, count: 89 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  },
  {
    name: "Samsung Galaxy S24",
    description: "Android flagship with AI features",
    price: NumberDecimal("899.99"),
    category: "Electronics",
    subcategory: "Smartphones", 
    brand: "Samsung",
    inventory: { quantity: 80, available: 75 },
    rating_summary: { average: 4.5, count: 67 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  },
  {
    name: "Sony WH-1000XM5",
    description: "Premium noise-canceling headphones",
    price: NumberDecimal("399.99"),
    category: "Electronics",
    subcategory: "Audio",
    brand: "Sony",
    inventory: { quantity: 60, available: 58 },
    rating_summary: { average: 4.8, count: 234 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  },
  // Books
  {
    name: "The Pragmatic Programmer",
    description: "Classic programming book",
    price: NumberDecimal("49.99"),
    category: "Books",
    subcategory: "Technology",
    brand: "Addison-Wesley",
    inventory: { quantity: 200, available: 195 },
    rating_summary: { average: 4.9, count: 1250 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  },
  {
    name: "Clean Code",
    description: "A handbook of agile software craftsmanship",
    price: NumberDecimal("44.99"),
    category: "Books",
    subcategory: "Technology",
    brand: "Prentice Hall",
    inventory: { quantity: 150, available: 148 },
    rating_summary: { average: 4.6, count: 890 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  },
  // Clothing
  {
    name: "Levi's 501 Jeans",
    description: "Classic straight-leg jeans",
    price: NumberDecimal("89.99"),
    category: "Clothing",
    subcategory: "Jeans",
    brand: "Levi's",
    inventory: { quantity: 120, available: 115 },
    rating_summary: { average: 4.4, count: 456 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  },
  {
    name: "Nike Air Max 90",
    description: "Iconic running shoes",
    price: NumberDecimal("119.99"),
    category: "Clothing",
    subcategory: "Shoes",
    brand: "Nike",
    inventory: { quantity: 90, available: 87 },
    rating_summary: { average: 4.3, count: 678 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  },
  // Home
  {
    name: "Instant Pot Duo 7-in-1",
    description: "Multi-use pressure cooker",
    price: NumberDecimal("79.99"),
    category: "Home",
    subcategory: "Kitchen",
    brand: "Instant Pot",
    inventory: { quantity: 75, available: 72 },
    rating_summary: { average: 4.7, count: 2340 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  },
  {
    name: "Dyson V15 Detect",
    description: "Cordless vacuum cleaner",
    price: NumberDecimal("749.99"),
    category: "Home",
    subcategory: "Cleaning",
    brand: "Dyson",
    inventory: { quantity: 40, available: 38 },
    rating_summary: { average: 4.6, count: 567 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  },
  // Sports
  {
    name: "Peloton Bike+",
    description: "Interactive exercise bike",
    price: NumberDecimal("2495.00"),
    category: "Sports",
    subcategory: "Fitness",
    brand: "Peloton",
    inventory: { quantity: 25, available: 23 },
    rating_summary: { average: 4.5, count: 123 },
    created_at: ISODate("2024-01-01T00:00:00Z"),
    status: "active"
  }
]);

// ============================================================================
// EXERCISE 3: Basic queries
// ============================================================================

// Solution 1: Find all users from a specific city
db.users.find({ "addresses.city": "New York" });

// Solution 2: Find products in a price range
db.products.find({ 
  price: { 
    $gte: NumberDecimal("100.00"), 
    $lte: NumberDecimal("500.00") 
  } 
});

// Solution 3: Find products by category with high ratings
db.products.find({ 
  category: "Electronics",
  "rating_summary.average": { $gte: 4.0 }
});

// Solution 4: Find orders by status
db.orders.find({ status: "shipped" });

// Solution 5: Find user's order history
// First, get a user ID
const userId = db.users.findOne({ email: "john@example.com" })._id;
db.orders.find({ user_id: userId });

// ============================================================================
// EXERCISE 4: Complex queries with arrays and nested documents
// ============================================================================

// Solution 1: Find users with specific preferences
db.users.find({ "preferences.notifications": "email" });

// Solution 2: Find products with specific features
db.products.find({ "specifications.processor": /Apple/i });

// Solution 3: Find orders containing specific products
db.orders.find({ "items.product_name": /MacBook/i });

// Solution 4: Text search on products
// Create text index
db.products.createIndex({ name: "text", description: "text" });

// Search for products
db.products.find({ $text: { $search: "laptop computer" } });

// ============================================================================
// EXERCISE 5: Aggregation pipeline queries
// ============================================================================

// Solution 1: Calculate total revenue by category
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $unwind: "$items" },
  { $lookup: {
      from: "products",
      localField: "items.product_id",
      foreignField: "_id",
      as: "product_info"
  }},
  { $unwind: "$product_info" },
  { $group: {
      _id: "$product_info.category",
      total_revenue: { $sum: "$items.subtotal" },
      total_quantity: { $sum: "$items.quantity" },
      order_count: { $sum: 1 }
  }},
  { $sort: { total_revenue: -1 } }
]);

// Solution 2: Find top customers by total spending
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $group: {
      _id: "$user_id",
      total_spent: { $sum: "$pricing.total" },
      order_count: { $sum: 1 }
  }},
  { $sort: { total_spent: -1 } },
  { $limit: 5 },
  { $lookup: {
      from: "users",
      localField: "_id",
      foreignField: "_id",
      as: "user_info"
  }},
  { $unwind: "$user_info" },
  { $project: {
      user_name: "$user_info.name",
      user_email: "$user_info.email",
      total_spent: 1,
      order_count: 1
  }}
]);

// Solution 3: Calculate average order value by month
db.orders.aggregate([
  { $match: { 
      created_at: { $gte: ISODate("2024-01-01") },
      status: "completed"
  }},
  { $group: {
      _id: {
          year: { $year: "$created_at" },
          month: { $month: "$created_at" }
      },
      avg_order_value: { $avg: "$pricing.total" },
      total_orders: { $sum: 1 },
      total_revenue: { $sum: "$pricing.total" }
  }},
  { $sort: { "_id.year": 1, "_id.month": 1 } }
]);

// Solution 4: Product performance analysis
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $unwind: "$items" },
  { $group: {
      _id: "$items.product_name",
      total_quantity_sold: { $sum: "$items.quantity" },
      total_revenue: { $sum: "$items.subtotal" },
      avg_price: { $avg: "$items.price" },
      order_count: { $sum: 1 }
  }},
  { $addFields: {
      revenue_per_unit: { $divide: ["$total_revenue", "$total_quantity_sold"] }
  }},
  { $sort: { total_revenue: -1 } },
  { $limit: 10 }
]);

// ============================================================================
// EXERCISE 6: Create appropriate indexes
// ============================================================================

// Solution: Create indexes for common queries
db.users.createIndex({ email: 1 });  // Login queries
db.users.createIndex({ "addresses.city": 1 });  // Location-based queries
db.users.createIndex({ status: 1 });  // Filter by user status

db.products.createIndex({ category: 1 });  // Category filtering
db.products.createIndex({ price: 1 });  // Price range queries
db.products.createIndex({ "rating_summary.average": -1 });  // Sort by rating
db.products.createIndex({ status: 1, category: 1 });  // Compound index

db.orders.createIndex({ user_id: 1 });  // User's orders
db.orders.createIndex({ status: 1 });  // Order status filtering
db.orders.createIndex({ created_at: -1 });  // Recent orders
db.orders.createIndex({ user_id: 1, created_at: -1 });  // User's recent orders

// Text search index
db.products.createIndex({ name: "text", description: "text" });

// Partial index for active products only
db.products.createIndex(
  { category: 1, price: 1 },
  { partialFilterExpression: { status: "active" } }
);

// TTL index for user sessions (if you had a sessions collection)
// db.sessions.createIndex({ created_at: 1 }, { expireAfterSeconds: 3600 });

// ============================================================================
// EXERCISE 7: Update operations
// ============================================================================

// Solution 1: Update user preferences
db.users.updateOne(
  { email: "john@example.com" },
  { $addToSet: { "preferences.notifications": "push" } }
);

// Solution 2: Add a review to a product
db.products.updateOne(
  { name: "iPhone 15 Pro" },
  { 
    $push: { 
      reviews: {
        _id: ObjectId(),
        user_id: ObjectId(),
        user_name: "John Doe",
        rating: 5,
        title: "Great phone!",
        comment: "Love the new titanium design and camera quality.",
        date: new Date(),
        verified_purchase: true,
        helpful_votes: 0
      }
    },
    $inc: { 
      "rating_summary.count": 1,
      "rating_summary.distribution.5": 1
    }
  }
);

// Solution 3: Update order status
db.orders.updateOne(
  { order_number: "ORD-2024-001" },
  { 
    $set: { 
      status: "processing",
      updated_at: new Date()
    },
    $push: {
      status_history: {
        status: "processing",
        timestamp: new Date(),
        note: "Payment confirmed, preparing for shipment"
      }
    }
  }
);

// Solution 4: Increment product view count
db.products.updateOne(
  { name: "MacBook Pro 16\"" },
  { $inc: { view_count: 1 } }
);

// Solution 5: Update inventory after purchase
db.products.updateOne(
  { name: "MacBook Pro 16\"" },
  { 
    $inc: { 
      "inventory.available": -1,
      "inventory.reserved": 1
    }
  }
);
// ============================================================================
// EXERCISE 8: Advanced operations
// ============================================================================

// Solution 1: Implement shopping cart functionality
// Create carts collection
db.carts.insertOne({
  _id: ObjectId(),
  user_id: ObjectId("user_id_here"),
  items: [
    {
      product_id: ObjectId("product_id_here"),
      product_name: "MacBook Pro 16\"",
      price: NumberDecimal("2499.99"),
      quantity: 1,
      added_at: new Date()
    }
  ],
  created_at: new Date(),
  updated_at: new Date()
});

// Add item to cart
function addToCart(userId, productId, quantity) {
  return db.carts.updateOne(
    { user_id: userId },
    { 
      $addToSet: { 
        items: {
          product_id: productId,
          quantity: quantity,
          added_at: new Date()
        }
      },
      $set: { updated_at: new Date() }
    },
    { upsert: true }
  );
}

// Remove item from cart
function removeFromCart(userId, productId) {
  return db.carts.updateOne(
    { user_id: userId },
    { 
      $pull: { items: { product_id: productId } },
      $set: { updated_at: new Date() }
    }
  );
}

// Update item quantity
function updateCartQuantity(userId, productId, newQuantity) {
  return db.carts.updateOne(
    { 
      user_id: userId,
      "items.product_id": productId
    },
    { 
      $set: { 
        "items.$.quantity": newQuantity,
        updated_at: new Date()
      }
    }
  );
}

// Calculate cart total
db.carts.aggregate([
  { $match: { user_id: ObjectId("user_id_here") } },
  { $unwind: "$items" },
  { $lookup: {
      from: "products",
      localField: "items.product_id",
      foreignField: "_id",
      as: "product"
  }},
  { $unwind: "$product" },
  { $group: {
      _id: "$_id",
      total: { $sum: { $multiply: ["$product.price", "$items.quantity"] } },
      item_count: { $sum: "$items.quantity" }
  }}
]);

// Solution 2: Implement product recommendations
// Find products in same category as user's previous purchases
db.orders.aggregate([
  { $match: { user_id: ObjectId("user_id_here"), status: "completed" } },
  { $unwind: "$items" },
  { $lookup: {
      from: "products",
      localField: "items.product_id",
      foreignField: "_id",
      as: "product"
  }},
  { $unwind: "$product" },
  { $group: { _id: "$product.category" } },
  { $lookup: {
      from: "products",
      let: { category: "$_id" },
      pipeline: [
        { $match: { 
            $expr: { $eq: ["$category", "$$category"] },
            status: "active"
        }},
        { $sort: { "rating_summary.average": -1 } },
        { $limit: 5 }
      ],
      as: "recommended_products"
  }}
]);

// Find frequently bought together products
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $match: { "items.product_id": ObjectId("target_product_id") } },
  { $unwind: "$items" },
  { $match: { "items.product_id": { $ne: ObjectId("target_product_id") } } },
  { $group: {
      _id: "$items.product_id",
      frequency: { $sum: 1 },
      product_name: { $first: "$items.product_name" }
  }},
  { $sort: { frequency: -1 } },
  { $limit: 5 }
]);

// Solution 3: Create data validation schema
db.createCollection("validated_products", {
  validator: {
    $jsonSchema: {
      bsonType: "object",
      required: ["name", "price", "category", "inventory"],
      properties: {
        name: {
          bsonType: "string",
          minLength: 1,
          maxLength: 200,
          description: "Product name is required and must be 1-200 characters"
        },
        price: {
          bsonType: "decimal",
          minimum: 0,
          description: "Price must be a positive decimal"
        },
        category: {
          bsonType: "string",
          enum: ["Electronics", "Books", "Clothing", "Home", "Sports"],
          description: "Category must be one of the predefined values"
        },
        inventory: {
          bsonType: "object",
          required: ["quantity", "available"],
          properties: {
            quantity: {
              bsonType: "int",
              minimum: 0,
              description: "Quantity must be a non-negative integer"
            },
            available: {
              bsonType: "int",
              minimum: 0,
              description: "Available must be a non-negative integer"
            }
          }
        },
        rating_summary: {
          bsonType: "object",
          properties: {
            average: {
              bsonType: "double",
              minimum: 0,
              maximum: 5,
              description: "Average rating must be between 0 and 5"
            }
          }
        }
      }
    }
  }
});

// ============================================================================
// EXERCISE 9: Performance analysis
// ============================================================================

// Solution: Analyze query performance
// Before creating index
db.users.find({ email: "john@example.com" }).explain("executionStats");
// Result: COLLSCAN (collection scan) - slow

// After creating index
db.users.createIndex({ email: 1 });
db.users.find({ email: "john@example.com" }).explain("executionStats");
// Result: IXSCAN (index scan) - fast

// Compare compound index usage
db.products.find({ category: "Electronics", price: { $gte: 100 } }).explain("executionStats");

// Check index usage statistics
db.products.aggregate([
  { $indexStats: {} }
]);

// ============================================================================
// EXERCISE 10: Real-world scenarios
// ============================================================================

// Solution 1: Handle inventory management
function checkInventory(productId, requestedQuantity) {
  const product = db.products.findOne({ _id: productId });
  return product && product.inventory.available >= requestedQuantity;
}

function reserveInventory(productId, quantity) {
  return db.products.updateOne(
    { 
      _id: productId,
      "inventory.available": { $gte: quantity }
    },
    { 
      $inc: { 
        "inventory.available": -quantity,
        "inventory.reserved": quantity
      }
    }
  );
}

function releaseInventory(productId, quantity) {
  return db.products.updateOne(
    { _id: productId },
    { 
      $inc: { 
        "inventory.available": quantity,
        "inventory.reserved": -quantity
      }
    }
  );
}

function confirmInventory(productId, quantity) {
  return db.products.updateOne(
    { _id: productId },
    { 
      $inc: { 
        "inventory.reserved": -quantity,
        "inventory.quantity": -quantity
      }
    }
  );
}

// Solution 2: Order processing workflow
function createOrder(userId, items, shippingAddress) {
  // Validate inventory
  for (let item of items) {
    if (!checkInventory(item.product_id, item.quantity)) {
      throw new Error(`Insufficient inventory for product ${item.product_id}`);
    }
  }
  
  // Reserve inventory
  for (let item of items) {
    reserveInventory(item.product_id, item.quantity);
  }
  
  // Calculate totals
  const subtotal = items.reduce((sum, item) => sum + (item.price * item.quantity), 0);
  const tax = subtotal * 0.08; // 8% tax
  const shipping = subtotal > 100 ? 0 : 9.99;
  const total = subtotal + tax + shipping;
  
  // Create order
  return db.orders.insertOne({
    order_number: `ORD-${Date.now()}`,
    user_id: userId,
    status: "pending",
    items: items,
    pricing: {
      subtotal: NumberDecimal(subtotal.toString()),
      tax: NumberDecimal(tax.toString()),
      shipping: NumberDecimal(shipping.toString()),
      total: NumberDecimal(total.toString())
    },
    shipping_address: shippingAddress,
    status_history: [{
      status: "pending",
      timestamp: new Date(),
      note: "Order created"
    }],
    created_at: new Date()
  });
}

// Solution 3: Generate business reports
// Daily sales report
db.orders.aggregate([
  { $match: { 
      created_at: { 
        $gte: ISODate("2024-01-15T00:00:00Z"),
        $lt: ISODate("2024-01-16T00:00:00Z")
      },
      status: "completed"
  }},
  { $group: {
      _id: null,
      total_orders: { $sum: 1 },
      total_revenue: { $sum: "$pricing.total" },
      avg_order_value: { $avg: "$pricing.total" }
  }}
]);

// Top-selling products by category
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $unwind: "$items" },
  { $lookup: {
      from: "products",
      localField: "items.product_id",
      foreignField: "_id",
      as: "product"
  }},
  { $unwind: "$product" },
  { $group: {
      _id: {
        category: "$product.category",
        product_name: "$items.product_name"
      },
      total_sold: { $sum: "$items.quantity" },
      total_revenue: { $sum: "$items.subtotal" }
  }},
  { $sort: { "_id.category": 1, total_sold: -1 } },
  { $group: {
      _id: "$_id.category",
      top_products: { 
        $push: {
          product_name: "$_id.product_name",
          total_sold: "$total_sold",
          total_revenue: "$total_revenue"
        }
      }
  }},
  { $project: {
      category: "$_id",
      top_products: { $slice: ["$top_products", 3] }
  }}
]);

// Customer lifetime value
db.orders.aggregate([
  { $match: { status: "completed" } },
  { $group: {
      _id: "$user_id",
      total_spent: { $sum: "$pricing.total" },
      order_count: { $sum: 1 },
      first_order: { $min: "$created_at" },
      last_order: { $max: "$created_at" }
  }},
  { $addFields: {
      customer_lifetime_days: { 
        $divide: [
          { $subtract: ["$last_order", "$first_order"] },
          1000 * 60 * 60 * 24
        ]
      }
  }},
  { $lookup: {
      from: "users",
      localField: "_id",
      foreignField: "_id",
      as: "user"
  }},
  { $unwind: "$user" },
  { $project: {
      user_name: "$user.name",
      user_email: "$user.email",
      total_spent: 1,
      order_count: 1,
      avg_order_value: { $divide: ["$total_spent", "$order_count"] },
      customer_lifetime_days: 1
  }},
  { $sort: { total_spent: -1 } }
]);

// ============================================================================
// BONUS: Advanced patterns
// ============================================================================

// Solution 1: Subset Pattern implementation
// Store frequently accessed product data in order items
db.orders.updateOne(
  { _id: ObjectId("order_id") },
  { $set: { 
      "items.$[].product_subset": {
        name: "Product Name",
        image: "product_image.jpg",
        category: "Electronics"
      }
  }}
);

// Solution 2: Bucket Pattern for time-series data
db.page_views.insertOne({
  _id: ObjectId(),
  date: ISODate("2024-01-15T10:00:00Z"),
  page: "/products/macbook-pro",
  views: [
    { timestamp: ISODate("2024-01-15T10:00:00Z"), user_id: ObjectId(), session_id: "sess1" },
    { timestamp: ISODate("2024-01-15T10:01:00Z"), user_id: ObjectId(), session_id: "sess2" },
    { timestamp: ISODate("2024-01-15T10:02:00Z"), user_id: ObjectId(), session_id: "sess3" }
    // ... up to 60 views per hour
  ],
  total_views: 3
});

// Solution 3: Polymorphic Pattern for events
db.events.insertMany([
  {
    _id: ObjectId(),
    type: "user_login",
    user_id: ObjectId(),
    timestamp: ISODate("2024-01-15T10:00:00Z"),
    ip_address: "192.168.1.1",
    user_agent: "Mozilla/5.0..."
  },
  {
    _id: ObjectId(),
    type: "product_view",
    user_id: ObjectId(),
    timestamp: ISODate("2024-01-15T10:01:00Z"),
    product_id: ObjectId(),
    product_name: "MacBook Pro 16\"",
    category: "Electronics"
  },
  {
    _id: ObjectId(),
    type: "purchase",
    user_id: ObjectId(),
    timestamp: ISODate("2024-01-15T10:30:00Z"),
    order_id: ObjectId(),
    total_amount: NumberDecimal("2499.99"),
    item_count: 1
  }
]);

// Query events by type
db.events.find({ type: "purchase" });

// Aggregate events by type and date
db.events.aggregate([
  { $match: { 
      timestamp: { $gte: ISODate("2024-01-15T00:00:00Z") }
  }},
  { $group: {
      _id: {
        type: "$type",
        date: { $dateToString: { format: "%Y-%m-%d", date: "$timestamp" } }
      },
      count: { $sum: 1 }
  }},
  { $sort: { "_id.date": 1, "_id.type": 1 } }
]);

// ============================================================================
// ANSWERS TO QUESTIONS
// ============================================================================

/*
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
*/

// ============================================================================
// CLEANUP (optional)
// ============================================================================

// Uncomment to drop collections and start fresh
// db.users.drop();
// db.products.drop();
// db.orders.drop();
// db.carts.drop();
// db.events.drop();
// db.page_views.drop();