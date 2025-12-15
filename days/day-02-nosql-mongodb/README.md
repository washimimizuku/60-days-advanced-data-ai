# Day 2: NoSQL Databases - MongoDB Patterns

## 📖 Learning Objectives

By the end of today, you will:
- Understand when to choose MongoDB over relational databases
- Master MongoDB document modeling patterns
- Implement efficient queries and aggregation pipelines
- Apply indexing strategies for MongoDB collections
- Design schemas for real-world applications

---

## Theory

### Why MongoDB?

MongoDB is a document-oriented NoSQL database that stores data in flexible, JSON-like documents. It's designed for modern applications that need:

**Key advantages**:
- **Flexible schema**: Add fields without migrations
- **Natural data structures**: Store objects as they exist in code
- **Horizontal scaling**: Built-in sharding support
- **High performance**: Optimized for read-heavy workloads
- **Rich queries**: Complex aggregations and text search

**When to use MongoDB**:
- Rapidly evolving schemas
- Hierarchical/nested data structures
- Content management systems
- Real-time analytics
- IoT and sensor data
- Catalog and inventory systems

**When NOT to use MongoDB**:
- Complex transactions across multiple entities
- Strong consistency requirements
- Frequent complex relational operations (MongoDB supports joins via $lookup but they're less efficient than SQL joins)
- Financial systems requiring strict ACID guarantees across documents

### Document Model vs Relational

**Relational approach** (normalized):
```sql
-- Users table
users: { id: 1, name: "John", email: "john@example.com" }

-- Addresses table  
addresses: { id: 1, user_id: 1, street: "123 Main St", city: "NYC" }

-- Orders table
orders: { id: 1, user_id: 1, total: 99.99, date: "2024-01-01" }

-- Order items table
order_items: { id: 1, order_id: 1, product: "Laptop", price: 99.99 }
```

**MongoDB approach** (denormalized):
```javascript
// Single document contains everything
{
  _id: ObjectId("..."),
  name: "John",
  email: "john@example.com",
  addresses: [
    { street: "123 Main St", city: "NYC", type: "home" },
    { street: "456 Work Ave", city: "NYC", type: "work" }
  ],
  orders: [
    {
      _id: ObjectId("..."),
      total: 99.99,
      date: ISODate("2024-01-01"),
      items: [
        { product: "Laptop", price: 99.99, quantity: 1 }
      ]
    }
  ]
}
```

### Core Concepts

#### 1. Documents and Collections

```javascript
// Document - like a JSON object
{
  _id: ObjectId("507f1f77bcf86cd799439011"),
  name: "John Doe",
  age: 30,
  email: "john@example.com",
  tags: ["developer", "mongodb"],
  address: {
    street: "123 Main St",
    city: "New York",
    zipcode: "10001"
  },
  created_at: ISODate("2024-01-01T00:00:00Z")
}

// Collection - group of documents (like a table)
db.users.insertOne({...})
```

**Key features**:
- **_id field**: Unique identifier (auto-generated if not provided)
- **Flexible schema**: Documents in same collection can have different fields
- **Rich data types**: Strings, numbers, dates, arrays, objects, binary data
- **Limited joins**: Related data is typically embedded or referenced; $lookup provides join capability but is less efficient than SQL joins

#### 2. Data Modeling Patterns

**Embedding Pattern** - Store related data together:
```javascript
// Blog post with comments embedded
{
  _id: ObjectId("..."),
  title: "MongoDB Patterns",
  content: "...",
  author: "John Doe",
  comments: [
    {
      author: "Jane",
      text: "Great post!",
      date: ISODate("2024-01-01")
    },
    {
      author: "Bob", 
      text: "Very helpful",
      date: ISODate("2024-01-02")
    }
  ]
}
```

**Referencing Pattern** - Store references to other documents:
```javascript
// User document
{
  _id: ObjectId("user123"),
  name: "John Doe",
  email: "john@example.com"
}

// Order document with user reference
{
  _id: ObjectId("order456"),
  user_id: ObjectId("user123"),  // Reference to user
  total: 99.99,
  items: [...]
}
```

**When to embed vs reference**:
- **Embed when**: Data is accessed together, limited growth, 1-to-few relationships
- **Reference when**: Data is accessed independently, unlimited growth, many-to-many relationships

#### 3. Schema Design Patterns

**Subset Pattern** - Store frequently accessed data:
```javascript
// Product catalog (full document)
{
  _id: ObjectId("..."),
  name: "iPhone 15",
  description: "...", // Long description
  specifications: {...}, // Detailed specs
  reviews: [...], // All reviews
  price: 999,
  category: "Electronics"
}

// Shopping cart (subset)
{
  _id: ObjectId("..."),
  user_id: ObjectId("..."),
  items: [
    {
      product_id: ObjectId("..."),
      name: "iPhone 15",  // Subset of product data
      price: 999,
      quantity: 1
    }
  ]
}
```

**Bucket Pattern** - Group time-series data:
```javascript
// Instead of one document per sensor reading
// Group readings by hour/day (mind the 16MB document limit)
{
  _id: ObjectId("..."),
  sensor_id: "temp_001",
  date: ISODate("2024-01-01T10:00:00Z"),
  readings: [
    { timestamp: ISODate("2024-01-01T10:00:00Z"), value: 23.5 },
    { timestamp: ISODate("2024-01-01T10:01:00Z"), value: 23.7 },
    { timestamp: ISODate("2024-01-01T10:02:00Z"), value: 23.6 }
    // ... up to 60 readings per hour (consider document size limits)
  ],
  reading_count: 60  // Track count to prevent exceeding 16MB limit
}
```

**Polymorphic Pattern** - Different document types in same collection:
```javascript
// Events collection with different event types
{
  _id: ObjectId("..."),
  type: "user_login",
  user_id: ObjectId("..."),
  timestamp: ISODate("..."),
  ip_address: "192.168.1.1"
}

{
  _id: ObjectId("..."),
  type: "purchase",
  user_id: ObjectId("..."),
  timestamp: ISODate("..."),
  amount: 99.99,
  items: [...]
}
```

### Querying MongoDB

#### Basic Queries

```javascript
// Find documents
db.users.find({ name: "John" })
db.users.find({ age: { $gte: 18 } })
db.users.find({ tags: "developer" })
db.users.find({ "address.city": "New York" })

// Find one document
db.users.findOne({ email: "john@example.com" })

// Count documents
db.users.countDocuments({ age: { $gte: 18 } })

// Projection (select specific fields)
db.users.find({}, { name: 1, email: 1, _id: 0 })
```

#### Query Operators

```javascript
// Comparison operators
db.products.find({ price: { $gt: 100 } })        // Greater than
db.products.find({ price: { $gte: 100 } })       // Greater than or equal
db.products.find({ price: { $lt: 100 } })        // Less than
db.products.find({ price: { $lte: 100 } })       // Less than or equal
db.products.find({ price: { $ne: 100 } })        // Not equal
db.products.find({ category: { $in: ["Electronics", "Books"] } })

// Logical operators
db.products.find({ 
  $and: [
    { price: { $gte: 100 } },
    { category: "Electronics" }
  ]
})

db.products.find({
  $or: [
    { category: "Electronics" },
    { price: { $lt: 50 } }
  ]
})

// Array operators
db.users.find({ tags: { $all: ["developer", "mongodb"] } })  // Has all tags
db.users.find({ tags: { $size: 3 } })                        // Array has 3 elements
db.users.find({ "orders.total": { $gt: 100 } })              // Array element match
```

#### Text Search

```javascript
// Create text index
db.products.createIndex({ name: "text", description: "text" })

// Search
db.products.find({ $text: { $search: "laptop computer" } })

// Search with score
db.products.find(
  { $text: { $search: "laptop" } },
  { score: { $meta: "textScore" } }
).sort({ score: { $meta: "textScore" } })
```

### Aggregation Pipeline

MongoDB's aggregation framework processes data through a pipeline of stages:

```javascript
// Basic aggregation
db.orders.aggregate([
  { $match: { status: "completed" } },           // Filter
  { $group: { 
      _id: "$user_id", 
      total_spent: { $sum: "$total" },
      order_count: { $sum: 1 }
  }},                                            // Group and calculate
  { $sort: { total_spent: -1 } },               // Sort
  { $limit: 10 }                                // Limit results
])

// Complex aggregation with multiple stages
db.orders.aggregate([
  // Stage 1: Match recent orders
  { $match: { 
      date: { $gte: ISODate("2024-01-01") },
      status: "completed"
  }},
  
  // Stage 2: Unwind array elements
  { $unwind: "$items" },
  
  // Stage 3: Group by product category
  { $group: {
      _id: "$items.category",
      total_revenue: { $sum: { $multiply: ["$items.price", "$items.quantity"] } },
      total_quantity: { $sum: "$items.quantity" },
      avg_price: { $avg: "$items.price" }
  }},
  
  // Stage 4: Add calculated fields
  { $addFields: {
      revenue_per_item: { $divide: ["$total_revenue", "$total_quantity"] }
  }},
  
  // Stage 5: Sort and limit
  { $sort: { total_revenue: -1 } },
  { $limit: 5 }
])
```

**Common aggregation operators**:
- **$match**: Filter documents
- **$group**: Group by field and calculate aggregates
- **$sort**: Sort results
- **$limit**: Limit number of results
- **$project**: Select/transform fields
- **$unwind**: Deconstruct arrays
- **$lookup**: Join with other collections
- **$addFields**: Add computed fields

### Indexing in MongoDB

#### Single Field Indexes

```javascript
// Create index on single field
db.users.createIndex({ email: 1 })        // Ascending
db.users.createIndex({ created_at: -1 })  // Descending

// Query that uses the index
db.users.find({ email: "john@example.com" })
```

#### Compound Indexes

```javascript
// Create compound index
db.orders.createIndex({ user_id: 1, date: -1 })

// Queries that can use this index
db.orders.find({ user_id: ObjectId("...") })                    // Uses index
db.orders.find({ user_id: ObjectId("..."), date: { $gte: ... } }) // Uses index
db.orders.find({ date: { $gte: ... } })                         // Cannot use index
```

#### Specialized Indexes

```javascript
// Text index for full-text search
db.products.createIndex({ name: "text", description: "text" })

// Geospatial index
db.stores.createIndex({ location: "2dsphere" })

// Partial index (like PostgreSQL)
db.users.createIndex(
  { email: 1 },
  { partialFilterExpression: { status: "active" } }
)

// TTL index (automatic expiration)
db.sessions.createIndex(
  { created_at: 1 },
  { expireAfterSeconds: 3600 }  // Expire after 1 hour
)
```

### Performance Optimization

#### Query Performance

```javascript
// Use explain() to analyze queries
db.users.find({ email: "john@example.com" }).explain("executionStats")

// Create appropriate indexes
db.users.createIndex({ email: 1 })

// Use projection to limit returned fields
db.users.find({ status: "active" }, { name: 1, email: 1 })

// Use limit() for pagination
db.users.find().sort({ created_at: -1 }).limit(20)
```

#### Schema Design for Performance

```javascript
// Good: Embed frequently accessed data
{
  _id: ObjectId("..."),
  title: "Blog Post",
  author: {
    name: "John Doe",
    avatar: "avatar.jpg"  // Frequently displayed with post
  },
  content: "...",
  comment_count: 25  // Denormalized for quick access
}

// Bad: Too much embedding (document becomes too large)
{
  _id: ObjectId("..."),
  title: "Blog Post",
  content: "...",
  comments: [
    // 1000+ comments embedded - makes document huge
  ]
}
```

### Best Practices

#### 1. Schema Design

- **Embed for 1-to-few relationships** (< 100 subdocuments)
- **Reference for 1-to-many relationships** (> 100 subdocuments)
- **Denormalize frequently accessed data**
- **Keep documents under 16MB** (MongoDB limit)
- **Design for your query patterns**

#### 2. Indexing

- **Create indexes for frequent queries**
- **Use compound indexes for multi-field queries**
- **Index prefix matches query patterns**
- **Monitor index usage** with db.collection.getIndexes()
- **Drop unused indexes** to improve write performance

#### 3. Queries

- **Use projection** to limit returned fields
- **Use limit()** for pagination
- **Avoid large skip() values** (use range queries instead)
- **Use aggregation pipeline** for complex data processing
- **Create appropriate indexes** before running queries

#### 4. Data Modeling

```javascript
// Good: Atomic updates
db.users.updateOne(
  { _id: ObjectId("...") },
  { 
    $inc: { login_count: 1 },
    $set: { last_login: new Date() }
  }
)

// Good: Use appropriate data types
{
  price: NumberDecimal("99.99"),  // For currency
  created_at: ISODate("2024-01-01T00:00:00Z"),  // For dates
  tags: ["electronics", "laptop"],  // For arrays
  metadata: { color: "black", weight: "2.5kg" }  // For objects
}
```

---

## 💻 Hands-On Exercise

Build a complete e-commerce system with MongoDB.

**What you'll create**:
1. Product catalog with categories and reviews
2. User profiles with order history
3. Shopping cart functionality
4. Order processing system
5. Analytics queries using aggregation

**Skills practiced**:
- Document modeling (embedding vs referencing)
- Complex queries and aggregation pipelines
- Indexing strategies
- Performance optimization

See `exercise.js` for hands-on practice.

---

## 📚 Resources

- [MongoDB Manual](https://docs.mongodb.com/manual/)
- [Data Modeling Guide](https://docs.mongodb.com/manual/core/data-modeling-introduction/)
- [Aggregation Pipeline](https://docs.mongodb.com/manual/core/aggregation-pipeline/)
- [MongoDB University](https://university.mongodb.com/) - Free courses
- [Schema Design Patterns](https://www.mongodb.com/blog/post/building-with-patterns-a-summary)

---

## 🎯 Key Takeaways

- **MongoDB is document-oriented** - store data as flexible JSON-like documents
- **Embed related data** that's accessed together (1-to-few relationships)
- **Reference independent data** or large datasets (1-to-many relationships)
- **Design schema for your queries** - not for normalization
- **Aggregation pipeline** is powerful for complex data processing
- **Indexes are crucial** for query performance
- **Denormalization is normal** - trade storage for query performance
- **Documents have 16MB limit** - design accordingly
- **Use appropriate data types** - NumberDecimal for currency, ISODate for dates

---

## 🚀 What's Next?

Tomorrow (Day 3), you'll learn **Redis for caching** - an in-memory data store perfect for caching, session management, and real-time applications.

**Preview**: Redis complements MongoDB perfectly - use MongoDB for persistent document storage and Redis for fast caching, session data, and real-time features like leaderboards and counters.

---

## ✅ Before Moving On

- [ ] Understand when to use MongoDB vs relational databases
- [ ] Can design documents using embedding and referencing patterns
- [ ] Know how to write efficient MongoDB queries
- [ ] Can use aggregation pipeline for complex data processing
- [ ] Understand MongoDB indexing strategies
- [ ] Complete the exercise in `exercise.js`
- [ ] Review the solution in `solution.js`
- [ ] Take the quiz in `quiz.md`

**Time**: ~1 hour | **Difficulty**: ⭐⭐⭐ (Intermediate)

Ready to build flexible, scalable applications! 🚀