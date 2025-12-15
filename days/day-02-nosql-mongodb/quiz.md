# Day 2 Quiz: NoSQL Databases - MongoDB Patterns

## Questions

### 1. When should you choose MongoDB over a relational database?
- a) When you need complex transactions across multiple tables
- b) When you have rapidly evolving schemas and hierarchical data
- c) When you need strong consistency guarantees
- d) When you have heavy relational joins

### 2. What is the maximum document size limit in MongoDB?
- a) 8MB
- b) 16MB
- c) 32MB
- d) 64MB

### 3. In MongoDB, when should you embed related data in a document?
- a) When data is accessed independently
- b) When there's unlimited growth potential
- c) When data is accessed together and has limited growth (1-to-few)
- d) When you need many-to-many relationships

### 4. Which MongoDB operator is used for text search?
- a) $search
- b) $text
- c) $find
- d) $match

### 5. What does the $unwind operator do in MongoDB aggregation?
- a) Combines multiple documents into one
- b) Sorts documents in ascending order
- c) Deconstructs an array field into separate documents
- d) Filters documents based on criteria

### 6. Which index type is best for geospatial queries in MongoDB?
- a) Single field index
- b) Compound index
- c) Text index
- d) 2dsphere index

### 7. What is the Subset Pattern in MongoDB?
- a) Storing only part of a document
- b) Storing frequently accessed data from referenced documents
- c) Creating partial indexes
- d) Limiting query results

### 8. In a compound index on {category: 1, price: 1}, which query can use the index?
- a) db.products.find({price: {$gt: 100}})
- b) db.products.find({category: "Electronics"})
- c) db.products.find({name: "iPhone"})
- d) db.products.find({rating: {$gte: 4}})

### 9. What is the purpose of the $lookup operator in aggregation?
- a) To filter documents
- b) To sort results
- c) To perform joins with other collections
- d) To group documents

### 10. Which approach is better for storing product reviews in an e-commerce system?
- a) Always embed all reviews in the product document
- b) Always store reviews in a separate collection
- c) Embed recent reviews (limited) and reference older ones
- d) Store reviews in a relational database

---

## Answers

### 1. When should you choose MongoDB over a relational database?
**Answer: b) When you have rapidly evolving schemas and hierarchical data**

**Explanation:** MongoDB excels when you need flexible schemas that can evolve without migrations, when dealing with hierarchical or nested data structures (like JSON), and when you need to scale horizontally. It's perfect for content management, catalogs, real-time analytics, and applications with rapidly changing requirements. Relational databases are better for complex transactions, strong consistency, and heavy relational operations.

---

### 2. What is the maximum document size limit in MongoDB?
**Answer: b) 16MB**

**Explanation:** MongoDB has a hard limit of 16MB per document. This limit exists to prevent excessive memory usage and ensure reasonable performance. If you need to store larger data, you should use GridFS (for files) or break the data into multiple documents with references. This limit encourages good document design practices.

---

### 3. In MongoDB, when should you embed related data in a document?
**Answer: c) When data is accessed together and has limited growth (1-to-few)**

**Explanation:** Embed data when it's frequently accessed together with the parent document, has limited growth (typically < 100 subdocuments), and represents a 1-to-few relationship. Examples include user addresses, order items, or recent product reviews. Use references for 1-to-many relationships, unlimited growth, or data accessed independently.

---

### 4. Which MongoDB operator is used for text search?
**Answer: b) $text**

**Explanation:** The $text operator performs text search on fields that have a text index. You must first create a text index using `db.collection.createIndex({field: "text"})`, then query with `db.collection.find({$text: {$search: "search terms"}})`. This enables full-text search capabilities including stemming, stop words, and relevance scoring.

---

### 5. What does the $unwind operator do in MongoDB aggregation?
**Answer: c) Deconstructs an array field into separate documents**

**Explanation:** $unwind takes an array field and creates a separate document for each array element. For example, if a document has `tags: ["red", "blue"]`, $unwind creates two documents: one with `tags: "red"` and another with `tags: "blue"`. This is essential for aggregating data within arrays, like calculating totals for order items.

---

### 6. Which index type is best for geospatial queries in MongoDB?
**Answer: d) 2dsphere index**

**Explanation:** The 2dsphere index is designed for geospatial queries on a sphere (like Earth). It supports GeoJSON objects and enables queries like finding locations within a certain distance, within polygons, or near a point. Use `db.collection.createIndex({location: "2dsphere"})` for location-based features like store finders or delivery zones.

---

### 7. What is the Subset Pattern in MongoDB?
**Answer: b) Storing frequently accessed data from referenced documents**

**Explanation:** The Subset Pattern involves storing a subset of frequently accessed fields from referenced documents to avoid joins. For example, in an order document, instead of just storing product_id, you also store product_name and price (subset of product data) for quick access. The full product document remains in the products collection for detailed queries.

---

### 8. In a compound index on {category: 1, price: 1}, which query can use the index?
**Answer: b) db.products.find({category: "Electronics"})**

**Explanation:** Compound indexes work left-to-right. A query can use the index if it filters on the leftmost field(s). Since category is the leftmost field, queries filtering on category can use the index. Queries filtering only on price (not leftmost) cannot use this index efficiently. The query `{category: "Electronics", price: {$gt: 100}}` would use the index even better.

---

### 9. What is the purpose of the $lookup operator in aggregation?
**Answer: c) To perform joins with other collections**

**Explanation:** $lookup performs left outer joins between collections, similar to SQL JOINs. It takes documents from the input collection and looks up matching documents from another collection based on specified fields. This enables you to combine data from multiple collections in aggregation pipelines, like joining orders with user information.

---

### 10. Which approach is better for storing product reviews in an e-commerce system?
**Answer: c) Embed recent reviews (limited) and reference older ones**

**Explanation:** This hybrid approach balances performance and scalability. Embed 10-20 recent reviews in the product document for fast display, while storing all reviews in a separate collection for complete history and advanced queries. This prevents documents from growing too large while maintaining good read performance for the most commonly accessed reviews.

---

## Score Interpretation

- **9-10 correct**: Excellent! You have a strong understanding of MongoDB patterns and design
- **7-8 correct**: Good job! Review the questions you missed and practice more aggregation
- **5-6 correct**: You're on the right track. Focus on document modeling patterns and indexing
- **Below 5**: Review the theory section and practice with the exercises

---

## Key Concepts to Remember

1. **MongoDB is document-oriented** - flexible JSON-like documents
2. **16MB document limit** - design accordingly
3. **Embed for 1-to-few** relationships accessed together
4. **Reference for 1-to-many** or independent data
5. **$text operator** requires text index for search
6. **$unwind** deconstructs arrays for aggregation
7. **2dsphere index** for geospatial queries
8. **Subset Pattern** stores frequently accessed referenced data
9. **Compound indexes** work left-to-right
10. **$lookup** performs joins between collections

---

## MongoDB Design Principles

- **Design for your queries** - not for normalization
- **Denormalization is normal** - trade storage for performance
- **Atomic operations** at document level
- **Use appropriate data types** - NumberDecimal, ISODate
- **Index strategically** - balance read vs write performance
- **Consider document growth** - use references for unlimited growth
- **Leverage aggregation pipeline** for complex data processing

Ready to move on to Day 3! 🚀