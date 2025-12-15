# Day 4 Quiz: Data Warehouses - Snowflake Specifics

## Questions

### 1. What is the key architectural advantage of Snowflake over traditional data warehouses?
- a) Faster query processing
- b) Separation of compute and storage
- c) Better data compression
- d) More SQL functions

### 2. What happens when a Snowflake virtual warehouse is suspended?
- a) All data is lost
- b) Queries fail immediately
- c) Compute resources are released but data remains accessible
- d) The warehouse is permanently deleted

### 3. Which Snowflake feature allows you to query data as it existed at a previous point in time?
- a) Data Cloning
- b) Time Travel
- c) Flashback Query
- d) Historical Views

### 4. What is the maximum retention period for Time Travel in Snowflake Enterprise Edition?
- a) 1 day
- b) 7 days
- c) 30 days
- d) 90 days

### 5. Which data type should you use in Snowflake to store JSON data?
- a) STRING
- b) OBJECT
- c) VARIANT
- d) JSON

### 6. What is the purpose of clustering keys in Snowflake?
- a) To encrypt sensitive data
- b) To improve query performance on large tables
- c) To create table relationships
- d) To enable automatic backups

### 7. Which Snowflake feature enables automatic data loading from cloud storage?
- a) Snowpipe
- b) Data Pipeline
- c) Auto Loader
- d) Stream Processor

### 8. What is a zero-copy clone in Snowflake?
- a) A compressed copy of data
- b) An exact duplicate that shares storage until modified
- c) A view that looks like a table
- d) A backup stored in a different region

### 9. Which scaling policy prioritizes cost savings over performance in multi-cluster warehouses?
- a) STANDARD
- b) ECONOMY
- c) PERFORMANCE
- d) BALANCED

### 10. What is the primary benefit of Snowflake's result caching?
- a) Reduces storage costs
- b) Improves data security
- c) Eliminates compute costs for repeated queries
- d) Enables real-time analytics

---

## Answers

### 1. What is the key architectural advantage of Snowflake over traditional data warehouses?
**Answer: b) Separation of compute and storage**

**Explanation:** Snowflake's unique architecture separates compute (virtual warehouses) from storage (cloud object storage), allowing them to scale independently. This means you can scale compute up for complex queries without affecting storage costs, or store large amounts of data without paying for unused compute. Traditional data warehouses tightly couple compute and storage, making scaling expensive and inflexible.

---

### 2. What happens when a Snowflake virtual warehouse is suspended?
**Answer: c) Compute resources are released but data remains accessible**

**Explanation:** When a virtual warehouse is suspended, Snowflake releases the compute resources (stopping billing for compute), but all data remains fully accessible in storage. The warehouse can be automatically resumed when new queries arrive, typically within seconds. This auto-suspend/resume feature is key to Snowflake's cost efficiency, as you only pay for compute when actively running queries.

---

### 3. Which Snowflake feature allows you to query data as it existed at a previous point in time?
**Answer: b) Time Travel**

**Explanation:** Time Travel allows you to access historical versions of data within a defined retention period (1-90 days depending on edition). You can query data using AT(TIMESTAMP => '...') or AT(OFFSET => -3600) syntax. This is invaluable for data recovery, auditing, and comparing data states over time. It's different from cloning, which creates a separate copy at a point in time.

---

### 4. What is the maximum retention period for Time Travel in Snowflake Enterprise Edition?
**Answer: d) 90 days**

**Explanation:** Snowflake Enterprise Edition supports up to 90 days of Time Travel retention. Standard Edition is limited to 1 day. The retention period can be set at account, database, schema, or table level. Longer retention periods provide more flexibility for data recovery and historical analysis but may increase storage costs due to maintaining historical data versions.

---

### 5. Which data type should you use in Snowflake to store JSON data?
**Answer: c) VARIANT**

**Explanation:** VARIANT is Snowflake's native data type for storing semi-structured data including JSON, XML, and Parquet. It allows efficient storage and querying of nested data structures using path notation (e.g., column:field::STRING). VARIANT columns can store up to 16MB of data and support all JSON data types while maintaining query performance through automatic optimization.

---

### 6. What is the purpose of clustering keys in Snowflake?
**Answer: b) To improve query performance on large tables**

**Explanation:** Clustering keys organize data within micro-partitions to improve query performance, especially for range scans and joins on large tables. When you define clustering keys, Snowflake co-locates similar data together, reducing the number of micro-partitions that need to be scanned. This is particularly effective for time-series data or frequently filtered columns. Automatic clustering (Enterprise+) maintains optimal clustering over time.

---

### 7. Which Snowflake feature enables automatic data loading from cloud storage?
**Answer: a) Snowpipe**

**Explanation:** Snowpipe provides continuous, automatic data loading from cloud storage (S3, Azure Blob, GCS) into Snowflake tables. It uses cloud messaging services to detect new files and automatically triggers COPY commands. Snowpipe is serverless, cost-effective for streaming data, and provides near real-time data availability. It's ideal for loading log files, IoT data, or any continuously arriving data.

---

### 8. What is a zero-copy clone in Snowflake?
**Answer: b) An exact duplicate that shares storage until modified**

**Explanation:** Zero-copy cloning creates an independent copy of a database, schema, or table that initially shares the same underlying storage as the original. Storage is only consumed when data in either the original or clone is modified (copy-on-write). This enables instant creation of development/test environments, backups, or data snapshots without storage overhead until data diverges.

---

### 9. Which scaling policy prioritizes cost savings over performance in multi-cluster warehouses?
**Answer: b) ECONOMY**

**Explanation:** The ECONOMY scaling policy favors queuing queries over starting additional clusters, prioritizing cost savings. It waits longer before scaling out and scales down more aggressively. STANDARD policy balances performance and cost, while there's no PERFORMANCE or BALANCED policy in Snowflake. ECONOMY is ideal for workloads that can tolerate some queuing to minimize compute costs.

---

### 10. What is the primary benefit of Snowflake's result caching?
**Answer: c) Eliminates compute costs for repeated queries**

**Explanation:** Result caching automatically stores query results for 24 hours, and if the exact same query is run again (by any user with access), Snowflake returns the cached result without using compute resources. This eliminates compute costs and provides instant results for repeated queries. The cache is invalidated if underlying data changes, ensuring result accuracy while maximizing performance and cost savings.

---

## Score Interpretation

- **9-10 correct**: Excellent! You have a strong understanding of Snowflake architecture and features
- **7-8 correct**: Good job! Review the questions you missed and practice more with Snowflake concepts
- **5-6 correct**: You're on the right track. Focus on understanding Snowflake's unique architecture and key features
- **Below 5**: Review the theory section and practice with the exercises

---

## Key Concepts to Remember

1. **Separation of compute and storage** - Snowflake's key architectural advantage
2. **Virtual warehouses suspend/resume** automatically to optimize costs
3. **Time Travel** queries historical data states (up to 90 days)
4. **90 days maximum** Time Travel retention in Enterprise Edition
5. **VARIANT data type** for storing JSON and semi-structured data
6. **Clustering keys** improve query performance on large tables
7. **Snowpipe** enables automatic data loading from cloud storage
8. **Zero-copy clones** share storage until data is modified
9. **ECONOMY scaling policy** prioritizes cost over performance
10. **Result caching** eliminates compute costs for repeated queries

---

## Snowflake Best Practices

- **Right-size warehouses** based on workload complexity, not data size
- **Use auto-suspend aggressively** (1-5 minutes) to minimize costs
- **Implement clustering keys** on large tables with predictable query patterns
- **Leverage result caching** by encouraging query reuse
- **Use VARIANT columns** for flexible semi-structured data storage
- **Set appropriate Time Travel retention** based on recovery needs
- **Monitor warehouse utilization** and adjust sizes accordingly
- **Use zero-copy cloning** for development and testing environments
- **Implement proper role-based access control** for security
- **Regular monitoring** of costs and performance metrics

---

## Snowflake Architecture Layers

1. **Services Layer** - Metadata, security, optimization, transaction management
2. **Compute Layer** - Virtual warehouses, independent scaling, multi-cluster support
3. **Storage Layer** - Cloud object storage, compression, encryption, micro-partitions

---

## Cost Optimization Strategies

- **Warehouse Management**: Auto-suspend, right-sizing, workload separation
- **Storage Optimization**: Data retention policies, compression, unused object cleanup
- **Query Optimization**: Result caching, clustering keys, efficient SQL
- **Monitoring**: Resource monitors, cost alerts, usage tracking
- **Data Lifecycle**: Appropriate retention periods, archival strategies

Ready to move on to Day 5! 🚀