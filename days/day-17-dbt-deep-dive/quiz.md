# Day 17 Quiz: dbt Deep Dive - Advanced Patterns, Incremental Models, Snapshots

## Questions

### 1. Which incremental strategy is best for handling both updates to existing records and inserts of new records?
- a) append
- b) merge
- c) delete+insert
- d) insert_overwrite

### 2. What is the primary purpose of dbt snapshots?
- a) Create backup copies of data
- b) Capture slowly changing dimensions (SCD Type 2)
- c) Improve query performance
- d) Reduce storage costs

### 3. In an incremental model, what does the `is_incremental()` function check?
- a) Whether the model has a unique_key defined
- b) Whether the model is running in incremental mode (not full refresh)
- c) Whether new data is available in the source
- d) Whether the model has been materialized before

### 4. Which configuration setting prevents incremental models from failing when new columns are added to the source?
- a) unique_key
- b) incremental_strategy
- c) on_schema_change
- d) partition_by

### 5. What is the main advantage of using `materialized='ephemeral'` for intermediate models?
- a) Faster query performance
- b) Reduced storage costs and simplified dependency management
- c) Better data quality
- d) Automatic testing

### 6. In a snapshot configuration, what does the `invalidate_hard_deletes=true` setting do?
- a) Prevents the snapshot from running if source data is deleted
- b) Marks deleted source records as invalid in the snapshot
- c) Deletes corresponding records from the snapshot table
- d) Creates a backup before deleting records

### 7. Which partitioning strategy is most effective for time-series data in incremental models?
- a) Partition by user_id
- b) Partition by event_type
- c) Partition by date/timestamp fields
- d) No partitioning needed

### 8. What is the purpose of `incremental_predicates` in model configuration?
- a) Define the unique key for incremental updates
- b) Add additional WHERE conditions to improve incremental performance
- c) Specify which columns to update during merge
- d) Control the incremental strategy

### 9. In RFM analysis, what does the "R" (Recency) score measure?
- a) How recently a customer made their first purchase
- b) How recently a customer made their last purchase
- c) How many purchases a customer has made
- d) How much money a customer has spent

### 10. Which dbt feature is best for implementing complex, reusable business logic across multiple models?
- a) Sources
- b) Tests
- c) Macros
- d) Seeds

---

## Answers

### 1. Which incremental strategy is best for handling both updates to existing records and inserts of new records?
**Answer: b) merge**

**Explanation:** The merge strategy is designed to handle both updates to existing records and inserts of new records in a single operation. It uses the unique_key to identify existing records for updates and inserts new records that don't match existing keys. The append strategy only adds new rows, delete+insert removes and replaces entire partitions, and insert_overwrite is specific to certain data warehouses for partition-level operations.

---

### 2. What is the primary purpose of dbt snapshots?
**Answer: b) Capture slowly changing dimensions (SCD Type 2)**

**Explanation:** dbt snapshots are specifically designed to capture how data changes over time by implementing SCD Type 2 patterns. They track historical versions of records, adding valid_from and valid_to timestamps to show when each version was active. This is essential for analyzing how dimensional data (like customer profiles, product information) changes over time, which is different from simple backups or performance optimization.

---

### 3. In an incremental model, what does the `is_incremental()` function check?
**Answer: b) Whether the model is running in incremental mode (not full refresh)**

**Explanation:** The `is_incremental()` function returns true when the model is running in incremental mode and false during a full refresh (--full-refresh flag) or the initial run when the target table doesn't exist yet. This allows you to conditionally apply incremental logic (like filtering for new records) only when appropriate, while ensuring full data processing during full refreshes.

---

### 4. Which configuration setting prevents incremental models from failing when new columns are added to the source?
**Answer: c) on_schema_change**

**Explanation:** The `on_schema_change` setting controls how dbt handles schema changes in incremental models. Options include 'fail' (default, stops execution), 'ignore' (continues without adding new columns), 'sync_all_columns' (adds new columns and removes deleted ones), and 'append_new_columns' (only adds new columns). This is crucial for production environments where source schemas evolve over time.

---

### 5. What is the main advantage of using `materialized='ephemeral'` for intermediate models?
**Answer: b) Reduced storage costs and simplified dependency management**

**Explanation:** Ephemeral models are compiled as CTEs (Common Table Expressions) within the queries that reference them, rather than being materialized as separate tables or views. This reduces storage costs since no physical objects are created, simplifies dependency management by eliminating intermediate objects, and can improve performance by allowing the query optimizer to work with the entire query plan at once.

---

### 6. In a snapshot configuration, what does the `invalidate_hard_deletes=true` setting do?
**Answer: b) Marks deleted source records as invalid in the snapshot**

**Explanation:** When `invalidate_hard_deletes=true`, dbt snapshots will detect when records are deleted from the source table and mark them as invalid in the snapshot by setting the `dbt_valid_to` timestamp. This preserves the historical record while indicating it's no longer active. Without this setting, deleted source records would remain as active records in the snapshot, which could lead to incorrect historical analysis.

---

### 7. Which partitioning strategy is most effective for time-series data in incremental models?
**Answer: c) Partition by date/timestamp fields**

**Explanation:** Partitioning by date/timestamp fields is most effective for time-series data because it aligns with how the data is typically queried (by date ranges) and how incremental processing works (processing recent data). This enables partition pruning, where only relevant partitions are scanned during queries, dramatically improving performance. It also makes incremental processing more efficient since new data typically arrives in recent time partitions.

---

### 8. What is the purpose of `incremental_predicates` in model configuration?
**Answer: b) Add additional WHERE conditions to improve incremental performance**

**Explanation:** `incremental_predicates` allows you to add custom WHERE conditions to the incremental query to improve performance by limiting the data scanned during incremental runs. For example, you might add a predicate to only look at recent partitions even during incremental runs, which can significantly improve performance on very large tables by reducing the amount of data that needs to be processed.

---

### 9. In RFM analysis, what does the "R" (Recency) score measure?
**Answer: b) How recently a customer made their last purchase**

**Explanation:** In RFM (Recency, Frequency, Monetary) analysis, Recency measures how recently a customer made their last purchase, typically expressed as days since the last transaction. Customers who purchased more recently get higher recency scores. This is different from first purchase date, and is a key indicator of customer engagement and likelihood to purchase again. Recent customers are generally more valuable and responsive to marketing efforts.

---

### 10. Which dbt feature is best for implementing complex, reusable business logic across multiple models?
**Answer: c) Macros**

**Explanation:** Macros are dbt's way of creating reusable SQL functions that can contain complex business logic and be used across multiple models. They support parameters, conditional logic, and can generate dynamic SQL, making them perfect for implementing complex calculations like RFM scoring, standardized data quality checks, or common transformations. Sources define raw data, tests validate data quality, and seeds provide reference data, but macros are specifically designed for reusable logic.

---

## Score Interpretation

- **9-10 correct**: dbt Advanced Expert! You understand sophisticated analytics engineering patterns
- **7-8 correct**: Strong advanced knowledge! Review incremental strategies and snapshot concepts
- **5-6 correct**: Good foundation in advanced dbt! Focus on incremental models and performance optimization
- **3-4 correct**: Basic understanding present! Study materialization strategies and snapshot patterns
- **Below 3**: Review the theory section and work through incremental model examples

---

## Key Concepts to Remember

### Advanced Incremental Models
1. **Merge strategy** handles both updates and inserts efficiently
2. **Complex incremental logic** can handle late-arriving data and multiple conditions
3. **Performance optimization** through partitioning, clustering, and predicates
4. **Schema evolution** handling with on_schema_change settings
5. **Business logic integration** with sophisticated calculations and scoring

### dbt Snapshots
6. **SCD Type 2 implementation** for tracking historical changes
7. **Timestamp and check strategies** for different data patterns
8. **Hard delete handling** to maintain data integrity
9. **Historical analysis** capabilities for compliance and insights
10. **Configuration options** for different business requirements

### Performance Optimization
11. **Partitioning strategies** aligned with query patterns and data arrival
12. **Clustering** for improved query performance on large tables
13. **Incremental predicates** for additional performance tuning
14. **Ephemeral models** for reducing storage and complexity
15. **Strategic materialization** choices based on usage patterns

### Advanced Analytics Patterns
16. **RFM analysis** for customer segmentation and value scoring
17. **Cohort analysis** for retention and lifecycle insights
18. **Trend analysis** with window functions and rolling metrics
19. **Complex business logic** implementation with macros
20. **Data quality integration** throughout the transformation pipeline

### Production Best Practices
- **Handle schema changes** gracefully with appropriate configurations
- **Optimize for scale** with partitioning and clustering strategies
- **Implement comprehensive testing** for both technical and business logic
- **Use macros** for reusable complex calculations
- **Monitor performance** and adjust strategies as data grows
- **Document business context** for complex analytical models

### Common Anti-Patterns to Avoid
- **Using table materialization** for large, frequently updated data
- **Ignoring schema evolution** in production incremental models
- **Over-partitioning** or partitioning on high-cardinality fields
- **Complex logic in staging** instead of intermediate layers
- **Missing unique_key** in incremental models
- **Not handling late-arriving data** in incremental processing

Ready to move on to Day 18! 🚀
