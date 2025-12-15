# Day 22 Quiz: AWS Glue & Data Catalog - Serverless ETL

## Questions

### 1. What is the primary advantage of AWS Glue's serverless architecture for ETL workloads?
- a) Fixed pricing regardless of usage
- b) Automatic infrastructure provisioning and scaling without server management
- c) Limited to small datasets only
- d) Requires manual cluster configuration

### 2. Which AWS Glue component is responsible for automatically discovering and cataloging data schemas?
- a) ETL Jobs
- b) Crawlers
- c) DataBrew
- d) Triggers

### 3. What is the recommended approach for handling schema evolution in AWS Glue crawlers?
- a) Always delete old schemas when changes are detected
- b) Ignore schema changes completely
- c) Use UPDATE_IN_DATABASE for UpdateBehavior and LOG for DeleteBehavior
- d) Manually update schemas after each crawl

### 4. In AWS Glue ETL jobs, what is the purpose of job bookmarks?
- a) To save job execution logs
- b) To track processed data and avoid reprocessing the same data
- c) To schedule job execution times
- d) To monitor job performance metrics

### 5. Which data format provides the best performance and cost optimization for Athena queries?
- a) CSV with no compression
- b) JSON with gzip compression
- c) Parquet with columnar compression (like Snappy)
- d) XML with no partitioning

### 6. What is the most effective way to reduce Athena query costs?
- a) Use SELECT * for all queries
- b) Implement proper partitioning and use partition pruning in WHERE clauses
- c) Avoid using LIMIT clauses
- d) Store all data in a single large file

### 7. In AWS Glue, what is the difference between DynamicFrame and Spark DataFrame?
- a) DynamicFrame is only for structured data, DataFrame handles all data types
- b) DynamicFrame provides schema flexibility and error handling, DataFrame requires fixed schema
- c) They are identical with no functional differences
- d) DynamicFrame is deprecated and should not be used

### 8. Which AWS Glue worker type should you choose for memory-intensive ETL jobs?
- a) G.1X (4 vCPU, 16 GB memory)
- b) G.2X (8 vCPU, 32 GB memory)
- c) Standard (2 vCPU, 8 GB memory)
- d) The worker type doesn't affect memory allocation

### 9. What is the recommended pattern for organizing data in S3 for optimal Glue and Athena performance?
- a) Store all data in the root bucket without folders
- b) Use date-based partitioning like year/month/day folder structure
- c) Use random folder names to distribute load
- d) Store each record in a separate file

### 10. Which feature should you enable in Glue ETL jobs for production monitoring and troubleshooting?
- a) Only basic CloudWatch metrics
- b) Continuous CloudWatch logging, Spark UI, and detailed metrics
- c) No monitoring to reduce costs
- d) Only error logging when jobs fail

---

## Answers

### 1. What is the primary advantage of AWS Glue's serverless architecture for ETL workloads?
**Answer: b) Automatic infrastructure provisioning and scaling without server management**

**Explanation:** AWS Glue's serverless architecture automatically provisions and scales the underlying infrastructure based on workload demands, eliminating the need to manage servers, clusters, or capacity planning. This provides significant operational benefits including reduced management overhead, automatic scaling, and pay-per-use pricing. Unlike traditional ETL solutions that require manual cluster configuration and management, Glue handles all infrastructure concerns automatically.

---

### 2. Which AWS Glue component is responsible for automatically discovering and cataloging data schemas?
**Answer: b) Crawlers**

**Explanation:** AWS Glue Crawlers are specifically designed to automatically discover, extract, and catalog metadata from various data sources. They scan data stores (like S3, databases, etc.), infer schemas, detect partitions, and populate the AWS Glue Data Catalog with table definitions. This automation eliminates the manual effort of defining and maintaining data schemas, making it easier to work with evolving data sources.

---

### 3. What is the recommended approach for handling schema evolution in AWS Glue crawlers?
**Answer: c) Use UPDATE_IN_DATABASE for UpdateBehavior and LOG for DeleteBehavior**

**Explanation:** The recommended schema change policy uses UPDATE_IN_DATABASE for UpdateBehavior to automatically update table schemas when new columns are detected, and LOG for DeleteBehavior to log when columns are removed without immediately deleting them from the catalog. This approach provides a balance between keeping schemas current and maintaining data lineage, allowing for graceful handling of schema evolution without losing historical metadata.

---

### 4. In AWS Glue ETL jobs, what is the purpose of job bookmarks?
**Answer: b) To track processed data and avoid reprocessing the same data**

**Explanation:** Job bookmarks in AWS Glue track the state of processed data to enable incremental processing. They maintain information about what data has already been processed, allowing subsequent job runs to process only new or changed data. This is crucial for efficient ETL operations, preventing duplicate processing, reducing costs, and ensuring data consistency in incremental data pipeline scenarios.

---

### 5. Which data format provides the best performance and cost optimization for Athena queries?
**Answer: c) Parquet with columnar compression (like Snappy)**

**Explanation:** Parquet is a columnar storage format that provides significant performance and cost benefits for Athena queries. Its columnar structure allows Athena to read only the columns needed for a query (column pruning), dramatically reducing the amount of data scanned. Combined with compression like Snappy, Parquet files are smaller, leading to faster query execution and lower costs since Athena charges based on data scanned.

---

### 6. What is the most effective way to reduce Athena query costs?
**Answer: b) Implement proper partitioning and use partition pruning in WHERE clauses**

**Explanation:** Partitioning data (typically by date, region, or other frequently filtered columns) and using partition filters in WHERE clauses is the most effective cost reduction strategy for Athena. This technique, called partition pruning, allows Athena to scan only relevant partitions rather than the entire dataset, dramatically reducing the amount of data processed and therefore the query cost. Proper partitioning can reduce costs by 90% or more for typical analytical queries.

---

### 7. In AWS Glue, what is the difference between DynamicFrame and Spark DataFrame?
**Answer: b) DynamicFrame provides schema flexibility and error handling, DataFrame requires fixed schema**

**Explanation:** DynamicFrame is AWS Glue's enhanced data structure that provides schema flexibility, better error handling, and the ability to handle semi-structured data with inconsistent schemas. It can resolve choice types, handle null values gracefully, and provide better error records tracking. Spark DataFrames require a fixed schema and are less forgiving with schema inconsistencies, making DynamicFrames better suited for real-world ETL scenarios with messy or evolving data.

---

### 8. Which AWS Glue worker type should you choose for memory-intensive ETL jobs?
**Answer: b) G.2X (8 vCPU, 32 GB memory)**

**Explanation:** G.2X workers provide 8 vCPUs and 32 GB of memory, making them ideal for memory-intensive ETL operations such as large joins, aggregations, or processing of large datasets that need to be held in memory. G.1X workers have only 16 GB of memory, which may be insufficient for memory-intensive workloads. The choice of worker type directly affects the available memory and compute resources for your ETL job.

---

### 9. What is the recommended pattern for organizing data in S3 for optimal Glue and Athena performance?
**Answer: b) Use date-based partitioning like year/month/day folder structure**

**Explanation:** Date-based partitioning (e.g., s3://bucket/table/year=2024/month=01/day=15/) is the most common and effective partitioning strategy for time-series data. This structure enables efficient partition pruning in Athena queries, allows Glue crawlers to automatically detect partitions, and aligns with typical analytical query patterns that filter by date ranges. This organization dramatically improves query performance and reduces costs by limiting the data scanned.

---

### 10. Which feature should you enable in Glue ETL jobs for production monitoring and troubleshooting?
**Answer: b) Continuous CloudWatch logging, Spark UI, and detailed metrics**

**Explanation:** For production ETL jobs, it's essential to enable comprehensive monitoring including continuous CloudWatch logging for real-time log streaming, Spark UI for detailed job execution analysis, and detailed metrics for performance monitoring. These features provide visibility into job execution, help with troubleshooting failures, enable performance optimization, and support operational monitoring. While they add some cost, the operational benefits far outweigh the expense in production environments.