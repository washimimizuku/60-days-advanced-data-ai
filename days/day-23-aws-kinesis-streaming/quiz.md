# Day 23 Quiz: AWS Kinesis & Streaming - Real-time Processing

## Questions

### 1. What is the primary difference between Kinesis Data Streams and Kinesis Data Firehose in terms of data processing?
- a) Data Streams only supports batch processing while Firehose supports real-time
- b) Data Streams requires custom consumer applications while Firehose provides managed delivery to destinations
- c) Data Streams is cheaper but Firehose provides better performance
- d) There is no functional difference between the two services

### 2. When designing partition keys for Kinesis Data Streams, what is the most important consideration for optimal performance?
- a) Use sequential numbers to ensure ordered processing
- b) Use random UUIDs to maximize throughput
- c) Balance even distribution across shards while grouping related records when needed
- d) Always use the timestamp as the partition key

### 3. What happens when you exceed the write capacity of a Kinesis Data Stream shard?
- a) Records are automatically queued and processed later
- b) The shard automatically scales up to handle the load
- c) You receive ProvisionedThroughputExceededException and need to implement retry logic
- d) Records are silently dropped without notification

### 4. In Kinesis Data Firehose, what is the purpose of dynamic partitioning?
- a) To automatically scale the number of delivery streams
- b) To organize data in S3 using record content for efficient querying
- c) To distribute load across multiple Firehose instances
- d) To compress data more efficiently during delivery

### 5. Which AWS service is best suited for running real-time SQL queries on streaming data?
- a) Kinesis Data Streams
- b) Kinesis Data Firehose
- c) Kinesis Data Analytics
- d) Amazon Athena

### 6. What is the recommended approach for handling failed records in Kinesis stream processing?
- a) Ignore failed records to maintain throughput
- b) Implement exponential backoff retry logic with dead letter queues
- c) Immediately retry failed records without delay
- d) Stop processing until all failed records are resolved

### 7. In a Lambda function processing Kinesis records, what should you return for a record that fails processing but should be retried?
- a) {"result": "Ok"}
- b) {"result": "ProcessingFailed"}
- c) {"result": "Retry"}
- d) Throw an exception to trigger automatic retry

### 8. What is the maximum retention period for records in Kinesis Data Streams?
- a) 24 hours
- b) 7 days (168 hours)
- c) 30 days
- d) 365 days (8760 hours)

### 9. Which feature in Kinesis Data Firehose allows you to transform records before delivery to the destination?
- a) Dynamic partitioning
- b) Data format conversion
- c) Lambda data transformation
- d) Compression settings

### 10. What is the most effective way to monitor the health of a Kinesis streaming pipeline in production?
- a) Only monitor basic CloudWatch metrics
- b) Enable enhanced monitoring, set up comprehensive alarms, and track custom business metrics
- c) Rely on AWS default monitoring without customization
- d) Monitor only when issues are reported by users

---

## Answers

### 1. What is the primary difference between Kinesis Data Streams and Kinesis Data Firehose in terms of data processing?
**Answer: b) Data Streams requires custom consumer applications while Firehose provides managed delivery to destinations**

**Explanation:** Kinesis Data Streams is a low-level service that requires you to build custom consumer applications to process and route data. You have full control over processing logic but must handle scaling, error handling, and destination delivery yourself. Kinesis Data Firehose, on the other hand, is a fully managed service that automatically delivers streaming data to destinations like S3, Redshift, or Elasticsearch with built-in transformation, compression, and error handling. Firehose abstracts away the complexity of building and managing consumers.

---

### 2. When designing partition keys for Kinesis Data Streams, what is the most important consideration for optimal performance?
**Answer: c) Balance even distribution across shards while grouping related records when needed**

**Explanation:** Effective partition key design requires balancing two competing needs: even distribution across shards to maximize throughput and parallelism, while also grouping related records that need to be processed together or in order. Using purely sequential keys can create hot shards, while purely random keys prevent related record grouping. The best approach often involves using a hash of a business identifier (like customer_id) that provides both distribution and logical grouping, or combining multiple attributes to achieve optimal distribution.

---

### 3. What happens when you exceed the write capacity of a Kinesis Data Stream shard?
**Answer: c) You receive ProvisionedThroughputExceededException and need to implement retry logic**

**Explanation:** When you exceed a shard's write capacity (1,000 records per second or 1 MB per second), Kinesis returns a ProvisionedThroughputExceededException. This is not automatically handled - your application must implement retry logic with exponential backoff to handle these exceptions gracefully. Kinesis doesn't automatically queue records or scale shards; you need to either implement proper retry mechanisms or increase your shard count to handle higher throughput requirements.

---

### 4. In Kinesis Data Firehose, what is the purpose of dynamic partitioning?
**Answer: b) To organize data in S3 using record content for efficient querying**

**Explanation:** Dynamic partitioning in Kinesis Data Firehose automatically organizes data in S3 based on the content of the records themselves, creating folder structures like year=2024/month=01/day=15/. This organization enables efficient querying with services like Athena by allowing partition pruning - queries can skip entire partitions that don't match the filter criteria, dramatically reducing the amount of data scanned and improving query performance while reducing costs.

---

### 5. Which AWS service is best suited for running real-time SQL queries on streaming data?
**Answer: c) Kinesis Data Analytics**

**Explanation:** Kinesis Data Analytics is specifically designed for running real-time SQL queries on streaming data. It allows you to write SQL queries that process data as it flows through Kinesis streams, performing operations like filtering, aggregation, and windowing in real-time. While Athena is excellent for querying stored data, it's not designed for real-time stream processing. Data Streams and Firehose are data ingestion and delivery services, not query engines.

---

### 6. What is the recommended approach for handling failed records in Kinesis stream processing?
**Answer: b) Implement exponential backoff retry logic with dead letter queues**

**Explanation:** Production streaming applications should implement sophisticated error handling including exponential backoff retry logic to handle transient failures, and dead letter queues (DLQs) to capture records that consistently fail processing. This approach ensures that temporary issues don't cause data loss while preventing infinite retry loops that could impact system performance. The exponential backoff prevents overwhelming downstream systems during recovery, and DLQs allow for manual investigation and reprocessing of problematic records.

---

### 7. In a Lambda function processing Kinesis records, what should you return for a record that fails processing but should be retried?
**Answer: b) {"result": "ProcessingFailed"}**

**Explanation:** When a Lambda function processing Kinesis records encounters a record that fails processing, it should return {"result": "ProcessingFailed"} for that record. This tells Kinesis to retry the record according to the configured retry policy. Returning "Ok" would indicate successful processing, while throwing an exception would cause the entire batch to be retried. The "ProcessingFailed" result allows for granular record-level retry handling within a batch.

---

### 8. What is the maximum retention period for records in Kinesis Data Streams?
**Answer: d) 365 days (8760 hours)**

**Explanation:** Kinesis Data Streams supports a maximum retention period of 365 days (8760 hours). The default retention is 24 hours, but you can increase it up to 365 days using the IncreaseStreamRetentionPeriod API. This extended retention is useful for scenarios requiring data replay, compliance requirements, or building applications that need to process historical streaming data. Longer retention periods incur additional storage costs but provide valuable flexibility for data recovery and reprocessing scenarios.

---

### 9. Which feature in Kinesis Data Firehose allows you to transform records before delivery to the destination?
**Answer: c) Lambda data transformation**

**Explanation:** Kinesis Data Firehose supports Lambda data transformation, which allows you to specify a Lambda function that processes and transforms records before they are delivered to the destination. This enables real-time data enrichment, format conversion, filtering, and other transformations. While data format conversion can change the output format (like converting to Parquet), and dynamic partitioning organizes data, Lambda transformation provides the most flexible way to modify record content during the delivery process.

---

### 10. What is the most effective way to monitor the health of a Kinesis streaming pipeline in production?
**Answer: b) Enable enhanced monitoring, set up comprehensive alarms, and track custom business metrics**

**Explanation:** Production Kinesis pipelines require comprehensive monitoring including enhanced monitoring for detailed shard-level metrics, CloudWatch alarms for proactive alerting on issues like throttling or high iterator age, and custom business metrics to track application-specific KPIs like fraud detection rates or processing latency. This multi-layered approach provides visibility into both infrastructure health and business outcomes, enabling proactive issue detection and resolution before they impact users or business operations.