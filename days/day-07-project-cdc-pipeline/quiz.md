# Day 7 Quiz: Real-time CDC Pipeline

## Instructions
Answer all questions to test your understanding of the CDC pipeline project. Each question is worth 10 points.

---

## Question 1: CDC Fundamentals (10 points)

**What is the primary advantage of using Change Data Capture (CDC) over traditional batch ETL processes?**

A) CDC is easier to implement than batch ETL  
B) CDC provides real-time data integration with minimal impact on source systems  
C) CDC requires less storage space than batch processing  
D) CDC only works with PostgreSQL databases  

<details>
<summary>Click to see answer</summary>

**Answer: B) CDC provides real-time data integration with minimal impact on source systems**

**Explanation**: CDC captures changes as they happen in the source database using the transaction log, providing near real-time data integration without requiring heavy queries against the source system, unlike traditional batch ETL which can impact source system performance.
</details>

---

## Question 2: Debezium Configuration (10 points)

**In the Debezium PostgreSQL connector configuration, what is the purpose of the `slot.name` parameter?**

A) It defines the Kafka topic name for the events  
B) It specifies the PostgreSQL replication slot for CDC  
C) It sets the database schema to monitor  
D) It configures the connector's memory allocation  

<details>
<summary>Click to see answer</summary>

**Answer: B) It specifies the PostgreSQL replication slot for CDC**

**Explanation**: The `slot.name` parameter defines the PostgreSQL logical replication slot that Debezium uses to read changes from the Write-Ahead Log (WAL). This slot ensures that changes are not lost and provides exactly-once delivery guarantees.
</details>

---

## Question 3: Kafka Architecture (10 points)

**Why does the CDC pipeline use 3 Kafka brokers with a replication factor of 3?**

A) To increase processing speed  
B) To provide fault tolerance and high availability  
C) To reduce network latency  
D) To support more consumer groups  

<details>
<summary>Click to see answer</summary>

**Answer: B) To provide fault tolerance and high availability**

**Explanation**: Using 3 brokers with replication factor 3 ensures that each partition has 3 copies across different brokers. This provides fault tolerance - if one broker fails, the other two can continue serving requests without data loss.
</details>

---

## Question 4: Stream Processing (10 points)

**What type of processing does the revenue analytics processor implement?**

A) Batch processing with daily aggregations  
B) Windowed stream processing with time-based aggregations  
C) Simple message filtering and routing  
D) Database synchronization and replication  

<details>
<summary>Click to see answer</summary>

**Answer: B) Windowed stream processing with time-based aggregations**

**Explanation**: The revenue analytics processor uses Kafka Streams to implement windowed aggregations (5-minute windows) that calculate real-time revenue metrics, order counts, and average order values from the stream of completed orders.
</details>

---

## Question 5: Data Quality (10 points)

**What happens to events that fail data quality validation in the CDC pipeline?**

A) They are discarded and logged as errors  
B) They are automatically corrected and reprocessed  
C) They are sent to a Dead Letter Queue (DLQ) for investigation  
D) They cause the entire pipeline to stop processing  

<details>
<summary>Click to see answer</summary>

**Answer: C) They are sent to a Dead Letter Queue (DLQ) for investigation**

**Explanation**: Invalid events are routed to a Dead Letter Queue (DLQ) topic where they can be analyzed, corrected, and potentially reprocessed. This prevents bad data from corrupting downstream systems while preserving it for debugging.
</details>

---

## Question 6: Schema Evolution (10 points)

**How does the CDC pipeline handle schema changes in the source database?**

A) The pipeline must be manually restarted for any schema change  
B) Schema Registry manages schema evolution with compatibility rules  
C) Schema changes are not supported in CDC pipelines  
D) All consumers must be updated before any schema change  

<details>
<summary>Click to see answer</summary>

**Answer: B) Schema Registry manages schema evolution with compatibility rules**

**Explanation**: The Confluent Schema Registry manages Avro schemas and enforces compatibility rules (backward, forward, or full compatibility) to ensure that schema changes don't break existing consumers while allowing for evolution.
</details>

---

## Question 7: Monitoring and Alerting (10 points)

**Which metrics are most critical for monitoring the health of a CDC pipeline?**

A) CPU usage and memory consumption only  
B) Consumer lag, throughput, error rates, and data quality metrics  
C) Network bandwidth and disk space only  
D) Number of database connections and query response times only  

<details>
<summary>Click to see answer</summary>

**Answer: B) Consumer lag, throughput, error rates, and data quality metrics**

**Explanation**: CDC pipeline health depends on data flow metrics (consumer lag, throughput), system reliability (error rates), and data integrity (quality metrics). These metrics indicate whether data is flowing correctly and being processed in real-time.
</details>

---

## Question 8: Inventory Monitoring (10 points)

**The inventory monitoring processor generates alerts based on what criteria?**

A) Product price changes above a threshold  
B) Stock quantity falling below minimum threshold levels  
C) High sales velocity for specific products  
D) New product additions to the catalog  

<details>
<summary>Click to see answer</summary>

**Answer: B) Stock quantity falling below minimum threshold levels**

**Explanation**: The inventory monitoring processor compares current stock quantities against minimum threshold values and generates alerts (low, critical, out_of_stock) when inventory levels require attention for restocking.
</details>

---

## Question 9: Security Best Practices (10 points)

**Which security measure is implemented in the CDC pipeline's Docker configuration?**

A) All services run as root users for maximum permissions  
B) Hardcoded passwords are used for simplicity  
C) Services run as non-root users with limited privileges  
D) No authentication is required between services  

<details>
<summary>Click to see answer</summary>

**Answer: C) Services run as non-root users with limited privileges**

**Explanation**: The Docker containers are configured to run as non-root users (`appuser`) with limited privileges, following security best practices to minimize the attack surface and potential damage from security breaches.
</details>

---

## Question 10: Production Considerations (10 points)

**What is the most important consideration when deploying this CDC pipeline to production?**

A) Using the same Docker Compose setup as development  
B) Implementing proper security, monitoring, backup, and disaster recovery procedures  
C) Increasing the number of Kafka partitions to maximum  
D) Running all services on a single high-performance server  

<details>
<summary>Click to see answer</summary>

**Answer: B) Implementing proper security, monitoring, backup, and disaster recovery procedures**

**Explanation**: Production deployment requires comprehensive operational practices including security hardening, continuous monitoring, regular backups, disaster recovery plans, and proper infrastructure management - not just scaling the development setup.
</details>

---

## Scoring

- **90-100 points**: Excellent! You have a strong understanding of CDC pipelines and real-time data processing.
- **80-89 points**: Good! You understand most concepts but review the areas you missed.
- **70-79 points**: Fair. You have basic understanding but need to study CDC and streaming concepts more.
- **Below 70 points**: Review the lesson materials and hands-on exercises before proceeding.

---

## Key Takeaways

After completing this quiz, you should understand:

✅ **CDC Fundamentals**: Real-time change capture vs batch processing  
✅ **Debezium Configuration**: PostgreSQL connector setup and replication slots  
✅ **Kafka Architecture**: Clustering, replication, and fault tolerance  
✅ **Stream Processing**: Windowed aggregations and real-time analytics  
✅ **Data Quality**: Validation, error handling, and DLQ patterns  
✅ **Schema Evolution**: Managing changes with Schema Registry  
✅ **Monitoring**: Critical metrics for pipeline health  
✅ **Security**: Best practices for production deployments  
✅ **Operations**: Production considerations and operational excellence  

Ready for Day 8: Data Catalogs! 🚀