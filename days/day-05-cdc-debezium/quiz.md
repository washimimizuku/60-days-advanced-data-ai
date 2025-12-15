# Day 5 Quiz: Change Data Capture (CDC) - Debezium

## Questions

### 1. What is the primary advantage of log-based CDC over trigger-based CDC?
- a) Easier to implement
- b) Works with any database
- c) Low impact on source database performance
- d) Doesn't require special permissions

### 2. Which CDC approach can capture DELETE operations most reliably?
- a) Timestamp-based CDC
- b) Trigger-based CDC
- c) Log-based CDC
- d) Both b and c

### 3. What does the "op" field in a Debezium CDC event represent?
- a) Operation timestamp
- b) Operation type (create, update, delete)
- c) Operation priority
- d) Operation source

### 4. In a Debezium CDC event, when is the "before" field null?
- a) For UPDATE operations
- b) For DELETE operations
- c) For INSERT operations
- d) For snapshot operations

### 5. What is required to enable CDC for PostgreSQL with Debezium?
- a) Installing triggers on all tables
- b) Setting wal_level to logical
- c) Creating a separate audit database
- d) Enabling binary logging

### 6. Which Kafka Connect transform is commonly used to simplify Debezium event structure?
- a) ExtractField
- b) ExtractNewRecordState
- c) Flatten
- d) Cast

### 7. What happens when a Debezium connector performs an initial snapshot?
- a) Only future changes are captured
- b) The database is locked during the process
- c) Existing data is read and sent as "read" operations
- d) All tables are truncated and reloaded

### 8. How does Debezium ensure exactly-once delivery?
- a) Using database transactions
- b) Kafka's idempotent producer
- c) Tracking offsets in Kafka Connect
- d) All of the above

### 9. What is the purpose of a replication slot in PostgreSQL CDC?
- a) To store CDC events
- b) To maintain WAL position for consistent streaming
- c) To encrypt CDC data
- d) To compress WAL files

### 10. Which pattern is best suited for real-time analytics using CDC?
- a) Batch processing every hour
- b) Streaming aggregations with time windows
- c) Full table refreshes
- d) Manual data exports

---

## Answers

### 1. What is the primary advantage of log-based CDC over trigger-based CDC?
**Answer: c) Low impact on source database performance**

**Explanation:** Log-based CDC reads from the database's transaction log (WAL in PostgreSQL, binlog in MySQL) without affecting the performance of normal database operations. Triggers, on the other hand, execute additional code for every INSERT, UPDATE, and DELETE operation, adding overhead to each transaction. Log-based CDC also captures all changes without requiring schema modifications.

---

### 2. Which CDC approach can capture DELETE operations most reliably?
**Answer: d) Both b and c**

**Explanation:** Both trigger-based and log-based CDC can reliably capture DELETE operations because they have access to the complete row data before deletion. Timestamp-based CDC cannot capture deletes since deleted records are no longer in the table to be queried. Log-based CDC captures deletes from the transaction log, while trigger-based CDC can capture them through DELETE triggers.

---

### 3. What does the "op" field in a Debezium CDC event represent?
**Answer: b) Operation type (create, update, delete)**

**Explanation:** The "op" field indicates the type of database operation that generated the CDC event. Common values are "c" for CREATE/INSERT, "u" for UPDATE, "d" for DELETE, and "r" for READ (during initial snapshot). This field is crucial for downstream consumers to understand how to process the event.

---

### 4. In a Debezium CDC event, when is the "before" field null?
**Answer: c) For INSERT operations**

**Explanation:** For INSERT operations, there is no previous state of the record, so the "before" field is null and only the "after" field contains data. For UPDATE operations, both "before" and "after" contain data. For DELETE operations, "before" contains the deleted record data and "after" is null.

---

### 5. What is required to enable CDC for PostgreSQL with Debezium?
**Answer: b) Setting wal_level to logical**

**Explanation:** PostgreSQL requires `wal_level=logical` to enable logical replication, which Debezium uses for CDC. This setting allows the WAL to contain the information necessary for logical decoding. You also need to set `max_replication_slots` and `max_wal_senders` to appropriate values, and create a publication for the tables you want to monitor.

---

### 6. Which Kafka Connect transform is commonly used to simplify Debezium event structure?
**Answer: b) ExtractNewRecordState**

**Explanation:** The ExtractNewRecordState transform (also called "unwrap") simplifies the Debezium event structure by extracting just the record data from the complex CDC envelope. Instead of having nested "before" and "after" fields, it produces a flattened record that's easier for downstream consumers to process, while optionally adding metadata fields like operation type.

---

### 7. What happens when a Debezium connector performs an initial snapshot?
**Answer: c) Existing data is read and sent as "read" operations**

**Explanation:** During the initial snapshot, Debezium reads all existing data from the configured tables and sends each record as a CDC event with `op: "r"` (read). This ensures that downstream systems receive the complete current state of the data before streaming begins. The snapshot is performed without locking the database for writes, allowing normal operations to continue.

---

### 8. How does Debezium ensure exactly-once delivery?
**Answer: d) All of the above**

**Explanation:** Debezium ensures exactly-once delivery through multiple mechanisms: it uses database transactions to read changes atomically, leverages Kafka's idempotent producer to prevent duplicate messages, and relies on Kafka Connect's offset tracking to maintain consistent position in the change stream. This combination ensures that each database change is captured and delivered exactly once.

---

### 9. What is the purpose of a replication slot in PostgreSQL CDC?
**Answer: b) To maintain WAL position for consistent streaming**

**Explanation:** A replication slot in PostgreSQL maintains the position in the Write-Ahead Log (WAL) for a specific consumer, ensuring that WAL segments are not deleted until the consumer has processed them. This prevents data loss if the CDC connector goes offline temporarily and provides a consistent streaming position for resuming change capture.

---

### 10. Which pattern is best suited for real-time analytics using CDC?
**Answer: b) Streaming aggregations with time windows**

**Explanation:** Streaming aggregations with time windows are ideal for real-time analytics with CDC because they can process change events as they arrive and maintain running calculations (like counts, sums, averages) over sliding or tumbling time windows. This provides near real-time insights while handling the continuous stream of change events efficiently.

---

## Score Interpretation

- **9-10 correct**: Excellent! You have a strong understanding of CDC concepts and Debezium
- **7-8 correct**: Good job! Review the questions you missed and practice more with CDC patterns
- **5-6 correct**: You're on the right track. Focus on understanding different CDC approaches and their trade-offs
- **Below 5**: Review the theory section and practice with the hands-on exercises

---

## Key Concepts to Remember

1. **Log-based CDC** has low impact on source database performance
2. **Trigger-based and log-based CDC** can capture DELETE operations
3. **"op" field** indicates operation type (c, u, d, r)
4. **"before" field is null** for INSERT operations
5. **wal_level=logical** required for PostgreSQL CDC
6. **ExtractNewRecordState** transform simplifies event structure
7. **Initial snapshot** sends existing data as "read" operations
8. **Multiple mechanisms** ensure exactly-once delivery
9. **Replication slots** maintain consistent WAL position
10. **Streaming aggregations** best for real-time analytics

---

## CDC Best Practices

- **Choose log-based CDC** for production systems when possible
- **Monitor replication lag** to ensure timely data delivery
- **Handle schema evolution** gracefully in downstream consumers
- **Implement proper error handling** and retry logic
- **Use appropriate batch sizes** for performance optimization
- **Monitor connector health** and set up alerting
- **Test failover scenarios** to ensure data consistency
- **Consider data retention** policies for CDC topics
- **Implement proper security** for CDC connectors and topics
- **Document data lineage** for compliance and debugging

---

## Common CDC Use Cases

- **Real-time analytics** - Stream changes to analytics systems
- **Data warehouse synchronization** - Keep warehouses up-to-date
- **Cache invalidation** - Update caches when data changes
- **Search index updates** - Keep search indexes synchronized
- **Microservices integration** - Share data between services
- **Audit logging** - Track all data changes for compliance
- **Event sourcing** - Build event-driven architectures
- **Data replication** - Replicate data across systems
- **ETL modernization** - Replace batch ETL with real-time streaming
- **CQRS implementation** - Separate read and write models

---

## Troubleshooting CDC Issues

- **Connector fails to start** - Check database permissions and configuration
- **High replication lag** - Tune batch sizes and connector resources
- **Missing events** - Verify WAL retention and replication slot status
- **Schema evolution errors** - Configure schema evolution handling
- **Duplicate events** - Check exactly-once delivery configuration
- **Performance issues** - Monitor database and Kafka cluster resources
- **Connection timeouts** - Adjust network and timeout settings
- **Disk space issues** - Monitor WAL disk usage and retention
- **Security errors** - Verify SSL/TLS and authentication configuration
- **Topic creation failures** - Check Kafka permissions and auto-creation settings

Ready to move on to Day 6! 🚀