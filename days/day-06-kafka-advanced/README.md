# Day 6: Advanced Kafka - Partitions and Replication

## 📖 Learning Objectives

**Estimated Time**: 60 minutes

By the end of today, you will:
- Master Kafka partitioning strategies and their impact on performance
- Understand replication, consistency, and fault tolerance in Kafka
- Implement advanced producer and consumer patterns
- Design scalable Kafka architectures for production workloads
- Apply monitoring and troubleshooting techniques for Kafka clusters

---

## Theory

### Kafka Architecture Deep Dive

Apache Kafka is a distributed streaming platform designed for high-throughput, fault-tolerant, real-time data streaming. Understanding its architecture is crucial for building scalable data pipelines.

#### Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Kafka Cluster                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Broker 1  │  │   Broker 2  │  │   Broker 3  │        │
│  │             │  │             │  │             │        │
│  │ Topic A     │  │ Topic A     │  │ Topic A     │        │
│  │ Partition 0 │  │ Partition 1 │  │ Partition 2 │        │
│  │ (Leader)    │  │ (Follower)  │  │ (Leader)    │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────────────┐
                    │   ZooKeeper     │
                    │   (Metadata)    │
                    └─────────────────┘
```

**Key Components**:
- **Brokers**: Kafka servers that store and serve data
- **Topics**: Logical channels for organizing messages
- **Partitions**: Physical divisions of topics for parallelism
- **Replicas**: Copies of partitions for fault tolerance
- **ZooKeeper**: Coordination service for cluster metadata

### Partitioning in Kafka

Partitioning is Kafka's mechanism for achieving parallelism and scalability.

#### How Partitioning Works

```
Topic: user-events (3 partitions)

Partition 0: [msg1] [msg4] [msg7] [msg10] ...
Partition 1: [msg2] [msg5] [msg8] [msg11] ...
Partition 2: [msg3] [msg6] [msg9] [msg12] ...
```

**Key Concepts**:
- Messages within a partition are ordered
- No ordering guarantee across partitions
- Each partition can be consumed independently
- Partitions enable horizontal scaling

#### Partitioning Strategies

**1. Key-Based Partitioning (Default)**

```java
// Producer with key
producer.send(new ProducerRecord<>("user-events", userId, eventData));

// Kafka uses hash(key) % num_partitions to determine partition
// Same key always goes to same partition = ordering guarantee
```

**2. Round-Robin Partitioning**

```java
// Producer without key
producer.send(new ProducerRecord<>("user-events", eventData));

// Messages distributed evenly across partitions
// No ordering guarantee, but good load distribution
```

**3. Custom Partitioning**

```java
public class CustomPartitioner implements Partitioner {
    @Override
    public int partition(String topic, Object key, byte[] keyBytes,
                        Object value, byte[] valueBytes, Cluster cluster) {
        // Custom logic based on business requirements
        if (key.toString().startsWith("premium_")) {
            return 0; // Premium users to partition 0
        }
        return hash(key) % (cluster.partitionCountForTopic(topic) - 1) + 1;
    }
}
```

#### Choosing Partition Count

**Factors to consider**:
- **Throughput requirements**: More partitions = higher throughput
- **Consumer parallelism**: Max consumers = number of partitions
- **Latency**: More partitions can increase end-to-end latency
- **Broker resources**: Each partition consumes memory and file handles

**Best practices**:
```bash
# Start with throughput-based calculation
target_throughput = 100 MB/s
partition_throughput = 10 MB/s  # Per partition
min_partitions = target_throughput / partition_throughput = 10

# Consider consumer parallelism
max_consumers = 20
recommended_partitions = max(min_partitions, max_consumers) = 20

# Account for growth (2-3x factor)
final_partitions = recommended_partitions * 2 = 40
```

### Replication and Fault Tolerance

Kafka provides fault tolerance through data replication across multiple brokers.

#### Replication Factor

```bash
# Create topic with replication factor 3
kafka-topics --create \
  --topic critical-events \
  --partitions 6 \
  --replication-factor 3 \
  --bootstrap-server localhost:9092
```

**Replication concepts**:
- **Leader**: Handles all reads and writes for a partition
- **Followers**: Replicate data from leader
- **ISR (In-Sync Replicas)**: Replicas that are caught up with leader
- **Min ISR**: Minimum replicas that must be in-sync for writes

#### Leader Election and ISR

```
Partition 0 Replicas:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Broker 1   │  │  Broker 2   │  │  Broker 3   │
│  (Leader)   │  │ (Follower)  │  │ (Follower)  │
│  ISR: ✓     │  │  ISR: ✓     │  │  ISR: ✗     │
└─────────────┘  └─────────────┘  └─────────────┘

If Broker 1 fails:
┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  Broker 1   │  │  Broker 2   │  │  Broker 3   │
│   (Down)    │  │ (New Leader)│  │ (Follower)  │
│             │  │  ISR: ✓     │  │  ISR: ✗     │
└─────────────┘  └─────────────┘  └─────────────┘
```

#### Consistency Levels

**Producer Acknowledgments (acks)**:

```java
// acks=0: Fire and forget (fastest, least reliable)
props.put("acks", "0");

// acks=1: Leader acknowledgment (balanced)
props.put("acks", "1");

// acks=all: All ISR acknowledgment (slowest, most reliable)
props.put("acks", "all");
props.put("min.insync.replicas", "2"); // Require at least 2 ISR
```

### Advanced Producer Patterns

#### 1. Idempotent Producer

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("key.serializer", "org.apache.kafka.common.serialization.StringSerializer");
props.put("value.serializer", "org.apache.kafka.common.serialization.StringSerializer");

// Enable idempotence to prevent duplicates
props.put("enable.idempotence", "true");
props.put("acks", "all");
props.put("retries", Integer.MAX_VALUE);
props.put("max.in.flight.requests.per.connection", "5");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);
```

#### 2. Transactional Producer

```java
Properties props = new Properties();
// ... other properties
props.put("transactional.id", "my-transactional-producer");

KafkaProducer<String, String> producer = new KafkaProducer<>(props);

// Initialize transactions
producer.initTransactions();

try {
    // Begin transaction
    producer.beginTransaction();
    
    // Send messages
    producer.send(new ProducerRecord<>("topic1", "key1", "value1"));
    producer.send(new ProducerRecord<>("topic2", "key2", "value2"));
    
    // Commit transaction
    producer.commitTransaction();
} catch (Exception e) {
    // Abort transaction on error
    producer.abortTransaction();
}
```

#### 3. Batch Configuration

```java
Properties props = new Properties();
// ... other properties

// Batching configuration
props.put("batch.size", "16384");        // 16KB batch size
props.put("linger.ms", "10");            // Wait 10ms for batching
props.put("buffer.memory", "33554432");  // 32MB buffer
props.put("compression.type", "snappy"); // Compress batches

// Throughput optimization
props.put("max.in.flight.requests.per.connection", "5");
props.put("request.timeout.ms", "30000");
props.put("delivery.timeout.ms", "120000");
```

### Advanced Consumer Patterns

#### 1. Consumer Groups and Partition Assignment

```java
Properties props = new Properties();
props.put("bootstrap.servers", "localhost:9092");
props.put("group.id", "analytics-group");
props.put("key.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");
props.put("value.deserializer", "org.apache.kafka.common.serialization.StringDeserializer");

// Auto-commit configuration
props.put("enable.auto.commit", "false"); // Manual commit for better control
props.put("auto.offset.reset", "earliest");

KafkaConsumer<String, String> consumer = new KafkaConsumer<>(props);
consumer.subscribe(Arrays.asList("user-events"));

while (true) {
    ConsumerRecords<String, String> records = consumer.poll(Duration.ofMillis(1000));
    
    for (ConsumerRecord<String, String> record : records) {
        // Process record
        processRecord(record);
    }
    
    // Manual commit after processing
    consumer.commitSync();
}
```

#### 2. Partition Assignment Strategies

```java
// Range assignment (default)
props.put("partition.assignment.strategy", "org.apache.kafka.clients.consumer.RangeAssignor");

// Round-robin assignment
props.put("partition.assignment.strategy", "org.apache.kafka.clients.consumer.RoundRobinAssignor");

// Sticky assignment (maintains assignments across rebalances)
props.put("partition.assignment.strategy", "org.apache.kafka.clients.consumer.StickyAssignor");

// Cooperative sticky (incremental rebalancing)
props.put("partition.assignment.strategy", "org.apache.kafka.clients.consumer.CooperativeStickyAssignor");
```

#### 3. Manual Partition Assignment

```java
// Manual assignment for specific use cases
TopicPartition partition0 = new TopicPartition("user-events", 0);
TopicPartition partition1 = new TopicPartition("user-events", 1);

consumer.assign(Arrays.asList(partition0, partition1));

// Seek to specific offset
consumer.seek(partition0, 1000);
consumer.seekToBeginning(Arrays.asList(partition1));
```

#### 4. Offset Management

```java
// Manual offset management
Map<TopicPartition, OffsetAndMetadata> offsets = new HashMap<>();

for (ConsumerRecord<String, String> record : records) {
    processRecord(record);
    
    // Track offset for each partition
    offsets.put(
        new TopicPartition(record.topic(), record.partition()),
        new OffsetAndMetadata(record.offset() + 1)
    );
}

// Commit specific offsets
consumer.commitSync(offsets);
```

### Kafka Streams for Stream Processing

#### 1. Basic Stream Processing

```java
Properties props = new Properties();
props.put(StreamsConfig.APPLICATION_ID_CONFIG, "user-analytics");
props.put(StreamsConfig.BOOTSTRAP_SERVERS_CONFIG, "localhost:9092");
props.put(StreamsConfig.DEFAULT_KEY_SERDE_CLASS_CONFIG, Serdes.String().getClass());
props.put(StreamsConfig.DEFAULT_VALUE_SERDE_CLASS_CONFIG, Serdes.String().getClass());

StreamsBuilder builder = new StreamsBuilder();

// Create stream from topic
KStream<String, String> userEvents = builder.stream("user-events");

// Transform and filter
KStream<String, String> processedEvents = userEvents
    .filter((key, value) -> value.contains("purchase"))
    .mapValues(value -> processEvent(value));

// Write to output topic
processedEvents.to("processed-events");

KafkaStreams streams = new KafkaStreams(builder.build(), props);
streams.start();
```

#### 2. Windowed Aggregations

```java
// Count events per user in 5-minute windows
KTable<Windowed<String>, Long> eventCounts = userEvents
    .groupByKey()
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
    .count();

// Convert to stream and write results
eventCounts.toStream()
    .map((windowedKey, count) -> KeyValue.pair(
        windowedKey.key() + "@" + windowedKey.window().start(),
        count.toString()
    ))
    .to("event-counts");
```

#### 3. Stream-Table Joins

```java
// User profile table
KTable<String, String> userProfiles = builder.table("user-profiles");

// Join events with user profiles
KStream<String, String> enrichedEvents = userEvents
    .join(userProfiles, (event, profile) -> {
        return enrichEvent(event, profile);
    });
```

### Performance Optimization

#### 1. Producer Optimization

```java
Properties props = new Properties();

// Throughput optimization
props.put("acks", "1");                    // Balance reliability/performance
props.put("batch.size", "65536");          // 64KB batches
props.put("linger.ms", "20");              // Wait for batching
props.put("compression.type", "lz4");      // Fast compression
props.put("buffer.memory", "67108864");    // 64MB buffer

// Network optimization
props.put("max.in.flight.requests.per.connection", "5");
props.put("request.timeout.ms", "30000");
props.put("retry.backoff.ms", "100");
```

#### 2. Consumer Optimization

```java
Properties props = new Properties();

// Fetch optimization
props.put("fetch.min.bytes", "50000");     // 50KB minimum fetch
props.put("fetch.max.wait.ms", "500");     // Max wait for fetch.min.bytes
props.put("max.partition.fetch.bytes", "1048576"); // 1MB per partition
props.put("max.poll.records", "1000");     // Records per poll

// Session management
props.put("session.timeout.ms", "30000");  // 30s session timeout
props.put("heartbeat.interval.ms", "10000"); // 10s heartbeat
props.put("max.poll.interval.ms", "300000"); // 5min max poll interval
```

#### 3. Broker Configuration

```properties
# Server configuration (server.properties)

# Network threads
num.network.threads=8
num.io.threads=16

# Log configuration
log.segment.bytes=1073741824    # 1GB segments
log.retention.hours=168         # 7 days retention
log.retention.bytes=1073741824  # 1GB retention per partition

# Replication
default.replication.factor=3
min.insync.replicas=2
unclean.leader.election.enable=false

# Performance
socket.send.buffer.bytes=102400
socket.receive.buffer.bytes=102400
socket.request.max.bytes=104857600

# JVM heap
export KAFKA_HEAP_OPTS="-Xmx6g -Xms6g"
```

### Monitoring and Observability

#### 1. Key Metrics to Monitor

**Broker Metrics**:
```bash
# Throughput metrics
kafka.server:type=BrokerTopicMetrics,name=MessagesInPerSec
kafka.server:type=BrokerTopicMetrics,name=BytesInPerSec
kafka.server:type=BrokerTopicMetrics,name=BytesOutPerSec

# Replication metrics
kafka.server:type=ReplicaManager,name=LeaderCount
kafka.server:type=ReplicaManager,name=PartitionCount
kafka.server:type=ReplicaManager,name=UnderReplicatedPartitions

# Request metrics
kafka.network:type=RequestMetrics,name=RequestsPerSec,request=Produce
kafka.network:type=RequestMetrics,name=TotalTimeMs,request=Produce
```

**Consumer Metrics**:
```bash
# Lag metrics
kafka.consumer:type=consumer-fetch-manager-metrics,client-id=*,topic=*,partition=*

# Throughput metrics
kafka.consumer:type=consumer-fetch-manager-metrics,client-id=*
```

#### 2. Monitoring Tools

**JMX Monitoring**:
```java
// Enable JMX in Kafka
export JMX_PORT=9999
export KAFKA_JMX_OPTS="-Dcom.sun.management.jmxremote -Dcom.sun.management.jmxremote.authenticate=false -Dcom.sun.management.jmxremote.ssl=false"
```

**Kafka Manager/AKHQ**:
```yaml
# docker-compose.yml for AKHQ
akhq:
  image: tchiotludo/akhq
  ports:
    - "8080:8080"
  environment:
    AKHQ_CONFIGURATION: |
      akhq:
        connections:
          docker-kafka-server:
            properties:
              bootstrap.servers: "kafka:9092"
```

#### 3. Alerting Rules

```yaml
# Prometheus alerting rules
groups:
- name: kafka
  rules:
  - alert: KafkaConsumerLag
    expr: kafka_consumer_lag_sum > 1000
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Kafka consumer lag is high"
      
  - alert: KafkaUnderReplicatedPartitions
    expr: kafka_server_replicamanager_underreplicatedpartitions > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Kafka has under-replicated partitions"
```

### Production Best Practices

#### 1. Cluster Sizing

```bash
# Broker sizing calculation
target_throughput = 1 GB/s
broker_throughput = 100 MB/s  # Per broker
min_brokers = target_throughput / broker_throughput = 10

# Add replication overhead (3x for RF=3)
storage_brokers = min_brokers * 3 = 30

# Final broker count (max of throughput and storage requirements)
recommended_brokers = max(min_brokers, storage_brokers) = 30
```

#### 2. Topic Design

```bash
# Topic naming convention
<environment>.<team>.<domain>.<entity>.<version>
# Example: prod.analytics.user.events.v1

# Partition strategy
# High-throughput topics: 30-50 partitions
# Medium-throughput topics: 10-20 partitions
# Low-throughput topics: 3-6 partitions

# Replication factor
# Critical data: RF=3
# Important data: RF=2
# Development: RF=1
```

#### 3. Security Configuration

```properties
# SSL/TLS configuration
listeners=SSL://localhost:9093
security.inter.broker.protocol=SSL
ssl.keystore.location=/path/to/kafka.server.keystore.jks
ssl.keystore.password=password
ssl.key.password=password
ssl.truststore.location=/path/to/kafka.server.truststore.jks
ssl.truststore.password=password

# SASL authentication
sasl.enabled.mechanisms=PLAIN
sasl.mechanism.inter.broker.protocol=PLAIN
```

#### 4. Disaster Recovery

```bash
# Cross-cluster replication with MirrorMaker 2.0
connect-mirror-maker.sh mm2.properties

# mm2.properties
clusters = primary, backup
primary.bootstrap.servers = primary-cluster:9092
backup.bootstrap.servers = backup-cluster:9092

primary->backup.enabled = true
primary->backup.topics = .*
backup->primary.enabled = false

replication.factor = 3
checkpoints.topic.replication.factor = 3
heartbeats.topic.replication.factor = 3
offset-syncs.topic.replication.factor = 3
```

### Troubleshooting Common Issues

#### 1. Consumer Lag

```bash
# Check consumer lag
kafka-consumer-groups --bootstrap-server localhost:9092 \
  --group analytics-group --describe

# Causes and solutions:
# - Slow processing: Increase consumer instances or optimize processing
# - Network issues: Check network connectivity and bandwidth
# - Partition skew: Improve key distribution or use custom partitioner
```

#### 2. Under-Replicated Partitions

```bash
# Check under-replicated partitions
kafka-topics --bootstrap-server localhost:9092 --describe --under-replicated-partitions

# Causes and solutions:
# - Broker failure: Replace failed broker
# - Network issues: Check inter-broker connectivity
# - Disk issues: Check disk space and I/O performance
```

#### 3. Leader Election Issues

```bash
# Check partition leadership
kafka-topics --bootstrap-server localhost:9092 --describe --topic my-topic

# Causes and solutions:
# - ZooKeeper issues: Check ZooKeeper cluster health
# - Broker overload: Reduce load or add more brokers
# - Network partitions: Check network connectivity
```

---

## 💻 Hands-On Exercise

Build a production-ready Kafka cluster with advanced patterns.

**What you'll create**:
1. Multi-broker Kafka cluster with replication
2. Advanced producer with idempotence and transactions
3. Consumer groups with different assignment strategies
4. Stream processing with Kafka Streams
5. Monitoring and alerting setup
6. Performance optimization and tuning

**Skills practiced**:
- Kafka cluster configuration and management
- Advanced producer and consumer patterns
- Stream processing and windowed aggregations
- Performance monitoring and optimization
- Troubleshooting and maintenance

See `exercise.py` for hands-on practice and `exercise/` directory for complete infrastructure setup.

> **🔒 Security Note**: This lesson uses development configurations for learning purposes. In production, implement proper security with SSL/TLS encryption, SASL authentication, ACLs, and secure credential management. Never use default passwords or expose Kafka clusters without proper security controls.

---

## 📚 Resources

- [Apache Kafka Documentation](https://kafka.apache.org/documentation/)
- [Kafka: The Definitive Guide](https://www.confluent.io/resources/kafka-the-definitive-guide/)
- [Kafka Streams Documentation](https://kafka.apache.org/documentation/streams/)
- [Confluent Platform](https://docs.confluent.io/platform/current/overview.html)
- [Kafka Performance Tuning](https://kafka.apache.org/documentation/#performance)

---

## 🎯 Key Takeaways

- **Partitioning enables parallelism** and is key to Kafka's scalability
- **Replication provides fault tolerance** through leader-follower architecture
- **ISR (In-Sync Replicas)** ensures data consistency and availability
- **Producer acknowledgments** control durability vs performance trade-offs
- **Consumer groups** enable horizontal scaling of message processing
- **Kafka Streams** provides powerful stream processing capabilities
- **Monitoring is crucial** for production Kafka deployments
- **Proper configuration** dramatically impacts performance and reliability
- **Security and disaster recovery** are essential for production systems
- **Understanding trade-offs** helps optimize for specific use cases

---

## 🚀 What's Next?

Tomorrow (Day 7), you'll build a **Real-time CDC Pipeline Project** - combining everything learned so far to create a complete end-to-end streaming data pipeline from PostgreSQL through Kafka to analytics systems.

**Preview**: This project will integrate PostgreSQL CDC (Day 5), advanced Kafka patterns (Day 6), real-time processing, and analytics - demonstrating how all the pieces work together in a production system.

---

## ✅ Before Moving On

- [ ] Understand Kafka partitioning strategies and their impact
- [ ] Can configure replication and understand ISR concepts
- [ ] Know how to implement advanced producer patterns
- [ ] Can design consumer groups for different use cases
- [ ] Understand Kafka Streams for stream processing
- [ ] Can monitor and troubleshoot Kafka clusters
- [ ] Complete the exercise in `exercise.py`
- [ ] Set up the infrastructure using `exercise/docker-compose.yml`
- [ ] Review the solution in `solution.py`
- [ ] Follow the setup guide in `SETUP.md`
- [ ] Take the quiz in `quiz.md`

**Time**: ~1 hour | **Difficulty**: ⭐⭐⭐⭐ (Advanced)

Ready to master distributed streaming! 🚀