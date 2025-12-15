# Day 6 Quiz: Advanced Kafka - Partitions and Replication

## Questions

### 1. What is the maximum number of consumers that can effectively consume from a topic with 6 partitions?
- a) 3 consumers
- b) 6 consumers
- c) 12 consumers
- d) Unlimited consumers

### 2. In Kafka replication, what does ISR stand for?
- a) Internal System Replicas
- b) In-Sync Replicas
- c) Isolated Sync Replicas
- d) Indexed Storage Replicas

### 3. Which producer acknowledgment setting provides the highest durability?
- a) acks=0
- b) acks=1
- c) acks=all
- d) acks=2

### 4. What happens when a Kafka consumer group rebalances?
- a) All messages are deleted
- b) Partitions are redistributed among consumers
- c) The topic is recreated
- d) Brokers restart automatically

### 5. Which configuration enables exactly-once semantics in Kafka producers?
- a) enable.idempotence=true only
- b) acks=all only
- c) enable.idempotence=true and transactional.id set
- d) retries=0

### 6. What is the purpose of the linger.ms setting in Kafka producers?
- a) Maximum time to wait for acknowledgments
- b) Time to wait for additional messages before sending a batch
- c) Consumer polling interval
- d) Broker response timeout

### 7. In key-based partitioning, messages with the same key will:
- a) Be distributed randomly across partitions
- b) Always go to the same partition
- c) Be load-balanced across all brokers
- d) Be replicated to all partitions

### 8. What is the minimum number of brokers required for a replication factor of 3?
- a) 1 broker
- b) 2 brokers
- c) 3 brokers
- d) 6 brokers

### 9. Which Kafka Streams operation is used for windowed aggregations?
- a) groupBy()
- b) windowedBy()
- c) aggregate()
- d) Both b and c

### 10. What happens when min.insync.replicas=2 and only 1 replica is in-sync?
- a) Writes succeed normally
- b) Writes are rejected with an error
- c) The partition becomes read-only
- d) Kafka automatically creates more replicas

---

## Answers

### 1. What is the maximum number of consumers that can effectively consume from a topic with 6 partitions?
**Answer: b) 6 consumers**

**Explanation:** In Kafka, each partition can be consumed by only one consumer within a consumer group. Therefore, the maximum number of consumers that can effectively consume from a topic is equal to the number of partitions. With 6 partitions, you can have at most 6 consumers in a consumer group. Additional consumers would remain idle. However, you can have multiple consumer groups, each with up to 6 consumers.

---

### 2. In Kafka replication, what does ISR stand for?
**Answer: b) In-Sync Replicas**

**Explanation:** ISR stands for In-Sync Replicas, which are the set of replicas that are fully caught up with the leader partition. Only ISR replicas are eligible to become the new leader if the current leader fails. The ISR list is dynamically maintained by Kafka - replicas that fall too far behind are removed from the ISR, and those that catch up are added back.

---

### 3. Which producer acknowledgment setting provides the highest durability?
**Answer: c) acks=all**

**Explanation:** `acks=all` (equivalent to `acks=-1`) provides the highest durability by requiring acknowledgment from all in-sync replicas before considering a write successful. This ensures that the message is replicated to all ISR members. `acks=0` provides no durability (fire-and-forget), `acks=1` waits only for the leader acknowledgment, and `acks=2` is not a valid setting.

---

### 4. What happens when a Kafka consumer group rebalances?
**Answer: b) Partitions are redistributed among consumers**

**Explanation:** During a rebalance, Kafka redistributes partition assignments among the active consumers in the group. This happens when consumers join or leave the group, or when the number of partitions changes. During rebalancing, consumption is temporarily paused, consumers commit their current offsets, and then partitions are reassigned according to the partition assignment strategy (range, round-robin, sticky, etc.).

---

### 5. Which configuration enables exactly-once semantics in Kafka producers?
**Answer: c) enable.idempotence=true and transactional.id set**

**Explanation:** Exactly-once semantics requires both idempotent producers (`enable.idempotence=true`) and transactions (`transactional.id` set). Idempotence prevents duplicate messages within a single producer session, while transactions provide atomicity across multiple partitions and topics. You also need `acks=all` and appropriate retry settings, but the key requirement is both idempotence and transactions.

---

### 6. What is the purpose of the linger.ms setting in Kafka producers?
**Answer: b) Time to wait for additional messages before sending a batch**

**Explanation:** `linger.ms` controls how long the producer waits for additional messages to arrive before sending a batch. This allows for better batching and higher throughput at the cost of slightly increased latency. For example, `linger.ms=10` means the producer will wait up to 10 milliseconds for more messages to fill the batch before sending it to the broker.

---

### 7. In key-based partitioning, messages with the same key will:
**Answer: b) Always go to the same partition**

**Explanation:** Kafka uses a hash of the message key to determine which partition to send the message to: `partition = hash(key) % number_of_partitions`. This ensures that all messages with the same key go to the same partition, which guarantees ordering for messages with the same key. This is crucial for maintaining order in event streams where sequence matters.

---

### 8. What is the minimum number of brokers required for a replication factor of 3?
**Answer: c) 3 brokers**

**Explanation:** The replication factor determines how many copies of each partition exist across the cluster. For a replication factor of 3, you need at least 3 brokers to store the 3 replicas (1 leader + 2 followers). While you could technically create the topic with fewer brokers, you wouldn't achieve the desired replication level, and some replicas would be missing.

---

### 9. Which Kafka Streams operation is used for windowed aggregations?
**Answer: d) Both b and c**

**Explanation:** Windowed aggregations in Kafka Streams require both `windowedBy()` to define the time window and `aggregate()` (or similar operations like `count()`, `reduce()`) to perform the aggregation. The typical pattern is: `stream.groupByKey().windowedBy(TimeWindows.of(Duration.ofMinutes(5))).count()`. The `windowedBy()` operation creates windowed groups, and then aggregation operations are applied.

---

### 10. What happens when min.insync.replicas=2 and only 1 replica is in-sync?
**Answer: b) Writes are rejected with an error**

**Explanation:** When `min.insync.replicas=2` but only 1 replica is in-sync (including the leader), Kafka will reject write requests with a `NotEnoughReplicasException`. This setting ensures that writes only succeed when there are at least the specified number of in-sync replicas, providing a durability guarantee. The partition remains available for reads, but writes are blocked until more replicas come back in-sync.

---

## Score Interpretation

- **9-10 correct**: Excellent! You have a strong understanding of advanced Kafka concepts
- **7-8 correct**: Good job! Review the questions you missed and practice more with Kafka configuration
- **5-6 correct**: You're on the right track. Focus on understanding partitioning, replication, and consumer groups
- **Below 5**: Review the theory section and practice with the hands-on exercises

---

## Key Concepts to Remember

1. **Maximum consumers = number of partitions** per consumer group
2. **ISR (In-Sync Replicas)** are replicas caught up with the leader
3. **acks=all** provides highest durability by waiting for all ISR
4. **Rebalancing redistributes partitions** among consumers in a group
5. **Exactly-once needs idempotence + transactions** (transactional.id)
6. **linger.ms waits for batching** to improve throughput
7. **Same key always goes to same partition** for ordering guarantees
8. **Replication factor = minimum brokers** needed for full replication
9. **Windowed aggregations need windowedBy() + aggregate()** operations
10. **min.insync.replicas enforces durability** by rejecting writes when not met

---

## Kafka Best Practices

### Partitioning Strategy
- **Choose partition count** based on target throughput and consumer parallelism
- **Use meaningful keys** for ordering requirements
- **Consider future scaling** when setting partition count (can't decrease)
- **Monitor partition skew** and adjust key distribution if needed

### Replication and Durability
- **Use replication factor 3** for production topics
- **Set min.insync.replicas=2** for critical data
- **Use acks=all** for important messages
- **Monitor ISR shrinks** and under-replicated partitions

### Producer Optimization
- **Enable idempotence** for production producers
- **Use appropriate batch.size** and linger.ms for throughput
- **Configure retries** and timeout settings properly
- **Use compression** (snappy, lz4) for better network utilization

### Consumer Optimization
- **Use manual offset management** for critical applications
- **Configure session.timeout.ms** and heartbeat.interval.ms appropriately
- **Handle rebalances gracefully** with proper cleanup
- **Monitor consumer lag** and scale consumers as needed

### Monitoring and Operations
- **Monitor key metrics**: throughput, latency, consumer lag, ISR health
- **Set up alerting** for under-replicated partitions and consumer lag
- **Plan for capacity** based on retention and throughput requirements
- **Test failover scenarios** regularly

Ready to move on to Day 7! 🚀