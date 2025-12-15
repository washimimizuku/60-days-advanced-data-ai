# Day 6: Advanced Kafka - Setup Guide

## 🚀 Quick Start (15 minutes)

### Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ with pip
- 16GB+ RAM available (for 3-broker cluster)
- Ports 9092-9094, 2181, 8080-8083, 3000, 9090 available

### 1. Start Kafka Cluster

```bash
# Navigate to exercise directory
cd days/day-06-kafka-advanced/exercise/

# Start all services (3-broker cluster + monitoring)
docker-compose up -d

# Verify all services are running
docker-compose ps
```

### 2. Install Python Dependencies

```bash
# Install requirements
pip install -r requirements.txt

# Or install in virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Verify Cluster Setup

```bash
# Check Kafka brokers
docker exec -it kafka-1 kafka-topics --bootstrap-server localhost:9092 --list

# Check cluster metadata
docker exec -it kafka-1 kafka-broker-api-versions --bootstrap-server localhost:9092

# Test connectivity to all brokers
docker exec -it kafka-1 kafka-topics --bootstrap-server kafka-1:29092,kafka-2:29093,kafka-3:29094 --list
```

### 4. Access Monitoring Tools

- **AKHQ (Kafka UI)**: http://localhost:8080
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### 5. Run Initial Tests

```bash
# Run the solution to test all components
python solution.py

# Or run individual exercises
python exercise.py
```

## 🔧 Detailed Setup

### Multi-Broker Kafka Cluster

The setup includes:
- **3 Kafka Brokers** (kafka-1, kafka-2, kafka-3)
- **ZooKeeper** for coordination
- **Schema Registry** for schema management
- **Kafka Connect** for connectors
- **AKHQ** for cluster management
- **Prometheus + Grafana** for monitoring

### Broker Configuration

Each broker is configured with:
- **Replication Factor**: 3 (default)
- **Min ISR**: 2 (minimum in-sync replicas)
- **Auto Topic Creation**: Disabled (explicit topic management)
- **Log Retention**: 7 days
- **JMX Metrics**: Enabled for monitoring

### Topic Configuration

Default topic settings:
- **Partitions**: Varies by use case (3-12)
- **Replication Factor**: 2-3
- **Cleanup Policy**: Delete or Compact
- **Compression**: LZ4 for performance

## 📊 Monitoring Setup

### Prometheus Configuration

Monitors:
- Kafka broker metrics (JMX)
- Producer/consumer metrics
- System metrics (CPU, memory, disk)
- Custom application metrics

### Grafana Dashboards

Pre-configured dashboards for:
- Kafka cluster overview
- Broker performance
- Topic and partition metrics
- Consumer lag monitoring
- Producer throughput

### AKHQ Features

- Topic management and browsing
- Consumer group monitoring
- Schema registry integration
- Kafka Connect management
- Real-time message viewing

## 🧪 Testing the Setup

### 1. Create Test Topics

```bash
# Create high-throughput topic
docker exec -it kafka-1 kafka-topics \\\n  --create \\\n  --topic user-events \\\n  --partitions 12 \\\n  --replication-factor 3 \\\n  --bootstrap-server localhost:9092\n\n# Create low-latency topic\ndocker exec -it kafka-1 kafka-topics \\\n  --create \\\n  --topic real-time-alerts \\\n  --partitions 3 \\\n  --replication-factor 3 \\\n  --config min.insync.replicas=2 \\\n  --bootstrap-server localhost:9092\n```\n\n### 2. Test Producer Performance\n\n```bash\n# High-throughput producer test\ndocker exec -it kafka-1 kafka-producer-perf-test \\\n  --topic user-events \\\n  --num-records 100000 \\\n  --record-size 1024 \\\n  --throughput 10000 \\\n  --producer-props bootstrap.servers=localhost:9092\n```\n\n### 3. Test Consumer Performance\n\n```bash\n# Consumer performance test\ndocker exec -it kafka-1 kafka-consumer-perf-test \\\n  --topic user-events \\\n  --messages 100000 \\\n  --bootstrap-server localhost:9092\n```\n\n### 4. Verify Replication\n\n```bash\n# Check topic details and ISR\ndocker exec -it kafka-1 kafka-topics \\\n  --describe \\\n  --topic user-events \\\n  --bootstrap-server localhost:9092\n```\n\n## 🚨 Troubleshooting\n\n### Common Issues\n\n**Brokers not starting:**\n```bash\n# Check broker logs\ndocker logs kafka-1\ndocker logs kafka-2\ndocker logs kafka-3\n\n# Check ZooKeeper\ndocker logs zookeeper-1\n```\n\n**Out of memory errors:**\n```bash\n# Reduce JVM heap size in docker-compose.yml\n# Add to broker environment:\nKAFKA_HEAP_OPTS: \"-Xmx2G -Xms2G\"\n```\n\n**Port conflicts:**\n```bash\n# Check port usage\nnetstat -tulpn | grep :9092\n\n# Stop conflicting services or change ports in docker-compose.yml\n```\n\n**Consumer lag issues:**\n```bash\n# Check consumer group status\ndocker exec -it kafka-1 kafka-consumer-groups \\\n  --bootstrap-server localhost:9092 \\\n  --describe --group analytics-group\n```\n\n### Performance Tuning\n\n**Broker Optimization:**\n```properties\n# Add to broker configuration\nnum.network.threads=8\nnum.io.threads=16\nsocket.send.buffer.bytes=102400\nsocket.receive.buffer.bytes=102400\n```\n\n**Producer Optimization:**\n```python\n# High-throughput settings\nproducer = KafkaProducer(\n    batch_size=65536,\n    linger_ms=20,\n    compression_type='lz4',\n    buffer_memory=67108864\n)\n```\n\n**Consumer Optimization:**\n```python\n# High-throughput settings\nconsumer = KafkaConsumer(\n    fetch_min_bytes=50000,\n    fetch_max_wait_ms=500,\n    max_partition_fetch_bytes=1048576\n)\n```\n\n## 🔍 Monitoring and Alerting\n\n### Key Metrics to Watch\n\n**Broker Metrics:**\n- Messages in/out per second\n- Bytes in/out per second\n- Under-replicated partitions\n- Leader election rate\n\n**Producer Metrics:**\n- Request latency\n- Batch size\n- Compression ratio\n- Error rate\n\n**Consumer Metrics:**\n- Consumer lag\n- Fetch latency\n- Rebalance frequency\n- Processing rate\n\n### Alerting Rules\n\n```yaml\n# Prometheus alerting rules\ngroups:\n- name: kafka\n  rules:\n  - alert: KafkaConsumerLag\n    expr: kafka_consumer_lag_sum > 10000\n    for: 5m\n    \n  - alert: KafkaUnderReplicatedPartitions\n    expr: kafka_server_replicamanager_underreplicatedpartitions > 0\n    for: 1m\n    \n  - alert: KafkaBrokerDown\n    expr: up{job=\"kafka\"} == 0\n    for: 30s\n```\n\n## 🧹 Cleanup\n\n```bash\n# Stop all services\ndocker-compose down\n\n# Remove volumes (WARNING: deletes all data)\ndocker-compose down -v\n\n# Remove images\ndocker-compose down --rmi all\n\n# Clean up Docker system\ndocker system prune -f\n```\n\n## 📚 Next Steps\n\n1. Complete the exercise in `exercise.py`\n2. Run the solution in `solution.py`\n3. Explore AKHQ at http://localhost:8080\n4. Monitor metrics in Grafana at http://localhost:3000\n5. Take the quiz in `quiz.md`\n\n## 🔗 Useful Commands\n\n```bash\n# List all topics\ndocker exec -it kafka-1 kafka-topics --bootstrap-server localhost:9092 --list\n\n# Describe topic\ndocker exec -it kafka-1 kafka-topics --bootstrap-server localhost:9092 --describe --topic user-events\n\n# List consumer groups\ndocker exec -it kafka-1 kafka-consumer-groups --bootstrap-server localhost:9092 --list\n\n# Reset consumer group offset\ndocker exec -it kafka-1 kafka-consumer-groups \\\n  --bootstrap-server localhost:9092 \\\n  --group analytics-group \\\n  --reset-offsets \\\n  --to-earliest \\\n  --topic user-events \\\n  --execute\n\n# Check cluster metadata\ndocker exec -it kafka-1 kafka-metadata-shell --snapshot /var/lib/kafka/data/__cluster_metadata-0/00000000000000000000.log\n\n# Monitor log segments\ndocker exec -it kafka-1 kafka-log-dirs --bootstrap-server localhost:9092 --describe\n```\n\n## 🎯 Learning Objectives Verification\n\nAfter setup, you should be able to:\n- ✅ Access 3-broker Kafka cluster\n- ✅ Create topics with different partition strategies\n- ✅ Monitor cluster health via AKHQ\n- ✅ View metrics in Grafana dashboards\n- ✅ Run producer/consumer performance tests\n- ✅ Simulate broker failures and recovery\n\nReady to master advanced Kafka patterns! 🚀