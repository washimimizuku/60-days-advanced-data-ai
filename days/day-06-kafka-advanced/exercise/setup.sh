#!/bin/bash
# Day 6: Advanced Kafka - Setup Script

set -e

echo "🚀 Setting up Advanced Kafka Exercise Environment"
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose and try again."
    exit 1
fi

echo "✅ Docker and Docker Compose are available"

# Create necessary directories
mkdir -p monitoring/grafana/{dashboards,datasources}
mkdir -p logs
mkdir -p data

echo "📁 Created necessary directories"

# Create Prometheus configuration
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'kafka-jmx'
    static_configs:
      - targets: ['jmx-kafka-1:5556']
    scrape_interval: 10s
    metrics_path: /metrics

  - job_name: 'kafka-connect'
    static_configs:
      - targets: ['connect:8083']
    metrics_path: /metrics
    scrape_interval: 10s
EOF

# Create Grafana datasource configuration
cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

# Create basic Kafka dashboard
cat > monitoring/grafana/dashboards/kafka-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Kafka Cluster Overview",
    "tags": ["kafka"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Messages In Per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(kafka_server_brokertopicmetrics_messagesinpersec[5m])",
            "legendFormat": "{{topic}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "Bytes In Per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(kafka_server_brokertopicmetrics_bytesinpersec[5m])",
            "legendFormat": "{{topic}}"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "10s"
  }
}
EOF

echo "📊 Created monitoring configuration"

# Start ZooKeeper first
echo "🐘 Starting ZooKeeper..."
docker-compose up -d zookeeper-1

# Wait for ZooKeeper
echo "⏳ Waiting for ZooKeeper to start..."
sleep 15

# Start Kafka brokers
echo "🎯 Starting Kafka brokers..."
docker-compose up -d kafka-1 kafka-2 kafka-3

# Wait for Kafka brokers
echo "⏳ Waiting for Kafka brokers to start..."
sleep 30

# Check if Kafka is ready
echo "🔍 Checking Kafka connectivity..."
timeout 60 bash -c 'until docker-compose exec kafka-1 kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; do sleep 2; done'

# Start additional services
echo "🔗 Starting additional services..."
docker-compose up -d schema-registry connect redis

# Wait for services
echo "⏳ Waiting for services to start..."
sleep 20

# Start monitoring stack
echo "📊 Starting monitoring stack..."
docker-compose up -d prometheus jmx-kafka-1 grafana akhq

echo "⏳ Waiting for monitoring services..."
sleep 15

# Create sample topics with different configurations
echo "📝 Creating sample topics..."

# High-throughput topic with many partitions
docker-compose exec kafka-1 kafka-topics --create \
  --topic user-events \
  --partitions 12 \
  --replication-factor 3 \
  --config min.insync.replicas=2 \
  --config cleanup.policy=delete \
  --config retention.ms=604800000 \
  --bootstrap-server localhost:9092

# Compacted topic for user profiles
docker-compose exec kafka-1 kafka-topics --create \
  --topic user-profiles \
  --partitions 6 \
  --replication-factor 3 \
  --config min.insync.replicas=2 \
  --config cleanup.policy=compact \
  --config segment.ms=86400000 \
  --bootstrap-server localhost:9092

# Low-latency topic for alerts
docker-compose exec kafka-1 kafka-topics --create \
  --topic alerts \
  --partitions 3 \
  --replication-factor 3 \
  --config min.insync.replicas=2 \
  --config flush.ms=1000 \
  --bootstrap-server localhost:9092

# Analytics results topic
docker-compose exec kafka-1 kafka-topics --create \
  --topic analytics-results \
  --partitions 6 \
  --replication-factor 3 \
  --config min.insync.replicas=2 \
  --bootstrap-server localhost:9092

echo "✅ Created sample topics"

# List topics to verify
echo "📋 Verifying topics:"
docker-compose exec kafka-1 kafka-topics --list --bootstrap-server localhost:9092

# Show topic details
echo "📊 Topic details:"
docker-compose exec kafka-1 kafka-topics --describe --bootstrap-server localhost:9092

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
if command -v python3 &> /dev/null; then
    python3 -m pip install -r requirements.txt
elif command -v python &> /dev/null; then
    python -m pip install -r requirements.txt
else
    echo "⚠️  Python not found. Please install Python dependencies manually:"
    echo "   pip install -r requirements.txt"
fi

# Create performance test script
cat > performance_test.py << 'EOF'
#!/usr/bin/env python3
"""Performance testing script for Kafka cluster"""

import time
import json
import threading
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
import random
import uuid

class KafkaPerformanceTest:
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.bootstrap_servers = bootstrap_servers
        self.results = {
            'messages_sent': 0,
            'messages_received': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
        
    def create_producer(self, **config):
        """Create optimized producer"""
        default_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'key_serializer': lambda x: x.encode('utf-8') if x else None,
            'value_serializer': lambda x: json.dumps(x).encode('utf-8'),
            'acks': 'all',
            'retries': 3,
            'batch_size': 16384,
            'linger_ms': 10,
            'buffer_memory': 33554432,
            'compression_type': 'snappy',
            'enable_idempotence': True
        }
        default_config.update(config)
        return KafkaProducer(**default_config)
    
    def create_consumer(self, topics, group_id, **config):
        """Create optimized consumer"""
        default_config = {
            'bootstrap_servers': self.bootstrap_servers,
            'group_id': group_id,
            'key_deserializer': lambda x: x.decode('utf-8') if x else None,
            'value_deserializer': lambda x: json.loads(x.decode('utf-8')),
            'auto_offset_reset': 'earliest',
            'enable_auto_commit': False,
            'fetch_min_bytes': 1024,
            'fetch_max_wait_ms': 500,
            'max_poll_records': 500
        }
        default_config.update(config)
        consumer = KafkaConsumer(**default_config)
        consumer.subscribe(topics)
        return consumer
    
    def produce_messages(self, topic, num_messages, message_size=1024):
        """Produce messages to test throughput"""
        producer = self.create_producer()
        
        # Generate sample message
        sample_data = {
            'user_id': str(uuid.uuid4()),
            'event_type': 'page_view',
            'timestamp': datetime.now().isoformat(),
            'data': 'x' * (message_size - 200)  # Approximate message size
        }
        
        start_time = time.time()
        
        for i in range(num_messages):
            try:
                # Use user_id as key for consistent partitioning
                key = f"user_{i % 1000}"
                sample_data['sequence'] = i
                
                future = producer.send(topic, key=key, value=sample_data)
                
                if i % 1000 == 0:
                    print(f"Sent {i} messages...")
                    
            except KafkaError as e:
                print(f"Error sending message {i}: {e}")
                self.results['errors'] += 1
        
        # Flush remaining messages
        producer.flush()
        producer.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Produced {num_messages} messages in {duration:.2f} seconds")
        print(f"Throughput: {num_messages/duration:.2f} messages/sec")
        
        return duration
    
    def consume_messages(self, topics, group_id, max_messages=None, timeout=30):
        """Consume messages to test throughput"""
        consumer = self.create_consumer(topics, group_id)
        
        messages_consumed = 0
        start_time = time.time()
        last_message_time = start_time
        
        try:
            while True:
                msg_pack = consumer.poll(timeout_ms=1000)
                
                if not msg_pack:
                    # Check timeout
                    if time.time() - last_message_time > timeout:
                        print(f"No messages received for {timeout} seconds, stopping...")
                        break
                    continue
                
                for tp, messages in msg_pack.items():
                    for message in messages:
                        messages_consumed += 1
                        last_message_time = time.time()
                        
                        if messages_consumed % 1000 == 0:
                            print(f"Consumed {messages_consumed} messages...")
                        
                        if max_messages and messages_consumed >= max_messages:
                            break
                    
                    if max_messages and messages_consumed >= max_messages:
                        break
                
                # Commit offsets
                consumer.commit()
                
                if max_messages and messages_consumed >= max_messages:
                    break
                    
        except KeyboardInterrupt:
            print("Consumer interrupted by user")
        finally:
            consumer.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"Consumed {messages_consumed} messages in {duration:.2f} seconds")
        if duration > 0:
            print(f"Throughput: {messages_consumed/duration:.2f} messages/sec")
        
        return messages_consumed, duration

def run_performance_test():
    """Run comprehensive performance test"""
    test = KafkaPerformanceTest()
    
    print("🚀 Starting Kafka Performance Test")
    print("=" * 50)
    
    # Test 1: High-throughput producer
    print("\n📤 Test 1: Producer Throughput")
    print("-" * 30)
    produce_duration = test.produce_messages('user-events', 10000, 1024)
    
    # Test 2: Consumer throughput
    print("\n📥 Test 2: Consumer Throughput")
    print("-" * 30)
    consumed, consume_duration = test.consume_messages(['user-events'], 'perf-test-group', 10000, 60)
    
    # Test 3: Multiple consumers (consumer group)
    print("\n👥 Test 3: Consumer Group Performance")
    print("-" * 40)
    
    def consumer_worker(group_id, consumer_id):
        test_consumer = KafkaPerformanceTest()
        consumed, duration = test_consumer.consume_messages(
            ['user-events'], 
            group_id, 
            max_messages=2000,
            timeout=30
        )
        print(f"Consumer {consumer_id} consumed {consumed} messages in {duration:.2f}s")
    
    # Start multiple consumers
    threads = []
    for i in range(3):
        thread = threading.Thread(
            target=consumer_worker, 
            args=('multi-consumer-group', f'consumer-{i}')
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all consumers to finish
    for thread in threads:
        thread.join()
    
    print("\n✅ Performance test completed!")

if __name__ == "__main__":
    run_performance_test()
EOF

chmod +x performance_test.py

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Services Status:"
docker-compose ps
echo ""
echo "🔗 Access Points:"
echo "- AKHQ (Kafka UI): http://localhost:8080"
echo "- Grafana: http://localhost:3000 (admin/admin)"
echo "- Prometheus: http://localhost:9090"
echo ""
echo "📊 Kafka Brokers:"
echo "- Broker 1: localhost:9092"
echo "- Broker 2: localhost:9093"
echo "- Broker 3: localhost:9094"
echo ""
echo "🧪 Next Steps:"
echo "1. Run performance test:"
echo "   python3 performance_test.py"
echo ""
echo "2. Check topic details:"
echo "   docker-compose exec kafka-1 kafka-topics --describe --bootstrap-server localhost:9092"
echo ""
echo "3. Monitor consumer groups:"
echo "   docker-compose exec kafka-1 kafka-consumer-groups --bootstrap-server localhost:9092 --list"
echo ""
echo "4. Start the advanced exercises:"
echo "   python3 advanced_producer.py"
echo "   python3 advanced_consumer.py"
echo "   python3 kafka_streams_example.py"
echo ""
echo "🔧 Troubleshooting:"
echo "- Check logs: docker-compose logs [service_name]"
echo "- Restart cluster: docker-compose restart"
echo "- Clean restart: docker-compose down && docker-compose up -d"