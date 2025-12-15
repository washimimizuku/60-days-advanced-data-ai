# Day 7: Project - Real-time CDC Pipeline

## 🎯 Project Overview

Build a complete end-to-end Change Data Capture (CDC) pipeline that captures real-time changes from a PostgreSQL e-commerce database, streams them through Kafka, processes them for analytics, and provides monitoring and alerting capabilities.

**Duration**: 3 hours  
**Difficulty**: ⭐⭐⭐⭐⭐ (Expert)  
**Type**: Integration Project

---

## 🎯 Learning Objectives

- **Integration Mastery**: Combine PostgreSQL, Debezium, Kafka, and stream processing
- **Production Readiness**: Build scalable, monitored, fault-tolerant systems
- **Real-time Analytics**: Process streaming data for immediate business insights
- **Data Quality**: Implement validation, error handling, and data governance
- **Operational Excellence**: Deploy monitoring, alerting, and observability

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CDC Pipeline Architecture                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │    Debezium     │    │     Kafka       │
│                 │───▶│   Connector     │───▶│   Cluster       │
│ ┌─────────────┐ │    │                 │    │ ┌─────────────┐ │
│ │ users       │ │    │ • WAL Reader    │    │ │ users       │ │
│ │ orders      │ │    │ • Schema Detect │    │ │ orders      │ │
│ │ products    │ │    │ • Event Format  │    │ │ products    │ │
│ │ order_items │ │    │ • Offset Track  │    │ │ order_items │ │
│ └─────────────┘ │    └─────────────────┘    │ └─────────────┘ │
└─────────────────┘                           └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   Analytics     │    │ Stream Processor│◀────────────┘
│   Database      │◀───│                 │
│                 │    │ ┌─────────────┐ │
│ ┌─────────────┐ │    │ │ Revenue     │ │
│ │ revenue     │ │    │ │ Analytics   │ │
│ │ user_stats  │ │    │ │ User Behav  │ │
│ │ inventory   │ │    │ │ Inventory   │ │
│ │ alerts      │ │    │ │ Monitoring  │ │
│ └─────────────┘ │    │ └─────────────┘ │
└─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │ Data Quality    │    │ Schema Registry │
│                 │    │                 │    │                 │
│ • Prometheus    │    │ • Validation    │    │ • Avro Schemas  │
│ • Grafana       │    │ • Dead Letter Q │    │ • Evolution     │
│ • Alertmanager  │    │ • Error Metrics │    │ • Compatibility │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## 📋 Project Requirements

### 🎯 Functional Requirements

#### 1. Real-time Data Capture
- [x] Capture all PostgreSQL changes (INSERT, UPDATE, DELETE)
- [x] Handle initial snapshot and ongoing changes
- [x] Maintain event ordering per partition
- [x] Support schema evolution without downtime

#### 2. Stream Processing
- [x] Real-time revenue analytics (5-minute windows)
- [x] User behavior tracking and segmentation
- [x] Inventory monitoring with low-stock alerts
- [x] Data enrichment through stream-table joins

#### 3. Data Quality & Governance
- [x] Schema validation for all events
- [x] Business rule validation (positive prices, valid statuses)
- [x] Duplicate detection and deduplication
- [x] Dead letter queue for invalid events

#### 4. Analytics & Insights
- [x] Real-time dashboards for business metrics
- [x] Automated alerting for critical events
- [x] Historical trend analysis
- [x] Data lineage tracking

### ⚡ Non-Functional Requirements

#### 1. Performance
- **Throughput**: Handle 10,000+ events per second
- **Latency**: End-to-end processing < 100ms (95th percentile)
- **Scalability**: Horizontal scaling for all components
- **Efficiency**: Optimal resource utilization

#### 2. Reliability
- **Availability**: 99.9% uptime SLA
- **Durability**: Zero data loss guarantee
- **Recovery**: RTO < 5 minutes, RPO < 1 minute
- **Fault Tolerance**: Automatic failover and recovery

#### 3. Observability
- **Monitoring**: Real-time metrics and dashboards
- **Alerting**: Proactive notifications for issues
- **Logging**: Comprehensive audit trail
- **Tracing**: End-to-end request tracking

---

## 🚀 Implementation Plan

### Phase 1: Infrastructure Setup (45 minutes)

#### 1.1 Environment Preparation
```bash
# Create project directory
mkdir cdc-pipeline-project
cd cdc-pipeline-project

# Initialize Docker Compose environment
touch docker-compose.yml
mkdir -p {postgres,kafka,debezium,monitoring,stream-processor}/{config,data,logs}
```

#### 1.2 Core Services Configuration
- **PostgreSQL**: Configure for logical replication
- **Kafka Cluster**: 3-broker setup with proper partitioning
- **Debezium Connect**: Distributed mode with PostgreSQL connector
- **Schema Registry**: Avro schema management
- **Zookeeper**: Kafka coordination service

#### 1.3 Sample Data Setup
```sql
-- Create e-commerce database schema
-- Load sample data (users, products, orders)
-- Configure replication slot and publication
-- Set up triggers for updated_at timestamps
```

### Phase 2: CDC Pipeline Implementation (60 minutes)

#### 2.1 Debezium Connector Configuration
```json
{
  "name": "ecommerce-postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium_user",
    "database.password": "debezium_password",
    "database.dbname": "ecommerce",
    "database.server.name": "ecommerce-db",
    "table.include.list": "public.users,public.orders,public.order_items,public.products",
    "plugin.name": "pgoutput",
    "slot.name": "debezium_slot",
    "publication.name": "debezium_publication",
    "transforms": "route,unwrap",
    "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.route.regex": "([^.]+)\\.([^.]+)\\.([^.]+)",
    "transforms.route.replacement": "$3",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": "false",
    "key.converter": "io.confluent.connect.avro.AvroConverter",
    "value.converter": "io.confluent.connect.avro.AvroConverter",
    "key.converter.schema.registry.url": "http://schema-registry:8081",
    "value.converter.schema.registry.url": "http://schema-registry:8081"
  }
}
```

#### 2.2 Kafka Topic Configuration
```bash
# Create topics with appropriate partitioning
kafka-topics --create --topic users --partitions 6 --replication-factor 3
kafka-topics --create --topic orders --partitions 12 --replication-factor 3
kafka-topics --create --topic products --partitions 3 --replication-factor 3
kafka-topics --create --topic order_items --partitions 12 --replication-factor 3

# Create processed data topics
kafka-topics --create --topic revenue-analytics --partitions 6 --replication-factor 3
kafka-topics --create --topic user-behavior --partitions 6 --replication-factor 3
kafka-topics --create --topic inventory-alerts --partitions 3 --replication-factor 3
```

#### 2.3 Schema Registry Setup
```bash
# Register Avro schemas for all entities
# Configure compatibility settings
# Set up schema evolution policies
```

### Phase 3: Stream Processing (75 minutes)

#### 3.1 Revenue Analytics Stream
```java
@Component
public class RevenueAnalyticsProcessor {
    
    @StreamListener("orders")
    @SendTo("revenue-analytics")
    public KStream<String, RevenueMetrics> processRevenue(
            KStream<String, Order> orders) {
        
        return orders
            .filter((key, order) -> "completed".equals(order.getStatus()))
            .groupBy((key, order) -> "revenue")
            .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
            .aggregate(
                RevenueMetrics::new,
                (key, order, metrics) -> metrics.addOrder(order),
                Materialized.with(Serdes.String(), revenueMetricsSerde)
            )
            .toStream()
            .map((windowedKey, metrics) -> KeyValue.pair(
                windowedKey.key() + "-" + windowedKey.window().start(),
                metrics
            ));
    }
}
```

#### 3.2 User Behavior Tracking
```java
@Component
public class UserBehaviorProcessor {
    
    @StreamListener("user-events")
    @SendTo("user-behavior")
    public KStream<String, UserSession> trackUserBehavior(
            KStream<String, UserEvent> userEvents) {
        
        return userEvents
            .groupByKey()
            .windowedBy(SessionWindows.with(Duration.ofMinutes(30)))
            .aggregate(
                UserSession::new,
                (key, event, session) -> session.addEvent(event),
                (key, session1, session2) -> session1.merge(session2),
                Materialized.with(Serdes.String(), userSessionSerde)
            )
            .toStream()
            .map((windowedKey, session) -> KeyValue.pair(
                windowedKey.key(),
                session
            ));
    }
}
```

#### 3.3 Inventory Monitoring
```java
@Component
public class InventoryMonitoringProcessor {
    
    @StreamListener("products")
    @SendTo("inventory-alerts")
    public KStream<String, InventoryAlert> monitorInventory(
            KStream<String, Product> products) {
        
        return products
            .filter((key, product) -> product.getStockQuantity() < product.getMinThreshold())
            .map((key, product) -> KeyValue.pair(
                key,
                InventoryAlert.builder()
                    .productId(product.getProductId())
                    .productName(product.getName())
                    .currentStock(product.getStockQuantity())
                    .threshold(product.getMinThreshold())
                    .alertLevel(determineAlertLevel(product))
                    .timestamp(Instant.now())
                    .build()
            ));
    }
    
    private AlertLevel determineAlertLevel(Product product) {
        int stock = product.getStockQuantity();
        int threshold = product.getMinThreshold();
        
        if (stock == 0) return AlertLevel.CRITICAL;
        if (stock < threshold * 0.5) return AlertLevel.HIGH;
        return AlertLevel.MEDIUM;
    }
}
```

#### 3.4 Data Enrichment
```java
@Component
public class DataEnrichmentProcessor {
    
    @StreamListener("orders")
    @SendTo("enriched-orders")
    public KStream<String, EnrichedOrder> enrichOrders(
            KStream<String, Order> orders,
            KTable<String, User> users,
            KTable<String, Product> products) {
        
        return orders
            .join(users, 
                (order, user) -> EnrichedOrder.builder()
                    .order(order)
                    .user(user)
                    .build(),
                Joined.with(Serdes.String(), orderSerde, userSerde))
            .transformValues(() -> new ValueTransformerWithKey<String, EnrichedOrder, EnrichedOrder>() {
                private KeyValueStore<String, Product> productStore;
                
                @Override
                public void init(ProcessorContext context) {
                    productStore = (KeyValueStore<String, Product>) 
                        context.getStateStore("products-store");
                }
                
                @Override
                public EnrichedOrder transform(String key, EnrichedOrder enrichedOrder) {
                    // Enrich with product information for each order item
                    List<EnrichedOrderItem> enrichedItems = enrichedOrder.getOrder()
                        .getOrderItems()
                        .stream()
                        .map(item -> {
                            Product product = productStore.get(String.valueOf(item.getProductId()));
                            return EnrichedOrderItem.builder()
                                .orderItem(item)
                                .product(product)
                                .build();
                        })
                        .collect(Collectors.toList());
                    
                    return enrichedOrder.toBuilder()
                        .enrichedItems(enrichedItems)
                        .build();
                }
            });
    }
}
```

### Phase 4: Data Quality & Error Handling (30 minutes)

#### 4.1 Schema Validation
```java
@Component
public class DataQualityProcessor {
    
    private final SchemaRegistry schemaRegistry;
    private final DeadLetterQueueProducer dlqProducer;
    
    @StreamListener("raw-events")
    @SendTo("validated-events")
    public KStream<String, ValidatedEvent> validateEvents(
            KStream<String, GenericRecord> rawEvents) {
        
        return rawEvents
            .mapValues(this::validateAndTransform)
            .filter((key, validatedEvent) -> validatedEvent != null);
    }
    
    private ValidatedEvent validateAndTransform(GenericRecord record) {
        try {
            // Schema validation
            Schema schema = schemaRegistry.getLatestSchema(record.getSchema().getName());
            if (!isCompatible(record.getSchema(), schema)) {
                dlqProducer.sendToDLQ(record, "Schema incompatibility");
                return null;
            }
            
            // Business rule validation
            List<ValidationError> errors = validateBusinessRules(record);
            if (!errors.isEmpty()) {
                dlqProducer.sendToDLQ(record, "Business rule violations: " + errors);
                return null;
            }
            
            return ValidatedEvent.builder()
                .originalRecord(record)
                .validationTimestamp(Instant.now())
                .validationStatus(ValidationStatus.VALID)
                .build();
                
        } catch (Exception e) {
            dlqProducer.sendToDLQ(record, "Validation error: " + e.getMessage());
            return null;
        }
    }
    
    private List<ValidationError> validateBusinessRules(GenericRecord record) {
        List<ValidationError> errors = new ArrayList<>();
        
        // Example validations
        if ("orders".equals(record.getSchema().getName())) {
            Object totalAmount = record.get("total_amount");
            if (totalAmount != null && ((BigDecimal) totalAmount).compareTo(BigDecimal.ZERO) <= 0) {
                errors.add(new ValidationError("total_amount", "Must be positive"));
            }
            
            Object status = record.get("status");
            if (status != null && !Arrays.asList("pending", "processing", "completed", "cancelled")
                    .contains(status.toString())) {
                errors.add(new ValidationError("status", "Invalid status value"));
            }
        }
        
        return errors;
    }
}
```

#### 4.2 Dead Letter Queue Handler
```java
@Component
public class DeadLetterQueueHandler {
    
    @StreamListener("dlq-topic")
    public void handleDeadLetterEvents(KStream<String, DLQEvent> dlqEvents) {
        dlqEvents
            .foreach((key, dlqEvent) -> {
                // Log the error
                log.error("DLQ Event: key={}, error={}, originalEvent={}", 
                    key, dlqEvent.getError(), dlqEvent.getOriginalEvent());
                
                // Update metrics
                meterRegistry.counter("dlq.events", 
                    "error_type", dlqEvent.getErrorType(),
                    "topic", dlqEvent.getOriginalTopic())
                    .increment();
                
                // Send alert if error rate is high
                if (shouldAlert(dlqEvent)) {
                    alertService.sendAlert(
                        AlertLevel.HIGH,
                        "High error rate in DLQ",
                        dlqEvent.toString()
                    );
                }
            });
    }
}
```

### Phase 5: Monitoring & Observability (30 minutes)

#### 5.1 Prometheus Metrics Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "kafka_rules.yml"
  - "application_rules.yml"

scrape_configs:
  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka1:9308', 'kafka2:9308', 'kafka3:9308']
  
  - job_name: 'kafka-connect'
    static_configs:
      - targets: ['kafka-connect:8083']
  
  - job_name: 'stream-processor'
    static_configs:
      - targets: ['stream-processor:8080']
  
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

#### 5.2 Grafana Dashboards
```json
{
  "dashboard": {
    "title": "CDC Pipeline Monitoring",
    "panels": [
      {
        "title": "Message Throughput",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(kafka_server_brokertopicmetrics_messagesinpersec_total[5m])",
            "legendFormat": "Messages/sec - {{topic}}"
          }
        ]
      },
      {
        "title": "Consumer Lag",
        "type": "graph",
        "targets": [
          {
            "expr": "kafka_consumer_lag_sum",
            "legendFormat": "Lag - {{consumergroup}} - {{topic}}"
          }
        ]
      },
      {
        "title": "Processing Latency",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(stream_processing_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Data Quality Metrics",
        "type": "stat",
        "targets": [
          {
            "expr": "rate(dlq_events_total[5m])",
            "legendFormat": "DLQ Events/sec"
          }
        ]
      }
    ]
  }
}
```

#### 5.3 Alerting Rules
```yaml
# kafka_rules.yml
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
      description: "Consumer group {{ $labels.consumergroup }} has lag of {{ $value }} on topic {{ $labels.topic }}"
  
  - alert: KafkaUnderReplicatedPartitions
    expr: kafka_server_replicamanager_underreplicatedpartitions > 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Kafka has under-replicated partitions"
      description: "{{ $value }} partitions are under-replicated"
  
  - alert: HighDLQRate
    expr: rate(dlq_events_total[5m]) > 10
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High dead letter queue rate"
      description: "DLQ rate is {{ $value }} events/sec"
  
  - alert: StreamProcessingLatency
    expr: histogram_quantile(0.95, rate(stream_processing_duration_seconds_bucket[5m])) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Stream processing latency is high"
      description: "95th percentile latency is {{ $value }}s"
```

### Phase 6: Testing & Validation (30 minutes)

#### 6.1 Data Generation Script
```python
#!/usr/bin/env python3
"""
Data generator for CDC pipeline testing
"""
import psycopg2
import random
import time
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

def generate_test_data():
    conn = psycopg2.connect(
        host="localhost",
        database="ecommerce",
        user="postgres",
        password="postgres"
    )
    
    cursor = conn.cursor()
    
    # Generate users
    for i in range(1000):
        cursor.execute("""
            INSERT INTO users (email, first_name, last_name, status)
            VALUES (%s, %s, %s, %s)
        """, (
            fake.email(),
            fake.first_name(),
            fake.last_name(),
            random.choice(['active', 'inactive'])
        ))
    
    # Generate products
    categories = ['Electronics', 'Clothing', 'Books', 'Home', 'Sports']
    for i in range(500):
        cursor.execute("""
            INSERT INTO products (name, category, price, stock_quantity)
            VALUES (%s, %s, %s, %s)
        """, (
            fake.catch_phrase(),
            random.choice(categories),
            round(random.uniform(10.0, 500.0), 2),
            random.randint(0, 1000)
        ))
    
    conn.commit()
    conn.close()

def simulate_order_activity():
    """Simulate realistic order activity"""
    conn = psycopg2.connect(
        host="localhost",
        database="ecommerce",
        user="postgres",
        password="postgres"
    )
    
    cursor = conn.cursor()
    
    while True:
        # Create random orders
        user_id = random.randint(1, 1000)
        total_amount = round(random.uniform(20.0, 500.0), 2)
        status = random.choice(['pending', 'processing', 'completed', 'cancelled'])
        
        cursor.execute("""
            INSERT INTO orders (user_id, total_amount, status)
            VALUES (%s, %s, %s) RETURNING order_id
        """, (user_id, total_amount, status))
        
        order_id = cursor.fetchone()[0]
        
        # Add order items
        num_items = random.randint(1, 5)
        for _ in range(num_items):
            product_id = random.randint(1, 500)
            quantity = random.randint(1, 3)
            unit_price = round(random.uniform(10.0, 100.0), 2)
            
            cursor.execute("""
                INSERT INTO order_items (order_id, product_id, quantity, unit_price)
                VALUES (%s, %s, %s, %s)
            """, (order_id, product_id, quantity, unit_price))
        
        # Occasionally update existing orders
        if random.random() < 0.3:
            cursor.execute("""
                UPDATE orders 
                SET status = %s, updated_at = CURRENT_TIMESTAMP
                WHERE order_id = %s
            """, (
                random.choice(['processing', 'completed', 'cancelled']),
                random.randint(max(1, order_id - 100), order_id)
            ))
        
        # Occasionally update product inventory
        if random.random() < 0.2:
            cursor.execute("""
                UPDATE products 
                SET stock_quantity = stock_quantity - %s,
                    updated_at = CURRENT_TIMESTAMP
                WHERE product_id = %s AND stock_quantity > %s
            """, (
                random.randint(1, 10),
                random.randint(1, 500),
                5
            ))
        
        conn.commit()
        time.sleep(random.uniform(0.1, 2.0))  # Random delay between operations

if __name__ == "__main__":
    print("Generating initial test data...")
    generate_test_data()
    print("Starting order simulation...")
    simulate_order_activity()
```

#### 6.2 Integration Tests
```python
import pytest
import kafka
import psycopg2
import time
from kafka import KafkaConsumer, KafkaProducer

class TestCDCPipeline:
    
    def setup_method(self):
        self.pg_conn = psycopg2.connect(
            host="localhost",
            database="ecommerce",
            user="postgres",
            password="postgres"
        )
        
        self.kafka_consumer = KafkaConsumer(
            'orders',
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest'
        )
    
    def test_order_insert_captured(self):
        """Test that order inserts are captured by CDC"""
        cursor = self.pg_conn.cursor()
        
        # Insert test order
        test_order = {
            'user_id': 999,
            'total_amount': 123.45,
            'status': 'pending'
        }
        
        cursor.execute("""
            INSERT INTO orders (user_id, total_amount, status)
            VALUES (%(user_id)s, %(total_amount)s, %(status)s)
            RETURNING order_id
        """, test_order)
        
        order_id = cursor.fetchone()[0]
        self.pg_conn.commit()
        
        # Wait for CDC event
        time.sleep(2)
        
        # Check Kafka topic
        messages = []
        for message in self.kafka_consumer:
            messages.append(message.value)
            if len(messages) >= 1:
                break
        
        assert len(messages) > 0
        event = messages[0]
        assert event['payload']['after']['order_id'] == order_id
        assert event['payload']['after']['user_id'] == 999
        assert float(event['payload']['after']['total_amount']) == 123.45
    
    def test_order_update_captured(self):
        """Test that order updates are captured by CDC"""
        cursor = self.pg_conn.cursor()
        
        # Create and then update order
        cursor.execute("""
            INSERT INTO orders (user_id, total_amount, status)
            VALUES (998, 100.00, 'pending')
            RETURNING order_id
        """)
        order_id = cursor.fetchone()[0]
        self.pg_conn.commit()
        
        time.sleep(1)
        
        # Update the order
        cursor.execute("""
            UPDATE orders 
            SET status = 'completed', updated_at = CURRENT_TIMESTAMP
            WHERE order_id = %s
        """, (order_id,))
        self.pg_conn.commit()
        
        # Wait and check for update event
        time.sleep(2)
        
        messages = []
        for message in self.kafka_consumer:
            if message.value['payload']['after']['order_id'] == order_id:
                messages.append(message.value)
                if message.value['payload']['after']['status'] == 'completed':
                    break
        
        assert len(messages) > 0
        update_event = messages[-1]
        assert update_event['payload']['op'] == 'u'  # Update operation
        assert update_event['payload']['after']['status'] == 'completed'
    
    def test_stream_processing_latency(self):
        """Test end-to-end processing latency"""
        start_time = time.time()
        
        cursor = self.pg_conn.cursor()
        cursor.execute("""
            INSERT INTO orders (user_id, total_amount, status)
            VALUES (997, 200.00, 'completed')
        """)
        self.pg_conn.commit()
        
        # Check analytics database for processed result
        analytics_conn = psycopg2.connect(
            host="localhost",
            database="analytics",
            user="postgres",
            password="postgres"
        )
        
        analytics_cursor = analytics_conn.cursor()
        
        # Poll for processed data
        for _ in range(30):  # Wait up to 30 seconds
            analytics_cursor.execute("""
                SELECT COUNT(*) FROM revenue_analytics 
                WHERE window_end > %s
            """, (datetime.now() - timedelta(minutes=1),))
            
            count = analytics_cursor.fetchone()[0]
            if count > 0:
                end_time = time.time()
                latency = end_time - start_time
                assert latency < 10.0  # Should process within 10 seconds
                break
            
            time.sleep(1)
        else:
            pytest.fail("Stream processing did not complete within timeout")
```

---

## 📊 Success Criteria & Validation

### ✅ Functional Validation

1. **CDC Capture Verification**
   - [ ] All table changes captured (INSERT, UPDATE, DELETE)
   - [ ] Event ordering maintained per partition
   - [ ] Schema evolution handled gracefully
   - [ ] Initial snapshot completed successfully

2. **Stream Processing Validation**
   - [ ] Revenue analytics updated within 5 seconds
   - [ ] User behavior tracking functional
   - [ ] Inventory alerts triggered correctly
   - [ ] Data enrichment working properly

3. **Data Quality Validation**
   - [ ] Schema validation catching invalid events
   - [ ] Business rules enforced correctly
   - [ ] Dead letter queue handling errors
   - [ ] Duplicate detection working

### ⚡ Performance Validation

1. **Throughput Testing**
   ```bash
   # Generate load and measure throughput
   python load_generator.py --rate 10000 --duration 300
   
   # Expected results:
   # - Kafka ingestion: >10,000 events/sec
   # - Stream processing: >8,000 events/sec
   # - End-to-end: >5,000 events/sec
   ```

2. **Latency Testing**
   ```bash
   # Measure end-to-end latency
   python latency_test.py --samples 1000
   
   # Expected results:
   # - 50th percentile: <50ms
   # - 95th percentile: <100ms
   # - 99th percentile: <200ms
   ```

3. **Scalability Testing**
   ```bash
   # Test horizontal scaling
   docker-compose scale stream-processor=3
   docker-compose scale kafka-consumer=6
   
   # Verify linear throughput scaling
   ```

### 🔍 Reliability Validation

1. **Fault Tolerance Testing**
   ```bash
   # Test broker failure
   docker-compose stop kafka2
   # Verify: No data loss, automatic failover
   
   # Test database connection loss
   docker-compose stop postgres
   # Verify: Graceful degradation, automatic recovery
   
   # Test stream processor failure
   docker-compose stop stream-processor
   # Verify: Processing resumes from last checkpoint
   ```

2. **Data Consistency Testing**
   ```bash
   # Compare source and processed data
   python data_consistency_check.py
   
   # Expected: 100% consistency within SLA timeframe
   ```

### 📈 Monitoring Validation

1. **Metrics Collection**
   - [ ] All components reporting metrics to Prometheus
   - [ ] Grafana dashboards displaying real-time data
   - [ ] Custom business metrics tracked

2. **Alerting Verification**
   - [ ] Consumer lag alerts triggered at threshold
   - [ ] Data quality alerts sent for DLQ events
   - [ ] System health alerts functional

---

## 🎯 Deliverables

### 📁 Code Deliverables

1. **Infrastructure as Code**
   - `docker-compose.yml` - Complete environment setup
   - `kafka/` - Kafka cluster configuration
   - `postgres/` - Database setup and sample data
   - `debezium/` - CDC connector configurations

2. **Stream Processing Applications**
   - `stream-processor/` - Java/Python stream processing apps
   - `data-quality/` - Validation and error handling
   - `analytics/` - Real-time analytics processors

3. **Monitoring & Observability**
   - `monitoring/prometheus/` - Metrics collection config
   - `monitoring/grafana/` - Dashboards and visualizations
   - `monitoring/alerting/` - Alert rules and notifications

### 📋 Documentation Deliverables

1. **Architecture Documentation**
   - System architecture diagrams
   - Data flow documentation
   - Component interaction specifications

2. **Operational Documentation**
   - Deployment guide
   - Configuration management
   - Troubleshooting runbook
   - Performance tuning guide

3. **Testing Documentation**
   - Test strategy and test cases
   - Performance benchmarks
   - Validation procedures

### 📊 Demonstration Deliverables

1. **Live Demo**
   - Real-time data pipeline demonstration
   - Monitoring dashboard walkthrough
   - Failure scenario and recovery

2. **Performance Report**
   - Throughput and latency measurements
   - Scalability test results
   - Resource utilization analysis

---

## 🚨 Common Challenges & Solutions

### Challenge 1: Schema Evolution
**Problem**: Database schema changes break downstream consumers
**Solution**: 
- Implement backward-compatible schema changes
- Use Schema Registry with compatibility rules
- Version control all schema changes
- Test schema evolution in staging environment

### Challenge 2: Exactly-Once Processing
**Problem**: Duplicate events in stream processing
**Solution**:
- Enable Kafka idempotent producers
- Use transactional processing in Kafka Streams
- Implement application-level deduplication
- Monitor duplicate rates with metrics

### Challenge 3: Backpressure Management
**Problem**: Downstream systems overwhelmed by event volume
**Solution**:
- Implement circuit breakers and bulkheads
- Use adaptive batching and buffering
- Scale consumers horizontally
- Implement graceful degradation

### Challenge 4: Data Consistency
**Problem**: Eventual consistency issues across systems
**Solution**:
- Use event sourcing patterns
- Implement saga patterns for distributed transactions
- Monitor data reconciliation
- Set appropriate consistency SLAs

### Challenge 5: Operational Complexity
**Problem**: Managing multiple distributed systems
**Solution**:
- Comprehensive monitoring and alerting
- Infrastructure as Code for reproducibility
- Automated deployment and rollback procedures
- Clear operational runbooks

---

## 🎓 Learning Outcomes

Upon completion of this project, you will have:

### Technical Skills
- **Distributed Systems**: Understanding of CDC, event streaming, and stream processing
- **Data Engineering**: Real-time data pipeline design and implementation
- **Monitoring**: Production-grade observability and alerting
- **Quality Assurance**: Data validation, testing, and reliability patterns

### Architectural Skills
- **System Design**: End-to-end data architecture planning
- **Scalability**: Horizontal scaling patterns and performance optimization
- **Reliability**: Fault tolerance, disaster recovery, and operational excellence
- **Integration**: Connecting multiple data systems seamlessly

### Operational Skills
- **DevOps**: Infrastructure as Code and automated deployments
- **Monitoring**: Metrics, logging, and alerting best practices
- **Troubleshooting**: Debugging distributed systems issues
- **Performance**: Capacity planning and optimization techniques

---

## 🚀 Next Steps & Extensions

### Immediate Extensions
1. **Add More Data Sources**: Integrate additional databases or APIs
2. **Advanced Analytics**: Implement machine learning on streaming data
3. **Data Lake Integration**: Stream data to S3/HDFS for long-term storage
4. **Multi-Region Deployment**: Implement cross-region replication

### Advanced Features
1. **Event Sourcing**: Implement full event sourcing patterns
2. **CQRS**: Separate command and query responsibilities
3. **Microservices**: Break down into smaller, focused services
4. **Kubernetes**: Deploy on Kubernetes for production scalability

### Production Readiness
1. **Security**: Implement authentication, authorization, and encryption
2. **Compliance**: Add GDPR, SOX, or other regulatory compliance
3. **Cost Optimization**: Implement resource optimization strategies
4. **Disaster Recovery**: Multi-region backup and recovery procedures

---

## 📚 Additional Resources

### Documentation & Guides
- [Debezium Tutorial](https://debezium.io/documentation/reference/tutorial.html)
- [Kafka Streams Developer Guide](https://kafka.apache.org/documentation/streams/developer-guide/)
- [Event-Driven Architecture Patterns](https://microservices.io/patterns/data/event-driven-architecture.html)

### Best Practices
- [CDC Best Practices](https://debezium.io/blog/2020/02/25/lessons-learned-running-debezium-with-postgresql-on-rds/)
- [Kafka Production Checklist](https://kafka.apache.org/documentation/#productionchecklist)
- [Stream Processing Patterns](https://www.confluent.io/blog/build-services-backbone-events/)

### Tools & Libraries
- [Kafka Manager](https://github.com/yahoo/CMAK) - Cluster management
- [Schema Registry UI](https://github.com/lensesio/schema-registry-ui) - Schema management
- [Kafka Connect UI](https://github.com/lensesio/kafka-connect-ui) - Connector management

---

## ✅ Project Completion Checklist

### Infrastructure ✅
- [ ] Docker Compose environment running
- [ ] PostgreSQL configured for logical replication
- [ ] Kafka cluster (3 brokers) operational
- [ ] Debezium Connect cluster deployed
- [ ] Schema Registry configured
- [ ] Monitoring stack (Prometheus/Grafana) running

### Data Pipeline ✅
- [ ] CDC connector capturing all table changes
- [ ] Kafka topics properly partitioned and replicated
- [ ] Stream processing applications deployed
- [ ] Real-time analytics functional
- [ ] Data quality validation implemented
- [ ] Error handling and DLQ operational

### Testing & Validation ✅
- [ ] Unit tests passing (>90% coverage)
- [ ] Integration tests successful
- [ ] Performance benchmarks met
- [ ] Fault tolerance tested
- [ ] Data consistency validated
- [ ] End-to-end testing completed

### Documentation ✅
- [ ] Architecture documentation complete
- [ ] Deployment guide written
- [ ] Operational runbook created
- [ ] Performance report generated
- [ ] Demo presentation prepared

### Demonstration ✅
- [ ] Live demo successful
- [ ] Monitoring dashboards functional
- [ ] Alerting system operational
- [ ] Performance metrics documented
- [ ] Project presentation delivered

**Estimated Completion Time**: 3-4 hours  
**Difficulty Level**: ⭐⭐⭐⭐⭐ (Expert)  
**Success Criteria**: All checklist items completed with documented evidence

---

🎉 **Congratulations!** You've built a production-ready real-time CDC pipeline that demonstrates mastery of modern data engineering practices. This project showcases your ability to integrate multiple technologies, handle real-time data at scale, and implement production-grade monitoring and reliability patterns.

**Ready for Day 8: Data Catalogs!** 🚀
