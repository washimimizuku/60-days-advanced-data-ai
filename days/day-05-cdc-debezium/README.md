# Day 5: Change Data Capture (CDC) - Debezium

## 📖 Learning Objectives

**Estimated Time**: 60 minutes

By the end of today, you will:
- Understand Change Data Capture (CDC) concepts and use cases
- Master Debezium for real-time data streaming
- Implement CDC pipelines from databases to data warehouses
- Design event-driven architectures with CDC
- Apply CDC patterns for microservices and analytics

---

## Theory

### What is Change Data Capture (CDC)?

Change Data Capture is a design pattern that identifies and captures changes made to data in a database, then delivers those changes in real-time to downstream systems. CDC enables real-time data integration, event-driven architectures, and keeps multiple systems synchronized.

**Key benefits**:
- **Real-time data synchronization**: Keep systems in sync with minimal latency
- **Event-driven architecture**: React to data changes as they happen
- **Audit trails**: Track all changes for compliance and debugging
- **Zero-impact extraction**: Capture changes without affecting source systems
- **Scalable data integration**: Handle high-volume change streams efficiently

**Common use cases**:
- Real-time analytics and reporting
- Data warehouse synchronization
- Microservices data synchronization
- Cache invalidation
- Search index updates
- Audit logging and compliance

### CDC Approaches

#### 1. Log-Based CDC (Recommended)

Reads database transaction logs to capture changes.

**Advantages**:
- Low impact on source database
- Captures all changes (including deletes)
- Preserves transaction order
- No schema modifications required

**Disadvantages**:
- Database-specific implementation
- Requires log access permissions
- Log retention policies affect history

#### 2. Trigger-Based CDC

Uses database triggers to capture changes.

```sql
-- Example trigger for PostgreSQL
CREATE OR REPLACE FUNCTION audit_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO change_log (table_name, operation, new_data, timestamp)
        VALUES (TG_TABLE_NAME, 'INSERT', row_to_json(NEW), NOW());
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO change_log (table_name, operation, old_data, new_data, timestamp)
        VALUES (TG_TABLE_NAME, 'UPDATE', row_to_json(OLD), row_to_json(NEW), NOW());
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO change_log (table_name, operation, old_data, timestamp)
        VALUES (TG_TABLE_NAME, 'DELETE', row_to_json(OLD), NOW());
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER users_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON users
    FOR EACH ROW EXECUTE FUNCTION audit_changes();
```

**Advantages**:
- Database-agnostic approach
- Easy to implement and understand
- Can add custom logic

**Disadvantages**:
- Performance impact on source database
- Requires schema modifications
- Can miss changes if triggers fail

#### 3. Timestamp-Based CDC

Uses timestamp columns to identify changed records.

```sql
-- Poll for changes every minute
SELECT * FROM orders 
WHERE updated_at > '2024-01-15 10:00:00'
  AND updated_at <= '2024-01-15 10:01:00';
```

**Advantages**:
- Simple to implement
- Works with any database
- No special permissions required

**Disadvantages**:
- Cannot capture deletes
- Requires timestamp columns
- Polling introduces latency
- Can miss rapid changes

### What is Debezium?

Debezium is an open-source platform for Change Data Capture built on Apache Kafka Connect. It monitors databases and captures row-level changes, streaming them to Kafka topics in real-time.

**Key features**:
- **Multiple database support**: PostgreSQL, MySQL, MongoDB, SQL Server, Oracle, Cassandra
- **Kafka integration**: Built on Kafka Connect framework
- **Schema evolution**: Handles schema changes gracefully
- **Exactly-once delivery**: Ensures data consistency
- **Fault tolerance**: Automatic recovery and resumption
- **Monitoring**: Built-in metrics and health checks

### Debezium Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Source DB     │    │   Debezium      │    │   Kafka         │
│   (PostgreSQL)  │───▶│   Connector     │───▶│   Topics        │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                       │
                                                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Consumers     │    │   Kafka         │    │   Data          │
│   (Analytics)   │◀───│   Connect       │◀───│   Warehouse     │
│                 │    │   Sink          │    │   (Snowflake)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Setting Up Debezium

#### 1. Prerequisites

```bash
# Start Kafka and Zookeeper
docker-compose up -d zookeeper kafka

# Start Kafka Connect with Debezium
docker-compose up -d connect

# Verify services
docker-compose ps
```

#### 2. Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'
services:
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  kafka:
    image: confluentinc/cp-kafka:7.4.0
    depends_on:
      - zookeeper
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  connect:
    image: debezium/connect:2.4
    depends_on:
      - kafka
    ports:
      - "8083:8083"
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      GROUP_ID: 1
      CONFIG_STORAGE_TOPIC: connect_configs
      OFFSET_STORAGE_TOPIC: connect_offsets
      STATUS_STORAGE_TOPIC: connect_status

  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      POSTGRES_DB: ecommerce
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
    command:
      - "postgres"
      - "-c"
      - "wal_level=logical"
      - "-c"
      - "max_replication_slots=4"
      - "-c"
      - "max_wal_senders=4"
```

#### 3. PostgreSQL Configuration

```sql
-- Enable logical replication
ALTER SYSTEM SET wal_level = logical;
ALTER SYSTEM SET max_replication_slots = 4;
ALTER SYSTEM SET max_wal_senders = 4;

-- Restart PostgreSQL to apply changes
-- sudo systemctl restart postgresql

-- Create replication user
CREATE USER debezium WITH REPLICATION LOGIN PASSWORD 'debezium';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO debezium;
GRANT USAGE ON SCHEMA public TO debezium;

-- Create publication for tables to monitor
CREATE PUBLICATION dbz_publication FOR ALL TABLES;
```

### Configuring Debezium Connectors

#### 1. PostgreSQL Connector

```json
{
  "name": "postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "debezium",
    "database.dbname": "ecommerce",
    "database.server.name": "ecommerce-db",
    "table.include.list": "public.customers,public.orders,public.products",
    "plugin.name": "pgoutput",
    "publication.name": "dbz_publication",
    "slot.name": "debezium_slot",
    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "key.converter.schemas.enable": false,
    "value.converter.schemas.enable": false,
    "transforms": "unwrap",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": false,
    "transforms.unwrap.delete.handling.mode": "rewrite"
  }
}
```

#### 2. Deploy Connector

```bash
# Deploy connector configuration
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d @postgres-connector.json

# Check connector status
curl http://localhost:8083/connectors/postgres-connector/status

# List all connectors
curl http://localhost:8083/connectors
```

### Understanding CDC Events

#### 1. Event Structure

```json
{
  "before": null,
  "after": {
    "id": 1001,
    "customer_id": 123,
    "total_amount": 299.99,
    "status": "pending",
    "created_at": 1705123456789
  },
  "source": {
    "version": "2.4.0.Final",
    "connector": "postgresql",
    "name": "ecommerce-db",
    "ts_ms": 1705123456789,
    "snapshot": "false",
    "db": "ecommerce",
    "sequence": "[\"24023928\",\"24023928\"]",
    "schema": "public",
    "table": "orders",
    "txId": 747,
    "lsn": 24023928,
    "xmin": null
  },
  "op": "c",
  "ts_ms": 1705123456790,
  "transaction": null
}
```

**Event fields**:
- **before**: Row state before change (null for INSERT)
- **after**: Row state after change (null for DELETE)
- **source**: Metadata about the change
- **op**: Operation type (c=create, u=update, d=delete, r=read)
- **ts_ms**: Timestamp when change was processed

#### 2. Operation Types

```json
// INSERT event
{
  "op": "c",
  "before": null,
  "after": { "id": 1, "name": "John", "email": "john@example.com" }
}

// UPDATE event
{
  "op": "u",
  "before": { "id": 1, "name": "John", "email": "john@old.com" },
  "after": { "id": 1, "name": "John", "email": "john@new.com" }
}

// DELETE event
{
  "op": "d",
  "before": { "id": 1, "name": "John", "email": "john@example.com" },
  "after": null
}
```

### Processing CDC Events

#### 1. Python Consumer

```python
from kafka import KafkaConsumer
import json
import logging

class CDCProcessor:
    def __init__(self, bootstrap_servers=['localhost:9092']):
        self.consumer = KafkaConsumer(
            'ecommerce-db.public.orders',
            bootstrap_servers=bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            group_id='cdc-processor',
            auto_offset_reset='earliest'
        )
        
    def process_events(self):
        for message in self.consumer:
            event = message.value
            self.handle_event(event)
    
    def handle_event(self, event):
        operation = event.get('op')
        table = event.get('source', {}).get('table')
        
        if operation == 'c':  # INSERT
            self.handle_insert(table, event['after'])
        elif operation == 'u':  # UPDATE
            self.handle_update(table, event['before'], event['after'])
        elif operation == 'd':  # DELETE
            self.handle_delete(table, event['before'])
        elif operation == 'r':  # READ (snapshot)
            self.handle_snapshot(table, event['after'])
    
    def handle_insert(self, table, record):
        print(f"INSERT into {table}: {record}")
        # Update cache, search index, analytics, etc.
        
    def handle_update(self, table, old_record, new_record):
        print(f"UPDATE {table}: {old_record} -> {new_record}")
        # Invalidate cache, update indexes, trigger workflows
        
    def handle_delete(self, table, record):
        print(f"DELETE from {table}: {record}")
        # Remove from cache, cleanup related data

# Usage
processor = CDCProcessor()
processor.process_events()
```

#### 2. Real-Time Analytics

```python
import pandas as pd
from datetime import datetime
import redis

class RealTimeAnalytics:
    def __init__(self):
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
    def process_order_event(self, event):
        operation = event.get('op')
        
        if operation in ['c', 'u']:  # INSERT or UPDATE
            order = event['after']
            self.update_metrics(order)
            
        elif operation == 'd':  # DELETE
            order = event['before']
            self.remove_metrics(order)
    
    def update_metrics(self, order):
        order_date = datetime.fromtimestamp(order['created_at'] / 1000).date()
        amount = float(order['total_amount'])
        
        # Update daily revenue
        daily_key = f"revenue:daily:{order_date}"
        self.redis_client.incrbyfloat(daily_key, amount)
        
        # Update order count
        count_key = f"orders:daily:{order_date}"
        self.redis_client.incr(count_key)
        
        # Update customer metrics
        customer_key = f"customer:{order['customer_id']}:total"
        self.redis_client.incrbyfloat(customer_key, amount)
        
        # Update real-time dashboard
        self.redis_client.publish('dashboard_updates', json.dumps({
            'type': 'order_update',
            'order_id': order['id'],
            'amount': amount,
            'timestamp': datetime.now().isoformat()
        }))
```

### CDC to Data Warehouse

#### 1. Snowflake Integration

```python
import snowflake.connector
from datetime import datetime

class SnowflakeCDCLoader:
    def __init__(self, connection_params):
        self.conn = snowflake.connector.connect(**connection_params)
        
    def process_cdc_event(self, event):
        operation = event.get('op')
        table = event.get('source', {}).get('table')
        
        if operation == 'c':  # INSERT
            self.upsert_record(table, event['after'])
        elif operation == 'u':  # UPDATE
            self.upsert_record(table, event['after'])
        elif operation == 'd':  # DELETE
            self.soft_delete_record(table, event['before'])
    
    def upsert_record(self, table, record):
        # Convert CDC record to Snowflake format
        columns = list(record.keys())
        values = list(record.values())
        
        # Add CDC metadata
        record['_cdc_timestamp'] = datetime.now()
        record['_cdc_operation'] = 'upsert'
        
        # Use MERGE for upsert
        merge_sql = f"""
        MERGE INTO {table}_cdc AS target
        USING (SELECT {', '.join([f"'{v}' as {k}" for k, v in record.items()])}) AS source
        ON target.id = source.id
        WHEN MATCHED THEN UPDATE SET {', '.join([f"{k} = source.{k}" for k in columns])}
        WHEN NOT MATCHED THEN INSERT ({', '.join(record.keys())}) 
                             VALUES ({', '.join([f"source.{k}" for k in record.keys()])})
        """
        
        cursor = self.conn.cursor()
        cursor.execute(merge_sql)
        cursor.close()
    
    def soft_delete_record(self, table, record):
        # Mark record as deleted instead of hard delete
        update_sql = f"""
        UPDATE {table}_cdc 
        SET _cdc_deleted = TRUE, 
            _cdc_timestamp = CURRENT_TIMESTAMP(),
            _cdc_operation = 'delete'
        WHERE id = {record['id']}
        """
        
        cursor = self.conn.cursor()
        cursor.execute(update_sql)
        cursor.close()
```

#### 2. Kafka Connect Sink

```json
{
  "name": "snowflake-sink-connector",
  "config": {
    "connector.class": "com.snowflake.kafka.connector.SnowflakeSinkConnector",
    "topics": "ecommerce-db.public.orders,ecommerce-db.public.customers",
    "snowflake.url.name": "https://account.snowflakecomputing.com",
    "snowflake.user.name": "kafka_user",
    "snowflake.private.key": "<your-snowflake-private-key>",
    "snowflake.database.name": "ECOMMERCE_CDC",
    "snowflake.schema.name": "RAW",
    "key.converter": "org.apache.kafka.connect.json.JsonConverter",
    "value.converter": "org.apache.kafka.connect.json.JsonConverter",
    "buffer.count.records": 10000,
    "buffer.flush.time": 60,
    "buffer.size.bytes": 5000000
  }
}
```

### Advanced CDC Patterns

#### 1. Event Sourcing

```python
class EventStore:
    def __init__(self, kafka_producer):
        self.producer = kafka_producer
    
    def handle_cdc_event(self, event):
        # Transform CDC event to domain event
        domain_event = self.transform_to_domain_event(event)
        
        # Store in event stream
        self.producer.send(
            topic=f"events.{domain_event['aggregate_type']}",
            key=domain_event['aggregate_id'],
            value=domain_event
        )
    
    def transform_to_domain_event(self, cdc_event):
        operation = cdc_event.get('op')
        table = cdc_event.get('source', {}).get('table')
        
        if table == 'orders' and operation == 'c':
            return {
                'event_type': 'OrderCreated',
                'aggregate_type': 'Order',
                'aggregate_id': cdc_event['after']['id'],
                'event_data': cdc_event['after'],
                'timestamp': cdc_event['ts_ms']
            }
        elif table == 'orders' and operation == 'u':
            return {
                'event_type': 'OrderUpdated',
                'aggregate_type': 'Order',
                'aggregate_id': cdc_event['after']['id'],
                'event_data': {
                    'before': cdc_event['before'],
                    'after': cdc_event['after']
                },
                'timestamp': cdc_event['ts_ms']
            }
```

#### 2. CQRS (Command Query Responsibility Segregation)

```python
class ReadModelUpdater:
    def __init__(self, read_db_connection):
        self.read_db = read_db_connection
    
    def handle_order_event(self, event):
        operation = event.get('op')
        
        if operation in ['c', 'u']:
            self.update_order_summary(event['after'])
            self.update_customer_summary(event['after'])
        elif operation == 'd':
            self.remove_order_summary(event['before'])
    
    def update_order_summary(self, order):
        # Update denormalized read model
        summary = {
            'order_id': order['id'],
            'customer_id': order['customer_id'],
            'total_amount': order['total_amount'],
            'status': order['status'],
            'order_date': order['created_at']
        }
        
        # Enrich with customer data
        customer = self.get_customer(order['customer_id'])
        summary.update({
            'customer_name': customer['name'],
            'customer_segment': customer['segment']
        })
        
        # Store in read-optimized format
        self.read_db.upsert('order_summary', summary)
    
    def update_customer_summary(self, order):
        # Update customer aggregates
        customer_id = order['customer_id']
        
        # Recalculate customer metrics
        metrics = self.calculate_customer_metrics(customer_id)
        self.read_db.upsert('customer_summary', metrics)
```

### Monitoring and Troubleshooting

#### 1. Connector Health Monitoring

```python
import requests
import json

class DebeziumMonitor:
    def __init__(self, connect_url='http://localhost:8083'):
        self.connect_url = connect_url
    
    def check_connector_health(self, connector_name):
        response = requests.get(f"{self.connect_url}/connectors/{connector_name}/status")
        status = response.json()
        
        return {
            'connector_state': status['connector']['state'],
            'tasks': [task['state'] for task in status['tasks']],
            'healthy': all(task['state'] == 'RUNNING' for task in status['tasks'])
        }
    
    def get_connector_metrics(self, connector_name):
        # Get JMX metrics (requires JMX setup)
        metrics = {
            'total_records_processed': self.get_jmx_metric('TotalNumberOfEventsSeen'),
            'snapshot_completed': self.get_jmx_metric('SnapshotCompleted'),
            'last_event_timestamp': self.get_jmx_metric('MilliSecondsSinceLastEvent')
        }
        return metrics
    
    def restart_connector(self, connector_name):
        response = requests.post(f"{self.connect_url}/connectors/{connector_name}/restart")
        return response.status_code == 204
```

#### 2. Lag Monitoring

```python
from kafka import KafkaConsumer
from kafka.admin import KafkaAdminClient, ConfigResource, ConfigResourceType

class CDCLagMonitor:
    def __init__(self, bootstrap_servers):
        self.admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    
    def check_consumer_lag(self, group_id, topic):
        # Get consumer group offsets
        group_offsets = self.admin_client.list_consumer_group_offsets(group_id)
        
        # Get topic partition high water marks
        consumer = KafkaConsumer(bootstrap_servers=self.admin_client._bootstrap_servers)
        partitions = consumer.partitions_for_topic(topic)
        
        lag_info = {}
        for partition in partitions:
            tp = TopicPartition(topic, partition)
            high_water_mark = consumer.end_offsets([tp])[tp]
            current_offset = group_offsets[group_id].get(tp, 0)
            
            lag_info[partition] = {
                'current_offset': current_offset,
                'high_water_mark': high_water_mark,
                'lag': high_water_mark - current_offset
            }
        
        return lag_info
```

### Best Practices

#### 1. Schema Evolution

```json
{
  "name": "postgres-connector",
  "config": {
    "schema.evolution": "true",
    "schema.refresh.mode": "columns_diff",
    "include.schema.changes": "true",
    "provide.transaction.metadata": "true"
  }
}
```

#### 2. Error Handling

```python
class RobustCDCProcessor:
    def __init__(self):
        self.dead_letter_queue = []
        self.retry_count = {}
        self.max_retries = 3
    
    def process_event_safely(self, event):
        event_id = self.get_event_id(event)
        
        try:
            self.process_event(event)
            # Reset retry count on success
            self.retry_count.pop(event_id, None)
            
        except Exception as e:
            retry_count = self.retry_count.get(event_id, 0)
            
            if retry_count < self.max_retries:
                self.retry_count[event_id] = retry_count + 1
                self.schedule_retry(event, retry_count + 1)
            else:
                # Send to dead letter queue
                self.dead_letter_queue.append({
                    'event': event,
                    'error': str(e),
                    'timestamp': datetime.now()
                })
                logging.error(f"Event {event_id} failed after {self.max_retries} retries: {e}")
```

#### 3. Performance Optimization

```json
{
  "name": "postgres-connector",
  "config": {
    "max.batch.size": "2048",
    "max.queue.size": "8192",
    "poll.interval.ms": "1000",
    "snapshot.fetch.size": "10240",
    "incremental.snapshot.chunk.size": "1024"
  }
}
```

---

## 💻 Hands-On Exercise

Build a complete CDC pipeline from PostgreSQL to Snowflake.

**What you'll create**:
1. PostgreSQL database with sample e-commerce data
2. Debezium connector for real-time change capture
3. Kafka topics for change events
4. Python consumers for event processing
5. Real-time analytics dashboard
6. Data warehouse synchronization

**Skills practiced**:
- Setting up Debezium connectors
- Processing CDC events
- Real-time data streaming
- Event-driven architecture patterns
- Data warehouse integration

See `exercise.py` for hands-on practice and `exercise/` directory for complete infrastructure setup.

> **🔒 Security Note**: This lesson uses placeholder credentials (`<username>`, `<password>`) in configuration examples. In production, use proper credential management with environment variables, secrets management systems, or IAM roles. Never commit real credentials to version control.

---

## 📚 Resources

- [Debezium Documentation](https://debezium.io/documentation/)
- [Kafka Connect Guide](https://kafka.apache.org/documentation/#connect)
- [CDC Patterns](https://microservices.io/patterns/data/change-data-capture.html)
- [Event Sourcing](https://martinfowler.com/eaaDev/EventSourcing.html)
- [CQRS Pattern](https://docs.microsoft.com/en-us/azure/architecture/patterns/cqrs)

---

## 🎯 Key Takeaways

- **CDC enables real-time data integration** without impacting source systems
- **Log-based CDC is preferred** for production systems (low impact, complete capture)
- **Debezium provides robust CDC** with exactly-once delivery and fault tolerance
- **Event structure includes before/after states** plus rich metadata
- **CDC enables event-driven architectures** and real-time analytics
- **Proper monitoring is essential** for production CDC pipelines
- **Schema evolution support** is crucial for long-running systems
- **Error handling and retry logic** ensure data consistency
- **Performance tuning** balances throughput and latency
- **CDC bridges operational and analytical systems** effectively

---

## 🚀 What's Next?

Tomorrow (Day 6), you'll learn **Advanced Kafka** - partitions, replication, and production patterns for building scalable streaming platforms.

**Preview**: Kafka is the backbone of modern data streaming architectures. You'll master advanced Kafka concepts like partitioning strategies, replication, consumer groups, and production deployment patterns that enable CDC pipelines to scale.

---

## ✅ Before Moving On

- [ ] Understand CDC concepts and approaches
- [ ] Can set up Debezium connectors for PostgreSQL
- [ ] Know how to process CDC events in real-time
- [ ] Can integrate CDC with data warehouses
- [ ] Understand event-driven architecture patterns
- [ ] Can monitor and troubleshoot CDC pipelines
- [ ] Complete the exercise in `exercise.py`
- [ ] Set up the infrastructure using `exercise/docker-compose.yml`
- [ ] Review the solution in `solution.py`
- [ ] Follow the setup guide in `SETUP.md`
- [ ] Take the quiz in `quiz.md`

**Time**: ~1 hour | **Difficulty**: ⭐⭐⭐⭐ (Advanced)

Ready to build real-time data pipelines! 🚀