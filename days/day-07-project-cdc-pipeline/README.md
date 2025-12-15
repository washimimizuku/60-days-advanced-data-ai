# Day 7: Project - Real-time CDC Pipeline

## 🎯 Project Overview

Build a complete real-time Change Data Capture (CDC) pipeline that captures changes from PostgreSQL, streams them through Kafka, and processes them for analytics. This project integrates everything learned in Days 1-6.

**What you'll build**:
- PostgreSQL database with sample e-commerce data
- Debezium CDC connector capturing database changes
- Kafka cluster handling real-time event streaming
- Stream processing for real-time analytics
- Monitoring and alerting system
- Data quality validation

---

## 📖 Learning Objectives

By the end of this project, you will:
- Integrate PostgreSQL, Debezium, Kafka, and stream processing
- Build a production-ready CDC pipeline with monitoring
- Handle real-time data transformations and aggregations
- Implement data quality checks and error handling
- Deploy and monitor a complete streaming data system
- Apply best practices for scalable data architectures

---

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │    Debezium     │    │     Kafka       │
│                 │───▶│   Connector     │───▶│   (3 brokers)   │
│ • Users         │    │                 │    │                 │
│ • Orders        │    │ • CDC Capture   │    │ • Raw Events    │
│ • Products      │    │ • Schema Reg    │    │ • Processed     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐             │
│   Analytics     │    │ Stream Processor│◀────────────┘
│                 │◀───│                 │
│ • Real-time     │    │ • Kafka Streams │
│   Dashboards    │    │ • Aggregations  │
│ • Alerts        │    │ • Transformations│
└─────────────────┘    └─────────────────┘

┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │ Data Quality    │
│                 │    │                 │
│ • Prometheus    │    │ • Schema Valid  │
│ • Grafana       │    │ • Data Checks   │
│ • Alertmanager  │    │ • Error Handling│
└─────────────────┘    └─────────────────┘
```

---

## 🛠️ Technology Stack

**Data Sources**:
- PostgreSQL 15 (source database)
- Sample e-commerce dataset

**CDC & Streaming**:
- Debezium 2.4 (CDC connector)
- Apache Kafka 3.6 (event streaming)
- Kafka Connect (connector management)
- Schema Registry (schema evolution)

**Processing**:
- Kafka Streams (stream processing)
- Python/Java for custom processors

**Storage & Analytics**:
- PostgreSQL (analytics database)
- Redis (caching layer)

**Monitoring**:
- Prometheus (metrics collection)
- Grafana (visualization)
- Kafka Manager (cluster monitoring)

---

## 📋 Project Requirements

### Functional Requirements

1. **Real-time Data Capture**
   - Capture all changes (INSERT, UPDATE, DELETE) from PostgreSQL
   - Handle schema evolution gracefully
   - Maintain data consistency and ordering

2. **Stream Processing**
   - Real-time order analytics (revenue, top products)
   - User behavior tracking and segmentation
   - Inventory level monitoring with alerts

3. **Data Quality**
   - Schema validation for all events
   - Data completeness checks
   - Duplicate detection and handling

4. **Error Handling**
   - Dead letter queues for failed messages
   - Retry mechanisms with exponential backoff
   - Comprehensive logging and alerting

### Non-Functional Requirements

1. **Performance**
   - Handle 10,000+ events per second
   - End-to-end latency < 100ms (95th percentile)
   - Zero data loss guarantee

2. **Scalability**
   - Horizontally scalable stream processors
   - Auto-scaling based on load
   - Partition strategy for optimal throughput

3. **Reliability**
   - 99.9% uptime requirement
   - Automatic failover and recovery
   - Data replication across availability zones

4. **Monitoring**
   - Real-time metrics and dashboards
   - Proactive alerting on anomalies
   - End-to-end data lineage tracking

---

## 🚀 Implementation Guide

### Phase 1: Infrastructure Setup (30 minutes)

**Set up the foundation**:
1. Docker Compose environment with all services
2. PostgreSQL with sample e-commerce data
3. Kafka cluster (3 brokers) with proper configuration
4. Debezium Connect cluster
5. Schema Registry for schema management

**Key configurations**:
- Kafka: Replication factor 3, appropriate partitioning
- PostgreSQL: WAL level logical, proper replication slots
- Debezium: Incremental snapshots, schema evolution

### Phase 2: CDC Pipeline (45 minutes)

**Implement data capture**:
1. Configure Debezium PostgreSQL connector
2. Set up topic naming and partitioning strategy
3. Handle initial snapshot and ongoing changes
4. Implement schema registry integration

**Data flow**:
```
PostgreSQL Tables → Debezium → Kafka Topics
• users → ecommerce.public.users
• orders → ecommerce.public.orders  
• order_items → ecommerce.public.order_items
• products → ecommerce.public.products
```

### Phase 3: Stream Processing (45 minutes)

**Build real-time analytics**:
1. Order processing stream (revenue calculations)
2. User activity stream (behavior tracking)
3. Inventory monitoring (stock level alerts)
4. Data enrichment (joining streams and tables)

**Processing patterns**:
- Windowed aggregations (5-minute, 1-hour windows)
- Stream-table joins for data enrichment
- Stateful processing for user sessions
- Complex event processing for business rules

### Phase 4: Data Quality & Error Handling (30 minutes)

**Implement quality controls**:
1. Schema validation processors
2. Data completeness checks
3. Business rule validation
4. Dead letter queue handling

**Quality checks**:
- Required field validation
- Data type consistency
- Business logic validation (e.g., positive prices)
- Referential integrity checks

### Phase 5: Monitoring & Alerting (30 minutes)

**Set up observability**:
1. Prometheus metrics collection
2. Grafana dashboards for visualization
3. Alerting rules for critical issues
4. Log aggregation and analysis

**Key metrics**:
- Message throughput and latency
- Consumer lag and processing rates
- Error rates and data quality scores
- System resource utilization

---

## 📊 Sample Data Model

### Source Tables (PostgreSQL)

```sql
-- Users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active'
);

-- Products table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100),
    price DECIMAL(10,2) NOT NULL,
    stock_quantity INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    total_amount DECIMAL(10,2) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Order items table
CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id),
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL,
    unit_price DECIMAL(10,2) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### Stream Processing Outputs

```sql
-- Real-time revenue analytics
CREATE TABLE revenue_analytics (
    window_start TIMESTAMP,
    window_end TIMESTAMP,
    total_revenue DECIMAL(15,2),
    order_count INTEGER,
    avg_order_value DECIMAL(10,2),
    top_category VARCHAR(100),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User activity summary
CREATE TABLE user_activity (
    user_id INTEGER,
    session_start TIMESTAMP,
    session_end TIMESTAMP,
    page_views INTEGER,
    orders_placed INTEGER,
    total_spent DECIMAL(10,2),
    last_activity TIMESTAMP
);

-- Inventory alerts
CREATE TABLE inventory_alerts (
    product_id INTEGER,
    product_name VARCHAR(255),
    current_stock INTEGER,
    threshold INTEGER,
    alert_level VARCHAR(20), -- 'low', 'critical', 'out_of_stock'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 🔧 Key Implementation Details

### Debezium Configuration

```json
{
  "name": "ecommerce-postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "debezium",
    "database.dbname": "ecommerce",
    "database.server.name": "ecommerce",
    "table.include.list": "public.users,public.orders,public.order_items,public.products",
    "plugin.name": "pgoutput",
    "slot.name": "debezium_slot",
    "publication.name": "debezium_publication",
    "transforms": "route",
    "transforms.route.type": "org.apache.kafka.connect.transforms.RegexRouter",
    "transforms.route.regex": "([^.]+)\\.([^.]+)\\.([^.]+)",
    "transforms.route.replacement": "$3",
    "key.converter": "io.confluent.connect.avro.AvroConverter",
    "value.converter": "io.confluent.connect.avro.AvroConverter",
    "key.converter.schema.registry.url": "http://schema-registry:8081",
    "value.converter.schema.registry.url": "http://schema-registry:8081"
  }
}
```

### Kafka Streams Processing

```java
// Revenue analytics stream
KStream<String, Order> orders = builder.stream("orders");

KTable<Windowed<String>, Double> revenueByWindow = orders
    .filter((key, order) -> "completed".equals(order.getStatus()))
    .groupBy((key, order) -> "revenue")
    .windowedBy(TimeWindows.of(Duration.ofMinutes(5)))
    .aggregate(
        () -> 0.0,
        (key, order, aggregate) -> aggregate + order.getTotalAmount(),
        Materialized.with(Serdes.String(), Serdes.Double())
    );

// User session tracking
KStream<String, UserEvent> userEvents = builder.stream("user_events");

KTable<String, UserSession> userSessions = userEvents
    .groupByKey()
    .aggregate(
        UserSession::new,
        (key, event, session) -> session.addEvent(event),
        Materialized.with(Serdes.String(), userSessionSerde)
    );
```

### Data Quality Validation

```python
def validate_order_event(event):
    """Validate order event data quality"""
    errors = []
    
    # Required fields
    required_fields = ['order_id', 'user_id', 'total_amount', 'status']
    for field in required_fields:
        if field not in event or event[field] is None:
            errors.append(f"Missing required field: {field}")
    
    # Business rules
    if 'total_amount' in event:
        if event['total_amount'] <= 0:
            errors.append("Total amount must be positive")
    
    if 'status' in event:
        valid_statuses = ['pending', 'processing', 'completed', 'cancelled']
        if event['status'] not in valid_statuses:
            errors.append(f"Invalid status: {event['status']}")
    
    return errors

def process_with_quality_check(event):
    """Process event with quality validation"""
    errors = validate_order_event(event)
    
    if errors:
        # Send to dead letter queue
        send_to_dlq(event, errors)
        return None
    
    # Process valid event
    return transform_event(event)
```

---

## 📈 Success Metrics

### Performance Metrics
- **Throughput**: > 10,000 events/second
- **Latency**: < 100ms end-to-end (95th percentile)
- **Availability**: > 99.9% uptime
- **Data Loss**: Zero tolerance

### Data Quality Metrics
- **Schema Compliance**: > 99.9%
- **Data Completeness**: > 99.5%
- **Duplicate Rate**: < 0.1%
- **Processing Accuracy**: > 99.9%

### Business Metrics
- **Real-time Insights**: Revenue updates within 5 seconds
- **Alert Response**: Inventory alerts within 30 seconds
- **Data Freshness**: Analytics data < 1 minute old
- **System Recovery**: < 5 minutes RTO/RPO

---

## 🚨 Common Challenges & Solutions

### Challenge 1: Schema Evolution
**Problem**: Database schema changes break downstream consumers
**Solution**: 
- Use Schema Registry with compatibility rules
- Implement backward/forward compatible schemas
- Version control for schema changes

### Challenge 2: Exactly-Once Processing
**Problem**: Duplicate events in stream processing
**Solution**:
- Enable Kafka idempotent producers
- Use transactional processing in Kafka Streams
- Implement deduplication logic

### Challenge 3: Backpressure Handling
**Problem**: Downstream systems can't keep up with event volume
**Solution**:
- Implement circuit breakers
- Use adaptive batching
- Scale consumers horizontally

### Challenge 4: Data Consistency
**Problem**: Eventual consistency issues across systems
**Solution**:
- Use event sourcing patterns
- Implement saga patterns for distributed transactions
- Monitor data reconciliation

---

## 🔍 Testing Strategy

### Unit Tests
- Individual component testing
- Mock external dependencies
- Data transformation logic validation

### Integration Tests
- End-to-end pipeline testing
- Schema compatibility testing
- Error handling scenarios

### Performance Tests
- Load testing with realistic data volumes
- Latency benchmarking
- Scalability testing

### Chaos Engineering
- Network partition simulation
- Service failure scenarios
- Data corruption handling

---

## 📚 Resources

### Documentation
- [Debezium Documentation](https://debezium.io/documentation/)
- [Kafka Streams Developer Guide](https://kafka.apache.org/documentation/streams/)
- [PostgreSQL Logical Replication](https://www.postgresql.org/docs/current/logical-replication.html)

### Best Practices
- [CDC Best Practices](https://debezium.io/blog/2020/02/25/lessons-learned-running-debezium-with-postgresql-on-rds/)
- [Kafka Streams Patterns](https://kafka.apache.org/documentation/streams/developer-guide/)
- [Stream Processing Patterns](https://www.confluent.io/blog/build-services-backbone-events/)

### Monitoring & Operations
- [Kafka Monitoring](https://kafka.apache.org/documentation/#monitoring)
- [Debezium Monitoring](https://debezium.io/documentation/reference/operations/monitoring.html)
- [Stream Processing Observability](https://www.confluent.io/blog/monitoring-kafka-streams/)

---

## 🎯 Key Takeaways

- **CDC enables real-time data integration** without impacting source systems
- **Kafka provides reliable, scalable event streaming** for high-volume data
- **Stream processing enables real-time analytics** and business insights
- **Data quality validation is crucial** for production systems
- **Monitoring and alerting are essential** for operational excellence
- **Schema evolution must be planned** for long-term maintainability
- **Error handling and recovery** are critical for system reliability
- **Performance testing validates** system capacity and scalability

---

## 🚀 What's Next?

Tomorrow (Day 8), you'll learn about **Data Catalogs** - understanding how to organize, discover, and govern data assets across your organization using tools like DataHub and Amundsen.

**Preview**: Data catalogs provide:
- Centralized metadata management
- Data discovery and search capabilities
- Data lineage and impact analysis
- Governance and compliance tracking

This CDC pipeline you built today will serve as a foundation for understanding how data flows through systems and why cataloging and governance become critical at scale.

---

## ✅ Project Completion Checklist

- [ ] Infrastructure deployed and running
- [ ] PostgreSQL CDC configured with Debezium
- [ ] Kafka cluster handling events reliably
- [ ] Stream processing generating real-time analytics
- [ ] Data quality validation implemented
- [ ] Monitoring and alerting operational
- [ ] Performance requirements met
- [ ] Error handling tested
- [ ] Documentation completed
- [ ] Project demo prepared

**Time**: ~3 hours | **Difficulty**: ⭐⭐⭐⭐⭐ (Expert)

## 📋 Additional Resources

### Security
- [SECURITY.md](./SECURITY.md) - Comprehensive security guidelines
- [.env.example](./.env.example) - Environment variable template

### Operations
- [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) - Common issues and solutions
- [tests/](./tests/) - Integration test suite

### Development
- [requirements.txt](./requirements.txt) - Python dependencies
- [stream-processor/pom.xml](./stream-processor/pom.xml) - Java dependencies

Ready to build production-grade streaming systems! 🚀
