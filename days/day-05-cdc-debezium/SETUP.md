# Day 5: CDC with Debezium - Setup Guide

## 🚀 Quick Start (10 minutes)

### Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ with pip
- 8GB+ RAM available
- Ports 5432, 6379, 8080, 8081, 8083, 9092 available

### 1. Start Infrastructure

```bash
# Navigate to exercise directory
cd days/day-05-cdc-debezium/exercise/

# Start all services
docker-compose up -d

# Verify services are running
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

### 3. Verify Setup

```bash
# Check PostgreSQL
docker exec -it postgres psql -U postgres -d ecommerce -c "SELECT version();"

# Check Kafka
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --list

# Check Kafka Connect
curl http://localhost:8083/connectors

# Check Redis
docker exec -it redis redis-cli ping
```

### 4. Initialize Database

```bash
# Database tables are created automatically via init scripts
# Verify tables exist
docker exec -it postgres psql -U postgres -d ecommerce -c "\dt"
```

### 5. Deploy Debezium Connector

```bash
# Run the solution to deploy connector
python solution.py

# Or deploy manually
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d @connectors/postgres-connector.json
```

## 🔧 Detailed Setup

### PostgreSQL Configuration

The PostgreSQL container is pre-configured with:
- Logical replication enabled (`wal_level=logical`)
- Replication slots configured
- Debezium user with proper permissions
- Sample e-commerce schema

### Kafka Configuration

- Single broker setup for development
- Auto-topic creation enabled
- JSON serialization configured
- Retention policies set for CDC topics

### Debezium Configuration

Key connector settings:
- `plugin.name`: pgoutput (built-in PostgreSQL plugin)
- `snapshot.mode`: initial (capture existing data)
- `transforms`: ExtractNewRecordState (simplify event structure)
- `decimal.handling.mode`: string (avoid precision issues)

### Redis Configuration

Used for:
- Real-time metrics storage
- Customer/product data caching
- Dashboard data aggregation
- Event publishing for notifications

## 📊 Monitoring

### Kafka UI
Access at http://localhost:8080
- View topics and messages
- Monitor consumer lag
- Check connector status

### Connector Status
```bash
# Check connector health
curl http://localhost:8083/connectors/postgres-connector/status

# View connector configuration
curl http://localhost:8083/connectors/postgres-connector/config
```

### Database Activity
```bash
# Monitor replication slot
docker exec -it postgres psql -U postgres -d ecommerce -c "SELECT * FROM pg_replication_slots;"

# Check publication
docker exec -it postgres psql -U postgres -d ecommerce -c "SELECT * FROM pg_publication_tables;"
```

## 🧪 Testing the Pipeline

### 1. Generate Sample Data

```sql
-- Connect to PostgreSQL
docker exec -it postgres psql -U postgres -d ecommerce

-- Insert test customer
INSERT INTO customers (customer_id, first_name, last_name, email, customer_segment) 
VALUES (1001, 'John', 'Doe', 'john@example.com', 'Premium');

-- Insert test product
INSERT INTO products (product_id, product_name, category, unit_price, inventory_quantity) 
VALUES (2001, 'Test Product', 'Electronics', 99.99, 100);

-- Insert test order
INSERT INTO orders (order_id, customer_id, total_amount, order_status) 
VALUES (3001, 1001, 99.99, 'pending');
```

### 2. Verify CDC Events

```bash
# Check Kafka topics
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --list

# Consume CDC events
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-db.public.customers \
  --from-beginning
```

### 3. Check Real-time Metrics

```bash
# Connect to Redis
docker exec -it redis redis-cli

# Check metrics
HGETALL customer_counts
GET "metrics:daily:$(date +%Y-%m-%d):revenue"
KEYS "customer:*"
```

## 🚨 Troubleshooting

### Common Issues

**Connector fails to start:**
```bash
# Check PostgreSQL permissions
docker exec -it postgres psql -U postgres -d ecommerce -c "SELECT * FROM pg_user WHERE usename='debezium';"

# Verify replication settings
docker exec -it postgres psql -U postgres -d ecommerce -c "SHOW wal_level;"
```

**No CDC events:**
```bash
# Check replication slot activity
docker exec -it postgres psql -U postgres -d ecommerce -c "SELECT * FROM pg_stat_replication;"

# Verify publication
docker exec -it postgres psql -U postgres -d ecommerce -c "SELECT * FROM pg_publication;"
```

**Consumer lag:**
```bash
# Check consumer group status
docker exec -it kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe --group cdc-processor
```

### Log Analysis

```bash
# Kafka Connect logs
docker logs connect

# PostgreSQL logs
docker logs postgres

# Kafka logs
docker logs kafka
```

## 🧹 Cleanup

```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: deletes all data)
docker-compose down -v

# Remove images
docker-compose down --rmi all
```

## 📚 Next Steps

1. Complete the exercise in `exercise.py`
2. Run the solution in `solution.py`
3. Explore the Kafka UI at http://localhost:8080
4. Monitor metrics in Redis
5. Take the quiz in `quiz.md`

## 🔗 Useful Commands

```bash
# Restart connector
curl -X POST http://localhost:8083/connectors/postgres-connector/restart

# Delete connector
curl -X DELETE http://localhost:8083/connectors/postgres-connector

# Check Kafka Connect plugins
curl http://localhost:8083/connector-plugins

# Monitor topic messages
docker exec -it kafka kafka-console-consumer \
  --bootstrap-server localhost:9092 \
  --topic ecommerce-db.public.orders \
  --property print.key=true \
  --property print.timestamp=true
```