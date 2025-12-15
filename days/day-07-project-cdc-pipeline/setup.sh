#!/bin/bash

# CDC Pipeline Setup Script
# This script sets up the complete CDC pipeline environment

set -e

echo "🚀 Setting up CDC Pipeline Project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    mkdir -p postgres/{init-scripts,config}
    mkdir -p debezium/connectors
    mkdir -p stream-processor/{src,config}
    mkdir -p analytics/init-scripts
    mkdir -p monitoring/{prometheus/{rules,config},grafana/{dashboards,datasources}}
    mkdir -p data-generator/src
    mkdir -p logs
    mkdir -p tests
    
    print_success "Directory structure created"
}

# Create PostgreSQL configuration
create_postgres_config() {
    print_status "Creating PostgreSQL configuration..."
    
    cat > postgres/postgresql.conf << 'EOF'
# PostgreSQL configuration for CDC
listen_addresses = '*'
port = 5432
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200

# WAL configuration for logical replication
wal_level = logical
max_wal_senders = 10
max_replication_slots = 10
wal_sender_timeout = 60s
max_wal_size = 1GB
min_wal_size = 80MB

# Logging
log_destination = 'stderr'
logging_collector = on
log_directory = 'log'
log_filename = 'postgresql-%Y-%m-%d_%H%M%S.log'
log_truncate_on_rotation = on
log_rotation_age = 1d
log_rotation_size = 10MB
log_min_messages = warning
log_min_error_statement = error
log_min_duration_statement = 1000
log_checkpoints = on
log_connections = on
log_disconnections = on
log_lock_waits = on
log_temp_files = 0
log_autovacuum_min_duration = 0
log_error_verbosity = default
EOF

    print_success "PostgreSQL configuration created"
}

# Create database initialization scripts
create_db_init_scripts() {
    print_status "Creating database initialization scripts..."
    
    # Main database schema
    cat > postgres/init-scripts/01-create-schema.sql << 'EOF'
-- Create ecommerce database schema
\c ecommerce;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended'))
);

-- Create products table
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    category VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL CHECK (price > 0),
    stock_quantity INTEGER DEFAULT 0 CHECK (stock_quantity >= 0),
    min_threshold INTEGER DEFAULT 10,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create orders table
CREATE TABLE orders (
    order_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    total_amount DECIMAL(10,2) NOT NULL CHECK (total_amount > 0),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'processing', 'completed', 'cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create order_items table
CREATE TABLE order_items (
    item_id SERIAL PRIMARY KEY,
    order_id INTEGER REFERENCES orders(order_id) ON DELETE CASCADE,
    product_id INTEGER REFERENCES products(product_id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price DECIMAL(10,2) NOT NULL CHECK (unit_price > 0),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_status ON users(status);
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_products_category ON products(category);
CREATE INDEX idx_products_stock ON products(stock_quantity);

-- Create triggers for updated_at timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_products_updated_at BEFORE UPDATE ON products
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
EOF

    # Create replication user and setup
    cat > postgres/init-scripts/02-setup-replication.sql << 'EOF'
-- Create replication user for Debezium
CREATE USER debezium WITH REPLICATION PASSWORD 'debezium';

-- Set connection limits
ALTER USER debezium CONNECTION LIMIT 10;

-- Grant necessary permissions
GRANT SELECT ON ALL TABLES IN SCHEMA public TO debezium;
GRANT USAGE ON SCHEMA public TO debezium;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO debezium;

-- Create publication for logical replication
CREATE PUBLICATION debezium_publication FOR ALL TABLES;

-- Create replication slot
SELECT pg_create_logical_replication_slot('debezium_slot', 'pgoutput');
EOF

    # Sample data
    cat > postgres/init-scripts/03-sample-data.sql << 'EOF'
-- Insert sample data
\c ecommerce;

-- Sample users
INSERT INTO users (email, first_name, last_name, status) VALUES
('john.doe@example.com', 'John', 'Doe', 'active'),
('jane.smith@example.com', 'Jane', 'Smith', 'active'),
('bob.johnson@example.com', 'Bob', 'Johnson', 'active'),
('alice.brown@example.com', 'Alice', 'Brown', 'inactive'),
('charlie.wilson@example.com', 'Charlie', 'Wilson', 'active');

-- Sample products
INSERT INTO products (name, category, price, stock_quantity, min_threshold) VALUES
('Laptop Pro 15"', 'Electronics', 1299.99, 50, 10),
('Wireless Mouse', 'Electronics', 29.99, 200, 20),
('Coffee Mug', 'Home', 12.99, 100, 15),
('Running Shoes', 'Sports', 89.99, 75, 10),
('Bluetooth Headphones', 'Electronics', 149.99, 30, 5),
('Desk Chair', 'Home', 199.99, 25, 5),
('Water Bottle', 'Sports', 19.99, 150, 25),
('Smartphone', 'Electronics', 699.99, 40, 8),
('Backpack', 'Sports', 59.99, 60, 12),
('Table Lamp', 'Home', 39.99, 80, 10);

-- Sample orders
INSERT INTO orders (user_id, total_amount, status) VALUES
(1, 1329.98, 'completed'),
(2, 149.99, 'processing'),
(3, 89.99, 'pending'),
(1, 32.98, 'completed'),
(4, 699.99, 'cancelled');

-- Sample order items
INSERT INTO order_items (order_id, product_id, quantity, unit_price) VALUES
(1, 1, 1, 1299.99),
(1, 2, 1, 29.99),
(2, 5, 1, 149.99),
(3, 4, 1, 89.99),
(4, 3, 1, 12.99),
(4, 7, 1, 19.99),
(5, 8, 1, 699.99);
EOF

    # Analytics database schema
    cat > analytics/init-scripts/01-analytics-schema.sql << 'EOF'
-- Analytics database schema
\c analytics;

-- Revenue analytics table
CREATE TABLE revenue_analytics (
    id SERIAL PRIMARY KEY,
    window_start TIMESTAMP NOT NULL,
    window_end TIMESTAMP NOT NULL,
    total_revenue DECIMAL(15,2) NOT NULL,
    order_count INTEGER NOT NULL,
    avg_order_value DECIMAL(10,2) NOT NULL,
    top_category VARCHAR(100),
    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User activity summary
CREATE TABLE user_activity (
    id SERIAL PRIMARY KEY,
    user_id INTEGER NOT NULL,
    session_start TIMESTAMP NOT NULL,
    session_end TIMESTAMP,
    page_views INTEGER DEFAULT 0,
    orders_placed INTEGER DEFAULT 0,
    total_spent DECIMAL(10,2) DEFAULT 0,
    last_activity TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Inventory alerts
CREATE TABLE inventory_alerts (
    id SERIAL PRIMARY KEY,
    product_id INTEGER NOT NULL,
    product_name VARCHAR(255) NOT NULL,
    current_stock INTEGER NOT NULL,
    threshold INTEGER NOT NULL,
    alert_level VARCHAR(20) NOT NULL CHECK (alert_level IN ('low', 'critical', 'out_of_stock')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
);

-- Data quality metrics
CREATE TABLE data_quality_metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    threshold_value DECIMAL(10,4),
    status VARCHAR(20) DEFAULT 'ok' CHECK (status IN ('ok', 'warning', 'critical')),
    measured_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_revenue_analytics_window ON revenue_analytics(window_start, window_end);
CREATE INDEX idx_user_activity_user_id ON user_activity(user_id);
CREATE INDEX idx_user_activity_session ON user_activity(session_start, session_end);
CREATE INDEX idx_inventory_alerts_product ON inventory_alerts(product_id);
CREATE INDEX idx_inventory_alerts_level ON inventory_alerts(alert_level);
CREATE INDEX idx_data_quality_metrics_name ON data_quality_metrics(metric_name);
EOF

    print_success "Database initialization scripts created"
}

# Create Kafka topics
create_kafka_topics() {
    print_status "Creating Kafka topic creation script..."
    
    cat > create-topics.sh << 'EOF'
#!/bin/bash

# Wait for Kafka to be ready
echo "Waiting for Kafka to be ready..."
sleep 30

# Create topics
echo "Creating Kafka topics..."

# Source data topics
kafka-topics --create --bootstrap-server kafka1:29092 --topic users --partitions 6 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server kafka1:29092 --topic orders --partitions 12 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server kafka1:29092 --topic products --partitions 3 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server kafka1:29092 --topic order_items --partitions 12 --replication-factor 3 --if-not-exists

# Processed data topics
kafka-topics --create --bootstrap-server kafka1:29092 --topic revenue-analytics --partitions 6 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server kafka1:29092 --topic user-behavior --partitions 6 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server kafka1:29092 --topic inventory-alerts --partitions 3 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server kafka1:29092 --topic data-quality-events --partitions 3 --replication-factor 3 --if-not-exists

# Dead letter queue
kafka-topics --create --bootstrap-server kafka1:29092 --topic dlq-events --partitions 3 --replication-factor 3 --if-not-exists

# Connect topics (if not auto-created)
kafka-topics --create --bootstrap-server kafka1:29092 --topic connect-configs --partitions 1 --replication-factor 3 --config cleanup.policy=compact --if-not-exists
kafka-topics --create --bootstrap-server kafka1:29092 --topic connect-offsets --partitions 25 --replication-factor 3 --config cleanup.policy=compact --if-not-exists
kafka-topics --create --bootstrap-server kafka1:29092 --topic connect-status --partitions 5 --replication-factor 3 --config cleanup.policy=compact --if-not-exists

echo "Topics created successfully!"

# List all topics
echo "Current topics:"
kafka-topics --list --bootstrap-server kafka1:29092
EOF

    chmod +x create-topics.sh
    print_success "Kafka topic creation script created"
}

# Create Debezium connector configuration
create_debezium_config() {
    print_status "Creating Debezium connector configuration..."
    
    cat > debezium-connector.json << 'EOF'
{
  "name": "ecommerce-postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "${DEBEZIUM_PASSWORD}",
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
    "transforms.unwrap.delete.handling.mode": "rewrite",
    "key.converter": "io.confluent.connect.avro.AvroConverter",
    "value.converter": "io.confluent.connect.avro.AvroConverter",
    "key.converter.schema.registry.url": "http://schema-registry:8081",
    "value.converter.schema.registry.url": "http://schema-registry:8081",
    "snapshot.mode": "initial",
    "decimal.handling.mode": "double",
    "time.precision.mode": "adaptive",
    "include.schema.changes": "true",
    "provide.transaction.metadata": "true",
    "max.batch.size": "2048",
    "max.queue.size": "8192",
    "poll.interval.ms": "1000"
  }
}
EOF

    print_success "Debezium connector configuration created"
}

# Create monitoring configuration
create_monitoring_config() {
    print_status "Creating monitoring configuration..."
    
    # Prometheus configuration
    cat > monitoring/prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'kafka'
    static_configs:
      - targets: ['kafka1:9999', 'kafka2:9998', 'kafka3:9997']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'kafka-connect'
    static_configs:
      - targets: ['kafka-connect:8083']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'stream-processor'
    static_configs:
      - targets: ['stream-processor:8080']
    metrics_path: /actuator/prometheus
    scrape_interval: 15s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    # Alerting rules
    cat > monitoring/prometheus/rules/kafka-rules.yml << 'EOF'
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

  - alert: KafkaBrokerDown
    expr: up{job="kafka"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Kafka broker is down"
      description: "Kafka broker {{ $labels.instance }} is down"

  - alert: HighDLQRate
    expr: rate(dlq_events_total[5m]) > 10
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High dead letter queue rate"
      description: "DLQ rate is {{ $value }} events/sec"
EOF

    # Grafana datasource
    cat > monitoring/grafana/datasources/prometheus.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    print_success "Monitoring configuration created"
}

# Create deployment script
create_deployment_script() {
    print_status "Creating deployment script..."
    
    cat > deploy.sh << 'EOF'
#!/bin/bash

set -e

echo "🚀 Deploying CDC Pipeline..."

# Start infrastructure services first
echo "Starting infrastructure services..."
docker-compose up -d zookeeper kafka1 kafka2 kafka3 postgres analytics-db redis

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Start remaining services
echo "Starting application services..."
docker-compose up -d schema-registry kafka-connect

# Wait for Kafka Connect to be ready
echo "Waiting for Kafka Connect to be ready..."
sleep 30

# Create Kafka topics
echo "Creating Kafka topics..."
docker-compose exec kafka1 /bin/bash -c "
kafka-topics --create --bootstrap-server localhost:29092 --topic users --partitions 6 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server localhost:29092 --topic orders --partitions 12 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server localhost:29092 --topic products --partitions 3 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server localhost:29092 --topic order_items --partitions 12 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server localhost:29092 --topic revenue-analytics --partitions 6 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server localhost:29092 --topic user-behavior --partitions 6 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server localhost:29092 --topic inventory-alerts --partitions 3 --replication-factor 3 --if-not-exists
kafka-topics --create --bootstrap-server localhost:29092 --topic dlq-events --partitions 3 --replication-factor 3 --if-not-exists
"

# Deploy Debezium connector
echo "Deploying Debezium connector..."
curl -X POST -H "Content-Type: application/json" --data @debezium-connector.json http://localhost:8083/connectors

# Start monitoring and processing services
echo "Starting monitoring and processing services..."
docker-compose up -d prometheus grafana kafka-manager stream-processor data-generator

echo "✅ Deployment complete!"
echo ""
echo "🔗 Service URLs:"
echo "  - Kafka Manager: http://localhost:9000"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo "  - Kafka Connect: http://localhost:8083"
echo "  - Schema Registry: http://localhost:8081"
echo ""
echo "📊 Database Connections:"
echo "  - Source DB: postgresql://postgres:postgres@localhost:5432/ecommerce"
echo "  - Analytics DB: postgresql://postgres:postgres@localhost:5433/analytics"
echo "  - Redis: redis://localhost:6379"
EOF

    chmod +x deploy.sh
    print_success "Deployment script created"
}

# Create README
create_readme() {
    print_status "Creating project README..."
    
    cat > README.md << 'EOF'
# CDC Pipeline Project

A complete real-time Change Data Capture (CDC) pipeline built with PostgreSQL, Debezium, Kafka, and stream processing.

## Architecture

```
PostgreSQL → Debezium → Kafka → Stream Processing → Analytics
     ↓           ↓        ↓           ↓              ↓
  Source DB   CDC Conn  Events   Real-time      Dashboards
                                Processing
```

## Quick Start

1. **Prerequisites**
   - Docker and Docker Compose
   - At least 8GB RAM available
   - Ports 3000, 5432, 5433, 6379, 8080-8083, 9000, 9090-9094 available

2. **Deploy the Pipeline**
   ```bash
   ./setup.sh    # Run this setup script first
   ./deploy.sh   # Deploy all services
   ```

3. **Verify Deployment**
   ```bash
   # Check all services are running
   docker-compose ps
   
   # Check Kafka topics
   docker-compose exec kafka1 kafka-topics --list --bootstrap-server localhost:29092
   
   # Check Debezium connector status
   curl http://localhost:8083/connectors/ecommerce-postgres-connector/status
   ```

## Services

| Service | Port | Description |
|---------|------|-------------|
| PostgreSQL (Source) | 5432 | E-commerce database |
| PostgreSQL (Analytics) | 5433 | Analytics database |
| Kafka Brokers | 9092-9094 | Event streaming |
| Schema Registry | 8081 | Schema management |
| Kafka Connect | 8083 | CDC connector |
| Stream Processor | 8080 | Real-time processing |
| Kafka Manager | 9000 | Cluster management |
| Prometheus | 9090 | Metrics collection |
| Grafana | 3000 | Dashboards |
| Redis | 6379 | Caching |

## Monitoring

- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090
- **Kafka Manager**: http://localhost:9000

## Testing

Generate test data:
```bash
docker-compose exec data-generator python generate_data.py
```

Monitor events:
```bash
# Watch Kafka topics
docker-compose exec kafka1 kafka-console-consumer --bootstrap-server localhost:29092 --topic orders --from-beginning

# Check analytics results
docker-compose exec analytics-db psql -U postgres -d analytics -c "SELECT * FROM revenue_analytics ORDER BY window_start DESC LIMIT 10;"
```

## Troubleshooting

1. **Services not starting**: Check Docker resources and port availability
2. **Connector fails**: Verify PostgreSQL replication setup
3. **No events**: Check Debezium connector status and logs
4. **High lag**: Scale stream processors or optimize processing logic

## Project Structure

```
├── docker-compose.yml          # Service definitions
├── postgres/                   # PostgreSQL configuration
├── debezium/                   # CDC connector configs
├── stream-processor/           # Stream processing apps
├── analytics/                  # Analytics database setup
├── monitoring/                 # Prometheus & Grafana configs
├── data-generator/             # Test data generation
└── tests/                      # Integration tests
```

## Learning Objectives

- Real-time data integration with CDC
- Event streaming with Kafka
- Stream processing patterns
- Data quality and monitoring
- Production deployment practices
EOF

    print_success "Project README created"
}

# Main execution
main() {
    echo "🎯 CDC Pipeline Project Setup"
    echo "=============================="
    
    check_prerequisites
    create_directories
    create_postgres_config
    create_db_init_scripts
    create_kafka_topics
    create_debezium_config
    create_monitoring_config
    create_deployment_script
    create_readme
    
    print_success "Setup completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Review the generated configuration files"
    echo "2. Run './deploy.sh' to start all services"
    echo "3. Access the monitoring dashboards"
    echo "4. Generate test data and observe the pipeline"
    echo ""
    echo "🔗 Quick Links:"
    echo "  - Project README: ./README.md"
    echo "  - Deployment: ./deploy.sh"
    echo "  - Connector Config: ./debezium-connector.json"
    echo ""
    print_warning "Make sure you have at least 8GB RAM available for Docker!"
}

# Run main function
main "$@"