# CDC Pipeline Troubleshooting Guide

## Common Issues and Solutions

### 1. Services Not Starting

#### Symptoms
- Docker containers failing to start
- Connection refused errors
- Services in unhealthy state

#### Diagnosis
```bash
# Check container status
docker-compose ps

# Check container logs
docker-compose logs [service-name]

# Check system resources
docker system df
docker stats
```

#### Solutions

**Insufficient Resources:**
```bash
# Check available memory
free -h

# Check disk space
df -h

# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > 8GB+
```

**Port Conflicts:**
```bash
# Check port usage
netstat -tulpn | grep :5432
lsof -i :9092

# Stop conflicting services
sudo systemctl stop postgresql
sudo systemctl stop kafka
```

**Permission Issues:**
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER .
```

### 2. Database Connection Issues

#### Symptoms
- "Connection refused" to PostgreSQL
- Authentication failures
- Timeout errors

#### Diagnosis
```bash
# Test database connection
docker-compose exec postgres psql -U postgres -d ecommerce -c "SELECT 1;"

# Check PostgreSQL logs
docker-compose logs postgres

# Verify network connectivity
docker-compose exec data-generator ping postgres
```

#### Solutions

**Connection Refused:**
```bash
# Restart PostgreSQL
docker-compose restart postgres

# Check PostgreSQL configuration
docker-compose exec postgres cat /etc/postgresql/postgresql.conf | grep listen_addresses

# Verify port binding
docker-compose ps postgres
```

**Authentication Failures:**
```bash
# Check environment variables
docker-compose config | grep POSTGRES

# Reset passwords
docker-compose down
docker volume rm cdc-pipeline_postgres_data
docker-compose up -d postgres
```

**Replication Issues:**
```bash
# Check replication slot
docker-compose exec postgres psql -U postgres -d ecommerce -c "SELECT * FROM pg_replication_slots;"

# Check publication
docker-compose exec postgres psql -U postgres -d ecommerce -c "SELECT * FROM pg_publication;"

# Recreate replication setup
docker-compose exec postgres psql -U postgres -d ecommerce -c "DROP PUBLICATION IF EXISTS debezium_publication;"
docker-compose exec postgres psql -U postgres -d ecommerce -c "CREATE PUBLICATION debezium_publication FOR ALL TABLES;"
```

### 3. Kafka Issues

#### Symptoms
- Kafka brokers not forming cluster
- Topics not created
- Consumer lag increasing
- Messages not being produced/consumed

#### Diagnosis
```bash
# Check Kafka cluster status
docker-compose exec kafka1 kafka-topics --bootstrap-server localhost:29092 --list

# Check broker logs
docker-compose logs kafka1 kafka2 kafka3

# Check topic details
docker-compose exec kafka1 kafka-topics --bootstrap-server localhost:29092 --describe --topic orders

# Monitor consumer groups
docker-compose exec kafka1 kafka-consumer-groups --bootstrap-server localhost:29092 --list
```

#### Solutions

**Cluster Formation Issues:**
```bash
# Restart Zookeeper first
docker-compose restart zookeeper
sleep 30

# Restart Kafka brokers in sequence
docker-compose restart kafka1
sleep 30
docker-compose restart kafka2
sleep 30
docker-compose restart kafka3
```

**Topic Creation Failures:**
```bash
# Manually create topics
docker-compose exec kafka1 kafka-topics --create --bootstrap-server localhost:29092 --topic orders --partitions 12 --replication-factor 3

# Check topic configuration
docker-compose exec kafka1 kafka-topics --bootstrap-server localhost:29092 --describe --topic orders
```

**Consumer Lag:**
```bash
# Check consumer group status
docker-compose exec kafka1 kafka-consumer-groups --bootstrap-server localhost:29092 --describe --group cdc-stream-processor

# Reset consumer offsets (if needed)
docker-compose exec kafka1 kafka-consumer-groups --bootstrap-server localhost:29092 --group cdc-stream-processor --reset-offsets --to-earliest --topic orders --execute
```

### 4. Debezium Connector Issues

#### Symptoms
- Connector not starting
- No CDC events being produced
- Connector in FAILED state
- Schema evolution errors

#### Diagnosis
```bash
# Check connector status
curl http://localhost:8083/connectors/ecommerce-postgres-connector/status

# Check connector configuration
curl http://localhost:8083/connectors/ecommerce-postgres-connector/config

# Check Kafka Connect logs
docker-compose logs kafka-connect

# List all connectors
curl http://localhost:8083/connectors
```

#### Solutions

**Connector Not Starting:**
```bash
# Delete and recreate connector
curl -X DELETE http://localhost:8083/connectors/ecommerce-postgres-connector

# Wait a moment, then recreate
curl -X POST -H "Content-Type: application/json" --data @debezium-connector.json http://localhost:8083/connectors
```

**No Events Being Produced:**
```bash
# Check PostgreSQL WAL level
docker-compose exec postgres psql -U postgres -c "SHOW wal_level;"

# Check replication slot activity
docker-compose exec postgres psql -U postgres -d ecommerce -c "SELECT * FROM pg_stat_replication;"

# Verify table changes are being made
docker-compose exec postgres psql -U postgres -d ecommerce -c "INSERT INTO users (email, first_name, last_name) VALUES ('test@example.com', 'Test', 'User');"
```

**Schema Registry Issues:**
```bash
# Check Schema Registry health
curl http://localhost:8081/subjects

# Check connector schema configuration
curl http://localhost:8083/connectors/ecommerce-postgres-connector/config | jq '.config."key.converter.schema.registry.url"'

# Restart Schema Registry
docker-compose restart schema-registry
```

### 5. Stream Processing Issues

#### Symptoms
- Stream processor not consuming messages
- Processing errors in logs
- No output to analytics topics
- High memory usage

#### Diagnosis
```bash
# Check stream processor logs
docker-compose logs stream-processor

# Check JVM metrics
docker-compose exec stream-processor jps -v

# Monitor Kafka Streams state
# (Check application logs for state store information)

# Check output topics
docker-compose exec kafka1 kafka-console-consumer --bootstrap-server localhost:29092 --topic revenue-analytics --from-beginning
```

#### Solutions

**Consumer Not Starting:**
```bash
# Restart stream processor
docker-compose restart stream-processor

# Check Kafka connectivity from processor
docker-compose exec stream-processor ping kafka1

# Verify topic existence
docker-compose exec kafka1 kafka-topics --bootstrap-server localhost:29092 --list | grep -E "(orders|users|products)"
```

**Processing Errors:**
```bash
# Check for serialization issues
docker-compose logs stream-processor | grep -i "serialization\|deserialization"

# Reset stream processor state (if needed)
docker-compose stop stream-processor
docker-compose exec kafka1 kafka-streams-application-reset --application-id cdc-stream-processor --bootstrap-servers localhost:29092
docker-compose start stream-processor
```

**Memory Issues:**
```bash
# Increase JVM heap size
# Edit docker-compose.yml:
# JAVA_OPTS: "-Xmx1G -Xms1G"

# Monitor memory usage
docker stats stream-processor
```

### 6. Data Quality Issues

#### Symptoms
- High number of DLQ events
- Data validation errors
- Missing or incorrect data
- Schema compatibility issues

#### Diagnosis
```bash
# Check DLQ topic
docker-compose exec kafka1 kafka-console-consumer --bootstrap-server localhost:29092 --topic dlq-events --from-beginning

# Check data quality metrics
docker-compose exec kafka1 kafka-console-consumer --bootstrap-server localhost:29092 --topic data-quality-events --from-beginning

# Verify source data
docker-compose exec postgres psql -U postgres -d ecommerce -c "SELECT * FROM orders ORDER BY created_at DESC LIMIT 10;"
```

#### Solutions

**High DLQ Rate:**
```bash
# Analyze DLQ messages
docker-compose exec kafka1 kafka-console-consumer --bootstrap-server localhost:29092 --topic dlq-events --from-beginning | jq '.errors'

# Fix data generation issues
docker-compose logs data-generator

# Update validation rules if needed
# (Modify stream processor validation logic)
```

**Schema Issues:**
```bash
# Check schema registry subjects
curl http://localhost:8081/subjects

# Check schema compatibility
curl http://localhost:8081/compatibility/subjects/orders-value/versions/latest

# Update schema if needed
# (Use Schema Registry API or recreate connector)
```

### 7. Monitoring Issues

#### Symptoms
- Grafana dashboards not loading
- Prometheus not scraping metrics
- Missing metrics data
- Alert notifications not working

#### Diagnosis
```bash
# Check Grafana health
curl http://localhost:3000/api/health

# Check Prometheus targets
curl http://localhost:9090/api/v1/targets

# Check metrics endpoints
curl http://localhost:8080/actuator/prometheus  # Stream processor
curl http://localhost:8000/metrics             # Data generator

# Check Grafana logs
docker-compose logs grafana
```

#### Solutions

**Grafana Issues:**
```bash
# Restart Grafana
docker-compose restart grafana

# Check datasource configuration
# Login to Grafana (admin/admin) and verify Prometheus datasource

# Reset Grafana data (if needed)
docker-compose down
docker volume rm cdc-pipeline_grafana_data
docker-compose up -d grafana
```

**Prometheus Issues:**
```bash
# Check Prometheus configuration
docker-compose exec prometheus cat /etc/prometheus/prometheus.yml

# Restart Prometheus
docker-compose restart prometheus

# Check service discovery
curl http://localhost:9090/api/v1/targets | jq '.data.activeTargets[] | select(.health != "up")'
```

### 8. Performance Issues

#### Symptoms
- High latency in processing
- Low throughput
- Resource exhaustion
- Slow query performance

#### Diagnosis
```bash
# Monitor system resources
docker stats

# Check Kafka metrics
# Use Kafka Manager: http://localhost:9000

# Monitor database performance
docker-compose exec postgres psql -U postgres -d ecommerce -c "SELECT * FROM pg_stat_activity;"

# Check processing latency
# Monitor Grafana dashboards for latency metrics
```

#### Solutions

**Scale Kafka:**
```bash
# Increase partition count for high-volume topics
docker-compose exec kafka1 kafka-topics --bootstrap-server localhost:29092 --alter --topic orders --partitions 24

# Tune Kafka configuration
# Edit docker-compose.yml to increase memory and adjust settings
```

**Scale Stream Processing:**
```bash
# Increase stream processor instances
docker-compose up -d --scale stream-processor=3

# Tune JVM settings
# Increase heap size and tune GC settings
```

**Database Optimization:**
```bash
# Add indexes for frequently queried columns
docker-compose exec postgres psql -U postgres -d ecommerce -c "CREATE INDEX CONCURRENTLY idx_orders_status_created ON orders(status, created_at);"

# Analyze query performance
docker-compose exec postgres psql -U postgres -d ecommerce -c "EXPLAIN ANALYZE SELECT * FROM orders WHERE status = 'pending';"
```

## Emergency Procedures

### Complete System Reset

```bash
# Stop all services
docker-compose down

# Remove all volumes (WARNING: Data loss!)
docker volume prune -f

# Remove all containers and networks
docker system prune -f

# Restart from clean state
docker-compose up -d
```

### Backup and Recovery

```bash
# Backup PostgreSQL data
docker-compose exec postgres pg_dump -U postgres ecommerce > backup_$(date +%Y%m%d_%H%M%S).sql

# Backup Kafka topics (if needed)
# Use Kafka backup tools or export to files

# Restore PostgreSQL data
docker-compose exec -T postgres psql -U postgres ecommerce < backup_file.sql
```

### Log Collection

```bash
# Collect all logs
mkdir -p logs/$(date +%Y%m%d_%H%M%S)
docker-compose logs > logs/$(date +%Y%m%d_%H%M%S)/all_services.log

# Collect individual service logs
for service in postgres kafka1 kafka-connect stream-processor; do
    docker-compose logs $service > logs/$(date +%Y%m%d_%H%M%S)/${service}.log
done
```

## Getting Help

### Log Analysis
- Check service-specific logs first
- Look for ERROR and WARN level messages
- Check for connection and authentication issues
- Monitor resource usage patterns

### Community Resources
- [Debezium Community](https://debezium.io/community/)
- [Kafka Users Mailing List](https://kafka.apache.org/contact)
- [PostgreSQL Community](https://www.postgresql.org/community/)

### Professional Support
- Consider commercial support for production deployments
- Engage with vendors for enterprise features
- Consult with data engineering experts for complex issues