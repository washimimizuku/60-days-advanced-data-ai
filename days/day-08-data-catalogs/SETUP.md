# Day 8: Data Catalogs Setup Guide

## Quick Start (10 minutes)

### 1. Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ with pip
- At least 8GB RAM available for Docker
- Ports 3000, 8080, 9002, 9200 available

### 2. Environment Setup
```bash
# Clone and navigate to Day 8
cd days/day-08-data-catalogs

# Copy environment template
cp .env.example .env

# Edit .env with your settings (optional for development)
nano .env

# Install Python dependencies
pip install -r requirements.txt
```

### 3. Start DataHub Services
```bash
# Create Docker Compose file (from README.md)
# Copy the complete docker-compose.yml content from README.md

# Start all services
docker-compose up -d

# Wait for services to be ready (2-3 minutes)
docker-compose logs -f datahub-gms
```

### 4. Verify Installation
```bash
# Check service health
curl http://localhost:8080/health
curl http://localhost:9002/authenticate

# Run setup verification
python -c "
import requests
try:
    r = requests.get('http://localhost:8080/health', timeout=5)
    print('✅ DataHub GMS is running' if r.status_code == 200 else '❌ DataHub GMS failed')
except:
    print('❌ DataHub GMS not accessible')
"
```

## Detailed Setup Instructions

### DataHub Architecture Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │      GMS        │    │  Elasticsearch  │
│   (Port 9002)   │───▶│   (Port 8080)   │───▶│   (Port 9200)   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐    ┌─────────────────┐
                       │     MySQL       │    │     Kafka       │
                       │   (Port 3306)   │    │   (Port 9092)   │
                       └─────────────────┘    └─────────────────┘
```

### Service Dependencies
1. **MySQL** - Metadata storage (starts first)
2. **Elasticsearch** - Search and indexing
3. **Kafka + Zookeeper** - Real-time updates
4. **Schema Registry** - Schema management
5. **DataHub GMS** - Backend API service
6. **DataHub Frontend** - Web interface

### Complete Docker Compose Setup

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  # Elasticsearch for search
  elasticsearch:
    image: elasticsearch:7.17.0
    container_name: datahub-elasticsearch
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
      - ES_JAVA_OPTS=-Xms1g -Xmx1g
    ports:
      - "9200:9200"
    volumes:
      - elasticsearch_data:/usr/share/elasticsearch/data
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:9200/_cluster/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 5

  # MySQL for metadata storage
  mysql:
    image: mysql:8.0
    container_name: datahub-mysql
    environment:
      MYSQL_DATABASE: datahub
      MYSQL_USER: datahub
      MYSQL_PASSWORD: datahub
      MYSQL_ROOT_PASSWORD: datahub
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
    healthcheck:
      test: ["CMD", "mysqladmin", "ping", "-h", "localhost"]
      interval: 30s
      timeout: 10s
      retries: 5

  # Zookeeper for Kafka
  zookeeper:
    image: confluentinc/cp-zookeeper:7.4.0
    container_name: datahub-zookeeper
    environment:
      ZOOKEEPER_CLIENT_PORT: 2181
      ZOOKEEPER_TICK_TIME: 2000

  # Kafka for real-time updates
  kafka:
    image: confluentinc/cp-kafka:7.4.0
    container_name: datahub-kafka
    depends_on:
      - zookeeper
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_ADVERTISED_LISTENERS: PLAINTEXT://kafka:9092
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1
    ports:
      - "9092:9092"

  # Schema Registry
  schema-registry:
    image: confluentinc/cp-schema-registry:7.4.0
    container_name: datahub-schema-registry
    depends_on:
      - kafka
    environment:
      SCHEMA_REGISTRY_HOST_NAME: schema-registry
      SCHEMA_REGISTRY_KAFKASTORE_BOOTSTRAP_SERVERS: kafka:9092
    ports:
      - "8081:8081"

  # DataHub GMS (Backend)
  datahub-gms:
    image: linkedin/datahub-gms:v0.11.0
    container_name: datahub-gms
    depends_on:
      - mysql
      - elasticsearch
      - kafka
    environment:
      - EBEAN_DATASOURCE_URL=jdbc:mysql://mysql:3306/datahub?verifyServerCertificate=false&useSSL=false
      - EBEAN_DATASOURCE_USERNAME=datahub
      - EBEAN_DATASOURCE_PASSWORD=datahub
      - KAFKA_BOOTSTRAP_SERVER=kafka:9092
      - ELASTICSEARCH_HOST=elasticsearch
      - ELASTICSEARCH_PORT=9200
    ports:
      - "8080:8080"
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8080/health || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 10

  # DataHub Frontend
  datahub-frontend:
    image: linkedin/datahub-frontend-react:v0.11.0
    container_name: datahub-frontend
    depends_on:
      - datahub-gms
    environment:
      - DATAHUB_GMS_HOST=datahub-gms
      - DATAHUB_GMS_PORT=8080
    ports:
      - "9002:9002"

volumes:
  elasticsearch_data:
  mysql_data:
```

### Startup Sequence
```bash
# 1. Start infrastructure services
docker-compose up -d mysql elasticsearch zookeeper

# 2. Wait for infrastructure (30 seconds)
sleep 30

# 3. Start Kafka services
docker-compose up -d kafka schema-registry

# 4. Wait for Kafka (30 seconds)
sleep 30

# 5. Start DataHub services
docker-compose up -d datahub-gms datahub-frontend

# 6. Check all services
docker-compose ps
```

### Verification Steps
```bash
# 1. Check service health
curl http://localhost:9200/_cluster/health  # Elasticsearch
curl http://localhost:8080/health           # DataHub GMS
curl http://localhost:9002/authenticate     # DataHub Frontend

# 2. Access DataHub UI
open http://localhost:9002
# Default login: datahub / datahub

# 3. Test Python SDK connection
python -c "
from datahub.emitter.rest_emitter import DatahubRestEmitter
emitter = DatahubRestEmitter('http://localhost:8080')
print('✅ DataHub SDK connection successful')
"
```

## Troubleshooting

### Common Issues

#### Services Not Starting
```bash
# Check Docker resources
docker system df
docker stats

# Check logs
docker-compose logs datahub-gms
docker-compose logs elasticsearch

# Restart specific service
docker-compose restart datahub-gms
```

#### Port Conflicts
```bash
# Check port usage
netstat -tulpn | grep :8080
lsof -i :9002

# Stop conflicting services
sudo systemctl stop apache2  # If using port 8080
```

#### Memory Issues
```bash
# Increase Docker memory limit to 8GB+
# Docker Desktop: Settings > Resources > Memory

# Reduce Elasticsearch memory
# Edit docker-compose.yml:
# ES_JAVA_OPTS=-Xms512m -Xmx512m
```

#### Connection Timeouts
```bash
# Wait longer for services to start
sleep 60

# Check service dependencies
docker-compose logs mysql
docker-compose logs kafka
```

### Reset Everything
```bash
# Stop all services
docker-compose down

# Remove volumes (WARNING: Data loss!)
docker-compose down -v

# Remove images (optional)
docker-compose down --rmi all

# Start fresh
docker-compose up -d
```

## Sample Data Setup

### Create Sample PostgreSQL Data
```bash
# Connect to PostgreSQL (if you have a separate instance)
docker run --rm -it postgres:15 psql -h host.docker.internal -U postgres

# Or use the exercise script
python exercise.py
```

### Sample CSV Files
Create `sample_data/` directory with sample CSV files:
```bash
mkdir -p sample_data
echo "user_id,name,email,created_at
1,John Doe,john@example.com,2024-01-01
2,Jane Smith,jane@example.com,2024-01-02" > sample_data/users.csv

echo "order_id,user_id,amount,status,created_at
1,1,99.99,completed,2024-01-01
2,2,149.99,pending,2024-01-02" > sample_data/orders.csv
```

## Next Steps

1. **Complete Exercise**: Run `python exercise.py`
2. **Explore DataHub UI**: Visit http://localhost:9002
3. **Check Solution**: Review `solution.py` for complete implementation
4. **Take Quiz**: Complete `quiz.md` assessment

## Resources

- [DataHub Quickstart](https://datahubproject.io/docs/quickstart)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [DataHub Python SDK](https://datahubproject.io/docs/metadata-ingestion/as-a-library)