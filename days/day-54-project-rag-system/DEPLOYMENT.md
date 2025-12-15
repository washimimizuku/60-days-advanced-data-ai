# Day 54: Production RAG System - Deployment Guide

## Overview
This guide provides comprehensive instructions for deploying the Production RAG System in various environments, from local development to enterprise production.

## Prerequisites

### System Requirements
- **CPU**: 4+ cores (8+ recommended for production)
- **RAM**: 8GB minimum (16GB+ recommended)
- **Storage**: 50GB+ available space
- **Network**: Stable internet connection for LLM APIs

### Software Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+ (for local development)
- Git

### API Keys and Services
- OpenAI API key (or alternative LLM provider)
- Optional: Pinecone, Anthropic, or other service keys

## Quick Start (Local Development)

### 1. Clone and Setup
```bash
# Clone the repository
git clone <repository-url>
cd day-54-project-rag-system

# Copy environment configuration
cp .env.example .env

# Edit .env with your API keys
nano .env  # or your preferred editor
```

### 2. Environment Configuration
Update `.env` file with your settings:
```bash
# Required: OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Adjust other settings
LOG_LEVEL=INFO
MAX_CHUNK_SIZE=1000
FAITHFULNESS_THRESHOLD=0.7
```

### 3. Start with Docker Compose
```bash
# Start all services
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f rag-api
```

### 4. Verify Deployment
```bash
# Health check
curl http://localhost:8000/health

# Test query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I authenticate with the API?"}'

# View API documentation
open http://localhost:8000/docs
```

## Production Deployment

### Option 1: Docker Compose (Recommended for Small-Medium Scale)

#### 1. Production Environment Setup
```bash
# Create production directory
mkdir -p /opt/rag-system
cd /opt/rag-system

# Copy deployment files
cp docker-compose.yml .
cp .env.example .env

# Create data directories
mkdir -p data/{documents,vectordb,logs}
mkdir -p monitoring/{prometheus,grafana}
```

#### 2. Production Configuration
```bash
# Edit production environment
nano .env

# Key production settings:
DEBUG_MODE=false
LOG_LEVEL=INFO
API_WORKERS=4
ENABLE_PROMETHEUS_METRICS=true
ENABLE_HTTPS=true
```

#### 3. SSL/TLS Setup (Production)
```bash
# Generate SSL certificates (example with Let's Encrypt)
certbot certonly --standalone -d your-domain.com

# Update nginx configuration
cp nginx/nginx.conf.prod nginx/nginx.conf

# Update docker-compose.yml with SSL paths
```

#### 4. Start Production Services
```bash
# Start in production mode
docker-compose -f docker-compose.yml up -d

# Enable auto-restart
docker-compose -f docker-compose.yml up -d --restart unless-stopped
```

### Option 2: Kubernetes Deployment

#### 1. Kubernetes Manifests
Create `k8s/` directory with manifests:

**Namespace:**
```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
```

**ConfigMap:**
```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  LOG_LEVEL: "INFO"
  API_WORKERS: "4"
  MAX_CHUNK_SIZE: "1000"
```

**Secret:**
```yaml
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: rag-secrets
  namespace: rag-system
type: Opaque
data:
  OPENAI_API_KEY: <base64-encoded-key>
```

**Deployment:**
```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: rag-system:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: OPENAI_API_KEY
        envFrom:
        - configMapRef:
            name: rag-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
```

**Service:**
```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
spec:
  selector:
    app: rag-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

#### 2. Deploy to Kubernetes
```bash
# Apply manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n rag-system
kubectl get services -n rag-system

# View logs
kubectl logs -f deployment/rag-api -n rag-system
```

### Option 3: Cloud Platform Deployment

#### AWS ECS Deployment

**1. Create Task Definition:**
```json
{
  "family": "rag-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "rag-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/rag-system:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "secrets": [
        {
          "name": "OPENAI_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:rag-secrets"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/rag-system",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

**2. Create ECS Service:**
```bash
# Create cluster
aws ecs create-cluster --cluster-name rag-system

# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster rag-system \
  --service-name rag-api \
  --task-definition rag-system:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

## Monitoring and Observability

### 1. Prometheus Metrics
Access metrics at: `http://localhost:9090`

Key metrics to monitor:
- `rag_requests_total` - Total requests
- `rag_request_duration_seconds` - Response times
- `rag_evaluation_scores` - Quality metrics
- `rag_active_connections` - Concurrent users

### 2. Grafana Dashboards
Access dashboards at: `http://localhost:3000`

Default login: `admin/admin`

Import provided dashboard configurations:
- System Performance Dashboard
- RAG Quality Metrics Dashboard
- Error Rate and Alerting Dashboard

### 3. Log Aggregation
```bash
# View application logs
docker-compose logs -f rag-api

# Search logs with grep
docker-compose logs rag-api | grep ERROR

# Export logs for analysis
docker-compose logs --no-color rag-api > rag-system.log
```

### 4. Health Monitoring
```bash
# Automated health checks
curl -f http://localhost:8000/health || echo "Service unhealthy"

# Detailed health status
curl -s http://localhost:8000/health | jq '.'
```

## Scaling and Performance

### Horizontal Scaling

#### Docker Compose Scaling
```bash
# Scale API service
docker-compose up -d --scale rag-api=3

# Add load balancer
docker-compose -f docker-compose.yml -f docker-compose.scale.yml up -d
```

#### Kubernetes Scaling
```bash
# Scale deployment
kubectl scale deployment rag-api --replicas=5 -n rag-system

# Auto-scaling
kubectl autoscale deployment rag-api --cpu-percent=70 --min=2 --max=10 -n rag-system
```

### Performance Optimization

#### 1. Database Optimization
```bash
# Optimize vector database
# For FAISS: Use IVF indexes for large datasets
# For ChromaDB: Configure appropriate batch sizes

# Redis optimization
redis-cli CONFIG SET maxmemory 2gb
redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

#### 2. Application Tuning
```bash
# Increase worker processes
export API_WORKERS=8

# Optimize chunk processing
export MAX_CHUNK_SIZE=800
export CHUNK_OVERLAP=150

# Enable caching
export ENABLE_CACHING=true
export CACHE_TTL_SECONDS=7200
```

#### 3. Load Testing
```bash
# Install load testing tools
pip install locust

# Run load test
locust -f load_test.py --host=http://localhost:8000
```

## Security Configuration

### 1. API Security
```bash
# Enable API authentication
export ENABLE_API_AUTHENTICATION=true
export API_KEY=your-secure-api-key

# Configure JWT
export JWT_SECRET_KEY=your-jwt-secret
export JWT_EXPIRATION_HOURS=24
```

### 2. Network Security
```bash
# Configure firewall rules
ufw allow 80/tcp
ufw allow 443/tcp
ufw deny 8000/tcp  # Block direct API access

# Use reverse proxy
nginx -t && nginx -s reload
```

### 3. Data Security
```bash
# Enable data encryption
export ENCRYPT_STORED_DATA=true
export ENCRYPTION_KEY=your-encryption-key

# Configure HTTPS
export ENABLE_HTTPS=true
export SSL_CERT_PATH=/path/to/cert.pem
export SSL_KEY_PATH=/path/to/key.pem
```

## Backup and Recovery

### 1. Data Backup
```bash
# Backup vector database
tar -czf vectordb-backup-$(date +%Y%m%d).tar.gz data/vectordb/

# Backup documents
rsync -av data/documents/ backup/documents/

# Backup configuration
cp .env backup/env-$(date +%Y%m%d)
```

### 2. Database Backup
```bash
# Redis backup
redis-cli BGSAVE
cp /var/lib/redis/dump.rdb backup/redis-$(date +%Y%m%d).rdb

# SQLite backup (if used)
sqlite3 data/rag_system.db ".backup backup/rag_system-$(date +%Y%m%d).db"
```

### 3. Automated Backup Script
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/opt/backups/rag-system"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR/$DATE

# Backup data
tar -czf $BACKUP_DIR/$DATE/vectordb.tar.gz data/vectordb/
tar -czf $BACKUP_DIR/$DATE/documents.tar.gz data/documents/
cp .env $BACKUP_DIR/$DATE/

# Cleanup old backups (keep 30 days)
find $BACKUP_DIR -type d -mtime +30 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR/$DATE"
```

## Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check logs
docker-compose logs rag-api

# Common causes:
# - Missing API keys
# - Port conflicts
# - Insufficient memory
# - Missing dependencies
```

#### 2. Poor Performance
```bash
# Check resource usage
docker stats

# Monitor metrics
curl http://localhost:9090/metrics | grep rag_

# Common solutions:
# - Increase memory allocation
# - Scale horizontally
# - Optimize chunk size
# - Enable caching
```

#### 3. Quality Issues
```bash
# Check evaluation metrics
curl http://localhost:8000/evaluation/summary

# Common causes:
# - Poor document quality
# - Incorrect chunking
# - Suboptimal retrieval parameters
# - LLM configuration issues
```

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
export DEBUG_MODE=true

# Restart services
docker-compose restart rag-api

# View detailed logs
docker-compose logs -f rag-api
```

### Performance Profiling
```bash
# Enable profiling
export ENABLE_PROFILING=true
export PROFILE_OUTPUT_DIR=./profiles

# Analyze profiles
python -m cProfile -o profile.stats solution.py
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
```

## Maintenance

### Regular Tasks

#### Daily
- Monitor system health and metrics
- Check error logs for issues
- Verify backup completion

#### Weekly
- Review performance metrics
- Update security patches
- Clean up old logs and backups

#### Monthly
- Update dependencies
- Review and optimize configuration
- Conduct security audit
- Performance testing

### Update Procedure
```bash
# 1. Backup current system
./backup.sh

# 2. Pull latest changes
git pull origin main

# 3. Update dependencies
docker-compose pull

# 4. Deploy updates
docker-compose up -d

# 5. Verify deployment
curl http://localhost:8000/health

# 6. Monitor for issues
docker-compose logs -f rag-api
```

## Support and Resources

### Documentation
- API Documentation: `http://localhost:8000/docs`
- Metrics Dashboard: `http://localhost:3000`
- Health Status: `http://localhost:8000/health`

### Monitoring Endpoints
- `/health` - System health check
- `/metrics` - Prometheus metrics
- `/evaluation/summary` - Quality metrics
- `/documents` - Document inventory

### Log Locations
- Application logs: `./logs/`
- Docker logs: `docker-compose logs`
- System logs: `/var/log/`

### Performance Benchmarks
- Response time: < 2 seconds (95th percentile)
- Throughput: 100+ concurrent requests
- Accuracy: > 0.8 average RAGAS scores
- Uptime: 99.9% availability target

This deployment guide provides comprehensive instructions for running the Production RAG System in various environments with proper monitoring, security, and maintenance procedures.