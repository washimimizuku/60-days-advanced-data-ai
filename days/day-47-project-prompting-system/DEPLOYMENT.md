# Day 47: Advanced Prompting System - Deployment Guide

## Overview

This guide provides comprehensive instructions for deploying the Advanced Prompting System in production environments, including local development, staging, and production deployments.

## Prerequisites

### System Requirements
- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 50GB+ available space
- **Network**: Stable internet connection for LLM API calls

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Python 3.11+
- PostgreSQL 15+ (if not using Docker)
- Redis 7+ (if not using Docker)

### API Keys Required
- OpenAI API key (required)
- Anthropic API key (optional)
- Cohere API key (optional)

## Quick Start (Development)

### 1. Clone and Setup
```bash
git clone <repository-url>
cd day-47-project-prompting-system

# Copy environment configuration
cp .env.example .env

# Edit .env with your API keys and settings
nano .env
```

### 2. Start with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f prompting-system

# Check service status
docker-compose ps
```

### 3. Verify Installation
```bash
# Health check
curl http://localhost:8000/health

# Register a test user
curl -X POST http://localhost:8000/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "securepassword123"
  }'

# Login and get token
curl -X POST "http://localhost:8000/auth/login?username=testuser&password=securepassword123"
```

## Production Deployment

### Option 1: Docker Compose (Recommended for Small-Medium Scale)

#### 1. Production Environment Setup
```bash
# Create production directory
mkdir -p /opt/prompting-system
cd /opt/prompting-system

# Copy application files
cp -r /path/to/source/* .

# Create production environment file
cp .env.example .env.production
```

#### 2. Configure Production Environment
```bash
# Edit production configuration
nano .env.production
```

Key production settings:
```bash
# Security
JWT_SECRET_KEY=your-super-secure-jwt-secret-here
BCRYPT_ROUNDS=12

# Database (use external managed database in production)
DATABASE_URL=postgresql://user:pass@your-db-host:5432/prompting_system

# API Keys
OPENAI_API_KEY=your-production-openai-key
ANTHROPIC_API_KEY=your-production-anthropic-key

# Performance
API_WORKERS=4
MAX_CONCURRENT_REQUESTS=200
CONNECTION_POOL_SIZE=50

# Security
ENABLE_SECURITY_SCANNING=true
SECURITY_SCAN_THRESHOLD=0.9
BLOCK_SUSPICIOUS_USERS=true

# Monitoring
SENTRY_DSN=your-sentry-dsn
ENABLE_PROMETHEUS_METRICS=true

# CORS and Security
CORS_ORIGINS=https://yourdomain.com
TRUSTED_HOSTS=yourdomain.com
ENABLE_DOCS=false
```

#### 3. Start Production Services
```bash
# Use production compose file
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify all services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

### Option 2: Kubernetes Deployment (Recommended for Large Scale)

#### 1. Create Kubernetes Manifests
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: prompting-system
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prompting-system-config
  namespace: prompting-system
data:
  DATABASE_URL: "postgresql://user:pass@postgres:5432/prompting_system"
  REDIS_URL: "redis://redis:6379/0"
  API_WORKERS: "4"
  MAX_CONCURRENT_REQUESTS: "200"
---
# secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: prompting-system-secrets
  namespace: prompting-system
type: Opaque
stringData:
  JWT_SECRET_KEY: "your-jwt-secret"
  OPENAI_API_KEY: "your-openai-key"
  ANTHROPIC_API_KEY: "your-anthropic-key"
---
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: prompting-system
  namespace: prompting-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: prompting-system
  template:
    metadata:
      labels:
        app: prompting-system
    spec:
      containers:
      - name: prompting-system
        image: prompting-system:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: prompting-system-config
        - secretRef:
            name: prompting-system-secrets
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
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
---
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: prompting-system-service
  namespace: prompting-system
spec:
  selector:
    app: prompting-system
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: prompting-system-ingress
  namespace: prompting-system
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - api.yourdomain.com
    secretName: prompting-system-tls
  rules:
  - host: api.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: prompting-system-service
            port:
              number: 80
```

#### 2. Deploy to Kubernetes
```bash
# Apply all manifests
kubectl apply -f k8s/

# Check deployment status
kubectl get pods -n prompting-system
kubectl get services -n prompting-system

# View logs
kubectl logs -f deployment/prompting-system -n prompting-system

# Scale deployment
kubectl scale deployment prompting-system --replicas=5 -n prompting-system
```

## Database Setup

### PostgreSQL Configuration
```sql
-- Create database and user
CREATE DATABASE prompting_system;
CREATE USER prompting_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE prompting_system TO prompting_user;

-- Connect to the database
\c prompting_system;

-- Create tables (run from init.sql)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT true,
    last_login TIMESTAMP
);

CREATE TABLE requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    prompt TEXT NOT NULL,
    response TEXT,
    model_used VARCHAR(50),
    processing_time FLOAT,
    cost_estimate FLOAT,
    security_level VARCHAR(20),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB
);

CREATE TABLE security_events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    event_type VARCHAR(50) NOT NULL,
    threat_level VARCHAR(20) NOT NULL,
    description TEXT,
    blocked BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    details JSONB
);

-- Create indexes for performance
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_requests_user_id ON requests(user_id);
CREATE INDEX idx_requests_created_at ON requests(created_at);
CREATE INDEX idx_security_events_user_id ON security_events(user_id);
CREATE INDEX idx_security_events_created_at ON security_events(created_at);
```

### Redis Configuration
```bash
# Redis configuration for production
# /etc/redis/redis.conf

# Security
requirepass your_redis_password
bind 127.0.0.1

# Performance
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Logging
loglevel notice
logfile /var/log/redis/redis-server.log
```

## Monitoring Setup

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prompting-system'
    static_configs:
      - targets: ['prompting-system:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Advanced Prompting System",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Security Threats",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(security_threats_total[5m])",
            "legendFormat": "Threats/sec"
          }
        ]
      },
      {
        "title": "Active Users",
        "type": "singlestat",
        "targets": [
          {
            "expr": "active_users",
            "legendFormat": "Users"
          }
        ]
      }
    ]
  }
}
```

## SSL/TLS Configuration

### Nginx SSL Configuration
```nginx
# nginx/nginx.conf
server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req zone=api burst=20 nodelay;

    location / {
        proxy_pass http://prompting-system:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

## Backup and Recovery

### Database Backup
```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="prompting_system"

# Create backup
pg_dump -h localhost -U prompting_user -d $DB_NAME > $BACKUP_DIR/backup_$DATE.sql

# Compress backup
gzip $BACKUP_DIR/backup_$DATE.sql

# Remove backups older than 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: backup_$DATE.sql.gz"
```

### Automated Backup with Cron
```bash
# Add to crontab
0 2 * * * /opt/prompting-system/backup.sh >> /var/log/backup.log 2>&1
```

## Security Hardening

### Application Security
```bash
# 1. Use non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# 2. Limit file permissions
RUN chmod 600 .env
RUN chmod 700 logs/

# 3. Use secrets management
# Store sensitive data in Kubernetes secrets or AWS Secrets Manager
```

### Network Security
```bash
# 1. Firewall rules (UFW example)
ufw allow 22/tcp    # SSH
ufw allow 80/tcp    # HTTP
ufw allow 443/tcp   # HTTPS
ufw deny 8000/tcp   # Block direct access to app
ufw enable

# 2. Fail2ban for SSH protection
apt install fail2ban
systemctl enable fail2ban
```

## Performance Tuning

### Application Tuning
```python
# Increase worker processes for production
API_WORKERS=4

# Optimize database connections
CONNECTION_POOL_SIZE=50
MAX_CONCURRENT_REQUESTS=200

# Enable caching
CACHE_TTL_SECONDS=3600
ENABLE_REQUEST_CACHING=true
```

### Database Tuning
```sql
-- PostgreSQL performance tuning
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;

-- Reload configuration
SELECT pg_reload_conf();
```

## Troubleshooting

### Common Issues

#### 1. Application Won't Start
```bash
# Check logs
docker-compose logs prompting-system

# Common causes:
# - Missing environment variables
# - Database connection issues
# - Port conflicts
```

#### 2. Database Connection Issues
```bash
# Test database connection
docker-compose exec prompting-system python -c "
import asyncpg
import asyncio
async def test():
    conn = await asyncpg.connect('postgresql://user:pass@postgres:5432/prompting_system')
    await conn.close()
    print('Database connection successful')
asyncio.run(test())
"
```

#### 3. High Memory Usage
```bash
# Monitor memory usage
docker stats

# Reduce worker processes if needed
API_WORKERS=2
MAX_CONCURRENT_REQUESTS=100
```

#### 4. Slow Response Times
```bash
# Check metrics
curl http://localhost:8000/metrics | grep request_duration

# Optimize database queries
# Enable caching
# Scale horizontally
```

### Health Checks
```bash
# Application health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready

# Redis health
docker-compose exec redis redis-cli ping
```

## Scaling Strategies

### Horizontal Scaling
```bash
# Scale with Docker Compose
docker-compose up -d --scale prompting-system=3

# Scale with Kubernetes
kubectl scale deployment prompting-system --replicas=5
```

### Load Balancing
```nginx
# Nginx load balancer configuration
upstream prompting_backend {
    server prompting-system-1:8000;
    server prompting-system-2:8000;
    server prompting-system-3:8000;
}

server {
    location / {
        proxy_pass http://prompting_backend;
    }
}
```

## Maintenance

### Regular Maintenance Tasks
```bash
# 1. Update dependencies
pip install -r requirements.txt --upgrade

# 2. Database maintenance
VACUUM ANALYZE;
REINDEX DATABASE prompting_system;

# 3. Log rotation
logrotate /etc/logrotate.d/prompting-system

# 4. Security updates
apt update && apt upgrade -y
```

### Monitoring Checklist
- [ ] Application health endpoints responding
- [ ] Database performance metrics within limits
- [ ] Security alerts configured and working
- [ ] Backup processes running successfully
- [ ] SSL certificates not expiring soon
- [ ] Resource usage within acceptable ranges

This deployment guide provides a comprehensive approach to deploying the Advanced Prompting System in production environments with proper security, monitoring, and scalability considerations.