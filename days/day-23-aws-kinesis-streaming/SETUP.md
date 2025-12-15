# Day 23: AWS Kinesis & Streaming - Setup Guide

## Quick Start (5 minutes)

```bash
# 1. Navigate to day 23 directory
cd days/day-23-aws-kinesis-streaming

# 2. Run automated setup
./setup.sh

# 3. Start interactive demo
python demo.py
```

## Manual Setup

### Prerequisites
- Docker and Docker Compose
- Python 3.8+
- 4GB+ available RAM

### Step-by-Step Setup

1. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env if needed (defaults work for local development)
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start Services**
   ```bash
   docker-compose up -d
   ```

4. **Initialize AWS Resources**
   ```bash
   # Wait 30 seconds for LocalStack to start
   python scripts/setup.py
   ```

### Verify Setup

```bash
# Check services are running
docker-compose ps

# Test Kinesis connection
python -c "
import boto3
import os
from dotenv import load_dotenv
load_dotenv()

kinesis = boto3.client('kinesis', endpoint_url=os.getenv('AWS_ENDPOINT_URL'))
print('Streams:', kinesis.list_streams()['StreamNames'])
"
```

## Services

- **LocalStack**: http://localhost:4566 (AWS services simulation)
- **Grafana**: http://localhost:3000 (admin/admin)
- **Prometheus**: http://localhost:9090

## Troubleshooting

### LocalStack not starting
```bash
docker-compose down
docker-compose up -d localstack
docker-compose logs localstack
```

### Permission errors
```bash
chmod +x setup.sh scripts/init-aws.sh
```

### Port conflicts
Edit `docker-compose.yml` to change port mappings if needed.

## What's Included

- ✅ LocalStack AWS simulation environment
- ✅ Kinesis Data Streams setup
- ✅ S3 bucket for data storage
- ✅ DynamoDB for customer profiles
- ✅ Sample data generation
- ✅ Interactive demo script
- ✅ Monitoring with Grafana/Prometheus
- ✅ Automated setup and teardown