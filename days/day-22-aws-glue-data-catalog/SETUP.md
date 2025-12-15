# Day 22: AWS Glue & Data Catalog - Setup Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ (if running locally)
- 4GB+ RAM available
- 2GB+ free disk space

### Automated Setup
```bash
# 1. Navigate to directory
cd day-22-aws-glue-data-catalog

# 2. Run automated setup
./scripts/setup.sh

# 3. Verify setup
docker-compose exec glue-dev python -c "import boto3; print('✅ Environment ready')"
```

## 🛠️ Manual Setup

### 1. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit configuration if needed
nano .env
```

### 2. Start Services
```bash
# Start LocalStack and development environment
docker-compose up -d

# Check service status
docker-compose ps
```

### 3. Initialize AWS Resources
```bash
# Initialize S3 buckets and IAM roles
docker-compose exec glue-dev python scripts/init_aws_resources.py

# Generate sample data
docker-compose exec glue-dev python scripts/generate_sample_data.py
```

## 📊 Running the Exercise

### Interactive Demo
```bash
# Run complete demonstration
docker-compose exec glue-dev python demo.py
```

### Step-by-Step Exercise
```bash
# Access development environment
docker-compose exec glue-dev bash

# Run exercise components
python exercise.py
```

### Jupyter Notebook
```bash
# Access Jupyter at http://localhost:8888
# No token required for local development
```

## 🔧 Development Environment

### Local Development
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export AWS_ENDPOINT_URL=http://localhost:4566
export USE_LOCALSTACK=true
```

### AWS CLI Configuration
```bash
# Configure AWS CLI for LocalStack
aws configure set aws_access_key_id test
aws configure set aws_secret_access_key test
aws configure set region us-east-1

# Test LocalStack connection
aws --endpoint-url=http://localhost:4566 s3 ls
```

## 📈 Monitoring and Verification

### LocalStack Dashboard
- **URL**: http://localhost:4566
- **Health Check**: http://localhost:4566/_localstack/health

### Service Verification
```bash
# Check S3 buckets
docker-compose exec glue-dev aws --endpoint-url=http://localhost:4566 s3 ls

# List Glue databases
docker-compose exec glue-dev aws --endpoint-url=http://localhost:4566 glue get-databases

# Check sample data
docker-compose exec glue-dev aws --endpoint-url=http://localhost:4566 s3 ls s3://serverlessdata-datalake/raw/ --recursive
```

## 🛠️ Configuration Options

### Environment Variables (.env)
```bash
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCOUNT_ID=123456789012

# LocalStack
LOCALSTACK_ENDPOINT=http://localhost:4566
USE_LOCALSTACK=true

# Glue Configuration
GLUE_DATABASE_NAME=serverlessdata_analytics
GLUE_IAM_ROLE=arn:aws:iam::123456789012:role/GlueServiceRole

# S3 Buckets
S3_DATA_BUCKET=serverlessdata-datalake
S3_SCRIPTS_BUCKET=serverlessdata-glue-scripts
S3_ATHENA_RESULTS=serverlessdata-athena-results
```

### Docker Compose Services
- **localstack**: AWS services simulation
- **glue-dev**: Development environment with AWS CLI
- **jupyter**: Interactive notebook environment

## 📁 Project Structure

```
day-22-aws-glue-data-catalog/
├── README.md                    # Main documentation
├── SETUP.md                     # This setup guide
├── requirements.txt             # Python dependencies
├── docker-compose.yml           # Docker services
├── Dockerfile                   # Development environment
├── .env.example                 # Environment template
│
├── scripts/                     # Setup and utility scripts
│   ├── setup.sh               # Automated setup
│   ├── init_aws_resources.py  # AWS resource initialization
│   └── generate_sample_data.py # Sample data generator
│
├── data/                        # Generated sample data
├── notebooks/                   # Jupyter notebooks
│
├── exercise.py                  # Hands-on exercises
├── solution.py                  # Complete solutions
├── demo.py                      # Interactive demonstration
└── quiz.md                      # Knowledge assessment
```

## 🔍 Troubleshooting

### Common Issues

#### LocalStack Not Starting
```bash
# Check Docker status
docker --version
docker-compose --version

# Restart LocalStack
docker-compose restart localstack

# Check logs
docker-compose logs localstack
```

#### AWS Services Not Available
```bash
# Wait for LocalStack to be ready
docker-compose exec localstack curl http://localhost:4566/_localstack/health

# Reinitialize resources
docker-compose exec glue-dev python scripts/init_aws_resources.py
```

#### Permission Issues
```bash
# Fix script permissions
chmod +x scripts/setup.sh

# Fix directory ownership
sudo chown -R $USER:$USER .
```

#### Memory Issues
```bash
# Check available memory
free -h

# Reduce LocalStack services
# Edit docker-compose.yml and remove unused services
```

### Verification Commands
```bash
# Test environment
docker-compose exec glue-dev python -c "
import boto3
import os
print('✅ Python environment ready')
print(f'AWS Region: {os.getenv(\"AWS_REGION\")}')
print(f'LocalStack: {os.getenv(\"USE_LOCALSTACK\")}')
"

# Test AWS connectivity
docker-compose exec glue-dev python -c "
import boto3
s3 = boto3.client('s3', endpoint_url='http://localstack:4566')
buckets = s3.list_buckets()
print(f'✅ S3 buckets: {len(buckets[\"Buckets\"])}')
"
```

## 📚 Additional Resources

### AWS Glue Documentation
- [AWS Glue Developer Guide](https://docs.aws.amazon.com/glue/)
- [Glue ETL Programming Guide](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-programming.html)
- [Data Catalog API Reference](https://docs.aws.amazon.com/glue/latest/dg/aws-glue-api-catalog.html)

### LocalStack Resources
- [LocalStack Documentation](https://docs.localstack.cloud/)
- [AWS CLI with LocalStack](https://docs.localstack.cloud/user-guide/integrations/aws-cli/)

### Best Practices
- [Glue Best Practices](https://docs.aws.amazon.com/glue/latest/dg/best-practices.html)
- [Athena Performance Tuning](https://docs.aws.amazon.com/athena/latest/ug/performance-tuning.html)
- [Data Lake Architecture](https://aws.amazon.com/big-data/datalakes-and-analytics/)

## 🆘 Getting Help

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
docker-compose exec glue-dev python demo.py --verbose
```

### Support Checklist
- [ ] Docker services running (`docker-compose ps`)
- [ ] LocalStack healthy (`curl http://localhost:4566/_localstack/health`)
- [ ] AWS resources initialized (`aws --endpoint-url=http://localhost:4566 s3 ls`)
- [ ] Sample data generated (`aws --endpoint-url=http://localhost:4566 s3 ls s3://serverlessdata-datalake/raw/`)
- [ ] Python environment working (`python -c "import boto3; print('OK')"`)

---

## ✅ Verification Checklist

After setup, verify these components work:

- [ ] LocalStack services running
- [ ] S3 buckets created and accessible
- [ ] Sample data uploaded to S3
- [ ] Glue database and tables defined
- [ ] Python environment with all dependencies
- [ ] Demo script runs successfully
- [ ] Jupyter notebook accessible

🎉 **Setup Complete!** You're ready to explore serverless ETL with AWS Glue and Data Catalog.