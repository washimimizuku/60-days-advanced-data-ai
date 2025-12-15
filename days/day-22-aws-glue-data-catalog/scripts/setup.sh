#!/bin/bash
# Day 22: AWS Glue & Data Catalog - Setup Script

set -e

echo "🚀 Setting up Day 22: AWS Glue & Data Catalog Environment"
echo "========================================================"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Create environment file
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
fi

# Start services
echo "🐳 Starting LocalStack and development environment..."
docker-compose up -d

# Wait for LocalStack
echo "⏳ Waiting for LocalStack to be ready..."
sleep 30

# Initialize AWS resources
echo "🔧 Initializing AWS resources..."
docker-compose exec -T glue-dev python scripts/init_aws_resources.py

# Generate sample data
echo "📊 Generating sample data..."
docker-compose exec -T glue-dev python scripts/generate_sample_data.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Quick Start Commands:"
echo "  • Interactive development: docker-compose exec glue-dev bash"
echo "  • Jupyter notebook: http://localhost:8888"
echo "  • Run demo: docker-compose exec glue-dev python demo.py"
echo "  • LocalStack dashboard: http://localhost:4566"