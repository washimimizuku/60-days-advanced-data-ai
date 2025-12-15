#!/bin/bash

# Day 20: Data Observability - Setup Script
# This script sets up the complete observability environment

set -e

echo "🚀 Setting up Data Observability Environment..."
echo "================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install it and try again."
    exit 1
fi

echo "✅ Docker is running"

# Create environment file
echo "📝 Creating environment configuration..."
cat > .env << EOF
# Database Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=observability_db
DB_USER=obs_user
DB_PASSWORD=obs_password

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_ADMIN_PASSWORD=admin

# Application Configuration
JUPYTER_PORT=8888
JUPYTER_TOKEN=observability
EOF

echo "✅ Environment file created"

# Start the services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check if PostgreSQL is ready
echo "🔍 Checking PostgreSQL connection..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if docker-compose exec -T postgres pg_isready -U obs_user -d observability_db > /dev/null 2>&1; then
        echo "✅ PostgreSQL is ready"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        echo "❌ PostgreSQL failed to start after $max_attempts attempts"
        exit 1
    fi
    
    echo "   Attempt $attempt/$max_attempts - waiting for PostgreSQL..."
    sleep 2
    ((attempt++))
done

# Check if Grafana is ready
echo "🔍 Checking Grafana connection..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -s http://localhost:3000/api/health > /dev/null 2>&1; then
        echo "✅ Grafana is ready"
        break
    fi
    
    if [ $attempt -eq $max_attempts ]; then
        echo "❌ Grafana failed to start after $max_attempts attempts"
        exit 1
    fi
    
    echo "   Attempt $attempt/$max_attempts - waiting for Grafana..."
    sleep 2
    ((attempt++))
done

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if command -v pip3 &> /dev/null; then
    pip3 install -r requirements.txt
elif command -v pip &> /dev/null; then
    pip install -r requirements.txt
else
    echo "⚠️ pip not found. Please install Python dependencies manually:"
    echo "   pip install -r requirements.txt"
fi

# Run the demo
echo "🎯 Running observability demo..."
python3 demo.py

echo ""
echo "🎉 Setup Complete!"
echo "=================="
echo ""
echo "🔗 Access Points:"
echo "   • Grafana Dashboard: http://localhost:3000 (admin/admin)"
echo "   • Prometheus Metrics: http://localhost:9090"
echo "   • Jupyter Notebooks: http://localhost:8888 (token: observability)"
echo "   • PostgreSQL Database: localhost:5432 (obs_user/obs_password)"
echo ""
echo "📚 Next Steps:"
echo "   1. Explore the Grafana dashboards for real-time monitoring"
echo "   2. Check Prometheus for custom metrics"
echo "   3. Run 'python3 demo.py' to see the observability system in action"
echo "   4. Examine the Jupyter notebooks for interactive analysis"
echo ""
echo "🛑 To stop services: docker-compose down"
echo "🔄 To restart services: docker-compose restart"