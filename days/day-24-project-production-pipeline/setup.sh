#!/bin/bash

echo "🚀 Setting up Day 24: Production Pipeline"

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file from template"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Initialize Airflow database
echo "🗄️ Initializing Airflow database..."
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init

# Create Airflow admin user
echo "👤 Creating Airflow admin user..."
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 60

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run 'python demo.py' for interactive demo"
echo "2. Access Airflow at http://localhost:8080 (admin/admin)"
echo "3. Access Grafana at http://localhost:3000 (admin/admin)"
echo "4. Access Prometheus at http://localhost:9090"