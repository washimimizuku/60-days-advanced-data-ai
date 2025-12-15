#!/bin/bash

# DataCorp Governed Platform Setup Script

echo "🚀 Setting up DataCorp Governed Data Platform..."

# Check prerequisites
echo "📋 Checking prerequisites..."
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Copy environment template if .env doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating environment configuration..."
    cp .env.example .env
    
    # Generate Fernet key for Airflow
    echo "🔐 Generating Airflow Fernet key..."
    python3 -c "from cryptography.fernet import Fernet; print('AIRFLOW_FERNET_KEY=' + Fernet.generate_key().decode())" >> .env
fi

# Create required directories
echo "📁 Creating directory structure..."
mkdir -p {airflow/{dags,logs,plugins},dbt/{models,tests,macros},governance/{policies,scripts,logs},monitoring/{prometheus,grafana,alerts},sql,sample_data}

# Set proper permissions
echo "🔒 Setting permissions..."
chmod -R 755 airflow/
chmod -R 755 governance/scripts/
chmod +x scripts/setup.sh

# Start core services first
echo "🗄️ Starting core services..."
docker-compose up -d postgres airflow-db redis

# Wait for databases to be ready
echo "⏳ Waiting for databases to initialize..."
sleep 30

# Check database health
echo "🏥 Checking database health..."
docker-compose exec postgres pg_isready -U platform_user || {
    echo "❌ PostgreSQL is not ready. Check logs: docker-compose logs postgres"
    exit 1
}

# Initialize Airflow database
echo "✈️ Initializing Airflow..."
docker-compose run --rm airflow-webserver airflow db init

# Create Airflow admin user
echo "👤 Creating Airflow admin user..."
docker-compose run --rm airflow-webserver airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@datacorp.com \
    --password admin

# Start all services
echo "🏃 Starting all services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 60

# Health check
echo "🏥 Running health checks..."
services=("airflow-webserver:8080/health" "grafana:3000/api/health" "prometheus:9090/-/healthy")
for service in "${services[@]}"; do
    if curl -f "http://localhost:${service#*:}" &> /dev/null; then
        echo "✅ ${service%:*} is healthy"
    else
        echo "⚠️ ${service%:*} may not be ready yet"
    fi
done

# Load sample data
echo "📊 Loading sample data..."
docker-compose exec postgres psql -U platform_user -d datacorp_platform -f /docker-entrypoint-initdb.d/init.sql

# Setup dbt
echo "🔧 Setting up dbt..."
docker-compose exec airflow-webserver dbt deps --project-dir /opt/dbt || echo "⚠️ dbt deps failed - this is normal on first run"

# Enable DAGs
echo "📋 Enabling Airflow DAGs..."
sleep 10
docker-compose exec airflow-webserver airflow dags unpause governance_daily_pipeline || echo "⚠️ DAG not found yet - will be available after restart"

echo ""
echo "✅ DataCorp Governed Platform setup complete!"
echo ""
echo "🌐 Access URLs:"
echo "   Airflow: http://localhost:8080 (admin/admin)"
echo "   Grafana: http://localhost:3000 (admin/admin)"
echo "   DataHub: http://localhost:8090"
echo "   Prometheus: http://localhost:9090"
echo ""
echo "📚 Next steps:"
echo "   1. Visit Airflow to see governance DAGs"
echo "   2. Check Grafana for governance dashboards"
echo "   3. Review governance policies in ./governance/policies/"
echo "   4. Run: docker-compose exec airflow-webserver airflow dags trigger governance_daily_pipeline"
echo ""
echo "🆘 Troubleshooting:"
echo "   - Check logs: docker-compose logs <service-name>"
echo "   - Health check: docker-compose ps"
echo "   - Restart: docker-compose restart"