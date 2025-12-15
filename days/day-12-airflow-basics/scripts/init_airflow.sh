#!/bin/bash
# Initialize Airflow environment

set -e

echo "=== Initializing Airflow Environment ==="

# Set AIRFLOW_UID for Docker
export AIRFLOW_UID=$(id -u)
echo "AIRFLOW_UID=$AIRFLOW_UID" > .env

# Create necessary directories
echo "Creating directories..."
mkdir -p ./dags ./logs ./plugins ./config ./data
chmod 755 ./dags ./logs ./plugins ./config ./data

# Initialize Airflow database
echo "Initializing Airflow database..."
docker-compose up airflow-init

# Start Airflow services
echo "Starting Airflow services..."
docker-compose up -d

# Wait for services to be ready
echo "Waiting for services to start..."
sleep 30

# Check service status
echo "Checking service status..."
docker-compose ps

echo ""
echo "=== Airflow Setup Complete ==="
echo "Airflow UI: http://localhost:8080"
echo "Username: admin"
echo "Password: admin"
echo ""
echo "To stop Airflow: docker-compose down"
echo "To view logs: docker-compose logs -f"