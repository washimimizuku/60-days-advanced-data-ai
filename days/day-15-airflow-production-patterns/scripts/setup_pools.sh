#!/bin/bash

# Setup Airflow Pools for Resource Management
# This script creates all required pools for the TechCorp data platform

echo "🏊 Setting up Airflow pools for resource management..."

# Check if Airflow is running
if ! docker-compose exec airflow-webserver airflow version &> /dev/null; then
    echo "❌ Airflow is not running. Please start with: docker-compose up -d"
    exit 1
fi

# Create pools
echo "Creating fast_processing_pool..."
docker-compose exec airflow-webserver airflow pools set fast_processing_pool 3 "High-priority, small data processing"

echo "Creating standard_pool..."
docker-compose exec airflow-webserver airflow pools set standard_pool 10 "Standard data processing"

echo "Creating batch_processing_pool..."
docker-compose exec airflow-webserver airflow pools set batch_processing_pool 2 "Large data batch processing"

echo "Creating monitoring_pool..."
docker-compose exec airflow-webserver airflow pools set monitoring_pool 5 "Monitoring and alerting tasks"

echo "Creating external_api_pool..."
docker-compose exec airflow-webserver airflow pools set external_api_pool 5 "External API calls"

# Verify pools were created
echo ""
echo "✅ Pools created successfully!"
echo ""
echo "📊 Current pool status:"
docker-compose exec airflow-webserver airflow pools list

echo ""
echo "🎯 Pool setup complete! You can now run the TechCorp data pipelines."