#!/bin/bash

# Day 18: dbt Advanced - Setup Script
# Sets up the complete development environment for InnovateCorp Analytics Toolkit

set -e

echo "🚀 Setting up Day 18: dbt Advanced Development Environment"
echo "=========================================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Build and start the environment
echo "📦 Building Docker containers..."
docker-compose build

echo "🔄 Starting services..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 10

# Check if PostgreSQL is ready
until docker-compose exec postgres pg_isready -U dbt_user -d innovatecorp_analytics; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done

echo "✅ PostgreSQL is ready!"

# Install dbt dependencies
echo "📚 Installing dbt dependencies..."
docker-compose exec dbt dbt deps || echo "No packages to install"

# Test dbt connection
echo "🔗 Testing dbt connection..."
docker-compose exec dbt dbt debug

# Run dbt models
echo "🏗️ Running dbt models..."
docker-compose exec dbt dbt run

# Run dbt tests
echo "🧪 Running dbt tests..."
docker-compose exec dbt dbt test || echo "Some tests may fail on first run"

echo ""
echo "🎉 Setup complete! Your dbt Advanced environment is ready."
echo ""
echo "📋 Next steps:"
echo "1. Access the dbt container: docker-compose exec dbt bash"
echo "2. Run dbt commands: dbt run, dbt test, dbt docs generate"
echo "3. View logs: docker-compose logs -f"
echo "4. Stop environment: docker-compose down"
echo ""
echo "🔗 Database connection:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: innovatecorp_analytics"
echo "   Username: dbt_user"
echo "   Password: dbt_password"
echo ""
echo "📖 Example commands:"
echo "   docker-compose exec dbt dbt run --models example_attribution_analysis"
echo "   docker-compose exec dbt dbt test --models example_cohort_analysis"
echo "   docker-compose exec dbt dbt docs generate && dbt docs serve"