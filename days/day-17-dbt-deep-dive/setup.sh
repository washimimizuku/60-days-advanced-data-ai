#!/bin/bash

# Day 17: dbt Deep Dive - Setup Script
# Sets up the complete dbt development environment

set -e

echo "🚀 Setting up dbt Deep Dive environment..."

# Check prerequisites
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.example .env
fi

# Start the database
echo "🐘 Starting PostgreSQL database..."
docker-compose up -d postgres

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
sleep 10

# Install dbt packages (if dbt_project exists)
if [ -d "dbt_project" ]; then
    echo "📦 Installing dbt packages..."
    docker-compose run --rm dbt dbt deps --project-dir dbt_project
fi

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run 'docker-compose run --rm dbt bash' to enter dbt environment"
echo "2. Navigate to dbt_project directory"
echo "3. Run 'dbt run' to execute models"
echo "4. Run 'dbt test' to validate data quality"
echo ""
echo "📊 Database connection:"
echo "Host: localhost:5432"
echo "Database: datacorp_dev"
echo "User: dbt_user"
echo "Password: dbt_password"