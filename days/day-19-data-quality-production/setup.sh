#!/bin/bash

# Day 19: Data Quality in Production - Setup Script
# Sets up the complete development environment for QualityFirst Corp data quality system

set -e

echo "🚀 Setting up Day 19: Data Quality in Production Environment"
echo "==========================================================="

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
sleep 15

# Check if PostgreSQL is ready
until docker-compose exec postgres pg_isready -U quality_user -d qualityfirst_data; do
    echo "Waiting for PostgreSQL..."
    sleep 2
done

echo "✅ PostgreSQL is ready!"

# Initialize Great Expectations
echo "📊 Initializing Great Expectations..."
docker-compose exec great_expectations python -c "
import great_expectations as gx
context = gx.get_context()
print('Great Expectations initialized successfully!')
"

# Run sample data quality validation
echo "🧪 Running sample data quality validation..."
docker-compose exec great_expectations python -c "
from exercise import create_customer_data_expectations, create_transaction_data_expectations
import great_expectations as gx

context = gx.get_context()

# Create expectation suites
customer_suite = create_customer_data_expectations()
transaction_suite = create_transaction_data_expectations()

print('Sample expectation suites created successfully!')
print(f'Customer suite: {len(customer_suite.expectations)} expectations')
print(f'Transaction suite: {len(transaction_suite.expectations)} expectations')
"

echo ""
echo "🎉 Setup complete! Your data quality environment is ready."
echo ""
echo "📋 Next steps:"
echo "1. Access Great Expectations container: docker-compose exec great_expectations bash"
echo "2. Run quality validations: python exercise.py"
echo "3. View Grafana dashboard: http://localhost:3000 (admin/admin)"
echo "4. Stop environment: docker-compose down"
echo ""
echo "🔗 Database connection:"
echo "   Host: localhost"
echo "   Port: 5432"
echo "   Database: qualityfirst_data"
echo "   Username: quality_user"
echo "   Password: quality_password"
echo ""
echo "📖 Example commands:"
echo "   docker-compose exec great_expectations python exercise.py"
echo "   docker-compose exec postgres psql -U quality_user -d qualityfirst_data"