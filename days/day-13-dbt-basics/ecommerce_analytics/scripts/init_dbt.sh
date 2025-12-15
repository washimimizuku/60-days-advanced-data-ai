#!/bin/bash
# Initialize dbt project

set -e

echo "=== Initializing dbt Project ==="

# Set environment variables
export DBT_PROFILES_DIR=$(pwd)/..

# Install dbt packages
echo "Installing dbt packages..."
dbt deps

# Load sample data
echo "Loading sample data..."
dbt seed

# Run staging models
echo "Running staging models..."
dbt run --models staging

# Run all models
echo "Running all models..."
dbt run

# Run tests
echo "Running tests..."
dbt test

# Generate documentation
echo "Generating documentation..."
dbt docs generate

echo ""
echo "=== dbt Setup Complete ==="
echo "Documentation: dbt docs serve"
echo "View models: dbt run --models model_name"
echo "Run tests: dbt test --models model_name"
echo ""