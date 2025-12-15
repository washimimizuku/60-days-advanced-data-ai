#!/bin/bash
# Day 21: Testing Strategies - Setup Script

set -e

echo "🎯 Setting up Day 21: Testing Strategies Environment"
echo "=================================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs test_results test_baselines

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating environment file..."
    cp .env.example .env
    echo "✅ Created .env file. Please review and update as needed."
fi

# Build and start services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check if PostgreSQL is ready
echo "🔍 Checking PostgreSQL connection..."
docker-compose exec -T postgres pg_isready -U testuser -d testing_db

# Install Python dependencies in container
echo "📦 Installing Python dependencies..."
docker-compose exec -T testing-env pip install -r requirements.txt

# Run initial tests to verify setup
echo "🧪 Running setup verification tests..."
docker-compose exec -T testing-env python -m pytest tests/unit/test_setup.py -v

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 Quick Start Commands:"
echo "  • Run all tests: docker-compose exec testing-env pytest"
echo "  • Run unit tests: docker-compose exec testing-env pytest tests/unit/"
echo "  • Run with coverage: docker-compose exec testing-env pytest --cov=."
echo "  • Access Jupyter: http://localhost:8888"
echo "  • Access Grafana: http://localhost:3000 (admin/admin123)"
echo "  • Access Prometheus: http://localhost:9090"
echo ""
echo "📚 Documentation:"
echo "  • README.md - Complete guide"
echo "  • exercise.py - Hands-on exercises"
echo "  • solution.py - Complete solutions"
echo ""