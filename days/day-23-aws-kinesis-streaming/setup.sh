#!/bin/bash

echo "🚀 Setting up Day 23: AWS Kinesis & Streaming"

# Copy environment file
if [ ! -f .env ]; then
    cp .env.example .env
    echo "📝 Created .env file from template"
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Run setup script
echo "🔧 Running setup script..."
python scripts/setup.py

echo "✅ Setup complete!"
echo ""
echo "🎯 Next steps:"
echo "1. Run 'python demo.py' for interactive demo"
echo "2. Run 'python exercise.py' for the main exercise"
echo "3. Access Grafana at http://localhost:3000 (admin/admin)"
echo "4. Access Prometheus at http://localhost:9090"