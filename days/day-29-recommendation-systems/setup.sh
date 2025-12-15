#!/bin/bash

# Day 29: Recommendation Systems Setup Script
# Sets up complete recommendation system infrastructure with Docker services

set -e

echo "🚀 Setting up Day 29: Recommendation Systems Environment"
echo "======================================================"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if Docker Compose is available
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose."
    exit 1
fi

echo "✅ Docker is running"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed logs models grafana/dashboards grafana/datasources

# Create init.sql for PostgreSQL
echo "📄 Creating database initialization script..."
cat > init.sql << 'EOF'
-- Initialize recommendation systems database

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    age INTEGER,
    gender VARCHAR(10),
    location VARCHAR(50),
    income_level VARCHAR(20),
    preferred_categories TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create items table
CREATE TABLE IF NOT EXISTS items (
    item_id SERIAL PRIMARY KEY,
    category VARCHAR(50) NOT NULL,
    price FLOAT,
    brand VARCHAR(100),
    avg_rating FLOAT,
    num_ratings INTEGER DEFAULT 0,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create interactions table
CREATE TABLE IF NOT EXISTS interactions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    item_id INTEGER REFERENCES items(item_id),
    interaction_type VARCHAR(20) NOT NULL,
    rating FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, item_id, interaction_type)
);

-- Create recommendations table
CREATE TABLE IF NOT EXISTS recommendations (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    item_id INTEGER REFERENCES items(item_id),
    algorithm VARCHAR(50) NOT NULL,
    score FLOAT NOT NULL,
    rank INTEGER,
    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model metadata table
CREATE TABLE IF NOT EXISTS recommendation_models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    algorithm VARCHAR(50) NOT NULL,
    parameters JSONB,
    performance_metrics JSONB,
    model_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_interactions_user ON interactions(user_id);
CREATE INDEX IF NOT EXISTS idx_interactions_item ON interactions(item_id);
CREATE INDEX IF NOT EXISTS idx_interactions_timestamp ON interactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_recommendations_user ON recommendations(user_id);
CREATE INDEX IF NOT EXISTS idx_recommendations_algorithm ON recommendations(algorithm);

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO recsys_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO recsys_user;
EOF

# Create Prometheus configuration
echo "📊 Creating Prometheus configuration..."
cat > prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'recsys-api'
    static_configs:
      - targets: ['recsys-api:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres:5432']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'elasticsearch'
    static_configs:
      - targets: ['elasticsearch:9200']
EOF

# Create Grafana datasource configuration
echo "📈 Creating Grafana configuration..."
mkdir -p grafana/datasources
cat > grafana/datasources/datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true

  - name: PostgreSQL
    type: postgres
    access: proxy
    url: postgres:5432
    database: recsys_db
    user: recsys_user
    password: recsys_pass

  - name: Elasticsearch
    type: elasticsearch
    access: proxy
    url: http://elasticsearch:9200
    database: recommendations
EOF

# Create Grafana dashboard
mkdir -p grafana/dashboards
cat > grafana/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

# Create API Dockerfile
echo "🐳 Creating API Dockerfile..."
cat > Dockerfile.api << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

# Create Data Generator Dockerfile
echo "🐳 Creating Data Generator Dockerfile..."
cat > Dockerfile.generator << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Run the data generator
CMD ["python", "data_generator.py"]
EOF

# Create Model Trainer Dockerfile
echo "🐳 Creating Model Trainer Dockerfile..."
cat > Dockerfile.trainer << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create models directory
RUN mkdir -p /app/models

# Run the model trainer
CMD ["python", "model_trainer.py"]
EOF

# Create model trainer script
echo "🤖 Creating model trainer script..."
cat > model_trainer.py << 'EOF'
#!/usr/bin/env python3
"""
Model Training Service for Recommendation Systems
Trains and updates recommendation models periodically
"""

import os
import pandas as pd
import numpy as np
import psycopg2
import joblib
import time
import logging
from datetime import datetime

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_models():
    """Train recommendation models"""
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=os.getenv('POSTGRES_HOST', 'localhost'),
            port=os.getenv('POSTGRES_PORT', 5432),
            database=os.getenv('POSTGRES_DB', 'recsys_db'),
            user=os.getenv('POSTGRES_USER', 'recsys_user'),
            password=os.getenv('POSTGRES_PASSWORD', 'recsys_pass')
        )
        
        # Load data
        users_df = pd.read_sql("SELECT * FROM users", conn)
        items_df = pd.read_sql("SELECT * FROM items", conn)
        interactions_df = pd.read_sql("SELECT * FROM interactions WHERE rating IS NOT NULL", conn)
        
        if len(interactions_df) == 0:
            logger.warning("No interaction data available for training")
            return
        
        logger.info(f"Training with {len(users_df)} users, {len(items_df)} items, {len(interactions_df)} interactions")
        
        # Train collaborative filtering model
        logger.info("Training collaborative filtering model...")
        
        # Create user-item matrix
        user_item_matrix = interactions_df.pivot_table(
            index='user_id', columns='item_id', values='rating', fill_value=0
        )
        
        # Train SVD model
        svd = TruncatedSVD(n_components=50, random_state=42)
        user_factors = svd.fit_transform(user_item_matrix.values)
        
        # Save model
        model_data = {
            'svd_model': svd,
            'user_factors': user_factors,
            'user_index': user_item_matrix.index.tolist(),
            'item_index': user_item_matrix.columns.tolist()
        }
        joblib.dump(model_data, '/app/models/collaborative_model.pkl')
        
        # Train content-based model
        logger.info("Training content-based model...")
        
        if 'description' in items_df.columns:
            tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
            item_features = tfidf.fit_transform(items_df['description'].fillna(''))
            item_similarity = cosine_similarity(item_features)
            
            content_model = {
                'tfidf': tfidf,
                'item_similarity': item_similarity,
                'item_index': items_df['item_id'].tolist()
            }
            joblib.dump(content_model, '/app/models/content_model.pkl')
        
        logger.info("Model training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model training: {e}")

def main():
    """Main training loop"""
    while True:
        try:
            logger.info("Starting model training...")
            train_models()
            
            # Wait 1 hour before next training
            logger.info("Waiting 1 hour before next training...")
            time.sleep(3600)
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            time.sleep(300)  # Wait 5 minutes on error

if __name__ == "__main__":
    main()
EOF

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if command -v python3 &> /dev/null; then
    python3 -m pip install --user -r requirements.txt
    echo "✅ Python dependencies installed"
else
    echo "⚠️  Python3 not found. Dependencies will be installed in Docker containers."
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose down --remove-orphans
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 45

# Check service health
echo "🔍 Checking service health..."

# Check PostgreSQL
if docker-compose exec -T postgres pg_isready -U recsys_user -d recsys_db > /dev/null 2>&1; then
    echo "✅ PostgreSQL is ready"
else
    echo "❌ PostgreSQL is not ready"
fi

# Check Redis
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is ready"
else
    echo "❌ Redis is not ready"
fi

# Check Elasticsearch
if curl -f http://localhost:9200/_cluster/health > /dev/null 2>&1; then
    echo "✅ Elasticsearch is ready"
else
    echo "❌ Elasticsearch is not ready"
fi

# Generate sample data
echo "📊 Generating sample recommendation system data..."
sleep 15  # Additional wait for data generator

# Display service URLs
echo ""
echo "🎯 Service URLs:"
echo "================================"
echo "📊 Recommendation API:   http://localhost:8000"
echo "📈 Grafana Dashboard:    http://localhost:3000 (admin/recsys123)"
echo "📊 Prometheus:           http://localhost:9090"
echo "🔍 Elasticsearch:        http://localhost:9200"
echo "💾 MLflow:               http://localhost:5000"
echo "📓 Jupyter Notebook:     http://localhost:8888 (token: recsys123)"
echo ""

# Display API endpoints
echo "🔗 API Endpoints:"
echo "================================"
echo "GET  /health                     - Health check"
echo "POST /recommendations            - Get user recommendations"
echo "GET  /users/{user_id}/profile    - Get user profile"
echo "GET  /items/{item_id}/similar    - Get similar items"
echo ""

# Display demo instructions
echo "🚀 Getting Started:"
echo "================================"
echo "1. Run the interactive demo:"
echo "   python demo.py"
echo ""
echo "2. Test the API:"
echo "   curl http://localhost:8000/health"
echo ""
echo "3. Get recommendations:"
echo "   curl -X POST http://localhost:8000/recommendations \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"user_id\": 1, \"num_recommendations\": 5}'"
echo ""
echo "4. Open Jupyter for analysis:"
echo "   http://localhost:8888 (token: recsys123)"
echo ""
echo "5. View monitoring in Grafana:"
echo "   http://localhost:3000 (admin/recsys123)"
echo ""

echo "✅ Day 29 Recommendation Systems environment is ready!"
echo "🎯 You can now work with collaborative filtering, content-based, and hybrid recommendation systems!"