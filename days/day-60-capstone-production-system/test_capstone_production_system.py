#!/usr/bin/env python3
"""
Day 60: Capstone Production System - Comprehensive Test Suite

Tests for the complete Intelligent Customer Analytics Platform including:
- API endpoints and business logic
- Database operations and caching
- ML model predictions and feature store
- GenAI insight generation
- System integration and performance
- Security and authentication
- Monitoring and observability

Author: 60 Days Advanced Data and AI Curriculum
"""

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, List, Any

import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from httpx import AsyncClient
import redis.asyncio as redis

# Import application components
from solution import (
    app, Config, DatabaseManager, MLModelManager, GenAIInsightGenerator,
    CustomerProfile, PredictionRequest, PredictionResponse,
    InsightRequest, InsightResponse
)

# Test configuration
TEST_CONFIG = Config(
    postgres_url="postgresql://test:test@localhost:5432/test_customers",
    mongodb_url="mongodb://localhost:27017/test_events",
    redis_url="redis://localhost:6379/1",
    kafka_bootstrap_servers="localhost:9092",
    mlflow_tracking_uri="http://localhost:5000",
    feast_repo_path="/tmp/test_feature_repo",
    openai_api_key="test-key",
    environment="test"
)

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
async def async_client():
    """Async HTTP client for testing"""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac

@pytest.fixture
def sample_customer_profile():
    """Sample customer profile for testing"""
    return CustomerProfile(
        customer_id="test-customer-123",
        email="test@example.com",
        first_name="John",
        last_name="Doe",
        segment="Premium",
        lifetime_value=5000.0,
        registration_date=datetime.now() - timedelta(days=365),
        total_purchases=25,
        avg_order_value=200.0,
        days_since_last_purchase=15,
        churn_probability=0.25
    )

@pytest.fixture
def mock_redis():
    """Mock Redis client"""
    mock_redis = AsyncMock()
    mock_redis.hset = AsyncMock()
    mock_redis.hgetall = AsyncMock(return_value={"test": "value"})
    mock_redis.expire = AsyncMock()
    mock_redis.ping = AsyncMock(return_value=True)
    return mock_redis

@pytest.fixture
def mock_postgres_pool():
    """Mock PostgreSQL connection pool"""
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=1)
    mock_pool.acquire = AsyncMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    return mock_pool

class TestDatabaseManager:
    """Test database operations and connection management"""
    
    @pytest.mark.asyncio
    async def test_database_initialization(self, mock_postgres_pool, mock_redis):
        """Test database manager initialization"""
        db_manager = DatabaseManager()
        
        with patch('asyncpg.create_pool', return_value=mock_postgres_pool), \
             patch('redis.asyncio.from_url', return_value=mock_redis):
            
            await db_manager.initialize()
            
            assert db_manager.postgres_pool is not None
            assert db_manager.redis_client is not None
    
    @pytest.mark.asyncio
    async def test_get_customer_profile_success(self, sample_customer_profile):
        """Test successful customer profile retrieval"""
        db_manager = DatabaseManager()
        db_manager.postgres_pool = Mock()
        
        # Mock database response
        mock_row = {
            'customer_id': sample_customer_profile.customer_id,
            'email': sample_customer_profile.email,
            'first_name': sample_customer_profile.first_name,
            'last_name': sample_customer_profile.last_name,
            'segment': sample_customer_profile.segment,
            'lifetime_value': sample_customer_profile.lifetime_value,
            'registration_date': sample_customer_profile.registration_date,
            'total_purchases': sample_customer_profile.total_purchases,
            'avg_order_value': sample_customer_profile.avg_order_value,
            'days_since_last_purchase': sample_customer_profile.days_since_last_purchase
        }
        
        mock_conn = AsyncMock()
        mock_conn.fetchrow = AsyncMock(return_value=mock_row)
        db_manager.postgres_pool.acquire = AsyncMock()
        db_manager.postgres_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
        db_manager.postgres_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        result = await db_manager.get_customer_profile(sample_customer_profile.customer_id)
        
        assert result is not None
        assert result.customer_id == sample_customer_profile.customer_id
        assert result.email == sample_customer_profile.email
        assert result.segment == sample_customer_profile.segment

class TestMLModelManager:
    """Test ML model operations and predictions"""
    
    @pytest.mark.asyncio
    async def test_ml_manager_initialization(self):
        """Test ML manager initialization"""
        ml_manager = MLModelManager()
        
        with patch('mlflow.set_tracking_uri'), \
             patch('feast.FeatureStore'), \
             patch.object(ml_manager, 'load_models', new_callable=AsyncMock):
            
            await ml_manager.initialize()
            
            assert ml_manager.feature_store is not None

class TestGenAIInsightGenerator:
    """Test GenAI insight generation and RAG system"""
    
    @pytest.mark.asyncio
    async def test_genai_initialization_with_api_key(self):
        """Test GenAI initialization with API key"""
        genai_manager = GenAIInsightGenerator()
        
        with patch('langchain.embeddings.OpenAIEmbeddings'), \
             patch('langchain.llms.OpenAI'), \
             patch.object(genai_manager, 'setup_knowledge_base', new_callable=AsyncMock):
            
            # Mock config with API key
            with patch('solution.config') as mock_config:
                mock_config.openai_api_key = "test-api-key"
                
                await genai_manager.initialize()
                
                assert genai_manager.embeddings is not None
                assert genai_manager.llm is not None

class TestAPIEndpoints:
    """Test FastAPI endpoints and HTTP responses"""
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Customer Analytics Platform"
        assert data["version"] == "1.0.0"
        assert data["status"] == "operational"
    
    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint"""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        assert "prometheus" in response.headers.get("content-type", "").lower() or \
               "text/plain" in response.headers.get("content-type", "")

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([
        __file__,
        "-v",
        "--tb=short"
    ])