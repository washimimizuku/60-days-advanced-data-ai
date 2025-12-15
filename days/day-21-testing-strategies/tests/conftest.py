"""
Day 21: Testing Strategies - Pytest Configuration
Shared fixtures and configuration for all tests
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import tempfile
import shutil
from typing import Dict, Any

# Set random seed for reproducible tests
np.random.seed(42)

@pytest.fixture(scope="session")
def database_url():
    """Database URL for testing"""
    return os.getenv("DATABASE_URL", "postgresql://testuser:testpass123@localhost:5432/testing_db")

@pytest.fixture
def sample_transaction_data():
    """Sample transaction data for testing"""
    return pd.DataFrame({
        'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004', 'TXN005'],
        'customer_id': ['CUST001', 'CUST001', 'CUST002', 'CUST002', 'CUST003'],
        'amount': [100.0, 150.0, 200.0, 250.0, 75.0],
        'currency': ['USD', 'USD', 'EUR', 'EUR', 'GBP'],
        'merchant_name': ['Store A', 'Store B', 'Store C', 'Store D', 'Store E'],
        'transaction_date': [
            '2024-01-01 10:00:00',
            '2024-01-01 14:00:00',
            '2024-01-02 09:00:00',
            '2024-01-02 15:00:00',
            '2024-01-03 11:00:00'
        ]
    })

@pytest.fixture
def large_transaction_data():
    """Large transaction dataset for performance testing"""
    size = 10000
    return pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(size)],
        'customer_id': [f'CUST{i % 1000:04d}' for i in range(size)],
        'amount': np.random.uniform(10, 1000, size),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], size),
        'merchant_name': [f'Merchant {i % 100}' for i in range(size)],
        'transaction_date': pd.date_range('2024-01-01', periods=size, freq='5min')
    })

@pytest.fixture
def problematic_transaction_data():
    """Transaction data with quality issues for testing error handling"""
    return pd.DataFrame({
        'transaction_id': ['TXN001', 'TXN002', 'TXN002', 'TXN003', 'TXN004'],  # Duplicate
        'customer_id': ['CUST001', 'CUST002', 'CUST002', None, 'CUST004'],     # Missing value
        'amount': [100.0, -50.0, 200.0, 'invalid', 0.0],                      # Invalid amounts
        'currency': ['USD', 'EUR', 'XXX', 'GBP', ''],                         # Invalid currency
        'merchant_name': ['Store A', None, 'Store C', 'Store D', 'Store E']    # Missing merchant
    })

@pytest.fixture
def temp_directory():
    """Temporary directory for test files"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def test_config():
    """Test configuration settings"""
    return {
        'performance_threshold_seconds': 5.0,
        'memory_limit_mb': 1024,
        'throughput_threshold': 1000,
        'regression_tolerance': 0.05,
        'coverage_threshold': 0.85
    }

@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup test environment before each test"""
    # Set environment variables for testing
    os.environ['TESTING_MODE'] = 'true'
    os.environ['LOG_LEVEL'] = 'WARNING'  # Reduce log noise during tests
    
    yield
    
    # Cleanup after test
    # Remove any test-specific environment variables if needed
    pass

# Performance testing fixtures
@pytest.fixture
def performance_baseline():
    """Performance baseline metrics for regression testing"""
    return {
        'execution_time_seconds': 2.0,
        'memory_usage_mb': 100.0,
        'throughput_records_per_second': 5000.0
    }

# Markers for different test types
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "regression: Regression tests")

# Test collection customization
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location"""
    for item in items:
        # Add markers based on test file location
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
            item.add_marker(pytest.mark.slow)