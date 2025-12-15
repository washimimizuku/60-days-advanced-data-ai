"""
Day 21: Testing Strategies - Setup Verification Tests
Tests to verify that the testing environment is properly configured
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

class TestEnvironmentSetup:
    """Test that the testing environment is properly configured"""
    
    def test_python_version(self):
        """Test that Python version is compatible"""
        assert sys.version_info >= (3, 8), "Python 3.8+ is required"
    
    def test_required_packages_installed(self):
        """Test that all required packages are installed"""
        required_packages = [
            'pandas',
            'numpy',
            'pytest',
            'psutil'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                pytest.fail(f"Required package {package} is not installed")
    
    def test_pandas_functionality(self):
        """Test basic pandas functionality"""
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        assert len(df) == 3
        assert list(df.columns) == ['a', 'b']
        assert df['a'].sum() == 6
    
    def test_numpy_functionality(self):
        """Test basic numpy functionality"""
        arr = np.array([1, 2, 3, 4, 5])
        assert arr.sum() == 15
        assert arr.mean() == 3.0
    
    def test_environment_variables(self):
        """Test that required environment variables are set"""
        testing_mode = os.getenv('TESTING_MODE', 'false')
        assert testing_mode.lower() == 'true', "TESTING_MODE should be set to 'true'"
    
    def test_directory_structure(self):
        """Test that required directories exist"""
        required_dirs = [
            'tests',
            'tests/unit',
            'tests/integration',
            'tests/e2e',
            'tests/performance',
            'data'
        ]
        
        for dir_path in required_dirs:
            assert Path(dir_path).exists(), f"Required directory {dir_path} does not exist"
    
    def test_database_connection_config(self):
        """Test that database connection configuration is available"""
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            assert 'postgresql://' in db_url, "DATABASE_URL should be a PostgreSQL connection string"
    
    def test_file_permissions(self):
        """Test that required files have proper permissions"""
        executable_files = [
            'scripts/setup.sh',
            'scripts/run_tests.py'
        ]
        
        for file_path in executable_files:
            if Path(file_path).exists():
                assert os.access(file_path, os.X_OK), f"File {file_path} should be executable"

class TestDataProcessingSetup:
    """Test data processing capabilities"""
    
    def test_sample_data_creation(self):
        """Test that we can create sample data"""
        sample_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002'],
            'customer_id': ['CUST001', 'CUST002'],
            'amount': [100.0, 200.0],
            'currency': ['USD', 'EUR']
        })
        
        assert len(sample_data) == 2
        assert sample_data['amount'].sum() == 300.0
    
    def test_data_validation_functions(self):
        """Test basic data validation functions"""
        # Test data type validation
        df = pd.DataFrame({'amount': [100.0, 200.0, 'invalid']})
        numeric_amounts = pd.to_numeric(df['amount'], errors='coerce')
        valid_amounts = numeric_amounts.dropna()
        
        assert len(valid_amounts) == 2
        assert valid_amounts.sum() == 300.0
    
    def test_performance_measurement(self):
        """Test that we can measure performance"""
        import time
        
        start_time = time.time()
        # Simulate some work
        df = pd.DataFrame({'x': range(1000)})
        result = df['x'].sum()
        end_time = time.time()
        
        execution_time = end_time - start_time
        assert execution_time >= 0
        assert result == 499500  # Sum of 0 to 999

class TestConfigurationSetup:
    """Test configuration and settings"""
    
    def test_pytest_configuration(self):
        """Test that pytest is properly configured"""
        # This test runs, so pytest is working
        assert True
    
    def test_coverage_configuration(self):
        """Test that coverage measurement is available"""
        try:
            import coverage
            assert True
        except ImportError:
            pytest.skip("Coverage package not installed")
    
    def test_logging_configuration(self):
        """Test logging configuration"""
        import logging
        
        logger = logging.getLogger('test_logger')
        logger.info("Test log message")
        
        # If we get here without error, logging is working
        assert True

if __name__ == '__main__':
    # Run setup verification tests
    pytest.main([__file__, '-v'])