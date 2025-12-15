"""
Day 21: Testing Strategies - Exercise

Build a comprehensive testing framework for data pipelines with unit, integration, 
and end-to-end tests, plus CI/CD integration.

Scenario:
You're the Data Engineering Lead at "TestDriven Analytics", a company that processes 
critical business data for decision-making. You need to implement a comprehensive 
testing strategy that ensures data quality, prevents regressions, and maintains 
high performance standards.

Business Context:
- Processing 5M+ transactions daily with zero tolerance for errors
- Multiple data sources with evolving schemas
- Real-time analytics requiring sub-second response times
- Regulatory compliance with full audit trails
- Multiple teams depending on data pipeline reliability

Your Task:
Build a comprehensive testing framework covering all aspects of data pipeline testing.

Requirements:
1. Unit testing framework for data transformations
2. Integration testing for pipeline component interactions
3. End-to-end testing for full workflow validation
4. Performance testing for load and scalability
5. Regression testing with automated detection
6. CI/CD integration with automated testing pipelines
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch
import json
import hashlib
import time
import psutil

# =============================================================================
# DATA TRANSFORMATION FUNCTIONS (TO BE TESTED)
# =============================================================================

def clean_transaction_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate transaction data with comprehensive error handling"""
    
    if df.empty:
        return df
    
    # Validate required columns
    required_columns = ['transaction_id', 'customer_id', 'amount', 'currency']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Remove duplicates based on transaction_id
    initial_count = len(df)
    df = df.drop_duplicates(subset=['transaction_id'])
    duplicates_removed = initial_count - len(df)
    
    # Validate and filter transaction amounts (convert to numeric first)
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df = df[df['amount'] > 0]
    
    # Standardize currency codes to uppercase
    df['currency'] = df['currency'].astype(str).str.upper().str.strip()
    
    # Fill missing merchant names if column exists
    if 'merchant_name' in df.columns:
        df['merchant_name'] = df['merchant_name'].fillna('Unknown Merchant')
    
    # Add metadata for tracking
    df['_cleaned_at'] = pd.Timestamp.now()
    df['_duplicates_removed'] = duplicates_removed
    
    return df
def calculate_customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive customer-level metrics from transaction data"""
    
    if df.empty:
        return pd.DataFrame()
    
    # Ensure transaction_date is datetime if it exists
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')
    
    # Calculate customer metrics
    agg_dict = {
        'amount': ['sum', 'mean', 'count', 'std', 'min', 'max']
    }
    
    if 'transaction_date' in df.columns:
        agg_dict['transaction_date'] = ['min', 'max']
    
    customer_metrics = df.groupby('customer_id').agg(agg_dict).round(2)
    
    # Flatten column names
    if 'transaction_date' in df.columns:
        customer_metrics.columns = [
            'total_spent', 'avg_amount', 'transaction_count', 'amount_std',
            'min_amount', 'max_amount', 'first_transaction', 'last_transaction'
        ]
        
        # Calculate customer lifetime (days)
        customer_metrics['customer_lifetime_days'] = (
            customer_metrics['last_transaction'] - customer_metrics['first_transaction']
        ).dt.days
    else:
        customer_metrics.columns = [
            'total_spent', 'avg_amount', 'transaction_count', 'amount_std',
            'min_amount', 'max_amount'
        ]
    
    # Calculate additional metrics
    customer_metrics['spending_consistency'] = (
        customer_metrics['amount_std'] / customer_metrics['avg_amount']
    ).fillna(0)
    
    return customer_metrics.reset_index()

def detect_fraud_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Detect comprehensive fraud patterns with multiple algorithms"""
    
    if df.empty:
        return df
    
    df = df.copy()
    df['is_suspicious'] = False
    df['fraud_score'] = 0.0
    df['fraud_reasons'] = ''
    
    # Convert transaction_date to datetime if it exists
    if 'transaction_date' in df.columns:
        df['transaction_datetime'] = pd.to_datetime(df['transaction_date'], errors='coerce')
        df = df.sort_values(['customer_id', 'transaction_datetime'])
    
    # Rule 1: High amount transactions (use percentile-based threshold)
    if len(df) > 0:
        high_amount_threshold = df['amount'].quantile(0.99)  # Top 1%
        high_amount_mask = df['amount'] > high_amount_threshold
        df.loc[high_amount_mask, 'is_suspicious'] = True
        df.loc[high_amount_mask, 'fraud_score'] += 0.3
        df.loc[high_amount_mask, 'fraud_reasons'] += 'high_amount;'
    
    # Rule 2: Rapid successive transactions
    if 'transaction_datetime' in df.columns:
        df['time_diff'] = df.groupby('customer_id')['transaction_datetime'].diff()
        rapid_mask = df['time_diff'] < pd.Timedelta(minutes=1)
        df.loc[rapid_mask, 'is_suspicious'] = True
        df.loc[rapid_mask, 'fraud_score'] += 0.4
        df.loc[rapid_mask, 'fraud_reasons'] += 'rapid_transactions;'
    
    # Rule 3: Statistical outliers per customer
    customer_stats = df.groupby('customer_id')['amount'].agg(['mean', 'std']).reset_index()
    df = df.merge(customer_stats, on='customer_id', suffixes=('', '_customer_avg'))
    
    # Z-score based detection (avoid division by zero)
    df['amount_zscore'] = np.abs(
        (df['amount'] - df['mean']) / df['std'].replace(0, np.inf)
    ).fillna(0)
    
    outlier_mask = df['amount_zscore'] > 3  # 3 standard deviations
    df.loc[outlier_mask, 'is_suspicious'] = True
    df.loc[outlier_mask, 'fraud_score'] += 0.2
    df.loc[outlier_mask, 'fraud_reasons'] += 'statistical_outlier;'
    
    # Update suspicious flag based on fraud score
    df.loc[df['fraud_score'] >= 0.5, 'is_suspicious'] = True
    
    # Clean up fraud_reasons and temporary columns
    df['fraud_reasons'] = df['fraud_reasons'].str.rstrip(';')
    df = df.drop(['mean', 'std', 'amount_zscore'], axis=1, errors='ignore')
    
    return df

# =============================================================================
# UNIT TESTING FRAMEWORK
# =============================================================================

class TestTransactionDataCleaning:
    """Comprehensive unit tests for transaction data cleaning"""
    
    def test_removes_duplicate_transactions(self):
        """Test that duplicate transactions are removed"""
        
        # TODO: Implement test for duplicate removal
        # Create test data with duplicates and verify they're removed
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN001', 'TXN002', 'TXN003'],
            'customer_id': ['CUST001', 'CUST001', 'CUST002', 'CUST003'],
            'amount': [100.0, 100.0, 250.0, 75.0],
            'currency': ['USD', 'USD', 'EUR', 'USD'],
            'merchant_name': ['Store A', 'Store A', 'Store B', 'Store C']
        })
        
        # Act
        result = clean_transaction_data(input_data)
        
        # Assert
        assert len(result) == 3, "Should remove duplicate transactions"
        assert result['transaction_id'].nunique() == 3, "All transaction IDs should be unique"
    
    def test_filters_invalid_amounts(self):
        """Test that transactions with invalid amounts are filtered"""
        
        # TODO: Implement test for amount validation
        # Test negative amounts, zero amounts, etc.
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004'],
            'amount': [100.0, -50.0, 0.0, 250.0],
            'currency': ['USD', 'USD', 'USD', 'EUR'],
            'merchant_name': ['Store A', 'Store B', 'Store C', 'Store D']
        })
        
        # Act
        result = clean_transaction_data(input_data)
        
        # Assert
        assert len(result) == 2, "Should filter out invalid amounts"
        assert all(result['amount'] > 0), "All amounts should be positive"
    
    def test_standardizes_currency_codes(self):
        """Test that currency codes are standardized to uppercase"""
        
        # TODO: Implement test for currency standardization
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003'],
            'amount': [100.0, 250.0, 75.0],
            'currency': ['usd', 'EUR', 'gbp'],
            'merchant_name': ['Store A', 'Store B', 'Store C']
        })
        
        # Act
        result = clean_transaction_data(input_data)
        
        # Assert
        expected_currencies = ['USD', 'EUR', 'GBP']
        assert result['currency'].tolist() == expected_currencies, "Currency codes should be uppercase"
    
    def test_handles_missing_merchant_names(self):
        """Test that missing merchant names are handled properly"""
        
        # TODO: Implement test for missing data handling
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003'],
            'amount': [100.0, 250.0, 75.0],
            'currency': ['USD', 'EUR', 'GBP'],
            'merchant_name': ['Store A', None, 'Store C']
        })
        
        # Act
        result = clean_transaction_data(input_data)
        
        # Assert
        assert result.loc[1, 'merchant_name'] == 'Unknown Merchant', "Missing merchant should be filled"
        assert not result['merchant_name'].isnull().any(), "No merchant names should be null"

class TestCustomerMetricsCalculation:
    """Unit tests for customer metrics calculation"""
    
    def test_calculates_customer_totals(self):
        """Test that customer totals are calculated correctly"""
        
        # TODO: Implement test for customer metrics calculation
        
        # Arrange
        input_data = pd.DataFrame({
            'customer_id': ['CUST001', 'CUST001', 'CUST002', 'CUST002'],
            'amount': [100.0, 150.0, 200.0, 300.0],
            'transaction_date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-03']
        })
        
        # Act
        result = calculate_customer_metrics(input_data)
        
        # Assert
        cust001_metrics = result[result['customer_id'] == 'CUST001'].iloc[0]
        assert cust001_metrics['total_spent'] == 250.0, "Should calculate correct total"
        assert cust001_metrics['avg_amount'] == 125.0, "Should calculate correct average"
        assert cust001_metrics['transaction_count'] == 2, "Should count transactions correctly"
    
    def test_handles_single_transaction_customers(self):
        """Test metrics calculation for customers with single transactions"""
        
        # TODO: Implement test for edge case handling
        
        # Arrange
        input_data = pd.DataFrame({
            'customer_id': ['CUST001'],
            'amount': [100.0],
            'transaction_date': ['2024-01-01']
        })
        
        # Act
        result = calculate_customer_metrics(input_data)
        
        # Assert
        assert len(result) == 1, "Should handle single transaction"
        assert result.iloc[0]['total_spent'] == 100.0, "Should calculate single transaction correctly"
        assert result.iloc[0]['transaction_count'] == 1, "Should count single transaction"

class TestFraudDetection:
    """Unit tests for fraud detection functionality"""
    
    def test_flags_high_amount_transactions(self):
        """Test that high amount transactions are flagged as suspicious"""
        
        # TODO: Implement test for high amount fraud detection
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003'],
            'amount': [100.0, 15000.0, 250.0],
            'transaction_date': ['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00']
        })
        
        # Act
        result = detect_fraud_patterns(input_data)
        
        # Assert
        assert result.iloc[1]['is_suspicious'] == True, "High amount transaction should be flagged"
        assert result.iloc[0]['is_suspicious'] == False, "Normal amount should not be flagged"
        assert result.iloc[2]['is_suspicious'] == False, "Normal amount should not be flagged"
    
    def test_flags_rapid_transactions(self):
        """Test that rapid successive transactions are flagged"""
        
        # TODO: Implement test for rapid transaction detection
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'customer_id': ['CUST001', 'CUST001', 'CUST002'],
            'amount': [100.0, 150.0, 200.0],
            'transaction_date': ['2024-01-01 10:00:00', '2024-01-01 10:00:30', '2024-01-01 11:00:00']
        })
        
        # Act
        result = detect_fraud_patterns(input_data)
        
        # Assert
        # Second transaction should be flagged (30 seconds after first)
        suspicious_transactions = result[result['is_suspicious'] == True]
        assert len(suspicious_transactions) > 0, "Rapid transactions should be flagged"

# =============================================================================
# INTEGRATION TESTING FRAMEWORK
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for pipeline components working together"""
    
    @pytest.fixture
    def sample_pipeline_data(self):
        """Sample data for pipeline integration testing"""
        
        # TODO: Create comprehensive sample data for integration testing
        
        return pd.DataFrame({
            'transaction_id': [f'TXN{i:03d}' for i in range(1, 101)],
            'customer_id': [f'CUST{i:03d}' for i in range(1, 51)] * 2,
            'amount': np.random.uniform(10, 1000, 100),
            'currency': np.random.choice(['USD', 'EUR', 'GBP'], 100),
            'merchant_name': [f'Merchant {i}' for i in range(1, 21)] * 5,
            'transaction_date': pd.date_range('2024-01-01', periods=100, freq='H')
        })
    
    def test_end_to_end_pipeline_flow(self, sample_pipeline_data):
        """Test complete pipeline flow from raw data to final metrics"""
        
        # TODO: Implement end-to-end pipeline integration test
        # Test: raw data -> cleaning -> metrics -> fraud detection
        
        # Act - Run complete pipeline
        cleaned_data = clean_transaction_data(sample_pipeline_data)
        customer_metrics = calculate_customer_metrics(cleaned_data)
        fraud_results = detect_fraud_patterns(cleaned_data)
        
        # Assert - Verify pipeline integration
        assert len(cleaned_data) > 0, "Cleaning should produce data"
        assert len(customer_metrics) > 0, "Should generate customer metrics"
        assert 'is_suspicious' in fraud_results.columns, "Should add fraud detection results"
        
        # Verify data consistency across pipeline stages
        unique_customers_cleaned = cleaned_data['customer_id'].nunique()
        unique_customers_metrics = customer_metrics['customer_id'].nunique()
        assert unique_customers_cleaned == unique_customers_metrics, "Customer counts should match"
    
    def test_pipeline_error_handling(self):
        """Test pipeline handles errors gracefully"""
        
        # TODO: Implement error handling tests
        # Test with malformed data, missing columns, etc.
        
        # Arrange - Create problematic data
        problematic_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002'],
            'customer_id': ['CUST001', None],  # Missing customer ID
            'amount': [100.0, 'invalid'],  # Invalid amount
            'currency': ['USD', 'EUR']
            # Missing merchant_name column
        })
        
        # Act & Assert - Should handle errors gracefully
        try:
            result = clean_transaction_data(problematic_data)
            # Should not crash, but may filter out invalid records
            assert isinstance(result, pd.DataFrame), "Should return DataFrame even with errors"
        except Exception as e:
            pytest.fail(f"Pipeline should handle errors gracefully, but got: {e}")
    
    def test_pipeline_performance_integration(self, sample_pipeline_data):
        """Test pipeline performance with realistic data volumes"""
        
        # TODO: Implement performance integration test
        
        # Arrange - Scale up data for performance testing
        large_data = pd.concat([sample_pipeline_data] * 10, ignore_index=True)
        large_data['transaction_id'] = [f'TXN{i:06d}' for i in range(len(large_data))]
        
        # Act - Measure pipeline performance
        start_time = time.time()
        
        cleaned_data = clean_transaction_data(large_data)
        customer_metrics = calculate_customer_metrics(cleaned_data)
        fraud_results = detect_fraud_patterns(cleaned_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert - Performance requirements
        assert processing_time < 10.0, "Pipeline should complete within 10 seconds"
        assert len(fraud_results) > 0, "Should produce fraud detection results"
        
        # Calculate throughput
        throughput = len(large_data) / processing_time
        assert throughput > 100, "Should process at least 100 records per second"

# =============================================================================
# PERFORMANCE TESTING FRAMEWORK
# =============================================================================

class TestPerformanceRequirements:
    """Performance testing for data pipeline components"""
    
    def test_cleaning_performance_scalability(self):
        """Test data cleaning performance scales with data size"""
        
        # TODO: Implement scalability performance test
        
        data_sizes = [1000, 5000, 10000]
        performance_results = []
        
        for size in data_sizes:
            # Generate test data
            test_data = pd.DataFrame({
                'transaction_id': [f'TXN{i:06d}' for i in range(size)],
                'customer_id': [f'CUST{i:04d}' for i in range(size // 10)] * 10,
                'amount': np.random.uniform(10, 1000, size),
                'currency': np.random.choice(['USD', 'EUR', 'GBP'], size),
                'merchant_name': [f'Merchant {i}' for i in range(100)] * (size // 100 + 1)
            })[:size]
            
            # Measure performance
            start_time = time.time()
            result = clean_transaction_data(test_data)
            end_time = time.time()
            
            processing_time = end_time - start_time
            throughput = size / processing_time
            
            performance_results.append({
                'size': size,
                'time': processing_time,
                'throughput': throughput
            })
        
        # Assert - Performance should scale reasonably
        for result in performance_results:
            assert result['throughput'] > 1000, f"Should maintain >1000 records/sec for {result['size']} records"
        
        # Check that performance doesn't degrade significantly with size
        small_throughput = performance_results[0]['throughput']
        large_throughput = performance_results[-1]['throughput']
        degradation = (small_throughput - large_throughput) / small_throughput
        
        assert degradation < 0.5, "Performance degradation should be less than 50%"
    
    def test_memory_usage_limits(self):
        """Test that pipeline stays within memory limits"""
        
        # TODO: Implement memory usage testing
        
        # Generate large dataset
        large_size = 50000
        test_data = pd.DataFrame({
            'transaction_id': [f'TXN{i:06d}' for i in range(large_size)],
            'customer_id': [f'CUST{i:04d}' for i in range(large_size // 100)] * 100,
            'amount': np.random.uniform(10, 1000, large_size),
            'currency': np.random.choice(['USD', 'EUR', 'GBP'], large_size),
            'merchant_name': [f'Merchant {i}' for i in range(1000)] * (large_size // 1000 + 1)
        })[:large_size]
        
        # Monitor memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Process data
        result = clean_transaction_data(test_data)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        # Assert - Memory usage limits
        assert memory_used < 500, "Should use less than 500MB additional memory"
        assert len(result) > 0, "Should produce valid results"

# =============================================================================
# REGRESSION TESTING FRAMEWORK
# =============================================================================

class TestRegressionDetection:
    """Regression testing to detect changes in pipeline behavior"""
    
    def test_output_consistency_regression(self):
        """Test that pipeline outputs remain consistent"""
        
        # TODO: Implement regression test for output consistency
        
        # Arrange - Fixed test data for consistent results
        fixed_test_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],
            'customer_id': ['CUST001', 'CUST001', 'CUST002', 'CUST002'],
            'amount': [100.0, 150.0, 200.0, 250.0],
            'currency': ['USD', 'USD', 'EUR', 'EUR'],
            'merchant_name': ['Store A', 'Store B', 'Store C', 'Store D']
        })
        
        # Act
        cleaned_data = clean_transaction_data(fixed_test_data)
        customer_metrics = calculate_customer_metrics(cleaned_data)
        
        # Assert - Expected consistent results
        expected_customer_count = 2
        expected_total_amount = 700.0
        
        assert len(customer_metrics) == expected_customer_count, "Customer count should be consistent"
        assert customer_metrics['total_spent'].sum() == expected_total_amount, "Total amounts should be consistent"
    
    def test_performance_regression(self):
        """Test that performance doesn't regress"""
        
        # TODO: Implement performance regression test
        
        # Arrange - Standard test dataset
        standard_size = 10000
        test_data = pd.DataFrame({
            'transaction_id': [f'TXN{i:06d}' for i in range(standard_size)],
            'customer_id': [f'CUST{i:04d}' for i in range(standard_size // 50)] * 50,
            'amount': np.random.uniform(10, 1000, standard_size),
            'currency': ['USD'] * standard_size,
            'merchant_name': ['Test Merchant'] * standard_size
        })
        
        # Act - Measure performance
        start_time = time.time()
        result = clean_transaction_data(test_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = standard_size / processing_time
        
        # Assert - Performance baseline (these would be updated based on historical performance)
        baseline_throughput = 5000  # records per second
        acceptable_degradation = 0.2  # 20%
        
        min_acceptable_throughput = baseline_throughput * (1 - acceptable_degradation)
        assert throughput >= min_acceptable_throughput, f"Performance regression detected: {throughput} < {min_acceptable_throughput}"

# =============================================================================
# EXERCISE INSTRUCTIONS
# =============================================================================

def print_exercise_instructions():
    """Print detailed exercise instructions"""
    
    print("🎯 Testing Strategies Exercise - TestDriven Analytics")
    print("=" * 60)
    
    print("\n📋 REQUIREMENTS:")
    print("1. Implement comprehensive unit tests for all data transformation functions")
    print("2. Build integration tests for pipeline component interactions")
    print("3. Create end-to-end tests for complete workflow validation")
    print("4. Implement performance tests for scalability and load testing")
    print("5. Build regression tests to detect changes in behavior")
    print("6. Set up CI/CD integration with automated test execution")
    
    print("\n🏗️ TESTING ARCHITECTURE:")
    print("""
    TestDriven Analytics Testing Framework:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Unit Tests (70%)                             │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │ Transform   │  │  Metrics    │  │   Fraud     │             │
    │  │   Tests     │  │   Tests     │  │  Detection  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                Integration Tests (20%)                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │  Pipeline   │  │   Error     │  │Performance  │             │
    │  │    Flow     │  │  Handling   │  │Integration  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                End-to-End Tests (10%)                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │  Complete   │  │ Production  │  │   System    │             │
    │  │ Workflows   │  │ Scenarios   │  │Integration  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n🎯 SUCCESS CRITERIA:")
    print("• Comprehensive test coverage across all pipeline components")
    print("• Automated detection of regressions and performance issues")
    print("• Fast feedback loop with CI/CD integration")
    print("• Reliable error handling and edge case coverage")
    print("• Performance validation under realistic load conditions")
    print("• Production-ready testing infrastructure")
    
    print("\n🚀 GETTING STARTED:")
    print("1. Complete the unit tests for all transformation functions")
    print("2. Implement integration tests for pipeline component interactions")
    print("3. Build end-to-end tests for complete workflow validation")
    print("4. Add performance tests for scalability requirements")
    print("5. Create regression tests to prevent quality degradation")
    print("6. Set up automated CI/CD pipeline with comprehensive testing")

if __name__ == "__main__":
    print_exercise_instructions()
    
    print("\n" + "="*60)
    print("🎯 Ready to build bulletproof data pipeline testing!")
    print("Complete the TODOs above to create a comprehensive testing framework.")
    print("="*60)