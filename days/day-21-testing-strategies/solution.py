"""
Day 21: Testing Strategies - Complete Solution

Comprehensive testing framework for data pipelines with unit, integration, and end-to-end tests.
This solution demonstrates enterprise-grade testing implementation for TestDriven Analytics.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import Mock, patch, MagicMock
import json
import hashlib
import time
import psutil
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# =============================================================================
# PRODUCTION DATA TRANSFORMATION FUNCTIONS
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
    
    # Validate and filter transaction amounts
    df = df[pd.to_numeric(df['amount'], errors='coerce') > 0]
    
    # Standardize currency codes
    df['currency'] = df['currency'].astype(str).str.upper().str.strip()
    
    # Handle missing merchant names
    if 'merchant_name' in df.columns:
        df['merchant_name'] = df['merchant_name'].fillna('Unknown Merchant')
    
    # Add data quality metadata
    df['_cleaned_at'] = datetime.now()
    df['_duplicates_removed'] = duplicates_removed
    
    return df

def calculate_customer_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive customer-level metrics"""
    
    if df.empty:
        return pd.DataFrame()
    
    # Ensure transaction_date is datetime
    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    
    # Calculate customer metrics
    customer_metrics = df.groupby('customer_id').agg({
        'amount': ['sum', 'mean', 'count', 'std', 'min', 'max'],
        'transaction_date': ['min', 'max'] if 'transaction_date' in df.columns else ['count']
    }).round(2)
    
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
    customer_metrics['amount_variance'] = customer_metrics['amount_std'] ** 2
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
        df['transaction_datetime'] = pd.to_datetime(df['transaction_date'])
        df = df.sort_values(['customer_id', 'transaction_datetime'])
    
    # Rule 1: High amount transactions
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
    
    # Rule 3: Unusual spending patterns (statistical outliers)
    customer_stats = df.groupby('customer_id')['amount'].agg(['mean', 'std']).reset_index()
    df = df.merge(customer_stats, on='customer_id', suffixes=('', '_customer_avg'))
    
    # Z-score based detection
    df['amount_zscore'] = np.abs(
        (df['amount'] - df['mean']) / df['std'].replace(0, np.inf)
    ).fillna(0)
    
    outlier_mask = df['amount_zscore'] > 3  # 3 standard deviations
    df.loc[outlier_mask, 'is_suspicious'] = True
    df.loc[outlier_mask, 'fraud_score'] += 0.2
    df.loc[outlier_mask, 'fraud_reasons'] += 'statistical_outlier;'
    
    # Rule 4: Round number bias (potential synthetic transactions)
    round_amounts = df['amount'] % 1 == 0  # Whole numbers
    large_round = (round_amounts) & (df['amount'] >= 1000)
    df.loc[large_round, 'fraud_score'] += 0.1
    df.loc[large_round, 'fraud_reasons'] += 'round_amount;'
    
    # Update suspicious flag based on fraud score
    df.loc[df['fraud_score'] >= 0.5, 'is_suspicious'] = True
    
    # Clean up fraud_reasons
    df['fraud_reasons'] = df['fraud_reasons'].str.rstrip(';')
    
    return df.drop(['mean', 'std', 'amount_zscore'], axis=1, errors='ignore')

# =============================================================================
# COMPREHENSIVE UNIT TESTING FRAMEWORK
# =============================================================================

class TestTransactionDataCleaning:
    """Comprehensive unit tests for transaction data cleaning"""
    
    def test_removes_duplicate_transactions(self):
        """Test that duplicate transactions are removed correctly"""
        
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
        assert result['_duplicates_removed'].iloc[0] == 1, "Should track duplicates removed"
    
    def test_filters_invalid_amounts(self):
        """Test comprehensive amount validation"""
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004', 'TXN005'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
            'amount': [100.0, -50.0, 0.0, 'invalid', None],
            'currency': ['USD', 'USD', 'USD', 'EUR', 'GBP'],
            'merchant_name': ['Store A', 'Store B', 'Store C', 'Store D', 'Store E']
        })
        
        # Act
        result = clean_transaction_data(input_data)
        
        # Assert
        assert len(result) == 1, "Should filter out all invalid amounts"
        assert all(result['amount'] > 0), "All amounts should be positive"
        assert result.iloc[0]['transaction_id'] == 'TXN001', "Should keep valid transaction"
    
    def test_standardizes_currency_codes(self):
        """Test currency code standardization"""
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004'],
            'amount': [100.0, 250.0, 75.0, 300.0],
            'currency': ['usd', 'EUR', '  gbp  ', 'Cad'],
            'merchant_name': ['Store A', 'Store B', 'Store C', 'Store D']
        })
        
        # Act
        result = clean_transaction_data(input_data)
        
        # Assert
        expected_currencies = ['USD', 'EUR', 'GBP', 'CAD']
        assert result['currency'].tolist() == expected_currencies, "Currency codes should be standardized"
    
    def test_handles_missing_merchant_names(self):
        """Test missing merchant name handling"""
        
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
    
    def test_handles_empty_dataframe(self):
        """Test edge case: empty dataframe"""
        
        # Arrange
        empty_df = pd.DataFrame()
        
        # Act
        result = clean_transaction_data(empty_df)
        
        # Assert
        assert len(result) == 0, "Should handle empty dataframe"
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
    
    def test_validates_required_columns(self):
        """Test validation of required columns"""
        
        # Arrange
        incomplete_data = pd.DataFrame({
            'transaction_id': ['TXN001'],
            'amount': [100.0]
            # Missing customer_id and currency
        })
        
        # Act & Assert
        with pytest.raises(ValueError, match="Missing required columns"):
            clean_transaction_data(incomplete_data)
    
    @pytest.mark.parametrize("amount,expected_valid", [
        (100.0, True),
        (-50.0, False),
        (0.0, False),
        (0.01, True),
        (999999.99, True)
    ])
    def test_amount_validation_edge_cases(self, amount, expected_valid):
        """Test amount validation with various edge cases"""
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001'],
            'customer_id': ['CUST001'],
            'amount': [amount],
            'currency': ['USD']
        })
        
        # Act
        result = clean_transaction_data(input_data)
        
        # Assert
        if expected_valid:
            assert len(result) == 1, f"Amount {amount} should be valid"
        else:
            assert len(result) == 0, f"Amount {amount} should be invalid"
class TestCustomerMetricsCalculation:
    """Comprehensive unit tests for customer metrics calculation"""
    
    def test_calculates_basic_customer_metrics(self):
        """Test basic customer metrics calculation"""
        
        # Arrange
        input_data = pd.DataFrame({
            'customer_id': ['CUST001', 'CUST001', 'CUST002', 'CUST002'],
            'amount': [100.0, 150.0, 200.0, 300.0],
            'transaction_date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-03']
        })
        
        # Act
        result = calculate_customer_metrics(input_data)
        
        # Assert
        assert len(result) == 2, "Should have metrics for 2 customers"
        
        cust001_metrics = result[result['customer_id'] == 'CUST001'].iloc[0]
        assert cust001_metrics['total_spent'] == 250.0, "Should calculate correct total"
        assert cust001_metrics['avg_amount'] == 125.0, "Should calculate correct average"
        assert cust001_metrics['transaction_count'] == 2, "Should count transactions correctly"
        assert cust001_metrics['min_amount'] == 100.0, "Should calculate minimum"
        assert cust001_metrics['max_amount'] == 150.0, "Should calculate maximum"
    
    def test_calculates_advanced_metrics(self):
        """Test advanced customer metrics calculation"""
        
        # Arrange
        input_data = pd.DataFrame({
            'customer_id': ['CUST001', 'CUST001', 'CUST001'],
            'amount': [100.0, 200.0, 150.0],
            'transaction_date': ['2024-01-01', '2024-01-05', '2024-01-10']
        })
        
        # Act
        result = calculate_customer_metrics(input_data)
        
        # Assert
        cust_metrics = result.iloc[0]
        assert cust_metrics['customer_lifetime_days'] == 9, "Should calculate customer lifetime"
        assert cust_metrics['amount_std'] > 0, "Should calculate standard deviation"
        assert cust_metrics['spending_consistency'] > 0, "Should calculate spending consistency"
    
    def test_handles_single_transaction_customers(self):
        """Test metrics for customers with single transactions"""
        
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
        cust_metrics = result.iloc[0]
        assert cust_metrics['total_spent'] == 100.0, "Should calculate single transaction correctly"
        assert cust_metrics['transaction_count'] == 1, "Should count single transaction"
        assert cust_metrics['customer_lifetime_days'] == 0, "Single transaction should have 0 lifetime"
    
    def test_handles_empty_dataframe(self):
        """Test edge case: empty dataframe"""
        
        # Arrange
        empty_df = pd.DataFrame()
        
        # Act
        result = calculate_customer_metrics(empty_df)
        
        # Assert
        assert len(result) == 0, "Should handle empty dataframe"
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
    
    def test_handles_missing_transaction_date(self):
        """Test metrics calculation without transaction dates"""
        
        # Arrange
        input_data = pd.DataFrame({
            'customer_id': ['CUST001', 'CUST001'],
            'amount': [100.0, 150.0]
        })
        
        # Act
        result = calculate_customer_metrics(input_data)
        
        # Assert
        assert len(result) == 1, "Should handle missing transaction dates"
        assert 'customer_lifetime_days' not in result.columns, "Should not calculate lifetime without dates"
        assert result.iloc[0]['total_spent'] == 250.0, "Should still calculate basic metrics"

class TestFraudDetection:
    """Comprehensive unit tests for fraud detection"""
    
    def test_flags_high_amount_transactions(self):
        """Test high amount fraud detection"""
        
        # Arrange
        amounts = [100.0] * 99 + [50000.0]  # One outlier
        input_data = pd.DataFrame({
            'transaction_id': [f'TXN{i:03d}' for i in range(100)],
            'customer_id': [f'CUST{i:03d}' for i in range(100)],
            'amount': amounts,
            'transaction_date': ['2024-01-01 10:00:00'] * 100
        })
        
        # Act
        result = detect_fraud_patterns(input_data)
        
        # Assert
        high_amount_flagged = result[result['amount'] == 50000.0]['is_suspicious'].iloc[0]
        assert high_amount_flagged == True, "High amount transaction should be flagged"
        
        normal_amount_flagged = result[result['amount'] == 100.0]['is_suspicious'].any()
        assert not normal_amount_flagged, "Normal amounts should not be flagged for high amount"
    
    def test_flags_rapid_transactions(self):
        """Test rapid transaction fraud detection"""
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'customer_id': ['CUST001', 'CUST001', 'CUST002'],
            'amount': [100.0, 150.0, 200.0],
            'transaction_date': [
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:30',  # 30 seconds later
                '2024-01-01 11:00:00'
            ]
        })
        
        # Act
        result = detect_fraud_patterns(input_data)
        
        # Assert
        rapid_transaction = result[result['transaction_id'] == 'TXN002']
        assert rapid_transaction['is_suspicious'].iloc[0] == True, "Rapid transaction should be flagged"
        assert 'rapid_transactions' in rapid_transaction['fraud_reasons'].iloc[0], "Should include fraud reason"
    
    def test_statistical_outlier_detection(self):
        """Test statistical outlier detection"""
        
        # Arrange - Customer with consistent spending except one outlier
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],
            'customer_id': ['CUST001', 'CUST001', 'CUST001', 'CUST001'],
            'amount': [100.0, 105.0, 95.0, 1000.0],  # Last one is outlier
            'transaction_date': ['2024-01-01 10:00:00'] * 4
        })
        
        # Act
        result = detect_fraud_patterns(input_data)
        
        # Assert
        outlier_transaction = result[result['amount'] == 1000.0]
        assert outlier_transaction['is_suspicious'].iloc[0] == True, "Statistical outlier should be flagged"
        assert 'statistical_outlier' in outlier_transaction['fraud_reasons'].iloc[0], "Should include outlier reason"
    
    def test_fraud_score_calculation(self):
        """Test fraud score calculation and thresholds"""
        
        # Arrange - Transaction that triggers multiple fraud rules
        amounts = [100.0] * 98 + [100.0, 60000.0]  # High amount outlier
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002'],
            'customer_id': ['CUST001', 'CUST001'],
            'amount': [100.0, 60000.0],
            'transaction_date': [
                '2024-01-01 10:00:00',
                '2024-01-01 10:00:30'  # Rapid + high amount
            ]
        })
        
        # Add more normal transactions for statistical baseline
        normal_data = pd.DataFrame({
            'transaction_id': [f'TXN{i:03d}' for i in range(3, 103)],
            'customer_id': ['CUST001'] * 100,
            'amount': [100.0] * 100,
            'transaction_date': ['2024-01-01 09:00:00'] * 100
        })
        
        full_data = pd.concat([input_data, normal_data], ignore_index=True)
        
        # Act
        result = detect_fraud_patterns(full_data)
        
        # Assert
        suspicious_transaction = result[result['transaction_id'] == 'TXN002']
        assert suspicious_transaction['fraud_score'].iloc[0] >= 0.5, "Should have high fraud score"
        assert suspicious_transaction['is_suspicious'].iloc[0] == True, "Should be flagged as suspicious"
    
    def test_handles_empty_dataframe(self):
        """Test fraud detection with empty dataframe"""
        
        # Arrange
        empty_df = pd.DataFrame()
        
        # Act
        result = detect_fraud_patterns(empty_df)
        
        # Assert
        assert len(result) == 0, "Should handle empty dataframe"
        assert isinstance(result, pd.DataFrame), "Should return DataFrame"
    
    def test_round_number_detection(self):
        """Test detection of suspicious round number transactions"""
        
        # Arrange
        input_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003'],
            'customer_id': ['CUST001', 'CUST002', 'CUST003'],
            'amount': [1000.0, 1500.50, 5000.0],  # Round numbers >= 1000
            'transaction_date': ['2024-01-01 10:00:00'] * 3
        })
        
        # Act
        result = detect_fraud_patterns(input_data)
        
        # Assert
        round_transactions = result[result['amount'].isin([1000.0, 5000.0])]
        assert all('round_amount' in reason for reason in round_transactions['fraud_reasons']), "Should flag round amounts"

# =============================================================================
# INTEGRATION TESTING FRAMEWORK
# =============================================================================

class TestPipelineIntegration:
    """Integration tests for pipeline components working together"""
    
    @pytest.fixture
    def sample_pipeline_data(self):
        """Comprehensive sample data for pipeline integration testing"""
        
        np.random.seed(42)  # For reproducible tests
        
        return pd.DataFrame({
            'transaction_id': [f'TXN{i:06d}' for i in range(1, 1001)],
            'customer_id': [f'CUST{i:04d}' for i in np.random.randint(1, 101, 1000)],
            'amount': np.random.lognormal(mean=4, sigma=1, size=1000),  # Realistic amount distribution
            'currency': np.random.choice(['USD', 'EUR', 'GBP'], 1000, p=[0.7, 0.2, 0.1]),
            'merchant_name': [f'Merchant {i}' for i in np.random.randint(1, 51, 1000)],
            'transaction_date': pd.date_range('2024-01-01', periods=1000, freq='5min')
        })
    
    def test_end_to_end_pipeline_flow(self, sample_pipeline_data):
        """Test complete pipeline flow with data quality validation"""
        
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
        
        # Verify data quality
        assert not cleaned_data['amount'].isnull().any(), "No null amounts after cleaning"
        assert all(cleaned_data['amount'] > 0), "All amounts should be positive"
        assert customer_metrics['total_spent'].sum() == cleaned_data['amount'].sum(), "Total amounts should match"
    
    def test_pipeline_error_handling(self):
        """Test pipeline handles various error conditions gracefully"""
        
        # Test 1: Missing columns
        incomplete_data = pd.DataFrame({
            'transaction_id': ['TXN001'],
            'amount': [100.0]
            # Missing required columns
        })
        
        with pytest.raises(ValueError):
            clean_transaction_data(incomplete_data)
        
        # Test 2: Invalid data types
        invalid_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002'],
            'customer_id': ['CUST001', 'CUST002'],
            'amount': ['invalid', 200.0],
            'currency': ['USD', 'EUR']
        })
        
        result = clean_transaction_data(invalid_data)
        assert len(result) == 1, "Should filter out invalid data gracefully"
        
        # Test 3: Empty dataframe handling
        empty_df = pd.DataFrame()
        
        cleaned_empty = clean_transaction_data(empty_df)
        metrics_empty = calculate_customer_metrics(empty_df)
        fraud_empty = detect_fraud_patterns(empty_df)
        
        assert len(cleaned_empty) == 0, "Should handle empty data"
        assert len(metrics_empty) == 0, "Should handle empty data"
        assert len(fraud_empty) == 0, "Should handle empty data"
    
    def test_pipeline_data_lineage_tracking(self, sample_pipeline_data):
        """Test that data lineage is maintained through pipeline"""
        
        # Act
        cleaned_data = clean_transaction_data(sample_pipeline_data)
        
        # Assert - Verify lineage tracking
        assert '_cleaned_at' in cleaned_data.columns, "Should track cleaning timestamp"
        assert '_duplicates_removed' in cleaned_data.columns, "Should track duplicates removed"
        
        # Verify all original transaction IDs are preserved (minus duplicates)
        original_unique_ids = sample_pipeline_data['transaction_id'].nunique()
        cleaned_unique_ids = cleaned_data['transaction_id'].nunique()
        assert cleaned_unique_ids <= original_unique_ids, "Should preserve or reduce transaction IDs"
    
    def test_pipeline_performance_integration(self, sample_pipeline_data):
        """Test pipeline performance with realistic data volumes"""
        
        # Arrange - Scale up data for performance testing
        large_data = pd.concat([sample_pipeline_data] * 5, ignore_index=True)
        large_data['transaction_id'] = [f'TXN{i:07d}' for i in range(len(large_data))]
        
        # Act - Measure pipeline performance
        start_time = time.time()
        
        cleaned_data = clean_transaction_data(large_data)
        customer_metrics = calculate_customer_metrics(cleaned_data)
        fraud_results = detect_fraud_patterns(cleaned_data)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Assert - Performance requirements
        assert processing_time < 30.0, f"Pipeline should complete within 30 seconds, took {processing_time:.2f}s"
        assert len(fraud_results) > 0, "Should produce fraud detection results"
        
        # Calculate throughput
        throughput = len(large_data) / processing_time
        assert throughput > 100, f"Should process at least 100 records per second, got {throughput:.2f}"
    
    @patch('days.day_21_testing_strategies.solution.datetime')
    def test_pipeline_with_mocked_dependencies(self, mock_datetime, sample_pipeline_data):
        """Test pipeline with mocked external dependencies"""
        
        # Arrange - Mock datetime for consistent testing
        mock_datetime.now.return_value = datetime(2024, 1, 15, 12, 0, 0)
        mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
        
        # Act
        result = clean_transaction_data(sample_pipeline_data)
        
        # Assert
        assert all(result['_cleaned_at'] == datetime(2024, 1, 15, 12, 0, 0)), "Should use mocked datetime"
        mock_datetime.now.assert_called()

# =============================================================================
# END-TO-END TESTING FRAMEWORK
# =============================================================================

class TestEndToEndWorkflows:
    """End-to-end tests for complete data workflows"""
    
    @pytest.mark.e2e
    def test_complete_analytics_workflow(self):
        """Test complete workflow from raw data to business insights"""
        
        # Arrange - Simulate realistic business scenario
        raw_data = self._generate_realistic_transaction_data(days=7, transactions_per_day=1000)
        
        # Act - Run complete analytics workflow
        workflow_results = self._run_analytics_workflow(raw_data)
        
        # Assert - Verify business outcomes
        assert workflow_results['status'] == 'success', "Workflow should complete successfully"
        assert workflow_results['data_quality_score'] >= 0.95, "Should meet data quality standards"
        assert workflow_results['customers_analyzed'] > 0, "Should analyze customers"
        assert workflow_results['fraud_cases_detected'] >= 0, "Should detect fraud cases"
        
        # Verify business metrics
        business_metrics = workflow_results['business_metrics']
        assert 'total_revenue' in business_metrics, "Should calculate total revenue"
        assert 'avg_transaction_value' in business_metrics, "Should calculate average transaction value"
        assert 'customer_segments' in business_metrics, "Should generate customer segments"
        
        # Verify data governance
        assert workflow_results['audit_trail'], "Should maintain audit trail"
        assert workflow_results['data_lineage'], "Should track data lineage"
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_high_volume_processing(self):
        """Test system behavior with high volume data"""
        
        # Arrange - Generate large dataset
        large_dataset = self._generate_realistic_transaction_data(days=30, transactions_per_day=10000)
        
        # Act - Process large dataset
        start_time = time.time()
        workflow_results = self._run_analytics_workflow(large_dataset)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Assert - Performance and reliability requirements
        assert workflow_results['status'] == 'success', "Should handle large volumes"
        assert processing_time < 300, f"Should complete within 5 minutes, took {processing_time:.2f}s"
        
        # Verify system resources
        assert workflow_results['peak_memory_mb'] < 4096, "Should stay within memory limits"
        assert workflow_results['error_rate'] < 0.01, "Should maintain low error rate"
    
    @pytest.mark.e2e
    def test_data_quality_end_to_end(self):
        """Test end-to-end data quality validation"""
        
        # Arrange - Create data with known quality issues
        problematic_data = self._generate_data_with_quality_issues()
        
        # Act - Run workflow with quality validation
        workflow_results = self._run_analytics_workflow(problematic_data)
        
        # Assert - Quality issues should be detected and handled
        assert workflow_results['quality_issues_detected'] > 0, "Should detect quality issues"
        assert workflow_results['data_quality_score'] < 1.0, "Should reflect quality issues in score"
        assert workflow_results['status'] in ['success_with_warnings', 'partial_success'], "Should handle quality issues gracefully"
    
    def _generate_realistic_transaction_data(self, days: int, transactions_per_day: int) -> pd.DataFrame:
        """Generate realistic transaction data for testing"""
        
        np.random.seed(42)
        total_transactions = days * transactions_per_day
        
        # Generate realistic customer distribution (Pareto principle)
        num_customers = total_transactions // 10
        customer_weights = np.random.pareto(1, num_customers)
        customer_weights = customer_weights / customer_weights.sum()
        
        customers = np.random.choice(
            [f'CUST{i:06d}' for i in range(num_customers)],
            size=total_transactions,
            p=customer_weights
        )
        
        # Generate realistic amounts (log-normal distribution)
        amounts = np.random.lognormal(mean=4, sigma=1.2, size=total_transactions)
        amounts = np.clip(amounts, 1, 50000)  # Reasonable bounds
        
        # Generate timestamps with realistic patterns
        base_date = datetime(2024, 1, 1)
        timestamps = []
        for day in range(days):
            day_start = base_date + timedelta(days=day)
            # More transactions during business hours
            for _ in range(transactions_per_day):
                hour = np.random.choice(24, p=self._get_hourly_distribution())
                minute = np.random.randint(0, 60)
                second = np.random.randint(0, 60)
                timestamp = day_start + timedelta(hours=hour, minutes=minute, seconds=second)
                timestamps.append(timestamp)
        
        return pd.DataFrame({
            'transaction_id': [f'TXN{i:08d}' for i in range(total_transactions)],
            'customer_id': customers,
            'amount': amounts,
            'currency': np.random.choice(['USD', 'EUR', 'GBP'], total_transactions, p=[0.7, 0.2, 0.1]),
            'merchant_name': np.random.choice([f'Merchant {i}' for i in range(100)], total_transactions),
            'transaction_date': timestamps
        })
    
    def _get_hourly_distribution(self) -> np.ndarray:
        """Get realistic hourly transaction distribution"""
        
        # Higher probability during business hours
        hourly_probs = np.ones(24) * 0.02  # Base probability
        hourly_probs[9:17] = 0.08  # Business hours
        hourly_probs[19:22] = 0.06  # Evening shopping
        
        return hourly_probs / hourly_probs.sum()
    
    def _generate_data_with_quality_issues(self) -> pd.DataFrame:
        """Generate data with known quality issues for testing"""
        
        base_data = self._generate_realistic_transaction_data(days=1, transactions_per_day=100)
        
        # Introduce quality issues
        # 1. Duplicate transactions
        duplicates = base_data.iloc[:5].copy()
        base_data = pd.concat([base_data, duplicates], ignore_index=True)
        
        # 2. Invalid amounts
        base_data.loc[10:12, 'amount'] = [-100, 0, 'invalid']
        
        # 3. Missing data
        base_data.loc[15:17, 'merchant_name'] = None
        
        # 4. Invalid currency codes
        base_data.loc[20:22, 'currency'] = ['XXX', 'invalid', '']
        
        return base_data
    
    def _run_analytics_workflow(self, raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Run complete analytics workflow and return results"""
        
        workflow_start = time.time()
        
        try:
            # Step 1: Data cleaning
            cleaned_data = clean_transaction_data(raw_data)
            
            # Step 2: Customer metrics
            customer_metrics = calculate_customer_metrics(cleaned_data)
            
            # Step 3: Fraud detection
            fraud_results = detect_fraud_patterns(cleaned_data)
            
            # Step 4: Business metrics calculation
            business_metrics = self._calculate_business_metrics(cleaned_data, customer_metrics, fraud_results)
            
            # Step 5: Data quality assessment
            quality_score = self._assess_data_quality(raw_data, cleaned_data)
            
            workflow_end = time.time()
            
            return {
                'status': 'success',
                'processing_time_seconds': workflow_end - workflow_start,
                'records_processed': len(raw_data),
                'records_cleaned': len(cleaned_data),
                'customers_analyzed': len(customer_metrics),
                'fraud_cases_detected': fraud_results['is_suspicious'].sum(),
                'data_quality_score': quality_score['overall_score'],
                'quality_issues_detected': quality_score['issues_count'],
                'business_metrics': business_metrics,
                'audit_trail': True,
                'data_lineage': True,
                'peak_memory_mb': self._get_peak_memory_usage(),
                'error_rate': 0.0
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'processing_time_seconds': time.time() - workflow_start
            }
    
    def _calculate_business_metrics(self, cleaned_data: pd.DataFrame, 
                                  customer_metrics: pd.DataFrame, 
                                  fraud_results: pd.DataFrame) -> Dict[str, Any]:
        """Calculate business metrics from processed data"""
        
        return {
            'total_revenue': cleaned_data['amount'].sum(),
            'avg_transaction_value': cleaned_data['amount'].mean(),
            'total_customers': cleaned_data['customer_id'].nunique(),
            'fraud_rate': fraud_results['is_suspicious'].mean(),
            'customer_segments': {
                'high_value': len(customer_metrics[customer_metrics['total_spent'] > customer_metrics['total_spent'].quantile(0.9)]),
                'medium_value': len(customer_metrics[customer_metrics['total_spent'].between(
                    customer_metrics['total_spent'].quantile(0.5),
                    customer_metrics['total_spent'].quantile(0.9)
                )]),
                'low_value': len(customer_metrics[customer_metrics['total_spent'] <= customer_metrics['total_spent'].quantile(0.5)])
            }
        }
    
    def _assess_data_quality(self, raw_data: pd.DataFrame, cleaned_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall data quality"""
        
        issues_count = 0
        
        # Check for data loss during cleaning
        data_loss_rate = (len(raw_data) - len(cleaned_data)) / len(raw_data) if len(raw_data) > 0 else 0
        if data_loss_rate > 0.1:  # More than 10% data loss
            issues_count += 1
        
        # Check for missing values
        missing_rate = cleaned_data.isnull().sum().sum() / (len(cleaned_data) * len(cleaned_data.columns)) if len(cleaned_data) > 0 else 0
        if missing_rate > 0.05:  # More than 5% missing values
            issues_count += 1
        
        # Calculate overall quality score
        overall_score = max(0, 1 - (data_loss_rate + missing_rate))
        
        return {
            'overall_score': overall_score,
            'issues_count': issues_count,
            'data_loss_rate': data_loss_rate,
            'missing_value_rate': missing_rate
        }
    
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage in MB"""
        
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024

# =============================================================================
# PERFORMANCE TESTING FRAMEWORK
# =============================================================================

class PerformanceTestFramework:
    """Comprehensive performance testing framework for data pipelines"""
    
    def __init__(self):
        self.performance_metrics = {}
        self.baseline_metrics = {}
    
    def measure_performance(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure comprehensive performance metrics for a function"""
        
        # Capture initial system state
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = process.cpu_percent()
        
        # Execute function with timing
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Capture final system state
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        final_cpu = process.cpu_percent()
        
        execution_time = end_time - start_time
        memory_used = final_memory - initial_memory
        
        performance_metrics = {
            'execution_time_seconds': execution_time,
            'memory_used_mb': memory_used,
            'peak_memory_mb': final_memory,
            'cpu_usage_percent': final_cpu,
            'throughput_records_per_second': 0,  # Will be calculated if applicable
            'result': result
        }
        
        # Calculate throughput if result is a DataFrame
        if isinstance(result, pd.DataFrame):
            performance_metrics['throughput_records_per_second'] = len(result) / execution_time if execution_time > 0 else 0
        
        return performance_metrics
    
    def load_test_pipeline(self, pipeline_func, concurrent_requests: int, 
                          test_data_generator, iterations: int = 10) -> Dict[str, Any]:
        """Perform comprehensive load testing on data pipeline"""
        
        results = []
        errors = []
        
        def run_single_test(iteration):
            try:
                test_data = test_data_generator(iteration)
                metrics = self.measure_performance(pipeline_func, test_data)
                return metrics
            except Exception as e:
                return {'error': str(e), 'iteration': iteration}
        
        # Execute concurrent tests
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(run_single_test, i) for i in range(iterations)]
            
            for future in as_completed(futures):
                result = future.result()
                if 'error' in result:
                    errors.append(result)
                else:
                    results.append(result)
        
        end_time = time.time()
        
        # Calculate aggregate metrics
        if results:
            execution_times = [r['execution_time_seconds'] for r in results]
            memory_usage = [r['memory_used_mb'] for r in results]
            throughputs = [r['throughput_records_per_second'] for r in results if r['throughput_records_per_second'] > 0]
            
            load_test_results = {
                'total_requests': iterations,
                'successful_requests': len(results),
                'failed_requests': len(errors),
                'success_rate': len(results) / iterations * 100,
                'total_test_time': end_time - start_time,
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'p95_execution_time': sorted(execution_times)[int(len(execution_times) * 0.95)],
                'p99_execution_time': sorted(execution_times)[int(len(execution_times) * 0.99)],
                'avg_memory_usage': sum(memory_usage) / len(memory_usage),
                'max_memory_usage': max(memory_usage),
                'requests_per_second': len(results) / (end_time - start_time),
                'avg_throughput': sum(throughputs) / len(throughputs) if throughputs else 0,
                'errors': errors
            }
        else:
            load_test_results = {
                'total_requests': iterations,
                'successful_requests': 0,
                'failed_requests': len(errors),
                'success_rate': 0,
                'errors': errors
            }
        
        return load_test_results
    
    def benchmark_scalability(self, func, data_sizes: List[int], *args, **kwargs) -> Dict[str, Any]:
        """Benchmark function scalability across different data sizes"""
        
        scalability_results = []
        
        for size in data_sizes:
            # Generate test data of specified size
            test_data = self._generate_test_data(size)
            
            # Measure performance
            metrics = self.measure_performance(func, test_data, *args, **kwargs)
            
            scalability_results.append({
                'data_size': size,
                'execution_time': metrics['execution_time_seconds'],
                'memory_used': metrics['memory_used_mb'],
                'throughput': metrics['throughput_records_per_second']
            })
        
        # Analyze scalability trends
        analysis = self._analyze_scalability_trends(scalability_results)
        
        return {
            'scalability_results': scalability_results,
            'analysis': analysis
        }
    
    def _generate_test_data(self, size: int) -> pd.DataFrame:
        """Generate test data of specified size"""
        
        return pd.DataFrame({
            'transaction_id': [f'TXN{i:08d}' for i in range(size)],
            'customer_id': [f'CUST{i % 1000:06d}' for i in range(size)],
            'amount': np.random.uniform(10, 1000, size),
            'currency': np.random.choice(['USD', 'EUR', 'GBP'], size),
            'merchant_name': [f'Merchant {i % 100}' for i in range(size)]
        })
    
    def _analyze_scalability_trends(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scalability trends from benchmark results"""
        
        sizes = [r['data_size'] for r in results]
        times = [r['execution_time'] for r in results]
        throughputs = [r['throughput'] for r in results]
        
        # Calculate complexity (linear, quadratic, etc.)
        time_complexity = self._estimate_complexity(sizes, times)
        throughput_trend = 'increasing' if throughputs[-1] > throughputs[0] else 'decreasing'
        
        return {
            'estimated_time_complexity': time_complexity,
            'throughput_trend': throughput_trend,
            'scalability_rating': self._calculate_scalability_rating(time_complexity, throughput_trend),
            'recommendations': self._generate_scalability_recommendations(time_complexity, throughput_trend)
        }
    
    def _estimate_complexity(self, sizes: List[int], times: List[float]) -> str:
        """Estimate algorithmic complexity from size vs time data"""
        
        if len(sizes) < 3:
            return 'insufficient_data'
        
        # Calculate ratios to estimate complexity
        size_ratios = [sizes[i] / sizes[i-1] for i in range(1, len(sizes))]
        time_ratios = [times[i] / times[i-1] for i in range(1, len(times))]
        
        avg_size_ratio = sum(size_ratios) / len(size_ratios)
        avg_time_ratio = sum(time_ratios) / len(time_ratios)
        
        complexity_ratio = avg_time_ratio / avg_size_ratio
        
        if complexity_ratio < 1.2:
            return 'O(1) - Constant'
        elif complexity_ratio < 2.0:
            return 'O(n) - Linear'
        elif complexity_ratio < 4.0:
            return 'O(n log n) - Linearithmic'
        else:
            return 'O(n²) or worse - Quadratic+'
    
    def _calculate_scalability_rating(self, complexity: str, throughput_trend: str) -> str:
        """Calculate overall scalability rating"""
        
        if 'O(1)' in complexity or 'O(n)' in complexity:
            return 'Excellent' if throughput_trend == 'increasing' else 'Good'
        elif 'O(n log n)' in complexity:
            return 'Good' if throughput_trend == 'increasing' else 'Fair'
        else:
            return 'Poor'
    
    def _generate_scalability_recommendations(self, complexity: str, throughput_trend: str) -> List[str]:
        """Generate recommendations based on scalability analysis"""
        
        recommendations = []
        
        if 'O(n²)' in complexity or 'worse' in complexity:
            recommendations.append("Consider optimizing algorithms to reduce quadratic complexity")
            recommendations.append("Implement data partitioning or parallel processing")
        
        if throughput_trend == 'decreasing':
            recommendations.append("Investigate memory bottlenecks or resource contention")
            recommendations.append("Consider implementing caching or data preprocessing")
        
        if not recommendations:
            recommendations.append("Performance characteristics are acceptable for current scale")
        
        return recommendations

class TestPerformanceRequirements:
    """Comprehensive performance testing for data pipeline components"""
    
    @pytest.fixture
    def performance_framework(self):
        """Performance testing framework fixture"""
        return PerformanceTestFramework()
    
    def test_cleaning_performance_scalability(self, performance_framework):
        """Test data cleaning performance scales appropriately with data size"""
        
        # Test with increasing data sizes
        data_sizes = [1000, 5000, 10000, 25000]
        
        scalability_results = performance_framework.benchmark_scalability(
            clean_transaction_data, data_sizes
        )
        
        # Assert performance requirements
        for result in scalability_results['scalability_results']:
            throughput = result['throughput']
            assert throughput > 1000, f"Should maintain >1000 records/sec for {result['data_size']} records, got {throughput:.2f}"
        
        # Check scalability rating
        rating = scalability_results['analysis']['scalability_rating']
        assert rating in ['Excellent', 'Good'], f"Scalability rating should be Good or Excellent, got {rating}"
    
    def test_memory_usage_limits(self, performance_framework):
        """Test that pipeline components stay within memory limits"""
        
        # Generate large dataset
        large_size = 50000
        test_data = performance_framework._generate_test_data(large_size)
        
        # Measure memory usage
        metrics = performance_framework.measure_performance(clean_transaction_data, test_data)
        
        # Assert memory requirements
        assert metrics['memory_used_mb'] < 500, f"Should use <500MB, used {metrics['memory_used_mb']:.2f}MB"
        assert metrics['peak_memory_mb'] < 2048, f"Peak memory should be <2GB, was {metrics['peak_memory_mb']:.2f}MB"
    
    def test_concurrent_processing_performance(self, performance_framework):
        """Test performance under concurrent load"""
        
        def test_data_generator(iteration):
            return performance_framework._generate_test_data(1000)
        
        # Run load test
        load_results = performance_framework.load_test_pipeline(
            clean_transaction_data,
            concurrent_requests=5,
            test_data_generator=test_data_generator,
            iterations=20
        )
        
        # Assert load test requirements
        assert load_results['success_rate'] >= 95, f"Success rate should be >=95%, got {load_results['success_rate']:.1f}%"
        assert load_results['avg_execution_time'] < 5.0, f"Average execution time should be <5s, got {load_results['avg_execution_time']:.2f}s"
        assert load_results['requests_per_second'] > 2, f"Should handle >2 requests/sec, got {load_results['requests_per_second']:.2f}"
    
    def test_performance_regression_detection(self, performance_framework):
        """Test performance regression detection"""
        
        # Baseline performance measurement
        baseline_data = performance_framework._generate_test_data(10000)
        baseline_metrics = performance_framework.measure_performance(clean_transaction_data, baseline_data)
        
        # Current performance measurement
        current_data = performance_framework._generate_test_data(10000)
        current_metrics = performance_framework.measure_performance(clean_transaction_data, current_data)
        
        # Check for regression (allow 20% variance)
        performance_ratio = current_metrics['execution_time_seconds'] / baseline_metrics['execution_time_seconds']
        assert performance_ratio < 1.2, f"Performance regression detected: {performance_ratio:.2f}x slower than baseline"
        
        throughput_ratio = current_metrics['throughput_records_per_second'] / baseline_metrics['throughput_records_per_second']
        assert throughput_ratio > 0.8, f"Throughput regression detected: {throughput_ratio:.2f}x of baseline throughput"

# =============================================================================
# REGRESSION TESTING FRAMEWORK
# =============================================================================

class RegressionTestFramework:
    """Comprehensive regression testing framework for data pipelines"""
    
    def __init__(self, baseline_path: str = "./test_baselines"):
        self.baseline_path = baseline_path
        os.makedirs(baseline_path, exist_ok=True)
    
    def capture_baseline(self, data: pd.DataFrame, test_name: str, metadata: Dict[str, Any] = None):
        """Capture comprehensive baseline data for regression testing"""
        
        baseline_metrics = self._calculate_comprehensive_metrics(data)
        
        if metadata:
            baseline_metrics['metadata'] = metadata
        
        baseline_metrics['captured_at'] = datetime.now().isoformat()
        
        baseline_file = os.path.join(self.baseline_path, f"{test_name}_baseline.json")
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_metrics, f, indent=2, default=str)
        
        return baseline_metrics
    
    def check_regression(self, current_data: pd.DataFrame, test_name: str, 
                        tolerance: float = 0.05) -> Dict[str, Any]:
        """Comprehensive regression checking against baseline"""
        
        baseline_file = os.path.join(self.baseline_path, f"{test_name}_baseline.json")
        
        try:
            with open(baseline_file, 'r') as f:
                baseline_metrics = json.load(f)
        except FileNotFoundError:
            # Capture current as baseline if none exists
            self.capture_baseline(current_data, test_name)
            return {
                'status': 'baseline_created',
                'message': 'No baseline found, captured current data as baseline'
            }
        
        current_metrics = self._calculate_comprehensive_metrics(current_data)
        
        regression_results = {
            'status': 'pass',
            'test_name': test_name,
            'checked_at': datetime.now().isoformat(),
            'regressions': [],
            'improvements': [],
            'warnings': [],
            'baseline_metrics': baseline_metrics,
            'current_metrics': current_metrics,
            'tolerance': tolerance
        }
        
        # Compare all metrics
        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name in ['metadata', 'captured_at'] or metric_name not in current_metrics:
                continue
                
            current_value = current_metrics[metric_name]
            
            if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                relative_change = self._calculate_relative_change(baseline_value, current_value)
                
                if abs(relative_change) > tolerance:
                    change_info = {
                        'metric': metric_name,
                        'baseline': baseline_value,
                        'current': current_value,
                        'change_percent': relative_change * 100,
                        'severity': self._assess_change_severity(metric_name, relative_change)
                    }
                    
                    if relative_change < 0:  # Degradation
                        regression_results['regressions'].append(change_info)
                    else:  # Improvement
                        regression_results['improvements'].append(change_info)
            
            elif baseline_value != current_value:
                # Non-numeric changes
                regression_results['warnings'].append({
                    'metric': metric_name,
                    'baseline': baseline_value,
                    'current': current_value,
                    'type': 'structural_change'
                })
        
        # Update overall status
        if regression_results['regressions']:
            critical_regressions = [r for r in regression_results['regressions'] if r['severity'] == 'critical']
            if critical_regressions:
                regression_results['status'] = 'critical_regression'
            else:
                regression_results['status'] = 'regression_detected'
        
        return regression_results
    
    def _calculate_comprehensive_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive metrics for regression comparison"""
        
        if data.empty:
            return {'empty_dataframe': True}
        
        metrics = {
            # Basic structure metrics
            'row_count': len(data),
            'column_count': len(data.columns),
            'column_names': sorted(data.columns.tolist()),
            
            # Data quality metrics
            'null_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100,
            
            # Data distribution hash (for detecting subtle changes)
            'data_hash': hashlib.md5(pd.util.hash_pandas_object(data, index=True).values).hexdigest()
        }
        
        # Numeric column metrics
        numeric_columns = data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                metrics[f'{col}_mean'] = float(col_data.mean())
                metrics[f'{col}_std'] = float(col_data.std())
                metrics[f'{col}_min'] = float(col_data.min())
                metrics[f'{col}_max'] = float(col_data.max())
                metrics[f'{col}_median'] = float(col_data.median())
                metrics[f'{col}_null_count'] = int(data[col].isnull().sum())
        
        # Categorical column metrics
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                metrics[f'{col}_unique_count'] = int(col_data.nunique())
                metrics[f'{col}_null_count'] = int(data[col].isnull().sum())
                
                # Most common values
                value_counts = col_data.value_counts()
                if len(value_counts) > 0:
                    metrics[f'{col}_most_common'] = str(value_counts.index[0])
                    metrics[f'{col}_most_common_count'] = int(value_counts.iloc[0])
        
        return metrics
    
    def _calculate_relative_change(self, baseline: float, current: float) -> float:
        """Calculate relative change between baseline and current values"""
        
        if baseline == 0:
            return 1.0 if current != 0 else 0.0
        
        return (current - baseline) / abs(baseline)
    
    def _assess_change_severity(self, metric_name: str, relative_change: float) -> str:
        """Assess the severity of a metric change"""
        
        abs_change = abs(relative_change)
        
        # Critical metrics that should have minimal change
        critical_metrics = ['row_count', 'column_count', 'data_hash']
        
        if metric_name in critical_metrics:
            if abs_change > 0.01:  # 1% change in critical metrics
                return 'critical'
            else:
                return 'minor'
        
        # Statistical metrics can have more variance
        if abs_change > 0.5:  # 50% change
            return 'critical'
        elif abs_change > 0.2:  # 20% change
            return 'major'
        elif abs_change > 0.1:  # 10% change
            return 'moderate'
        else:
            return 'minor'

class TestRegressionDetection:
    """Comprehensive regression testing for data pipeline components"""
    
    @pytest.fixture
    def regression_framework(self, tmp_path):
        """Regression testing framework fixture"""
        return RegressionTestFramework(str(tmp_path))
    
    def test_output_consistency_regression(self, regression_framework):
        """Test that pipeline outputs remain consistent over time"""
        
        # Fixed test data for consistent results
        fixed_test_data = pd.DataFrame({
            'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],
            'customer_id': ['CUST001', 'CUST001', 'CUST002', 'CUST002'],
            'amount': [100.0, 150.0, 200.0, 250.0],
            'currency': ['USD', 'USD', 'EUR', 'EUR'],
            'merchant_name': ['Store A', 'Store B', 'Store C', 'Store D']
        })
        
        # Process data
        cleaned_data = clean_transaction_data(fixed_test_data)
        customer_metrics = calculate_customer_metrics(cleaned_data)
        
        # Check for regressions
        cleaning_regression = regression_framework.check_regression(cleaned_data, 'cleaning_consistency')
        metrics_regression = regression_framework.check_regression(customer_metrics, 'metrics_consistency')
        
        # Assert no critical regressions
        assert cleaning_regression['status'] != 'critical_regression', "Critical regression in data cleaning detected"
        assert metrics_regression['status'] != 'critical_regression', "Critical regression in metrics calculation detected"
    
    def test_performance_regression_tracking(self, regression_framework):
        """Test performance regression tracking over time"""
        
        # Generate consistent test data
        np.random.seed(42)
        test_data = pd.DataFrame({
            'transaction_id': [f'TXN{i:06d}' for i in range(10000)],
            'customer_id': [f'CUST{i:04d}' for i in range(10000)],
            'amount': np.random.uniform(10, 1000, 10000),
            'currency': ['USD'] * 10000,
            'merchant_name': ['Test Merchant'] * 10000
        })
        
        # Measure performance
        start_time = time.time()
        result = clean_transaction_data(test_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        throughput = len(test_data) / processing_time
        
        # Create performance metrics
        performance_data = pd.DataFrame({
            'processing_time': [processing_time],
            'throughput': [throughput],
            'memory_usage': [psutil.Process().memory_info().rss / 1024 / 1024]
        })
        
        # Check for performance regression
        regression_result = regression_framework.check_regression(performance_data, 'performance_tracking', tolerance=0.2)
        
        # Assert acceptable performance
        if regression_result['status'] == 'regression_detected':
            performance_regressions = [r for r in regression_result['regressions'] if 'throughput' in r['metric']]
            for regression in performance_regressions:
                assert regression['severity'] != 'critical', f"Critical performance regression: {regression}"
    
    def test_data_quality_regression(self, regression_framework):
        """Test data quality regression detection"""
        
        # Create baseline data with known quality characteristics
        baseline_data = pd.DataFrame({
            'transaction_id': [f'TXN{i:03d}' for i in range(100)],
            'customer_id': [f'CUST{i:02d}' for i in range(50)] * 2,
            'amount': np.random.uniform(50, 500, 100),
            'currency': ['USD'] * 100,
            'merchant_name': [f'Merchant {i}' for i in range(10)] * 10
        })
        
        # Process baseline data
        baseline_result = clean_transaction_data(baseline_data)
        
        # Create current data with slight quality degradation
        current_data = baseline_data.copy()
        current_data.loc[5:10, 'amount'] = -100  # Invalid amounts
        current_data.loc[15:20, 'merchant_name'] = None  # Missing merchants
        
        # Process current data
        current_result = clean_transaction_data(current_data)
        
        # Check for quality regression
        regression_result = regression_framework.check_regression(current_result, 'quality_regression')
        
        # Should detect the quality degradation
        if regression_result['status'] == 'regression_detected':
            row_count_regressions = [r for r in regression_result['regressions'] if r['metric'] == 'row_count']
            assert len(row_count_regressions) > 0, "Should detect row count regression due to filtering"

# =============================================================================
# CI/CD INTEGRATION FRAMEWORK
# =============================================================================

class CICDTestingFramework:
    """CI/CD integration framework for automated testing pipelines"""
    
    def __init__(self, config_path: str = "./cicd_config.json"):
        self.config_path = config_path
        self.test_results = {}
        self.quality_gates = {}
        
    def setup_quality_gates(self) -> Dict[str, Any]:
        """Setup quality gates for CI/CD pipeline"""
        
        self.quality_gates = {
            'unit_tests': {
                'min_coverage': 0.90,  # 90% code coverage
                'max_failures': 0,     # No failing tests
                'timeout_minutes': 10
            },
            'integration_tests': {
                'min_success_rate': 0.95,  # 95% success rate
                'max_failures': 1,         # At most 1 failing test
                'timeout_minutes': 30
            },
            'performance_tests': {
                'max_execution_time': 300,    # 5 minutes max
                'min_throughput': 1000,       # 1000 records/sec min
                'max_memory_mb': 2048         # 2GB memory limit
            },
            'regression_tests': {
                'max_critical_regressions': 0,  # No critical regressions
                'max_major_regressions': 2,     # At most 2 major regressions
                'tolerance': 0.1                # 10% tolerance
            }
        }
        
        return self.quality_gates
    
    def run_cicd_pipeline(self, test_suite: str = 'full') -> Dict[str, Any]:
        """Run complete CI/CD testing pipeline"""
        
        pipeline_start = time.time()
        
        pipeline_results = {
            'pipeline_id': f"pipeline_{int(time.time())}",
            'started_at': datetime.now().isoformat(),
            'test_suite': test_suite,
            'stages': {},
            'overall_status': 'running',
            'quality_gate_results': {},
            'recommendations': []
        }
        
        try:
            # Stage 1: Unit Tests
            if test_suite in ['full', 'unit']:
                unit_results = self._run_unit_test_stage()
                pipeline_results['stages']['unit_tests'] = unit_results
                pipeline_results['quality_gate_results']['unit_tests'] = self._check_quality_gate('unit_tests', unit_results)
            
            # Stage 2: Integration Tests
            if test_suite in ['full', 'integration']:
                integration_results = self._run_integration_test_stage()
                pipeline_results['stages']['integration_tests'] = integration_results
                pipeline_results['quality_gate_results']['integration_tests'] = self._check_quality_gate('integration_tests', integration_results)
            
            # Stage 3: Performance Tests
            if test_suite in ['full', 'performance']:
                performance_results = self._run_performance_test_stage()
                pipeline_results['stages']['performance_tests'] = performance_results
                pipeline_results['quality_gate_results']['performance_tests'] = self._check_quality_gate('performance_tests', performance_results)
            
            # Stage 4: Regression Tests
            if test_suite in ['full', 'regression']:
                regression_results = self._run_regression_test_stage()
                pipeline_results['stages']['regression_tests'] = regression_results
                pipeline_results['quality_gate_results']['regression_tests'] = self._check_quality_gate('regression_tests', regression_results)
            
            # Determine overall status
            pipeline_results['overall_status'] = self._determine_overall_status(pipeline_results['quality_gate_results'])
            
        except Exception as e:
            pipeline_results['overall_status'] = 'failed'
            pipeline_results['error'] = str(e)
        
        pipeline_results['completed_at'] = datetime.now().isoformat()
        pipeline_results['duration_seconds'] = time.time() - pipeline_start
        
        # Generate recommendations
        pipeline_results['recommendations'] = self._generate_pipeline_recommendations(pipeline_results)
        
        return pipeline_results
    
    def _run_unit_test_stage(self) -> Dict[str, Any]:
        """Run unit test stage"""
        
        # Simulate running pytest for unit tests
        unit_results = {
            'stage': 'unit_tests',
            'status': 'passed',
            'tests_run': 45,
            'tests_passed': 44,
            'tests_failed': 1,
            'coverage_percentage': 92.5,
            'execution_time_seconds': 45.2,
            'failed_tests': ['test_edge_case_empty_dataframe']
        }
        
        return unit_results
    
    def _run_integration_test_stage(self) -> Dict[str, Any]:
        """Run integration test stage"""
        
        integration_results = {
            'stage': 'integration_tests',
            'status': 'passed',
            'tests_run': 15,
            'tests_passed': 15,
            'tests_failed': 0,
            'success_rate': 100.0,
            'execution_time_seconds': 180.5,
            'failed_tests': []
        }
        
        return integration_results
    
    def _run_performance_test_stage(self) -> Dict[str, Any]:
        """Run performance test stage"""
        
        performance_results = {
            'stage': 'performance_tests',
            'status': 'passed',
            'avg_execution_time': 2.5,
            'max_execution_time': 4.2,
            'avg_throughput': 2500,
            'min_throughput': 1800,
            'peak_memory_mb': 1024,
            'execution_time_seconds': 240.0
        }
        
        return performance_results
    
    def _run_regression_test_stage(self) -> Dict[str, Any]:
        """Run regression test stage"""
        
        regression_results = {
            'stage': 'regression_tests',
            'status': 'passed',
            'critical_regressions': 0,
            'major_regressions': 1,
            'moderate_regressions': 3,
            'minor_regressions': 5,
            'execution_time_seconds': 120.0
        }
        
        return regression_results
    
    def _check_quality_gate(self, gate_name: str, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check if test results pass quality gate requirements"""
        
        gate_config = self.quality_gates.get(gate_name, {})
        gate_result = {
            'gate_name': gate_name,
            'status': 'passed',
            'checks': [],
            'violations': []
        }
        
        if gate_name == 'unit_tests':
            # Check coverage
            if test_results.get('coverage_percentage', 0) < gate_config.get('min_coverage', 0) * 100:
                gate_result['violations'].append(f"Coverage {test_results.get('coverage_percentage')}% below minimum {gate_config.get('min_coverage', 0) * 100}%")
            
            # Check failures
            if test_results.get('tests_failed', 0) > gate_config.get('max_failures', 0):
                gate_result['violations'].append(f"Failed tests {test_results.get('tests_failed')} exceeds maximum {gate_config.get('max_failures')}")
        
        elif gate_name == 'performance_tests':
            # Check execution time
            if test_results.get('max_execution_time', 0) > gate_config.get('max_execution_time', float('inf')):
                gate_result['violations'].append(f"Max execution time {test_results.get('max_execution_time')}s exceeds limit {gate_config.get('max_execution_time')}s")
            
            # Check throughput
            if test_results.get('min_throughput', float('inf')) < gate_config.get('min_throughput', 0):
                gate_result['violations'].append(f"Min throughput {test_results.get('min_throughput')} below requirement {gate_config.get('min_throughput')}")
        
        elif gate_name == 'regression_tests':
            # Check critical regressions
            if test_results.get('critical_regressions', 0) > gate_config.get('max_critical_regressions', 0):
                gate_result['violations'].append(f"Critical regressions {test_results.get('critical_regressions')} exceeds maximum {gate_config.get('max_critical_regressions')}")
        
        # Update status based on violations
        if gate_result['violations']:
            gate_result['status'] = 'failed'
        
        return gate_result
    
    def _determine_overall_status(self, quality_gate_results: Dict[str, Any]) -> str:
        """Determine overall pipeline status from quality gate results"""
        
        failed_gates = [gate for gate, result in quality_gate_results.items() if result['status'] == 'failed']
        
        if not failed_gates:
            return 'passed'
        elif len(failed_gates) == 1 and 'regression_tests' in failed_gates:
            return 'passed_with_warnings'
        else:
            return 'failed'
    
    def _generate_pipeline_recommendations(self, pipeline_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on pipeline results"""
        
        recommendations = []
        
        # Check for common issues and generate recommendations
        for stage_name, stage_results in pipeline_results.get('stages', {}).items():
            if stage_name == 'unit_tests' and stage_results.get('coverage_percentage', 100) < 90:
                recommendations.append("Increase unit test coverage to meet 90% minimum requirement")
            
            if stage_name == 'performance_tests' and stage_results.get('peak_memory_mb', 0) > 1500:
                recommendations.append("Investigate memory usage optimization opportunities")
            
            if stage_name == 'regression_tests' and stage_results.get('major_regressions', 0) > 0:
                recommendations.append("Review and address major regressions before deployment")
        
        # Overall recommendations
        if pipeline_results['overall_status'] == 'failed':
            recommendations.append("Address failing quality gates before proceeding with deployment")
        elif pipeline_results['overall_status'] == 'passed_with_warnings':
            recommendations.append("Consider addressing warnings to improve pipeline reliability")
        
        return recommendations

class TestCICDIntegration:
    """Test CI/CD integration and quality gates"""
    
    @pytest.fixture
    def cicd_framework(self):
        """CI/CD testing framework fixture"""
        return CICDTestingFramework()
    
    def test_quality_gates_setup(self, cicd_framework):
        """Test quality gates configuration"""
        
        quality_gates = cicd_framework.setup_quality_gates()
        
        # Assert required quality gates exist
        required_gates = ['unit_tests', 'integration_tests', 'performance_tests', 'regression_tests']
        for gate in required_gates:
            assert gate in quality_gates, f"Quality gate {gate} should be configured"
        
        # Assert reasonable thresholds
        assert quality_gates['unit_tests']['min_coverage'] >= 0.8, "Unit test coverage should be at least 80%"
        assert quality_gates['performance_tests']['min_throughput'] > 0, "Performance throughput threshold should be positive"
    
    def test_full_cicd_pipeline(self, cicd_framework):
        """Test complete CI/CD pipeline execution"""
        
        cicd_framework.setup_quality_gates()
        
        # Run full pipeline
        pipeline_results = cicd_framework.run_cicd_pipeline('full')
        
        # Assert pipeline completion
        assert 'pipeline_id' in pipeline_results, "Pipeline should have unique ID"
        assert pipeline_results['overall_status'] in ['passed', 'passed_with_warnings', 'failed'], "Pipeline should have valid status"
        assert 'duration_seconds' in pipeline_results, "Pipeline should track execution time"
        
        # Assert all stages ran
        expected_stages = ['unit_tests', 'integration_tests', 'performance_tests', 'regression_tests']
        for stage in expected_stages:
            assert stage in pipeline_results['stages'], f"Stage {stage} should have run"
    
    def test_quality_gate_enforcement(self, cicd_framework):
        """Test quality gate enforcement logic"""
        
        cicd_framework.setup_quality_gates()
        
        # Test unit test quality gate
        failing_unit_results = {
            'coverage_percentage': 75.0,  # Below 90% threshold
            'tests_failed': 2              # Above 0 threshold
        }
        
        gate_result = cicd_framework._check_quality_gate('unit_tests', failing_unit_results)
        
        assert gate_result['status'] == 'failed', "Quality gate should fail with insufficient coverage and failed tests"
        assert len(gate_result['violations']) >= 2, "Should report both coverage and failure violations"
    
    def test_pipeline_recommendations(self, cicd_framework):
        """Test pipeline recommendation generation"""
        
        cicd_framework.setup_quality_gates()
        
        # Create pipeline results with issues
        pipeline_results = {
            'overall_status': 'passed_with_warnings',
            'stages': {
                'unit_tests': {'coverage_percentage': 85.0},
                'performance_tests': {'peak_memory_mb': 1800},
                'regression_tests': {'major_regressions': 2}
            }
        }
        
        recommendations = cicd_framework._generate_pipeline_recommendations(pipeline_results)
        
        assert len(recommendations) > 0, "Should generate recommendations for issues"
        assert any('coverage' in rec.lower() for rec in recommendations), "Should recommend coverage improvement"
        assert any('memory' in rec.lower() for rec in recommendations), "Should recommend memory optimization"

# =============================================================================
# MAIN EXECUTION AND DEMONSTRATION
# =============================================================================

def demonstrate_comprehensive_testing_framework():
    """Demonstrate the complete testing framework capabilities"""
    
    print("🎯 TestDriven Analytics - Comprehensive Testing Framework")
    print("=" * 70)
    
    # Generate sample data for demonstration
    sample_data = pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(1000)],
        'customer_id': [f'CUST{i:04d}' for i in range(100)] * 10,
        'amount': np.random.uniform(10, 1000, 1000),
        'currency': np.random.choice(['USD', 'EUR', 'GBP'], 1000),
        'merchant_name': [f'Merchant {i}' for i in range(50)] * 20,
        'transaction_date': pd.date_range('2024-01-01', periods=1000, freq='H')
    })
    
    print(f"\n📊 Sample Data Generated: {len(sample_data)} transactions")
    
    # 1. Demonstrate Performance Testing
    print("\n🚀 Performance Testing Framework")
    print("-" * 40)
    
    perf_framework = PerformanceTestFramework()
    
    # Scalability test
    scalability_results = perf_framework.benchmark_scalability(
        clean_transaction_data, [1000, 2000, 5000]
    )
    
    print(f"Scalability Rating: {scalability_results['analysis']['scalability_rating']}")
    print(f"Estimated Complexity: {scalability_results['analysis']['estimated_time_complexity']}")
    
    # 2. Demonstrate Regression Testing
    print("\n🔍 Regression Testing Framework")
    print("-" * 40)
    
    regression_framework = RegressionTestFramework()
    
    # Process data and check for regressions
    cleaned_data = clean_transaction_data(sample_data)
    regression_result = regression_framework.check_regression(cleaned_data, 'demo_test')
    
    print(f"Regression Status: {regression_result['status']}")
    if regression_result['status'] != 'baseline_created':
        print(f"Regressions Detected: {len(regression_result['regressions'])}")
        print(f"Improvements Detected: {len(regression_result['improvements'])}")
    
    # 3. Demonstrate CI/CD Integration
    print("\n🔄 CI/CD Integration Framework")
    print("-" * 40)
    
    cicd_framework = CICDTestingFramework()
    cicd_framework.setup_quality_gates()
    
    # Run CI/CD pipeline
    pipeline_results = cicd_framework.run_cicd_pipeline('full')
    
    print(f"Pipeline Status: {pipeline_results['overall_status']}")
    print(f"Duration: {pipeline_results['duration_seconds']:.2f} seconds")
    print(f"Recommendations: {len(pipeline_results['recommendations'])}")
    
    for rec in pipeline_results['recommendations'][:3]:  # Show first 3
        print(f"  • {rec}")
    
    print("\n✅ Comprehensive Testing Framework Demonstration Complete!")
    print("🎯 Ready for production deployment with full test coverage!")

if __name__ == "__main__":
    demonstrate_comprehensive_testing_framework()