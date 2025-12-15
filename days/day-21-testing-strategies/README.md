# Day 21: Testing Strategies - Unit, Integration, End-to-End Testing

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** comprehensive testing strategies for data pipelines and analytics systems
- **Implement** unit, integration, and end-to-end testing frameworks for data workflows
- **Build** automated CI/CD pipelines with comprehensive test coverage and quality gates
- **Design** data-specific testing patterns including schema, regression, and performance testing
- **Deploy** production-ready testing infrastructure with monitoring and alerting

---

## Theory

### Production Data Testing Framework

Testing data pipelines requires specialized approaches that go beyond traditional software testing. Data systems have unique challenges including data quality, schema evolution, performance at scale, and complex dependencies that require comprehensive testing strategies.

#### 1. The Data Testing Pyramid

**Unit Tests (Foundation - 70%)**:
```python
# Example: Unit test for data transformation function
import pytest
import pandas as pd
from datetime import datetime, timedelta

def clean_customer_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize customer data"""
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['customer_id'])
    
    # Standardize email format
    df['email'] = df['email'].str.lower().str.strip()
    
    # Validate age range
    df = df[(df['age'] >= 13) & (df['age'] <= 120)]
    
    # Fill missing names with 'Unknown'
    df['first_name'] = df['first_name'].fillna('Unknown')
    df['last_name'] = df['last_name'].fillna('Unknown')
    
    return df

class TestCustomerDataCleaning:
    """Comprehensive unit tests for customer data cleaning"""
    
    def test_removes_duplicates(self):
        """Test that duplicate customer records are removed"""
        
        # Arrange
        input_data = pd.DataFrame({
            'customer_id': [1, 1, 2, 3],
            'email': ['test@example.com', 'test@example.com', 'user@test.com', 'admin@test.com'],
            'age': [25, 25, 30, 35],
            'first_name': ['John', 'John', 'Jane', 'Bob'],
            'last_name': ['Doe', 'Doe', 'Smith', 'Johnson']
        })
        
        # Act
        result = clean_customer_data(input_data)
        
        # Assert
        assert len(result) == 3, "Should remove duplicate customer records"
        assert result['customer_id'].nunique() == 3, "All customer IDs should be unique"
    
    def test_standardizes_email_format(self):
        """Test that email addresses are standardized to lowercase"""
        
        # Arrange
        input_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'email': ['TEST@EXAMPLE.COM', '  user@Test.com  ', 'Admin@TEST.COM'],
            'age': [25, 30, 35],
            'first_name': ['John', 'Jane', 'Bob'],
            'last_name': ['Doe', 'Smith', 'Johnson']
        })
        
        # Act
        result = clean_customer_data(input_data)
        
        # Assert
        expected_emails = ['test@example.com', 'user@test.com', 'admin@test.com']
        assert result['email'].tolist() == expected_emails, "Emails should be lowercase and trimmed"
    
    def test_validates_age_range(self):
        """Test that invalid ages are filtered out"""
        
        # Arrange
        input_data = pd.DataFrame({
            'customer_id': [1, 2, 3, 4, 5],
            'email': ['test1@example.com', 'test2@example.com', 'test3@example.com', 'test4@example.com', 'test5@example.com'],
            'age': [12, 25, 150, 30, -5],  # Invalid ages: 12, 150, -5
            'first_name': ['John', 'Jane', 'Bob', 'Alice', 'Charlie'],
            'last_name': ['Doe', 'Smith', 'Johnson', 'Brown', 'Wilson']
        })
        
        # Act
        result = clean_customer_data(input_data)
        
        # Assert
        assert len(result) == 2, "Should filter out invalid ages"
        assert all(result['age'].between(13, 120)), "All ages should be in valid range"
    
    def test_handles_missing_names(self):
        """Test that missing names are filled with 'Unknown'"""
        
        # Arrange
        input_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'email': ['test1@example.com', 'test2@example.com', 'test3@example.com'],
            'age': [25, 30, 35],
            'first_name': [None, 'Jane', None],
            'last_name': ['Doe', None, 'Johnson']
        })
        
        # Act
        result = clean_customer_data(input_data)
        
        # Assert
        assert result.loc[0, 'first_name'] == 'Unknown', "Missing first name should be filled"
        assert result.loc[1, 'last_name'] == 'Unknown', "Missing last name should be filled"
        assert not result[['first_name', 'last_name']].isnull().any().any(), "No names should be null"
    
    @pytest.mark.parametrize("input_df,expected_count", [
        (pd.DataFrame({'customer_id': [], 'email': [], 'age': [], 'first_name': [], 'last_name': []}), 0),
        (pd.DataFrame({'customer_id': [1], 'email': ['test@example.com'], 'age': [25], 'first_name': ['John'], 'last_name': ['Doe']}), 1)
    ])
    def test_edge_cases(self, input_df, expected_count):
        """Test edge cases like empty dataframes"""
        
        # Act
        result = clean_customer_data(input_df)
        
        # Assert
        assert len(result) == expected_count, f"Should handle edge case correctly"
```

**Integration Tests (Middle - 20%)**:
```python
# Example: Integration test for pipeline components
import pytest
from unittest.mock import Mock, patch
import pandas as pd

class TestDataPipelineIntegration:
    """Integration tests for data pipeline components"""
    
    @pytest.fixture
    def sample_raw_data(self):
        """Sample raw data for testing"""
        return pd.DataFrame({
            'customer_id': [1, 2, 3, 4],
            'email': ['test@example.com', 'user@test.com', 'admin@test.com', 'guest@test.com'],
            'purchase_amount': [100.50, 250.75, 75.25, 300.00],
            'purchase_date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18'],
            'product_category': ['electronics', 'clothing', 'books', 'electronics']
        })
    
    def test_extract_transform_load_pipeline(self, sample_raw_data):
        """Test complete ETL pipeline integration"""
        
        # Mock external dependencies
        with patch('pipeline.extract_from_source') as mock_extract, \
             patch('pipeline.load_to_warehouse') as mock_load:
            
            # Arrange
            mock_extract.return_value = sample_raw_data
            mock_load.return_value = True
            
            # Act
            from pipeline import run_etl_pipeline
            result = run_etl_pipeline('2024-01-15')
            
            # Assert
            assert result['status'] == 'success', "Pipeline should complete successfully"
            assert result['records_processed'] == 4, "Should process all records"
            mock_extract.assert_called_once_with('2024-01-15')
            mock_load.assert_called_once()
    
    def test_data_quality_integration(self, sample_raw_data):
        """Test integration between transformation and quality validation"""
        
        # Act
        from pipeline import transform_data, validate_data_quality
        
        transformed_data = transform_data(sample_raw_data)
        quality_results = validate_data_quality(transformed_data)
        
        # Assert
        assert quality_results['overall_score'] >= 0.95, "Data quality should meet threshold"
        assert quality_results['completeness_score'] == 1.0, "All required fields should be complete"
        assert len(quality_results['failed_checks']) == 0, "No quality checks should fail"
    
    def test_schema_evolution_handling(self):
        """Test pipeline handles schema changes gracefully"""
        
        # Arrange - data with new column
        evolved_data = pd.DataFrame({
            'customer_id': [1, 2],
            'email': ['test@example.com', 'user@test.com'],
            'purchase_amount': [100.50, 250.75],
            'purchase_date': ['2024-01-15', '2024-01-16'],
            'product_category': ['electronics', 'clothing'],
            'loyalty_tier': ['gold', 'silver']  # New column
        })
        
        # Act
        from pipeline import handle_schema_evolution, transform_data
        
        schema_result = handle_schema_evolution(evolved_data)
        transformed_data = transform_data(evolved_data)
        
        # Assert
        assert schema_result['new_columns'] == ['loyalty_tier'], "Should detect new column"
        assert 'loyalty_tier' in transformed_data.columns, "Should preserve new column"
        assert len(transformed_data) == 2, "Should process all records"
```

**End-to-End Tests (Top - 10%)**:
```python
# Example: End-to-end test for complete data workflow
import pytest
import pandas as pd
from datetime import datetime, timedelta

class TestEndToEndDataWorkflow:
    """End-to-end tests for complete data workflows"""
    
    @pytest.mark.e2e
    def test_complete_customer_analytics_workflow(self):
        """Test complete workflow from raw data to analytics"""
        
        # Arrange - Set up test environment
        test_date = '2024-01-15'
        
        # Act - Run complete workflow
        from workflows import CustomerAnalyticsWorkflow
        
        workflow = CustomerAnalyticsWorkflow()
        result = workflow.run_daily_processing(test_date)
        
        # Assert - Verify end-to-end results
        assert result['status'] == 'success', "Workflow should complete successfully"
        
        # Verify data was processed correctly
        assert result['customers_processed'] > 0, "Should process customer data"
        assert result['analytics_generated'], "Should generate analytics"
        
        # Verify output data quality
        output_data = workflow.get_output_data(test_date)
        assert len(output_data) > 0, "Should produce output data"
        assert output_data['customer_id'].nunique() == len(output_data), "Customer IDs should be unique"
        
        # Verify analytics accuracy
        analytics = workflow.get_analytics_results(test_date)
        assert 'total_revenue' in analytics, "Should calculate total revenue"
        assert 'customer_segments' in analytics, "Should generate customer segments"
        assert analytics['data_quality_score'] >= 0.95, "Should meet quality standards"
    
    @pytest.mark.e2e
    @pytest.mark.slow
    def test_performance_at_scale(self):
        """Test pipeline performance with large dataset"""
        
        # Arrange - Generate large test dataset
        large_dataset_size = 100000
        
        # Act
        from workflows import PerformanceTestWorkflow
        
        workflow = PerformanceTestWorkflow()
        start_time = datetime.now()
        result = workflow.process_large_dataset(large_dataset_size)
        end_time = datetime.now()
        
        processing_time = (end_time - start_time).total_seconds()
        
        # Assert - Performance requirements
        assert result['status'] == 'success', "Should handle large dataset"
        assert processing_time < 300, "Should complete within 5 minutes"  # SLA requirement
        assert result['records_per_second'] > 300, "Should meet throughput requirement"
        assert result['memory_usage_mb'] < 2048, "Should stay within memory limits"
```

#### 2. Data-Specific Testing Patterns

**Schema Testing**:
```python
# Example: Comprehensive schema testing
import pytest
import pandas as pd
from typing import Dict, Any

class SchemaValidator:
    """Validate data schemas and detect changes"""
    
    def __init__(self, expected_schema: Dict[str, Any]):
        self.expected_schema = expected_schema
    
    def validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataframe against expected schema"""
        
        validation_results = {
            'is_valid': True,
            'missing_columns': [],
            'extra_columns': [],
            'type_mismatches': [],
            'constraint_violations': []
        }
        
        # Check for missing columns
        expected_columns = set(self.expected_schema.keys())
        actual_columns = set(df.columns)
        
        validation_results['missing_columns'] = list(expected_columns - actual_columns)
        validation_results['extra_columns'] = list(actual_columns - expected_columns)
        
        # Check data types
        for column, expected_config in self.expected_schema.items():
            if column in df.columns:
                expected_type = expected_config['type']
                actual_type = str(df[column].dtype)
                
                if not self._types_compatible(actual_type, expected_type):
                    validation_results['type_mismatches'].append({
                        'column': column,
                        'expected': expected_type,
                        'actual': actual_type
                    })
                
                # Check constraints
                if 'constraints' in expected_config:
                    constraint_violations = self._check_constraints(df[column], expected_config['constraints'])
                    if constraint_violations:
                        validation_results['constraint_violations'].extend(constraint_violations)
        
        # Overall validation status
        validation_results['is_valid'] = (
            len(validation_results['missing_columns']) == 0 and
            len(validation_results['type_mismatches']) == 0 and
            len(validation_results['constraint_violations']) == 0
        )
        
        return validation_results
    
    def _types_compatible(self, actual_type: str, expected_type: str) -> bool:
        """Check if data types are compatible"""
        
        type_mappings = {
            'int64': ['integer', 'int', 'number'],
            'float64': ['float', 'number', 'decimal'],
            'object': ['string', 'text', 'varchar'],
            'datetime64[ns]': ['datetime', 'timestamp'],
            'bool': ['boolean', 'bool']
        }
        
        compatible_types = type_mappings.get(actual_type, [])
        return expected_type.lower() in compatible_types
    
    def _check_constraints(self, series: pd.Series, constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check data constraints"""
        
        violations = []
        
        # Check null constraints
        if constraints.get('nullable', True) is False:
            null_count = series.isnull().sum()
            if null_count > 0:
                violations.append({
                    'type': 'null_constraint',
                    'column': series.name,
                    'violation': f'{null_count} null values found'
                })
        
        # Check range constraints
        if 'min_value' in constraints:
            min_violations = (series < constraints['min_value']).sum()
            if min_violations > 0:
                violations.append({
                    'type': 'min_value_constraint',
                    'column': series.name,
                    'violation': f'{min_violations} values below minimum {constraints["min_value"]}'
                })
        
        if 'max_value' in constraints:
            max_violations = (series > constraints['max_value']).sum()
            if max_violations > 0:
                violations.append({
                    'type': 'max_value_constraint',
                    'column': series.name,
                    'violation': f'{max_violations} values above maximum {constraints["max_value"]}'
                })
        
        return violations

class TestSchemaValidation:
    """Test schema validation functionality"""
    
    @pytest.fixture
    def customer_schema(self):
        """Expected customer data schema"""
        return {
            'customer_id': {
                'type': 'integer',
                'constraints': {'nullable': False, 'min_value': 1}
            },
            'email': {
                'type': 'string',
                'constraints': {'nullable': False}
            },
            'age': {
                'type': 'integer',
                'constraints': {'nullable': True, 'min_value': 13, 'max_value': 120}
            },
            'created_at': {
                'type': 'datetime',
                'constraints': {'nullable': False}
            }
        }
    
    def test_valid_schema_passes(self, customer_schema):
        """Test that valid schema passes validation"""
        
        # Arrange
        valid_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'email': ['test@example.com', 'user@test.com', 'admin@test.com'],
            'age': [25, 30, 35],
            'created_at': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        })
        
        validator = SchemaValidator(customer_schema)
        
        # Act
        result = validator.validate_schema(valid_data)
        
        # Assert
        assert result['is_valid'], "Valid schema should pass validation"
        assert len(result['missing_columns']) == 0, "No columns should be missing"
        assert len(result['type_mismatches']) == 0, "No type mismatches should occur"
    
    def test_missing_columns_detected(self, customer_schema):
        """Test that missing columns are detected"""
        
        # Arrange - Missing 'age' column
        invalid_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'email': ['test@example.com', 'user@test.com', 'admin@test.com'],
            'created_at': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        })
        
        validator = SchemaValidator(customer_schema)
        
        # Act
        result = validator.validate_schema(invalid_data)
        
        # Assert
        assert not result['is_valid'], "Invalid schema should fail validation"
        assert 'age' in result['missing_columns'], "Missing age column should be detected"
    
    def test_constraint_violations_detected(self, customer_schema):
        """Test that constraint violations are detected"""
        
        # Arrange - Age constraint violation
        invalid_data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'email': ['test@example.com', 'user@test.com', 'admin@test.com'],
            'age': [25, 150, 35],  # 150 exceeds max_value constraint
            'created_at': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03'])
        })
        
        validator = SchemaValidator(customer_schema)
        
        # Act
        result = validator.validate_schema(invalid_data)
        
        # Assert
        assert not result['is_valid'], "Schema with constraint violations should fail"
        assert len(result['constraint_violations']) > 0, "Constraint violations should be detected"
```

**Regression Testing**:
```python
# Example: Regression testing for data pipelines
import pytest
import pandas as pd
import hashlib
from typing import Dict, Any

class RegressionTestFramework:
    """Framework for detecting regressions in data pipelines"""
    
    def __init__(self, baseline_path: str):
        self.baseline_path = baseline_path
    
    def capture_baseline(self, data: pd.DataFrame, test_name: str):
        """Capture baseline data for regression testing"""
        
        baseline_metrics = self._calculate_data_metrics(data)
        baseline_file = f"{self.baseline_path}/{test_name}_baseline.json"
        
        with open(baseline_file, 'w') as f:
            json.dump(baseline_metrics, f, indent=2, default=str)
    
    def check_regression(self, current_data: pd.DataFrame, test_name: str, 
                        tolerance: float = 0.05) -> Dict[str, Any]:
        """Check for regressions against baseline"""
        
        baseline_file = f"{self.baseline_path}/{test_name}_baseline.json"
        
        try:
            with open(baseline_file, 'r') as f:
                baseline_metrics = json.load(f)
        except FileNotFoundError:
            return {'status': 'no_baseline', 'message': 'No baseline found, capturing current as baseline'}
        
        current_metrics = self._calculate_data_metrics(current_data)
        
        regression_results = {
            'status': 'pass',
            'regressions': [],
            'improvements': [],
            'baseline_metrics': baseline_metrics,
            'current_metrics': current_metrics
        }
        
        # Compare metrics
        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                
                if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                    relative_change = abs(current_value - baseline_value) / abs(baseline_value) if baseline_value != 0 else 0
                    
                    if relative_change > tolerance:
                        change_type = 'regression' if current_value < baseline_value else 'improvement'
                        
                        regression_results[f'{change_type}s'].append({
                            'metric': metric_name,
                            'baseline': baseline_value,
                            'current': current_value,
                            'change_percent': relative_change * 100
                        })
        
        if regression_results['regressions']:
            regression_results['status'] = 'regression_detected'
        
        return regression_results
    
    def _calculate_data_metrics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data metrics for comparison"""
        
        metrics = {
            'row_count': len(data),
            'column_count': len(data.columns),
            'null_percentage': (data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100 if len(data) > 0 else 0,
            'data_hash': hashlib.md5(pd.util.hash_pandas_object(data, index=True).values).hexdigest()
        }
        
        # Numeric column metrics
        numeric_columns = data.select_dtypes(include=['number']).columns
        for col in numeric_columns:
            metrics[f'{col}_mean'] = data[col].mean()
            metrics[f'{col}_std'] = data[col].std()
            metrics[f'{col}_min'] = data[col].min()
            metrics[f'{col}_max'] = data[col].max()
        
        # Categorical column metrics
        categorical_columns = data.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            metrics[f'{col}_unique_count'] = data[col].nunique()
            metrics[f'{col}_most_common'] = data[col].mode().iloc[0] if len(data[col].mode()) > 0 else None
        
        return metrics

class TestRegressionDetection:
    """Test regression detection functionality"""
    
    @pytest.fixture
    def sample_baseline_data(self):
        """Sample baseline data for testing"""
        return pd.DataFrame({
            'customer_id': range(1, 101),
            'purchase_amount': [100 + i * 2.5 for i in range(100)],
            'category': ['electronics'] * 50 + ['clothing'] * 50
        })
    
    def test_no_regression_detected(self, sample_baseline_data, tmp_path):
        """Test that identical data shows no regression"""
        
        # Arrange
        framework = RegressionTestFramework(str(tmp_path))
        framework.capture_baseline(sample_baseline_data, 'test_no_regression')
        
        # Act
        result = framework.check_regression(sample_baseline_data, 'test_no_regression')
        
        # Assert
        assert result['status'] == 'pass', "Identical data should show no regression"
        assert len(result['regressions']) == 0, "No regressions should be detected"
    
    def test_regression_detected(self, sample_baseline_data, tmp_path):
        """Test that significant changes are detected as regressions"""
        
        # Arrange
        framework = RegressionTestFramework(str(tmp_path))
        framework.capture_baseline(sample_baseline_data, 'test_regression')
        
        # Create regressed data (50% fewer rows)
        regressed_data = sample_baseline_data.iloc[:50].copy()
        
        # Act
        result = framework.check_regression(regressed_data, 'test_regression')
        
        # Assert
        assert result['status'] == 'regression_detected', "Significant change should be detected"
        assert len(result['regressions']) > 0, "Regressions should be found"
        
        # Check specific regression
        row_count_regression = next((r for r in result['regressions'] if r['metric'] == 'row_count'), None)
        assert row_count_regression is not None, "Row count regression should be detected"
        assert row_count_regression['change_percent'] == 50.0, "Should detect 50% reduction"
```

#### 3. Performance Testing

**Load Testing**:
```python
# Example: Performance and load testing for data pipelines
import pytest
import pandas as pd
import time
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

class PerformanceTestFramework:
    """Framework for performance testing data pipelines"""
    
    def __init__(self):
        self.performance_metrics = {}
    
    def measure_performance(self, func, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance metrics for a function"""
        
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
            'result': result
        }
        
        return performance_metrics
    
    def load_test_pipeline(self, pipeline_func, concurrent_requests: int, 
                          test_data_generator, iterations: int = 10) -> Dict[str, Any]:
        """Perform load testing on data pipeline"""
        
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
                'avg_memory_usage': sum(memory_usage) / len(memory_usage),
                'max_memory_usage': max(memory_usage),
                'requests_per_second': len(results) / (end_time - start_time),
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

class TestPerformanceRequirements:
    """Test performance requirements for data pipelines"""
    
    @pytest.fixture
    def performance_framework(self):
        """Performance testing framework"""
        return PerformanceTestFramework()
    
    @pytest.fixture
    def sample_data_generator(self):
        """Generate sample data for performance testing"""
        def generator(size_multiplier=1):
            base_size = 1000
            return pd.DataFrame({
                'id': range(base_size * size_multiplier),
                'value': [i * 1.5 for i in range(base_size * size_multiplier)],
                'category': ['A', 'B', 'C'] * (base_size * size_multiplier // 3 + 1)
            })[:base_size * size_multiplier]
        
        return generator
    
    def test_single_request_performance(self, performance_framework, sample_data_generator):
        """Test single request performance meets requirements"""
        
        # Arrange
        test_data = sample_data_generator(10)  # 10k records
        
        def sample_pipeline(data):
            # Simulate data processing
            result = data.groupby('category')['value'].agg(['mean', 'sum', 'count'])
            return result
        
        # Act
        metrics = performance_framework.measure_performance(sample_pipeline, test_data)
        
        # Assert - Performance SLAs
        assert metrics['execution_time_seconds'] < 5.0, "Should complete within 5 seconds"
        assert metrics['memory_used_mb'] < 100, "Should use less than 100MB additional memory"
        assert metrics['result'] is not None, "Should produce valid result"
    
    @pytest.mark.slow
    def test_load_performance(self, performance_framework, sample_data_generator):
        """Test pipeline performance under load"""
        
        # Arrange
        def test_pipeline(data):
            # Simulate processing
            time.sleep(0.1)  # Simulate processing time
            return data.describe()
        
        # Act
        load_results = performance_framework.load_test_pipeline(
            test_pipeline,
            concurrent_requests=5,
            test_data_generator=lambda i: sample_data_generator(1),
            iterations=20
        )
        
        # Assert - Load testing SLAs
        assert load_results['success_rate'] >= 95, "Should maintain 95% success rate under load"
        assert load_results['p95_execution_time'] < 2.0, "95th percentile should be under 2 seconds"
        assert load_results['requests_per_second'] >= 2, "Should handle at least 2 requests per second"
    
    @pytest.mark.parametrize("data_size,max_time", [
        (1000, 1.0),    # 1k records in 1 second
        (10000, 5.0),   # 10k records in 5 seconds
        (100000, 30.0)  # 100k records in 30 seconds
    ])
    def test_scalability_requirements(self, performance_framework, sample_data_generator, 
                                    data_size, max_time):
        """Test pipeline scalability with different data sizes"""
        
        # Arrange
        test_data = sample_data_generator(data_size // 1000)
        
        def scalability_pipeline(data):
            # Simulate complex processing
            result = data.groupby('category').agg({
                'value': ['mean', 'std', 'min', 'max', 'count']
            })
            return result
        
        # Act
        metrics = performance_framework.measure_performance(scalability_pipeline, test_data)
        
        # Assert
        assert metrics['execution_time_seconds'] < max_time, f"Should process {data_size} records within {max_time} seconds"
        
        # Calculate throughput
        throughput = len(test_data) / metrics['execution_time_seconds']
        assert throughput >= 1000, "Should maintain at least 1000 records/second throughput"
```

#### 4. CI/CD Integration

**GitHub Actions Workflow**:
```yaml
# Example: .github/workflows/data-pipeline-ci.yml
name: Data Pipeline CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.9'
  
jobs:
  test:
    runs-on: ubuntu-latest
    
    services:
      postgres:
        image: postgres:13
        env:
          POSTGRES_PASSWORD: postgres
          POSTGRES_DB: test_db
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
        ports:
          - 5432:5432
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run linting
      run: |
        flake8 src/ tests/
        black --check src/ tests/
        isort --check-only src/ tests/
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=src --cov-report=xml
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        DATABASE_URL: postgresql://postgres:postgres@localhost:5432/test_db
    
    - name: Run data quality tests
      run: |
        pytest tests/data_quality/ -v
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
    
    - name: Run security scan
      run: |
        bandit -r src/
        safety check
  
  performance-test:
    runs-on: ubuntu-latest
    needs: test
    if: github.event_name == 'pull_request'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run performance tests
      run: |
        pytest tests/performance/ -v --benchmark-only
    
    - name: Performance regression check
      run: |
        python scripts/check_performance_regression.py
  
  e2e-test:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ env.PYTHON_VERSION }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Set up test environment
      run: |
        docker-compose -f docker-compose.test.yml up -d
        sleep 30  # Wait for services to be ready
    
    - name: Run end-to-end tests
      run: |
        pytest tests/e2e/ -v --maxfail=1
      env:
        TEST_ENVIRONMENT: staging
    
    - name: Cleanup test environment
      if: always()
      run: |
        docker-compose -f docker-compose.test.yml down
  
  deploy-staging:
    runs-on: ubuntu-latest
    needs: [test, e2e-test]
    if: github.ref == 'refs/heads/develop'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to staging
      run: |
        echo "Deploying to staging environment"
        # Add deployment commands here
    
    - name: Run staging validation tests
      run: |
        pytest tests/staging_validation/ -v
      env:
        ENVIRONMENT: staging
  
  deploy-production:
    runs-on: ubuntu-latest
    needs: [test, e2e-test]
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        echo "Deploying to production environment"
        # Add production deployment commands here
    
    - name: Run production smoke tests
      run: |
        pytest tests/smoke/ -v
      env:
        ENVIRONMENT: production
    
    - name: Monitor deployment
      run: |
        python scripts/monitor_deployment.py --timeout 300
```

---

## 💻 Hands-On Exercise (40 minutes)

Build a comprehensive testing framework for data pipelines with unit, integration, and end-to-end tests, plus CI/CD integration.

**Scenario**: You're the Data Engineering Lead at "TestDriven Analytics", a company that processes critical business data for decision-making. You need to implement a comprehensive testing strategy that ensures data quality, prevents regressions, and maintains high performance standards.

**Requirements**:
1. **Unit Testing Framework**: Implement comprehensive unit tests for data transformations
2. **Integration Testing**: Build tests for pipeline component interactions
3. **End-to-End Testing**: Create full workflow validation tests
4. **Performance Testing**: Implement load and scalability testing
5. **Regression Testing**: Build automated regression detection
6. **CI/CD Integration**: Create automated testing pipelines

**Data Sources**:
- Customer transaction data (high volume, critical accuracy)
- Product catalog (frequent updates, schema evolution)
- User behavior events (real-time, performance critical)
- Financial reporting data (regulatory compliance, zero tolerance for errors)

See `exercise.py` for starter code and detailed requirements.

---

## 📚 Resources

- **pytest**: [docs.pytest.org](https://docs.pytest.org/) - Python testing framework
- **Great Expectations**: [docs.greatexpectations.io](https://docs.greatexpectations.io/) - Data validation framework
- **dbt Testing**: [docs.getdbt.com/docs/build/tests](https://docs.getdbt.com/docs/build/tests) - dbt testing patterns
- **GitHub Actions**: [docs.github.com/actions](https://docs.github.com/actions) - CI/CD automation
- **Apache Airflow Testing**: [airflow.apache.org/docs/apache-airflow/stable/best-practices.html#testing](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html#testing)
- **Data Testing Patterns**: [martinfowler.com/articles/data-monolith-to-mesh.html](https://martinfowler.com/articles/data-monolith-to-mesh.html)

---

## 🎯 Key Takeaways

- **Testing pyramid approach** ensures comprehensive coverage with appropriate test distribution
- **Data-specific testing patterns** address unique challenges like schema evolution and data quality
- **Automated regression detection** prevents quality degradation over time
- **Performance testing** ensures systems meet SLA requirements under load
- **CI/CD integration** provides fast feedback and prevents production issues
- **Test-driven development** improves code quality and reduces debugging time
- **Comprehensive test coverage** builds confidence in data pipeline reliability
- **Production monitoring** complements testing with real-world validation

---

## 🚀 What's Next?

Tomorrow (Day 22), you'll learn **AWS Glue & Data Catalog** - serverless ETL and comprehensive data cataloging with AWS managed services.

**Preview**: You'll explore AWS Glue for serverless data processing, implement data catalogs for metadata management, and build scalable ETL workflows with AWS managed services!

---

## ✅ Before Moving On

- [ ] Understand the data testing pyramid and test distribution
- [ ] Can implement comprehensive unit tests for data transformations
- [ ] Know how to build integration and end-to-end tests
- [ ] Understand performance and regression testing strategies
- [ ] Can set up CI/CD pipelines with automated testing
- [ ] Complete the hands-on exercise
- [ ] Take the quiz

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐ (Advanced Testing Strategies)

Ready to ensure bulletproof data pipeline reliability! 🚀
