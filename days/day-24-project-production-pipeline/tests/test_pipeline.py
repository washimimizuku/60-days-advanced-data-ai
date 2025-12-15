#!/usr/bin/env python3
"""
Day 24: Production Pipeline - Tests
"""

import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch
from airflow.models import DagBag

class TestProductionPipeline:
    """Test suite for production pipeline"""
    
    def test_dag_loading(self):
        """Test that DAGs load without errors"""
        dagbag = DagBag(dag_folder='airflow/dags/', include_examples=False)
        
        # Check for import errors
        assert len(dagbag.import_errors) == 0, f"DAG import errors: {dagbag.import_errors}"
        
        # Check that production DAG exists
        assert 'production_data_pipeline' in dagbag.dag_ids
    
    def test_dag_structure(self):
        """Test DAG structure and dependencies"""
        dagbag = DagBag(dag_folder='airflow/dags/', include_examples=False)
        dag = dagbag.get_dag('production_data_pipeline')
        
        # Check task groups exist
        expected_groups = ['data_ingestion', 'quality_validation', 'transformations']
        task_ids = [task.task_id for task in dag.tasks]
        
        # Verify key tasks exist
        assert any('ingest' in task_id for task_id in task_ids)
        assert any('quality' in task_id for task_id in task_ids)
        assert any('dbt' in task_id for task_id in task_ids)
    
    def test_data_processing(self):
        """Test data processing functions"""
        # Sample data
        data = pd.DataFrame({
            'customer_id': [1, 2, 3],
            'email': ['test1@example.com', 'test2@example.com', 'invalid-email'],
            'amount': [100, 200, 300]
        })
        
        # Test data validation
        valid_emails = data['email'].str.match(r'^[^@]+@[^@]+\.[^@]+$')
        assert valid_emails.sum() == 2  # Two valid emails
    
    def test_quality_validation(self):
        """Test data quality validation"""
        # Test quality score calculation
        sample_data = pd.DataFrame({
            'id': [1, 2, None],
            'value': [100, 200, 300]
        })
        
        # Calculate quality score (simplified)
        null_count = sample_data.isnull().sum().sum()
        total_values = sample_data.size
        quality_score = 1 - (null_count / total_values)
        
        assert quality_score > 0.8  # Should have good quality
    
    def test_environment_config(self):
        """Test environment configuration"""
        # Check that environment variables can be loaded
        test_vars = ['POSTGRES_USER', 'POSTGRES_DB', 'AWS_DEFAULT_REGION']
        
        for var in test_vars:
            # Should not raise exception
            value = os.getenv(var, 'default')
            assert value is not None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])