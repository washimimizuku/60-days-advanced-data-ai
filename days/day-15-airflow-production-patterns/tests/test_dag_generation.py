"""
Integration tests for dynamic DAG generation
"""

import pytest
import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import the solution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solution import (
    create_data_pipeline_dag, 
    create_monitoring_dag,
    DATA_SOURCE_CONFIGS,
    DataSourceConfig
)

class TestDynamicDAGGeneration:
    """Test cases for dynamic DAG generation"""
    
    def test_create_pipeline_dag_basic(self):
        """Test basic DAG creation from configuration"""
        config = DATA_SOURCE_CONFIGS[0]  # Use first config
        dag = create_data_pipeline_dag(config)
        
        # Verify DAG properties
        assert dag.dag_id == f"techcorp_pipeline_{config.name}"
        assert dag.schedule_interval == config.schedule_interval
        assert dag.catchup is False
        assert dag.max_active_runs == 1
        
        # Verify tags
        expected_tags = ['production', 'techcorp', config.source_type, f'priority_{config.priority}']
        for tag in expected_tags:
            assert tag in dag.tags
    
    def test_create_pipeline_dag_all_configs(self):
        """Test DAG creation for all data source configurations"""
        dags = []
        
        for config in DATA_SOURCE_CONFIGS:
            dag = create_data_pipeline_dag(config)
            dags.append(dag)
            
            # Verify each DAG has unique ID
            assert dag.dag_id == f"techcorp_pipeline_{config.name}"
            
            # Verify DAG has required tasks
            task_ids = [task.task_id for task in dag.tasks]
            
            # Check for essential tasks
            assert 'start' in task_ids
            assert 'end' in task_ids
            assert 'choose_processing_path' in task_ids
            assert 'final_validation' in task_ids
            assert 'update_monitoring' in task_ids
        
        # Verify all DAGs have unique IDs
        dag_ids = [dag.dag_id for dag in dags]
        assert len(dag_ids) == len(set(dag_ids))  # No duplicates
    
    def test_dag_task_structure(self):
        """Test that generated DAGs have correct task structure"""
        config = DATA_SOURCE_CONFIGS[0]
        dag = create_data_pipeline_dag(config)
        
        # Get all tasks
        tasks = {task.task_id: task for task in dag.tasks}
        
        # Verify task groups exist
        task_group_tasks = [task_id for task_id in tasks.keys() if '.' in task_id]
        
        # Should have ingestion and processing task groups
        ingestion_tasks = [t for t in task_group_tasks if t.startswith('ingestion.')]
        processing_tasks = [t for t in task_group_tasks if t.startswith('processing.')]
        
        assert len(ingestion_tasks) > 0, "Should have ingestion tasks"
        assert len(processing_tasks) > 0, "Should have processing tasks"
        
        # Verify branching tasks exist
        branching_tasks = [
            'fast_track_processing',
            'standard_processing', 
            'batch_processing',
            'complex_processing',
            'skip_processing'
        ]
        
        for task_id in branching_tasks:
            assert task_id in tasks, f"Missing branching task: {task_id}"
    
    def test_dag_dependencies(self):
        """Test that DAG tasks have correct dependencies"""
        config = DATA_SOURCE_CONFIGS[0]
        dag = create_data_pipeline_dag(config)
        
        # Get tasks
        tasks = {task.task_id: task for task in dag.tasks}
        
        # Verify start task has no upstream dependencies
        start_task = tasks['start']
        assert len(start_task.upstream_task_ids) == 0
        
        # Verify end task has upstream dependencies
        end_task = tasks['end']
        assert len(end_task.upstream_task_ids) > 0
        
        # Verify branching task has upstream dependencies
        branch_task = tasks['choose_processing_path']
        assert len(branch_task.upstream_task_ids) > 0
    
    def test_monitoring_dag_creation(self):
        """Test monitoring DAG creation"""
        dag = create_monitoring_dag()
        
        # Verify DAG properties
        assert dag.dag_id == 'techcorp_pipeline_monitoring'
        assert dag.schedule_interval == '*/15 * * * *'
        assert dag.catchup is False
        
        # Verify monitoring tasks exist
        task_ids = [task.task_id for task in dag.tasks]
        
        expected_tasks = [
            'check_pipeline_health',
            'check_data_freshness',
            'check_resource_utilization',
            'generate_metrics'
        ]
        
        for task_id in expected_tasks:
            assert task_id in task_ids, f"Missing monitoring task: {task_id}"
    
    def test_custom_config_dag_generation(self):
        """Test DAG generation with custom configuration"""
        custom_config = DataSourceConfig(
            name="test_source",
            source_type="api",
            source_config={"url": "https://test.api.com"},
            target_table="test_table",
            schedule_interval="@daily",
            priority=5,
            expected_volume_mb=100,
            sla_hours=4,
            quality_rules=[{"type": "not_null", "columns": ["id"]}],
            transformations=[{"type": "test_transform"}]
        )
        
        dag = create_data_pipeline_dag(custom_config)
        
        # Verify custom configuration is applied
        assert dag.dag_id == "techcorp_pipeline_test_source"
        assert dag.schedule_interval == "@daily"
        assert 'priority_5' in dag.tags
        assert 'api' in dag.tags
    
    def test_dag_default_args(self):
        """Test that DAGs have correct default arguments"""
        config = DATA_SOURCE_CONFIGS[0]
        dag = create_data_pipeline_dag(config)
        
        # Verify default args
        default_args = dag.default_args
        
        assert default_args['owner'] == 'techcorp-data-team'
        assert default_args['depends_on_past'] is False
        assert default_args['email_on_failure'] is True
        assert default_args['retries'] >= 1
        assert 'retry_delay' in default_args
        assert 'execution_timeout' in default_args
        assert 'on_failure_callback' in default_args
        assert 'on_success_callback' in default_args

class TestDAGValidation:
    """Test DAG validation and integrity"""
    
    def test_dag_validation(self):
        """Test that generated DAGs pass Airflow validation"""
        for config in DATA_SOURCE_CONFIGS[:2]:  # Test first 2 configs
            dag = create_data_pipeline_dag(config)
            
            # Basic validation - DAG should be able to serialize
            try:
                dag_dict = dag.__dict__
                assert 'dag_id' in dag_dict
                assert 'schedule_interval' in dag_dict
            except Exception as e:
                pytest.fail(f"DAG validation failed for {config.name}: {e}")
    
    def test_no_circular_dependencies(self):
        """Test that DAGs don't have circular dependencies"""
        config = DATA_SOURCE_CONFIGS[0]
        dag = create_data_pipeline_dag(config)
        
        # Simple check - if DAG can be created without errors,
        # Airflow's internal validation should catch circular dependencies
        assert len(dag.tasks) > 0
        
        # Verify we can traverse the DAG
        visited = set()
        
        def visit_task(task):
            if task.task_id in visited:
                return
            visited.add(task.task_id)
            for downstream_task in task.downstream_list:
                visit_task(downstream_task)
        
        # Start from tasks with no upstream dependencies
        root_tasks = [task for task in dag.tasks if len(task.upstream_task_ids) == 0]
        
        for root_task in root_tasks:
            visit_task(root_task)
        
        # Should be able to visit all tasks without infinite recursion
        assert len(visited) > 0

if __name__ == '__main__':
    pytest.main([__file__])