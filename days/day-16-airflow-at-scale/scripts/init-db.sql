-- Initialize Airflow database with optimizations

-- Create additional indexes for performance
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_dag_run_dag_id_execution_date 
ON dag_run (dag_id, execution_date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_instance_dag_task_execution_date 
ON task_instance (dag_id, task_id, execution_date);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_task_instance_state_dag_task 
ON task_instance (state, dag_id, task_id);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_job_state_heartbeat 
ON job (state, latest_heartbeat);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_log_dag_task_execution_date 
ON log (dag_id, task_id, execution_date);

-- Create replication user
CREATE USER replicator WITH REPLICATION ENCRYPTED PASSWORD 'replicator_password';

-- Grant necessary permissions
GRANT CONNECT ON DATABASE airflow TO replicator;
GRANT USAGE ON SCHEMA public TO replicator;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO replicator;

-- Set up monitoring user
CREATE USER airflow_monitor WITH ENCRYPTED PASSWORD 'monitor_password';
GRANT CONNECT ON DATABASE airflow TO airflow_monitor;
GRANT USAGE ON SCHEMA public TO airflow_monitor;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO airflow_monitor;

-- Performance tuning
ALTER SYSTEM SET shared_preload_libraries = 'pg_stat_statements';
ALTER SYSTEM SET track_activity_query_size = 2048;
ALTER SYSTEM SET track_counts = on;
ALTER SYSTEM SET track_functions = all;