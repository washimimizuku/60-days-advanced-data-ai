# Day 12: Apache Airflow Basics - DAGs, Operators, Scheduling

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- Understand Apache Airflow architecture and core concepts
- Create and configure DAGs (Directed Acyclic Graphs)
- Use different operators (Python, Bash, Email)
- Implement task dependencies and scheduling
- Apply error handling and retry strategies

---

## Theory

### What is Apache Airflow?

Apache Airflow is an open-source platform to programmatically author, schedule, and monitor workflows. It's the industry standard for data pipeline orchestration, used by companies like Airbnb, Netflix, Adobe, and Spotify.

**Why Airflow?**
- **Code-based**: Define workflows as Python code
- **Scalable**: Handle thousands of tasks
- **Extensible**: Custom operators and plugins
- **Monitoring**: Built-in UI and alerting
- **Community**: Large ecosystem and support

### Core Concepts

#### 1. DAG (Directed Acyclic Graph)

A DAG is a collection of tasks with dependencies, representing your workflow.

```python
from airflow import DAG
from datetime import datetime, timedelta

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email': ['alerts@company.com'],
    'email_on_failure': True,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'my_data_pipeline',
    default_args=default_args,
    description='A simple data pipeline',
    schedule_interval='@daily',  # Run once per day
    catchup=False,  # Don't backfill
)
```

**Key Properties**:
- `dag_id`: Unique identifier
- `schedule_interval`: When to run (cron or preset)
- `start_date`: When DAG becomes active
- `catchup`: Whether to backfill missed runs
- `default_args`: Default settings for all tasks

#### 2. Operators

Operators define what work gets done. Each operator is a task in your DAG.

**PythonOperator** - Execute Python functions:
```python
from airflow.operators.python import PythonOperator

def extract_data(**context):
    """Extract data from source"""
    data = {'users': 100, 'orders': 500}
    # Push data to XCom for next task
    context['task_instance'].xcom_push(key='metrics', value=data)
    return data

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)
```

**BashOperator** - Run shell commands:
```python
from airflow.operators.bash import BashOperator

cleanup_task = BashOperator(
    task_id='cleanup_temp_files',
    bash_command='rm -rf /tmp/data_*.csv',
    dag=dag,
)
```

**EmailOperator** - Send notifications:
```python
from airflow.operators.email import EmailOperator

notify_task = EmailOperator(
    task_id='send_notification',
    to='team@company.com',
    subject='Pipeline Complete',
    html_content='<p>Data pipeline finished successfully!</p>',
    dag=dag,
)
```

#### 3. Task Dependencies

Define the order tasks run in:

```python
# Method 1: Bitshift operators
extract_task >> transform_task >> load_task

# Method 2: set_downstream/set_upstream
extract_task.set_downstream(transform_task)
transform_task.set_downstream(load_task)

# Method 3: Multiple dependencies
[extract_users, extract_orders] >> transform_task >> load_task
```

**Dependency Patterns**:
- **Linear**: A >> B >> C
- **Fan-out**: A >> [B, C, D]
- **Fan-in**: [A, B, C] >> D
- **Complex**: Mix of above

#### 4. Scheduling

**Cron Expressions**:
```python
'@once'      # Run once
'@hourly'    # Every hour
'@daily'     # Every day at midnight
'@weekly'    # Every Sunday at midnight
'@monthly'   # First day of month
'0 */2 * * *'  # Every 2 hours
'30 9 * * 1-5' # 9:30 AM weekdays
```

**Execution Date**:
- Airflow uses "execution_date" (logical date)
- Represents the start of the data interval
- Task runs AFTER the interval completes

```python
# For daily DAG with start_date 2024-01-01:
# First run: 2024-01-02 00:00 (processes 2024-01-01 data)
# Second run: 2024-01-03 00:00 (processes 2024-01-02 data)
```

#### 5. XCom (Cross-Communication)

Share data between tasks:

```python
def task_a(**context):
    # Push data
    context['task_instance'].xcom_push(key='my_data', value={'count': 100})

def task_b(**context):
    # Pull data
    ti = context['task_instance']
    data = ti.xcom_pull(key='my_data', task_ids='task_a')
    print(f"Received: {data}")
```

**XCom Best Practices**:
- ✅ Use for small metadata (IDs, counts, status)
- ❌ Don't use for large datasets
- ✅ Store large data in S3/database
- ✅ Pass references (paths, URLs) via XCom

### Error Handling and Retries

```python
task = PythonOperator(
    task_id='my_task',
    python_callable=my_function,
    retries=3,                    # Retry 3 times
    retry_delay=timedelta(minutes=5),  # Wait 5 min between retries
    retry_exponential_backoff=True,    # Increase delay each retry
    max_retry_delay=timedelta(hours=1), # Max delay
    execution_timeout=timedelta(hours=2), # Task timeout
    on_failure_callback=alert_on_failure, # Custom callback
    dag=dag,
)
```

### Production Best Practices

1. **Idempotency**: Tasks should produce same result when run multiple times
   ```python
   # Good: Overwrites existing data
   df.to_sql('table', engine, if_exists='replace')
   
   # Bad: Appends every time
   df.to_sql('table', engine, if_exists='append')
   ```

2. **Atomic Operations**: All or nothing
   ```python
   # Use transactions
   with engine.begin() as conn:
       conn.execute("DELETE FROM table WHERE date = %s", date)
       conn.execute("INSERT INTO table VALUES ...")
   ```

3. **Proper Logging**:
   ```python
   import logging
   
   def my_task():
       logging.info("Starting data extraction")
       logging.warning("Missing some records")
       logging.error("Failed to connect")
   ```

4. **Testing**:
   ```python
   # Test DAG validity
   from airflow.models import DagBag
   
   def test_dag_loaded():
       dagbag = DagBag()
       assert 'my_dag' in dagbag.dags
       assert len(dagbag.import_errors) == 0
   ```

### Common Patterns

**ETL Pattern**:
```python
extract >> transform >> load >> validate
```

**Parallel Processing**:
```python
start >> [process_a, process_b, process_c] >> combine >> end
```

**Conditional Execution**:
```python
from airflow.operators.python import BranchPythonOperator

def choose_branch(**context):
    if condition:
        return 'task_a'
    return 'task_b'

branch = BranchPythonOperator(
    task_id='branch',
    python_callable=choose_branch,
)

branch >> [task_a, task_b]
```

---

## 💻 Hands-On Exercise (40 minutes)

Build a complete ETL pipeline with Airflow.

**Scenario**: You need to build a daily pipeline that:
1. Extracts user activity data from an API
2. Transforms and cleans the data
3. Loads it into a database
4. Sends a success notification

**Requirements**:
- Use PythonOperator for ETL tasks
- Use BashOperator for cleanup
- Use EmailOperator for notifications
- Implement proper error handling
- Add logging
- Set up daily scheduling

See `exercise.py` for starter code.

---

## 📚 Resources

- **Official Docs**: [airflow.apache.org/docs](https://airflow.apache.org/docs/)
- **Best Practices**: [Airflow Best Practices Guide](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- **Tutorials**: [Airflow Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/tutorial.html)
- **Operators**: [Operator Guide](https://airflow.apache.org/docs/apache-airflow/stable/howto/operator/index.html)
- **Community**: [Airflow Slack](https://apache-airflow.slack.com/)

---

## 🎯 Key Takeaways

- **Airflow orchestrates workflows** using DAGs (Directed Acyclic Graphs)
- **Operators define tasks** - PythonOperator, BashOperator, etc.
- **Dependencies control execution order** using `>>` operator
- **Scheduling uses cron expressions** - @daily, @hourly, custom
- **XCom shares small data** between tasks
- **Error handling is critical** - retries, timeouts, callbacks
- **Idempotency ensures reliability** - same result on re-runs
- **Start simple, add complexity** as needed

---

## 🚀 What's Next?

Tomorrow (Day 13), you'll learn **dbt basics** - the transformation layer that works perfectly with Airflow for building data pipelines.

**Preview**: dbt (data build tool) transforms data in your warehouse using SQL, with built-in testing, documentation, and version control. Combined with Airflow, you'll have a complete orchestration + transformation stack!

---

## ✅ Before Moving On

- [ ] Understand what DAGs are and how they work
- [ ] Can create tasks using different operators
- [ ] Know how to set up task dependencies
- [ ] Understand scheduling with cron expressions
- [ ] Can implement error handling and retries
- [ ] Complete the exercise
- [ ] Take the quiz

**Time spent**: ~1 hour
**Difficulty**: ⭐⭐⭐ (Intermediate)

Ready to build production data pipelines! 🚀
