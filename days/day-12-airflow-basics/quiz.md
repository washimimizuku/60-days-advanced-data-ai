# Day 12 Quiz: Apache Airflow Basics

## Questions

### 1. What is a DAG in Apache Airflow?
- a) A database query language for data analysis
- b) A directed acyclic graph representing a workflow
- c) A data aggregation gateway for APIs
- d) A deployment automation guide for containers

### 2. Which operator would you use to execute a Python function in Airflow?
- a) BashOperator
- b) PythonOperator
- c) FunctionOperator
- d) ExecuteOperator

### 3. What is the primary purpose of XCom in Airflow?
- a) To execute shell commands across tasks
- b) To pass small amounts of data between tasks
- c) To compress large datasets for storage
- d) To communicate with external API systems

### 4. What does setting `catchup=False` do in a DAG configuration?
- a) Prevents the DAG from catching exceptions
- b) Disables backfilling of past scheduled runs
- c) Stops the DAG from running automatically
- d) Forces the DAG to catch up with missed schedules

### 5. What makes a task "idempotent" in Airflow?
- a) A task that runs only once per DAG execution
- b) A task that produces the same result when run multiple times
- c) A task that never fails under any circumstances
- d) A task that runs in parallel with other tasks

### 6. Which executor is recommended for production Airflow deployments at scale?
- a) SequentialExecutor (single-threaded)
- b) LocalExecutor (multi-threaded, single machine)
- c) CeleryExecutor or KubernetesExecutor (distributed)
- d) DebugExecutor (development only)

### 7. What is the recommended approach for passing large datasets between Airflow tasks?
- a) Use XCom to pass the data directly
- b) Store data in global Python variables
- c) Use external storage (S3, database) and pass references
- d) Pass data through task parameters

### 8. What does the `>>` operator do in Airflow DAG definitions?
- a) Compares the outputs of two tasks
- b) Defines task dependencies and execution order
- c) Shifts data from one task to another
- d) Increases the priority of a task

### 9. What is a Sensor in Apache Airflow?
- a) A monitoring tool for system performance
- b) An operator that waits for a specific condition to be met
- c) A data validation tool for quality checks
- d) A security feature for access control

### 10. Which scheduling expression would run a DAG every weekday at 9:30 AM?
- a) `'0 9 * * 1-5'`
- b) `'30 9 * * 1-5'`
- c) `'@daily'`
- d) `'30 9 * * *'`

---

## Answers

### 1. What is a DAG in Apache Airflow?
**Answer: b) A directed acyclic graph representing a workflow**

**Explanation:** A DAG (Directed Acyclic Graph) is the core concept in Airflow representing a workflow. It's a collection of tasks with defined dependencies that form a graph structure. "Directed" means the edges have direction (task A leads to task B), and "acyclic" means there are no circular dependencies. Each DAG represents a complete workflow that can be scheduled and executed.

---

### 2. Which operator would you use to execute a Python function in Airflow?
**Answer: b) PythonOperator**

**Explanation:** The PythonOperator is specifically designed to execute Python functions within Airflow tasks. You provide a `python_callable` parameter that points to your function, and Airflow will execute it when the task runs. BashOperator executes shell commands, while FunctionOperator and ExecuteOperator don't exist in Airflow's standard operator set.

---

### 3. What is the primary purpose of XCom in Airflow?
**Answer: b) To pass small amounts of data between tasks**

**Explanation:** XCom (Cross-Communication) is Airflow's mechanism for sharing small pieces of data between tasks. Tasks can "push" data to XCom and other tasks can "pull" that data. It's designed for metadata, IDs, status information, and small datasets. For large datasets, you should use external storage (S3, databases) and pass references through XCom instead.

---

### 4. What does setting `catchup=False` do in a DAG configuration?
**Answer: b) Disables backfilling of past scheduled runs**

**Explanation:** When `catchup=False`, Airflow will not automatically create DAG runs for past scheduled intervals that were missed. For example, if you deploy a daily DAG with a start_date of January 1st but deploy it on January 10th, Airflow won't create runs for January 1-9. This is often desired for ETL pipelines where you only want to process current data, not historical data.

---

### 5. What makes a task "idempotent" in Airflow?
**Answer: b) A task that produces the same result when run multiple times**

**Explanation:** Idempotency is a crucial concept in data pipelines. An idempotent task produces the same result whether it's run once or multiple times with the same inputs. This is important for retry scenarios and reprocessing. For example, using `INSERT ... ON DUPLICATE KEY UPDATE` or `MERGE` statements instead of simple `INSERT` statements makes database operations idempotent.

---

### 6. Which executor is recommended for production Airflow deployments at scale?
**Answer: c) CeleryExecutor or KubernetesExecutor (distributed)**

**Explanation:** For production at scale, you need distributed executors that can run tasks across multiple workers. CeleryExecutor uses Celery for distributed task execution across multiple machines, while KubernetesExecutor creates Kubernetes pods for each task. SequentialExecutor runs tasks one at a time (development only), and LocalExecutor runs tasks on a single machine with multiple processes.

---

### 7. What is the recommended approach for passing large datasets between Airflow tasks?
**Answer: c) Use external storage (S3, database) and pass references**

**Explanation:** XCom is designed for small metadata, not large datasets. The recommended pattern is to store large datasets in external systems (S3, databases, data lakes) and pass references (file paths, table names, URLs) through XCom. This approach is more efficient, doesn't overload Airflow's metadata database, and allows for better error handling and data persistence.

---

### 8. What does the `>>` operator do in Airflow DAG definitions?
**Answer: b) Defines task dependencies and execution order**

**Explanation:** The `>>` operator (bitshift operator) is used in Airflow to define task dependencies. `task_a >> task_b` means task_b will run after task_a completes successfully. You can also use `<<` for the reverse direction, or chain multiple tasks: `task_a >> task_b >> task_c`. This creates the directed edges in your DAG.

---

### 9. What is a Sensor in Apache Airflow?
**Answer: b) An operator that waits for a specific condition to be met**

**Explanation:** Sensors are special operators that wait for external conditions to be met before proceeding. Examples include FileSensor (waits for a file to exist), S3KeySensor (waits for an S3 object), HttpSensor (waits for an HTTP endpoint to return success), or TimeSensor (waits for a specific time). Sensors poll the condition at regular intervals until it's satisfied or a timeout is reached.

---

### 10. Which scheduling expression would run a DAG every weekday at 9:30 AM?
**Answer: b) `'30 9 * * 1-5'`**

**Explanation:** This is a cron expression where the format is `minute hour day_of_month month day_of_week`. `'30 9 * * 1-5'` means: minute 30, hour 9 (9:30 AM), any day of month (*), any month (*), Monday through Friday (1-5). Option a) would run at 9:00 AM, option c) runs daily including weekends, and option d) runs every day at 9:30 AM including weekends.

---

## Score Interpretation

- **9-10 correct**: Airflow Expert! You understand core concepts and are ready for production deployments
- **7-8 correct**: Strong foundation! Review the areas you missed and practice with real DAGs
- **5-6 correct**: Good start! Focus on understanding DAG structure, operators, and scheduling
- **Below 5**: Review the theory section and work through the hands-on exercises

---

## Key Concepts to Remember

1. **DAGs are directed acyclic graphs** representing workflows with task dependencies
2. **PythonOperator executes Python functions** as Airflow tasks
3. **XCom passes small data between tasks** - use external storage for large datasets
4. **catchup=False prevents backfilling** of missed historical runs
5. **Idempotent tasks produce consistent results** when run multiple times
6. **Distributed executors (Celery/Kubernetes)** are needed for production scale
7. **External storage is recommended** for large dataset transfers between tasks
8. **The >> operator defines task dependencies** and execution order
9. **Sensors wait for external conditions** before proceeding with workflow
10. **Cron expressions control scheduling** with minute-hour-day-month-weekday format

---

## Airflow Best Practices

### DAG Design
- **Keep DAGs simple and focused** on a single business process
- **Use meaningful task and DAG IDs** that describe their purpose
- **Set appropriate timeouts** to prevent hanging tasks
- **Implement proper error handling** with retries and callbacks

### Task Implementation
- **Make tasks idempotent** to handle retries safely
- **Use appropriate operators** for different types of work
- **Keep tasks atomic** - each task should do one thing well
- **Add comprehensive logging** for debugging and monitoring

### Data Handling
- **Use XCom for metadata only** - not for large datasets
- **Store large data externally** (S3, databases, data lakes)
- **Pass references through XCom** (file paths, table names, URLs)
- **Implement data validation** at task boundaries

### Production Deployment
- **Use distributed executors** (CeleryExecutor, KubernetesExecutor)
- **Set up monitoring and alerting** for DAG failures
- **Implement proper resource limits** to prevent resource exhaustion
- **Use version control** for DAG code and configuration

### Scheduling and Dependencies
- **Set realistic SLAs** based on business requirements
- **Use sensors for external dependencies** rather than polling in tasks
- **Implement proper backfill strategies** with catchup settings
- **Consider timezone implications** for scheduling

### Common Pitfalls to Avoid
- **Don't use XCom for large datasets** - it will overload the metadata database
- **Don't create dynamic DAGs** without careful consideration of performance
- **Don't ignore task timeouts** - they prevent hanging processes
- **Don't skip testing** - validate DAGs before production deployment
- **Don't forget about resource cleanup** - clean up temporary files and connections

Ready to move on to Day 13! 🚀
