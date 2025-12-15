# Day 13 Quiz: dbt Basics

## Questions

### 1. What does dbt stand for and what is its primary purpose?
- a) Database build tool - for creating database schemas
- b) Data build tool - for transforming data in warehouses using SQL
- c) Dynamic batch transformer - for real-time data processing
- d) Data backup tool - for creating data backups

### 2. What is a dbt model?
- a) A Python class that defines data transformations
- b) A configuration file that sets up database connections
- c) A SQL SELECT statement that creates tables or views
- d) A JSON schema that validates data structure

### 3. Which materialization strategy should you use for large fact tables that receive new data regularly?
- a) view - for fast queries without storage overhead
- b) table - for complete refresh on every run
- c) incremental - to process only new or changed data
- d) ephemeral - to avoid creating physical objects

### 4. What is the purpose of the staging layer in dbt's layered architecture?
- a) To store final analytics tables for business users
- b) To clean and standardize raw data with 1:1 source mapping
- c) To apply complex business logic and calculations
- d) To create aggregated summary tables

### 5. How do you reference another dbt model in your SQL?
- a) Using direct table names like `SELECT * FROM customers`
- b) Using the source() function like `{{ source('raw', 'customers') }}`
- c) Using the ref() function like `{{ ref('stg_customers') }}`
- d) Using JOIN statements with full table paths

### 6. What type of test would you use to ensure a column contains only specific values?
- a) unique - to check for duplicate values
- b) not_null - to check for missing values
- c) accepted_values - to validate against a list of allowed values
- d) relationships - to check foreign key constraints

### 7. What is the difference between schema tests and data tests in dbt?
- a) Schema tests are written in YAML, data tests are written in SQL
- b) Schema tests validate structure, data tests validate content
- c) Schema tests run faster, data tests are more comprehensive
- d) Schema tests are built-in, data tests are custom SQL queries

### 8. Which command generates and serves dbt documentation locally?
- a) `dbt docs build` followed by `dbt docs open`
- b) `dbt generate docs` followed by `dbt serve docs`
- c) `dbt docs generate` followed by `dbt docs serve`
- d) `dbt create docs` followed by `dbt view docs`

### 9. What is the recommended approach for handling large datasets in dbt?
- a) Use view materialization to avoid storage costs
- b) Use incremental materialization with proper unique_key
- c) Use ephemeral materialization for better performance
- d) Use table materialization with full refresh every time

### 10. In dbt's layered architecture, what should the intermediate layer contain?
- a) Raw data cleaning and standardization
- b) Final analytics tables for end users
- c) Business logic, joins, and complex transformations
- d) Data quality tests and validation rules

---

## Answers

### 1. What does dbt stand for and what is its primary purpose?
**Answer: b) Data build tool - for transforming data in warehouses using SQL**

**Explanation:** dbt (data build tool) is specifically designed for transforming data in data warehouses using SQL. It brings software engineering practices like version control, testing, and documentation to data transformation workflows. Unlike ETL tools that extract and load data, dbt focuses purely on the transformation layer, assuming data is already in your warehouse.

---

### 2. What is a dbt model?
**Answer: c) A SQL SELECT statement that creates tables or views**

**Explanation:** A dbt model is fundamentally a SQL SELECT statement saved in a .sql file. When dbt runs, it executes these SELECT statements and materializes them as tables, views, or other objects in your data warehouse. Models can reference other models using the ref() function, creating a dependency graph that dbt uses to determine execution order.

---

### 3. Which materialization strategy should you use for large fact tables that receive new data regularly?
**Answer: c) incremental - to process only new or changed data**

**Explanation:** Incremental materialization is ideal for large tables that grow over time. Instead of rebuilding the entire table on each run (like table materialization), incremental models only process new or changed records, making them much more efficient for large datasets. You define a unique_key and dbt handles the merge logic to update existing records and insert new ones.

---

### 4. What is the purpose of the staging layer in dbt's layered architecture?
**Answer: b) To clean and standardize raw data with 1:1 source mapping**

**Explanation:** The staging layer is the first transformation layer in dbt's architecture. It maintains a 1:1 relationship with source tables, focusing on basic data cleaning, standardization, and renaming. This includes tasks like lowercasing emails, trimming whitespace, standardizing date formats, and renaming columns to follow consistent naming conventions. Complex business logic belongs in the intermediate layer.

---

### 5. How do you reference another dbt model in your SQL?
**Answer: c) Using the ref() function like `{{ ref('stg_customers') }}`**

**Explanation:** The ref() function is how you reference other dbt models in your SQL. This creates dependencies in dbt's DAG and ensures models run in the correct order. dbt compiles `{{ ref('stg_customers') }}` to the actual table/view name in your warehouse. The source() function is used for raw data tables, not dbt models.

---

### 6. What type of test would you use to ensure a column contains only specific values?
**Answer: c) accepted_values - to validate against a list of allowed values**

**Explanation:** The accepted_values test is a schema test that validates a column contains only values from a specified list. For example, you might test that a status column only contains 'active', 'inactive', or 'pending'. This is different from unique (no duplicates), not_null (no missing values), or relationships (foreign key validation).

---

### 7. What is the difference between schema tests and data tests in dbt?
**Answer: a) Schema tests are written in YAML, data tests are written in SQL**

**Explanation:** Schema tests are the built-in tests (unique, not_null, accepted_values, relationships) that are configured in YAML files. Data tests (also called singular tests) are custom SQL queries that you write to test specific business rules. Both validate data quality, but schema tests are pre-built and configured declaratively, while data tests give you full flexibility to write custom validation logic.

---

### 8. Which command generates and serves dbt documentation locally?
**Answer: c) `dbt docs generate` followed by `dbt docs serve`**

**Explanation:** dbt documentation is a two-step process: `dbt docs generate` creates the documentation files by parsing your project and extracting metadata, then `dbt docs serve` starts a local web server to view the documentation in your browser. The documentation includes model descriptions, column details, data lineage graphs, and compiled SQL.

---

### 9. What is the recommended approach for handling large datasets in dbt?
**Answer: b) Use incremental materialization with proper unique_key**

**Explanation:** For large datasets, incremental materialization is the most efficient approach. It only processes new or changed data rather than rebuilding the entire table. You must specify a unique_key that dbt uses to identify which records to update or insert. This dramatically reduces processing time and warehouse costs for large, growing datasets.

---

### 10. In dbt's layered architecture, what should the intermediate layer contain?
**Answer: c) Business logic, joins, and complex transformations**

**Explanation:** The intermediate layer sits between staging and marts, containing business logic, complex joins, and calculations. This is where you combine multiple staging models, apply business rules, calculate metrics, and prepare data for the final mart layer. Intermediate models are typically materialized as ephemeral (CTEs) since they're stepping stones to final tables.

---

## Score Interpretation

- **9-10 correct**: dbt Expert! You understand core concepts and are ready for production analytics engineering
- **7-8 correct**: Strong foundation! Review the areas you missed and practice building dbt projects
- **5-6 correct**: Good start! Focus on understanding layered architecture, materializations, and testing
- **Below 5**: Review the theory section and work through the hands-on exercises

---

## Key Concepts to Remember

### Core dbt Concepts
1. **dbt transforms data in warehouses** using SQL with software engineering practices
2. **Models are SELECT statements** that create tables, views, or incremental objects
3. **Layered architecture** (staging → intermediate → marts) promotes maintainability
4. **ref() function creates dependencies** between models in the DAG
5. **Materializations control** how models are built (view, table, incremental, ephemeral)

### Testing and Quality
6. **Schema tests are configured in YAML** (unique, not_null, accepted_values, relationships)
7. **Data tests are custom SQL queries** for business rule validation
8. **Sources define raw data** with freshness and quality monitoring
9. **Comprehensive testing** ensures data quality throughout the pipeline
10. **Documentation is auto-generated** from code and stays current

### Production Best Practices
- **Use staging for cleaning** - 1:1 with sources, basic standardization
- **Use intermediate for logic** - business rules, joins, calculations
- **Use marts for final tables** - analytics-ready dimensional models
- **Incremental for large tables** - process only new/changed data
- **Test everything** - schema tests for basics, data tests for business rules
- **Document as you build** - descriptions, business context, assumptions

### Common Patterns
- **Staging models**: `stg_` prefix, view materialization, basic cleaning
- **Intermediate models**: `int_` prefix, ephemeral materialization, business logic
- **Dimension tables**: `dim_` prefix, table materialization, slowly changing
- **Fact tables**: `fct_` prefix, incremental materialization, additive measures
- **Aggregate tables**: summary metrics, table materialization, pre-computed

### dbt Commands
- **Development**: `dbt compile`, `dbt run`, `dbt test`, `dbt build`
- **Selective execution**: `--models`, `--select`, `+model_name`, `model_name+`
- **Documentation**: `dbt docs generate`, `dbt docs serve`
- **Production**: `--target prod`, `--full-refresh`, `--store-failures`

---

## dbt Best Practices

### Model Organization
- **Follow naming conventions** - clear prefixes and descriptive names
- **One model per file** - keeps code organized and maintainable
- **Use proper folder structure** - staging/, intermediate/, marts/
- **Group related models** - by business domain or data source

### SQL and Transformations
- **Keep models focused** - each model should have a single purpose
- **Use CTEs for readability** - break complex logic into steps
- **Add helpful comments** - explain business logic and assumptions
- **Handle edge cases** - null values, data quality issues

### Testing Strategy
- **Test all primary keys** - unique and not_null constraints
- **Test foreign keys** - relationships between tables
- **Test business rules** - custom data tests for domain logic
- **Test data freshness** - ensure sources are up-to-date

### Documentation
- **Document model purpose** - what it does and why it exists
- **Document column meanings** - business definitions and context
- **Document assumptions** - data quality expectations and business rules
- **Keep docs current** - update when models change

### Performance
- **Use appropriate materializations** - view for small, table for medium, incremental for large
- **Optimize incremental logic** - efficient unique_key and filtering
- **Consider indexing** - for frequently queried columns
- **Monitor query performance** - identify and optimize slow models

Ready to move on to Day 14! 🚀
