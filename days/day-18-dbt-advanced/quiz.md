# Day 18 Quiz: dbt Advanced - Custom Materializations, Packages, Advanced Analytics

## Questions

### 1. What is the primary purpose of Jinja templating in dbt macros?
- a) To improve query performance
- b) To generate dynamic SQL based on parameters and conditions
- c) To create database connections
- d) To validate data quality

### 2. Which function is used to execute SQL queries within a macro during compilation?
- a) execute_query()
- b) run_query()
- c) sql_query()
- d) compile_query()

### 3. In attribution modeling, what does "time decay" attribution give more weight to?
- a) The first touchpoint in the customer journey
- b) The last touchpoint before conversion
- c) More recent touchpoints before conversion
- d) All touchpoints equally

### 4. What is the main advantage of creating custom dbt packages?
- a) Faster query execution
- b) Better data quality
- c) Code reusability across projects and teams
- d) Automatic testing

### 5. In cohort analysis, what does "retention rate" measure?
- a) The percentage of customers who make repeat purchases
- b) The percentage of users from a cohort who remain active in subsequent periods
- c) The average revenue per customer
- d) The time between first and last purchase

### 6. Which configuration is required when creating a custom materialization?
- a) unique_key
- b) partition_by
- c) The materialization name and implementation logic
- d) on_schema_change

### 7. In predictive CLV modeling, what does the discount rate represent?
- a) The percentage discount offered to customers
- b) The time value of money for future cash flows
- c) The churn probability
- d) The customer acquisition cost

### 8. What is the purpose of the `dispatch` configuration in dbt packages?
- a) To schedule when models run
- b) To control macro search order across packages for cross-database compatibility
- c) To distribute data across partitions
- d) To manage package dependencies

### 9. Which attribution model gives equal weight to the first and last touchpoints, with remaining weight distributed among middle touchpoints?
- a) Linear attribution
- b) Time decay attribution
- c) U-shaped attribution
- d) First-touch attribution

### 10. What is a key benefit of using generic tests in dbt packages?
- a) They run faster than other tests
- b) They can be reused across multiple models with different parameters
- c) They don't require configuration
- d) They automatically fix data quality issues

---

## Answers

### 1. What is the primary purpose of Jinja templating in dbt macros?
**Answer: b) To generate dynamic SQL based on parameters and conditions**

**Explanation:** Jinja templating is the core feature that makes dbt macros powerful. It allows you to write dynamic SQL that changes based on parameters, conditions, loops, and other logic. This enables creating reusable, parameterized SQL functions that can generate different queries based on input parameters, making analytics code more maintainable and flexible.

---

### 2. Which function is used to execute SQL queries within a macro during compilation?
**Answer: b) run_query()**

**Explanation:** The `run_query()` function is used within dbt macros to execute SQL queries during compilation time. This is essential for macros that need to inspect database metadata, get distinct values for dynamic SQL generation, or perform other operations that require querying the database before generating the final SQL. It's commonly used in advanced macros for dynamic pivot tables or schema introspection.

---

### 3. In attribution modeling, what does "time decay" attribution give more weight to?
**Answer: c) More recent touchpoints before conversion**

**Explanation:** Time decay attribution assigns higher weights to touchpoints that occurred closer to the conversion event. The logic is that more recent interactions are more influential in driving the conversion decision. This model uses exponential decay, where the weight decreases as the time between the touchpoint and conversion increases, making it useful for understanding the immediate drivers of customer actions.

---

### 4. What is the main advantage of creating custom dbt packages?
**Answer: c) Code reusability across projects and teams**

**Explanation:** Custom dbt packages enable code reusability by packaging macros, models, tests, and other dbt components that can be shared across multiple projects and teams. This promotes consistency, reduces duplication, enables standardization of analytics patterns, and allows teams to build on each other's work. Packages can be distributed via Git repositories or dbt Hub for broader community use.

---

### 5. In cohort analysis, what does "retention rate" measure?
**Answer: b) The percentage of users from a cohort who remain active in subsequent periods**

**Explanation:** Retention rate in cohort analysis measures what percentage of users from a specific cohort (defined by a common characteristic like signup date) continue to be active in later time periods. For example, if 100 users signed up in January and 60 are still active in March, the 2-month retention rate is 60%. This metric is crucial for understanding user engagement and product stickiness over time.

---

### 6. Which configuration is required when creating a custom materialization?
**Answer: c) The materialization name and implementation logic**

**Explanation:** Custom materializations require defining the materialization name and implementing the core logic that determines how the model should be built in the database. This includes handling different scenarios (first run, incremental updates, full refresh) and defining the SQL operations needed to create or update the target relation. The implementation uses dbt's materialization framework with specific hooks and functions.

---

### 7. In predictive CLV modeling, what does the discount rate represent?
**Answer: b) The time value of money for future cash flows**

**Explanation:** The discount rate in CLV modeling represents the time value of money - the concept that money received in the future is worth less than money received today. It's used to calculate the present value of future customer revenue streams. A higher discount rate means future cash flows are discounted more heavily, resulting in lower CLV calculations. This reflects business concepts like opportunity cost and risk.

---

### 8. What is the purpose of the `dispatch` configuration in dbt packages?
**Answer: b) To control macro search order across packages for cross-database compatibility**

**Explanation:** The `dispatch` configuration controls the order in which dbt searches for macro implementations across different packages. This is crucial for cross-database compatibility because different data warehouses may require different SQL syntax. Packages can provide database-specific implementations of the same macro, and dispatch ensures the correct version is used based on the target database.

---

### 9. Which attribution model gives equal weight to the first and last touchpoints, with remaining weight distributed among middle touchpoints?
**Answer: c) U-shaped attribution**

**Explanation:** U-shaped (or position-based) attribution typically gives 40% weight each to the first and last touchpoints, with the remaining 20% distributed equally among middle touchpoints. This model recognizes that both the initial awareness touchpoint and the final conversion touchpoint are important, while still giving some credit to nurturing touchpoints in between. It's useful for understanding both acquisition and conversion drivers.

---

### 10. What is a key benefit of using generic tests in dbt packages?
**Answer: b) They can be reused across multiple models with different parameters**

**Explanation:** Generic tests in dbt packages are parameterized test templates that can be applied to different models with different configurations. Instead of writing custom SQL tests for each model, you can create a generic test once and reuse it across multiple models by passing different parameters. This promotes consistency in testing logic, reduces code duplication, and makes it easier to maintain and update test logic across an entire analytics codebase.

---

## Score Interpretation

- **9-10 correct**: dbt Advanced Master! You understand sophisticated analytics engineering and package development
- **7-8 correct**: Strong advanced knowledge! Review custom materializations and advanced analytics patterns
- **5-6 correct**: Good foundation in advanced dbt! Focus on Jinja templating and package development
- **3-4 correct**: Basic understanding present! Study macro development and analytics patterns
- **Below 3**: Review the theory section and work through advanced dbt examples

---

## Key Concepts to Remember

### Advanced Macro Development
1. **Jinja templating** enables dynamic SQL generation with parameters and conditions
2. **run_query() function** executes SQL during compilation for metadata inspection
3. **Complex logic** can be implemented with loops, conditionals, and variable assignments
4. **Cross-database compatibility** through conditional SQL generation
5. **Performance optimization** with intelligent caching and query generation

### Package Development
6. **Code reusability** across projects and teams through well-designed packages
7. **Semantic versioning** for package releases and dependency management
8. **Dispatch configuration** for cross-database macro compatibility
9. **Comprehensive documentation** and examples for package adoption
10. **Generic tests** for reusable validation logic across models

### Advanced Analytics Patterns
11. **Attribution modeling** with multiple strategies for marketing effectiveness
12. **Cohort analysis** for understanding user retention and behavior patterns
13. **Predictive CLV modeling** with churn probability and confidence intervals
14. **Customer segmentation** with sophisticated scoring and classification
15. **Performance monitoring** for analytics platform health and optimization

### Custom Materializations
16. **Materialization framework** for implementing custom build strategies
17. **Incremental logic** handling for complex update patterns
18. **Error handling** and recovery mechanisms for robust implementations
19. **Performance optimization** through intelligent caching and indexing
20. **Cross-database support** for portable materialization strategies

### Production Best Practices
- **Parameterize macros** for flexibility and reusability
- **Document thoroughly** with clear examples and use cases
- **Test comprehensively** including edge cases and error conditions
- **Version carefully** with semantic versioning and migration guides
- **Optimize performance** with efficient SQL generation and caching
- **Handle errors gracefully** with meaningful error messages and fallbacks

### Common Anti-Patterns to Avoid
- **Hardcoding values** in macros instead of using parameters
- **Complex logic in models** instead of reusable macros
- **Poor error handling** without meaningful error messages
- **Inadequate documentation** making packages hard to use
- **Performance issues** from inefficient SQL generation
- **Breaking changes** without proper versioning and migration paths

Ready to move on to Day 19! 🚀
