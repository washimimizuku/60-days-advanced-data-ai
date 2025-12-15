# Day 24 Quiz: Production Pipeline Integration

## Questions

1. **What is the correct order for a production pipeline?**
   - a) Load → Extract → Transform
   - b) Extract → Transform → Load → Validate
   - c) Transform → Extract → Load
   - d) Validate → Extract → Transform

2. **Where should quality checks run?**
   - a) Only at the end
   - b) Only at the beginning
   - c) After each major transformation
   - d) Never

3. **What happens when a quality check fails?**
   - a) Continue anyway
   - b) Stop pipeline and alert
   - c) Delete the data
   - d) Ignore it

4. **How should dbt be integrated with Airflow?**
   - a) Run manually
   - b) As a task in the DAG
   - c) Separately
   - d) Not integrated

5. **What should be monitored in production?**
   - a) Only errors
   - b) Only success rate
   - c) Freshness, volume, quality, performance
   - d) Nothing

## Answers
1. b, 2. c, 3. b, 4. b, 5. c

## Reflection Questions

1. How does orchestration improve pipeline reliability?
2. Why is data quality validation critical?
3. What are the benefits of incremental processing?
4. How does monitoring prevent production issues?
5. What makes a pipeline "production-ready"?
