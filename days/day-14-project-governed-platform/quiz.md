# Day 14 Quiz: Governed Data Platform Project

## Test your understanding of governed data platform concepts and implementation

### 1. **What is the primary purpose of integrating DataHub in a governed data platform?**
   - A) Data transformation and modeling
   - B) Metadata management and data discovery
   - C) Workflow orchestration and scheduling
   - D) Data quality testing and validation

### 2. **In the governed platform architecture, what role does Airflow play in governance workflows?**
   - A) Stores metadata and lineage information
   - B) Orchestrates governance tasks like PII detection and compliance reporting
   - C) Provides data visualization and dashboards
   - D) Manages user authentication and authorization

### 3. **Which dbt configuration ensures GDPR compliance in data models?**
   - A) Setting materialized='table' for all models
   - B) Using tags=['gdpr'] and implementing PII protection in transformations
   - C) Creating separate schemas for each data source
   - D) Enabling incremental model updates only

### 4. **What is the correct approach for PII protection in a governed platform?**
   - A) Delete all PII fields from datasets
   - B) Store PII in separate encrypted databases
   - C) Apply hashing, masking, or encryption based on data classification policies
   - D) Only allow admin users to access any PII data

### 5. **How should compliance reporting be automated in a governed platform?**
   - A) Manual generation of reports by compliance officers
   - B) Scheduled Airflow DAGs that generate compliance metrics and store results
   - C) Real-time streaming of all data changes to regulators
   - D) Weekly batch exports of raw data for manual analysis

### 6. **What is the purpose of the audit_logs table in the governed platform?**
   - A) Store application error messages and debugging information
   - B) Track all data access, modifications, and governance events for compliance
   - C) Cache frequently accessed data for performance optimization
   - D) Store user preferences and application configuration settings

### 7. **Which monitoring metric is most critical for GDPR compliance?**
   - A) Database query response time
   - B) Number of active user sessions
   - C) PII protection coverage percentage and consent compliance rate
   - D) Total data storage capacity utilization

### 8. **In the Docker Compose setup, why are health checks important for governance services?**
   - A) They improve application performance and reduce latency
   - B) They ensure services are ready before dependent governance workflows start
   - C) They automatically scale services based on resource usage
   - D) They provide user authentication and session management

### 9. **What is the benefit of implementing policy-as-code in data governance?**
   - A) Reduces database storage requirements
   - B) Enables version-controlled, consistent, and automated policy enforcement
   - C) Improves data transformation performance
   - D) Simplifies user interface design and navigation

### 10. **How does the governed platform handle data retention policies?**
   - A) Manually delete old data when storage is full
   - B) Automatically archive and delete data based on classification and retention rules
   - C) Keep all data indefinitely for historical analysis
   - D) Only retain data that is frequently accessed by users

---

**Answers:**
1. B - DataHub provides metadata management and data discovery capabilities
2. B - Airflow orchestrates governance workflows including PII detection and compliance reporting
3. B - Using GDPR tags and implementing PII protection ensures compliance
4. C - Apply appropriate protection methods based on data classification policies
5. B - Automated compliance reporting through scheduled Airflow DAGs
6. B - Audit logs track all governance events for compliance and regulatory requirements
7. C - PII protection and consent compliance are critical GDPR metrics
8. B - Health checks ensure service readiness before governance workflows execute
9. B - Policy-as-code enables consistent, version-controlled governance automation
10. B - Automated retention based on classification and policy rules
