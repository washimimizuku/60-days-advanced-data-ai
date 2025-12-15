# Day 14: Project - Governed Data Platform

## 📖 Learning Objectives (30 min)

By the end of today, you will:
- **Integrate** data governance concepts from Days 8-13 into a unified platform
- **Build** a production-ready governed data platform with Airflow orchestration and dbt transformations
- **Implement** comprehensive data catalog, lineage tracking, privacy controls, and access management
- **Deploy** a complete end-to-end data governance solution with monitoring and observability
- **Demonstrate** enterprise-grade data governance practices in a real-world scenario

---

## Theory

### What is a Governed Data Platform?

A governed data platform is a comprehensive data infrastructure that combines data processing capabilities with built-in governance, compliance, and quality controls. It ensures that data is discoverable, trustworthy, secure, and compliant throughout its lifecycle.

**Why Governed Platforms Matter**:
- **Regulatory Compliance**: Meet GDPR, CCPA, SOX, and industry regulations
- **Data Quality**: Ensure accuracy, completeness, and consistency
- **Risk Mitigation**: Prevent data breaches and compliance violations
- **Business Value**: Enable trusted analytics and AI initiatives
- **Operational Efficiency**: Automate governance processes and reduce manual overhead

### Platform Architecture Overview

Our governed data platform integrates the following components:

```
┌─────────────────────────────────────────────────────────────────┐
│                    Governed Data Platform                        │
├─────────────────────────────────────────────────────────────────┤
│  Data Catalog (Day 8)     │  Data Lineage (Day 9)              │
│  - DataHub Integration    │  - Apache Atlas                     │
│  - Metadata Management    │  - Impact Analysis                  │
│  - Search & Discovery     │  - Dependency Tracking             │
├─────────────────────────────────────────────────────────────────┤
│  Privacy Controls (Day 10) │  Access Control (Day 11)          │
│  - GDPR Compliance        │  - RBAC Implementation             │
│  - PII Detection          │  - Row-Level Security              │
│  - Data Anonymization     │  - Multi-tenant Isolation         │
├─────────────────────────────────────────────────────────────────┤
│  Orchestration (Day 12)   │  Transformations (Day 13)          │
│  - Airflow DAGs           │  - dbt Models                      │
│  - Workflow Management    │  - Data Quality Tests              │
│  - Scheduling & Monitoring│  - Documentation Generation        │
├─────────────────────────────────────────────────────────────────┤
│                    Governance Layer                             │
│  - Policy Engine          │  - Audit Logging                   │
│  - Compliance Reporting   │  - Data Quality Monitoring         │
│  - Access Requests        │  - Incident Management             │
└─────────────────────────────────────────────────────────────────┘
```

### Key Integration Points

#### 1. Airflow + dbt Integration
```python
# Airflow DAG with dbt integration
from airflow import DAG
from airflow_dbt.operators.dbt_operator import DbtRunOperator, DbtTestOperator

dag = DAG('governed_data_pipeline', schedule_interval='@daily')

# Data quality checks before transformation
data_quality_check = PythonOperator(
    task_id='data_quality_check',
    python_callable=run_data_quality_checks,
    dag=dag
)

# dbt transformations with governance
dbt_run = DbtRunOperator(
    task_id='dbt_run',
    dir='/opt/dbt',
    dag=dag
)

# Data lineage tracking
lineage_update = PythonOperator(
    task_id='update_lineage',
    python_callable=update_data_lineage,
    dag=dag
)

data_quality_check >> dbt_run >> lineage_update
```

#### 2. Automated Governance Workflows
```python
# Governance automation in Airflow
def governance_workflow():
    """Automated governance checks and updates"""
    
    # 1. Scan for PII in new datasets
    pii_scan_results = scan_for_pii()
    
    # 2. Update data catalog with new assets
    catalog_updates = update_data_catalog()
    
    # 3. Refresh access control policies
    access_control_refresh = refresh_access_policies()
    
    # 4. Generate compliance reports
    compliance_report = generate_compliance_report()
    
    # 5. Update data lineage
    lineage_update = update_lineage_graph()
    
    return {
        'pii_scan': pii_scan_results,
        'catalog': catalog_updates,
        'access_control': access_control_refresh,
        'compliance': compliance_report,
        'lineage': lineage_update
    }
```

#### 3. Data Quality Integration
```sql
-- dbt model with governance annotations
{{ config(
    materialized='table',
    tags=['pii', 'customer_data', 'gdpr'],
    meta={
        'data_classification': 'confidential',
        'retention_period': '7_years',
        'pii_fields': ['email', 'phone', 'address'],
        'access_level': 'restricted'
    }
) }}

with customer_data as (
    select
        customer_id,
        {{ hash_pii('email') }} as email_hash,  -- Privacy protection
        first_name,
        last_name,
        created_at
    from {{ source('raw', 'customers') }}
    where created_at >= '{{ var("start_date") }}'
),

-- Data quality tests embedded
final as (
    select *
    from customer_data
    where customer_id is not null  -- Quality constraint
      and email_hash is not null   -- Privacy constraint
)

select * from final
```

### Governance Automation Patterns

#### 1. Policy-as-Code
```yaml
# governance_policies.yml
data_governance:
  classification_rules:
    - pattern: ".*email.*"
      classification: "PII"
      retention: "7_years"
      access_level: "restricted"
    
    - pattern: ".*ssn.*|.*social_security.*"
      classification: "SENSITIVE_PII"
      retention: "permanent"
      access_level: "highly_restricted"
  
  access_policies:
    - role: "data_analyst"
      allowed_classifications: ["public", "internal"]
      row_level_filters:
        - "region = user_region"
    
    - role: "data_scientist"
      allowed_classifications: ["public", "internal", "confidential"]
      row_level_filters:
        - "department = user_department"
```

#### 2. Automated Compliance Monitoring
```python
class ComplianceMonitor:
    """Automated compliance monitoring and reporting"""
    
    def __init__(self):
        self.policies = load_governance_policies()
        self.audit_logger = AuditLogger()
    
    def monitor_data_access(self):
        """Monitor and log all data access"""
        access_logs = self.get_access_logs()
        
        for log in access_logs:
            # Check for policy violations
            violations = self.check_policy_violations(log)
            
            if violations:
                self.handle_violations(violations)
            
            # Log for audit trail
            self.audit_logger.log_access(log)
    
    def generate_compliance_report(self):
        """Generate automated compliance reports"""
        return {
            'gdpr_compliance': self.check_gdpr_compliance(),
            'data_retention': self.check_retention_policies(),
            'access_violations': self.get_access_violations(),
            'data_quality': self.check_data_quality_metrics()
        }
```

### Production Deployment Architecture

#### Infrastructure Components
```yaml
# docker-compose.yml for governed platform
version: '3.8'
services:
  # Orchestration
  airflow-webserver:
    image: apache/airflow:2.7.0
    environment:
      - AIRFLOW__CORE__EXECUTOR=CeleryExecutor
      - AIRFLOW__CORE__LOAD_EXAMPLES=False
    volumes:
      - ./dags:/opt/airflow/dags
      - ./governance:/opt/airflow/governance
  
  # Data Catalog
  datahub-gms:
    image: linkedin/datahub-gms:latest
    environment:
      - EBEAN_DATASOURCE_URL=jdbc:postgresql://postgres:5432/datahub
  
  # Data Lineage
  atlas:
    image: apache/atlas:2.3.0
    environment:
      - ATLAS_OPTS=-Xmx1024m
  
  # Database
  postgres:
    image: postgres:13
    environment:
      - POSTGRES_DB=governance_platform
      - POSTGRES_USER=platform_user
      - POSTGRES_PASSWORD=secure_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
  
  # Monitoring
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
  
  grafana:
    image: grafana/grafana:latest
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
```

### Monitoring and Observability

#### Key Metrics to Track
```python
# Governance metrics collection
GOVERNANCE_METRICS = {
    'data_quality': [
        'data_quality_score',
        'failed_quality_checks',
        'data_freshness_violations'
    ],
    'compliance': [
        'gdpr_compliance_score',
        'retention_policy_violations',
        'access_policy_violations'
    ],
    'platform_health': [
        'pipeline_success_rate',
        'data_processing_latency',
        'catalog_update_frequency'
    ],
    'security': [
        'unauthorized_access_attempts',
        'pii_exposure_incidents',
        'data_breach_alerts'
    ]
}
```

#### Alerting Framework
```python
class GovernanceAlerting:
    """Governance-specific alerting system"""
    
    def __init__(self):
        self.alert_channels = {
            'critical': ['pagerduty', 'slack_security'],
            'warning': ['slack_data_team'],
            'info': ['email_governance_team']
        }
    
    def check_governance_violations(self):
        """Check for governance violations and alert"""
        
        # Critical: Data breach or PII exposure
        if self.detect_pii_exposure():
            self.send_alert('critical', 'PII_EXPOSURE_DETECTED')
        
        # Warning: Data quality degradation
        if self.check_data_quality_degradation():
            self.send_alert('warning', 'DATA_QUALITY_DEGRADED')
        
        # Info: Policy updates needed
        if self.check_policy_updates_needed():
            self.send_alert('info', 'POLICY_UPDATE_REQUIRED')
```

---

## 💻 Project Implementation (90 minutes)

### Project Overview

**Business Scenario**: You're the Lead Data Engineer at "DataCorp", a financial services company. The company has been growing rapidly and now faces regulatory scrutiny. The CEO has mandated that all data operations must be fully governed, compliant, and auditable within 30 days.

**Your Mission**: Build a comprehensive governed data platform that processes customer financial data while ensuring GDPR compliance, maintaining data quality, and providing full audit trails.

### Project Requirements

#### Functional Requirements
1. **Data Ingestion**: Ingest customer transaction data with automatic PII detection
2. **Data Transformation**: Clean and transform data using dbt with quality tests
3. **Governance Integration**: Automatic catalog updates and lineage tracking
4. **Access Control**: Role-based access with row-level security
5. **Compliance Monitoring**: Automated GDPR compliance checks and reporting
6. **Orchestration**: Airflow DAGs managing the entire governance workflow

#### Non-Functional Requirements
1. **Security**: All PII must be encrypted or anonymized
2. **Auditability**: Complete audit trail of all data operations
3. **Performance**: Process 1M+ records within 30 minutes
4. **Reliability**: 99.9% uptime with automatic failover
5. **Scalability**: Handle 10x data growth without architecture changes

### Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DataCorp Governed Platform                    │
├─────────────────────────────────────────────────────────────────┤
│  Data Sources          │  Governance Layer    │  Outputs         │
│  ┌─────────────────┐   │  ┌─────────────────┐ │  ┌─────────────┐ │
│  │ Customer DB     │──▶│  │ PII Detection   │ │  │ Analytics   │ │
│  │ Transaction API │   │  │ Data Catalog    │ │  │ Dashboards  │ │
│  │ External Feeds  │   │  │ Access Control  │ │  │ Reports     │ │
│  └─────────────────┘   │  │ Quality Checks  │ │  │ APIs        │ │
│                        │  │ Audit Logging   │ │  └─────────────┘ │
│  Processing Layer      │  └─────────────────┘ │                  │
│  ┌─────────────────┐   │                      │  Monitoring      │
│  │ Airflow DAGs    │   │  Compliance Engine   │  ┌─────────────┐ │
│  │ dbt Models      │   │  ┌─────────────────┐ │  │ Grafana     │ │
│  │ Quality Tests   │   │  │ GDPR Checker    │ │  │ Prometheus  │ │
│  │ Lineage Tracker │   │  │ Retention Mgmt  │ │  │ Alerting    │ │
│  └─────────────────┘   │  │ Policy Engine   │ │  └─────────────┘ │
│                        │  └─────────────────┘ │                  │
└─────────────────────────────────────────────────────────────────┘
```

See `project.md` for detailed implementation steps and deliverables.

---

## 📚 Resources

- **Apache Airflow**: [airflow.apache.org](https://airflow.apache.org/) - Orchestration platform
- **dbt Documentation**: [docs.getdbt.com](https://docs.getdbt.com/) - Data transformation
- **DataHub**: [datahubproject.io](https://datahubproject.io/) - Data catalog platform
- **Apache Atlas**: [atlas.apache.org](https://atlas.apache.org/) - Data lineage tracking
- **GDPR Compliance**: [gdpr.eu](https://gdpr.eu/) - Regulatory requirements
- **Data Governance**: [DAMA-DMBOK](https://www.dama.org/cpages/body-of-knowledge) - Industry standards

---

## 🎯 Key Takeaways

- **Governed platforms integrate** multiple governance capabilities into unified solutions
- **Automation is critical** for scalable governance at enterprise scale
- **Policy-as-code** enables consistent, version-controlled governance rules
- **Monitoring and alerting** provide proactive governance violation detection
- **Compliance requires** comprehensive audit trails and automated reporting
- **Integration patterns** connect orchestration, transformation, and governance tools
- **Production deployment** needs robust infrastructure and monitoring
- **Security and privacy** must be built-in, not bolted-on

---

## 🚀 What's Next?

Tomorrow (Day 15), you'll learn **Airflow Production Patterns** - advanced techniques for deploying and scaling Airflow in production environments.

**Preview**: You'll explore advanced Airflow concepts like custom operators, dynamic DAG generation, distributed execution, monitoring strategies, and enterprise deployment patterns. This builds directly on the orchestration foundation you've established in this governed platform!

---

## ✅ Before Moving On

- [ ] Understand how governance components integrate into unified platforms
- [ ] Can implement automated governance workflows with Airflow and dbt
- [ ] Know how to deploy governed platforms with proper monitoring
- [ ] Understand compliance automation and policy-as-code patterns
- [ ] Can build end-to-end governed data pipelines
- [ ] Complete the project implementation
- [ ] Review the solution and architecture patterns

**Time spent**: ~2 hours  
**Difficulty**: ⭐⭐⭐⭐ (Advanced Integration Project)

Ready to build enterprise-grade governed data platforms! 🚀