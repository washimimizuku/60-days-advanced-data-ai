# Day 20: Data Observability - Metrics, Alerting, Dashboards

## 📖 Learning Objectives (15 min)

By the end of today, you will:
- **Master** comprehensive data observability frameworks for production environments
- **Implement** automated anomaly detection and intelligent alerting systems
- **Build** real-time monitoring dashboards with actionable insights
- **Design** incident response workflows and data reliability engineering practices
- **Deploy** end-to-end observability solutions with metrics, logs, and traces

---

## Theory

### Production Data Observability Framework

Data observability extends beyond traditional monitoring to provide comprehensive visibility into data system health, performance, and reliability. It combines metrics, logs, traces, and business context to enable proactive data reliability engineering.

#### 1. The Five Pillars of Data Observability

**Freshness (Timeliness)**:
```python
# Example: Comprehensive freshness monitoring
def monitor_data_freshness(table_name, expected_frequency_minutes=60):
    """Monitor data freshness with business context"""
    
    query = f"""
    SELECT 
        MAX(updated_at) as last_update,
        EXTRACT(EPOCH FROM (NOW() - MAX(updated_at)))/60 as minutes_since_update,
        COUNT(*) as total_records,
        COUNT(CASE WHEN updated_at >= NOW() - INTERVAL '{expected_frequency_minutes} minutes' 
              THEN 1 END) as recent_records
    FROM {table_name}
    """
    
    result = execute_query(query)
    
    freshness_metrics = {
        'table_name': table_name,
        'last_update': result['last_update'],
        'minutes_since_update': result['minutes_since_update'],
        'is_fresh': result['minutes_since_update'] <= expected_frequency_minutes,
        'freshness_score': max(0, 1 - (result['minutes_since_update'] / expected_frequency_minutes)),
        'recent_records_ratio': result['recent_records'] / result['total_records'],
        'expected_frequency_minutes': expected_frequency_minutes,
        'timestamp': datetime.now()
    }
    
    return freshness_metrics
```

**Volume (Completeness)**:
```python
# Example: Advanced volume monitoring with trend analysis
def monitor_data_volume(table_name, lookback_days=7):
    """Monitor data volume with statistical analysis"""
    
    query = f"""
    WITH daily_counts AS (
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as daily_count,
            COUNT(DISTINCT customer_id) as unique_customers,
            AVG(COUNT(*)) OVER (ORDER BY DATE(created_at) 
                               ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) as rolling_avg_7d
        FROM {table_name}
        WHERE created_at >= CURRENT_DATE - INTERVAL '{lookback_days} days'
        GROUP BY DATE(created_at)
        ORDER BY date DESC
    ),
    volume_stats AS (
        SELECT 
            AVG(daily_count) as avg_volume,
            STDDEV(daily_count) as stddev_volume,
            MIN(daily_count) as min_volume,
            MAX(daily_count) as max_volume,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY daily_count) as median_volume
        FROM daily_counts
    )
    SELECT 
        dc.*,
        vs.avg_volume,
        vs.stddev_volume,
        vs.median_volume,
        ABS(dc.daily_count - vs.avg_volume) / NULLIF(vs.stddev_volume, 0) as z_score
    FROM daily_counts dc
    CROSS JOIN volume_stats vs
    WHERE dc.date = CURRENT_DATE
    """
    
    result = execute_query(query)
    
    volume_metrics = {
        'table_name': table_name,
        'current_volume': result['daily_count'],
        'expected_volume': result['avg_volume'],
        'volume_deviation': result['daily_count'] - result['avg_volume'],
        'volume_deviation_percent': ((result['daily_count'] - result['avg_volume']) / result['avg_volume']) * 100,
        'z_score': result['z_score'],
        'is_anomaly': abs(result['z_score']) > 2.0,  # 2 standard deviations
        'rolling_avg_7d': result['rolling_avg_7d'],
        'unique_customers': result['unique_customers'],
        'timestamp': datetime.now()
    }
    
    return volume_metrics
```

**Schema (Structure)**:
```python
# Example: Schema drift detection and monitoring
def monitor_schema_changes(table_name, baseline_schema=None):
    """Monitor schema changes and drift detection"""
    
    current_schema_query = f"""
    SELECT 
        column_name,
        data_type,
        is_nullable,
        column_default,
        ordinal_position
    FROM information_schema.columns 
    WHERE table_name = '{table_name}'
    ORDER BY ordinal_position
    """
    
    current_schema = execute_query(current_schema_query)
    
    if baseline_schema is None:
        # First run - establish baseline
        return {
            'table_name': table_name,
            'schema_status': 'baseline_established',
            'current_schema': current_schema,
            'changes': [],
            'timestamp': datetime.now()
        }
    
    # Compare schemas
    changes = []
    
    # Check for added columns
    baseline_columns = {col['column_name'] for col in baseline_schema}
    current_columns = {col['column_name'] for col in current_schema}
    
    added_columns = current_columns - baseline_columns
    removed_columns = baseline_columns - current_columns
    
    for col in added_columns:
        changes.append({
            'type': 'column_added',
            'column_name': col,
            'details': next(c for c in current_schema if c['column_name'] == col)
        })
    
    for col in removed_columns:
        changes.append({
            'type': 'column_removed',
            'column_name': col,
            'details': next(c for c in baseline_schema if c['column_name'] == col)
        })
    
    # Check for type changes
    for baseline_col in baseline_schema:
        current_col = next((c for c in current_schema if c['column_name'] == baseline_col['column_name']), None)
        if current_col and current_col['data_type'] != baseline_col['data_type']:
            changes.append({
                'type': 'type_changed',
                'column_name': baseline_col['column_name'],
                'old_type': baseline_col['data_type'],
                'new_type': current_col['data_type']
            })
    
    schema_metrics = {
        'table_name': table_name,
        'schema_status': 'stable' if not changes else 'changed',
        'current_schema': current_schema,
        'changes': changes,
        'change_count': len(changes),
        'has_breaking_changes': any(c['type'] in ['column_removed', 'type_changed'] for c in changes),
        'timestamp': datetime.now()
    }
    
    return schema_metrics
```

**Distribution (Data Quality)**:
```python
# Example: Statistical distribution monitoring
def monitor_data_distribution(table_name, numeric_columns, categorical_columns):
    """Monitor data distribution and detect statistical anomalies"""
    
    distribution_metrics = {
        'table_name': table_name,
        'numeric_distributions': {},
        'categorical_distributions': {},
        'anomalies_detected': [],
        'timestamp': datetime.now()
    }
    
    # Monitor numeric columns
    for column in numeric_columns:
        query = f"""
        WITH stats AS (
            SELECT 
                AVG({column}) as mean_val,
                STDDEV({column}) as stddev_val,
                MIN({column}) as min_val,
                MAX({column}) as max_val,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY {column}) as q1,
                PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY {column}) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY {column}) as q3,
                COUNT(*) as total_count,
                COUNT({column}) as non_null_count
            FROM {table_name}
            WHERE created_at >= CURRENT_DATE
        ),
        outliers AS (
            SELECT COUNT(*) as outlier_count
            FROM {table_name}
            WHERE created_at >= CURRENT_DATE
            AND ({column} < (SELECT q1 - 1.5 * (q3 - q1) FROM stats)
                 OR {column} > (SELECT q3 + 1.5 * (q3 - q1) FROM stats))
        )
        SELECT s.*, o.outlier_count
        FROM stats s
        CROSS JOIN outliers o
        """
        
        result = execute_query(query)
        
        distribution_metrics['numeric_distributions'][column] = {
            'mean': result['mean_val'],
            'stddev': result['stddev_val'],
            'min': result['min_val'],
            'max': result['max_val'],
            'q1': result['q1'],
            'median': result['median'],
            'q3': result['q3'],
            'outlier_count': result['outlier_count'],
            'outlier_ratio': result['outlier_count'] / result['non_null_count'],
            'null_ratio': (result['total_count'] - result['non_null_count']) / result['total_count']
        }
        
        # Detect anomalies
        if result['outlier_count'] / result['non_null_count'] > 0.05:  # More than 5% outliers
            distribution_metrics['anomalies_detected'].append({
                'type': 'high_outlier_ratio',
                'column': column,
                'outlier_ratio': result['outlier_count'] / result['non_null_count']
            })
    
    # Monitor categorical columns
    for column in categorical_columns:
        query = f"""
        SELECT 
            {column} as category_value,
            COUNT(*) as count,
            COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () as percentage
        FROM {table_name}
        WHERE created_at >= CURRENT_DATE
        AND {column} IS NOT NULL
        GROUP BY {column}
        ORDER BY count DESC
        """
        
        result = execute_query(query)
        
        distribution_metrics['categorical_distributions'][column] = {
            'unique_values': len(result),
            'top_categories': result[:10],  # Top 10 categories
            'entropy': calculate_entropy([r['count'] for r in result])
        }
    
    return distribution_metrics

def calculate_entropy(counts):
    """Calculate Shannon entropy for categorical distribution"""
    import math
    
    total = sum(counts)
    probabilities = [count / total for count in counts]
    entropy = -sum(p * math.log2(p) for p in probabilities if p > 0)
    return entropy
```

**Lineage (Data Flow)**:
```python
# Example: Data lineage tracking and impact analysis
def track_data_lineage(table_name, lineage_graph):
    """Track data lineage and analyze downstream impact"""
    
    lineage_metrics = {
        'table_name': table_name,
        'upstream_dependencies': [],
        'downstream_consumers': [],
        'lineage_depth': 0,
        'impact_score': 0,
        'timestamp': datetime.now()
    }
    
    # Find upstream dependencies
    upstream = find_upstream_tables(table_name, lineage_graph)
    lineage_metrics['upstream_dependencies'] = upstream
    
    # Find downstream consumers
    downstream = find_downstream_tables(table_name, lineage_graph)
    lineage_metrics['downstream_consumers'] = downstream
    
    # Calculate impact metrics
    lineage_metrics['lineage_depth'] = max(len(upstream), len(downstream))
    lineage_metrics['impact_score'] = calculate_impact_score(downstream)
    
    return lineage_metrics

def calculate_impact_score(downstream_tables):
    """Calculate impact score based on downstream consumers"""
    
    # Weight different types of consumers
    weights = {
        'dashboard': 1.0,
        'ml_model': 2.0,
        'api': 3.0,
        'financial_report': 5.0
    }
    
    impact_score = 0
    for table in downstream_tables:
        table_type = table.get('type', 'dashboard')
        impact_score += weights.get(table_type, 1.0)
    
    return impact_score
```

#### 2. Advanced Anomaly Detection

**Statistical Anomaly Detection**:
```python
# Example: Multi-method anomaly detection
class StatisticalAnomalyDetector:
    """Advanced statistical anomaly detection for data observability"""
    
    def __init__(self):
        self.methods = {
            'z_score': self._z_score_detection,
            'iqr': self._iqr_detection,
            'isolation_forest': self._isolation_forest_detection,
            'seasonal_decomposition': self._seasonal_detection
        }
    
    def detect_anomalies(self, data, methods=['z_score', 'iqr'], threshold=2.0):
        """Detect anomalies using multiple methods"""
        
        anomalies = {
            'detected_anomalies': [],
            'method_results': {},
            'consensus_anomalies': [],
            'confidence_scores': {}
        }
        
        # Run each detection method
        for method in methods:
            if method in self.methods:
                method_anomalies = self.methods[method](data, threshold)
                anomalies['method_results'][method] = method_anomalies
        
        # Find consensus anomalies (detected by multiple methods)
        if len(methods) > 1:
            anomalies['consensus_anomalies'] = self._find_consensus_anomalies(
                anomalies['method_results']
            )
        
        return anomalies
    
    def _z_score_detection(self, data, threshold=2.0):
        """Z-score based anomaly detection"""
        
        mean = np.mean(data)
        std = np.std(data)
        
        z_scores = [(x - mean) / std for x in data]
        anomalies = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
        
        return {
            'anomaly_indices': anomalies,
            'z_scores': z_scores,
            'threshold': threshold,
            'method': 'z_score'
        }
    
    def _iqr_detection(self, data, threshold=1.5):
        """Interquartile Range based anomaly detection"""
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        anomalies = [i for i, x in enumerate(data) 
                    if x < lower_bound or x > upper_bound]
        
        return {
            'anomaly_indices': anomalies,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'method': 'iqr'
        }
    
    def _seasonal_detection(self, data, period=24):
        """Seasonal decomposition based anomaly detection"""
        
        # This would use libraries like statsmodels for seasonal decomposition
        # Simplified implementation for demonstration
        
        seasonal_pattern = self._extract_seasonal_pattern(data, period)
        residuals = [data[i] - seasonal_pattern[i % period] 
                    for i in range(len(data))]
        
        # Apply z-score to residuals
        return self._z_score_detection(residuals)
```

#### 3. Intelligent Alerting System

**Context-Aware Alerting**:
```python
# Example: Intelligent alerting with context and suppression
class IntelligentAlertingSystem:
    """Advanced alerting system with context awareness and noise reduction"""
    
    def __init__(self):
        self.alert_history = []
        self.suppression_rules = []
        self.escalation_policies = {}
        
    def evaluate_alert(self, metric_name, current_value, threshold, context=None):
        """Evaluate whether to send an alert with intelligent filtering"""
        
        alert_candidate = {
            'metric_name': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'context': context or {},
            'timestamp': datetime.now(),
            'severity': self._calculate_severity(current_value, threshold, context)
        }
        
        # Apply intelligent filtering
        if self._should_suppress_alert(alert_candidate):
            return {'action': 'suppressed', 'reason': 'intelligent_filtering'}
        
        # Check for alert fatigue
        if self._is_alert_fatigue(metric_name):
            return {'action': 'suppressed', 'reason': 'alert_fatigue'}
        
        # Enrich alert with context
        enriched_alert = self._enrich_alert_context(alert_candidate)
        
        # Send alert
        self._send_alert(enriched_alert)
        
        # Track alert history
        self.alert_history.append(enriched_alert)
        
        return {'action': 'sent', 'alert': enriched_alert}
    
    def _calculate_severity(self, current_value, threshold, context):
        """Calculate alert severity based on deviation and business context"""
        
        deviation = abs(current_value - threshold) / threshold
        
        # Base severity on deviation
        if deviation >= 0.5:  # 50% deviation
            base_severity = 'critical'
        elif deviation >= 0.2:  # 20% deviation
            base_severity = 'high'
        elif deviation >= 0.1:  # 10% deviation
            base_severity = 'medium'
        else:
            base_severity = 'low'
        
        # Adjust based on business context
        business_impact = context.get('business_impact', 'medium')
        if business_impact == 'critical' and base_severity in ['medium', 'high']:
            base_severity = 'critical'
        elif business_impact == 'low' and base_severity == 'high':
            base_severity = 'medium'
        
        return base_severity
    
    def _should_suppress_alert(self, alert_candidate):
        """Determine if alert should be suppressed based on intelligent rules"""
        
        # Check suppression rules
        for rule in self.suppression_rules:
            if rule['condition'](alert_candidate):
                return True
        
        # Check for duplicate recent alerts
        recent_alerts = [a for a in self.alert_history 
                        if (datetime.now() - a['timestamp']).seconds < 3600]  # Last hour
        
        similar_alerts = [a for a in recent_alerts 
                         if a['metric_name'] == alert_candidate['metric_name']]
        
        if len(similar_alerts) >= 3:  # More than 3 similar alerts in last hour
            return True
        
        return False
    
    def _enrich_alert_context(self, alert):
        """Enrich alert with additional context and suggested actions"""
        
        enriched = alert.copy()
        
        # Add runbook links
        enriched['runbook_url'] = f"https://runbooks.company.com/{alert['metric_name']}"
        
        # Add related metrics
        enriched['related_metrics'] = self._get_related_metrics(alert['metric_name'])
        
        # Add suggested actions
        enriched['suggested_actions'] = self._get_suggested_actions(alert)
        
        # Add impact assessment
        enriched['impact_assessment'] = self._assess_impact(alert)
        
        return enriched
    
    def _get_suggested_actions(self, alert):
        """Generate suggested actions based on alert type and context"""
        
        actions = []
        
        if alert['metric_name'] == 'data_freshness':
            actions.extend([
                "Check upstream data sources for delays",
                "Verify ETL pipeline status",
                "Review data ingestion logs"
            ])
        elif alert['metric_name'] == 'data_volume':
            actions.extend([
                "Compare with historical volume patterns",
                "Check for data source issues",
                "Verify data filtering logic"
            ])
        elif alert['metric_name'] == 'schema_change':
            actions.extend([
                "Review recent schema migrations",
                "Check downstream impact",
                "Validate data type compatibility"
            ])
        
        return actions
```

#### 4. Comprehensive Monitoring Dashboards

**Real-time Observability Dashboard**:
```python
# Example: Dashboard configuration for comprehensive observability
def create_observability_dashboard():
    """Create comprehensive data observability dashboard"""
    
    dashboard_config = {
        "dashboard": {
            "id": "data-observability-production",
            "title": "Production Data Observability Dashboard",
            "tags": ["observability", "data", "production"],
            "time": {"from": "now-24h", "to": "now"},
            "refresh": "30s",
            "timezone": "UTC",
            
            "variables": [
                {
                    "name": "environment",
                    "type": "custom",
                    "options": ["production", "staging", "development"],
                    "current": "production"
                },
                {
                    "name": "data_source",
                    "type": "query",
                    "query": "SELECT DISTINCT table_name FROM data_observability_metrics",
                    "multi": True,
                    "includeAll": True
                }
            ],
            
            "panels": [
                # Overall Health Score
                {
                    "id": 1,
                    "title": "Overall Data Health Score",
                    "type": "stat",
                    "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0},
                    "targets": [
                        {
                            "expr": "avg(data_health_score{environment=\"$environment\",table_name=~\"$data_source\"})",
                            "legendFormat": "Health Score"
                        }
                    ],
                    "fieldConfig": {
                        "defaults": {
                            "unit": "percent",
                            "min": 0,
                            "max": 100,
                            "thresholds": {
                                "steps": [
                                    {"color": "red", "value": 0},
                                    {"color": "yellow", "value": 70},
                                    {"color": "green", "value": 90}
                                ]
                            }
                        }
                    }
                },
                
                # Five Pillars Overview
                {
                    "id": 2,
                    "title": "Five Pillars Status",
                    "type": "bargauge",
                    "gridPos": {"h": 8, "w": 18, "x": 6, "y": 0},
                    "targets": [
                        {
                            "expr": "avg(freshness_score{environment=\"$environment\",table_name=~\"$data_source\"})",
                            "legendFormat": "Freshness"
                        },
                        {
                            "expr": "avg(volume_score{environment=\"$environment\",table_name=~\"$data_source\"})",
                            "legendFormat": "Volume"
                        },
                        {
                            "expr": "avg(schema_score{environment=\"$environment\",table_name=~\"$data_source\"})",
                            "legendFormat": "Schema"
                        },
                        {
                            "expr": "avg(distribution_score{environment=\"$environment\",table_name=~\"$data_source\"})",
                            "legendFormat": "Distribution"
                        },
                        {
                            "expr": "avg(lineage_score{environment=\"$environment\",table_name=~\"$data_source\"})",
                            "legendFormat": "Lineage"
                        }
                    ]
                },
                
                # Anomaly Detection Timeline
                {
                    "id": 3,
                    "title": "Anomaly Detection Timeline",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 24, "x": 0, "y": 8},
                    "targets": [
                        {
                            "expr": "sum(rate(anomalies_detected_total{environment=\"$environment\",table_name=~\"$data_source\"}[5m]))",
                            "legendFormat": "Anomalies per minute"
                        }
                    ]
                },
                
                # Data Freshness Heatmap
                {
                    "id": 4,
                    "title": "Data Freshness Heatmap",
                    "type": "heatmap",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
                    "targets": [
                        {
                            "expr": "freshness_minutes{environment=\"$environment\",table_name=~\"$data_source\"}",
                            "format": "heatmap"
                        }
                    ]
                },
                
                # Volume Trends
                {
                    "id": 5,
                    "title": "Volume Trends",
                    "type": "timeseries",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
                    "targets": [
                        {
                            "expr": "data_volume{environment=\"$environment\",table_name=~\"$data_source\"}",
                            "legendFormat": "{{ table_name }}"
                        }
                    ]
                },
                
                # Schema Changes
                {
                    "id": 6,
                    "title": "Recent Schema Changes",
                    "type": "table",
                    "gridPos": {"h": 8, "w": 12, "x": 0, "y": 24},
                    "targets": [
                        {
                            "expr": "schema_changes{environment=\"$environment\",table_name=~\"$data_source\"}",
                            "format": "table",
                            "instant": True
                        }
                    ]
                },
                
                # Active Alerts
                {
                    "id": 7,
                    "title": "Active Data Alerts",
                    "type": "alertlist",
                    "gridPos": {"h": 8, "w": 12, "x": 12, "y": 24},
                    "options": {
                        "showOptions": "current",
                        "maxItems": 20,
                        "sortOrder": 1,
                        "tags": ["data_observability"]
                    }
                }
            ]
        }
    }
    
    return dashboard_config
```

---

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose installed
- Python 3.8+ with pip
- 8GB+ RAM recommended
- 5GB+ free disk space

### Option 1: Automated Setup (Recommended)
```bash
# Clone and navigate to the directory
cd days/day-20-data-observability

# Run the automated setup script
./setup.sh
```

### Option 2: Manual Setup
```bash
# Start the infrastructure
docker-compose up -d

# Install Python dependencies
pip install -r requirements.txt

# Run the demo
python3 demo.py
```

### Option 3: Development Mode
```bash
# Start services
docker-compose up -d

# Access Jupyter for interactive analysis
open http://localhost:8888
# Token: observability
```

### Access Points
Once setup is complete:
- **Grafana Dashboards**: http://localhost:3000 (admin/admin)
- **Prometheus Metrics**: http://localhost:9090
- **Jupyter Analysis**: http://localhost:8888 (token: observability)
- **PostgreSQL**: localhost:5432 (obs_user/obs_password)

---

## 💻 Hands-On Exercise (40 minutes)

Build a comprehensive data observability system with real-time monitoring, anomaly detection, and intelligent alerting.

**Scenario**: You're the Data Reliability Engineer at "ObserveTech Solutions", a company processing critical business data across multiple systems. You need to implement a comprehensive observability solution that provides complete visibility into data health and proactively detects issues.

**Requirements**:
1. **Five Pillars Monitoring**: Implement monitoring for freshness, volume, schema, distribution, and lineage
2. **Anomaly Detection**: Build automated anomaly detection with multiple algorithms
3. **Intelligent Alerting**: Create context-aware alerting with noise reduction
4. **Real-time Dashboards**: Build comprehensive monitoring dashboards
5. **Incident Response**: Implement automated incident response workflows
6. **Impact Analysis**: Track data lineage and assess downstream impact

**Infrastructure Provided**:
- PostgreSQL database with sample transaction data
- Prometheus + Grafana monitoring stack
- Redis for real-time caching
- Jupyter notebooks for interactive analysis
- Complete Docker environment

**Data Sources**:
- Customer transaction streams (high-frequency, critical)
- Product catalog updates (schema changes, business impact)
- User behavior events (volume variations, seasonal patterns)
- Financial reporting data (accuracy critical, compliance requirements)

See `exercise.py` for starter code and `demo.py` for working examples.

---

## 📁 Files Structure

### Core Files
- `README.md` - This comprehensive guide
- `exercise.py` - Implementation exercises with TODOs
- `solution.py` - Complete working solution
- `quiz.md` - Knowledge assessment questions
- `demo.py` - Interactive demonstration script

### Infrastructure
- `docker-compose.yml` - Complete environment setup
- `requirements.txt` - Python dependencies
- `setup.sh` - Automated setup script
- `.env.example` - Environment configuration template

### Database
- `sql/init.sql` - Database schema initialization
- `sql/sample_data.sql` - Sample data generation

### Monitoring
- `monitoring/prometheus.yml` - Metrics collection config
- `monitoring/grafana/` - Dashboard and datasource configs

### Analysis
- `notebooks/data_observability_analysis.ipynb` - Interactive analysis
- `Dockerfile.jupyter` - Jupyter environment setup

---

## 📚 Resources

- **Monte Carlo**: [docs.getmontecarlo.com](https://docs.getmontecarlo.com/) - Data observability platform
- **Datafold**: [docs.datafold.com](https://docs.datafold.com/) - Data diff and monitoring
- **elementary**: [docs.elementary-data.com](https://docs.elementary-data.com/) - dbt-native observability
- **OpenTelemetry**: [opentelemetry.io](https://opentelemetry.io/) - Observability framework
- **Prometheus**: [prometheus.io/docs](https://prometheus.io/docs/) - Monitoring and alerting
- **Grafana**: [grafana.com/docs](https://grafana.com/docs/) - Observability dashboards

---

## 🎯 Key Takeaways

- **Five pillars approach** provides comprehensive coverage of data health dimensions
- **Multi-method anomaly detection** increases accuracy and reduces false positives
- **Intelligent alerting** prevents alert fatigue while ensuring critical issues are escalated
- **Real-time monitoring** enables proactive issue detection and faster resolution
- **Context-aware systems** provide actionable insights rather than just notifications
- **Impact analysis** helps prioritize issues based on downstream business effects
- **Automated response** reduces manual effort and improves mean time to resolution
- **Continuous monitoring** builds confidence in data reliability and system health

---

## 🚀 What's Next?

Tomorrow (Day 21), you'll learn **Testing Strategies** - comprehensive testing approaches for data pipelines including unit, integration, and end-to-end testing.

**Preview**: You'll explore advanced testing patterns, implement automated test suites, and build comprehensive validation frameworks that ensure data pipeline reliability and correctness!

---

## ✅ Before Moving On

- [ ] Understand the five pillars of data observability
- [ ] Can implement multi-method anomaly detection
- [ ] Know how to build intelligent alerting systems
- [ ] Understand real-time monitoring and dashboards
- [ ] Can design incident response workflows
- [ ] Complete the hands-on exercise
- [ ] Take the quiz

**Time spent**: ~1 hour  
**Difficulty**: ⭐⭐⭐⭐ (Advanced Data Observability)

Ready to ensure complete visibility into data system health! 🚀
