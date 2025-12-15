"""
Day 20: Data Observability - Exercise

Build a comprehensive data observability system with real-time monitoring, anomaly detection, and intelligent alerting.

Scenario:
You're the Data Reliability Engineer at "ObserveTech Solutions", a company 
processing critical business data across multiple systems. You need to implement 
a comprehensive observability solution that provides complete visibility into 
data health and proactively detects issues.

Business Context:
- Processing 50M+ events daily across multiple data streams
- Critical business dashboards depend on real-time data
- Regulatory compliance requires data accuracy and completeness
- Multiple teams consume data with different SLA requirements
- Need to minimize false positive alerts while catching real issues

Your Task:
Build a comprehensive data observability system with monitoring, detection, and alerting.

Requirements:
1. Five pillars monitoring (freshness, volume, schema, distribution, lineage)
2. Multi-method anomaly detection with statistical analysis
3. Intelligent alerting with context awareness and noise reduction
4. Real-time dashboards with actionable insights
5. Automated incident response workflows
6. Impact analysis and downstream effect assessment
"""

import os
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =============================================================================
# DATA OBSERVABILITY FRAMEWORK IMPLEMENTATION
# =============================================================================

class ObserveTechDataObservabilityFramework:
    """Comprehensive data observability framework for ObserveTech Solutions"""
    
    def __init__(self):
        self.monitoring_systems = {}
        self.anomaly_detectors = {}
        self.alerting_system = None
        self.dashboard_generator = None
        
    def initialize_monitoring_systems(self):
        """Initialize all monitoring systems for the five pillars"""
        
        # TODO: Initialize monitoring systems for each pillar
        # Set up freshness, volume, schema, distribution, and lineage monitoring
        
        self.monitoring_systems = {
            'freshness': FreshnessMonitor(),
            'volume': VolumeMonitor(),
            'schema': SchemaMonitor(),
            'distribution': DistributionMonitor(),
            'lineage': LineageMonitor()
        }
        
        return self.monitoring_systems

# =============================================================================
# FIVE PILLARS MONITORING IMPLEMENTATION
# =============================================================================

class FreshnessMonitor:
    """Monitor data freshness and timeliness"""
    
    def __init__(self):
        self.freshness_thresholds = {}
        
    def monitor_freshness(self, table_name: str, expected_frequency_minutes: int = 60) -> Dict[str, Any]:
        """Monitor data freshness with business context"""
        
        # TODO: Implement comprehensive freshness monitoring
        # Check last update time, calculate freshness score, detect delays
        
        # Simulate database query for last update
        last_update_query = f"""
        SELECT 
            MAX(updated_at) as last_update,
            COUNT(*) as total_records,
            COUNT(CASE WHEN updated_at >= NOW() - INTERVAL '{expected_frequency_minutes} minutes' 
                  THEN 1 END) as recent_records
        FROM {table_name}
        """
        
        # Simulate database query execution
        try:
            # In production, execute the query against actual database
            current_time = datetime.now()
            last_update = current_time - timedelta(minutes=45)  # Simulate 45 minutes old
            total_records = 150000
            recent_records = 12000
            
            minutes_since_update = (current_time - last_update).total_seconds() / 60
            is_fresh = minutes_since_update <= expected_frequency_minutes
            freshness_score = max(0, 1 - (minutes_since_update / expected_frequency_minutes))
            recent_records_ratio = recent_records / total_records if total_records > 0 else 0
            
            freshness_metrics = {
                'table_name': table_name,
                'last_update': last_update.isoformat(),
                'minutes_since_update': minutes_since_update,
                'is_fresh': is_fresh,
                'freshness_score': freshness_score,
                'recent_records_ratio': recent_records_ratio,
                'expected_frequency_minutes': expected_frequency_minutes,
                'timestamp': current_time.isoformat()
            }
        except Exception as e:
            print(f"Error monitoring freshness for {table_name}: {e}")
            freshness_metrics = {
                'table_name': table_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        return freshness_metrics

class VolumeMonitor:
    """Monitor data volume and detect volume anomalies"""
    
    def __init__(self):
        self.volume_baselines = {}
        
    def monitor_volume(self, table_name: str, lookback_days: int = 7) -> Dict[str, Any]:
        """Monitor data volume with statistical analysis"""
        
        # TODO: Implement comprehensive volume monitoring
        # Calculate daily volumes, rolling averages, detect anomalies using z-score
        
        volume_query = f"""
        WITH daily_counts AS (
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as daily_count,
                COUNT(DISTINCT customer_id) as unique_customers
            FROM {table_name}
            WHERE created_at >= CURRENT_DATE - INTERVAL '{lookback_days} days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        ),
        volume_stats AS (
            SELECT 
                AVG(daily_count) as avg_volume,
                STDDEV(daily_count) as stddev_volume
            FROM daily_counts
        )
        SELECT dc.*, vs.avg_volume, vs.stddev_volume
        FROM daily_counts dc
        CROSS JOIN volume_stats vs
        WHERE dc.date = CURRENT_DATE
        """
        
        # Simulate database query execution
        try:
            # In production, execute the query against actual database
            current_volume = 125000
            historical_volumes = [120000, 118000, 122000, 119000, 121000, 123000, 120500]
            unique_customers = 45000
            
            expected_volume = sum(historical_volumes) / len(historical_volumes)
            volume_deviation = current_volume - expected_volume
            volume_deviation_percent = (volume_deviation / expected_volume) * 100 if expected_volume > 0 else 0
            
            # Calculate z-score
            if len(historical_volumes) > 1:
                std_dev = np.std(historical_volumes)
                z_score = volume_deviation / std_dev if std_dev > 0 else 0
            else:
                z_score = 0
            
            is_anomaly = abs(z_score) > 2.0
            
            volume_metrics = {
                'table_name': table_name,
                'current_volume': current_volume,
                'expected_volume': expected_volume,
                'volume_deviation': volume_deviation,
                'volume_deviation_percent': volume_deviation_percent,
                'z_score': z_score,
                'is_anomaly': is_anomaly,
                'unique_customers': unique_customers,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error monitoring volume for {table_name}: {e}")
            volume_metrics = {
                'table_name': table_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        return volume_metrics

class SchemaMonitor:
    """Monitor schema changes and drift detection"""
    
    def __init__(self):
        self.baseline_schemas = {}
        
    def monitor_schema_changes(self, table_name: str, baseline_schema: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Monitor schema changes and detect drift"""
        
        # TODO: Implement schema monitoring
        # Get current schema, compare with baseline, detect changes
        
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
        
        # TODO: Execute query to get current schema
        current_schema = []  # TODO: Get from query result
        
        if baseline_schema is None:
            # First run - establish baseline
            return {
                'table_name': table_name,
                'schema_status': 'baseline_established',
                'current_schema': current_schema,
                'changes': [],
                'timestamp': datetime.now()
            }
        
        # TODO: Compare schemas and detect changes
        # Check for: added columns, removed columns, type changes
        
        changes = []  # TODO: Implement change detection
        
        schema_metrics = {
            'table_name': table_name,
            'schema_status': 'stable',  # TODO: Determine status
            'current_schema': current_schema,
            'changes': changes,
            'change_count': len(changes),
            'has_breaking_changes': False,  # TODO: Determine if breaking
            'timestamp': datetime.now()
        }
        
        return schema_metrics

class DistributionMonitor:
    """Monitor data distribution and statistical properties"""
    
    def __init__(self):
        self.distribution_baselines = {}
        
    def monitor_data_distribution(self, table_name: str, numeric_columns: List[str], 
                                categorical_columns: List[str]) -> Dict[str, Any]:
        """Monitor data distribution and detect statistical anomalies"""
        
        # TODO: Implement distribution monitoring
        # Monitor numeric distributions (mean, std, outliers) and categorical distributions (entropy, top categories)
        
        distribution_metrics = {
            'table_name': table_name,
            'numeric_distributions': {},
            'categorical_distributions': {},
            'anomalies_detected': [],
            'timestamp': datetime.now()
        }
        
        # Monitor numeric columns
        for column in numeric_columns:
            try:
                # Simulate numeric distribution analysis
                sample_data = np.random.normal(100, 15, 1000)  # Simulate data
                
                mean_val = float(np.mean(sample_data))
                stddev_val = float(np.std(sample_data))
                min_val = float(np.min(sample_data))
                max_val = float(np.max(sample_data))
                q1 = float(np.percentile(sample_data, 25))
                median = float(np.percentile(sample_data, 50))
                q3 = float(np.percentile(sample_data, 75))
                
                # Calculate outliers using IQR method
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = sample_data[(sample_data < lower_bound) | (sample_data > upper_bound)]
                
                distribution_metrics['numeric_distributions'][column] = {
                    'mean': mean_val,
                    'stddev': stddev_val,
                    'min': min_val,
                    'max': max_val,
                    'q1': q1,
                    'median': median,
                    'q3': q3,
                    'outlier_count': len(outliers),
                    'outlier_ratio': len(outliers) / len(sample_data)
                }
            except Exception as e:
                print(f"Error analyzing numeric column {column}: {e}")
                distribution_metrics['numeric_distributions'][column] = {'error': str(e)}
        
        # Monitor categorical columns
        for column in categorical_columns:
            try:
                # Simulate categorical distribution analysis
                categories = ['active', 'inactive', 'suspended', 'cancelled', 'trial']
                counts = [5000, 2000, 500, 1000, 1500]
                total_count = sum(counts)
                
                top_categories = [
                    {'category_value': cat, 'count': count, 'percentage': (count / total_count) * 100}
                    for cat, count in zip(categories, counts)
                ]
                
                # Calculate Shannon entropy
                probabilities = [count / total_count for count in counts]
                entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
                
                distribution_metrics['categorical_distributions'][column] = {
                    'unique_values': len(categories),
                    'top_categories': sorted(top_categories, key=lambda x: x['count'], reverse=True),
                    'entropy': entropy
                }
            except Exception as e:
                print(f"Error analyzing categorical column {column}: {e}")
                distribution_metrics['categorical_distributions'][column] = {'error': str(e)}
        
        return distribution_metrics

class LineageMonitor:
    """Monitor data lineage and impact analysis"""
    
    def __init__(self):
        self.lineage_graph = {}
        
    def track_data_lineage(self, table_name: str) -> Dict[str, Any]:
        """Track data lineage and analyze downstream impact"""
        
        # Implement lineage tracking
        try:
            # Simulate lineage discovery
            lineage_map = {
                'customer_data': {
                    'upstream': ['raw_events', 'crm_system'],
                    'downstream': ['analytics_dashboard', 'ml_pipeline', 'reports']
                },
                'transaction_data': {
                    'upstream': ['payment_gateway', 'order_system'],
                    'downstream': ['financial_reports', 'fraud_detection']
                }
            }
            
            table_lineage = lineage_map.get(table_name, {'upstream': [], 'downstream': []})
            
            upstream_dependencies = [
                {'table_name': dep, 'criticality': 'high'} 
                for dep in table_lineage['upstream']
            ]
            
            downstream_consumers = [
                {'table_name': consumer, 'type': 'dashboard' if 'dashboard' in consumer else 'pipeline'}
                for consumer in table_lineage['downstream']
            ]
            
            # Calculate impact score based on number and type of consumers
            impact_score = len(downstream_consumers) * 2.5
            lineage_depth = max(len(upstream_dependencies), len(downstream_consumers))
            
            lineage_metrics = {
                'table_name': table_name,
                'upstream_dependencies': upstream_dependencies,
                'downstream_consumers': downstream_consumers,
                'lineage_depth': lineage_depth,
                'impact_score': impact_score,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error tracking lineage for {table_name}: {e}")
            lineage_metrics = {
                'table_name': table_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        return lineage_metrics

# =============================================================================
# ANOMALY DETECTION SYSTEM
# =============================================================================

class StatisticalAnomalyDetector:
    """Advanced statistical anomaly detection for data observability"""
    
    def __init__(self):
        self.methods = {
            'z_score': self._z_score_detection,
            'iqr': self._iqr_detection,
            'isolation_forest': self._isolation_forest_detection
        }
        
    def detect_anomalies(self, data: List[float], methods: List[str] = ['z_score', 'iqr'], 
                        threshold: float = 2.0) -> Dict[str, Any]:
        """Detect anomalies using multiple methods"""
        
        # TODO: Implement multi-method anomaly detection
        # Run each detection method and find consensus anomalies
        
        anomalies = {
            'detected_anomalies': [],
            'method_results': {},
            'consensus_anomalies': [],
            'confidence_scores': {}
        }
        
        # TODO: Run each detection method
        for method in methods:
            if method in self.methods:
                # TODO: Execute method and store results
                method_anomalies = self.methods[method](data, threshold)
                anomalies['method_results'][method] = method_anomalies
        
        # TODO: Find consensus anomalies (detected by multiple methods)
        if len(methods) > 1:
            anomalies['consensus_anomalies'] = self._find_consensus_anomalies(
                anomalies['method_results']
            )
        
        return anomalies
    
    def _z_score_detection(self, data: List[float], threshold: float = 2.0) -> Dict[str, Any]:
        """Z-score based anomaly detection"""
        
        # TODO: Implement z-score anomaly detection
        # Calculate mean, std, z-scores, and identify anomalies
        
        if len(data) == 0:
            return {'anomaly_indices': [], 'z_scores': [], 'method': 'z_score'}
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return {'anomaly_indices': [], 'z_scores': [0] * len(data), 'method': 'z_score'}
        
        # TODO: Calculate z-scores and find anomalies
        z_scores = [(x - mean) / std for x in data]
        anomalies = [i for i, z in enumerate(z_scores) if abs(z) > threshold]
        
        return {
            'anomaly_indices': anomalies,
            'z_scores': z_scores,
            'threshold': threshold,
            'method': 'z_score'
        }
    
    def _iqr_detection(self, data: List[float], threshold: float = 1.5) -> Dict[str, Any]:
        """Interquartile Range based anomaly detection"""
        
        # TODO: Implement IQR anomaly detection
        # Calculate quartiles, IQR, bounds, and identify outliers
        
        if len(data) == 0:
            return {'anomaly_indices': [], 'method': 'iqr'}
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        
        # TODO: Find anomalies outside bounds
        anomalies = [i for i, x in enumerate(data) 
                    if x < lower_bound or x > upper_bound]
        
        return {
            'anomaly_indices': anomalies,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'method': 'iqr'
        }
    
    def _isolation_forest_detection(self, data: List[float], threshold: float = 0.1) -> Dict[str, Any]:
        """Isolation Forest based anomaly detection"""
        
        # TODO: Implement Isolation Forest anomaly detection
        # This would typically use scikit-learn's IsolationForest
        # For now, implement a simplified version
        
        # Simplified implementation - in practice, use sklearn.ensemble.IsolationForest
        anomalies = []  # TODO: Implement isolation forest logic
        
        return {
            'anomaly_indices': anomalies,
            'method': 'isolation_forest'
        }
    
    def _find_consensus_anomalies(self, method_results: Dict[str, Dict]) -> List[int]:
        """Find anomalies detected by multiple methods"""
        
        # TODO: Implement consensus anomaly detection
        # Find indices that appear in multiple method results
        
        all_anomalies = []
        for method, results in method_results.items():
            all_anomalies.extend(results.get('anomaly_indices', []))
        
        # TODO: Find indices that appear in at least 2 methods
        from collections import Counter
        anomaly_counts = Counter(all_anomalies)
        consensus_anomalies = [idx for idx, count in anomaly_counts.items() if count >= 2]
        
        return consensus_anomalies

# =============================================================================
# INTELLIGENT ALERTING SYSTEM
# =============================================================================

class IntelligentAlertingSystem:
    """Advanced alerting system with context awareness and noise reduction"""
    
    def __init__(self):
        self.alert_history = []
        self.suppression_rules = []
        self.escalation_policies = {}
        
    def evaluate_alert(self, metric_name: str, current_value: float, threshold: float, 
                      context: Optional[Dict] = None) -> Dict[str, Any]:
        """Evaluate whether to send an alert with intelligent filtering"""
        
        # TODO: Implement intelligent alert evaluation
        # Apply filtering, check for alert fatigue, enrich with context
        
        alert_candidate = {
            'metric_name': metric_name,
            'current_value': current_value,
            'threshold': threshold,
            'context': context or {},
            'timestamp': datetime.now(),
            'severity': self._calculate_severity(current_value, threshold, context)
        }
        
        # TODO: Apply intelligent filtering
        if self._should_suppress_alert(alert_candidate):
            return {'action': 'suppressed', 'reason': 'intelligent_filtering'}
        
        # TODO: Check for alert fatigue
        if self._is_alert_fatigue(metric_name):
            return {'action': 'suppressed', 'reason': 'alert_fatigue'}
        
        # TODO: Enrich alert with context
        enriched_alert = self._enrich_alert_context(alert_candidate)
        
        # TODO: Send alert through appropriate channels
        self._send_alert(enriched_alert)
        
        # Track alert history
        self.alert_history.append(enriched_alert)
        
        return {'action': 'sent', 'alert': enriched_alert}
    
    def _calculate_severity(self, current_value: float, threshold: float, 
                          context: Optional[Dict]) -> str:
        """Calculate alert severity based on deviation and business context"""
        
        # TODO: Implement severity calculation
        # Consider deviation magnitude and business context
        
        deviation = abs(current_value - threshold) / threshold
        
        # TODO: Calculate base severity from deviation
        if deviation >= 0.5:  # 50% deviation
            base_severity = 'critical'
        elif deviation >= 0.2:  # 20% deviation
            base_severity = 'high'
        elif deviation >= 0.1:  # 10% deviation
            base_severity = 'medium'
        else:
            base_severity = 'low'
        
        # TODO: Adjust based on business context
        business_impact = context.get('business_impact', 'medium') if context else 'medium'
        
        return base_severity
    
    def _should_suppress_alert(self, alert_candidate: Dict[str, Any]) -> bool:
        """Determine if alert should be suppressed"""
        
        # TODO: Implement suppression logic
        # Check suppression rules and recent alert history
        
        # Check for duplicate recent alerts
        recent_alerts = [a for a in self.alert_history 
                        if (datetime.now() - a['timestamp']).seconds < 3600]  # Last hour
        
        similar_alerts = [a for a in recent_alerts 
                         if a['metric_name'] == alert_candidate['metric_name']]
        
        # TODO: Suppress if too many similar alerts recently
        if len(similar_alerts) >= 3:  # More than 3 similar alerts in last hour
            return True
        
        return False
    
    def _is_alert_fatigue(self, metric_name: str) -> bool:
        """Check for alert fatigue conditions"""
        
        # TODO: Implement alert fatigue detection
        # Check recent alert frequency for this metric
        
        recent_alerts = [a for a in self.alert_history 
                        if (datetime.now() - a['timestamp']).seconds < 3600
                        and a['metric_name'] == metric_name]
        
        return len(recent_alerts) >= 5  # More than 5 alerts in last hour
    
    def _enrich_alert_context(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich alert with additional context and suggested actions"""
        
        # TODO: Implement alert enrichment
        # Add runbook links, related metrics, suggested actions, impact assessment
        
        enriched = alert.copy()
        
        # TODO: Add runbook links
        enriched['runbook_url'] = f"https://runbooks.observetech.com/{alert['metric_name']}"
        
        # TODO: Add suggested actions based on alert type
        enriched['suggested_actions'] = self._get_suggested_actions(alert)
        
        # TODO: Add impact assessment
        enriched['impact_assessment'] = self._assess_impact(alert)
        
        return enriched
    
    def _get_suggested_actions(self, alert: Dict[str, Any]) -> List[str]:
        """Generate suggested actions based on alert type"""
        
        # TODO: Implement action suggestions based on metric type
        
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
        # TODO: Add more metric-specific actions
        
        return actions
    
    def _assess_impact(self, alert: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the business impact of the alert"""
        
        # TODO: Implement impact assessment
        # Consider downstream consumers, business criticality, SLA impact
        
        return {
            'business_impact': 'medium',  # TODO: Calculate based on context
            'affected_systems': [],  # TODO: Identify affected downstream systems
            'sla_risk': 'low'  # TODO: Assess SLA violation risk
        }
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through appropriate channels"""
        
        # TODO: Implement alert sending
        # Route to appropriate channels based on severity
        
        print(f"ALERT: {alert['severity'].upper()} - {alert['metric_name']}")
        print(f"Value: {alert['current_value']}, Threshold: {alert['threshold']}")

# =============================================================================
# DASHBOARD GENERATOR
# =============================================================================

class ObservabilityDashboardGenerator:
    """Generate comprehensive observability dashboards"""
    
    def __init__(self):
        self.dashboard_configs = {}
        
    def create_observability_dashboard(self) -> Dict[str, Any]:
        """Create comprehensive data observability dashboard"""
        
        # TODO: Implement dashboard configuration
        # Create panels for five pillars, anomalies, alerts, trends
        
        dashboard_config = {
            "dashboard": {
                "id": "observetech-data-observability",
                "title": "ObserveTech Data Observability Dashboard",
                "tags": ["observability", "data", "production"],
                "time": {"from": "now-24h", "to": "now"},
                "refresh": "30s",
                
                "panels": [
                    # TODO: Add overall health score panel
                    {
                        "id": 1,
                        "title": "Overall Data Health Score",
                        "type": "stat",
                        "targets": [
                            {
                                "expr": "avg(data_health_score)",
                                "legendFormat": "Health Score"
                            }
                        ]
                    },
                    
                    # TODO: Add five pillars status panel
                    {
                        "id": 2,
                        "title": "Five Pillars Status",
                        "type": "bargauge",
                        "targets": [
                            {"expr": "avg(freshness_score)", "legendFormat": "Freshness"},
                            {"expr": "avg(volume_score)", "legendFormat": "Volume"},
                            {"expr": "avg(schema_score)", "legendFormat": "Schema"},
                            {"expr": "avg(distribution_score)", "legendFormat": "Distribution"},
                            {"expr": "avg(lineage_score)", "legendFormat": "Lineage"}
                        ]
                    }
                    
                    # TODO: Add more panels for anomalies, trends, alerts
                ]
            }
        }
        
        return dashboard_config

# =============================================================================
# EXERCISE INSTRUCTIONS
# =============================================================================

def print_exercise_instructions():
    """Print detailed exercise instructions"""
    
    print("🎯 Data Observability Exercise - ObserveTech Solutions")
    print("=" * 60)
    
    print("\n📋 REQUIREMENTS:")
    print("1. Implement five pillars monitoring (freshness, volume, schema, distribution, lineage)")
    print("2. Build multi-method anomaly detection with statistical analysis")
    print("3. Create intelligent alerting with context awareness and noise reduction")
    print("4. Generate real-time dashboards with actionable insights")
    print("5. Implement automated incident response workflows")
    print("6. Build impact analysis and downstream effect assessment")
    
    print("\n🏗️ SYSTEM ARCHITECTURE:")
    print("""
    ObserveTech Data Observability System:
    
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Data Sources                                 │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │ Transaction │  │   Product   │  │    User     │             │
    │  │   Streams   │  │   Catalog   │  │  Behavior   │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │                Five Pillars Monitoring                          │
    │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │
    │  │Freshness│ │ Volume  │ │ Schema  │ │Distrib. │ │ Lineage │   │
    │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘   │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │              Anomaly Detection Engine                           │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │   Z-Score   │  │     IQR     │  │ Isolation   │             │
    │  │  Detection  │  │  Detection  │  │   Forest    │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │            Intelligent Alerting System                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │   Context   │  │    Noise    │  │ Escalation  │             │
    │  │ Enrichment  │  │ Reduction   │  │   Policies  │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
                                    │
    ┌─────────────────────────────────────────────────────────────────┐
    │              Observability Dashboards                          │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
    │  │  Real-time  │  │   Trend     │  │   Impact    │             │
    │  │  Metrics    │  │  Analysis   │  │  Analysis   │             │
    │  └─────────────┘  └─────────────┘  └─────────────┘             │
    └─────────────────────────────────────────────────────────────────┘
    """)
    
    print("\n🎯 SUCCESS CRITERIA:")
    print("• Comprehensive monitoring across all five observability pillars")
    print("• Accurate anomaly detection with minimal false positives")
    print("• Intelligent alerting that reduces noise and provides context")
    print("• Real-time dashboards with actionable insights")
    print("• Automated incident response and impact analysis")
    print("• Scalable architecture supporting high-volume data streams")
    
    print("\n🚀 GETTING STARTED:")
    print("1. Initialize the observability framework and monitoring systems")
    print("2. Implement five pillars monitoring with appropriate metrics")
    print("3. Build multi-method anomaly detection with statistical analysis")
    print("4. Create intelligent alerting with context and suppression")
    print("5. Generate comprehensive dashboards and visualizations")
    print("6. Test the system with sample data and validate accuracy")

if __name__ == "__main__":
    print_exercise_instructions()
    
    print("\n" + "="*60)
    print("🎯 Ready to build comprehensive data observability!")
    print("Complete the TODOs above to create a production-ready system.")
    print("="*60)
