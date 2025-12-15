#!/usr/bin/env python3
"""
Day 20: Data Observability - Interactive Demo

This script demonstrates the complete data observability system with real database connections
and monitoring capabilities.
"""

import os
import sys
import time
import psycopg2
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solution import ObserveTechDataObservabilityFramework

class ObservabilityDemo:
    """Interactive demonstration of the data observability system"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DB_HOST', 'localhost'),
            'port': os.getenv('DB_PORT', '5432'),
            'database': os.getenv('DB_NAME', 'observability_db'),
            'user': os.getenv('DB_USER', 'obs_user'),
            'password': os.getenv('DB_PASSWORD', 'obs_password')
        }
        self.framework = None
        self.connection = None
    
    def connect_database(self) -> bool:
        """Connect to the PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            print("✅ Connected to PostgreSQL database")
            return True
        except Exception as e:
            print(f"❌ Failed to connect to database: {e}")
            print("💡 Make sure Docker containers are running: docker-compose up -d")
            return False
    
    def initialize_framework(self):
        """Initialize the observability framework"""
        try:
            self.framework = ObserveTechDataObservabilityFramework()
            self.framework.initialize_monitoring_systems()
            print("✅ Observability framework initialized")
        except Exception as e:
            print(f"❌ Failed to initialize framework: {e}")
    
    def check_data_availability(self) -> Dict[str, int]:
        """Check available data in the database"""
        try:
            cursor = self.connection.cursor()
            
            # Check customer transactions
            cursor.execute("SELECT COUNT(*) FROM data_sources.customer_transactions")
            transaction_count = cursor.fetchone()[0]
            
            # Check monitoring metrics
            cursor.execute("SELECT COUNT(*) FROM monitoring.data_quality_metrics")
            metrics_count = cursor.fetchone()[0]
            
            # Check alert history
            cursor.execute("SELECT COUNT(*) FROM alerts.alert_history")
            alerts_count = cursor.fetchone()[0]
            
            cursor.close()
            
            data_summary = {
                'transactions': transaction_count,
                'metrics': metrics_count,
                'alerts': alerts_count
            }
            
            print(f"📊 Data Summary:")
            print(f"   • Customer Transactions: {transaction_count:,}")
            print(f"   • Quality Metrics: {metrics_count}")
            print(f"   • Alert History: {alerts_count}")
            
            return data_summary
            
        except Exception as e:
            print(f"❌ Error checking data: {e}")
            return {}
    
    def run_monitoring_demo(self):
        """Run comprehensive monitoring demonstration"""
        print("\n🔍 Running Comprehensive Data Monitoring...")
        print("=" * 60)
        
        # Configuration for monitoring
        config = {
            'expected_frequency_minutes': 60,
            'lookback_days': 7,
            'numeric_columns': ['amount', 'quantity', 'price'],
            'categorical_columns': ['status', 'category', 'region']
        }
        
        # Run monitoring for customer transactions
        results = self.framework.run_comprehensive_monitoring('customer_transactions', config)
        
        # Display results
        print(f"\n📈 MONITORING RESULTS for 'customer_transactions':")
        print(f"   • Overall Health Score: {results['overall_health_score']:.3f}")
        print(f"   • Timestamp: {results['timestamp']}")
        
        print(f"\n🏛️ FIVE PILLARS STATUS:")
        for pillar, result in results['pillar_results'].items():
            if pillar == 'freshness':
                score = result.get('freshness_score', 0)
                status = "✅ Fresh" if result.get('is_fresh', False) else "⚠️ Stale"
                print(f"   • Freshness: {score:.3f} {status}")
            elif pillar == 'volume':
                anomaly = result.get('is_anomaly', False)
                status = "⚠️ Anomaly" if anomaly else "✅ Normal"
                print(f"   • Volume: {status}")
            elif pillar == 'schema':
                changes = result.get('change_count', 0)
                status = f"⚠️ {changes} changes" if changes > 0 else "✅ Stable"
                print(f"   • Schema: {status}")
            elif pillar == 'distribution':
                health = result.get('distribution_health_score', 1.0)
                status = "✅ Healthy" if health > 0.8 else "⚠️ Issues"
                print(f"   • Distribution: {health:.3f} {status}")
            elif pillar == 'lineage':
                impact = result.get('impact_score', 0)
                print(f"   • Lineage: Impact Score {impact:.1f}")
        
        # Display anomalies
        if results['anomalies_detected']:
            print(f"\n🚨 ANOMALIES DETECTED ({len(results['anomalies_detected'])}):")
            for anomaly in results['anomalies_detected']:
                severity_emoji = {'low': '🟡', 'medium': '🟠', 'high': '🔴', 'critical': '🚨'}
                emoji = severity_emoji.get(anomaly['severity'], '🔵')
                print(f"   {emoji} {anomaly['type']}: {anomaly['description']}")
        else:
            print(f"\n✅ NO ANOMALIES DETECTED")
        
        # Display alerts
        if results['alerts_triggered']:
            print(f"\n📢 ALERTS TRIGGERED ({len(results['alerts_triggered'])}):")
            for alert in results['alerts_triggered']:
                severity_emoji = {'low': '🟡', 'medium': '🟠', 'high': '🔴', 'critical': '🚨'}
                emoji = severity_emoji.get(alert['severity'], '🔵')
                print(f"   {emoji} {alert['metric_name']}: {alert.get('description', 'Alert triggered')}")
        else:
            print(f"\n✅ NO ALERTS TRIGGERED")
        
        # Display recommendations
        if results['recommendations']:
            print(f"\n💡 RECOMMENDATIONS ({len(results['recommendations'])}):")
            for rec in results['recommendations']:
                priority_emoji = {'low': '🟢', 'medium': '🟡', 'high': '🔴'}
                emoji = priority_emoji.get(rec['priority'], '🔵')
                print(f"   {emoji} {rec['type']}: {rec['description']}")
        
        return results
    
    def demonstrate_anomaly_detection(self):
        """Demonstrate anomaly detection capabilities"""
        print("\n🎯 Anomaly Detection Demonstration")
        print("=" * 50)
        
        # Generate sample data with anomalies
        normal_data = [100, 102, 98, 105, 99, 101, 103, 97, 104, 100]
        anomaly_data = normal_data + [150, 45, 200]  # Add clear anomalies
        
        detector = self.framework.anomaly_detectors['statistical']
        
        # Test different detection methods
        methods = ['z_score', 'iqr', 'seasonal_hybrid']
        
        for method in methods:
            print(f"\n🔍 Testing {method.upper()} method:")
            result = detector.detect_anomalies(anomaly_data, methods=[method])
            
            anomalies = result['detected_anomalies']
            if anomalies:
                print(f"   • Anomalies found at indices: {anomalies}")
                print(f"   • Anomalous values: {[anomaly_data[i] for i in anomalies]}")
            else:
                print(f"   • No anomalies detected")
        
        # Test consensus detection
        print(f"\n🤝 Testing CONSENSUS detection:")
        consensus_result = detector.detect_anomalies(anomaly_data, methods=methods)
        consensus_anomalies = consensus_result['consensus_anomalies']
        
        if consensus_anomalies:
            print(f"   • Consensus anomalies at indices: {consensus_anomalies}")
            print(f"   • High-confidence anomalous values: {[anomaly_data[i] for i in consensus_anomalies]}")
        else:
            print(f"   • No consensus anomalies found")
    
    def demonstrate_alerting_system(self):
        """Demonstrate intelligent alerting capabilities"""
        print("\n📢 Intelligent Alerting Demonstration")
        print("=" * 50)
        
        alerting_system = self.framework.alerting_system
        
        # Test different alert scenarios
        test_alerts = [
            {
                'metric_name': 'data_freshness',
                'current_value': 0.65,
                'threshold': 0.90,
                'context': {'business_impact': 'high', 'table_name': 'customer_transactions'}
            },
            {
                'metric_name': 'data_volume',
                'current_value': 50000,
                'threshold': 100000,
                'context': {'business_impact': 'critical', 'time_sensitivity': 'urgent'}
            },
            {
                'metric_name': 'overall_health_score',
                'current_value': 0.75,
                'threshold': 0.90,
                'context': {'business_impact': 'medium'}
            }
        ]
        
        for i, alert_data in enumerate(test_alerts, 1):
            print(f"\n🚨 Test Alert #{i}: {alert_data['metric_name']}")
            
            result = alerting_system.evaluate_alert(
                alert_data['metric_name'],
                alert_data['current_value'],
                alert_data['threshold'],
                alert_data['context']
            )
            
            if result['action'] == 'sent':
                alert = result['alert']
                print(f"   ✅ Alert SENT - Severity: {alert['severity']}")
                print(f"   📋 Suggested Actions: {len(alert.get('suggested_actions', []))}")
                if alert.get('suggested_actions'):
                    for action in alert['suggested_actions'][:2]:  # Show first 2 actions
                        print(f"      • {action}")
            else:
                print(f"   🚫 Alert SUPPRESSED - Reason: {result['reason']}")
    
    def show_dashboard_preview(self):
        """Show dashboard configuration preview"""
        print("\n📊 Dashboard Configuration Preview")
        print("=" * 50)
        
        dashboard_generator = self.framework.dashboard_generator
        
        # Generate main dashboard config
        main_dashboard = dashboard_generator.create_observability_dashboard()
        
        print(f"📈 Main Observability Dashboard:")
        print(f"   • Title: {main_dashboard['dashboard']['title']}")
        print(f"   • Panels: {len(main_dashboard['dashboard']['panels'])}")
        print(f"   • Refresh Rate: {main_dashboard['dashboard']['refresh']}")
        
        # Show panel summary
        print(f"\n🎛️ Dashboard Panels:")
        for panel in main_dashboard['dashboard']['panels']:
            print(f"   • {panel['title']} ({panel['type']})")
        
        # Generate executive dashboard
        exec_dashboard = dashboard_generator.create_executive_dashboard()
        
        print(f"\n👔 Executive Dashboard:")
        print(f"   • Title: {exec_dashboard['dashboard']['title']}")
        print(f"   • Panels: {len(exec_dashboard['dashboard']['panels'])}")
        
        print(f"\n💡 Access dashboards at: http://localhost:3000 (admin/admin)")
    
    def run_complete_demo(self):
        """Run the complete demonstration"""
        print("🚀 ObserveTech Data Observability System - Live Demo")
        print("=" * 70)
        
        # Step 1: Connect to database
        if not self.connect_database():
            return False
        
        # Step 2: Initialize framework
        self.initialize_framework()
        
        # Step 3: Check data availability
        data_summary = self.check_data_availability()
        
        if data_summary.get('transactions', 0) == 0:
            print("\n⚠️ No sample data found. Please run: docker-compose up -d")
            return False
        
        # Step 4: Run monitoring demo
        monitoring_results = self.run_monitoring_demo()
        
        # Step 5: Demonstrate anomaly detection
        self.demonstrate_anomaly_detection()
        
        # Step 6: Demonstrate alerting system
        self.demonstrate_alerting_system()
        
        # Step 7: Show dashboard preview
        self.show_dashboard_preview()
        
        # Summary
        print(f"\n🎉 DEMO COMPLETE!")
        print("=" * 70)
        print("✅ Five pillars monitoring demonstrated")
        print("✅ Anomaly detection with multiple methods tested")
        print("✅ Intelligent alerting system showcased")
        print("✅ Dashboard configurations generated")
        print("\n🔗 Next Steps:")
        print("   • Access Grafana dashboards: http://localhost:3000")
        print("   • View Prometheus metrics: http://localhost:9090")
        print("   • Explore Jupyter notebooks: http://localhost:8888")
        print("   • Check database: psql -h localhost -U obs_user observability_db")
        
        return True
    
    def cleanup(self):
        """Clean up resources"""
        if self.connection:
            self.connection.close()
            print("🔌 Database connection closed")

def main():
    """Main execution function"""
    demo = ObservabilityDemo()
    
    try:
        success = demo.run_complete_demo()
        if not success:
            print("\n❌ Demo failed. Please check the setup and try again.")
            return 1
    except KeyboardInterrupt:
        print("\n\n⏹️ Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        return 1
    finally:
        demo.cleanup()
    
    return 0

if __name__ == "__main__":
    exit(main())