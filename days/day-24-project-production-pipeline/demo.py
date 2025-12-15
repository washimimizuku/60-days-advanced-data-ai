#!/usr/bin/env python3
"""
Day 24: Production Pipeline - Interactive Demo
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class ProductionPipelineDemo:
    def __init__(self):
        self.airflow_url = "http://localhost:8080"
        self.grafana_url = "http://localhost:3000"
        self.prometheus_url = "http://localhost:9090"
    
    def check_services(self):
        """Check if all services are running"""
        services = {
            'Airflow': self.airflow_url,
            'Grafana': self.grafana_url,
            'Prometheus': self.prometheus_url
        }
        
        print("🔍 Checking service health...")
        for name, url in services.items():
            try:
                response = requests.get(f"{url}/health", timeout=5)
                status = "✅ Running" if response.status_code == 200 else "❌ Error"
            except:
                status = "❌ Not accessible"
            print(f"   {name}: {status}")
    
    def trigger_pipeline(self):
        """Trigger the production pipeline"""
        print("🚀 Triggering production pipeline...")
        
        # Simulate pipeline trigger
        dag_id = "production_data_pipeline"
        
        try:
            # In a real setup, this would trigger via Airflow API
            print(f"   📊 DAG ID: {dag_id}")
            print(f"   ⏰ Triggered at: {datetime.now()}")
            print("   🔄 Pipeline execution started...")
            
            # Simulate processing time
            time.sleep(3)
            
            print("   ✅ Pipeline completed successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Pipeline failed: {e}")
            return False
    
    def show_metrics(self):
        """Display pipeline metrics"""
        print("📊 Pipeline Metrics:")
        
        # Simulate metrics
        metrics = {
            'Records Processed': '10,000',
            'Data Quality Score': '0.98',
            'Processing Time': '45 seconds',
            'Success Rate': '99.5%'
        }
        
        for metric, value in metrics.items():
            print(f"   • {metric}: {value}")
    
    def run_demo(self):
        """Run interactive demo"""
        print("🚀 Production Data Pipeline Demo")
        print("=" * 50)
        
        while True:
            print("\nChoose an option:")
            print("1. Check service health")
            print("2. Trigger pipeline")
            print("3. View metrics")
            print("4. Open Airflow UI")
            print("5. Open Grafana")
            print("6. Exit")
            
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                self.check_services()
            elif choice == '2':
                self.trigger_pipeline()
            elif choice == '3':
                self.show_metrics()
            elif choice == '4':
                print(f"🌐 Open Airflow UI: {self.airflow_url}")
                print("   Username: admin, Password: admin")
            elif choice == '5':
                print(f"📊 Open Grafana: {self.grafana_url}")
                print("   Username: admin, Password: admin")
            elif choice == '6':
                print("👋 Demo completed!")
                break
            else:
                print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    demo = ProductionPipelineDemo()
    demo.run_demo()