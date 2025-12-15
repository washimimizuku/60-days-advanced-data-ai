#!/usr/bin/env python3
"""
Day 25: Feature Stores - Interactive Demo

This demo showcases the complete feature store workflow:
1. Environment setup and data generation
2. Feature store initialization and configuration
3. Feature materialization and serving
4. Real-time inference and monitoring
5. Feature quality validation and drift detection
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any
import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.live import Live
from rich.layout import Layout

console = Console()

class FeatureStoreDemo:
    """Interactive demo for RideShare Feature Store"""
    
    def __init__(self):
        self.api_base_url = "http://localhost:8000"
        self.console = Console()
        
    def print_header(self, title: str):
        """Print formatted header"""
        self.console.print(Panel(title, style="bold blue"))
    
    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"✅ {message}", style="green")
    
    def print_error(self, message: str):
        """Print error message"""
        self.console.print(f"❌ {message}", style="red")
    
    def print_info(self, message: str):
        """Print info message"""
        self.console.print(f"ℹ️  {message}", style="blue")
    
    def check_services(self) -> bool:
        """Check if all services are running"""
        
        self.print_header("🔍 Checking Service Health")
        
        services = {
            "Feature Serving API": f"{self.api_base_url}/health",
            "PostgreSQL": "localhost:5432",
            "Redis": "localhost:6379",
            "Grafana": "http://localhost:3000",
            "Prometheus": "http://localhost:9090"
        }
        
        all_healthy = True
        
        for service, endpoint in services.items():
            try:
                if endpoint.startswith("http"):
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code == 200:
                        self.print_success(f"{service} is healthy")
                    else:
                        self.print_error(f"{service} returned status {response.status_code}")
                        all_healthy = False
                else:
                    # For non-HTTP services, just mark as assumed healthy
                    self.print_success(f"{service} is assumed healthy")
            except Exception as e:
                self.print_error(f"{service} is not accessible: {str(e)}")
                all_healthy = False
        
        return all_healthy
    
    def demonstrate_feature_serving(self):
        """Demonstrate real-time feature serving"""
        
        self.print_header("🚀 Real-time Feature Serving Demo")
        
        # Sample entity rows for different use cases
        test_cases = [
            {
                "name": "Driver Performance Lookup",
                "entity_rows": [
                    {"driver_id": 1001},
                    {"driver_id": 1002},
                    {"driver_id": 1003}
                ],
                "features": [
                    "driver_performance_features:acceptance_rate_7d",
                    "driver_performance_features:avg_rating_7d",
                    "driver_performance_features:total_rides_7d"
                ]
            },
            {
                "name": "User Behavior Analysis",
                "entity_rows": [
                    {"user_id": 2001},
                    {"user_id": 2002}
                ],
                "features": [
                    "user_behavior_features:loyalty_score",
                    "user_behavior_features:ride_frequency_30d",
                    "user_behavior_features:avg_trip_distance_30d"
                ]
            },
            {
                "name": "Location Demand Forecasting",
                "entity_rows": [
                    {"location_id": "zone_1"},
                    {"location_id": "zone_2"}
                ],
                "features": [
                    "location_demand_features:current_demand_score",
                    "location_demand_features:supply_ratio",
                    "location_demand_features:surge_multiplier"
                ]
            }
        ]
        
        for test_case in test_cases:
            self.console.print(f"\n🔍 Testing: {test_case['name']}")
            
            try:
                # Make API request
                start_time = time.time()
                
                response = requests.post(
                    f"{self.api_base_url}/features",
                    json={
                        "entity_rows": test_case["entity_rows"],
                        "features": test_case["features"]
                    },
                    timeout=10
                )
                
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    result = response.json()
                    
                    self.print_success(f"Features served in {latency:.2f}ms")
                    
                    # Display results in a table
                    table = Table(title=f"{test_case['name']} Results")
                    table.add_column("Entity", style="cyan")
                    table.add_column("Feature", style="magenta")
                    table.add_column("Value", style="green")
                    
                    # Parse and display features
                    features = result.get("features", {})
                    for i, entity_row in enumerate(test_case["entity_rows"]):
                        entity_key = list(entity_row.keys())[0]
                        entity_value = entity_row[entity_key]
                        
                        for feature in test_case["features"]:
                            feature_name = feature.split(":")[-1]
                            feature_value = features.get(feature_name, ["N/A"])[i] if isinstance(features.get(feature_name), list) else "N/A"
                            
                            table.add_row(
                                f"{entity_key}={entity_value}",
                                feature_name,
                                str(feature_value)
                            )
                    
                    self.console.print(table)
                    
                    # Check latency requirement
                    if latency < 10:
                        self.print_success("✅ Latency requirement met (<10ms)")
                    else:
                        self.print_error(f"⚠️ Latency requirement not met: {latency:.2f}ms")
                
                else:
                    self.print_error(f"API request failed: {response.status_code}")
                    self.console.print(response.text)
            
            except Exception as e:
                self.print_error(f"Feature serving failed: {str(e)}")
    
    def demonstrate_feature_metadata(self):
        """Demonstrate feature metadata retrieval"""
        
        self.print_header("📋 Feature Metadata Demo")
        
        feature_views = [
            "driver_performance_features",
            "user_behavior_features", 
            "location_demand_features"
        ]
        
        for fv_name in feature_views:
            try:
                response = requests.get(f"{self.api_base_url}/features/metadata/{fv_name}")
                
                if response.status_code == 200:
                    metadata = response.json()
                    
                    self.console.print(f"\n📊 {fv_name}")
                    
                    # Create metadata table
                    table = Table()
                    table.add_column("Property", style="cyan")
                    table.add_column("Value", style="green")
                    
                    table.add_row("Feature View", metadata.get("feature_view", "N/A"))
                    table.add_row("Entities", ", ".join(metadata.get("entities", [])))
                    table.add_row("TTL", metadata.get("ttl", "N/A"))
                    table.add_row("Online", str(metadata.get("online", False)))
                    
                    # Add features
                    features = metadata.get("features", [])
                    feature_list = [f"{f['name']} ({f['type']})" for f in features]
                    table.add_row("Features", "\n".join(feature_list))
                    
                    # Add tags
                    tags = metadata.get("tags", {})
                    tag_list = [f"{k}: {v}" for k, v in tags.items()]
                    table.add_row("Tags", "\n".join(tag_list))
                    
                    self.console.print(table)
                    
                else:
                    self.print_error(f"Failed to get metadata for {fv_name}: {response.status_code}")
            
            except Exception as e:
                self.print_error(f"Metadata retrieval failed for {fv_name}: {str(e)}")
    
    def demonstrate_feature_monitoring(self):
        """Demonstrate feature quality validation and drift detection"""
        
        self.print_header("🔍 Feature Monitoring Demo")
        
        feature_views = ["driver_performance_features", "user_behavior_features"]
        
        # Feature Quality Validation
        self.console.print("\n📊 Feature Quality Validation")
        
        for fv_name in feature_views:
            try:
                response = requests.post(f"{self.api_base_url}/features/validate/{fv_name}")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    quality_score = result.get("quality_score", 0.0)
                    checks_passed = result.get("checks_passed", 0)
                    total_checks = result.get("total_checks", 0)
                    issues = result.get("issues", [])
                    
                    # Create quality table
                    table = Table(title=f"Quality Report: {fv_name}")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")
                    
                    table.add_row("Quality Score", f"{quality_score:.3f}")
                    table.add_row("Checks Passed", f"{checks_passed}/{total_checks}")
                    table.add_row("Issues", str(len(issues)))
                    
                    if issues:
                        table.add_row("Issue Details", "\n".join(issues))
                    
                    self.console.print(table)
                    
                    # Quality assessment
                    if quality_score >= 0.95:
                        self.print_success(f"Excellent quality: {quality_score:.3f}")
                    elif quality_score >= 0.8:
                        self.print_info(f"Good quality: {quality_score:.3f}")
                    else:
                        self.print_error(f"Poor quality: {quality_score:.3f}")
                
                else:
                    self.print_error(f"Quality validation failed for {fv_name}: {response.status_code}")
            
            except Exception as e:
                self.print_error(f"Quality validation error for {fv_name}: {str(e)}")
        
        # Feature Drift Detection
        self.console.print("\n🔄 Feature Drift Detection")
        
        features_to_check = [
            "driver_performance_features:acceptance_rate_7d",
            "driver_performance_features:avg_rating_7d",
            "user_behavior_features:loyalty_score"
        ]
        
        for feature_name in features_to_check:
            try:
                response = requests.post(
                    f"{self.api_base_url}/features/drift/{feature_name}",
                    params={"reference_hours": 168, "current_hours": 24}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    drift_score = result.get("drift_score", 0.0)
                    drift_detected = result.get("drift_detected", False)
                    p_value = result.get("p_value", 1.0)
                    
                    # Create drift table
                    table = Table(title=f"Drift Analysis: {feature_name}")
                    table.add_column("Metric", style="cyan")
                    table.add_column("Value", style="green")
                    
                    table.add_row("Drift Score", f"{drift_score:.4f}")
                    table.add_row("P-Value", f"{p_value:.4f}")
                    table.add_row("Drift Detected", "Yes" if drift_detected else "No")
                    table.add_row("Statistical Test", result.get("statistical_test", "N/A"))
                    
                    self.console.print(table)
                    
                    # Drift assessment
                    if drift_detected:
                        self.print_error(f"⚠️ Drift detected: {drift_score:.4f}")
                    else:
                        self.print_success(f"✅ No drift detected: {drift_score:.4f}")
                
                else:
                    self.print_error(f"Drift detection failed for {feature_name}: {response.status_code}")
            
            except Exception as e:
                self.print_error(f"Drift detection error for {feature_name}: {str(e)}")
    
    def demonstrate_performance_benchmarks(self):
        """Demonstrate performance benchmarks and load testing"""
        
        self.print_header("⚡ Performance Benchmarks")
        
        # Test different batch sizes
        batch_sizes = [1, 10, 50, 100]
        feature_refs = [
            "driver_performance_features:acceptance_rate_7d",
            "driver_performance_features:avg_rating_7d"
        ]
        
        results = []
        
        for batch_size in batch_sizes:
            self.console.print(f"\n🔍 Testing batch size: {batch_size}")
            
            # Generate entity rows
            entity_rows = [{"driver_id": 1000 + i} for i in range(batch_size)]
            
            # Measure latency over multiple requests
            latencies = []
            
            for _ in range(5):  # 5 requests per batch size
                try:
                    start_time = time.time()
                    
                    response = requests.post(
                        f"{self.api_base_url}/features",
                        json={
                            "entity_rows": entity_rows,
                            "features": feature_refs
                        },
                        timeout=10
                    )
                    
                    latency = (time.time() - start_time) * 1000
                    
                    if response.status_code == 200:
                        latencies.append(latency)
                    
                except Exception as e:
                    self.print_error(f"Request failed: {str(e)}")
            
            if latencies:
                avg_latency = np.mean(latencies)
                p95_latency = np.percentile(latencies, 95)
                throughput = batch_size / (avg_latency / 1000)  # entities per second
                
                results.append({
                    "batch_size": batch_size,
                    "avg_latency": avg_latency,
                    "p95_latency": p95_latency,
                    "throughput": throughput
                })
                
                self.print_success(f"Avg: {avg_latency:.2f}ms, P95: {p95_latency:.2f}ms, Throughput: {throughput:.1f} entities/sec")
        
        # Display performance summary
        if results:
            table = Table(title="Performance Summary")
            table.add_column("Batch Size", style="cyan")
            table.add_column("Avg Latency (ms)", style="green")
            table.add_column("P95 Latency (ms)", style="yellow")
            table.add_column("Throughput (entities/sec)", style="magenta")
            
            for result in results:
                table.add_row(
                    str(result["batch_size"]),
                    f"{result['avg_latency']:.2f}",
                    f"{result['p95_latency']:.2f}",
                    f"{result['throughput']:.1f}"
                )
            
            self.console.print(table)
            
            # Performance assessment
            best_latency = min(r["avg_latency"] for r in results if r["batch_size"] == 1)
            if best_latency < 10:
                self.print_success(f"✅ Latency requirement met: {best_latency:.2f}ms < 10ms")
            else:
                self.print_error(f"❌ Latency requirement not met: {best_latency:.2f}ms >= 10ms")
    
    def show_monitoring_dashboard_info(self):
        """Show information about monitoring dashboards"""
        
        self.print_header("📊 Monitoring Dashboard Information")
        
        dashboard_info = [
            {
                "Service": "Grafana Dashboard",
                "URL": "http://localhost:3000",
                "Credentials": "admin / admin123",
                "Description": "Feature store metrics, performance monitoring, and alerting"
            },
            {
                "Service": "Prometheus Metrics",
                "URL": "http://localhost:9090",
                "Credentials": "None required",
                "Description": "Raw metrics collection and querying interface"
            },
            {
                "Service": "Jupyter Lab",
                "URL": "http://localhost:8888",
                "Credentials": "None required",
                "Description": "Interactive feature store development and analysis"
            }
        ]
        
        for info in dashboard_info:
            table = Table(title=info["Service"])
            table.add_column("Property", style="cyan")
            table.add_column("Value", style="green")
            
            for key, value in info.items():
                if key != "Service":
                    table.add_row(key, value)
            
            self.console.print(table)
            self.console.print()
    
    def run_complete_demo(self):
        """Run the complete feature store demo"""
        
        self.console.print(Panel("🚀 RideShare Feature Store - Complete Demo", style="bold blue"))
        
        # Check services
        if not self.check_services():
            self.print_error("Some services are not healthy. Please run 'docker-compose up -d' first.")
            return
        
        # Wait for user input between sections
        input("\nPress Enter to continue to Feature Serving Demo...")
        self.demonstrate_feature_serving()
        
        input("\nPress Enter to continue to Feature Metadata Demo...")
        self.demonstrate_feature_metadata()
        
        input("\nPress Enter to continue to Feature Monitoring Demo...")
        self.demonstrate_feature_monitoring()
        
        input("\nPress Enter to continue to Performance Benchmarks...")
        self.demonstrate_performance_benchmarks()
        
        input("\nPress Enter to see Monitoring Dashboard Information...")
        self.show_monitoring_dashboard_info()
        
        # Final summary
        self.print_header("🎉 Demo Complete!")
        
        self.console.print("""
✅ Feature Store Demo Summary:
   • Real-time feature serving with sub-10ms latency
   • Comprehensive feature metadata and governance
   • Feature quality validation and drift detection
   • Performance benchmarking and load testing
   • Production monitoring and observability

🚀 Next Steps:
   1. Explore Grafana dashboards for detailed metrics
   2. Try the Jupyter notebooks for interactive development
   3. Complete the exercises in exercise.py
   4. Build your own feature definitions and transformations

📚 Resources:
   • README.md - Complete documentation
   • solution.py - Full implementation reference
   • notebooks/ - Interactive examples and tutorials
        """)

if __name__ == "__main__":
    demo = FeatureStoreDemo()
    demo.run_complete_demo()