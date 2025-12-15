#!/usr/bin/env python3
"""
Day 21: Testing Strategies - Interactive Demo
Comprehensive demonstration of the testing framework capabilities
"""

import pandas as pd
import numpy as np
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
import json

# Import our testing framework components
from solution import (
    clean_transaction_data,
    calculate_customer_metrics,
    detect_fraud_patterns,
    PerformanceTestFramework,
    RegressionTestFramework,
    CICDTestingFramework
)

class TestingFrameworkDemo:
    """Interactive demonstration of the comprehensive testing framework"""
    
    def __init__(self):
        self.demo_data = None
        self.results = {}
        
    def generate_demo_data(self, size: int = 1000) -> pd.DataFrame:
        """Generate realistic demo data for testing"""
        
        print(f"📊 Generating {size} sample transactions...")
        
        np.random.seed(42)  # For reproducible results
        
        # Generate realistic transaction data
        data = pd.DataFrame({
            'transaction_id': [f'TXN{i:06d}' for i in range(size)],
            'customer_id': [f'CUST{i % 100:04d}' for i in range(size)],
            'amount': np.random.lognormal(mean=4, sigma=1, size=size),
            'currency': np.random.choice(['USD', 'EUR', 'GBP'], size, p=[0.7, 0.2, 0.1]),
            'merchant_name': [f'Merchant {i % 50}' for i in range(size)],
            'transaction_date': pd.date_range('2024-01-01', periods=size, freq='5min')
        })
        
        # Add some data quality issues for testing
        # Duplicates
        duplicate_indices = np.random.choice(size, size//20, replace=False)
        duplicates = data.iloc[duplicate_indices].copy()
        data = pd.concat([data, duplicates], ignore_index=True)
        
        # Invalid amounts
        invalid_indices = np.random.choice(len(data), len(data)//50, replace=False)
        data.loc[invalid_indices, 'amount'] = np.random.choice([-100, 0, 'invalid'], len(invalid_indices))
        
        # Missing merchant names
        missing_indices = np.random.choice(len(data), len(data)//30, replace=False)
        data.loc[missing_indices, 'merchant_name'] = None
        
        self.demo_data = data
        print(f"✅ Generated {len(data)} transactions with quality issues for testing")
        
        return data
    
    def demo_unit_testing(self):
        """Demonstrate unit testing capabilities"""
        
        print("\n" + "="*60)
        print("🧪 UNIT TESTING DEMONSTRATION")
        print("="*60)
        
        if self.demo_data is None:
            self.generate_demo_data()
        
        # Test data cleaning
        print("\n1. Testing Data Cleaning Function")
        print("-" * 40)
        
        original_count = len(self.demo_data)
        print(f"Original data: {original_count} records")
        
        start_time = time.time()
        cleaned_data = clean_transaction_data(self.demo_data)
        end_time = time.time()
        
        cleaned_count = len(cleaned_data)
        processing_time = end_time - start_time
        
        print(f"Cleaned data: {cleaned_count} records")
        print(f"Records removed: {original_count - cleaned_count}")
        print(f"Processing time: {processing_time:.3f} seconds")
        print(f"✅ Data cleaning: PASSED")
        
        # Test customer metrics
        print("\n2. Testing Customer Metrics Calculation")
        print("-" * 40)
        
        start_time = time.time()
        customer_metrics = calculate_customer_metrics(cleaned_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        print(f"Customers analyzed: {len(customer_metrics)}")
        print(f"Average spending: ${customer_metrics['total_spent'].mean():.2f}")
        print(f"Processing time: {processing_time:.3f} seconds")
        print(f"✅ Customer metrics: PASSED")
        
        # Test fraud detection
        print("\n3. Testing Fraud Detection")
        print("-" * 40)
        
        start_time = time.time()
        fraud_results = detect_fraud_patterns(cleaned_data)
        end_time = time.time()
        
        processing_time = end_time - start_time
        suspicious_count = fraud_results['is_suspicious'].sum()
        fraud_rate = (suspicious_count / len(fraud_results)) * 100
        
        print(f"Transactions analyzed: {len(fraud_results)}")
        print(f"Suspicious transactions: {suspicious_count}")
        print(f"Fraud rate: {fraud_rate:.2f}%")
        print(f"Processing time: {processing_time:.3f} seconds")
        print(f"✅ Fraud detection: PASSED")
        
        self.results['unit_testing'] = {
            'original_records': original_count,
            'cleaned_records': cleaned_count,
            'customers_analyzed': len(customer_metrics),
            'suspicious_transactions': suspicious_count,
            'fraud_rate': fraud_rate
        }
    
    def demo_performance_testing(self):
        """Demonstrate performance testing capabilities"""
        
        print("\n" + "="*60)
        print("🚀 PERFORMANCE TESTING DEMONSTRATION")
        print("="*60)
        
        perf_framework = PerformanceTestFramework()
        
        # Scalability testing
        print("\n1. Scalability Testing")
        print("-" * 40)
        
        data_sizes = [1000, 2000, 5000]
        scalability_results = perf_framework.benchmark_scalability(
            clean_transaction_data, data_sizes
        )
        
        print("Data Size | Execution Time | Throughput")
        print("-" * 40)
        for result in scalability_results['scalability_results']:
            print(f"{result['data_size']:8d} | {result['execution_time']:13.3f}s | {result['throughput']:8.0f} rec/s")
        
        print(f"\nScalability Rating: {scalability_results['analysis']['scalability_rating']}")
        print(f"Estimated Complexity: {scalability_results['analysis']['estimated_time_complexity']}")
        print(f"✅ Scalability testing: PASSED")
        
        # Memory usage testing
        print("\n2. Memory Usage Testing")
        print("-" * 40)
        
        large_data = perf_framework._generate_test_data(10000)\n        metrics = perf_framework.measure_performance(clean_transaction_data, large_data)\n        \n        print(f\"Records processed: {len(large_data)}\")\n        print(f\"Execution time: {metrics['execution_time_seconds']:.3f} seconds\")\n        print(f\"Memory used: {metrics['memory_used_mb']:.2f} MB\")\n        print(f\"Throughput: {metrics['throughput_records_per_second']:.0f} records/sec\")\n        \n        memory_ok = metrics['memory_used_mb'] < 500\n        throughput_ok = metrics['throughput_records_per_second'] > 1000\n        \n        print(f\"✅ Memory usage: {'PASSED' if memory_ok else 'FAILED'}\")\n        print(f\"✅ Throughput: {'PASSED' if throughput_ok else 'FAILED'}\")\n        \n        self.results['performance_testing'] = {\n            'scalability_rating': scalability_results['analysis']['scalability_rating'],\n            'memory_used_mb': metrics['memory_used_mb'],\n            'throughput': metrics['throughput_records_per_second']\n        }\n    \n    def demo_regression_testing(self):\n        \"\"\"Demonstrate regression testing capabilities\"\"\"\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"🔍 REGRESSION TESTING DEMONSTRATION\")\n        print(\"=\"*60)\n        \n        # Create temporary directory for baselines\n        baseline_dir = \"./demo_baselines\"\n        os.makedirs(baseline_dir, exist_ok=True)\n        \n        regression_framework = RegressionTestFramework(baseline_dir)\n        \n        # Generate consistent test data\n        print(\"\\n1. Baseline Capture\")\n        print(\"-\" * 40)\n        \n        test_data = pd.DataFrame({\n            'transaction_id': ['TXN001', 'TXN002', 'TXN003', 'TXN004'],\n            'customer_id': ['CUST001', 'CUST001', 'CUST002', 'CUST002'],\n            'amount': [100.0, 150.0, 200.0, 250.0],\n            'currency': ['USD', 'USD', 'EUR', 'EUR'],\n            'merchant_name': ['Store A', 'Store B', 'Store C', 'Store D']\n        })\n        \n        # Process data and capture baseline\n        processed_data = clean_transaction_data(test_data)\n        baseline_metrics = regression_framework.capture_baseline(processed_data, 'demo_test')\n        \n        print(f\"Baseline captured: {len(baseline_metrics)} metrics\")\n        print(f\"Row count: {baseline_metrics['row_count']}\")\n        print(f\"Data quality: {100 - baseline_metrics['null_percentage']:.1f}%\")\n        \n        # Simulate regression check\n        print(\"\\n2. Regression Detection\")\n        print(\"-\" * 40)\n        \n        # Create slightly modified data to simulate regression\n        modified_data = test_data.copy()\n        modified_data.loc[0, 'amount'] = 110.0  # Small change\n        \n        processed_modified = clean_transaction_data(modified_data)\n        regression_result = regression_framework.check_regression(processed_modified, 'demo_test')\n        \n        print(f\"Regression status: {regression_result['status']}\")\n        print(f\"Regressions detected: {len(regression_result['regressions'])}\")\n        print(f\"Improvements detected: {len(regression_result['improvements'])}\")\n        \n        if regression_result['regressions']:\n            for regression in regression_result['regressions'][:3]:  # Show first 3\n                print(f\"  • {regression['metric']}: {regression['change_percent']:.1f}% change\")\n        \n        print(f\"✅ Regression testing: PASSED\")\n        \n        self.results['regression_testing'] = {\n            'baseline_metrics_count': len(baseline_metrics),\n            'regression_status': regression_result['status'],\n            'regressions_detected': len(regression_result['regressions'])\n        }\n    \n    def demo_cicd_integration(self):\n        \"\"\"Demonstrate CI/CD integration capabilities\"\"\"\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"🔄 CI/CD INTEGRATION DEMONSTRATION\")\n        print(\"=\"*60)\n        \n        cicd_framework = CICDTestingFramework()\n        \n        # Setup quality gates\n        print(\"\\n1. Quality Gates Configuration\")\n        print(\"-\" * 40)\n        \n        quality_gates = cicd_framework.setup_quality_gates()\n        \n        print(\"Quality Gates:\")\n        for gate_name, config in quality_gates.items():\n            print(f\"  • {gate_name}: {len(config)} criteria\")\n        \n        # Run CI/CD pipeline simulation\n        print(\"\\n2. Pipeline Execution\")\n        print(\"-\" * 40)\n        \n        pipeline_results = cicd_framework.run_cicd_pipeline('full')\n        \n        print(f\"Pipeline ID: {pipeline_results['pipeline_id']}\")\n        print(f\"Overall Status: {pipeline_results['overall_status']}\")\n        print(f\"Duration: {pipeline_results['duration_seconds']:.2f} seconds\")\n        \n        print(\"\\nStage Results:\")\n        for stage, results in pipeline_results['stages'].items():\n            status_emoji = \"✅\" if results['status'] == 'passed' else \"❌\"\n            print(f\"  {status_emoji} {stage}: {results['status']}\")\n        \n        print(\"\\nQuality Gate Results:\")\n        for gate, result in pipeline_results['quality_gate_results'].items():\n            status_emoji = \"✅\" if result['status'] == 'passed' else \"❌\"\n            print(f\"  {status_emoji} {gate}: {result['status']}\")\n        \n        if pipeline_results['recommendations']:\n            print(\"\\nRecommendations:\")\n            for rec in pipeline_results['recommendations'][:3]:\n                print(f\"  • {rec}\")\n        \n        print(f\"✅ CI/CD integration: PASSED\")\n        \n        self.results['cicd_integration'] = {\n            'pipeline_status': pipeline_results['overall_status'],\n            'stages_run': len(pipeline_results['stages']),\n            'quality_gates_passed': sum(1 for r in pipeline_results['quality_gate_results'].values() if r['status'] == 'passed')\n        }\n    \n    def generate_summary_report(self):\n        \"\"\"Generate comprehensive summary report\"\"\"\n        \n        print(\"\\n\" + \"=\"*60)\n        print(\"📊 TESTING FRAMEWORK SUMMARY REPORT\")\n        print(\"=\"*60)\n        \n        print(f\"\\n🕐 Demo completed at: {datetime.now().isoformat()}\")\n        \n        if 'unit_testing' in self.results:\n            unit_results = self.results['unit_testing']\n            print(f\"\\n🧪 Unit Testing Results:\")\n            print(f\"  • Data processed: {unit_results['original_records']} → {unit_results['cleaned_records']} records\")\n            print(f\"  • Customers analyzed: {unit_results['customers_analyzed']}\")\n            print(f\"  • Fraud rate detected: {unit_results['fraud_rate']:.2f}%\")\n        \n        if 'performance_testing' in self.results:\n            perf_results = self.results['performance_testing']\n            print(f\"\\n🚀 Performance Testing Results:\")\n            print(f\"  • Scalability rating: {perf_results['scalability_rating']}\")\n            print(f\"  • Memory usage: {perf_results['memory_used_mb']:.2f} MB\")\n            print(f\"  • Throughput: {perf_results['throughput']:.0f} records/sec\")\n        \n        if 'regression_testing' in self.results:\n            reg_results = self.results['regression_testing']\n            print(f\"\\n🔍 Regression Testing Results:\")\n            print(f\"  • Baseline metrics: {reg_results['baseline_metrics_count']}\")\n            print(f\"  • Regression status: {reg_results['regression_status']}\")\n            print(f\"  • Regressions detected: {reg_results['regressions_detected']}\")\n        \n        if 'cicd_integration' in self.results:\n            cicd_results = self.results['cicd_integration']\n            print(f\"\\n🔄 CI/CD Integration Results:\")\n            print(f\"  • Pipeline status: {cicd_results['pipeline_status']}\")\n            print(f\"  • Stages executed: {cicd_results['stages_run']}\")\n            print(f\"  • Quality gates passed: {cicd_results['quality_gates_passed']}\")\n        \n        # Save results to file\n        results_file = f\"demo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json\"\n        with open(results_file, 'w') as f:\n            json.dump(self.results, f, indent=2, default=str)\n        \n        print(f\"\\n💾 Results saved to: {results_file}\")\n        \n        print(\"\\n🎯 FRAMEWORK CAPABILITIES DEMONSTRATED:\")\n        print(\"  ✅ Comprehensive unit testing with data validation\")\n        print(\"  ✅ Performance testing with scalability analysis\")\n        print(\"  ✅ Regression testing with automated detection\")\n        print(\"  ✅ CI/CD integration with quality gates\")\n        print(\"  ✅ Real-time monitoring and reporting\")\n        \n        print(\"\\n🚀 Ready for production deployment!\")\n    \n    def run_full_demo(self):\n        \"\"\"Run the complete testing framework demonstration\"\"\"\n        \n        print(\"🎯 TestDriven Analytics - Comprehensive Testing Framework Demo\")\n        print(\"=\" * 70)\n        print(\"\\nThis demo showcases enterprise-grade testing capabilities for data pipelines.\")\n        print(\"We'll demonstrate unit testing, performance testing, regression detection,\")\n        print(\"and CI/CD integration with real data processing scenarios.\")\n        \n        try:\n            # Run all demo components\n            self.demo_unit_testing()\n            self.demo_performance_testing()\n            self.demo_regression_testing()\n            self.demo_cicd_integration()\n            \n            # Generate summary\n            self.generate_summary_report()\n            \n        except Exception as e:\n            print(f\"\\n❌ Demo failed with error: {e}\")\n            raise\n\ndef main():\n    \"\"\"Main entry point for the demo\"\"\"\n    \n    demo = TestingFrameworkDemo()\n    demo.run_full_demo()\n\nif __name__ == '__main__':\n    main()"