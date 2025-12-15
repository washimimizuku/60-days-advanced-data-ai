#!/usr/bin/env python3
"""
Day 33: Load Testing for Model Serving API

Comprehensive load testing using Locust to validate performance requirements:
- Target: >1000 RPS throughput
- Target: <50ms p95 latency
- Target: >99% success rate

Usage:
    # Install locust: pip install locust
    # Run load test: locust -f load_test.py --host=http://localhost:8000
    # Web UI: http://localhost:8089
"""

import json
import random
import time
from datetime import datetime
from locust import HttpUser, task, between
import numpy as np

class FraudDetectionUser(HttpUser):
    """Simulated user for fraud detection API load testing"""
    
    wait_time = between(0.1, 0.5)  # Wait 0.1-0.5 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        self.transaction_counter = 0
        
        # Test health endpoint first
        response = self.client.get("/health")
        if response.status_code != 200:
            print(f"Health check failed: {response.status_code}")
    
    @task(8)
    def predict_fraud_normal(self):
        """Normal transaction prediction (80% of traffic)"""
        transaction = self._generate_normal_transaction()
        self._make_prediction_request(transaction)
    
    @task(2)
    def predict_fraud_suspicious(self):
        """Suspicious transaction prediction (20% of traffic)"""
        transaction = self._generate_suspicious_transaction()
        self._make_prediction_request(transaction)
    
    @task(1)
    def batch_predict(self):
        """Batch prediction request (10% of traffic)"""
        transactions = [
            self._generate_normal_transaction() for _ in range(5)
        ]
        
        with self.client.post(
            "/batch-predict",
            json=transactions,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Batch prediction failed: {response.status_code}")
    
    @task(1)
    def health_check(self):
        """Health check (monitoring traffic)"""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")
    
    @task(1)
    def get_metrics(self):
        """Metrics endpoint (monitoring traffic)"""
        with self.client.get("/metrics", catch_response=True) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Metrics failed: {response.status_code}")
    
    def _make_prediction_request(self, transaction):
        """Make prediction request with performance validation"""
        
        start_time = time.time()
        
        with self.client.post(
            "/predict",
            json=transaction,
            catch_response=True
        ) as response:
            
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    
                    # Validate response structure
                    required_fields = [
                        'transaction_id', 'fraud_probability', 'fraud_prediction',
                        'model_version', 'processing_time_ms'
                    ]
                    
                    for field in required_fields:
                        if field not in data:
                            response.failure(f"Missing field: {field}")
                            return
                    
                    # Validate data types and ranges
                    if not (0 <= data['fraud_probability'] <= 1):
                        response.failure(f"Invalid probability: {data['fraud_probability']}")
                        return
                    
                    if data['fraud_prediction'] not in [0, 1]:
                        response.failure(f"Invalid prediction: {data['fraud_prediction']}")
                        return
                    
                    # Performance validation
                    if response_time > 100:  # 100ms threshold
                        response.failure(f"High latency: {response_time:.1f}ms")\n                        return
                    
                    response.success()
                    
                except json.JSONDecodeError:
                    response.failure("Invalid JSON response")
                except Exception as e:
                    response.failure(f"Response validation error: {e}")
            else:
                response.failure(f"HTTP {response.status_code}")
    
    def _generate_normal_transaction(self):
        """Generate normal transaction data"""
        self.transaction_counter += 1
        
        return {
            "transaction_id": f"load_test_{self.transaction_counter}_{int(time.time())}",
            "amount": float(np.random.lognormal(3, 0.8)),  # Normal amounts
            "merchant_category": random.choice([
                "grocery", "gas", "restaurant", "retail"
            ]),
            "timestamp": datetime.now().isoformat(),
            "user_id": random.randint(1, 10000),
            "location": "domestic",
            "user_age": random.randint(25, 65),
            "use_cache": random.choice([True, False])
        }
    
    def _generate_suspicious_transaction(self):
        """Generate suspicious transaction data"""
        self.transaction_counter += 1
        
        return {
            "transaction_id": f"suspicious_{self.transaction_counter}_{int(time.time())}",
            "amount": float(np.random.lognormal(5, 1.2)),  # Higher amounts
            "merchant_category": random.choice([
                "online", "international", "cash_advance"
            ]),
            "timestamp": datetime.now().isoformat(),
            "user_id": random.randint(1, 10000),
            "location": "international",
            "user_age": random.randint(18, 30),  # Younger users
            "use_cache": False  # Don't cache suspicious transactions
        }

class HighVolumeUser(HttpUser):
    """High-volume user for stress testing"""
    
    wait_time = between(0.01, 0.05)  # Very short wait times
    
    @task
    def rapid_fire_predictions(self):
        """Rapid-fire predictions for stress testing"""
        transaction = {
            "transaction_id": f"stress_{int(time.time() * 1000000)}",
            "amount": 100.0,
            "merchant_category": "grocery",
            "timestamp": datetime.now().isoformat(),
            "user_id": 1,
            "location": "domestic",
            "user_age": 30,
            "use_cache": True
        }
        
        self.client.post("/predict", json=transaction)

# Custom load test scenarios
class LoadTestScenarios:
    """Predefined load test scenarios"""
    
    @staticmethod
    def normal_load():
        """Normal business load scenario"""
        return {
            "users": 100,
            "spawn_rate": 10,
            "run_time": "5m"
        }
    
    @staticmethod
    def peak_load():
        """Peak business hours scenario"""
        return {
            "users": 500,
            "spawn_rate": 50,
            "run_time": "10m"
        }
    
    @staticmethod
    def stress_test():
        """Stress test to find breaking point"""
        return {
            "users": 1000,
            "spawn_rate": 100,
            "run_time": "15m"
        }
    
    @staticmethod
    def endurance_test():
        """Long-running endurance test"""
        return {
            "users": 200,
            "spawn_rate": 20,
            "run_time": "60m"
        }

# Performance validation
class PerformanceValidator:
    """Validate performance requirements"""
    
    def __init__(self):
        self.requirements = {
            "max_response_time_p95": 50,  # 50ms
            "min_throughput_rps": 1000,   # 1000 RPS
            "min_success_rate": 99.0,     # 99%
            "max_error_rate": 1.0         # 1%
        }
    
    def validate_results(self, stats):
        """Validate load test results against requirements"""
        
        results = {
            "passed": True,
            "failures": [],
            "metrics": {}
        }
        
        # Extract key metrics
        total_requests = stats.get("total_requests", 0)
        failed_requests = stats.get("failed_requests", 0)
        avg_response_time = stats.get("avg_response_time", 0)
        p95_response_time = stats.get("p95_response_time", 0)
        rps = stats.get("requests_per_second", 0)
        
        # Calculate derived metrics
        success_rate = ((total_requests - failed_requests) / total_requests * 100) if total_requests > 0 else 0
        error_rate = (failed_requests / total_requests * 100) if total_requests > 0 else 0
        
        results["metrics"] = {
            "total_requests": total_requests,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "avg_response_time": avg_response_time,
            "p95_response_time": p95_response_time,
            "throughput_rps": rps
        }
        
        # Validate against requirements
        if p95_response_time > self.requirements["max_response_time_p95"]:
            results["passed"] = False
            results["failures"].append(
                f"P95 response time {p95_response_time}ms exceeds {self.requirements['max_response_time_p95']}ms"
            )
        
        if rps < self.requirements["min_throughput_rps"]:
            results["passed"] = False
            results["failures"].append(
                f"Throughput {rps} RPS below {self.requirements['min_throughput_rps']} RPS"
            )
        
        if success_rate < self.requirements["min_success_rate"]:
            results["passed"] = False
            results["failures"].append(
                f"Success rate {success_rate:.1f}% below {self.requirements['min_success_rate']}%"
            )
        
        if error_rate > self.requirements["max_error_rate"]:
            results["passed"] = False
            results["failures"].append(
                f"Error rate {error_rate:.1f}% exceeds {self.requirements['max_error_rate']}%"
            )
        
        return results

if __name__ == "__main__":
    print("Day 33: Model Serving Load Test")
    print("=" * 50)
    print()
    print("Available test scenarios:")
    print("1. Normal Load: 100 users, 5 minutes")
    print("2. Peak Load: 500 users, 10 minutes") 
    print("3. Stress Test: 1000 users, 15 minutes")
    print("4. Endurance Test: 200 users, 60 minutes")
    print()
    print("To run load test:")
    print("  locust -f load_test.py --host=http://localhost:8000")
    print("  # Then open http://localhost:8089 for web UI")
    print()
    print("Performance Requirements:")
    validator = PerformanceValidator()
    for key, value in validator.requirements.items():
        print(f"  {key}: {value}")