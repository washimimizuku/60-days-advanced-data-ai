#!/usr/bin/env python3
"""
Day 58: Monitoring & Observability - Setup Verification
Comprehensive setup verification script for monitoring and observability stack
"""

import os
import sys
import subprocess
import json
import requests
import time
import importlib
from typing import Dict, List, Tuple, Optional
import platform


class ObservabilitySetupVerifier:
    """Verify monitoring and observability stack setup"""
    
    def __init__(self):
        self.results = []
        self.errors = []
        self.warnings = []
        self.services = {
            "Prometheus": {"url": "http://localhost:9090", "health_path": "/-/healthy"},
            "Grafana": {"url": "http://localhost:3000", "health_path": "/api/health"},
            "Jaeger": {"url": "http://localhost:16686", "health_path": "/"},
            "Elasticsearch": {"url": "http://localhost:9200", "health_path": "/_cluster/health"},
            "Kibana": {"url": "http://localhost:5601", "health_path": "/api/status"},
            "Alertmanager": {"url": "http://localhost:9093", "health_path": "/-/healthy"}
        }
    
    def run_command(self, command: str, capture_output: bool = True) -> Tuple[bool, str]:
        """Run shell command and return success status and output"""
        try:
            if capture_output:
                result = subprocess.run(
                    command.split(),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                return result.returncode == 0, result.stdout.strip()
            else:
                result = subprocess.run(command.split(), timeout=30)
                return result.returncode == 0, ""
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            return False, str(e)
    
    def check_system_info(self) -> Dict[str, str]:
        """Check system information"""
        print("🖥️  System Information")
        print("-" * 30)
        
        system_info = {
            "OS": platform.system(),
            "OS Version": platform.version(),
            "Architecture": platform.machine(),
            "Python Version": platform.python_version(),
            "Available Memory": self._get_memory_info(),
            "Available Disk": self._get_disk_info()
        }
        
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        
        return system_info
    
    def _get_memory_info(self) -> str:
        """Get system memory information"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return f"{memory.total // (1024**3)}GB total, {memory.available // (1024**3)}GB available"
        except ImportError:
            return "Unknown (psutil not installed)"
    
    def _get_disk_info(self) -> str:
        """Get disk space information"""
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return f"{disk.total // (1024**3)}GB total, {disk.free // (1024**3)}GB free"
        except ImportError:
            return "Unknown (psutil not installed)"
    
    def check_docker_installation(self) -> bool:
        """Check Docker and Docker Compose installation"""
        print("\\n🐳 Docker Installation")
        print("-" * 25)
        
        # Check Docker
        success, output = self.run_command("docker --version")
        if success:
            print(f"  ✅ Docker: {output}")
        else:
            print("  ❌ Docker not found")
            self.errors.append("Docker is not installed or not accessible")
            return False
        
        # Check Docker Compose
        success, output = self.run_command("docker-compose --version")
        if success:
            print(f"  ✅ Docker Compose: {output}")
        else:
            # Try newer docker compose command
            success, output = self.run_command("docker compose version")
            if success:
                print(f"  ✅ Docker Compose: {output}")
            else:
                print("  ❌ Docker Compose not found")
                self.errors.append("Docker Compose is not installed")
                return False
        
        # Check Docker daemon
        success, output = self.run_command("docker info")
        if success:
            print("  ✅ Docker daemon is running")
            return True
        else:
            print("  ❌ Docker daemon is not running")
            self.errors.append("Docker daemon is not running")
            return False
    
    def check_python_dependencies(self) -> Dict[str, bool]:
        """Check Python dependencies for observability"""
        print("\\n🐍 Python Dependencies")
        print("-" * 30)
        
        required_packages = [
            "prometheus_client",
            "opentelemetry",
            "structlog",
            "requests",
            "psutil"
        ]
        
        optional_packages = [
            "elasticsearch",
            "grafana_api",
            "jaeger_client",
            "evidently",
            "alibi_detect"
        ]
        
        results = {}
        
        # Check required packages
        print("  Required packages:")
        for package in required_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                print(f"    ✅ {package}")
                results[package] = True
            except ImportError:
                print(f"    ❌ {package}")
                results[package] = False
                self.errors.append(f"Required package {package} not installed")
        
        # Check optional packages
        print("\\n  Optional packages:")
        for package in optional_packages:
            try:
                importlib.import_module(package.replace('-', '_'))
                print(f"    ✅ {package}")
                results[package] = True
            except ImportError:
                print(f"    ⚠️  {package} (optional)")
                results[package] = False
        
        return results
    
    def check_configuration_files(self) -> Dict[str, bool]:
        """Check configuration files"""
        print("\\n📄 Configuration Files")
        print("-" * 25)
        
        config_files = {
            "docker-compose.yml": "Docker Compose configuration",
            "prometheus/prometheus.yml": "Prometheus configuration",
            "grafana/grafana.ini": "Grafana configuration",
            "elk/logstash.conf": "Logstash configuration",
            ".env": "Environment variables"
        }
        
        results = {}
        
        for file_path, description in config_files.items():
            if os.path.exists(file_path):
                print(f"  ✅ {file_path} - {description}")
                results[file_path] = True
            else:
                print(f"  ⚠️  {file_path} - {description} (not found)")
                results[file_path] = False
                if file_path == "docker-compose.yml":
                    self.warnings.append("docker-compose.yml not found - services may not start")
        
        return results
    
    def check_docker_services(self) -> Dict[str, bool]:
        """Check if Docker services are running"""
        print("\\n🔧 Docker Services")
        print("-" * 20)
        
        # Check if docker-compose.yml exists
        if not os.path.exists("docker-compose.yml"):
            print("  ⚠️  docker-compose.yml not found - skipping service checks")
            return {}
        
        # Get running services
        success, output = self.run_command("docker-compose ps")
        if not success:
            print("  ❌ Failed to get Docker Compose status")
            return {}
        
        services = ["prometheus", "grafana", "jaeger", "elasticsearch", "kibana", "alertmanager"]
        results = {}
        
        for service in services:
            if service in output and "Up" in output:
                print(f"  ✅ {service} - Running")
                results[service] = True
            else:
                print(f"  ⚠️  {service} - Not running")
                results[service] = False
        
        return results
    
    def check_service_health(self) -> Dict[str, bool]:
        """Check health of monitoring services"""
        print("\\n🏥 Service Health Checks")
        print("-" * 30)
        
        results = {}
        
        for service_name, config in self.services.items():
            url = config["url"] + config["health_path"]
            
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    print(f"  ✅ {service_name} - Healthy")
                    results[service_name] = True
                else:
                    print(f"  ⚠️  {service_name} - HTTP {response.status_code}")
                    results[service_name] = False
            except requests.exceptions.RequestException as e:
                print(f"  ❌ {service_name} - Connection failed")
                results[service_name] = False
        
        return results
    
    def test_metrics_collection(self) -> bool:
        """Test metrics collection functionality"""
        print("\\n📊 Metrics Collection Test")
        print("-" * 30)
        
        try:
            from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry
            
            # Create test registry
            registry = CollectorRegistry()
            
            # Create test metrics
            test_counter = Counter('test_counter_total', 'Test counter', registry=registry)
            test_histogram = Histogram('test_histogram_seconds', 'Test histogram', registry=registry)
            test_gauge = Gauge('test_gauge', 'Test gauge', registry=registry)
            
            # Record test values
            test_counter.inc(5)
            test_histogram.observe(0.5)
            test_gauge.set(42)
            
            # Verify metrics
            metrics = list(registry.collect())
            if len(metrics) >= 3:
                print("  ✅ Metrics creation and collection works")
                return True
            else:
                print("  ❌ Metrics collection failed")
                return False
                
        except Exception as e:
            print(f"  ❌ Metrics test failed: {e}")
            self.errors.append(f"Metrics collection test failed: {e}")
            return False
    
    def test_structured_logging(self) -> bool:
        """Test structured logging functionality"""
        print("\\n📝 Structured Logging Test")
        print("-" * 30)
        
        try:
            import json
            import logging
            from datetime import datetime
            
            # Create test logger
            logger = logging.getLogger("test_logger")
            
            # Create structured log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "service": "test-service",
                "level": "INFO",
                "message": "Test log entry",
                "context": {
                    "test_field": "test_value",
                    "numeric_field": 123
                }
            }
            
            # Test JSON serialization
            json_log = json.dumps(log_entry)
            parsed_log = json.loads(json_log)
            
            if parsed_log["service"] == "test-service":
                print("  ✅ Structured logging works")
                return True
            else:
                print("  ❌ Structured logging failed")
                return False
                
        except Exception as e:
            print(f"  ❌ Structured logging test failed: {e}")
            self.errors.append(f"Structured logging test failed: {e}")
            return False
    
    def test_tracing_setup(self) -> bool:
        """Test distributed tracing setup"""
        print("\\n🔍 Distributed Tracing Test")
        print("-" * 35)
        
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            
            # Set up tracer
            trace.set_tracer_provider(TracerProvider())
            tracer = trace.get_tracer(__name__)
            
            # Create test span
            with tracer.start_as_current_span("test_span") as span:
                span.set_attribute("test.attribute", "test_value")
                span.set_attribute("test.number", 42)
            
            print("  ✅ Distributed tracing setup works")
            return True
            
        except Exception as e:
            print(f"  ❌ Tracing test failed: {e}")
            self.errors.append(f"Distributed tracing test failed: {e}")
            return False
    
    def run_integration_test(self) -> bool:
        """Run integration test with all components"""
        print("\\n🧪 Integration Test")
        print("-" * 20)
        
        try:
            # Test ML metrics simulation
            from exercise import MockPrometheusClient
            
            client = MockPrometheusClient()
            
            # Simulate model metrics
            client.counter_inc("ml_predictions_total", {"model": "test"}, 1.0)
            client.gauge_set("ml_accuracy", 0.95, {"model": "test"})
            client.histogram_observe("ml_latency", 0.1, {"model": "test"})
            
            # Verify metrics
            predictions = client.get_metric_values("ml_predictions_total")
            accuracy = client.get_metric_values("ml_accuracy")
            latency = client.get_metric_values("ml_latency")
            
            if len(predictions) > 0 and len(accuracy) > 0 and len(latency) > 0:
                print("  ✅ ML metrics integration works")
                return True
            else:
                print("  ❌ ML metrics integration failed")
                return False
                
        except Exception as e:
            print(f"  ❌ Integration test failed: {e}")
            self.errors.append(f"Integration test failed: {e}")
            return False
    
    def check_resource_requirements(self) -> Dict[str, bool]:
        """Check system resource requirements"""
        print("\\n💾 Resource Requirements")
        print("-" * 30)
        
        requirements = {
            "memory_gb": 8,
            "disk_gb": 20,
            "cpu_cores": 4
        }
        
        results = {}
        
        try:
            import psutil
            
            # Check memory
            memory_gb = psutil.virtual_memory().total // (1024**3)
            if memory_gb >= requirements["memory_gb"]:
                print(f"  ✅ Memory: {memory_gb}GB (>= {requirements['memory_gb']}GB required)")
                results["memory"] = True
            else:
                print(f"  ⚠️  Memory: {memory_gb}GB (< {requirements['memory_gb']}GB required)")
                results["memory"] = False
                self.warnings.append(f"Insufficient memory: {memory_gb}GB < {requirements['memory_gb']}GB")
            
            # Check disk space
            disk_gb = psutil.disk_usage('/').free // (1024**3)
            if disk_gb >= requirements["disk_gb"]:
                print(f"  ✅ Disk Space: {disk_gb}GB free (>= {requirements['disk_gb']}GB required)")
                results["disk"] = True
            else:
                print(f"  ⚠️  Disk Space: {disk_gb}GB free (< {requirements['disk_gb']}GB required)")
                results["disk"] = False
                self.warnings.append(f"Insufficient disk space: {disk_gb}GB < {requirements['disk_gb']}GB")
            
            # Check CPU cores
            cpu_cores = psutil.cpu_count()
            if cpu_cores >= requirements["cpu_cores"]:
                print(f"  ✅ CPU Cores: {cpu_cores} (>= {requirements['cpu_cores']} required)")
                results["cpu"] = True
            else:
                print(f"  ⚠️  CPU Cores: {cpu_cores} (< {requirements['cpu_cores']} required)")
                results["cpu"] = False
                self.warnings.append(f"Insufficient CPU cores: {cpu_cores} < {requirements['cpu_cores']}")
            
        except ImportError:
            print("  ⚠️  Cannot check resources (psutil not installed)")
            results = {"memory": False, "disk": False, "cpu": False}
        
        return results
    
    def generate_report(self) -> None:
        """Generate setup verification report"""
        print("\\n" + "=" * 60)
        print("📊 OBSERVABILITY SETUP VERIFICATION REPORT")
        print("=" * 60)
        
        if not self.errors:
            print("\\n✅ SETUP COMPLETE")
            print("Your monitoring and observability stack is ready!")
        else:
            print("\\n⚠️  SETUP ISSUES FOUND")
            print("Please address the following issues:")
            for error in self.errors:
                print(f"  • {error}")
        
        if self.warnings:
            print("\\n⚠️  WARNINGS")
            for warning in self.warnings:
                print(f"  • {warning}")
        
        print("\\n📋 NEXT STEPS")
        print("-" * 20)
        if not self.errors:
            print("1. Start the observability stack: docker-compose up -d")
            print("2. Access Grafana: http://localhost:3000 (admin/admin123)")
            print("3. Access Prometheus: http://localhost:9090")
            print("4. Access Jaeger: http://localhost:16686")
            print("5. Complete the exercises in exercise.py")
        else:
            print("1. Install missing dependencies: pip install -r requirements.txt")
            print("2. Install Docker and Docker Compose")
            print("3. Create configuration files (see SETUP.md)")
            print("4. Re-run this verification script")
        
        print("\\n🔗 HELPFUL RESOURCES")
        print("-" * 20)
        print("• Setup Guide: SETUP.md")
        print("• Prometheus Docs: https://prometheus.io/docs/")
        print("• Grafana Docs: https://grafana.com/docs/")
        print("• Jaeger Docs: https://www.jaegertracing.io/docs/")
        print("• OpenTelemetry Docs: https://opentelemetry.io/docs/")


def main():
    """Main verification function"""
    print("🚀 Day 58: Monitoring & Observability - Setup Verification")
    print("=" * 70)
    
    verifier = ObservabilitySetupVerifier()
    
    # Run all checks
    verifier.check_system_info()
    verifier.check_docker_installation()
    verifier.check_python_dependencies()
    verifier.check_configuration_files()
    verifier.check_docker_services()
    verifier.check_service_health()
    verifier.test_metrics_collection()
    verifier.test_structured_logging()
    verifier.test_tracing_setup()
    verifier.run_integration_test()
    verifier.check_resource_requirements()
    
    # Generate final report
    verifier.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if not verifier.errors else 1)


if __name__ == "__main__":
    main()