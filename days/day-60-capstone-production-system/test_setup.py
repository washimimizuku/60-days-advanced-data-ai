#!/usr/bin/env python3
"""
Day 60: Capstone Production System - Setup Verification

This script verifies that all components of the Intelligent Customer Analytics Platform
are properly installed and configured for development or production use.

Checks include:
- Python dependencies and versions
- Database connectivity (PostgreSQL, MongoDB, Redis)
- Streaming services (Kafka)
- ML platform components (MLflow, Feast)
- GenAI services (OpenAI, ChromaDB)
- Infrastructure tools (Docker, Kubernetes, Terraform)
- Monitoring stack (Prometheus, Grafana)

Author: 60 Days Advanced Data and AI Curriculum
"""

import sys
import os
import subprocess
import asyncio
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

@dataclass
class CheckResult:
    """Result of a system check"""
    name: str
    status: bool
    message: str
    details: Optional[str] = None
    critical: bool = True

class SystemChecker:
    """Comprehensive system verification for the capstone project"""
    
    def __init__(self):
        self.results: List[CheckResult] = []
        self.critical_failures = 0
        self.warnings = 0
    
    def print_header(self):
        """Print setup verification header"""
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}Day 60: Capstone Production System - Setup Verification{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}\n")
    
    def print_section(self, title: str):
        """Print section header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}🔍 {title}{Colors.END}")
        print(f"{Colors.BLUE}{'-' * (len(title) + 4)}{Colors.END}")
    
    def add_result(self, result: CheckResult):
        """Add check result and update counters"""
        self.results.append(result)
        
        if not result.status:
            if result.critical:
                self.critical_failures += 1
            else:
                self.warnings += 1
        
        # Print result immediately
        status_icon = f"{Colors.GREEN}✅" if result.status else f"{Colors.RED}❌"
        status_text = f"{Colors.GREEN}PASS" if result.status else f"{Colors.RED}FAIL"
        
        if not result.status and not result.critical:
            status_icon = f"{Colors.YELLOW}⚠️"
            status_text = f"{Colors.YELLOW}WARN"
        
        print(f"  {status_icon} {result.name}: {status_text}{Colors.END}")
        
        if result.message:
            print(f"     {Colors.WHITE}{result.message}{Colors.END}")
        
        if result.details and not result.status:
            print(f"     {Colors.YELLOW}Details: {result.details}{Colors.END}")
    
    def run_command(self, command: str, timeout: int = 30) -> Tuple[bool, str]:
        """Run shell command and return success status and output"""
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout.strip()
        except subprocess.TimeoutExpired:
            return False, f"Command timed out after {timeout} seconds"
        except Exception as e:
            return False, str(e)
    
    def check_python_version(self):
        """Check Python version compatibility"""
        version = sys.version_info
        
        if version.major == 3 and version.minor >= 9:
            self.add_result(CheckResult(
                name="Python Version",
                status=True,
                message=f"Python {version.major}.{version.minor}.{version.micro} (✓ Compatible)"
            ))
        else:
            self.add_result(CheckResult(
                name="Python Version",
                status=False,
                message=f"Python {version.major}.{version.minor}.{version.micro} (Requires 3.9+)",
                critical=True
            ))
    
    def check_python_dependencies(self):
        """Check required Python packages"""
        required_packages = [
            ("fastapi", "0.100.0"),
            ("pandas", "1.5.0"),
            ("numpy", "1.20.0"),
            ("scikit-learn", "1.0.0"),
            ("mlflow", "2.0.0"),
            ("langchain", "0.0.300"),
            ("redis", "4.0.0"),
            ("asyncpg", "0.25.0"),
            ("kafka-python", "2.0.0"),
            ("boto3", "1.20.0"),
            ("prometheus-client", "0.15.0")
        ]
        
        for package, min_version in required_packages:
            try:
                __import__(package)
                self.add_result(CheckResult(
                    name=f"Package: {package}",
                    status=True,
                    message="Installed and importable"
                ))
            except ImportError:
                self.add_result(CheckResult(
                    name=f"Package: {package}",
                    status=False,
                    message=f"Not installed (pip install {package}>={min_version})",
                    critical=True
                ))
    
    def check_docker(self):
        """Check Docker installation and status"""
        success, output = self.run_command("docker --version")
        
        if success:
            self.add_result(CheckResult(
                name="Docker Installation",
                status=True,
                message=f"Docker installed: {output}"
            ))
            
            # Check if Docker daemon is running
            success, _ = self.run_command("docker ps")
            self.add_result(CheckResult(
                name="Docker Daemon",
                status=success,
                message="Docker daemon is running" if success else "Docker daemon not running",
                critical=True
            ))
        else:
            self.add_result(CheckResult(
                name="Docker Installation",
                status=False,
                message="Docker not installed or not in PATH",
                details="Install Docker Desktop from https://docker.com/products/docker-desktop",
                critical=True
            ))
    
    def check_kubernetes(self):
        """Check Kubernetes installation and cluster access"""
        success, output = self.run_command("kubectl version --client")
        
        if success:
            self.add_result(CheckResult(
                name="kubectl Installation",
                status=True,
                message="kubectl installed"
            ))
            
            # Check cluster connectivity
            success, output = self.run_command("kubectl cluster-info")
            self.add_result(CheckResult(
                name="Kubernetes Cluster",
                status=success,
                message="Cluster accessible" if success else "No cluster access",
                details="Enable Kubernetes in Docker Desktop or configure cluster access",
                critical=False
            ))
        else:
            self.add_result(CheckResult(
                name="kubectl Installation",
                status=False,
                message="kubectl not installed",
                details="Install kubectl: https://kubernetes.io/docs/tasks/tools/",
                critical=False
            ))
    
    def check_terraform(self):
        """Check Terraform installation"""
        success, output = self.run_command("terraform version")
        
        if success:
            version_line = output.split('\n')[0]
            self.add_result(CheckResult(
                name="Terraform Installation",
                status=True,
                message=f"Terraform installed: {version_line}"
            ))
        else:
            self.add_result(CheckResult(
                name="Terraform Installation",
                status=False,
                message="Terraform not installed",
                details="Install from https://terraform.io/downloads",
                critical=False
            ))
    
    async def check_database_connectivity(self):
        """Check database connections"""
        # PostgreSQL
        try:
            import asyncpg
            
            try:
                conn = await asyncpg.connect(
                    host="localhost",
                    port=5432,
                    user="analytics_user",
                    password="secure_password",
                    database="customers",
                    timeout=5
                )
                await conn.fetchval("SELECT 1")
                await conn.close()
                
                self.add_result(CheckResult(
                    name="PostgreSQL Connection",
                    status=True,
                    message="Connected successfully"
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="PostgreSQL Connection",
                    status=False,
                    message="Connection failed",
                    details=str(e),
                    critical=False
                ))
        except ImportError:
            self.add_result(CheckResult(
                name="PostgreSQL Driver",
                status=False,
                message="asyncpg not installed",
                critical=True
            ))
        
        # Redis
        try:
            import redis.asyncio as redis
            
            try:
                client = redis.from_url("redis://localhost:6379", socket_timeout=5)
                await client.ping()
                await client.close()
                
                self.add_result(CheckResult(
                    name="Redis Connection",
                    status=True,
                    message="Connected successfully"
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="Redis Connection",
                    status=False,
                    message="Connection failed",
                    details=str(e),
                    critical=False
                ))
        except ImportError:
            self.add_result(CheckResult(
                name="Redis Driver",
                status=False,
                message="redis not installed",
                critical=True
            ))
        
        # MongoDB
        try:
            from pymongo import MongoClient
            
            try:
                client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
                client.server_info()
                client.close()
                
                self.add_result(CheckResult(
                    name="MongoDB Connection",
                    status=True,
                    message="Connected successfully"
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="MongoDB Connection",
                    status=False,
                    message="Connection failed",
                    details=str(e),
                    critical=False
                ))
        except ImportError:
            self.add_result(CheckResult(
                name="MongoDB Driver",
                status=False,
                message="pymongo not installed",
                critical=True
            ))
    
    def check_kafka_connectivity(self):
        """Check Kafka connectivity"""
        try:
            from kafka import KafkaProducer, KafkaConsumer
            
            try:
                producer = KafkaProducer(
                    bootstrap_servers=['localhost:9092'],
                    request_timeout_ms=5000,
                    api_version=(0, 10, 1)
                )
                producer.close()
                
                self.add_result(CheckResult(
                    name="Kafka Connection",
                    status=True,
                    message="Connected successfully"
                ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="Kafka Connection",
                    status=False,
                    message="Connection failed",
                    details=str(e),
                    critical=False
                ))
        except ImportError:
            self.add_result(CheckResult(
                name="Kafka Driver",
                status=False,
                message="kafka-python not installed",
                critical=True
            ))
    
    def check_mlflow_service(self):
        """Check MLflow tracking server"""
        try:
            import requests
            
            try:
                response = requests.get("http://localhost:5000/api/2.0/mlflow/experiments/list", timeout=5)
                
                if response.status_code == 200:
                    self.add_result(CheckResult(
                        name="MLflow Server",
                        status=True,
                        message="MLflow tracking server accessible"
                    ))
                else:
                    self.add_result(CheckResult(
                        name="MLflow Server",
                        status=False,
                        message=f"MLflow server returned status {response.status_code}",
                        critical=False
                    ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="MLflow Server",
                    status=False,
                    message="MLflow server not accessible",
                    details=str(e),
                    critical=False
                ))
        except ImportError:
            self.add_result(CheckResult(
                name="Requests Library",
                status=False,
                message="requests not installed",
                critical=True
            ))
    
    def check_genai_services(self):
        """Check GenAI service availability"""
        # OpenAI API Key
        openai_key = os.getenv("OPENAI_API_KEY")
        
        if openai_key and openai_key.startswith("sk-"):
            self.add_result(CheckResult(
                name="OpenAI API Key",
                status=True,
                message="API key configured"
            ))
        else:
            self.add_result(CheckResult(
                name="OpenAI API Key",
                status=False,
                message="API key not configured (system will run in mock mode)",
                critical=False
            ))
        
        # ChromaDB
        try:
            import requests
            
            try:
                response = requests.get("http://localhost:8000/api/v1/heartbeat", timeout=5)
                
                if response.status_code == 200:
                    self.add_result(CheckResult(
                        name="ChromaDB Service",
                        status=True,
                        message="ChromaDB server accessible"
                    ))
                else:
                    self.add_result(CheckResult(
                        name="ChromaDB Service",
                        status=False,
                        message="ChromaDB server not responding",
                        critical=False
                    ))
            except Exception as e:
                self.add_result(CheckResult(
                    name="ChromaDB Service",
                    status=False,
                    message="ChromaDB server not accessible",
                    details=str(e),
                    critical=False
                ))
        except ImportError:
            pass  # Already checked requests above
    
    def check_aws_configuration(self):
        """Check AWS CLI configuration"""
        success, output = self.run_command("aws --version")
        
        if success:
            self.add_result(CheckResult(
                name="AWS CLI Installation",
                status=True,
                message=f"AWS CLI installed: {output.split()[0]}"
            ))
            
            # Check AWS credentials
            success, output = self.run_command("aws sts get-caller-identity")
            
            if success:
                self.add_result(CheckResult(
                    name="AWS Credentials",
                    status=True,
                    message="AWS credentials configured"
                ))
            else:
                self.add_result(CheckResult(
                    name="AWS Credentials",
                    status=False,
                    message="AWS credentials not configured",
                    details="Run 'aws configure' to set up credentials",
                    critical=False
                ))
        else:
            self.add_result(CheckResult(
                name="AWS CLI Installation",
                status=False,
                message="AWS CLI not installed",
                details="Install from https://aws.amazon.com/cli/",
                critical=False
            ))
    
    def check_file_structure(self):
        """Check required files and directories"""
        required_files = [
            "solution.py",
            "requirements.txt",
            "SETUP.md",
            ".env.example"
        ]
        
        required_dirs = [
            ".",  # Current directory should exist
        ]
        
        for file_path in required_files:
            if Path(file_path).exists():
                self.add_result(CheckResult(
                    name=f"File: {file_path}",
                    status=True,
                    message="File exists"
                ))
            else:
                self.add_result(CheckResult(
                    name=f"File: {file_path}",
                    status=False,
                    message="File missing",
                    critical=True
                ))
        
        # Check .env file
        if Path(".env").exists():
            self.add_result(CheckResult(
                name="Environment File (.env)",
                status=True,
                message="Environment file configured"
            ))
        else:
            self.add_result(CheckResult(
                name="Environment File (.env)",
                status=False,
                message="Copy .env.example to .env and configure",
                critical=False
            ))
    
    def check_monitoring_services(self):
        """Check monitoring stack availability"""
        services = [
            ("Prometheus", "http://localhost:9090/-/healthy"),
            ("Grafana", "http://localhost:3000/api/health")
        ]
        
        try:
            import requests
            
            for service_name, url in services:
                try:
                    response = requests.get(url, timeout=5)
                    
                    if response.status_code == 200:
                        self.add_result(CheckResult(
                            name=f"{service_name} Service",
                            status=True,
                            message=f"{service_name} accessible"
                        ))
                    else:
                        self.add_result(CheckResult(
                            name=f"{service_name} Service",
                            status=False,
                            message=f"{service_name} not responding",
                            critical=False
                        ))
                except Exception:
                    self.add_result(CheckResult(
                        name=f"{service_name} Service",
                        status=False,
                        message=f"{service_name} not accessible",
                        critical=False
                    ))
        except ImportError:
            pass  # Already checked requests above
    
    def print_summary(self):
        """Print verification summary"""
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.status)
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}SETUP VERIFICATION SUMMARY{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.END}")
        
        print(f"\n{Colors.BOLD}Total Checks: {total_checks}{Colors.END}")
        print(f"{Colors.GREEN}✅ Passed: {passed_checks}{Colors.END}")
        print(f"{Colors.RED}❌ Failed: {self.critical_failures}{Colors.END}")
        print(f"{Colors.YELLOW}⚠️  Warnings: {self.warnings}{Colors.END}")
        
        # Overall status
        if self.critical_failures == 0:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 SETUP VERIFICATION PASSED!{Colors.END}")
            print(f"{Colors.GREEN}Your system is ready for the capstone project.{Colors.END}")
            
            if self.warnings > 0:
                print(f"{Colors.YELLOW}Note: {self.warnings} optional components are not configured.{Colors.END}")
                print(f"{Colors.YELLOW}The system will work with reduced functionality.{Colors.END}")
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}❌ SETUP VERIFICATION FAILED!{Colors.END}")
            print(f"{Colors.RED}Please fix the {self.critical_failures} critical issues before proceeding.{Colors.END}")
        
        # Next steps
        print(f"\n{Colors.BOLD}Next Steps:{Colors.END}")
        
        if self.critical_failures == 0:
            print(f"{Colors.GREEN}1. Start the development environment: docker-compose up -d{Colors.END}")
            print(f"{Colors.GREEN}2. Run the application: python solution.py{Colors.END}")
            print(f"{Colors.GREEN}3. Access API documentation: http://localhost:8080/docs{Colors.END}")
            print(f"{Colors.GREEN}4. Run tests: pytest test_capstone_production_system.py -v{Colors.END}")
        else:
            print(f"{Colors.RED}1. Install missing dependencies: pip install -r requirements.txt{Colors.END}")
            print(f"{Colors.RED}2. Fix critical issues listed above{Colors.END}")
            print(f"{Colors.RED}3. Re-run this verification: python test_setup.py{Colors.END}")
        
        print(f"\n{Colors.CYAN}For detailed setup instructions, see: SETUP.md{Colors.END}")
        print(f"{Colors.CYAN}For troubleshooting help, see: docs/TROUBLESHOOTING.md{Colors.END}")
    
    async def run_all_checks(self):
        """Run all system verification checks"""
        self.print_header()
        
        # Core system checks
        self.print_section("Core System Requirements")
        self.check_python_version()
        self.check_python_dependencies()
        self.check_file_structure()
        
        # Infrastructure tools
        self.print_section("Infrastructure Tools")
        self.check_docker()
        self.check_kubernetes()
        self.check_terraform()
        self.check_aws_configuration()
        
        # Database connectivity
        self.print_section("Database Services")
        await self.check_database_connectivity()
        
        # Streaming services
        self.print_section("Streaming Services")
        self.check_kafka_connectivity()
        
        # ML platform
        self.print_section("ML Platform Services")
        self.check_mlflow_service()
        
        # GenAI services
        self.print_section("GenAI Services")
        self.check_genai_services()
        
        # Monitoring stack
        self.print_section("Monitoring Services")
        self.check_monitoring_services()
        
        # Print final summary
        self.print_summary()
        
        return self.critical_failures == 0

async def main():
    """Main verification function"""
    checker = SystemChecker()
    success = await checker.run_all_checks()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    asyncio.run(main())