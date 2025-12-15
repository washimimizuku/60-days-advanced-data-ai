#!/usr/bin/env python3
"""
Day 57: Terraform & Infrastructure as Code - Setup Verification
Comprehensive setup verification script for Terraform development environment
"""

import os
import sys
import subprocess
import json
import importlib
from typing import Dict, List, Tuple, Optional
import platform


class SetupVerifier:
    """Verify Terraform development environment setup"""
    
    def __init__(self):
        self.results = []
        self.errors = []
        self.warnings = []
    
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
            "Platform": platform.platform()
        }
        
        for key, value in system_info.items():
            print(f"  {key}: {value}")
        
        return system_info
    
    def check_terraform_installation(self) -> bool:
        """Check Terraform installation and version"""
        print("\\n🏗️  Terraform Installation")
        print("-" * 30)
        
        success, output = self.run_command("terraform version")
        if success:
            version_line = output.split('\\n')[0]
            print(f"  ✅ {version_line}")
            
            # Check if version is >= 1.0
            try:
                version = version_line.split('v')[1].split(' ')[0]
                major_version = int(version.split('.')[0])
                if major_version >= 1:
                    print(f"  ✅ Version {version} meets minimum requirement (>= 1.0)")
                    return True
                else:
                    print(f"  ⚠️  Version {version} is below recommended minimum (>= 1.0)")
                    self.warnings.append(f"Terraform version {version} is below recommended minimum")
                    return True
            except (IndexError, ValueError):
                print(f"  ⚠️  Could not parse version from: {version_line}")
                self.warnings.append("Could not parse Terraform version")
                return True
        else:
            print("  ❌ Terraform not found or not accessible")
            self.errors.append("Terraform is not installed or not in PATH")
            return False
    
    def check_cloud_clis(self) -> Dict[str, bool]:
        """Check cloud CLI installations"""
        print("\\n☁️  Cloud CLI Tools")
        print("-" * 30)
        
        cli_tools = {
            "AWS CLI": "aws --version",
            "Azure CLI": "az version",
            "Google Cloud SDK": "gcloud version"
        }
        
        results = {}
        
        for tool_name, command in cli_tools.items():
            success, output = self.run_command(command)
            if success:
                print(f"  ✅ {tool_name} installed")
                if tool_name == "AWS CLI":
                    # Extract AWS CLI version
                    try:
                        version = output.split('/')[1].split(' ')[0]
                        print(f"     Version: {version}")
                    except IndexError:
                        pass
                results[tool_name] = True
            else:
                print(f"  ⚠️  {tool_name} not found (optional for multi-cloud)")
                results[tool_name] = False
        
        return results
    
    def check_python_dependencies(self) -> Dict[str, bool]:
        """Check Python dependencies"""
        print("\\n🐍 Python Dependencies")
        print("-" * 30)
        
        required_packages = [
            "terraform",
            "boto3",
            "pyyaml",
            "jinja2",
            "checkov",
            "pytest"
        ]
        
        optional_packages = [
            "azure-mgmt-resource",
            "google-cloud-storage",
            "ansible",
            "prometheus-client"
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
    
    def check_environment_variables(self) -> Dict[str, bool]:
        """Check environment variables"""
        print("\\n🔧 Environment Variables")
        print("-" * 30)
        
        # Check for .env file
        env_file_exists = os.path.exists('.env')
        print(f"  .env file: {'✅ Found' if env_file_exists else '⚠️  Not found (optional)'}")
        
        # Check important environment variables
        important_vars = [
            "AWS_REGION",
            "AWS_PROFILE",
            "TF_STATE_BUCKET",
            "PROJECT_NAME"
        ]
        
        optional_vars = [
            "AZURE_SUBSCRIPTION_ID",
            "GOOGLE_PROJECT_ID",
            "KUBERNETES_VERSION",
            "ENVIRONMENT"
        ]
        
        results = {}
        
        print("\\n  Important variables:")
        for var in important_vars:
            value = os.getenv(var)
            if value:
                print(f"    ✅ {var}")
                results[var] = True
            else:
                print(f"    ⚠️  {var} (not set)")
                results[var] = False
        
        print("\\n  Optional variables:")
        for var in optional_vars:
            value = os.getenv(var)
            if value:
                print(f"    ✅ {var}")
                results[var] = True
            else:
                print(f"    ⚠️  {var} (not set)")
                results[var] = False
        
        return results
    
    def check_aws_configuration(self) -> bool:
        """Check AWS configuration"""
        print("\\n🔐 AWS Configuration")
        print("-" * 30)
        
        # Check AWS credentials
        success, output = self.run_command("aws sts get-caller-identity")
        if success:
            try:
                identity = json.loads(output)
                print(f"  ✅ AWS credentials configured")
                print(f"     Account: {identity.get('Account', 'Unknown')}")
                print(f"     User/Role: {identity.get('Arn', 'Unknown').split('/')[-1]}")
                return True
            except json.JSONDecodeError:
                print(f"  ⚠️  AWS credentials configured but response format unexpected")
                return True
        else:
            print("  ⚠️  AWS credentials not configured or invalid")
            print("     Run 'aws configure' to set up credentials")
            return False
    
    def check_terraform_state_backend(self) -> bool:
        """Check Terraform state backend configuration"""
        print("\\n💾 Terraform State Backend")
        print("-" * 30)
        
        # Check if S3 bucket exists for state
        bucket_name = os.getenv('TF_STATE_BUCKET')
        if not bucket_name:
            print("  ⚠️  TF_STATE_BUCKET environment variable not set")
            return False
        
        success, output = self.run_command(f"aws s3 ls s3://{bucket_name}")
        if success:
            print(f"  ✅ S3 state bucket '{bucket_name}' accessible")
        else:
            print(f"  ⚠️  S3 state bucket '{bucket_name}' not accessible")
            print("     Create bucket with: aws s3 mb s3://{bucket_name}")
        
        # Check DynamoDB table for state locking
        table_name = os.getenv('TF_STATE_DYNAMODB_TABLE', 'terraform-state-locks')
        success, output = self.run_command(f"aws dynamodb describe-table --table-name {table_name}")
        if success:
            print(f"  ✅ DynamoDB lock table '{table_name}' exists")
            return True
        else:
            print(f"  ⚠️  DynamoDB lock table '{table_name}' not found")
            print(f"     Create table for state locking")
            return False
    
    def check_terraform_configuration(self) -> bool:
        """Check Terraform configuration files"""
        print("\\n📄 Terraform Configuration")
        print("-" * 30)
        
        terraform_files = [
            "main.tf",
            "variables.tf",
            "outputs.tf",
            "versions.tf"
        ]
        
        found_files = []
        for tf_file in terraform_files:
            if os.path.exists(tf_file):
                print(f"  ✅ {tf_file}")
                found_files.append(tf_file)
            else:
                print(f"  ⚠️  {tf_file} (not found)")
        
        if found_files:
            # Try to validate Terraform configuration
            success, output = self.run_command("terraform validate")
            if success:
                print("  ✅ Terraform configuration is valid")
                return True
            else:
                print("  ⚠️  Terraform configuration validation failed")
                print(f"     Error: {output}")
                return False
        else:
            print("  ⚠️  No Terraform configuration files found")
            return False
    
    def check_security_tools(self) -> Dict[str, bool]:
        """Check security and compliance tools"""
        print("\\n🔒 Security Tools")
        print("-" * 30)
        
        tools = {
            "Checkov": "checkov --version",
            "TFLint": "tflint --version",
            "Terraform Compliance": "terraform-compliance --version"
        }
        
        results = {}
        for tool_name, command in tools.items():
            success, output = self.run_command(command)
            if success:
                print(f"  ✅ {tool_name}")
                results[tool_name] = True
            else:
                print(f"  ⚠️  {tool_name} (optional)")
                results[tool_name] = False
        
        return results
    
    def run_basic_tests(self) -> bool:
        """Run basic functionality tests"""
        print("\\n🧪 Basic Tests")
        print("-" * 30)
        
        try:
            # Test Python imports
            from exercise import TerraformConfigGenerator, MockTerraformRunner
            print("  ✅ Exercise modules import successfully")
            
            # Test basic functionality
            generator = TerraformConfigGenerator("test", "dev")
            generator.add_variable("test_var", "string", "Test variable")
            config = generator.generate_hcl()
            
            if "variables.tf" in config and "test_var" in config["variables.tf"]:
                print("  ✅ Terraform configuration generation works")
            else:
                print("  ❌ Terraform configuration generation failed")
                return False
            
            # Test mock runner
            runner = MockTerraformRunner()
            result = runner.init("/tmp")
            
            if result.get("success"):
                print("  ✅ Mock Terraform runner works")
            else:
                print("  ❌ Mock Terraform runner failed")
                return False
            
            return True
            
        except Exception as e:
            print(f"  ❌ Basic tests failed: {e}")
            self.errors.append(f"Basic tests failed: {e}")
            return False
    
    def generate_report(self) -> None:
        """Generate setup verification report"""
        print("\\n" + "=" * 60)
        print("📊 SETUP VERIFICATION REPORT")
        print("=" * 60)
        
        if not self.errors:
            print("\\n✅ SETUP COMPLETE")
            print("Your Terraform development environment is ready!")
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
            print("1. Complete the exercises in exercise.py")
            print("2. Review the solution implementations")
            print("3. Set up your first Terraform infrastructure")
            print("4. Configure CI/CD for automated deployments")
        else:
            print("1. Install missing dependencies")
            print("2. Configure cloud credentials")
            print("3. Set up Terraform state backend")
            print("4. Re-run this verification script")
        
        print("\\n🔗 HELPFUL RESOURCES")
        print("-" * 20)
        print("• Setup Guide: SETUP.md")
        print("• Terraform Documentation: https://www.terraform.io/docs/")
        print("• AWS Provider Docs: https://registry.terraform.io/providers/hashicorp/aws/")
        print("• Best Practices: https://www.terraform-best-practices.com/")


def main():
    """Main verification function"""
    print("🚀 Day 57: Terraform & Infrastructure as Code - Setup Verification")
    print("=" * 70)
    
    verifier = SetupVerifier()
    
    # Run all checks
    verifier.check_system_info()
    verifier.check_terraform_installation()
    verifier.check_cloud_clis()
    verifier.check_python_dependencies()
    verifier.check_environment_variables()
    verifier.check_aws_configuration()
    verifier.check_terraform_state_backend()
    verifier.check_terraform_configuration()
    verifier.check_security_tools()
    verifier.run_basic_tests()
    
    # Generate final report
    verifier.generate_report()
    
    # Exit with appropriate code
    sys.exit(0 if not verifier.errors else 1)


if __name__ == "__main__":
    main()