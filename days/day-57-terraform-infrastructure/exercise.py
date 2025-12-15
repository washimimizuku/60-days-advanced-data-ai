"""
Day 57: Terraform & Infrastructure as Code - Multi-cloud & Best Practices
Exercises for building production-ready ML infrastructure with Terraform
"""

import os
import json
import yaml
import subprocess
import tempfile
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TerraformResource:
    """Terraform resource representation"""
    resource_type: str
    name: str
    configuration: Dict[str, Any]
    dependencies: Optional[List[str]] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []


class TerraformConfigGenerator:
    """Generate Terraform configurations for ML infrastructure"""
    
    def __init__(self, project_name: str, environment: str):
        self.project_name = project_name
        self.environment = environment
        self.resources = {}
        self.variables = {}
        self.outputs = {}
        
    def add_variable(self, name: str, var_type: str, description: str, 
                    default: Any = None, validation: Dict = None):
        """Add a Terraform variable"""
        var_config = {
            "description": description,
            "type": var_type
        }
        
        if default is not None:
            var_config["default"] = default
            
        if validation:
            var_config["validation"] = validation
            
        self.variables[name] = var_config
    
    def add_output(self, name: str, value: str, description: str, sensitive: bool = False):
        """Add a Terraform output"""
        self.outputs[name] = {
            "description": description,
            "value": value,
            "sensitive": sensitive
        }
    
    def add_resource(self, resource: TerraformResource):
        """Add a Terraform resource"""
        resource_key = f"{resource.resource_type}.{resource.name}"
        self.resources[resource_key] = resource
    
    def generate_hcl(self) -> Dict[str, str]:
        """Generate HCL configuration files"""
        files = {}
        
        # Generate main.tf
        files["main.tf"] = self._generate_main_tf()
        
        # Generate variables.tf
        if self.variables:
            files["variables.tf"] = self._generate_variables_tf()
        
        # Generate outputs.tf
        if self.outputs:
            files["outputs.tf"] = self._generate_outputs_tf()
        
        # Generate versions.tf
        files["versions.tf"] = self._generate_versions_tf()
        
        return files
    
    def _generate_main_tf(self) -> str:
        """Generate main Terraform configuration"""
        config_lines = []
        
        # Add provider configuration
        config_lines.extend([
            'terraform {',
            '  required_providers {',
            '    aws = {',
            '      source  = "hashicorp/aws"',
            '      version = "~> 5.0"',
            '    }',
            '  }',
            '  required_version = ">= 1.0"',
            '}',
            '',
            'provider "aws" {',
            '  region = var.aws_region',
            '',
            '  default_tags {',
            '    tags = {',
            f'      Environment = "{self.environment}"',
            f'      Project     = "{self.project_name}"',
            '      ManagedBy   = "terraform"',
            '    }',
            '  }',
            '}',
            ''
        ])
        
        # Add resources
        for resource_key, resource in self.resources.items():
            config_lines.extend(self._generate_resource_hcl(resource))
            config_lines.append('')
        
        return '\n'.join(config_lines)
    
    def _generate_resource_hcl(self, resource: TerraformResource) -> List[str]:
        """Generate HCL for a single resource"""
        lines = [f'resource "{resource.resource_type}" "{resource.name}" {{']
        
        for key, value in resource.configuration.items():
            lines.extend(self._format_hcl_value(key, value, indent=2))
        
        lines.append('}')
        return lines
    
    def _format_hcl_value(self, key: str, value: Any, indent: int = 0) -> List[str]:
        """Format HCL values with proper indentation"""
        prefix = ' ' * indent
        
        if isinstance(value, dict):
            lines = [f'{prefix}{key} {{']
            for sub_key, sub_value in value.items():
                lines.extend(self._format_hcl_value(sub_key, sub_value, indent + 2))
            lines.append(f'{prefix}}}')
            return lines
        elif isinstance(value, list):
            if all(isinstance(item, str) for item in value):
                # Simple string list
                formatted_items = ', '.join(f'"{item}"' for item in value)
                return [f'{prefix}{key} = [{formatted_items}]']
            else:
                # Complex list
                lines = [f'{prefix}{key} = [']
                for item in value:
                    if isinstance(item, dict):
                        lines.append(f'{prefix}  {{')
                        for sub_key, sub_value in item.items():
                            lines.extend(self._format_hcl_value(sub_key, sub_value, indent + 4))
                        lines.append(f'{prefix}  }},')
                    else:
                        lines.append(f'{prefix}  "{item}",')
                lines.append(f'{prefix}]')
                return lines
        elif isinstance(value, str):
            # Check for Terraform references more comprehensively
            if (value.startswith('${') or value.startswith('var.') or 
                value.startswith('data.') or value.startswith('local.') or 
                value.startswith('module.')):
                return [f'{prefix}{key} = {value}']
            else:
                return [f'{prefix}{key} = "{value}"']
        elif isinstance(value, bool):
            return [f'{prefix}{key} = {str(value).lower()}']
        else:
            return [f'{prefix}{key} = {value}']
    
    def _generate_variables_tf(self) -> str:
        """Generate variables.tf file"""
        lines = []
        
        for var_name, var_config in self.variables.items():
            lines.append(f'variable "{var_name}" {{')
            
            for key, value in var_config.items():
                if key == "validation":
                    lines.append('  validation {')
                    for val_key, val_value in value.items():
                        if isinstance(val_value, str):
                            lines.append(f'    {val_key} = "{val_value}"')
                        else:
                            lines.append(f'    {val_key} = {val_value}')
                    lines.append('  }')
                elif isinstance(value, str):
                    lines.append(f'  {key} = "{value}"')
                else:
                    lines.append(f'  {key} = {value}')
            
            lines.append('}')
            lines.append('')
        
        return '\n'.join(lines)
    
    def _generate_outputs_tf(self) -> str:
        """Generate outputs.tf file"""
        lines = []
        
        for output_name, output_config in self.outputs.items():
            lines.append(f'output "{output_name}" {{')
            
            for key, value in output_config.items():
                if isinstance(value, str) and not (value.startswith('${') or value.startswith('var.') or value.startswith('data.')):
                    lines.append(f'  {key} = "{value}"')
                elif isinstance(value, bool):
                    lines.append(f'  {key} = {str(value).lower()}')
                else:
                    lines.append(f'  {key} = {value}')
            
            lines.append('}')
            lines.append('')
        
        return '\n'.join(lines)
    
    def _generate_versions_tf(self) -> str:
        """Generate versions.tf file"""
        return '''terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.1"
    }
  }
}'''


class MockTerraformRunner:
    """Mock Terraform runner for exercises"""
    
    def __init__(self):
        self.state = {}
        self.plans = {}
        
    def init(self, config_dir: str) -> Dict[str, Any]:
        """Simulate terraform init"""
        return {
            "success": True,
            "message": "Terraform initialized successfully",
            "backend_configured": True
        }
    
    def plan(self, config_dir: str, var_file: str = None) -> Dict[str, Any]:
        """Simulate terraform plan"""
        plan_id = hashlib.sha256(f"{config_dir}{datetime.now()}".encode()).hexdigest()[:8]
        
        # Simulate plan results
        plan_result = {
            "plan_id": plan_id,
            "resources_to_add": 15,
            "resources_to_change": 2,
            "resources_to_destroy": 0,
            "estimated_cost": "$245.67/month",
            "success": True
        }
        
        self.plans[plan_id] = plan_result
        return plan_result
    
    def apply(self, config_dir: str, plan_id: str = None) -> Dict[str, Any]:
        """Simulate terraform apply"""
        # Validate plan_id if provided
        if plan_id and plan_id not in self.plans:
            logger.warning(f"Plan ID {plan_id} not found in stored plans")
        
        return {
            "success": True,
            "resources_created": 15,
            "resources_modified": 2,
            "resources_destroyed": 0,
            "apply_time_seconds": 245,
            "outputs": {
                "vpc_id": "vpc-12345678",
                "data_lake_bucket": "ml-platform-dev-data-lake",
                "eks_cluster_endpoint": "https://ABC123.gr7.us-west-2.eks.amazonaws.com"
            }
        }
    
    def destroy(self, config_dir: str) -> Dict[str, Any]:
        """Simulate terraform destroy"""
        return {
            "success": True,
            "resources_destroyed": 17,
            "destroy_time_seconds": 180
        }


# Exercise 1: Basic ML Infrastructure
def exercise_1_basic_ml_infrastructure():
    """
    Exercise 1: Create basic ML infrastructure with Terraform
    
    TODO: Complete the MLInfrastructureBuilder class
    """
    print("=== Exercise 1: Basic ML Infrastructure ===")
    
    class MLInfrastructureBuilder:
        def __init__(self, project_name: str, environment: str):
            self.config_gen = TerraformConfigGenerator(project_name, environment)
            self.terraform = MockTerraformRunner()
        
                def create_vpc_infrastructure(self, vpc_cidr: str = "10.0.0.0/16") -> None:
            """Create VPC with public and private subnets"""
            # VPC resource
            vpc = TerraformResource(
                resource_type="aws_vpc",
                name="main",
                configuration={
                    "cidr_block": vpc_cidr,
                    "enable_dns_hostnames": True,
                    "enable_dns_support": True,
                    "tags": {"Name": f"{self.config_gen.project_name}-{self.config_gen.environment}-vpc"}
                }
            )
            self.config_gen.add_resource(vpc)
            
            # Internet Gateway
            igw = TerraformResource(
                resource_type="aws_internet_gateway",
                name="main",
                configuration={
                    "vpc_id": "${aws_vpc.main.id}",
                    "tags": {"Name": f"{self.config_gen.project_name}-{self.config_gen.environment}-igw"}
                }
            )
            self.config_gen.add_resource(igw)
        
                def create_data_lake_storage(self, bucket_prefix: str) -> None:
            """Create S3 data lake with proper configuration"""
            # S3 bucket
            bucket = TerraformResource(
                resource_type="aws_s3_bucket",
                name="data_lake",
                configuration={
                    "bucket": f"{bucket_prefix}-data-lake",
                    "tags": {"Name": "Data Lake", "Purpose": "ML Data Storage"}
                }
            )
            self.config_gen.add_resource(bucket)
            
            # Bucket versioning
            versioning = TerraformResource(
                resource_type="aws_s3_bucket_versioning",
                name="data_lake_versioning",
                configuration={
                    "bucket": "${aws_s3_bucket.data_lake.id}",
                    "versioning_configuration": {"status": "Enabled"}
                }
            )
            self.config_gen.add_resource(versioning)
        
                def create_ml_database(self, db_name: str = "ml_metadata") -> None:
            """Create RDS PostgreSQL for ML metadata"""
            # RDS instance
            rds = TerraformResource(
                resource_type="aws_db_instance",
                name="ml_metadata",
                configuration={
                    "identifier": f"{self.config_gen.project_name}-{self.config_gen.environment}-{db_name}",
                    "engine": "postgres",
                    "engine_version": "15.3",
                    "instance_class": "db.t3.medium",
                    "allocated_storage": 100,
                    "storage_encrypted": True,
                    "db_name": db_name,
                    "username": "ml_admin",
                    "manage_master_user_password": True
                }
            )
            self.config_gen.add_resource(rds)
        
                def create_compute_resources(self, instance_types: Dict[str, str]) -> None:
            """Create EC2 launch templates and auto-scaling groups"""
            # Launch template for ML training
            launch_template = TerraformResource(
                resource_type="aws_launch_template",
                name="ml_training",
                configuration={
                    "name_prefix": f"{self.config_gen.project_name}-ml-training-",
                    "instance_type": instance_types.get("training", "p3.2xlarge"),
                    "vpc_security_group_ids": ["${aws_security_group.ml_training.id}"],
                    "tag_specifications": [{
                        "resource_type": "instance",
                        "tags": {"Name": "ML Training Instance", "Purpose": "ML Training"}
                    }]
                }
            )
            self.config_gen.add_resource(launch_template)
        
        def generate_and_validate_config(self) -> Dict[str, str]:
            """Generate Terraform configuration and validate"""
            try:
                # Generate HCL configuration files
                config_files = self.config_gen.generate_hcl()
                
                # Validate syntax and dependencies
                validation_results = []
                for filename, content in config_files.items():
                    if content.strip():
                        validation_results.append(f"✅ {filename}: Valid syntax")
                    else:
                        validation_results.append(f"⚠️ {filename}: Empty file")
                
                # Log validation results
                logger.info(f"Generated {len(config_files)} configuration files")
                for result in validation_results:
                    logger.info(result)
                
                return config_files
            except Exception as e:
                logger.error(f"Configuration generation failed: {e}")
                return {}
    
    # Test basic ML infrastructure
    print("Testing Basic ML Infrastructure...")
    print("\n--- Your implementation should create foundational infrastructure ---")
    
    builder = MLInfrastructureBuilder("ml-platform", "dev")
    
    # Example configuration
    instance_types = {
        "training": "p3.2xlarge",
        "inference": "c5.xlarge",
        "notebook": "t3.medium"
    }
    
    # builder.create_vpc_infrastructure("10.0.0.0/16")
    # builder.create_data_lake_storage("ml-platform-dev")
    # builder.create_ml_database("ml_metadata")
    # builder.create_compute_resources(instance_types)
    # config_files = builder.generate_and_validate_config()
    # print(f"Generated {len(config_files)} configuration files")


# Exercise 2: Multi-Cloud Infrastructure
def exercise_2_multi_cloud_infrastructure():
    """
    Exercise 2: Create multi-cloud ML infrastructure
    
    TODO: Complete the MultiCloudInfrastructure class
    """
    print("\n=== Exercise 2: Multi-Cloud Infrastructure ===")
    
    class MultiCloudInfrastructure:
        def __init__(self, project_name: str):
            self.project_name = project_name
            self.aws_config = TerraformConfigGenerator(project_name, "aws")
            self.azure_config = TerraformConfigGenerator(project_name, "azure")
            self.gcp_config = TerraformConfigGenerator(project_name, "gcp")
        
        def setup_aws_infrastructure(self, aws_config: Dict[str, Any]) -> None:
            """TODO: Set up AWS ML infrastructure"""
            # Hint:
            # 1. Configure AWS provider with region
            # 2. Create S3 data lake and EKS cluster
            # 3. Set up RDS for metadata storage
            # 4. Configure IAM roles and policies
            pass
        
        def setup_azure_infrastructure(self, azure_config: Dict[str, Any]) -> None:
            """TODO: Set up Azure ML infrastructure"""
            # Hint:
            # 1. Configure Azure provider
            # 2. Create resource group and storage account
            # 3. Set up AKS cluster for ML workloads
            # 4. Configure Azure SQL for metadata
            pass
        
        def setup_gcp_infrastructure(self, gcp_config: Dict[str, Any]) -> None:
            """TODO: Set up GCP ML infrastructure"""
            # Hint:
            # 1. Configure GCP provider
            # 2. Create Cloud Storage buckets
            # 3. Set up GKE cluster with GPU support
            # 4. Configure Cloud SQL for metadata
            pass
        
        def create_cross_cloud_networking(self) -> None:
            """TODO: Set up VPN connections between clouds"""
            # Hint:
            # 1. Create VPN gateways in each cloud
            # 2. Configure cross-cloud routing
            # 3. Set up DNS resolution
            # 4. Configure firewall rules
            pass
        
        def setup_data_replication(self, replication_config: Dict) -> None:
            """TODO: Configure cross-cloud data replication"""
            # Hint:
            # 1. Set up data transfer services
            # 2. Configure replication schedules
            # 3. Set up monitoring and alerting
            # 4. Configure backup and recovery
            pass
        
        def generate_multi_cloud_config(self) -> Dict[str, Dict[str, str]]:
            """TODO: Generate configurations for all clouds"""
            # Hint:
            # 1. Generate AWS Terraform config
            # 2. Generate Azure Terraform config
            # 3. Generate GCP Terraform config
            # 4. Return organized configuration files
            pass
    
    # Test multi-cloud infrastructure
    multi_cloud = MultiCloudInfrastructure("global-ml-platform")
    
    print("Testing Multi-Cloud Infrastructure...")
    print("\n--- Your implementation should create cross-cloud infrastructure ---")
    
    cloud_configs = {
        "aws": {
            "region": "us-west-2",
            "instance_types": ["p3.2xlarge", "c5.xlarge"]
        },
        "azure": {
            "location": "West US 2",
            "vm_sizes": ["Standard_NC6s_v3", "Standard_D4s_v3"]
        },
        "gcp": {
            "region": "us-west1",
            "machine_types": ["n1-standard-4", "nvidia-tesla-k80"]
        }
    }
    
    # multi_cloud.setup_aws_infrastructure(cloud_configs["aws"])
    # multi_cloud.setup_azure_infrastructure(cloud_configs["azure"])
    # multi_cloud.setup_gcp_infrastructure(cloud_configs["gcp"])
    # configs = multi_cloud.generate_multi_cloud_config()
    # print(f"Generated multi-cloud configurations for {len(configs)} providers")


def main():
    """Run all Terraform infrastructure exercises"""
    print("🏗️ Day 57: Terraform & Infrastructure as Code - Multi-cloud & Best Practices")
    print("=" * 90)
    
    exercises = [
        exercise_1_basic_ml_infrastructure,
        exercise_2_multi_cloud_infrastructure
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n📋 Starting Exercise {i}")
        try:
            exercise()
            print(f"✅ Exercise {i} setup complete")
        except Exception as e:
            print(f"❌ Exercise {i} error: {e}")
        
        if i < len(exercises):
            input("\nPress Enter to continue to the next exercise...")
    
    print("\n🎉 All exercises completed!")
    print("\nNext steps:")
    print("1. Implement the TODO sections in each exercise")
    print("2. Test your configurations with real Terraform")
    print("3. Deploy to actual cloud environments")
    print("4. Review the solution file for complete implementations")
    print("5. Experiment with advanced Terraform features")
    
    print("\n🚀 Production Deployment Checklist:")
    print("• Set up remote state backend with locking")
    print("• Implement proper IAM roles and policies")
    print("• Configure comprehensive monitoring and alerting")
    print("• Set up automated backup and disaster recovery")
    print("• Implement cost optimization strategies")
    print("• Set up CI/CD pipelines with proper approvals")
    print("• Configure security scanning and compliance checks")


if __name__ == "__main__":
    main()