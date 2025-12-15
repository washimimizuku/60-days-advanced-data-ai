"""
Day 57: Terraform & Infrastructure as Code - Comprehensive Test Suite
Tests for Terraform configuration generation, validation, and infrastructure management
"""

import pytest
import tempfile
import os
import json
from unittest.mock import Mock, patch, MagicMock
from exercise import (
    TerraformResource, TerraformConfigGenerator, MockTerraformRunner
)
from solution import ProductionTerraformManager


class TestTerraformResource:
    """Test Terraform resource representation"""
    
    def test_terraform_resource_creation(self):
        """Test basic Terraform resource creation"""
        resource = TerraformResource(
            resource_type="aws_s3_bucket",
            name="data_lake",
            configuration={"bucket": "ml-platform-data-lake"},
            dependencies=["aws_kms_key.main"]
        )
        
        assert resource.resource_type == "aws_s3_bucket"
        assert resource.name == "data_lake"
        assert resource.configuration["bucket"] == "ml-platform-data-lake"
        assert "aws_kms_key.main" in resource.dependencies
    
    def test_terraform_resource_default_dependencies(self):
        """Test resource with default empty dependencies"""
        resource = TerraformResource(
            resource_type="aws_vpc",
            name="main",
            configuration={"cidr_block": "10.0.0.0/16"}
        )
        
        assert resource.dependencies == []
    
    def test_terraform_resource_complex_configuration(self):
        """Test resource with complex nested configuration"""
        config = {
            "instance_type": "t3.medium",
            "vpc_security_group_ids": ["${aws_security_group.web.id}"],
            "tags": {
                "Name": "web-server",
                "Environment": "production"
            }
        }
        
        resource = TerraformResource(
            resource_type="aws_instance",
            name="web",
            configuration=config
        )
        
        assert resource.configuration["tags"]["Environment"] == "production"
        assert "${aws_security_group.web.id}" in resource.configuration["vpc_security_group_ids"]


class TestTerraformConfigGenerator:
    """Test Terraform configuration generation"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.generator = TerraformConfigGenerator("test-project", "dev")
    
    def test_config_generator_initialization(self):
        """Test configuration generator initialization"""
        assert self.generator.project_name == "test-project"
        assert self.generator.environment == "dev"
        assert self.generator.resources == {}
        assert self.generator.variables == {}
        assert self.generator.outputs == {}
    
    def test_add_variable(self):
        """Test adding Terraform variables"""
        self.generator.add_variable(
            name="aws_region",
            var_type="string",
            description="AWS region for deployment",
            default="us-west-2",
            validation={
                "condition": "can(regex(\"^[a-z]{2}-[a-z]+-[0-9]$\", var.aws_region))",
                "error_message": "Invalid AWS region format"
            }
        )
        
        var_config = self.generator.variables["aws_region"]
        assert var_config["description"] == "AWS region for deployment"
        assert var_config["type"] == "string"
        assert var_config["default"] == "us-west-2"
        assert "validation" in var_config
    
    def test_add_output(self):
        """Test adding Terraform outputs"""
        self.generator.add_output(
            name="vpc_id",
            value="${aws_vpc.main.id}",
            description="ID of the VPC",
            sensitive=False
        )
        
        output_config = self.generator.outputs["vpc_id"]
        assert output_config["description"] == "ID of the VPC"
        assert output_config["value"] == "${aws_vpc.main.id}"
        assert output_config["sensitive"] is False
    
    def test_add_resource(self):
        """Test adding Terraform resources"""
        resource = TerraformResource(
            resource_type="aws_s3_bucket",
            name="data_lake",
            configuration={"bucket": "test-data-lake"}
        )
        
        self.generator.add_resource(resource)
        
        resource_key = "aws_s3_bucket.data_lake"
        assert resource_key in self.generator.resources
        assert self.generator.resources[resource_key] == resource
    
    def test_generate_hcl_basic(self):
        """Test basic HCL generation"""
        # Add a simple resource
        resource = TerraformResource(
            resource_type="aws_s3_bucket",
            name="test",
            configuration={"bucket": "test-bucket"}
        )
        self.generator.add_resource(resource)
        
        # Add a variable
        self.generator.add_variable(
            name="region",
            var_type="string",
            description="AWS region"
        )
        
        # Add an output
        self.generator.add_output(
            name="bucket_name",
            value="${aws_s3_bucket.test.bucket}",
            description="Bucket name"
        )
        
        files = self.generator.generate_hcl()
        
        assert "main.tf" in files
        assert "variables.tf" in files
        assert "outputs.tf" in files
        assert "versions.tf" in files
    
    def test_format_hcl_value_string(self):
        """Test HCL string value formatting"""
        lines = self.generator._format_hcl_value("name", "test-value")
        assert lines == ['name = "test-value"']
    
    def test_format_hcl_value_terraform_reference(self):
        """Test HCL Terraform reference formatting"""
        lines = self.generator._format_hcl_value("vpc_id", "${aws_vpc.main.id}")
        assert lines == ['vpc_id = ${aws_vpc.main.id}']
    
    def test_format_hcl_value_boolean(self):
        """Test HCL boolean value formatting"""
        lines = self.generator._format_hcl_value("enabled", True)
        assert lines == ['enabled = true']
        
        lines = self.generator._format_hcl_value("disabled", False)
        assert lines == ['disabled = false']
    
    def test_format_hcl_value_list(self):
        """Test HCL list value formatting"""
        lines = self.generator._format_hcl_value("subnets", ["subnet-1", "subnet-2"])
        assert lines == ['subnets = ["subnet-1", "subnet-2"]']
    
    def test_format_hcl_value_dict(self):
        """Test HCL dictionary value formatting"""
        tags = {"Environment": "dev", "Project": "test"}
        lines = self.generator._format_hcl_value("tags", tags)
        
        assert lines[0] == "tags {"
        assert lines[-1] == "}"
        assert any("Environment" in line for line in lines)
        assert any("Project" in line for line in lines)


class TestMockTerraformRunner:
    """Test mock Terraform runner"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.runner = MockTerraformRunner()
    
    def test_terraform_init(self):
        """Test Terraform init simulation"""
        result = self.runner.init("/tmp/terraform")
        
        assert result["success"] is True
        assert "initialized successfully" in result["message"]
        assert result["backend_configured"] is True
    
    def test_terraform_plan(self):
        """Test Terraform plan simulation"""
        result = self.runner.plan("/tmp/terraform")
        
        assert result["success"] is True
        assert "plan_id" in result
        assert result["resources_to_add"] > 0
        assert "estimated_cost" in result
        
        # Check plan is stored
        plan_id = result["plan_id"]
        assert plan_id in self.runner.plans
    
    def test_terraform_apply(self):
        """Test Terraform apply simulation"""
        result = self.runner.apply("/tmp/terraform")
        
        assert result["success"] is True
        assert result["resources_created"] > 0
        assert "outputs" in result
        assert "vpc_id" in result["outputs"]
    
    def test_terraform_destroy(self):
        """Test Terraform destroy simulation"""
        result = self.runner.destroy("/tmp/terraform")
        
        assert result["success"] is True
        assert result["resources_destroyed"] > 0
        assert "destroy_time_seconds" in result


class TestProductionTerraformManager:
    """Test production Terraform manager"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.manager = ProductionTerraformManager("ml-platform", "dev")
    
    def test_manager_initialization(self):
        """Test production manager initialization"""
        assert self.manager.project_name == "ml-platform"
        assert self.manager.environment == "dev"
        assert self.manager.modules == {}
        assert self.manager.resources == {}
    
    def test_create_ml_platform_infrastructure(self):
        """Test ML platform infrastructure creation"""
        config = self.manager.create_ml_platform_infrastructure()
        
        assert "main.tf" in config
        assert "variables.tf" in config
        assert "outputs.tf" in config
        
        # Check main.tf contains expected components
        main_tf = config["main.tf"]
        assert "terraform {" in main_tf
        assert "provider \"aws\"" in main_tf
        assert "module \"vpc\"" in main_tf
        assert "module \"data_lake\"" in main_tf
        assert "module \"eks_cluster\"" in main_tf
    
    def test_create_vpc_module(self):
        """Test VPC module creation"""
        vpc_config = self.manager.create_vpc_module()
        
        assert "main.tf" in vpc_config
        assert "variables.tf" in vpc_config
        assert "outputs.tf" in vpc_config
        
        # Check VPC configuration
        main_tf = vpc_config["main.tf"]
        assert "resource \"aws_vpc\" \"main\"" in main_tf
        assert "resource \"aws_subnet\" \"public\"" in main_tf
        assert "resource \"aws_subnet\" \"private\"" in main_tf
        assert "resource \"aws_nat_gateway\" \"main\"" in main_tf


class TestTerraformValidation:
    """Test Terraform configuration validation"""
    
    def test_variable_validation_aws_region(self):
        """Test AWS region validation"""
        generator = TerraformConfigGenerator("test", "dev")
        
        # Valid region format
        generator.add_variable(
            name="aws_region",
            var_type="string",
            description="AWS region",
            validation={
                "condition": "can(regex(\"^[a-z]{2}-[a-z]+-[0-9]$\", var.aws_region))",
                "error_message": "Invalid region format"
            }
        )
        
        var_config = generator.variables["aws_region"]
        assert "validation" in var_config
        assert "condition" in var_config["validation"]
    
    def test_environment_validation(self):
        """Test environment validation"""
        generator = TerraformConfigGenerator("test", "dev")
        
        generator.add_variable(
            name="environment",
            var_type="string",
            description="Environment name",
            validation={
                "condition": "contains([\"dev\", \"staging\", \"prod\"], var.environment)",
                "error_message": "Environment must be dev, staging, or prod"
            }
        )
        
        var_config = generator.variables["environment"]
        assert "validation" in var_config
        assert "contains" in var_config["validation"]["condition"]
    
    def test_resource_naming_convention(self):
        """Test resource naming conventions"""
        generator = TerraformConfigGenerator("ml-platform", "dev")
        
        resource = TerraformResource(
            resource_type="aws_s3_bucket",
            name="data_lake",
            configuration={
                "bucket": "${var.project_name}-${var.environment}-data-lake"
            }
        )
        
        generator.add_resource(resource)
        
        # Check resource naming follows convention
        resource_key = "aws_s3_bucket.data_lake"
        stored_resource = generator.resources[resource_key]
        bucket_name = stored_resource.configuration["bucket"]
        assert "${var.project_name}" in bucket_name
        assert "${var.environment}" in bucket_name


class TestMultiCloudConfiguration:
    """Test multi-cloud configuration patterns"""
    
    def test_multi_provider_configuration(self):
        """Test multiple provider configuration"""
        providers = {
            "aws": {
                "source": "hashicorp/aws",
                "version": "~> 5.0",
                "alias": "primary"
            },
            "azurerm": {
                "source": "hashicorp/azurerm",
                "version": "~> 3.0",
                "alias": "secondary"
            },
            "google": {
                "source": "hashicorp/google",
                "version": "~> 4.0",
                "alias": "tertiary"
            }
        }
        
        # Verify provider configuration structure
        for provider, config in providers.items():
            assert "source" in config
            assert "version" in config
            assert "alias" in config
    
    def test_cross_cloud_resource_references(self):
        """Test cross-cloud resource references"""
        # AWS S3 bucket
        aws_bucket = TerraformResource(
            resource_type="aws_s3_bucket",
            name="primary_data_lake",
            configuration={
                "bucket": "ml-platform-primary-data-lake",
                "provider": "aws.primary"
            }
        )
        
        # Azure Storage Account
        azure_storage = TerraformResource(
            resource_type="azurerm_storage_account",
            name="secondary_data_lake",
            configuration={
                "name": "mlplatformsecondarydl",
                "provider": "azurerm.secondary"
            }
        )
        
        assert aws_bucket.configuration["provider"] == "aws.primary"
        assert azure_storage.configuration["provider"] == "azurerm.secondary"


class TestSecurityConfiguration:
    """Test security configuration patterns"""
    
    def test_kms_encryption_configuration(self):
        """Test KMS encryption setup"""
        generator = TerraformConfigGenerator("secure-ml", "prod")
        
        kms_key = TerraformResource(
            resource_type="aws_kms_key",
            name="ml_platform",
            configuration={
                "description": "KMS key for ML platform encryption",
                "deletion_window_in_days": 30
            }
        )
        
        generator.add_resource(kms_key)
        
        # Verify KMS configuration
        stored_key = generator.resources["aws_kms_key.ml_platform"]
        assert stored_key.configuration["description"] == "KMS key for ML platform encryption"
        assert stored_key.configuration["deletion_window_in_days"] == 30
    
    def test_iam_role_configuration(self):
        """Test IAM role configuration"""
        iam_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:PutObject"],
                    "Resource": "arn:aws:s3:::ml-platform-data/*"
                }
            ]
        }
        
        iam_role = TerraformResource(
            resource_type="aws_iam_role",
            name="ml_training",
            configuration={
                "name": "ml-training-role",
                "assume_role_policy": iam_policy
            }
        )
        
        assert iam_role.configuration["assume_role_policy"]["Version"] == "2012-10-17"
        assert len(iam_role.configuration["assume_role_policy"]["Statement"]) == 1
    
    def test_security_group_configuration(self):
        """Test security group configuration"""
        security_group = TerraformResource(
            resource_type="aws_security_group",
            name="ml_training",
            configuration={
                "name": "ml-training-sg",
                "description": "Security group for ML training instances",
                "ingress": [
                    {
                        "from_port": 22,
                        "to_port": 22,
                        "protocol": "tcp",
                        "security_groups": ["${aws_security_group.bastion.id}"]
                    }
                ],
                "egress": [
                    {
                        "from_port": 0,
                        "to_port": 0,
                        "protocol": "-1",
                        "cidr_blocks": ["0.0.0.0/0"]
                    }
                ]
            }
        )
        
        ingress_rules = security_group.configuration["ingress"]
        assert len(ingress_rules) == 1
        assert ingress_rules[0]["from_port"] == 22
        assert ingress_rules[0]["protocol"] == "tcp"


class TestCostOptimization:
    """Test cost optimization patterns"""
    
    def test_spot_instance_configuration(self):
        """Test spot instance configuration"""
        spot_config = {
            "mixed_instances_policy": {
                "instances_distribution": {
                    "on_demand_percentage_above_base_capacity": 20,
                    "spot_allocation_strategy": "diversified",
                    "spot_instance_pools": 3,
                    "spot_max_price": "1.00"
                }
            }
        }
        
        asg = TerraformResource(
            resource_type="aws_autoscaling_group",
            name="ml_training_spot",
            configuration=spot_config
        )
        
        instances_dist = asg.configuration["mixed_instances_policy"]["instances_distribution"]
        assert instances_dist["on_demand_percentage_above_base_capacity"] == 20
        assert instances_dist["spot_allocation_strategy"] == "diversified"
    
    def test_lifecycle_policy_configuration(self):
        """Test S3 lifecycle policy configuration"""
        lifecycle_rules = [
            {
                "id": "ml_data_lifecycle",
                "status": "Enabled",
                "transition": [
                    {"days": 30, "storage_class": "STANDARD_IA"},
                    {"days": 90, "storage_class": "GLACIER"},
                    {"days": 365, "storage_class": "DEEP_ARCHIVE"}
                ],
                "expiration": {"days": 2555}
            }
        ]
        
        bucket_lifecycle = TerraformResource(
            resource_type="aws_s3_bucket_lifecycle_configuration",
            name="data_lake_lifecycle",
            configuration={"rule": lifecycle_rules}
        )
        
        rules = bucket_lifecycle.configuration["rule"]
        assert len(rules) == 1
        assert rules[0]["id"] == "ml_data_lifecycle"
        assert len(rules[0]["transition"]) == 3


def test_terraform_configuration_integration():
    """Integration test for complete Terraform configuration"""
    generator = TerraformConfigGenerator("integration-test", "dev")
    
    # Add variables
    generator.add_variable("aws_region", "string", "AWS region", "us-west-2")
    generator.add_variable("environment", "string", "Environment", "dev")
    
    # Add VPC resource
    vpc = TerraformResource(
        resource_type="aws_vpc",
        name="main",
        configuration={
            "cidr_block": "10.0.0.0/16",
            "enable_dns_hostnames": True,
            "enable_dns_support": True
        }
    )
    generator.add_resource(vpc)
    
    # Add S3 bucket
    bucket = TerraformResource(
        resource_type="aws_s3_bucket",
        name="data_lake",
        configuration={
            "bucket": "${var.project_name}-${var.environment}-data-lake"
        },
        dependencies=["aws_vpc.main"]
    )
    generator.add_resource(bucket)
    
    # Add outputs
    generator.add_output("vpc_id", "${aws_vpc.main.id}", "VPC ID")
    generator.add_output("bucket_name", "${aws_s3_bucket.data_lake.bucket}", "Bucket name")
    
    # Generate configuration
    config = generator.generate_hcl()
    
    # Verify all files are generated
    assert len(config) == 4
    assert all(file in config for file in ["main.tf", "variables.tf", "outputs.tf", "versions.tf"])
    
    # Verify content structure
    assert "resource \"aws_vpc\" \"main\"" in config["main.tf"]
    assert "resource \"aws_s3_bucket\" \"data_lake\"" in config["main.tf"]
    assert "variable \"aws_region\"" in config["variables.tf"]
    assert "output \"vpc_id\"" in config["outputs.tf"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])