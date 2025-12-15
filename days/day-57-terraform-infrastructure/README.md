# Day 57: Terraform & Infrastructure as Code - Multi-cloud & Best Practices

## Learning Objectives
By the end of this session, you will be able to:
- Design and implement Infrastructure as Code (IaC) using Terraform for ML and data platforms
- Create multi-cloud infrastructure deployments with consistent patterns and best practices
- Implement advanced Terraform patterns including modules, workspaces, and state management
- Build production-ready infrastructure with proper security, monitoring, and cost optimization
- Deploy complete ML/data infrastructure stacks with automated provisioning and management

## Theory (15 minutes)

### Infrastructure as Code with Terraform

Terraform is the leading Infrastructure as Code (IaC) tool that enables you to define, provision, and manage infrastructure using declarative configuration files. For ML and data platforms, Terraform provides consistent, repeatable, and version-controlled infrastructure deployment across multiple cloud providers.

### Core Terraform Concepts

#### 1. Terraform Configuration Language (HCL)

**Basic Resource Definition**
```hcl
# Configure the AWS Provider
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = var.environment
      Project     = var.project_name
      ManagedBy   = "terraform"
    }
  }
}

# Data Lake S3 Bucket
resource "aws_s3_bucket" "data_lake" {
  bucket = "${var.project_name}-${var.environment}-data-lake"
  
  tags = {
    Name        = "Data Lake"
    Environment = var.environment
    Purpose     = "ML Data Storage"
  }
}

resource "aws_s3_bucket_versioning" "data_lake_versioning" {
  bucket = aws_s3_bucket.data_lake.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "data_lake_encryption" {
  bucket = aws_s3_bucket.data_lake.id

  server_side_encryption_configuration {
    rule {
      apply_server_side_encryption_by_default {
        kms_master_key_id = aws_kms_key.data_lake_key.arn
        sse_algorithm     = "aws:kms"
      }
    }
  }
}
```

#### 2. Variables and Outputs

**Variables Definition**
```hcl
# variables.tf
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-2"
  
  validation {
    condition = can(regex("^[a-z]{2}-[a-z]+-[0-9]$", var.aws_region))
    error_message = "AWS region must be in format like us-west-2."
  }
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "Environment must be dev, staging, or prod."
  }
}

variable "ml_instance_types" {
  description = "EC2 instance types for ML workloads"
  type = object({
    training   = string
    inference  = string
    notebook   = string
  })
  default = {
    training  = "p3.2xlarge"
    inference = "c5.xlarge"
    notebook  = "t3.medium"
  }
}

variable "data_retention_days" {
  description = "Data retention period in days"
  type        = number
  default     = 90
  
  validation {
    condition     = var.data_retention_days >= 30 && var.data_retention_days <= 2555
    error_message = "Data retention must be between 30 and 2555 days."
  }
}
```

**Outputs Definition**
```hcl
# outputs.tf
output "data_lake_bucket_name" {
  description = "Name of the data lake S3 bucket"
  value       = aws_s3_bucket.data_lake.bucket
}

output "ml_cluster_endpoint" {
  description = "EKS cluster endpoint for ML workloads"
  value       = aws_eks_cluster.ml_cluster.endpoint
  sensitive   = true
}

output "database_connection_string" {
  description = "RDS database connection string"
  value       = "postgresql://${aws_db_instance.ml_database.username}:${random_password.db_password.result}@${aws_db_instance.ml_database.endpoint}/${aws_db_instance.ml_database.db_name}"
  sensitive   = true
}

output "infrastructure_summary" {
  description = "Summary of deployed infrastructure"
  value = {
    region           = var.aws_region
    environment      = var.environment
    data_lake_bucket = aws_s3_bucket.data_lake.bucket
    eks_cluster_name = aws_eks_cluster.ml_cluster.name
    vpc_id          = aws_vpc.ml_vpc.id
    private_subnets = aws_subnet.private[*].id
    public_subnets  = aws_subnet.public[*].id
  }
}
```

#### 3. Terraform Modules

**ML Infrastructure Module Structure**
```
modules/
├── ml-platform/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   ├── versions.tf
│   └── README.md
├── data-lake/
│   ├── main.tf
│   ├── variables.tf
│   ├── outputs.tf
│   └── s3.tf
└── kubernetes-cluster/
    ├── main.tf
    ├── variables.tf
    ├── outputs.tf
    ├── eks.tf
    ├── node-groups.tf
    └── addons.tf
```

**ML Platform Module Example**
```hcl
# modules/ml-platform/main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.20"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.10"
    }
  }
}

# Data Lake Module
module "data_lake" {
  source = "../data-lake"
  
  project_name    = var.project_name
  environment     = var.environment
  retention_days  = var.data_retention_days
  
  enable_versioning = true
  enable_encryption = true
  
  lifecycle_rules = [
    {
      id     = "ml_data_lifecycle"
      status = "Enabled"
      
      transition = [
        {
          days          = 30
          storage_class = "STANDARD_IA"
        },
        {
          days          = 90
          storage_class = "GLACIER"
        }
      ]
      
      expiration = {
        days = var.data_retention_days
      }
    }
  ]
}

# Kubernetes Cluster Module
module "ml_cluster" {
  source = "../kubernetes-cluster"
  
  cluster_name    = "${var.project_name}-${var.environment}-ml"
  cluster_version = var.kubernetes_version
  
  vpc_id     = aws_vpc.ml_vpc.id
  subnet_ids = aws_subnet.private[*].id
  
  node_groups = {
    general = {
      instance_types = ["t3.medium", "t3.large"]
      min_size      = 2
      max_size      = 10
      desired_size  = 3
    }
    
    ml_training = {
      instance_types = ["p3.2xlarge", "p3.8xlarge"]
      min_size      = 0
      max_size      = 5
      desired_size  = 0
      
      taints = [
        {
          key    = "nvidia.com/gpu"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
    }
    
    ml_inference = {
      instance_types = ["c5.xlarge", "c5.2xlarge"]
      min_size      = 1
      max_size      = 20
      desired_size  = 2
    }
  }
  
  enable_cluster_autoscaler = true
  enable_aws_load_balancer_controller = true
  enable_nvidia_device_plugin = true
}

# RDS Database for ML Metadata
resource "aws_db_instance" "ml_metadata" {
  identifier = "${var.project_name}-${var.environment}-ml-metadata"
  
  engine         = "postgres"
  engine_version = "15.3"
  instance_class = var.db_instance_class
  
  allocated_storage     = 100
  max_allocated_storage = 1000
  storage_type         = "gp3"
  storage_encrypted    = true
  
  db_name  = "ml_metadata"
  username = "ml_admin"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.ml_metadata.name
  
  backup_retention_period = 7
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = var.environment != "prod"
  deletion_protection = var.environment == "prod"
  
  tags = {
    Name        = "ML Metadata Database"
    Environment = var.environment
    Purpose     = "MLflow and Experiment Tracking"
  }
}
```

### Multi-Cloud Infrastructure Patterns

#### 1. Provider Configuration

**Multi-Cloud Setup**
```hcl
# Configure multiple cloud providers
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region
  alias  = "primary"
}

# Azure Provider
provider "azurerm" {
  features {}
  alias = "secondary"
}

# Google Cloud Provider
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
  alias   = "tertiary"
}

# Multi-cloud data replication
resource "aws_s3_bucket" "primary_data_lake" {
  provider = aws.primary
  bucket   = "${var.project_name}-primary-data-lake"
}

resource "azurerm_storage_account" "secondary_data_lake" {
  provider = azurerm.secondary
  
  name                     = "${var.project_name}secondarydl"
  resource_group_name      = azurerm_resource_group.ml_platform.name
  location                 = azurerm_resource_group.ml_platform.location
  account_tier             = "Standard"
  account_replication_type = "GRS"
}

resource "google_storage_bucket" "tertiary_data_lake" {
  provider = google.tertiary
  
  name     = "${var.project_name}-tertiary-data-lake"
  location = var.gcp_region
  
  versioning {
    enabled = true
  }
}
```

#### 2. Cross-Cloud Networking

**VPC Peering and VPN Setup**
```hcl
# AWS VPC
resource "aws_vpc" "ml_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "${var.project_name}-ml-vpc"
  }
}

# Azure Virtual Network
resource "azurerm_virtual_network" "ml_vnet" {
  provider = azurerm.secondary
  
  name                = "${var.project_name}-ml-vnet"
  address_space       = ["10.1.0.0/16"]
  location            = azurerm_resource_group.ml_platform.location
  resource_group_name = azurerm_resource_group.ml_platform.name
}

# VPN Gateway for cross-cloud connectivity
resource "aws_vpn_gateway" "ml_vpn_gw" {
  vpc_id = aws_vpc.ml_vpc.id
  
  tags = {
    Name = "${var.project_name}-vpn-gateway"
  }
}

resource "azurerm_virtual_network_gateway" "ml_vpn_gw" {
  provider = azurerm.secondary
  
  name                = "${var.project_name}-vpn-gateway"
  location            = azurerm_resource_group.ml_platform.location
  resource_group_name = azurerm_resource_group.ml_platform.name
  
  type     = "Vpn"
  vpn_type = "RouteBased"
  
  active_active = false
  enable_bgp    = false
  sku           = "VpnGw1"
  
  ip_configuration {
    name                          = "vnetGatewayConfig"
    public_ip_address_id          = azurerm_public_ip.vpn_gateway_ip.id
    private_ip_address_allocation = "Dynamic"
    subnet_id                     = azurerm_subnet.gateway_subnet.id
  }
}
```

### Advanced Terraform Patterns

#### 1. Workspaces for Environment Management

**Workspace Configuration**
```hcl
# terraform.tf
terraform {
  backend "s3" {
    bucket         = "terraform-state-bucket"
    key            = "ml-platform/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-locks"
  }
}

# Environment-specific configurations
locals {
  environment_configs = {
    dev = {
      instance_count = 2
      instance_size  = "small"
      enable_monitoring = false
      backup_retention = 7
    }
    
    staging = {
      instance_count = 3
      instance_size  = "medium"
      enable_monitoring = true
      backup_retention = 14
    }
    
    prod = {
      instance_count = 5
      instance_size  = "large"
      enable_monitoring = true
      backup_retention = 30
    }
  }
  
  current_config = local.environment_configs[terraform.workspace]
}

# Use workspace-specific configuration
resource "aws_instance" "ml_workers" {
  count         = local.current_config.instance_count
  instance_type = local.current_config.instance_size == "small" ? "t3.medium" : 
                  local.current_config.instance_size == "medium" ? "c5.xlarge" : "c5.2xlarge"
  
  tags = {
    Name        = "${var.project_name}-worker-${count.index + 1}"
    Environment = terraform.workspace
  }
}
```

#### 2. State Management and Locking

**Remote State Configuration**
```hcl
# backend.tf
terraform {
  backend "s3" {
    bucket         = "ml-platform-terraform-state"
    key            = "infrastructure/terraform.tfstate"
    region         = "us-west-2"
    encrypt        = true
    dynamodb_table = "terraform-state-locks"
    
    # Workspace support
    workspace_key_prefix = "workspaces"
  }
}

# State bucket setup (run once)
resource "aws_s3_bucket" "terraform_state" {
  bucket = "ml-platform-terraform-state"
  
  tags = {
    Name        = "Terraform State"
    Environment = "shared"
  }
}

resource "aws_s3_bucket_versioning" "terraform_state_versioning" {
  bucket = aws_s3_bucket.terraform_state.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "terraform_state_encryption" {
  bucket = aws_s3_bucket.terraform_state.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# DynamoDB table for state locking
resource "aws_dynamodb_table" "terraform_locks" {
  name           = "terraform-state-locks"
  billing_mode   = "PAY_PER_REQUEST"
  hash_key       = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }

  tags = {
    Name        = "Terraform State Locks"
    Environment = "shared"
  }
}
```

#### 3. Data Sources and Remote State

**Cross-Stack Data Sharing**
```hcl
# Data source for existing VPC
data "aws_vpc" "existing" {
  count = var.use_existing_vpc ? 1 : 0
  
  filter {
    name   = "tag:Name"
    values = [var.existing_vpc_name]
  }
}

# Remote state data source
data "terraform_remote_state" "network" {
  backend = "s3"
  
  config = {
    bucket = "ml-platform-terraform-state"
    key    = "network/terraform.tfstate"
    region = "us-west-2"
  }
}

# Use data from remote state
resource "aws_instance" "ml_server" {
  subnet_id = data.terraform_remote_state.network.outputs.private_subnet_ids[0]
  vpc_security_group_ids = [
    data.terraform_remote_state.network.outputs.ml_security_group_id
  ]
  
  # ... other configuration
}

# AWS SSM Parameter data source
data "aws_ssm_parameter" "ml_model_bucket" {
  name = "/ml-platform/${var.environment}/model-bucket-name"
}

# Use parameter in resource
resource "aws_iam_policy" "ml_model_access" {
  name = "ml-model-access-${var.environment}"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = "${data.aws_ssm_parameter.ml_model_bucket.value}/*"
      }
    ]
  })
}
```

### Production Infrastructure Patterns

#### 1. ML Training Infrastructure

**Auto-Scaling Training Cluster**
```hcl
# Launch template for ML training instances
resource "aws_launch_template" "ml_training" {
  name_prefix   = "${var.project_name}-ml-training-"
  image_id      = data.aws_ami.ml_optimized.id
  instance_type = var.training_instance_type
  
  vpc_security_group_ids = [aws_security_group.ml_training.id]
  
  user_data = base64encode(templatefile("${path.module}/scripts/ml-training-init.sh", {
    s3_bucket = aws_s3_bucket.data_lake.bucket
    region    = var.aws_region
  }))
  
  block_device_mappings {
    device_name = "/dev/sda1"
    ebs {
      volume_size = 100
      volume_type = "gp3"
      encrypted   = true
    }
  }
  
  iam_instance_profile {
    name = aws_iam_instance_profile.ml_training.name
  }
  
  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "${var.project_name}-ml-training"
      Environment = var.environment
      Purpose     = "ML Training"
    }
  }
}

# Auto Scaling Group for training instances
resource "aws_autoscaling_group" "ml_training" {
  name                = "${var.project_name}-ml-training-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.ml_training.arn]
  health_check_type   = "ELB"
  
  min_size         = 0
  max_size         = var.max_training_instances
  desired_capacity = 0
  
  launch_template {
    id      = aws_launch_template.ml_training.id
    version = "$Latest"
  }
  
  # Scale based on SQS queue depth
  enabled_metrics = [
    "GroupMinSize",
    "GroupMaxSize",
    "GroupDesiredCapacity",
    "GroupInServiceInstances",
    "GroupTotalInstances"
  ]
  
  tag {
    key                 = "Name"
    value               = "${var.project_name}-ml-training-asg"
    propagate_at_launch = false
  }
}

# CloudWatch metric for scaling
resource "aws_cloudwatch_metric_alarm" "training_queue_depth" {
  alarm_name          = "${var.project_name}-training-queue-depth"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "ApproximateNumberOfMessages"
  namespace           = "AWS/SQS"
  period              = "120"
  statistic           = "Average"
  threshold           = "10"
  alarm_description   = "This metric monitors training queue depth"
  
  dimensions = {
    QueueName = aws_sqs_queue.training_jobs.name
  }
  
  alarm_actions = [aws_autoscaling_policy.scale_up.arn]
}
```

#### 2. Model Serving Infrastructure

**Load-Balanced Model Serving**
```hcl
# Application Load Balancer for model serving
resource "aws_lb" "ml_serving" {
  name               = "${var.project_name}-ml-serving-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id
  
  enable_deletion_protection = var.environment == "prod"
  
  tags = {
    Name        = "${var.project_name}-ml-serving-alb"
    Environment = var.environment
  }
}

# Target group for model serving instances
resource "aws_lb_target_group" "ml_serving" {
  name     = "${var.project_name}-ml-serving-tg"
  port     = 8080
  protocol = "HTTP"
  vpc_id   = aws_vpc.ml_vpc.id
  
  health_check {
    enabled             = true
    healthy_threshold   = 2
    interval            = 30
    matcher             = "200"
    path                = "/health"
    port                = "traffic-port"
    protocol            = "HTTP"
    timeout             = 5
    unhealthy_threshold = 2
  }
  
  tags = {
    Name        = "${var.project_name}-ml-serving-tg"
    Environment = var.environment
  }
}

# ALB listener
resource "aws_lb_listener" "ml_serving" {
  load_balancer_arn = aws_lb.ml_serving.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS-1-2-2017-01"
  certificate_arn   = aws_acm_certificate.ml_serving.arn
  
  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.ml_serving.arn
  }
}

# Auto Scaling for model serving
resource "aws_autoscaling_group" "ml_serving" {
  name                = "${var.project_name}-ml-serving-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  target_group_arns   = [aws_lb_target_group.ml_serving.arn]
  health_check_type   = "ELB"
  
  min_size         = var.min_serving_instances
  max_size         = var.max_serving_instances
  desired_capacity = var.desired_serving_instances
  
  launch_template {
    id      = aws_launch_template.ml_serving.id
    version = "$Latest"
  }
}
```

### Security and Compliance

#### 1. IAM Roles and Policies

**Least Privilege IAM Setup**
```hcl
# ML Training Role
resource "aws_iam_role" "ml_training" {
  name = "${var.project_name}-ml-training-role"
  
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# ML Training Policy
resource "aws_iam_policy" "ml_training" {
  name = "${var.project_name}-ml-training-policy"
  
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:DeleteObject"
        ]
        Resource = [
          "${aws_s3_bucket.data_lake.arn}/training-data/*",
          "${aws_s3_bucket.data_lake.arn}/models/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "s3:ListBucket"
        ]
        Resource = aws_s3_bucket.data_lake.arn
        Condition = {
          StringLike = {
            "s3:prefix" = [
              "training-data/*",
              "models/*"
            ]
          }
        }
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = aws_sqs_queue.training_jobs.arn
      },
      {
        Effect = "Allow"
        Action = [
          "cloudwatch:PutMetricData"
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "cloudwatch:namespace" = "ML/Training"
          }
        }
      }
    ]
  })
}

# Attach policy to role
resource "aws_iam_role_policy_attachment" "ml_training" {
  role       = aws_iam_role.ml_training.name
  policy_arn = aws_iam_policy.ml_training.arn
}
```

#### 2. Network Security

**Security Groups and NACLs**
```hcl
# ML Training Security Group
resource "aws_security_group" "ml_training" {
  name        = "${var.project_name}-ml-training-sg"
  description = "Security group for ML training instances"
  vpc_id      = aws_vpc.ml_vpc.id
  
  # Allow SSH from bastion host only
  ingress {
    from_port       = 22
    to_port         = 22
    protocol        = "tcp"
    security_groups = [aws_security_group.bastion.id]
  }
  
  # Allow communication between training instances
  ingress {
    from_port = 0
    to_port   = 65535
    protocol  = "tcp"
    self      = true
  }
  
  # Allow all outbound traffic
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = {
    Name        = "${var.project_name}-ml-training-sg"
    Environment = var.environment
  }
}

# Database Security Group
resource "aws_security_group" "rds" {
  name        = "${var.project_name}-rds-sg"
  description = "Security group for RDS database"
  vpc_id      = aws_vpc.ml_vpc.id
  
  # Allow PostgreSQL from ML instances only
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [
      aws_security_group.ml_training.id,
      aws_security_group.ml_serving.id
    ]
  }
  
  tags = {
    Name        = "${var.project_name}-rds-sg"
    Environment = var.environment
  }
}
```

### Cost Optimization Strategies

#### 1. Spot Instances for Training

**Spot Instance Configuration**
```hcl
# Mixed instance policy for cost optimization
resource "aws_autoscaling_group" "ml_training_spot" {
  name                = "${var.project_name}-ml-training-spot-asg"
  vpc_zone_identifier = aws_subnet.private[*].id
  
  min_size         = 0
  max_size         = var.max_spot_instances
  desired_capacity = 0
  
  mixed_instances_policy {
    launch_template {
      launch_template_specification {
        launch_template_id = aws_launch_template.ml_training_spot.id
        version           = "$Latest"
      }
      
      override {
        instance_type     = "p3.2xlarge"
        weighted_capacity = "1"
      }
      
      override {
        instance_type     = "p3.8xlarge"
        weighted_capacity = "4"
      }
    }
    
    instances_distribution {
      on_demand_allocation_strategy            = "prioritized"
      on_demand_base_capacity                 = 0
      on_demand_percentage_above_base_capacity = 20
      spot_allocation_strategy                = "diversified"
      spot_instance_pools                     = 3
      spot_max_price                          = var.spot_max_price
    }
  }
  
  tag {
    key                 = "Name"
    value               = "${var.project_name}-ml-training-spot"
    propagate_at_launch = true
  }
}

# Launch template for spot instances
resource "aws_launch_template" "ml_training_spot" {
  name_prefix   = "${var.project_name}-ml-training-spot-"
  image_id      = data.aws_ami.ml_optimized.id
  
  vpc_security_group_ids = [aws_security_group.ml_training.id]
  
  # Spot instance configuration
  instance_market_options {
    market_type = "spot"
    spot_options {
      max_price = var.spot_max_price
    }
  }
  
  user_data = base64encode(templatefile("${path.module}/scripts/spot-training-init.sh", {
    s3_bucket = aws_s3_bucket.data_lake.bucket
    region    = var.aws_region
  }))
}
```

#### 2. Lifecycle Policies

**S3 Lifecycle Management**
```hcl
resource "aws_s3_bucket_lifecycle_configuration" "data_lake_lifecycle" {
  bucket = aws_s3_bucket.data_lake.id
  
  rule {
    id     = "ml_data_lifecycle"
    status = "Enabled"
    
    filter {
      prefix = "training-data/"
    }
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 90
      storage_class = "GLACIER"
    }
    
    transition {
      days          = 365
      storage_class = "DEEP_ARCHIVE"
    }
    
    expiration {
      days = var.data_retention_days
    }
  }
  
  rule {
    id     = "model_artifacts_lifecycle"
    status = "Enabled"
    
    filter {
      prefix = "models/"
    }
    
    transition {
      days          = 7
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 30
      storage_class = "GLACIER"
    }
    
    noncurrent_version_expiration {
      noncurrent_days = 90
    }
  }
}
```

### Why Terraform for ML Infrastructure Matters

1. **Consistency**: Identical infrastructure across environments
2. **Version Control**: Infrastructure changes tracked in Git
3. **Collaboration**: Team-based infrastructure development
4. **Automation**: CI/CD integration for infrastructure deployment
5. **Cost Management**: Predictable and optimized resource provisioning
6. **Compliance**: Auditable infrastructure configurations
7. **Disaster Recovery**: Rapid infrastructure recreation

### Real-world Use Cases

- **Netflix**: Uses Terraform for multi-region ML infrastructure deployment
- **Airbnb**: Manages data platform infrastructure with Terraform modules
- **Spotify**: Deploys ML serving infrastructure across multiple clouds
- **Uber**: Provisions training clusters dynamically with Terraform
- **Pinterest**: Manages feature store infrastructure with IaC patterns

## Exercise (40 minutes)
Complete the hands-on exercises in `exercise.py` to build production-ready Terraform configurations for ML and data infrastructure, including multi-cloud deployments, security configurations, and cost optimization strategies.

## Resources
- [Terraform Documentation](https://www.terraform.io/docs/)
- [AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Azure Provider Documentation](https://registry.terraform.io/providers/hashicorp/azurerm/latest/docs)
- [Google Cloud Provider Documentation](https://registry.terraform.io/providers/hashicorp/google/latest/docs)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
- [Infrastructure as Code Patterns](https://infrastructure-as-code.com/)

## Next Steps
- Complete the Terraform infrastructure exercises
- Review multi-cloud deployment patterns
- Take the quiz to test your understanding
- Move to Day 58: Monitoring & Observability
