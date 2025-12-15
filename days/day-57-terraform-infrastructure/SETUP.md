# Day 57: Terraform & Infrastructure as Code - Setup Guide

## Overview
This guide helps you set up a complete Terraform development environment for multi-cloud ML infrastructure deployment with proper tooling, security, and best practices.

## Prerequisites

### Required Software
- **Terraform** >= 1.0 (Infrastructure as Code)
- **Python** >= 3.8 (Automation scripts)
- **AWS CLI** >= 2.0 (AWS operations)
- **Azure CLI** >= 2.0 (Azure operations)
- **Google Cloud SDK** >= 400.0 (GCP operations)
- **Docker** >= 20.0 (Containerization)
- **Git** >= 2.30 (Version control)

### Cloud Accounts
- AWS account with appropriate permissions
- Azure subscription (optional for multi-cloud)
- Google Cloud project (optional for multi-cloud)

## Installation Steps

### 1. Install Terraform

#### macOS (using Homebrew)
```bash
# Install Terraform
brew tap hashicorp/tap
brew install hashicorp/tap/terraform

# Verify installation
terraform version
```

#### Linux (Ubuntu/Debian)
```bash
# Add HashiCorp GPG key
wget -O- https://apt.releases.hashicorp.com/gpg | gpg --dearmor | sudo tee /usr/share/keyrings/hashicorp-archive-keyring.gpg

# Add HashiCorp repository
echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list

# Install Terraform
sudo apt update && sudo apt install terraform

# Verify installation
terraform version
```

#### Windows (using Chocolatey)
```powershell
# Install Terraform
choco install terraform

# Verify installation
terraform version
```

### 2. Install Cloud CLIs

#### AWS CLI
```bash
# macOS
brew install awscli

# Linux
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Configure AWS CLI
aws configure
```

#### Azure CLI
```bash
# macOS
brew install azure-cli

# Linux
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login to Azure
az login
```

#### Google Cloud SDK
```bash
# macOS
brew install google-cloud-sdk

# Linux
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Initialize gcloud
gcloud init
```

### 3. Install Python Dependencies

```bash
# Create virtual environment
python -m venv terraform-env
source terraform-env/bin/activate  # On Windows: terraform-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import terraform; print('Terraform Python wrapper installed')"
```

### 4. Install Additional Tools

#### Terraform Security Scanner (Checkov)
```bash
# Already included in requirements.txt, but can install separately
pip install checkov

# Verify installation
checkov --version
```

#### Terraform Compliance
```bash
# Already included in requirements.txt
pip install terraform-compliance

# Verify installation
terraform-compliance --version
```

#### TFLint (Terraform Linter)
```bash
# macOS
brew install tflint

# Linux
curl -s https://raw.githubusercontent.com/terraform-linters/tflint/master/install_linux.sh | bash

# Verify installation
tflint --version
```

## Configuration

### 1. AWS Configuration

#### Set up AWS credentials
```bash
# Configure AWS CLI with your credentials
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-west-2"
```

#### Create S3 bucket for Terraform state
```bash
# Create state bucket
aws s3 mb s3://your-terraform-state-bucket --region us-west-2

# Enable versioning
aws s3api put-bucket-versioning \
    --bucket your-terraform-state-bucket \
    --versioning-configuration Status=Enabled

# Create DynamoDB table for state locking
aws dynamodb create-table \
    --table-name terraform-state-locks \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region us-west-2
```

### 2. Azure Configuration (Optional)

```bash
# Login to Azure
az login

# Set subscription
az account set --subscription "your-subscription-id"

# Create resource group for Terraform state
az group create --name terraform-state-rg --location "West US 2"

# Create storage account for Terraform state
az storage account create \
    --name terraformstatestorage \
    --resource-group terraform-state-rg \
    --location "West US 2" \
    --sku Standard_LRS
```

### 3. Google Cloud Configuration (Optional)

```bash
# Initialize gcloud
gcloud init

# Set project
gcloud config set project your-project-id

# Create bucket for Terraform state
gsutil mb gs://your-terraform-state-bucket

# Enable versioning
gsutil versioning set on gs://your-terraform-state-bucket
```

### 4. Environment Variables

Create a `.env` file (copy from `.env.example`):

```bash
# Copy example environment file
cp .env.example .env

# Edit with your values
nano .env
```

Example `.env` content:
```bash
# AWS Configuration
AWS_REGION=us-west-2
AWS_PROFILE=default

# Azure Configuration (Optional)
AZURE_SUBSCRIPTION_ID=your-subscription-id
AZURE_TENANT_ID=your-tenant-id

# Google Cloud Configuration (Optional)
GOOGLE_PROJECT_ID=your-project-id
GOOGLE_REGION=us-west1

# Terraform Configuration
TF_STATE_BUCKET=your-terraform-state-bucket
TF_STATE_DYNAMODB_TABLE=terraform-state-locks

# Project Configuration
PROJECT_NAME=ml-platform
ENVIRONMENT=dev
```

## Terraform Project Structure

### 1. Initialize Terraform Project

```bash
# Create project directory
mkdir terraform-ml-infrastructure
cd terraform-ml-infrastructure

# Initialize Terraform
terraform init

# Create basic structure
mkdir -p {modules/{vpc,data-lake,eks-cluster,rds-database},environments/{dev,staging,prod}}
```

### 2. Recommended Directory Structure

```
terraform-ml-infrastructure/
├── main.tf                     # Main configuration
├── variables.tf                # Input variables
├── outputs.tf                  # Output values
├── versions.tf                 # Provider versions
├── terraform.tfvars           # Variable values
├── .terraform.lock.hcl        # Provider lock file
├── modules/                    # Reusable modules
│   ├── vpc/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── outputs.tf
│   ├── data-lake/
│   ├── eks-cluster/
│   └── rds-database/
├── environments/               # Environment-specific configs
│   ├── dev/
│   │   ├── main.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   └── prod/
└── scripts/                   # Automation scripts
    ├── deploy.sh
    └── destroy.sh
```

## Validation and Testing

### 1. Validate Terraform Configuration

```bash
# Format Terraform files
terraform fmt -recursive

# Validate configuration
terraform validate

# Plan deployment
terraform plan

# Check for security issues
checkov -f main.tf

# Lint configuration
tflint
```

### 2. Run Tests

```bash
# Run Python tests
python -m pytest test_terraform_infrastructure.py -v

# Run Terraform compliance tests
terraform-compliance -f compliance-tests -p terraform-plan.json
```

### 3. Security Scanning

```bash
# Scan with Checkov
checkov -d . --framework terraform

# Scan with Terrascan (if installed)
terrascan scan -t terraform

# Custom security validation
python scripts/security_validation.py
```

## Deployment Workflow

### 1. Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/new-infrastructure

# 2. Make changes to Terraform configuration
# Edit main.tf, variables.tf, etc.

# 3. Format and validate
terraform fmt -recursive
terraform validate

# 4. Plan changes
terraform plan -out=tfplan

# 5. Review plan
terraform show tfplan

# 6. Apply changes (after review)
terraform apply tfplan

# 7. Commit changes
git add .
git commit -m "Add new infrastructure components"
git push origin feature/new-infrastructure
```

### 2. Production Deployment

```bash
# 1. Switch to production workspace
terraform workspace select prod

# 2. Plan with production variables
terraform plan -var-file="environments/prod/terraform.tfvars"

# 3. Apply with approval
terraform apply -var-file="environments/prod/terraform.tfvars"

# 4. Verify deployment
terraform output
```

## Troubleshooting

### Common Issues

#### 1. Provider Authentication
```bash
# Check AWS credentials
aws sts get-caller-identity

# Check Azure login
az account show

# Check Google Cloud authentication
gcloud auth list
```

#### 2. State Lock Issues
```bash
# Force unlock (use with caution)
terraform force-unlock LOCK_ID

# Check DynamoDB table
aws dynamodb scan --table-name terraform-state-locks
```

#### 3. Version Conflicts
```bash
# Upgrade providers
terraform init -upgrade

# Check provider versions
terraform version

# Lock provider versions
terraform providers lock
```

### Debug Mode

```bash
# Enable debug logging
export TF_LOG=DEBUG
export TF_LOG_PATH=terraform.log

# Run Terraform commands
terraform plan

# Check logs
tail -f terraform.log
```

## Best Practices

### 1. State Management
- Always use remote state backends
- Enable state locking with DynamoDB
- Use separate state files for different environments
- Regular state backups

### 2. Security
- Never commit sensitive values to Git
- Use Terraform variables for sensitive data
- Implement least privilege IAM policies
- Regular security scanning with Checkov

### 3. Code Organization
- Use modules for reusable components
- Separate environments with workspaces or directories
- Consistent naming conventions
- Comprehensive documentation

### 4. Version Control
- Pin provider versions
- Use semantic versioning for modules
- Regular dependency updates
- Automated testing in CI/CD

## Next Steps

1. **Complete the exercises** in `exercise.py`
2. **Deploy test infrastructure** in development environment
3. **Set up CI/CD pipeline** for automated deployments
4. **Implement monitoring** and alerting for infrastructure
5. **Create disaster recovery** procedures
6. **Optimize costs** with spot instances and lifecycle policies

## Resources

- [Terraform Documentation](https://www.terraform.io/docs/)
- [AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [Terraform Best Practices](https://www.terraform-best-practices.com/)
- [Infrastructure as Code Patterns](https://infrastructure-as-code.com/)
- [Checkov Security Policies](https://www.checkov.io/5.Policy%20Index/terraform.html)