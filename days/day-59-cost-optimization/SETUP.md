# Day 59: Cost Optimization - Setup Guide

## Overview
This guide helps you set up a comprehensive cost optimization and FinOps environment for ML systems, including AWS Cost Explorer, billing APIs, automated optimization tools, and cost governance frameworks.

## Prerequisites

### Required Software
- **Python** >= 3.8 (Development environment)
- **AWS CLI** >= 2.0 (AWS operations and billing)
- **Docker** >= 20.0 (Optional, for containerized tools)
- **Terraform** >= 1.0 (Optional, for infrastructure cost management)

### Required Access and Permissions
- **AWS Account** with billing and cost management permissions
- **Cost Explorer** enabled in AWS account
- **IAM Permissions** for cost and billing APIs
- **CloudWatch** access for metrics and monitoring

### System Requirements
- **Memory**: 4GB+ RAM
- **Storage**: 10GB+ free disk space
- **Network**: Internet access for cloud APIs

## Installation Steps

### 1. Install Python Dependencies

```bash
# Create virtual environment
python -m venv cost-optimization-env
source cost-optimization-env/bin/activate  # On Windows: cost-optimization-env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import boto3; print('AWS SDK installed')"
python -c "import pandas; print('Data analysis tools installed')"
```

### 2. Configure AWS Access

#### Set up AWS CLI and Credentials
```bash
# Configure AWS CLI
aws configure

# Or set environment variables
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-west-2"

# Verify access
aws sts get-caller-identity
```

#### Required IAM Permissions
Create an IAM policy with the following permissions:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "ce:GetCostAndUsage",
                "ce:GetUsageReport",
                "ce:GetReservationCoverage",
                "ce:GetReservationPurchaseRecommendation",
                "ce:GetReservationUtilization",
                "ce:GetRightsizingRecommendation",
                "ce:ListCostCategoryDefinitions",
                "ce:GetCostCategories",
                "budgets:ViewBudget",
                "budgets:ModifyBudget",
                "cur:DescribeReportDefinitions"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "ec2:DescribeInstances",
                "ec2:DescribeInstanceTypes",
                "ec2:DescribeSpotPriceHistory",
                "ec2:DescribeReservedInstances",
                "cloudwatch:GetMetricStatistics",
                "cloudwatch:ListMetrics"
            ],
            "Resource": "*"
        },
        {
            "Effect": "Allow",
            "Action": [
                "organizations:ListAccounts",
                "organizations:DescribeOrganization"
            ],
            "Resource": "*"
        }
    ]
}
```

### 3. Enable AWS Cost Explorer

#### Enable Cost Explorer in AWS Console
1. Navigate to AWS Cost Management Console
2. Go to Cost Explorer
3. Click "Enable Cost Explorer"
4. Wait for data to populate (24-48 hours for initial data)

#### Set up Cost and Usage Reports (CUR)
```bash
# Create S3 bucket for CUR reports
aws s3 mb s3://your-company-cost-reports --region us-east-1

# Create CUR report definition
aws cur put-report-definition \
    --report-definition '{
        "ReportName": "ml-cost-optimization-report",
        "TimeUnit": "DAILY",
        "Format": "textORcsv",
        "Compression": "GZIP",
        "AdditionalSchemaElements": ["RESOURCES"],
        "S3Bucket": "your-company-cost-reports",
        "S3Prefix": "cost-reports/",
        "S3Region": "us-east-1",
        "AdditionalArtifacts": ["REDSHIFT", "ATHENA"],
        "RefreshClosedReports": true,
        "ReportVersioning": "OVERWRITE_REPORT"
    }'
```

### 4. Set up Cost Tagging Strategy

#### Implement Comprehensive Tagging
Create `tagging-policy.json`:
```json
{
    "TaggingPolicy": {
        "RequiredTags": {
            "CostCenter": {
                "AllowedValues": ["ml-platform", "data-engineering", "research"]
            },
            "Team": {
                "AllowedValues": ["fraud-detection", "recommendations", "nlp", "cv"]
            },
            "Environment": {
                "AllowedValues": ["dev", "staging", "prod"]
            },
            "WorkloadType": {
                "AllowedValues": ["training", "serving", "data-pipeline", "development"]
            },
            "Project": {
                "ValidationRegex": "^[a-z][a-z0-9-]*[a-z0-9]$"
            }
        },
        "OptionalTags": {
            "ModelName": {},
            "ExperimentId": {},
            "SpotEligible": {
                "AllowedValues": ["true", "false"]
            },
            "ScheduleEligible": {
                "AllowedValues": ["true", "false"]
            }
        }
    }
}
```

#### Apply Tags to Existing Resources
```bash
# Tag EC2 instances
aws ec2 create-tags \
    --resources i-1234567890abcdef0 \
    --tags Key=CostCenter,Value=ml-platform \
           Key=Team,Value=fraud-detection \
           Key=Environment,Value=prod \
           Key=WorkloadType,Value=training

# Tag S3 buckets
aws s3api put-bucket-tagging \
    --bucket ml-training-data \
    --tagging 'TagSet=[
        {Key=CostCenter,Value=ml-platform},
        {Key=Team,Value=fraud-detection},
        {Key=Environment,Value=prod}
    ]'
```

### 5. Set up Cost Monitoring and Alerting

#### Create CloudWatch Cost Alarms
```bash
# Create billing alarm
aws cloudwatch put-metric-alarm \
    --alarm-name "ML-Monthly-Cost-Alarm" \
    --alarm-description "Alert when ML costs exceed monthly budget" \
    --metric-name EstimatedCharges \
    --namespace AWS/Billing \
    --statistic Maximum \
    --period 86400 \
    --threshold 5000 \
    --comparison-operator GreaterThanThreshold \
    --dimensions Name=Currency,Value=USD \
    --evaluation-periods 1 \
    --alarm-actions arn:aws:sns:us-west-2:123456789012:cost-alerts
```

#### Set up AWS Budgets
Create `budget-config.json`:
```json
{
    "BudgetName": "ML-Platform-Monthly-Budget",
    "BudgetLimit": {
        "Amount": "5000",
        "Unit": "USD"
    },
    "TimeUnit": "MONTHLY",
    "TimePeriod": {
        "Start": "2024-01-01T00:00:00Z",
        "End": "2087-06-15T00:00:00Z"
    },
    "BudgetType": "COST",
    "CostFilters": {
        "TagKey": ["Team"],
        "TagValue": ["ml-platform"]
    }
}
```

```bash
# Create budget
aws budgets create-budget \
    --account-id 123456789012 \
    --budget file://budget-config.json \
    --notifications-with-subscribers '[
        {
            "Notification": {
                "NotificationType": "ACTUAL",
                "ComparisonOperator": "GREATER_THAN",
                "Threshold": 80
            },
            "Subscribers": [
                {
                    "SubscriptionType": "EMAIL",
                    "Address": "finops@company.com"
                }
            ]
        }
    ]'
```

### 6. Configure Cost Optimization Tools

#### Set up Automated Right-Sizing
Create `rightsizing-config.py`:
```python
import boto3
from datetime import datetime, timedelta

def setup_rightsizing_analysis():
    """Set up automated right-sizing analysis"""
    ce_client = boto3.client('ce')
    
    # Get right-sizing recommendations
    response = ce_client.get_rightsizing_recommendation(
        Filter={
            'Dimensions': {
                'Key': 'SERVICE',
                'Values': ['Amazon Elastic Compute Cloud - Compute']
            }
        },
        Configuration={
            'BenefitsConsidered': True,
            'RecommendationTarget': 'SAME_INSTANCE_FAMILY'
        }
    )
    
    return response['RightsizingRecommendations']

if __name__ == "__main__":
    recommendations = setup_rightsizing_analysis()
    print(f"Found {len(recommendations)} right-sizing opportunities")
```

#### Configure Spot Instance Optimization
Create `spot-optimization.py`:
```python
import boto3
from datetime import datetime, timedelta

class SpotOptimizer:
    def __init__(self, region='us-west-2'):
        self.ec2_client = boto3.client('ec2', region_name=region)
        self.region = region
    
    def analyze_spot_opportunities(self):
        """Analyze spot instance opportunities"""
        # Get current on-demand instances
        instances = self.ec2_client.describe_instances(
            Filters=[
                {'Name': 'instance-state-name', 'Values': ['running']},
                {'Name': 'tag:SpotEligible', 'Values': ['true']}
            ]
        )
        
        opportunities = []
        for reservation in instances['Reservations']:
            for instance in reservation['Instances']:
                instance_type = instance['InstanceType']
                
                # Get spot price history
                spot_prices = self.ec2_client.describe_spot_price_history(
                    InstanceTypes=[instance_type],
                    ProductDescriptions=['Linux/UNIX'],
                    StartTime=datetime.now() - timedelta(days=7),
                    EndTime=datetime.now()
                )
                
                if spot_prices['SpotPriceHistory']:
                    avg_spot_price = sum(
                        float(price['SpotPrice']) 
                        for price in spot_prices['SpotPriceHistory']
                    ) / len(spot_prices['SpotPriceHistory'])
                    
                    opportunities.append({
                        'instance_id': instance['InstanceId'],
                        'instance_type': instance_type,
                        'avg_spot_price': avg_spot_price,
                        'potential_savings': self._calculate_savings(instance_type, avg_spot_price)
                    })
        
        return opportunities
    
    def _calculate_savings(self, instance_type, spot_price):
        """Calculate potential savings from spot instances"""
        # Simplified on-demand pricing
        on_demand_prices = {
            'c5.xlarge': 0.17,
            'c5.2xlarge': 0.34,
            'm5.xlarge': 0.192,
            'p3.2xlarge': 3.06
        }
        
        on_demand_price = on_demand_prices.get(instance_type, 0.20)
        monthly_hours = 720
        
        on_demand_monthly = on_demand_price * monthly_hours
        spot_monthly = spot_price * monthly_hours
        
        return {
            'monthly_savings': on_demand_monthly - spot_monthly,
            'savings_percentage': ((on_demand_monthly - spot_monthly) / on_demand_monthly) * 100
        }

# Usage
optimizer = SpotOptimizer()
opportunities = optimizer.analyze_spot_opportunities()
print(f"Found {len(opportunities)} spot optimization opportunities")
```

### 7. Set up Automated Scheduling

#### Create Lambda Function for Resource Scheduling
Create `resource-scheduler.py`:
```python
import boto3
import json
from datetime import datetime

def lambda_handler(event, context):
    """Lambda function for automated resource scheduling"""
    ec2_client = boto3.client('ec2')
    action = event.get('action', 'stop')
    
    # Find instances tagged for scheduling
    instances = ec2_client.describe_instances(
        Filters=[
            {'Name': 'tag:ScheduleEligible', 'Values': ['true']},
            {'Name': 'tag:Environment', 'Values': ['dev', 'staging']},
            {'Name': 'instance-state-name', 'Values': ['running' if action == 'stop' else 'stopped']}
        ]
    )
    
    instance_ids = []
    for reservation in instances['Reservations']:
        for instance in reservation['Instances']:
            instance_ids.append(instance['InstanceId'])
    
    if instance_ids:
        if action == 'stop':
            response = ec2_client.stop_instances(InstanceIds=instance_ids)
        elif action == 'start':
            response = ec2_client.start_instances(InstanceIds=instance_ids)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'action': action,
                'instances_affected': len(instance_ids),
                'instance_ids': instance_ids
            })
        }
    
    return {
        'statusCode': 200,
        'body': json.dumps({'message': 'No instances found for scheduling'})
    }
```

#### Deploy Scheduling Lambda
```bash
# Create deployment package
zip resource-scheduler.zip resource-scheduler.py

# Create Lambda function
aws lambda create-function \
    --function-name ml-resource-scheduler \
    --runtime python3.9 \
    --role arn:aws:iam::123456789012:role/lambda-execution-role \
    --handler resource-scheduler.lambda_handler \
    --zip-file fileb://resource-scheduler.zip \
    --description "Automated ML resource scheduling for cost optimization"

# Create CloudWatch Events rule for scheduling
aws events put-rule \
    --name ml-dev-shutdown \
    --schedule-expression "cron(0 19 * * MON-FRI)" \
    --description "Shutdown dev resources at 7 PM weekdays"

# Add Lambda target to rule
aws events put-targets \
    --rule ml-dev-shutdown \
    --targets "Id"="1","Arn"="arn:aws:lambda:us-west-2:123456789012:function:ml-resource-scheduler","Input"='{"action":"stop"}'
```

### 8. Set up Cost Reporting and Dashboards

#### Create Cost Analysis Dashboard
Create `cost-dashboard.py`:
```python
import boto3
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class CostDashboard:
    def __init__(self):
        self.ce_client = boto3.client('ce')
    
    def generate_cost_report(self, days=30):
        """Generate comprehensive cost report"""
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        # Get cost and usage data
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='DAILY',
            Metrics=['BlendedCost'],
            GroupBy=[
                {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                {'Type': 'TAG', 'Key': 'Team'}
            ]
        )
        
        # Process data for visualization
        cost_data = []
        for result in response['ResultsByTime']:
            date = result['TimePeriod']['Start']
            for group in result['Groups']:
                service = group['Keys'][0] if group['Keys'][0] else 'Unknown'
                team = group['Keys'][1] if len(group['Keys']) > 1 else 'Untagged'
                cost = float(group['Metrics']['BlendedCost']['Amount'])
                
                cost_data.append({
                    'date': date,
                    'service': service,
                    'team': team,
                    'cost': cost
                })
        
        df = pd.DataFrame(cost_data)
        return df
    
    def create_visualizations(self, df):
        """Create cost visualization charts"""
        # Daily cost trend
        daily_costs = df.groupby('date')['cost'].sum()
        
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Daily cost trend
        plt.subplot(2, 2, 1)
        daily_costs.plot(kind='line')
        plt.title('Daily Cost Trend')
        plt.xlabel('Date')
        plt.ylabel('Cost ($)')
        
        # Plot 2: Cost by service
        plt.subplot(2, 2, 2)
        service_costs = df.groupby('service')['cost'].sum().sort_values(ascending=False)
        service_costs.head(10).plot(kind='bar')
        plt.title('Top 10 Services by Cost')
        plt.xlabel('Service')
        plt.ylabel('Cost ($)')
        plt.xticks(rotation=45)
        
        # Plot 3: Cost by team
        plt.subplot(2, 2, 3)
        team_costs = df.groupby('team')['cost'].sum()
        team_costs.plot(kind='pie', autopct='%1.1f%%')
        plt.title('Cost Distribution by Team')
        
        # Plot 4: Service cost trend
        plt.subplot(2, 2, 4)
        top_services = df.groupby('service')['cost'].sum().nlargest(5).index
        for service in top_services:
            service_daily = df[df['service'] == service].groupby('date')['cost'].sum()
            service_daily.plot(label=service)
        plt.title('Top Services Cost Trend')
        plt.xlabel('Date')
        plt.ylabel('Cost ($)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('cost-analysis-dashboard.png', dpi=300, bbox_inches='tight')
        plt.show()

# Usage
dashboard = CostDashboard()
cost_df = dashboard.generate_cost_report(30)
dashboard.create_visualizations(cost_df)
```

### 9. Validation and Testing

#### Test Cost Analysis Tools
Create `test_cost_tools.py`:
```python
import boto3
from datetime import datetime, timedelta

def test_cost_explorer_access():
    """Test Cost Explorer API access"""
    try:
        ce_client = boto3.client('ce')
        
        # Test basic cost query
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        
        response = ce_client.get_cost_and_usage(
            TimePeriod={'Start': start_date, 'End': end_date},
            Granularity='DAILY',
            Metrics=['BlendedCost']
        )
        
        print("✅ Cost Explorer access successful")
        print(f"Retrieved {len(response['ResultsByTime'])} days of cost data")
        return True
        
    except Exception as e:
        print(f"❌ Cost Explorer access failed: {e}")
        return False

def test_ec2_pricing_access():
    """Test EC2 pricing and instance access"""
    try:
        ec2_client = boto3.client('ec2')
        
        # Test instance description
        instances = ec2_client.describe_instances(MaxResults=5)
        print(f"✅ EC2 access successful - found {len(instances['Reservations'])} reservations")
        
        # Test spot price history
        spot_prices = ec2_client.describe_spot_price_history(
            InstanceTypes=['c5.xlarge'],
            MaxResults=5
        )
        print(f"✅ Spot price access successful - {len(spot_prices['SpotPriceHistory'])} price points")
        return True
        
    except Exception as e:
        print(f"❌ EC2 access failed: {e}")
        return False

def run_all_tests():
    """Run all cost optimization tests"""
    print("🧪 Testing Cost Optimization Setup")
    print("=" * 40)
    
    tests = [
        test_cost_explorer_access,
        test_ec2_pricing_access
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Test Results: {success_rate:.0f}% passed")
    
    if success_rate == 100:
        print("✅ All tests passed - setup is ready!")
    else:
        print("⚠️ Some tests failed - check configuration")

if __name__ == "__main__":
    run_all_tests()
```

Run the tests:
```bash
python test_cost_tools.py
```

### 10. Production Deployment

#### Set up Continuous Cost Monitoring
Create `cost-monitoring-pipeline.yml` for CI/CD:
```yaml
name: Cost Monitoring Pipeline

on:
  schedule:
    - cron: '0 8 * * *'  # Daily at 8 AM
  workflow_dispatch:

jobs:
  cost-analysis:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
      
      - name: Run cost analysis
        env:
          AWS_ACCESS_KEY_ID: ${{ secrets.AWS_ACCESS_KEY_ID }}
          AWS_SECRET_ACCESS_KEY: ${{ secrets.AWS_SECRET_ACCESS_KEY }}
        run: |
          python cost-analysis.py
          python generate-cost-report.py
      
      - name: Upload reports
        uses: actions/upload-artifact@v3
        with:
          name: cost-reports
          path: reports/
```

#### Configure Production Monitoring
```bash
# Set up production cost monitoring
aws events put-rule \
    --name daily-cost-analysis \
    --schedule-expression "cron(0 8 * * *)" \
    --description "Daily cost analysis and reporting"

# Create SNS topic for cost alerts
aws sns create-topic --name cost-optimization-alerts

# Subscribe to alerts
aws sns subscribe \
    --topic-arn arn:aws:sns:us-west-2:123456789012:cost-optimization-alerts \
    --protocol email \
    --notification-endpoint finops@company.com
```

## Troubleshooting

### Common Issues

#### 1. Cost Explorer Data Not Available
```bash
# Check if Cost Explorer is enabled
aws ce get-cost-and-usage \
    --time-period Start=2024-01-01,End=2024-01-02 \
    --granularity DAILY \
    --metrics BlendedCost

# If error: Enable Cost Explorer in AWS Console
# Wait 24-48 hours for data to populate
```

#### 2. Insufficient IAM Permissions
```bash
# Test permissions
aws ce get-cost-and-usage --time-period Start=2024-01-01,End=2024-01-02 --granularity DAILY --metrics BlendedCost
aws ec2 describe-instances --max-items 1
aws budgets describe-budgets --account-id $(aws sts get-caller-identity --query Account --output text) --max-results 1
```

#### 3. Missing Cost Allocation Tags
```bash
# Audit existing tags
aws resourcegroupstaggingapi get-resources \
    --resource-type-filters EC2:Instance \
    --tag-filters Key=CostCenter

# Apply missing tags
aws ec2 create-tags \
    --resources $(aws ec2 describe-instances --query 'Reservations[].Instances[].InstanceId' --output text) \
    --tags Key=CostCenter,Value=ml-platform
```

## Next Steps

1. **Complete the exercises** in `exercise.py`
2. **Set up real cost monitoring** with AWS Cost Explorer
3. **Implement automated optimization** strategies
4. **Create cost governance** policies and procedures
5. **Establish FinOps culture** within your organization
6. **Monitor and iterate** on cost optimization strategies

## Resources

- [AWS Cost Management Documentation](https://docs.aws.amazon.com/cost-management/)
- [FinOps Foundation](https://www.finops.org/)
- [AWS Well-Architected Cost Optimization Pillar](https://docs.aws.amazon.com/wellarchitected/latest/cost-optimization-pillar/)
- [Cloud Cost Optimization Best Practices](https://aws.amazon.com/aws-cost-management/aws-cost-optimization/)
- [Kubernetes Cost Optimization](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/#cost-optimization)