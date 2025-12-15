# Day 59: Cost Optimization - Resource Management & FinOps

## Learning Objectives
By the end of this session, you will be able to:
- Implement comprehensive cost optimization strategies for ML and data infrastructure
- Design and deploy FinOps practices for cloud resource management and cost governance
- Build automated cost monitoring, alerting, and optimization systems
- Optimize ML training and inference costs through resource scheduling and right-sizing
- Create cost-aware architectures that balance performance, reliability, and efficiency

## Theory (15 minutes)

### Cost Optimization for ML & Data Systems

Cost optimization in ML and data systems requires a holistic approach that balances performance, reliability, and cost efficiency. Modern cloud-native ML workloads can consume significant resources, making cost optimization critical for sustainable operations.

### FinOps Framework for ML Systems

#### 1. Cost Visibility and Allocation

**Cloud Cost Tagging Strategy**
```yaml
# Comprehensive tagging strategy for cost allocation
cost_allocation_tags:
  # Organizational tags
  business_unit: "data-science" | "engineering" | "product"
  cost_center: "ml-platform" | "data-engineering" | "research"
  team: "fraud-detection" | "recommendations" | "nlp"
  
  # Technical tags
  environment: "dev" | "staging" | "prod"
  service: "model-training" | "model-serving" | "data-pipeline"
  workload_type: "batch" | "streaming" | "interactive"
  
  # ML-specific tags
  model_name: "fraud-detector-v2" | "recommendation-engine"
  experiment_id: "exp-2024-001"
  training_job_id: "train-job-12345"
  
  # Cost optimization tags
  optimization_candidate: "true" | "false"
  spot_eligible: "true" | "false"
  schedule_eligible: "true" | "false"
```

**Cost Allocation Dashboard**
```python
import boto3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any

class MLCostAnalyzer:
    """Analyze and allocate ML infrastructure costs"""
    
    def __init__(self, aws_region: str = "us-west-2"):
        self.ce_client = boto3.client('ce', region_name=aws_region)
        self.ec2_client = boto3.client('ec2', region_name=aws_region)
        
    def get_cost_by_service(self, start_date: str, end_date: str, 
                           group_by: List[str] = None) -> Dict[str, Any]:
        """Get costs grouped by AWS service and custom dimensions"""
        
        group_by = group_by or ['SERVICE', 'LINKED_ACCOUNT']
        
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['BlendedCost', 'UsageQuantity'],
            GroupBy=[{'Type': 'DIMENSION', 'Key': key} for key in group_by]
        )
        
        return response
    
    def get_ml_workload_costs(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Get costs specifically for ML workloads"""
        
        # Filter by ML-related tags
        response = self.ce_client.get_cost_and_usage(
            TimePeriod={
                'Start': start_date,
                'End': end_date
            },
            Granularity='DAILY',
            Metrics=['BlendedCost'],
            GroupBy=[
                {'Type': 'TAG', 'Key': 'workload_type'},
                {'Type': 'TAG', 'Key': 'model_name'},
                {'Type': 'DIMENSION', 'Key': 'SERVICE'}
            ],
            Filter={
                'Tags': {
                    'Key': 'service',
                    'Values': ['model-training', 'model-serving', 'data-pipeline']
                }
            }
        )
        
        # Convert to DataFrame for analysis
        cost_data = []
        for result in response['ResultsByTime']:
            date = result['TimePeriod']['Start']
            for group in result['Groups']:
                cost_data.append({
                    'date': date,
                    'workload_type': group['Keys'][0] if group['Keys'][0] else 'untagged',
                    'model_name': group['Keys'][1] if len(group['Keys']) > 1 else 'unknown',
                    'service': group['Keys'][2] if len(group['Keys']) > 2 else 'unknown',
                    'cost': float(group['Metrics']['BlendedCost']['Amount'])
                })
        
        return pd.DataFrame(cost_data)
    
    def analyze_cost_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze cost trends and identify optimization opportunities"""
        
        df['date'] = pd.to_datetime(df['date'])
        
        # Calculate daily cost trends
        daily_costs = df.groupby('date')['cost'].sum().reset_index()
        daily_costs['cost_change'] = daily_costs['cost'].pct_change()
        
        # Identify top cost drivers
        top_services = df.groupby('service')['cost'].sum().sort_values(ascending=False)
        top_models = df.groupby('model_name')['cost'].sum().sort_values(ascending=False)
        
        # Calculate cost per model prediction (if metrics available)
        cost_efficiency = {}
        for model in df['model_name'].unique():
            model_cost = df[df['model_name'] == model]['cost'].sum()
            # In production, would get prediction count from metrics
            estimated_predictions = 1000000  # Placeholder
            cost_efficiency[model] = model_cost / estimated_predictions
        
        return {
            'total_cost': df['cost'].sum(),
            'daily_average': daily_costs['cost'].mean(),
            'cost_trend': daily_costs['cost_change'].mean(),
            'top_services': top_services.to_dict(),
            'top_models': top_models.to_dict(),
            'cost_per_prediction': cost_efficiency,
            'optimization_candidates': [
                model for model, cost in top_models.items() 
                if cost > top_models.mean()
            ]
        }

# Example usage
cost_analyzer = MLCostAnalyzer()

# Analyze last 30 days
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')

ml_costs = cost_analyzer.get_ml_workload_costs(start_date, end_date)
cost_analysis = cost_analyzer.analyze_cost_trends(ml_costs)

print(f"Total ML costs (30 days): ${cost_analysis['total_cost']:.2f}")
print(f"Daily average: ${cost_analysis['daily_average']:.2f}")
print(f"Top cost driver: {list(cost_analysis['top_services'].keys())[0]}")
```

#### 2. Resource Right-Sizing

**ML Instance Optimization**
```python
import boto3
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class InstanceRecommendation:
    current_instance: str
    recommended_instance: str
    current_cost: float
    recommended_cost: float
    savings_potential: float
    confidence_score: float
    reasoning: str

class MLResourceOptimizer:
    """Optimize ML resource allocation and instance types"""
    
    def __init__(self):
        self.ec2_client = boto3.client('ec2')
        self.cloudwatch = boto3.client('cloudwatch')
        
        # ML workload instance type mappings
        self.ml_instance_families = {
            'training': {
                'cpu_intensive': ['c5.large', 'c5.xlarge', 'c5.2xlarge', 'c5.4xlarge'],
                'memory_intensive': ['r5.large', 'r5.xlarge', 'r5.2xlarge', 'r5.4xlarge'],
                'gpu_training': ['p3.2xlarge', 'p3.8xlarge', 'p3.16xlarge', 'p4d.24xlarge'],
                'gpu_inference': ['g4dn.xlarge', 'g4dn.2xlarge', 'g4dn.4xlarge']
            },
            'serving': {
                'low_latency': ['c5n.large', 'c5n.xlarge', 'c5n.2xlarge'],
                'high_throughput': ['m5.large', 'm5.xlarge', 'm5.2xlarge'],
                'gpu_inference': ['g4dn.xlarge', 'g4dn.2xlarge', 'inf1.xlarge']
            }
        }
    
    def analyze_instance_utilization(self, instance_id: str, 
                                   days: int = 14) -> Dict[str, float]:
        """Analyze instance utilization metrics"""
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        metrics = {}
        
        # Get CPU utilization
        cpu_response = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='CPUUtilization',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour periods
            Statistics=['Average', 'Maximum']
        )
        
        if cpu_response['Datapoints']:
            cpu_values = [dp['Average'] for dp in cpu_response['Datapoints']]
            metrics['cpu_avg'] = sum(cpu_values) / len(cpu_values)
            metrics['cpu_max'] = max([dp['Maximum'] for dp in cpu_response['Datapoints']])
        
        # Get memory utilization (requires CloudWatch agent)
        try:
            memory_response = self.cloudwatch.get_metric_statistics(
                Namespace='CWAgent',
                MetricName='mem_used_percent',
                Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
                StartTime=start_time,
                EndTime=end_time,
                Period=3600,
                Statistics=['Average', 'Maximum']
            )
            
            if memory_response['Datapoints']:
                memory_values = [dp['Average'] for dp in memory_response['Datapoints']]
                metrics['memory_avg'] = sum(memory_values) / len(memory_values)
                metrics['memory_max'] = max([dp['Maximum'] for dp in memory_response['Datapoints']])
        except:
            metrics['memory_avg'] = 50.0  # Default assumption
            metrics['memory_max'] = 70.0
        
        # Get network utilization
        network_in = self.cloudwatch.get_metric_statistics(
            Namespace='AWS/EC2',
            MetricName='NetworkIn',
            Dimensions=[{'Name': 'InstanceId', 'Value': instance_id}],
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,
            Statistics=['Average']
        )
        
        if network_in['Datapoints']:
            network_values = [dp['Average'] for dp in network_in['Datapoints']]
            metrics['network_avg'] = sum(network_values) / len(network_values)
        
        return metrics
    
    def recommend_instance_type(self, current_instance: str, 
                               utilization: Dict[str, float],
                               workload_type: str = 'training') -> InstanceRecommendation:
        """Recommend optimal instance type based on utilization"""
        
        # Get current instance pricing (simplified)
        current_cost = self._get_instance_hourly_cost(current_instance)
        
        # Determine workload characteristics
        cpu_avg = utilization.get('cpu_avg', 50.0)
        memory_avg = utilization.get('memory_avg', 50.0)
        
        # Recommendation logic
        if cpu_avg < 20 and memory_avg < 30:
            # Over-provisioned - recommend smaller instance
            recommended = self._get_smaller_instance(current_instance, workload_type)
            reasoning = "Low utilization detected - downsizing recommended"
            confidence = 0.9
        elif cpu_avg > 80 or memory_avg > 85:
            # Under-provisioned - recommend larger instance
            recommended = self._get_larger_instance(current_instance, workload_type)
            reasoning = "High utilization detected - upsizing recommended"
            confidence = 0.8
        elif workload_type == 'training' and 'gpu' not in current_instance.lower():
            # Training workload without GPU - check if GPU would be beneficial
            recommended = self._recommend_gpu_instance(current_instance)
            reasoning = "GPU acceleration may improve training performance"
            confidence = 0.6
        else:
            # Current instance is appropriately sized
            recommended = current_instance
            reasoning = "Current instance type is well-suited for workload"
            confidence = 0.95
        
        recommended_cost = self._get_instance_hourly_cost(recommended)
        savings_potential = (current_cost - recommended_cost) / current_cost * 100
        
        return InstanceRecommendation(
            current_instance=current_instance,
            recommended_instance=recommended,
            current_cost=current_cost,
            recommended_cost=recommended_cost,
            savings_potential=savings_potential,
            confidence_score=confidence,
            reasoning=reasoning
        )
    
    def _get_instance_hourly_cost(self, instance_type: str) -> float:
        """Get hourly cost for instance type (simplified pricing)"""
        # Simplified pricing - in production, use AWS Pricing API
        pricing_map = {
            'c5.large': 0.085, 'c5.xlarge': 0.17, 'c5.2xlarge': 0.34,
            'm5.large': 0.096, 'm5.xlarge': 0.192, 'm5.2xlarge': 0.384,
            'r5.large': 0.126, 'r5.xlarge': 0.252, 'r5.2xlarge': 0.504,
            'p3.2xlarge': 3.06, 'p3.8xlarge': 12.24, 'p3.16xlarge': 24.48,
            'g4dn.xlarge': 0.526, 'g4dn.2xlarge': 0.752, 'g4dn.4xlarge': 1.204
        }
        return pricing_map.get(instance_type, 0.10)
    
    def _get_smaller_instance(self, current: str, workload_type: str) -> str:
        """Get smaller instance type recommendation"""
        # Simplified logic - in production, use comprehensive instance family mapping
        size_map = {
            '2xlarge': 'xlarge', 'xlarge': 'large', 'large': 'medium',
            '4xlarge': '2xlarge', '8xlarge': '4xlarge', '16xlarge': '8xlarge'
        }
        
        for size, smaller in size_map.items():
            if size in current:
                return current.replace(size, smaller)
        
        return current
    
    def _get_larger_instance(self, current: str, workload_type: str) -> str:
        """Get larger instance type recommendation"""
        size_map = {
            'medium': 'large', 'large': 'xlarge', 'xlarge': '2xlarge',
            '2xlarge': '4xlarge', '4xlarge': '8xlarge', '8xlarge': '16xlarge'
        }
        
        for size, larger in size_map.items():
            if size in current:
                return current.replace(size, larger)
        
        return current
    
    def _recommend_gpu_instance(self, current: str) -> str:
        """Recommend GPU instance for training workloads"""
        # Map CPU instances to appropriate GPU instances
        if 'c5' in current or 'm5' in current:
            return 'g4dn.xlarge'  # General GPU instance
        elif 'r5' in current:
            return 'p3.2xlarge'   # Training-optimized GPU
        else:
            return 'g4dn.xlarge'
```

#### 3. Spot Instance Optimization

**Intelligent Spot Instance Management**
```python
import boto3
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import json

class SpotInstanceOptimizer:
    """Optimize ML workloads using spot instances"""
    
    def __init__(self, region: str = 'us-west-2'):
        self.ec2_client = boto3.client('ec2', region_name=region)
        self.region = region
        
    def get_spot_price_history(self, instance_types: List[str], 
                              availability_zones: List[str] = None,
                              days: int = 7) -> Dict[str, List]:
        """Get spot price history for analysis"""
        
        if not availability_zones:
            availability_zones = [f"{self.region}a", f"{self.region}b", f"{self.region}c"]
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(days=days)
        
        spot_prices = {}
        
        for instance_type in instance_types:
            spot_prices[instance_type] = []
            
            response = self.ec2_client.describe_spot_price_history(
                InstanceTypes=[instance_type],
                ProductDescriptions=['Linux/UNIX'],
                AvailabilityZones=availability_zones,
                StartTime=start_time,
                EndTime=end_time
            )
            
            for price_point in response['SpotPriceHistory']:
                spot_prices[instance_type].append({
                    'timestamp': price_point['Timestamp'],
                    'price': float(price_point['SpotPrice']),
                    'availability_zone': price_point['AvailabilityZone']
                })
        
        return spot_prices
    
    def analyze_spot_reliability(self, instance_type: str, 
                               availability_zone: str) -> Dict[str, float]:
        """Analyze spot instance reliability and interruption patterns"""
        
        # Get interruption frequency (simplified - in production use Spot Instance Advisor API)
        interruption_rates = {
            'p3.2xlarge': 0.15,   # 15% interruption rate
            'p3.8xlarge': 0.25,   # 25% interruption rate
            'g4dn.xlarge': 0.08,  # 8% interruption rate
            'c5.xlarge': 0.05,    # 5% interruption rate
            'm5.xlarge': 0.03     # 3% interruption rate
        }
        
        base_rate = interruption_rates.get(instance_type, 0.10)
        
        # Adjust based on availability zone (some AZs are more stable)
        az_multiplier = {
            f"{self.region}a": 1.0,
            f"{self.region}b": 0.8,
            f"{self.region}c": 1.2
        }
        
        adjusted_rate = base_rate * az_multiplier.get(availability_zone, 1.0)
        
        return {
            'interruption_rate': adjusted_rate,
            'reliability_score': 1.0 - adjusted_rate,
            'recommended_for_training': adjusted_rate < 0.20,
            'recommended_for_serving': adjusted_rate < 0.05
        }
    
    def create_spot_fleet_config(self, workload_config: Dict) -> Dict:
        """Create optimized spot fleet configuration"""
        
        target_capacity = workload_config.get('target_capacity', 2)
        instance_types = workload_config.get('instance_types', ['c5.xlarge', 'm5.xlarge'])
        max_price = workload_config.get('max_price_per_hour', 0.50)
        
        # Diversification strategy
        launch_specifications = []
        
        for instance_type in instance_types:
            for az in [f"{self.region}a", f"{self.region}b", f"{self.region}c"]:
                reliability = self.analyze_spot_reliability(instance_type, az)
                
                if reliability['recommended_for_training']:
                    launch_spec = {
                        'ImageId': 'ami-0abcdef1234567890',  # ML-optimized AMI
                        'InstanceType': instance_type,
                        'SubnetId': f"subnet-{az}",
                        'SecurityGroups': [{'GroupId': 'sg-ml-training'}],
                        'UserData': self._get_user_data_script(workload_config),
                        'WeightedCapacity': 1.0,
                        'SpotPrice': str(max_price)
                    }
                    launch_specifications.append(launch_spec)
        
        spot_fleet_config = {
            'SpotFleetRequestConfig': {
                'IamFleetRole': 'arn:aws:iam::123456789012:role/aws-ec2-spot-fleet-role',
                'AllocationStrategy': 'diversified',
                'TargetCapacity': target_capacity,
                'SpotPrice': str(max_price),
                'LaunchSpecifications': launch_specifications,
                'TerminateInstancesWithExpiration': True,
                'Type': 'maintain',
                'ReplaceUnhealthyInstances': True,
                'InstanceInterruptionBehavior': 'terminate'
            }
        }
        
        return spot_fleet_config
    
    def _get_user_data_script(self, workload_config: Dict) -> str:
        """Generate user data script for ML workload initialization"""
        
        script = f"""#!/bin/bash
# ML Training Instance Initialization
yum update -y
yum install -y docker

# Start Docker
service docker start
usermod -a -G docker ec2-user

# Pull ML training image
docker pull {workload_config.get('training_image', 'tensorflow/tensorflow:latest-gpu')}

# Set up training environment
mkdir -p /opt/ml/input/data
mkdir -p /opt/ml/output/model

# Download training data
aws s3 sync {workload_config.get('data_s3_path', 's3://ml-data/training')} /opt/ml/input/data/

# Start training job
docker run --gpus all \\
  -v /opt/ml/input/data:/data \\
  -v /opt/ml/output/model:/model \\
  -e TRAINING_JOB_NAME={workload_config.get('job_name', 'ml-training')} \\
  {workload_config.get('training_image', 'tensorflow/tensorflow:latest-gpu')} \\
  python train.py

# Upload results
aws s3 sync /opt/ml/output/model {workload_config.get('output_s3_path', 's3://ml-models/output')}/

# Signal completion and terminate
/opt/aws/bin/cfn-signal -e $? --stack {workload_config.get('stack_name', 'ml-training')} --resource AutoScalingGroup --region {self.region}
"""
        
        return script
    
    def calculate_spot_savings(self, instance_type: str, 
                             hours_per_month: float = 720) -> Dict[str, float]:
        """Calculate potential savings from using spot instances"""
        
        # Get current on-demand pricing
        on_demand_price = self._get_on_demand_price(instance_type)
        
        # Get average spot price
        spot_history = self.get_spot_price_history([instance_type], days=30)
        if spot_history[instance_type]:
            avg_spot_price = sum(p['price'] for p in spot_history[instance_type]) / len(spot_history[instance_type])
        else:
            avg_spot_price = on_demand_price * 0.3  # Assume 70% discount
        
        monthly_on_demand = on_demand_price * hours_per_month
        monthly_spot = avg_spot_price * hours_per_month
        
        savings_amount = monthly_on_demand - monthly_spot
        savings_percentage = (savings_amount / monthly_on_demand) * 100
        
        return {
            'on_demand_monthly': monthly_on_demand,
            'spot_monthly': monthly_spot,
            'savings_amount': savings_amount,
            'savings_percentage': savings_percentage,
            'avg_spot_price': avg_spot_price,
            'on_demand_price': on_demand_price
        }
    
    def _get_on_demand_price(self, instance_type: str) -> float:
        """Get on-demand price for instance type"""
        # Simplified pricing - in production, use AWS Pricing API
        pricing_map = {
            'c5.xlarge': 0.17, 'c5.2xlarge': 0.34, 'c5.4xlarge': 0.68,
            'm5.xlarge': 0.192, 'm5.2xlarge': 0.384, 'm5.4xlarge': 0.768,
            'p3.2xlarge': 3.06, 'p3.8xlarge': 12.24, 'p3.16xlarge': 24.48,
            'g4dn.xlarge': 0.526, 'g4dn.2xlarge': 0.752, 'g4dn.4xlarge': 1.204
        }
        return pricing_map.get(instance_type, 0.20)
```

### Automated Cost Optimization

#### 1. Resource Scheduling

**Intelligent Resource Scheduling**
```python
import boto3
from datetime import datetime, timedelta
import json
from typing import Dict, List, Any

class MLResourceScheduler:
    """Automated scheduling for ML resources to optimize costs"""
    
    def __init__(self):
        self.ec2_client = boto3.client('ec2')
        self.lambda_client = boto3.client('lambda')
        self.events_client = boto3.client('events')
        
    def create_training_schedule(self, schedule_config: Dict) -> str:
        """Create automated training schedule"""
        
        # Lambda function for training job management
        lambda_code = f"""
import boto3
import json

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')
    
    action = event.get('action', 'start')
    instance_ids = event.get('instance_ids', [])
    
    if action == 'start':
        # Start training instances
        if instance_ids:
            ec2.start_instances(InstanceIds=instance_ids)
        else:
            # Launch new spot fleet for training
            spot_config = {json.dumps(schedule_config.get('spot_config', {}))}
            response = ec2.request_spot_fleet(SpotFleetRequestConfig=spot_config)
            return {{'spot_fleet_id': response['SpotFleetRequestId']}}
    
    elif action == 'stop':
        # Stop training instances
        if instance_ids:
            ec2.stop_instances(InstanceIds=instance_ids)
        
        # Cancel spot fleet requests
        spot_fleet_id = event.get('spot_fleet_id')
        if spot_fleet_id:
            ec2.cancel_spot_fleet_requests(
                SpotFleetRequestIds=[spot_fleet_id],
                TerminateInstances=True
            )
    
    return {{'status': 'success', 'action': action}}
"""
        
        # Create Lambda function
        function_name = f"ml-training-scheduler-{schedule_config['name']}"
        
        lambda_response = self.lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.9',
            Role='arn:aws:iam::123456789012:role/lambda-execution-role',
            Handler='index.lambda_handler',
            Code={'ZipFile': lambda_code.encode()},
            Description='Automated ML training resource scheduler',
            Timeout=300,
            Tags={
                'Purpose': 'cost-optimization',
                'Component': 'ml-training-scheduler'
            }
        )
        
        # Create CloudWatch Events rules for scheduling
        start_schedule = schedule_config.get('start_schedule', 'cron(0 9 * * MON-FRI)')  # 9 AM weekdays
        stop_schedule = schedule_config.get('stop_schedule', 'cron(0 18 * * MON-FRI)')   # 6 PM weekdays
        
        # Start rule
        self.events_client.put_rule(
            Name=f"{function_name}-start",
            ScheduleExpression=start_schedule,
            Description='Start ML training resources',
            State='ENABLED'
        )
        
        self.events_client.put_targets(
            Rule=f"{function_name}-start",
            Targets=[{
                'Id': '1',
                'Arn': lambda_response['FunctionArn'],
                'Input': json.dumps({
                    'action': 'start',
                    'spot_config': schedule_config.get('spot_config', {})
                })
            }]
        )
        
        # Stop rule
        self.events_client.put_rule(
            Name=f"{function_name}-stop",
            ScheduleExpression=stop_schedule,
            Description='Stop ML training resources',
            State='ENABLED'
        )
        
        self.events_client.put_targets(
            Rule=f"{function_name}-stop",
            Targets=[{
                'Id': '1',
                'Arn': lambda_response['FunctionArn'],
                'Input': json.dumps({'action': 'stop'})
            }]
        )
        
        return function_name
    
    def create_dev_environment_schedule(self, environment_config: Dict) -> str:
        """Create schedule for development environments"""
        
        # Schedule to shut down dev environments outside business hours
        shutdown_schedule = environment_config.get('shutdown_schedule', 'cron(0 19 * * MON-FRI)')  # 7 PM weekdays
        startup_schedule = environment_config.get('startup_schedule', 'cron(0 8 * * MON-FRI)')     # 8 AM weekdays
        
        lambda_code = """
import boto3

def lambda_handler(event, context):
    ec2 = boto3.client('ec2')
    
    action = event.get('action', 'stop')
    
    # Find instances tagged as dev environment
    response = ec2.describe_instances(
        Filters=[
            {'Name': 'tag:Environment', 'Values': ['dev', 'development']},
            {'Name': 'tag:AutoSchedule', 'Values': ['true']},
            {'Name': 'instance-state-name', 'Values': ['running' if action == 'stop' else 'stopped']}
        ]
    )
    
    instance_ids = []
    for reservation in response['Reservations']:
        for instance in reservation['Instances']:
            instance_ids.append(instance['InstanceId'])
    
    if instance_ids:
        if action == 'stop':
            ec2.stop_instances(InstanceIds=instance_ids)
        elif action == 'start':
            ec2.start_instances(InstanceIds=instance_ids)
        
        return {'status': 'success', 'action': action, 'instances': len(instance_ids)}
    
    return {'status': 'no_instances_found'}
"""
        
        function_name = f"dev-environment-scheduler-{environment_config['name']}"
        
        # Create Lambda function
        self.lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.9',
            Role='arn:aws:iam::123456789012:role/lambda-execution-role',
            Handler='index.lambda_handler',
            Code={'ZipFile': lambda_code.encode()},
            Description='Automated dev environment scheduler for cost optimization'
        )
        
        return function_name
```

### Why Cost Optimization Matters

1. **Financial Sustainability**: Reduce infrastructure costs by 30-70% through intelligent optimization
2. **Resource Efficiency**: Maximize utilization and minimize waste in ML workloads
3. **Scalability**: Enable cost-effective scaling of ML operations
4. **Business Value**: Improve ROI on ML investments and infrastructure
5. **Environmental Impact**: Reduce carbon footprint through efficient resource usage
6. **Competitive Advantage**: Lower operational costs enable more experimentation and innovation

### Real-world Use Cases

- **Netflix**: Saves millions annually through spot instance optimization for ML training workloads
- **Airbnb**: Uses intelligent scheduling to reduce development environment costs by 60%
- **Spotify**: Optimizes recommendation model training costs through right-sizing and spot instances
- **Uber**: Implements FinOps practices to track and optimize ML infrastructure costs across teams
- **Pinterest**: Uses automated resource scheduling to reduce non-production costs by 80%

## Exercise (40 minutes)
Complete the hands-on exercises in `exercise.py` to build production-ready cost optimization solutions, including cost analysis, resource right-sizing, spot instance management, and automated scheduling systems.

## Resources
- [AWS Cost Management Documentation](https://docs.aws.amazon.com/cost-management/)
- [FinOps Foundation](https://www.finops.org/)
- [AWS Spot Instance Best Practices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/spot-best-practices.html)
- [Cloud Cost Optimization Guide](https://aws.amazon.com/aws-cost-management/aws-cost-optimization/)
- [Kubernetes Cost Optimization](https://kubernetes.io/docs/concepts/cluster-administration/manage-deployment/#cost-optimization)
- [ML Cost Optimization Strategies](https://ml-ops.org/content/cost-optimization)

## Next Steps
- Complete the cost optimization exercises
- Review FinOps implementation strategies
- Take the quiz to test your understanding
- Move to Day 60: Capstone Project