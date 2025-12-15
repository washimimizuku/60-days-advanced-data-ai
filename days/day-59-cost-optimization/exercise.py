"""
Day 59: Cost Optimization - Resource Management & FinOps
Exercises for building comprehensive cost optimization solutions for ML systems
"""

import time
import json
import random
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CostData:
    """Represents cost data for analysis"""
    date: str
    service: str
    resource_id: str
    cost: float
    usage_quantity: float
    tags: Dict[str, str]


@dataclass
class OptimizationRecommendation:
    """Represents a cost optimization recommendation"""
    resource_id: str
    current_cost: float
    optimized_cost: float
    savings_potential: float
    recommendation_type: str
    confidence_score: float
    implementation_effort: str
    description: str


class MockCostAnalyzer:
    """Mock cost analyzer for exercises"""
    
    def __init__(self):
        self.cost_data = []
        # Remove unused attribute
        self.pricing_data = self._initialize_pricing_data()
    
    # Cost to usage conversion ratio
    COST_TO_USAGE_RATIO = 0.10
    
    def _initialize_pricing_data(self) -> Dict[str, float]:
        """Initialize mock pricing data"""
        return {
            # EC2 On-Demand Pricing (per hour)
            'c5.large': 0.085, 'c5.xlarge': 0.17, 'c5.2xlarge': 0.34,
            'm5.large': 0.096, 'm5.xlarge': 0.192, 'm5.2xlarge': 0.384,
            'r5.large': 0.126, 'r5.xlarge': 0.252, 'r5.2xlarge': 0.504,
            'p3.2xlarge': 3.06, 'p3.8xlarge': 12.24, 'g4dn.xlarge': 0.526,
            
            # Spot Instance Discounts (percentage of on-demand)
            'spot_discount': 0.7,  # 70% discount on average
            
            # Storage Pricing (per GB per month)
            's3_standard': 0.023, 's3_ia': 0.0125, 's3_glacier': 0.004
        }
    
    def generate_mock_cost_data(self, days: int = 30) -> List[CostData]:
        """Generate mock cost data for analysis"""
        cost_data = []
        
        services = ['EC2-Instance', 'S3', 'RDS', 'Lambda', 'CloudWatch']
        workload_types = ['training', 'serving', 'data-pipeline', 'development']
        teams = ['fraud-detection', 'recommendations', 'nlp', 'computer-vision']
        
        for day in range(days):
            date = (datetime.now() - timedelta(days=day)).strftime('%Y-%m-%d')
            
            for service in services:
                for team in teams:
                    for workload in workload_types:
                        # Generate realistic cost variations
                        base_cost = random.uniform(50, 500)
                        
                        # Add day-of-week patterns (lower on weekends)
                        day_of_week = (datetime.now() - timedelta(days=day)).weekday()
                        if day_of_week >= 5:  # Weekend
                            base_cost *= 0.3
                        
                        # Add workload-specific patterns
                        if workload == 'development':
                            base_cost *= 0.2  # Dev environments are smaller
                        elif workload == 'training':
                            base_cost *= random.uniform(0.5, 2.0)  # Variable training loads
                        
                        cost_entry = CostData(
                            date=date,
                            service=service,
                            resource_id=f"{service}-{team}-{workload}-{uuid.uuid4().hex[:8]}",
                            cost=base_cost,
                            usage_quantity=base_cost / self.COST_TO_USAGE_RATIO,  # Mock usage
                            tags={
                                'Team': team,
                                'WorkloadType': workload,
                                'Environment': 'prod' if workload != 'development' else 'dev',
                                'OptimizationCandidate': str(base_cost > 200).lower()
                            }
                        )
                        cost_data.append(cost_entry)
        
        self.cost_data = cost_data
        return cost_data
    
    def query_costs(self, filters: Dict[str, Any] = None) -> List[CostData]:
        """Query cost data with filters"""
        filtered_data = self.cost_data
        
        if filters:
            for key, value in filters.items():
                if key == 'service':
                    filtered_data = [d for d in filtered_data if d.service == value]
                elif key == 'date_range':
                    start_date, end_date = value
                    filtered_data = [d for d in filtered_data if start_date <= d.date <= end_date]
                elif key.startswith('tag:'):
                    tag_key = key[4:]  # Remove 'tag:' prefix
                    filtered_data = [d for d in filtered_data if d.tags.get(tag_key) == value]
        
        return filtered_data


# Exercise 1: Cost Analysis and Allocation
def exercise_1_cost_analysis():
    """
    Exercise 1: Implement comprehensive cost analysis and allocation
    
    TODO: Complete the CostAnalysisManager class
    """
    print("=== Exercise 1: Cost Analysis and Allocation ===")
    
    class CostAnalysisManager:
        def __init__(self, cost_analyzer: MockCostAnalyzer):
            self.cost_analyzer = cost_analyzer
            self.analysis_cache = {}
        
        def analyze_cost_trends(self, days: int = 30) -> Dict[str, Any]:
            """Analyze cost trends and patterns"""
            # Get cost data for specified period
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            cost_data = self.cost_analyzer.query_costs({
                "date_range": (start_date, end_date)
            })
            
            if not cost_data:
                return {"error": "No cost data available for analysis"}
            
            # Calculate daily trends
            daily_costs = defaultdict(float)
            for item in cost_data:
                daily_costs[item.date] += item.cost
            
            # Calculate trend metrics
            costs_list = list(daily_costs.values())
            total_cost = sum(costs_list)
            avg_daily_cost = total_cost / len(costs_list) if costs_list else 0
            
            # Identify cost spikes (costs > 2x average)
            spike_threshold = avg_daily_cost * 2
            cost_spikes = [date for date, cost in daily_costs.items() if cost > spike_threshold]
            
            return {
                "total_cost": total_cost,
                "avg_daily_cost": avg_daily_cost,
                "cost_spikes": cost_spikes,
                "trend_direction": "increasing" if costs_list[-1] > costs_list[0] else "decreasing",
                "analysis_period_days": days
            }
        
        def allocate_costs_by_team(self, period: str = "30d") -> Dict[str, float]:
            """Allocate costs by team using tags"""
            days = int(period.replace('d', ''))
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            
            cost_data = self.cost_analyzer.query_costs({
                "date_range": (start_date, end_date)
            })
            
            # Aggregate costs by team
            team_costs = defaultdict(float)
            for item in cost_data:
                team = item.tags.get("Team", "Untagged")
                team_costs[team] += item.cost
            
            # Calculate percentages
            total_cost = sum(team_costs.values())
            team_percentages = {}
            for team, cost in team_costs.items():
                team_percentages[team] = {
                    "cost": cost,
                    "percentage": (cost / total_cost * 100) if total_cost > 0 else 0
                }
            
            return dict(team_percentages)
        
        def analyze_workload_costs(self) -> Dict[str, Dict[str, float]]:
            """Analyze costs by workload type"""
            cost_data = self.cost_analyzer.cost_data
            
            # Group costs by workload type
            workload_costs = defaultdict(list)
            for item in cost_data:
                workload_type = item.tags.get("WorkloadType", "Unknown")
                workload_costs[workload_type].append(item.cost)
            
            # Calculate metrics for each workload
            workload_analysis = {}
            for workload, costs in workload_costs.items():
                total_cost = sum(costs)
                avg_cost = total_cost / len(costs) if costs else 0
                max_cost = max(costs) if costs else 0
                min_cost = min(costs) if costs else 0
                
                workload_analysis[workload] = {
                    "total_cost": total_cost,
                    "average_cost": avg_cost,
                    "max_cost": max_cost,
                    "min_cost": min_cost,
                    "resource_count": len(costs)
                }
            
            return workload_analysis
        
        def identify_cost_anomalies(self, threshold: float = 2.0) -> List[Dict]:
            """Identify cost anomalies and spikes"""
            cost_data = self.cost_analyzer.cost_data
            
            # Group costs by date
            daily_costs = defaultdict(float)
            for item in cost_data:
                daily_costs[item.date] += item.cost
            
            # Calculate baseline (average cost)
            costs_list = list(daily_costs.values())
            if len(costs_list) < 3:
                return []  # Need at least 3 days for anomaly detection
            
            avg_cost = sum(costs_list) / len(costs_list)
            
            # Identify anomalies
            anomalies = []
            for date, cost in daily_costs.items():
                deviation_ratio = cost / avg_cost if avg_cost > 0 else 0
                
                if deviation_ratio > threshold:
                    severity = "high" if deviation_ratio > threshold * 1.5 else "medium"
                    anomalies.append({
                        "date": date,
                        "cost": cost,
                        "baseline_cost": avg_cost,
                        "deviation_ratio": deviation_ratio,
                        "severity": severity,
                        "anomaly_type": "cost_spike"
                    })
            
            return anomalies
    
    # Test cost analysis
    cost_analyzer = MockCostAnalyzer()
    cost_data = cost_analyzer.generate_mock_cost_data(30)
    
    analysis_manager = CostAnalysisManager(cost_analyzer)
    
    print("Testing Cost Analysis...")
    print(f"Generated {len(cost_data)} cost data points")
    print("\n--- Your implementation should analyze cost trends and patterns ---")
    
    print("Cost analysis simulation completed")


# Exercise 2: Resource Right-Sizing
def exercise_2_resource_rightsizing():
    """
    Exercise 2: Implement intelligent resource right-sizing
    
    TODO: Complete the ResourceRightSizer class
    """
    print("\n=== Exercise 2: Resource Right-Sizing ===")
    
    class ResourceRightSizer:
        def __init__(self, cost_analyzer: MockCostAnalyzer):
            self.cost_analyzer = cost_analyzer
            self.utilization_data = {}
        
        def analyze_resource_utilization(self, resource_id: str, days: int = 14) -> Dict[str, float]:
            """Analyze resource utilization patterns"""
            # Simulate utilization data (in production, would query CloudWatch)
            utilization = {
                "cpu_avg": random.uniform(10, 90),
                "cpu_max": random.uniform(50, 100),
                "memory_avg": random.uniform(15, 85),
                "memory_max": random.uniform(40, 100),
                "gpu_avg": random.uniform(0, 95),
                "network_avg": random.uniform(5, 60)
            }
            
            # Store utilization data
            self.utilization_data[resource_id] = utilization
            
            # Determine provisioning status
            cpu_status = "under" if utilization["cpu_avg"] < 30 else "over" if utilization["cpu_avg"] > 80 else "optimal"
            memory_status = "under" if utilization["memory_avg"] < 30 else "over" if utilization["memory_avg"] > 80 else "optimal"
            
            utilization["cpu_status"] = cpu_status
            utilization["memory_status"] = memory_status
            utilization["analysis_days"] = days
            
            return utilization
        
        def recommend_instance_type(self, current_instance: str, 
                                   utilization: Dict[str, float]) -> OptimizationRecommendation:
            """Recommend optimal instance type"""
            current_cost = self.cost_analyzer.pricing_data.get(current_instance, 0.20) * 720  # Monthly cost
            
            cpu_avg = utilization.get("cpu_avg", 50)
            memory_avg = utilization.get("memory_avg", 50)
            
            # Recommendation logic
            if cpu_avg < 20 and memory_avg < 30:
                # Downsize recommendation
                recommended_instance = self._get_smaller_instance(current_instance)
                recommendation_type = "downsize"
                confidence = 0.9
                description = f"Low utilization detected. Downsize from {current_instance} to {recommended_instance}"
            elif cpu_avg > 80 or memory_avg > 85:
                # Upsize recommendation
                recommended_instance = self._get_larger_instance(current_instance)
                recommendation_type = "upsize"
                confidence = 0.8
                description = f"High utilization detected. Upsize from {current_instance} to {recommended_instance}"
            else:
                # Current instance is optimal
                recommended_instance = current_instance
                recommendation_type = "no_change"
                confidence = 0.95
                description = f"Current instance {current_instance} is optimally sized"
            
            optimized_cost = self.cost_analyzer.pricing_data.get(recommended_instance, 0.20) * 720
            savings_potential = max(0, current_cost - optimized_cost)
            
            return OptimizationRecommendation(
                resource_id=f"resource-{uuid.uuid4().hex[:8]}",
                current_cost=current_cost,
                optimized_cost=optimized_cost,
                savings_potential=savings_potential,
                recommendation_type=recommendation_type,
                confidence_score=confidence,
                implementation_effort="low" if recommendation_type == "no_change" else "medium",
                description=description
            )
        
        def identify_oversized_resources(self, utilization_threshold: float = 30.0) -> List[Dict]:
            """Identify oversized resources for downsizing"""
            oversized_resources = []
            
            # Simulate multiple resources with utilization data
            for i in range(5):
                resource_id = f"i-{uuid.uuid4().hex[:8]}"
                instance_type = random.choice(["c5.xlarge", "c5.2xlarge", "m5.xlarge", "r5.xlarge"])
                
                utilization = self.analyze_resource_utilization(resource_id, 14)
                
                # Check if resource is oversized
                if (utilization["cpu_avg"] < utilization_threshold and 
                    utilization["memory_avg"] < utilization_threshold):
                    
                    current_cost = self.cost_analyzer.pricing_data.get(instance_type, 0.20) * 720
                    smaller_instance = self._get_smaller_instance(instance_type)
                    optimized_cost = self.cost_analyzer.pricing_data.get(smaller_instance, 0.15) * 720
                    
                    oversized_resources.append({
                        "resource_id": resource_id,
                        "current_instance": instance_type,
                        "recommended_instance": smaller_instance,
                        "cpu_utilization": utilization["cpu_avg"],
                        "memory_utilization": utilization["memory_avg"],
                        "current_monthly_cost": current_cost,
                        "optimized_monthly_cost": optimized_cost,
                        "monthly_savings": current_cost - optimized_cost,
                        "risk_level": "low"
                    })
            
            # Sort by savings potential
            oversized_resources.sort(key=lambda x: x["monthly_savings"], reverse=True)
            
            return oversized_resources
    
    # Test resource right-sizing
    cost_analyzer = MockCostAnalyzer()
    rightsizer = ResourceRightSizer(cost_analyzer)
    
    print("Testing Resource Right-Sizing...")
    print("\n--- Your implementation should optimize resource allocation ---")
    
    # Simulate resource utilization data
    mock_utilization = {
        'cpu_avg': random.uniform(15, 85),
        'memory_avg': random.uniform(20, 90),
        'gpu_avg': random.uniform(10, 95),
        'network_avg': random.uniform(5, 50)
    }
    
    print(f"Mock utilization: CPU {mock_utilization['cpu_avg']:.1f}%, Memory {mock_utilization['memory_avg']:.1f}%")
    
    print("Resource right-sizing simulation completed")


def main():
    """Run all cost optimization exercises"""
    print("💰 Day 59: Cost Optimization - Resource Management & FinOps")
    print("=" * 70)
    
    exercises = [
        exercise_1_cost_analysis,
        exercise_2_resource_rightsizing
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
    print("2. Set up real cost monitoring and optimization tools")
    print("3. Deploy cost optimization strategies to actual infrastructure")
    print("4. Review the solution file for complete implementations")
    print("5. Experiment with advanced FinOps practices")
    
    print("\n🚀 Production Deployment Checklist:")
    print("• Implement comprehensive cost tagging strategy")
    print("• Set up automated cost monitoring and alerting")
    print("• Deploy resource right-sizing recommendations")
    print("• Configure spot instance optimization for training")
    print("• Establish FinOps governance and accountability")
    print("• Create cost optimization culture and practices")


if __name__ == "__main__":
    main()