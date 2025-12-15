"""
Day 59: Cost Optimization - Comprehensive Test Suite
Tests for cost optimization, FinOps, and resource management systems
"""

import pytest
import uuid
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from exercise import (
    CostData, OptimizationRecommendation, MockCostAnalyzer
)
from solution import (
    ProductionCostAnalyzer, ResourceOptimizer, SpotInstanceManager, 
    FinOpsManager, CostForecastingEngine
)


class TestCostData:
    """Test CostData dataclass"""
    
    def test_cost_data_creation(self):
        """Test CostData creation and attributes"""
        tags = {"Team": "ml-platform", "Environment": "prod"}
        cost_data = CostData(
            date="2024-01-15",
            service="EC2-Instance",
            resource_id="i-1234567890abcdef0",
            cost=125.50,
            usage_quantity=744.0,
            tags=tags
        )
        
        assert cost_data.date == "2024-01-15"
        assert cost_data.service == "EC2-Instance"
        assert cost_data.cost == 125.50
        assert cost_data.tags["Team"] == "ml-platform"


class TestOptimizationRecommendation:
    """Test OptimizationRecommendation dataclass"""
    
    def test_recommendation_creation(self):
        """Test recommendation creation and calculations"""
        recommendation = OptimizationRecommendation(
            resource_id="i-1234567890abcdef0",
            current_cost=200.0,
            optimized_cost=120.0,
            savings_potential=80.0,
            recommendation_type="right_sizing",
            confidence_score=0.85,
            implementation_effort="low",
            description="Downsize from c5.2xlarge to c5.xlarge"
        )
        
        assert recommendation.savings_potential == 80.0
        assert recommendation.confidence_score == 0.85
        assert recommendation.recommendation_type == "right_sizing"
        
        # Calculate savings percentage
        savings_percentage = (recommendation.savings_potential / recommendation.current_cost) * 100
        assert savings_percentage == 40.0


class TestMockCostAnalyzer:
    """Test mock cost analyzer functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = MockCostAnalyzer()
    
    def test_pricing_data_initialization(self):
        """Test pricing data initialization"""
        pricing = self.analyzer.pricing_data
        
        assert "c5.xlarge" in pricing
        assert "p3.2xlarge" in pricing
        assert pricing["c5.xlarge"] == 0.17
        assert pricing["spot_discount"] == 0.7
    
    def test_generate_mock_cost_data(self):
        """Test mock cost data generation"""
        cost_data = self.analyzer.generate_mock_cost_data(7)
        
        assert len(cost_data) > 0
        assert all(isinstance(item, CostData) for item in cost_data)
        
        # Check data variety
        services = set(item.service for item in cost_data)
        teams = set(item.tags.get("Team") for item in cost_data)
        
        assert len(services) > 1
        assert len(teams) > 1
    
    def test_query_costs_with_filters(self):
        """Test cost querying with filters"""
        # Generate test data
        self.analyzer.generate_mock_cost_data(10)
        
        # Test service filter
        ec2_costs = self.analyzer.query_costs({"service": "EC2-Instance"})
        assert all(item.service == "EC2-Instance" for item in ec2_costs)
        
        # Test tag filter
        prod_costs = self.analyzer.query_costs({"tag:Environment": "prod"})
        assert all(item.tags.get("Environment") == "prod" for item in prod_costs)
    
    def test_date_range_filtering(self):
        """Test date range filtering"""
        self.analyzer.generate_mock_cost_data(30)
        
        start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        filtered_costs = self.analyzer.query_costs({
            "date_range": (start_date, end_date)
        })
        
        assert len(filtered_costs) > 0
        for item in filtered_costs:
            assert start_date <= item.date <= end_date


class TestProductionCostAnalyzer:
    """Test production cost analyzer"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.analyzer = ProductionCostAnalyzer("us-west-2")
    
    @patch('boto3.client')
    def test_cost_analyzer_initialization(self, mock_boto_client):
        """Test cost analyzer initialization"""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        analyzer = ProductionCostAnalyzer("us-east-1")
        assert analyzer.region == "us-east-1"
        mock_boto_client.assert_called_with('ce', region_name="us-east-1")
    
    def test_calculate_cost_trends(self):
        """Test cost trend calculations"""
        # Mock cost data
        cost_data = [
            {"date": "2024-01-01", "cost": 100.0},
            {"date": "2024-01-02", "cost": 110.0},
            {"date": "2024-01-03", "cost": 105.0},
            {"date": "2024-01-04", "cost": 120.0}
        ]
        
        trends = self.analyzer.calculate_cost_trends(cost_data)
        
        assert "total_cost" in trends
        assert "daily_average" in trends
        assert "trend_direction" in trends
        assert trends["total_cost"] == 435.0
        assert trends["daily_average"] == 108.75
    
    def test_identify_cost_anomalies(self):
        """Test cost anomaly detection"""
        # Create data with anomaly
        normal_costs = [100.0] * 10
        anomaly_costs = [100.0] * 8 + [500.0, 100.0]  # One spike
        
        anomalies = self.analyzer.identify_cost_anomalies(anomaly_costs, threshold=2.0)
        
        assert len(anomalies) > 0
        assert any(anomaly["severity"] == "high" for anomaly in anomalies)
    
    def test_cost_allocation_by_tags(self):
        """Test cost allocation by tags"""
        mock_cost_data = [
            CostData("2024-01-01", "EC2", "i-123", 100.0, 744.0, {"Team": "ml-platform"}),
            CostData("2024-01-01", "S3", "bucket-1", 50.0, 1000.0, {"Team": "data-eng"}),
            CostData("2024-01-01", "EC2", "i-456", 75.0, 744.0, {"Team": "ml-platform"})
        ]
        
        allocation = self.analyzer.allocate_costs_by_tag(mock_cost_data, "Team")
        
        assert "ml-platform" in allocation
        assert "data-eng" in allocation
        assert allocation["ml-platform"] == 175.0
        assert allocation["data-eng"] == 50.0


class TestResourceOptimizer:
    """Test resource optimization functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.optimizer = ResourceOptimizer()
    
    def test_analyze_instance_utilization(self):
        """Test instance utilization analysis"""
        # Mock utilization data
        utilization_data = {
            "cpu_avg": 25.0,
            "memory_avg": 40.0,
            "network_avg": 15.0,
            "disk_avg": 30.0
        }
        
        analysis = self.optimizer.analyze_utilization_patterns(utilization_data)
        
        assert "utilization_category" in analysis
        assert "optimization_potential" in analysis
        assert analysis["utilization_category"] in ["low", "medium", "high"]
    
    def test_right_sizing_recommendation(self):
        """Test right-sizing recommendations"""
        current_instance = "c5.2xlarge"
        utilization = {"cpu_avg": 20.0, "memory_avg": 25.0}
        
        recommendation = self.optimizer.recommend_instance_size(
            current_instance, utilization
        )
        
        assert isinstance(recommendation, OptimizationRecommendation)
        assert recommendation.current_cost > 0
        assert recommendation.optimized_cost >= 0
        assert 0 <= recommendation.confidence_score <= 1.0
    
    def test_spot_instance_suitability(self):
        """Test spot instance suitability analysis"""
        workload_characteristics = {
            "fault_tolerant": True,
            "duration_hours": 4.0,
            "checkpointing_enabled": True,
            "priority": "low"
        }
        
        suitability = self.optimizer.analyze_spot_suitability(workload_characteristics)
        
        assert "suitable_for_spot" in suitability
        assert "risk_level" in suitability
        assert "expected_savings" in suitability
        assert isinstance(suitability["suitable_for_spot"], bool)
    
    def test_cost_optimization_recommendations(self):
        """Test comprehensive cost optimization recommendations"""
        resource_data = {
            "instance_type": "c5.2xlarge",
            "utilization": {"cpu_avg": 15.0, "memory_avg": 20.0},
            "workload_type": "training",
            "current_cost": 200.0
        }
        
        recommendations = self.optimizer.generate_optimization_recommendations(resource_data)
        
        assert len(recommendations) > 0
        assert all(isinstance(rec, OptimizationRecommendation) for rec in recommendations)
        
        # Check recommendation types
        rec_types = [rec.recommendation_type for rec in recommendations]
        assert any(rec_type in ["right_sizing", "spot_instance", "scheduling"] for rec_type in rec_types)


class TestSpotInstanceManager:
    """Test spot instance management"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.spot_manager = SpotInstanceManager("us-west-2")
    
    @patch('boto3.client')
    def test_spot_price_analysis(self, mock_boto_client):
        """Test spot price analysis"""
        mock_client = Mock()
        mock_boto_client.return_value = mock_client
        
        # Mock spot price history
        mock_client.describe_spot_price_history.return_value = {
            'SpotPriceHistory': [
                {
                    'Timestamp': datetime.now(),
                    'SpotPrice': '0.05',
                    'InstanceType': 'c5.xlarge',
                    'AvailabilityZone': 'us-west-2a'
                }
            ]
        }
        
        spot_manager = SpotInstanceManager("us-west-2")
        price_analysis = spot_manager.analyze_spot_prices(['c5.xlarge'])
        
        assert 'c5.xlarge' in price_analysis
        assert 'average_price' in price_analysis['c5.xlarge']
        assert 'price_volatility' in price_analysis['c5.xlarge']
    
    def test_interruption_risk_assessment(self):
        """Test spot instance interruption risk assessment"""
        instance_type = "p3.2xlarge"
        availability_zone = "us-west-2a"
        
        risk_assessment = self.spot_manager.assess_interruption_risk(
            instance_type, availability_zone
        )
        
        assert "interruption_rate" in risk_assessment
        assert "risk_level" in risk_assessment
        assert "mitigation_strategies" in risk_assessment
        assert 0 <= risk_assessment["interruption_rate"] <= 1.0
    
    def test_spot_fleet_configuration(self):
        """Test spot fleet configuration generation"""
        fleet_config = {
            "target_capacity": 4,
            "instance_types": ["c5.xlarge", "m5.xlarge"],
            "max_price": 0.20,
            "workload_type": "training"
        }
        
        spot_fleet = self.spot_manager.create_spot_fleet_config(fleet_config)
        
        assert "SpotFleetRequestConfig" in spot_fleet
        assert spot_fleet["SpotFleetRequestConfig"]["TargetCapacity"] == 4
        assert "LaunchSpecifications" in spot_fleet["SpotFleetRequestConfig"]
    
    def test_savings_calculation(self):
        """Test spot instance savings calculation"""
        instance_type = "c5.xlarge"
        hours_per_month = 720
        
        savings = self.spot_manager.calculate_spot_savings(instance_type, hours_per_month)
        
        assert "on_demand_cost" in savings
        assert "spot_cost" in savings
        assert "savings_amount" in savings
        assert "savings_percentage" in savings
        assert savings["savings_percentage"] > 0


class TestFinOpsManager:
    """Test FinOps management functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.finops_manager = FinOpsManager()
    
    def test_cost_governance_rules(self):
        """Test cost governance rule creation"""
        governance_rules = {
            "max_monthly_spend": 10000.0,
            "require_approval_above": 1000.0,
            "auto_shutdown_dev": True,
            "spot_instance_minimum": 0.7
        }
        
        rules = self.finops_manager.create_governance_rules(governance_rules)
        
        assert "budget_alerts" in rules
        assert "approval_workflows" in rules
        assert "automation_policies" in rules
        assert len(rules["budget_alerts"]) > 0
    
    def test_cost_attribution_model(self):
        """Test cost attribution model"""
        attribution_config = {
            "primary_dimension": "Team",
            "secondary_dimension": "Environment",
            "allocation_method": "proportional"
        }
        
        model = self.finops_manager.create_attribution_model(attribution_config)
        
        assert "allocation_rules" in model
        assert "reporting_structure" in model
        assert model["allocation_method"] == "proportional"
    
    def test_budget_monitoring(self):
        """Test budget monitoring and alerting"""
        budget_config = {
            "monthly_budget": 5000.0,
            "alert_thresholds": [50, 80, 95],
            "notification_channels": ["email", "slack"]
        }
        
        monitoring = self.finops_manager.setup_budget_monitoring(budget_config)
        
        assert "budget_alerts" in monitoring
        assert "notification_config" in monitoring
        assert len(monitoring["budget_alerts"]) == 3
    
    def test_roi_calculation(self):
        """Test ROI calculation for ML investments"""
        investment_data = {
            "infrastructure_cost": 10000.0,
            "operational_cost": 5000.0,
            "revenue_impact": 25000.0,
            "cost_savings": 8000.0,
            "time_period_months": 12
        }
        
        roi_analysis = self.finops_manager.calculate_ml_roi(investment_data)
        
        assert "roi_percentage" in roi_analysis
        assert "payback_period_months" in roi_analysis
        assert "net_present_value" in roi_analysis
        assert roi_analysis["roi_percentage"] > 0


class TestCostForecastingEngine:
    """Test cost forecasting functionality"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.forecasting_engine = CostForecastingEngine()
    
    def test_historical_trend_analysis(self):
        """Test historical cost trend analysis"""
        # Generate mock historical data
        historical_data = []
        base_cost = 1000.0
        
        for i in range(30):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            cost = base_cost + (i * 10) + (i % 7 * 50)  # Trend with weekly pattern
            historical_data.append({"date": date, "cost": cost})
        
        trend_analysis = self.forecasting_engine.analyze_historical_trends(historical_data)
        
        assert "trend_direction" in trend_analysis
        assert "seasonality" in trend_analysis
        assert "growth_rate" in trend_analysis
        assert trend_analysis["trend_direction"] in ["increasing", "decreasing", "stable"]
    
    def test_cost_forecasting(self):
        """Test cost forecasting models"""
        historical_costs = [1000 + i * 50 for i in range(30)]  # Increasing trend
        
        forecast = self.forecasting_engine.forecast_costs(
            historical_costs, forecast_days=7
        )
        
        assert "forecasted_costs" in forecast
        assert "confidence_intervals" in forecast
        assert "model_accuracy" in forecast
        assert len(forecast["forecasted_costs"]) == 7
    
    def test_budget_variance_analysis(self):
        """Test budget variance analysis"""
        budget_data = {
            "planned_budget": 5000.0,
            "actual_spend": 5500.0,
            "time_period": "monthly"
        }
        
        variance_analysis = self.forecasting_engine.analyze_budget_variance(budget_data)
        
        assert "variance_amount" in variance_analysis
        assert "variance_percentage" in variance_analysis
        assert "variance_category" in variance_analysis
        assert variance_analysis["variance_amount"] == 500.0
        assert variance_analysis["variance_percentage"] == 10.0
    
    def test_anomaly_detection(self):
        """Test cost anomaly detection"""
        # Normal costs with one anomaly
        cost_series = [100.0] * 20 + [500.0] + [100.0] * 9
        
        anomalies = self.forecasting_engine.detect_cost_anomalies(cost_series)
        
        assert len(anomalies) > 0
        assert any(anomaly["anomaly_score"] > 0.8 for anomaly in anomalies)
        assert any(anomaly["anomaly_type"] == "spike" for anomaly in anomalies)


class TestCostOptimizationIntegration:
    """Integration tests for cost optimization components"""
    
    def setup_method(self):
        """Set up integration test fixtures"""
        self.cost_analyzer = ProductionCostAnalyzer("us-west-2")
        self.resource_optimizer = ResourceOptimizer()
        self.spot_manager = SpotInstanceManager("us-west-2")
        self.finops_manager = FinOpsManager()
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete cost optimization workflow"""
        # 1. Generate mock cost data
        mock_analyzer = MockCostAnalyzer()
        cost_data = mock_analyzer.generate_mock_cost_data(30)
        
        # 2. Analyze costs and identify optimization opportunities
        high_cost_resources = [
            item for item in cost_data 
            if item.cost > 200 and item.tags.get("OptimizationCandidate") == "true"
        ]
        
        # 3. Generate optimization recommendations
        recommendations = []
        for resource in high_cost_resources[:5]:  # Limit for testing
            if "EC2" in resource.service:
                utilization = {"cpu_avg": 25.0, "memory_avg": 30.0}
                rec = self.resource_optimizer.recommend_instance_size(
                    "c5.2xlarge", utilization
                )
                recommendations.append(rec)
        
        # 4. Calculate total savings potential
        total_savings = sum(rec.savings_potential for rec in recommendations)
        
        # 5. Verify workflow results
        assert len(recommendations) > 0
        assert total_savings > 0
        assert all(rec.confidence_score > 0 for rec in recommendations)
    
    def test_multi_cloud_cost_analysis(self):
        """Test multi-cloud cost analysis"""
        cloud_costs = {
            "aws": {"compute": 5000, "storage": 1000, "network": 500},
            "azure": {"compute": 3000, "storage": 800, "network": 300},
            "gcp": {"compute": 2000, "storage": 600, "network": 200}
        }
        
        total_cost = sum(
            sum(services.values()) 
            for services in cloud_costs.values()
        )
        
        # Calculate cost distribution
        cloud_percentages = {
            cloud: (sum(services.values()) / total_cost) * 100
            for cloud, services in cloud_costs.items()
        }
        
        assert total_cost == 12400
        assert cloud_percentages["aws"] > cloud_percentages["azure"]
        assert cloud_percentages["azure"] > cloud_percentages["gcp"]
    
    def test_cost_optimization_roi_calculation(self):
        """Test ROI calculation for cost optimization initiatives"""
        optimization_investment = {
            "implementation_cost": 10000.0,  # One-time cost
            "monthly_savings": 2000.0,       # Recurring savings
            "implementation_time_months": 2
        }
        
        # Calculate ROI over 12 months
        total_savings = optimization_investment["monthly_savings"] * 12
        net_benefit = total_savings - optimization_investment["implementation_cost"]
        roi_percentage = (net_benefit / optimization_investment["implementation_cost"]) * 100
        
        assert total_savings == 24000.0
        assert net_benefit == 14000.0
        assert roi_percentage == 140.0  # 140% ROI


def test_cost_optimization_configuration():
    """Test cost optimization configuration validation"""
    config = {
        "cost_thresholds": {
            "daily_alert": 1000.0,
            "monthly_budget": 25000.0,
            "anomaly_threshold": 2.0
        },
        "optimization_settings": {
            "enable_right_sizing": True,
            "enable_spot_instances": True,
            "enable_scheduling": True,
            "min_savings_threshold": 100.0
        },
        "notification_channels": {
            "email": "finops@company.com",
            "slack": "#cost-alerts",
            "webhook": "https://api.company.com/cost-alerts"
        }
    }
    
    # Validate configuration structure
    assert "cost_thresholds" in config
    assert "optimization_settings" in config
    assert "notification_channels" in config
    assert config["cost_thresholds"]["monthly_budget"] > 0
    assert config["optimization_settings"]["enable_right_sizing"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])