#!/usr/bin/env python3
"""
Day 40: ML Systems Review & Assessment - Exercise

This exercise provides a comprehensive assessment of your Phase 3 knowledge through
practical implementation challenges. You'll demonstrate your mastery of MLOps concepts
by building a mini ML system health check and optimization tool.

Learning Objectives:
- Apply Phase 3 concepts in a practical assessment scenario
- Demonstrate understanding of ML system architecture and monitoring
- Show ability to diagnose and optimize ML system performance
- Practice explaining technical concepts and recommendations

Time: 40 minutes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Simulated ML system metrics data
np.random.seed(42)

class MLSystemHealthChecker:
    """
    ML System Health Check and Optimization Tool
    
    This class simulates a production ML system and provides tools to assess
    its health, identify issues, and recommend optimizations.
    """
    
    def __init__(self):
        self.system_metrics = self._generate_system_metrics()
        self.model_performance = self._generate_model_performance()
        self.feature_metrics = self._generate_feature_metrics()
        self.business_metrics = self._generate_business_metrics()
        
    def _generate_system_metrics(self) -> Dict[str, Any]:
        """Generate simulated system performance metrics"""
        # TODO: Analyze the system metrics below and identify potential issues
        # HINT: Look for metrics that exceed typical production thresholds
        
        return {
            "api_latency_p95_ms": 150,  # Target: <100ms
            "api_latency_p99_ms": 300,  # Target: <200ms
            "throughput_rps": 850,      # Current capacity: 1000 RPS
            "error_rate_percent": 2.5,  # Target: <1%
            "cpu_utilization_percent": 85,  # Target: <80%
            "memory_utilization_percent": 78,  # Target: <80%
            "disk_usage_percent": 92,   # Target: <90%
            "cache_hit_rate_percent": 65,  # Target: >90%
            "database_connection_pool_usage": 0.95,  # Target: <0.8
            "feature_store_latency_ms": 25,  # Target: <10ms
            "model_loading_time_seconds": 45,  # Target: <30s
            "health_check_success_rate": 0.98  # Target: >0.99
        }
    
    def _generate_model_performance(self) -> Dict[str, Any]:
        """Generate simulated model performance metrics"""
        # TODO: Analyze model performance trends and identify degradation
        # HINT: Compare current performance with baseline expectations
        
        # Simulate 30 days of model performance data
        days = 30
        dates = [datetime.now() - timedelta(days=i) for i in range(days)]
        
        # Simulate gradual performance degradation
        base_auc = 0.85
        auc_scores = [base_auc - (i * 0.002) + np.random.normal(0, 0.01) for i in range(days)]
        auc_scores = [max(0.7, min(0.9, score)) for score in auc_scores]  # Clamp values
        
        precision_scores = [0.82 - (i * 0.001) + np.random.normal(0, 0.015) for i in range(days)]
        precision_scores = [max(0.6, min(0.9, score)) for score in precision_scores]
        
        recall_scores = [0.78 - (i * 0.0015) + np.random.normal(0, 0.02) for i in range(days)]
        recall_scores = [max(0.6, min(0.9, score)) for score in recall_scores]
        
        return {
            "dates": dates,
            "auc_scores": auc_scores,
            "precision_scores": precision_scores,
            "recall_scores": recall_scores,
            "current_auc": auc_scores[0],
            "baseline_auc": 0.85,
            "auc_threshold": 0.80,
            "prediction_volume_daily": 50000,
            "model_version": "v2.3.1",
            "last_retrained": "2 weeks ago"
        }
    
    def _generate_feature_metrics(self) -> Dict[str, Any]:
        """Generate simulated feature quality and drift metrics"""
        # TODO: Identify features with quality issues or drift
        # HINT: Look for high missing rates, drift scores, or staleness
        
        features = [
            "customer_age", "account_balance", "transaction_frequency",
            "credit_score", "employment_status", "geographic_region",
            "product_usage", "support_interactions", "payment_history"
        ]
        
        return {
            "feature_names": features,
            "missing_value_rates": [0.02, 0.15, 0.05, 0.08, 0.12, 0.01, 0.25, 0.03, 0.07],
            "drift_scores": [0.05, 0.18, 0.08, 0.12, 0.22, 0.04, 0.15, 0.09, 0.11],
            "feature_importance": [0.15, 0.22, 0.18, 0.20, 0.08, 0.05, 0.07, 0.03, 0.02],
            "data_freshness_hours": [1, 6, 2, 12, 4, 1, 24, 3, 8],
            "schema_violations": [0, 2, 0, 1, 3, 0, 5, 0, 1],
            "drift_threshold": 0.10,
            "missing_threshold": 0.10,
            "freshness_threshold_hours": 6
        }
    
    def _generate_business_metrics(self) -> Dict[str, Any]:
        """Generate simulated business impact metrics"""
        # TODO: Analyze business metrics to understand ML system impact
        # HINT: Look for trends in conversion rates, revenue, and customer satisfaction
        
        return {
            "monthly_revenue_impact": 125000,  # USD
            "baseline_monthly_revenue": 150000,  # USD
            "conversion_rate_current": 0.045,
            "conversion_rate_baseline": 0.052,
            "customer_satisfaction_score": 7.2,  # Out of 10
            "customer_satisfaction_baseline": 7.8,
            "false_positive_rate": 0.08,
            "false_negative_rate": 0.15,
            "operational_cost_monthly": 25000,  # USD
            "roi_current": 4.0,  # Revenue / Cost ratio
            "roi_target": 6.0
        }

def analyze_system_health(health_checker: MLSystemHealthChecker) -> Dict[str, Any]:
    """
    TODO: Implement comprehensive system health analysis
    
    Analyze all system metrics and identify issues across:
    1. System performance (latency, throughput, errors)
    2. Model performance (accuracy, drift, staleness)
    3. Feature quality (missing values, drift, freshness)
    4. Business impact (revenue, conversion, satisfaction)
    
    Args:
        health_checker: MLSystemHealthChecker instance with system data
        
    Returns:
        Dictionary containing:
        - overall_health_score (0-100)
        - critical_issues (list of critical problems)
        - recommendations (list of improvement suggestions)
        - priority_actions (ordered list of next steps)
    
    HINT: Use the thresholds defined in the metrics to identify issues
    """
    
    # Your implementation here
    system_metrics = health_checker.system_metrics
    model_performance = health_checker.model_performance
    feature_metrics = health_checker.feature_metrics
    business_metrics = health_checker.business_metrics
    
    # Calculate overall health score (0-100)
    health_score = 100
    
    # System performance penalties
    if system_metrics['api_latency_p95_ms'] > 100:
        health_score -= 10
    if system_metrics['error_rate_percent'] > 1:
        health_score -= 15
    if system_metrics['disk_usage_percent'] > 90:
        health_score -= 20
    
    # Model performance penalties
    if model_performance['current_auc'] < model_performance['auc_threshold']:
        health_score -= 25
    
    # Feature quality penalties
    high_drift_features = sum(1 for score in feature_metrics['drift_scores'] if score > feature_metrics['drift_threshold'])
    health_score -= high_drift_features * 3
    
    overall_health_score = max(0, health_score)
    
    # Identify critical issues
    critical_issues = []
    if system_metrics['disk_usage_percent'] > 90:
        critical_issues.append("Disk usage critical (>90%)")
    if system_metrics['error_rate_percent'] > 1:
        critical_issues.append("High error rate detected")
    if model_performance['current_auc'] < 0.8:
        critical_issues.append("Model performance below threshold")
    
    # Generate recommendations
    recommendations = []
    if system_metrics['api_latency_p95_ms'] > 100:
        recommendations.append("Optimize API response time")
    if system_metrics['cache_hit_rate_percent'] < 90:
        recommendations.append("Improve caching strategy")
    if high_drift_features > 2:
        recommendations.append("Retrain model due to feature drift")
    
    # Prioritize actions
    priority_actions = critical_issues + recommendations[:3]
    
    return {
        "overall_health_score": overall_health_score,
        "critical_issues": critical_issues,
        "recommendations": recommendations,
        "priority_actions": priority_actions
    }

def create_monitoring_dashboard(health_checker: MLSystemHealthChecker):
    """
    TODO: Create a comprehensive monitoring dashboard
    
    Create visualizations for:
    1. System performance trends
    2. Model performance over time
    3. Feature drift analysis
    4. Business impact metrics
    
    Args:
        health_checker: MLSystemHealthChecker instance with system data
        
    HINT: Use matplotlib/seaborn to create informative plots
    """
    
    # Create a 2x2 subplot layout for the dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ML System Health Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1 - System Performance Metrics (top-left)
    metrics = ['Latency', 'Error Rate', 'CPU', 'Memory', 'Disk']
    values = [health_checker.system_metrics['api_latency_p95_ms']/100, 
              health_checker.system_metrics['error_rate_percent'],
              health_checker.system_metrics['cpu_utilization_percent']/100,
              health_checker.system_metrics['memory_utilization_percent']/100,
              health_checker.system_metrics['disk_usage_percent']/100]
    axes[0,0].bar(metrics, values)
    axes[0,0].set_title('System Performance')
    axes[0,0].set_ylabel('Normalized Values')
    
    # Plot 2 - Model Performance Trends (top-right)
    dates = health_checker.model_performance['dates']
    auc_scores = health_checker.model_performance['auc_scores']
    axes[0,1].plot(dates, auc_scores, 'b-', label='AUC')
    axes[0,1].axhline(y=0.8, color='r', linestyle='--', label='Threshold')
    axes[0,1].set_title('Model Performance Trend')
    axes[0,1].legend()
    
    # Plot 3 - Feature Drift Analysis (bottom-left)
    features = health_checker.feature_metrics['feature_names']
    drift_scores = health_checker.feature_metrics['drift_scores']
    axes[1,0].bar(range(len(features)), drift_scores)
    axes[1,0].axhline(y=0.1, color='r', linestyle='--', label='Threshold')
    axes[1,0].set_title('Feature Drift Scores')
    axes[1,0].set_xticks(range(len(features)))
    axes[1,0].set_xticklabels(features, rotation=45)
    
    # Plot 4 - Business Impact Summary (bottom-right)
    business = health_checker.business_metrics
    metrics = ['Revenue', 'Conversion', 'Satisfaction']
    current = [business['monthly_revenue_impact']/1000, 
               business['conversion_rate_current']*100,
               business['customer_satisfaction_score']]
    baseline = [business['baseline_monthly_revenue']/1000,
                business['conversion_rate_baseline']*100,
                business['customer_satisfaction_baseline']]
    
    x = np.arange(len(metrics))
    axes[1,1].bar(x - 0.2, current, 0.4, label='Current')
    axes[1,1].bar(x + 0.2, baseline, 0.4, label='Baseline')
    axes[1,1].set_title('Business Metrics')
    axes[1,1].set_xticks(x)
    axes[1,1].set_xticklabels(metrics)
    axes[1,1].legend()
    
    plt.tight_layout()
    plt.show()

def generate_optimization_plan(health_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    TODO: Generate a detailed optimization plan based on health analysis
    
    Create an actionable plan that includes:
    1. Immediate fixes (0-1 week)
    2. Short-term improvements (1-4 weeks)
    3. Long-term optimizations (1-3 months)
    4. Resource requirements and timelines
    
    Args:
        health_analysis: Results from analyze_system_health function
        
    Returns:
        Dictionary containing detailed optimization plan with timelines and resources
        
    HINT: Prioritize based on business impact and implementation difficulty
    """
    
    # Categorize issues by urgency and impact
    critical_issues = health_analysis.get('critical_issues', [])
    
    immediate_fixes = [issue for issue in critical_issues if 'critical' in issue.lower() or 'high' in issue.lower()]
    short_term_improvements = health_analysis.get('recommendations', [])[:3]
    long_term_optimizations = ["Implement advanced monitoring", "Upgrade infrastructure", "Enhance model architecture"]
    
    # Estimate resource requirements
    resource_requirements = {
        "immediate_fixes": {"engineering_hours": len(immediate_fixes) * 8, "infrastructure_cost": len(immediate_fixes) * 2000},
        "short_term_improvements": {"engineering_hours": len(short_term_improvements) * 20, "infrastructure_cost": len(short_term_improvements) * 5000},
        "long_term_optimizations": {"engineering_hours": len(long_term_optimizations) * 40, "infrastructure_cost": len(long_term_optimizations) * 10000}
    }
    
    # Create implementation timeline
    timeline = {
        "week_1": immediate_fixes,
        "weeks_2_4": short_term_improvements,
        "months_1_3": long_term_optimizations
    }
    
    return {
        "immediate_fixes": immediate_fixes,
        "short_term_improvements": short_term_improvements,
        "long_term_optimizations": long_term_optimizations,
        "resource_requirements": resource_requirements,
        "timeline": timeline,
        "expected_roi": 0.0,  # Calculate expected return on investment
        "risk_assessment": "Low/Medium/High"
    }

def calculate_business_impact(optimization_plan: Dict[str, Any], 
                            current_metrics: Dict[str, Any]) -> Dict[str, float]:
    """
    TODO: Calculate the expected business impact of the optimization plan
    
    Estimate improvements in:
    1. Revenue increase from better model performance
    2. Cost savings from system optimizations
    3. Customer satisfaction improvements
    4. Operational efficiency gains
    
    Args:
        optimization_plan: Generated optimization plan
        current_metrics: Current business metrics
        
    Returns:
        Dictionary with projected business impact metrics
        
    HINT: Use industry benchmarks and historical data for realistic projections
    """
    
    # Calculate projected revenue impact (10% improvement)
    baseline_revenue = current_metrics.get('baseline_monthly_revenue', 150000)
    projected_revenue_increase = baseline_revenue * 0.10 * 12  # Annual
    
    # Calculate projected cost savings (5% reduction in operational costs)
    operational_cost = current_metrics.get('operational_cost_monthly', 25000)
    projected_cost_savings = operational_cost * 0.05 * 12  # Annual
    
    # Calculate projected efficiency gains (20% improvement)
    projected_efficiency_improvement = 20.0
    
    # Calculate overall ROI
    total_investment = sum(req.get('infrastructure_cost', 0) + req.get('engineering_hours', 0) * 150 
                          for req in optimization_plan.get('resource_requirements', {}).values())
    total_benefit = projected_revenue_increase + projected_cost_savings
    projected_roi = total_benefit / max(total_investment, 1)
    
    return {
        "projected_revenue_increase_annual": projected_revenue_increase,
        "projected_cost_savings_annual": projected_cost_savings,
        "projected_efficiency_improvement_percent": projected_efficiency_improvement,
        "total_investment_required": total_investment,
        "projected_roi": projected_roi,
        "payback_period_months": max(1, total_investment / max(total_benefit / 12, 1))
    }

def main():
    """
    Main function to run the ML System Health Check assessment
    """
    print("🔍 ML System Health Check & Assessment")
    print("=" * 50)
    
    # Initialize the health checker
    health_checker = MLSystemHealthChecker()
    
    print("\n📊 System Overview:")
    print(f"API Latency (P95): {health_checker.system_metrics['api_latency_p95_ms']}ms")
    print(f"Current AUC: {health_checker.model_performance['current_auc']:.3f}")
    print(f"Monthly Revenue Impact: ${health_checker.business_metrics['monthly_revenue_impact']:,}")
    
    # Implement the health analysis
    print("\n🔍 Analyzing System Health...")
    try:
        health_analysis = analyze_system_health(health_checker)
    except Exception as e:
        print(f"❌ Health analysis failed: {str(e)}")
        return None
    
    print(f"\n📈 Overall Health Score: {health_analysis['overall_health_score']}/100")
    
    if health_analysis['critical_issues']:
        print(f"\n🚨 Critical Issues Found: {len(health_analysis['critical_issues'])}")
        for issue in health_analysis['critical_issues']:
            print(f"  • {issue}")
    
    if health_analysis['recommendations']:
        print(f"\n💡 Recommendations: {len(health_analysis['recommendations'])}")
        for rec in health_analysis['recommendations']:
            print(f"  • {rec}")
    
    # Create the monitoring dashboard
    print("\n📊 Generating Monitoring Dashboard...")
    try:
        create_monitoring_dashboard(health_checker)
    except Exception as e:
        print(f"❌ Dashboard creation failed: {str(e)}")
    
    # Generate optimization plan
    print("\n🎯 Creating Optimization Plan...")
    try:
        optimization_plan = generate_optimization_plan(health_analysis)
    except Exception as e:
        print(f"❌ Optimization plan failed: {str(e)}")
        return None
    
    # Calculate business impact
    print("\n💰 Calculating Business Impact...")
    try:
        business_impact = calculate_business_impact(optimization_plan, health_checker.business_metrics)
    except Exception as e:
        print(f"❌ Business impact calculation failed: {str(e)}")
        return None
    
    print(f"\n📋 Assessment Complete!")
    print(f"Projected ROI: {business_impact['projected_roi']:.1f}x")
    print(f"Payback Period: {business_impact['payback_period_months']:.1f} months")
    
    print(f"\n🎯 Next Steps:")
    priority_actions = health_analysis.get('priority_actions', [])
    if priority_actions:
        for i, action in enumerate(priority_actions[:3], 1):
            print(f"  {i}. {action}")
    else:
        print("  No priority actions identified")
    
    return {
        "health_analysis": health_analysis,
        "optimization_plan": optimization_plan,
        "business_impact": business_impact
    }

if __name__ == "__main__":
    # Run the assessment
    results = main()
    
    print(f"\n✅ Assessment completed successfully!")
    print(f"Use the results to improve your ML system's performance and business impact.")

"""
ASSESSMENT CRITERIA:

Your implementation will be evaluated on:

1. Problem Identification (25 points)
   - Correctly identifies critical system issues
   - Understands the relationship between metrics and thresholds
   - Recognizes patterns in performance degradation

2. Analysis Quality (25 points)
   - Comprehensive health score calculation
   - Logical prioritization of issues
   - Clear and actionable recommendations

3. Technical Implementation (25 points)
   - Clean, well-structured code
   - Appropriate use of data structures and algorithms
   - Effective visualization and dashboard creation

4. Business Understanding (25 points)
   - Realistic ROI and impact calculations
   - Understanding of business metrics and their importance
   - Practical optimization plans with timelines

BONUS POINTS (up to 10 additional points):
- Creative visualizations or additional metrics
- Advanced optimization techniques
- Consideration of edge cases and error handling
- Clear documentation and code comments

TOTAL POSSIBLE SCORE: 110 points

SCORING GUIDE:
- 95-110: Exceptional mastery - Ready for senior MLOps architect roles
- 85-94: Strong performance - Ready for senior MLOps engineer roles
- 75-84: Good understanding - Ready for MLOps engineer roles
- 65-74: Adequate knowledge - Some areas need improvement
- Below 65: Needs significant review before proceeding to Phase 4

Good luck with your assessment! This exercise demonstrates the real-world
skills you'll use as an MLOps professional.
"""