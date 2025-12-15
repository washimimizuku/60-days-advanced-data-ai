"""
Day 34: A/B Testing for ML - Exercise

Business Scenario:
You're the Senior Data Scientist at StreamTech Entertainment, a video streaming platform
with 50 million users. The company wants to improve user engagement through better
recommendations. You need to design and implement an A/B testing framework to compare
three recommendation algorithms:

1. Control: Collaborative filtering (current system)
2. Treatment A: Deep learning recommendation model
3. Treatment B: Hybrid model (collaborative + content-based)

Your task is to build a complete A/B testing platform that can:
- Assign users consistently to experiment variants
- Collect and analyze engagement metrics in real-time
- Implement multi-armed bandit for dynamic optimization
- Monitor guardrail metrics and implement early stopping
- Provide statistical analysis and recommendations

Success Criteria:
- Proper statistical design with 80% power and 5% significance
- Consistent user assignment across sessions
- Real-time monitoring with guardrail protection
- Multi-armed bandit implementation for traffic optimization
- Comprehensive reporting and recommendation system
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
import uuid
from abc import ABC, abstractmethod

# Statistical libraries
try:
    import scipy.stats as stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

@dataclass
class ExperimentConfig:
    """Configuration for A/B test experiment"""
    experiment_id: str
    name: str
    description: str
    
    # Variant configurations
    variants: Dict[str, Dict[str, Any]]  # {"control": {...}, "treatment_a": {...}}
    traffic_allocation: Dict[str, float]  # {"control": 0.5, "treatment_a": 0.3, "treatment_b": 0.2}
    
    # Success metrics
    primary_metric: str
    secondary_metrics: List[str] = field(default_factory=list)
    
    # Statistical parameters
    significance_level: float = 0.05
    power: float = 0.8
    minimum_detectable_effect: float = 0.05
    
    # Runtime parameters
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    min_sample_size: int = 1000
    
    # Guardrail metrics (metrics that shouldn't degrade)
    guardrail_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Eligibility criteria
    eligibility_criteria: Dict[str, Any] = field(default_factory=dict)

# TODO: Implement the traffic splitter
class TrafficSplitter:
    """Handles user assignment to experiment variants"""
    
    def __init__(self, experiment_config: ExperimentConfig):
        # TODO: Initialize traffic splitter
        # HINT: Store config and calculate hash boundaries for variants
        pass
    
    def assign_variant(self, user_id: str, experiment_id: str) -> str:
        """
        Assign user to experiment variant using consistent hashing
        
        Args:
            user_id: Unique user identifier
            experiment_id: Experiment identifier
            
        Returns:
            Variant name (e.g., 'control', 'treatment_a', 'treatment_b')
        """
        # Create deterministic hash
        hash_input = f"{user_id}:{experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Convert to 0-1 range
        normalized_hash = (hash_value % 100000) / 100000.0
        
        # Find variant based on boundaries
        cumulative = 0.0
        for variant, allocation in self.config.traffic_allocation.items():
            if cumulative <= normalized_hash < cumulative + allocation:
                return variant
            cumulative += allocation
        
        # Fallback to first variant
        return list(self.config.variants.keys())[0]
    
    def is_user_eligible(self, user_id: str, user_attributes: Dict[str, Any]) -> bool:
        """
        Check if user is eligible for the experiment
        
        Args:
            user_id: User identifier
            user_attributes: User characteristics (country, subscription_type, etc.)
            
        Returns:
            True if user is eligible
        """
        # TODO: Implement eligibility criteria
        # HINT: Check user attributes like country, subscription type, account age
        pass

# TODO: Implement statistical analyzer
class StatisticalAnalyzer:
    """Performs statistical analysis of A/B test results"""
    
    def __init__(self, significance_level: float = 0.05):
        # TODO: Initialize analyzer
        self.significance_level = significance_level
    
    def analyze_continuous_metric(self, control_data: np.array, treatment_data: np.array) -> Dict[str, Any]:
        """
        Analyze continuous metrics (e.g., watch time, engagement score)
        
        Args:
            control_data: Control group metric values
            treatment_data: Treatment group metric values
            
        Returns:
            Statistical analysis results
        """
        if len(control_data) == 0 or len(treatment_data) == 0:
            return {'error': 'Insufficient data for analysis'}
        
        # Basic statistics
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        control_std = np.std(control_data, ddof=1)
        treatment_std = np.std(treatment_data, ddof=1)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_data) - 1) * control_std**2 + 
                             (len(treatment_data) - 1) * treatment_std**2) / 
                            (len(control_data) + len(treatment_data) - 2))
        
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std > 0 else 0
        
        # Statistical test
        if SCIPY_AVAILABLE:
            t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
        else:
            # Simplified t-test
            se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
            t_stat = (treatment_mean - control_mean) / se_diff if se_diff > 0 else 0
            p_value = 0.05 if abs(t_stat) > 1.96 else 0.5
        
        # Confidence interval
        diff = treatment_mean - control_mean
        se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
        ci_lower = diff - 1.96 * se_diff
        ci_upper = diff + 1.96 * se_diff
        
        # Relative improvement
        relative_improvement = (treatment_mean - control_mean) / control_mean * 100 if control_mean != 0 else 0
        
        return {
            'control_mean': float(control_mean),
            'treatment_mean': float(treatment_mean),
            'difference': float(diff),
            'relative_improvement_pct': float(relative_improvement),
            'cohens_d': float(cohens_d),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < self.significance_level,
            'confidence_interval': (float(ci_lower), float(ci_upper)),
            'sample_sizes': {'control': len(control_data), 'treatment': len(treatment_data)}
        }
    
    def analyze_binary_metric(self, control_successes: int, control_total: int,
                            treatment_successes: int, treatment_total: int) -> Dict[str, Any]:
        """
        Analyze binary metrics (e.g., click-through rate, conversion rate)
        
        Args:
            control_successes: Number of successes in control group
            control_total: Total observations in control group
            treatment_successes: Number of successes in treatment group
            treatment_total: Total observations in treatment group
            
        Returns:
            Statistical analysis results
        """
        # TODO: Implement binary metric analysis
        # HINT: Calculate rates, z-test for proportions, confidence intervals
        pass
    
    def calculate_sample_size(self, baseline_rate: float, minimum_detectable_effect: float,
                            power: float = 0.8) -> int:
        """
        Calculate required sample size for experiment
        
        Args:
            baseline_rate: Current metric baseline
            minimum_detectable_effect: Minimum effect size to detect
            power: Statistical power (1 - Type II error rate)
            
        Returns:
            Required sample size per variant
        """
        # TODO: Implement sample size calculation
        # HINT: Use power analysis formulas for your metric type
        pass

# TODO: Implement multi-armed bandit algorithms
class MultiArmedBandit(ABC):
    """Abstract base class for multi-armed bandit algorithms"""
    
    @abstractmethod
    def select_arm(self) -> int:
        """Select which arm (variant) to pull"""
        pass
    
    @abstractmethod
    def update(self, arm: int, reward: float):
        """Update algorithm with reward from selected arm"""
        pass

class EpsilonGreedyBandit(MultiArmedBandit):
    """Epsilon-greedy bandit algorithm"""
    
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        # TODO: Initialize epsilon-greedy bandit
        # HINT: Store number of arms, epsilon, and initialize counts and values
        pass
    
    def select_arm(self) -> int:
        """Select arm using epsilon-greedy strategy"""
        if np.random.random() < self.epsilon:
            # Explore: choose random arm
            return np.random.randint(self.n_arms)
        else:
            # Exploit: choose best arm
            return np.argmax(self.values)
    
    def update(self, arm: int, reward: float):
        """Update arm statistics with new reward"""
        self.counts[arm] += 1
        n = self.counts[arm]
        
        # Incremental average update
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward

class UCBBandit(MultiArmedBandit):
    """Upper Confidence Bound bandit algorithm"""
    
    def __init__(self, n_arms: int, confidence_level: float = 2.0):
        # TODO: Initialize UCB bandit
        pass
    
    def select_arm(self) -> int:
        """Select arm using Upper Confidence Bound strategy"""
        # TODO: Implement UCB selection
        # HINT: Calculate UCB = average_reward + confidence_bonus for each arm
        pass
    
    def update(self, arm: int, reward: float):
        """Update arm statistics"""
        # TODO: Update counts and values
        pass

# TODO: Implement experiment manager
class ExperimentManager:
    """Manages A/B test experiments"""
    
    def __init__(self):
        # TODO: Initialize experiment manager
        self.experiments = {}
        self.active_experiments = {}
        self.results_log = []
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """
        Create a new A/B test experiment
        
        Args:
            config: Experiment configuration
            
        Returns:
            Experiment ID
        """
        # TODO: Create and validate experiment
        # HINT: Validate config, generate ID, store experiment
        pass
    
    def start_experiment(self, experiment_id: str) -> bool:
        """
        Start an experiment
        
        Args:
            experiment_id: ID of experiment to start
            
        Returns:
            True if started successfully
        """
        # TODO: Start experiment
        # HINT: Validate experiment can start, initialize traffic splitter and bandit
        pass
    
    def get_assignment(self, experiment_id: str, user_id: str, 
                      user_attributes: Dict[str, Any] = None) -> Optional[str]:
        """
        Get user assignment for experiment
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID
            user_attributes: User characteristics
            
        Returns:
            Assigned variant or None if not eligible
        """
        # TODO: Get user assignment
        # HINT: Check eligibility, get assignment from splitter or bandit
        pass
    
    def record_result(self, experiment_id: str, user_id: str, 
                     metric_name: str, metric_value: float) -> bool:
        """
        Record experiment result
        
        Args:
            experiment_id: Experiment ID
            user_id: User ID
            metric_name: Name of metric
            metric_value: Metric value
            
        Returns:
            True if recorded successfully
        """
        # TODO: Record result and update bandit if applicable
        pass

# TODO: Implement recommendation system models
class RecommendationModel(ABC):
    """Abstract base class for recommendation models"""
    
    @abstractmethod
    def get_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[str]:
        """Get recommendations for user"""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name"""
        pass

class CollaborativeFilteringModel(RecommendationModel):
    """Collaborative filtering recommendation model (Control)"""
    
    def __init__(self):
        # TODO: Initialize collaborative filtering model
        pass
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[str]:
        """Get collaborative filtering recommendations"""
        # TODO: Implement collaborative filtering logic
        # HINT: For exercise, you can simulate recommendations
        pass
    
    def get_model_name(self) -> str:
        return "collaborative_filtering"

class DeepLearningModel(RecommendationModel):
    """Deep learning recommendation model (Treatment A)"""
    
    def __init__(self):
        # TODO: Initialize deep learning model
        pass
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[str]:
        """Get deep learning recommendations"""
        # TODO: Implement deep learning logic (simulated)
        pass
    
    def get_model_name(self) -> str:
        return "deep_learning"

class HybridModel(RecommendationModel):
    """Hybrid recommendation model (Treatment B)"""
    
    def __init__(self):
        # TODO: Initialize hybrid model
        pass
    
    def get_recommendations(self, user_id: str, n_recommendations: int = 10) -> List[str]:
        """Get hybrid recommendations"""
        # TODO: Implement hybrid logic (simulated)
        pass
    
    def get_model_name(self) -> str:
        return "hybrid"

# TODO: Implement experiment monitor
class ExperimentMonitor:
    """Monitors experiment progress and provides analysis"""
    
    def __init__(self, experiment_manager: ExperimentManager):
        # TODO: Initialize monitor
        self.experiment_manager = experiment_manager
        self.analyzer = StatisticalAnalyzer()
    
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get current experiment status and results
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment status and analysis
        """
        # TODO: Analyze experiment results
        # HINT: Get results, perform statistical analysis, check guardrails
        pass
    
    def check_early_stopping(self, experiment_id: str) -> Dict[str, Any]:
        """
        Check if experiment should stop early
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Early stopping recommendation
        """
        # TODO: Implement early stopping logic
        # HINT: Check statistical significance, guardrail violations, sample size
        pass
    
    def generate_report(self, experiment_id: str) -> Dict[str, Any]:
        """
        Generate comprehensive experiment report
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Detailed experiment report
        """
        # TODO: Generate comprehensive report
        pass

# TODO: Implement the main streaming platform
class StreamingPlatform:
    """Main streaming platform with A/B testing integration"""
    
    def __init__(self):
        # TODO: Initialize platform
        self.experiment_manager = ExperimentManager()
        self.monitor = ExperimentMonitor(self.experiment_manager)
        
        # Initialize recommendation models
        self.models = {
            'control': CollaborativeFilteringModel(),
            'treatment_a': DeepLearningModel(),
            'treatment_b': HybridModel()
        }
    
    def get_recommendations_for_user(self, user_id: str, user_attributes: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get recommendations for user (with A/B testing)
        
        Args:
            user_id: User ID
            user_attributes: User characteristics
            
        Returns:
            Recommendations and experiment info
        """
        # TODO: Get user assignment and return appropriate recommendations
        # HINT: Get assignment from experiment manager, use appropriate model
        pass
    
    def record_user_interaction(self, user_id: str, interaction_type: str, 
                              item_id: str, value: float = 1.0):
        """
        Record user interaction for experiment tracking
        
        Args:
            user_id: User ID
            interaction_type: Type of interaction (click, watch, like, etc.)
            item_id: Item that was interacted with
            value: Interaction value (watch time, rating, etc.)
        """
        # TODO: Record interaction and update experiment results
        pass
    
    def run_experiment_simulation(self, experiment_config: ExperimentConfig, 
                                n_users: int = 10000, n_days: int = 14) -> Dict[str, Any]:
        """
        Simulate experiment with synthetic user interactions
        
        Args:
            experiment_config: Experiment configuration
            n_users: Number of users to simulate
            n_days: Number of days to simulate
            
        Returns:
            Simulation results
        """
        # TODO: Run experiment simulation
        # HINT: Create users, assign to variants, simulate interactions, analyze results
        pass

# TODO: Main demonstration function
def main():
    """
    Main function demonstrating A/B testing for ML models
    """
    print("Day 34: A/B Testing for ML - Exercise")
    print("=" * 60)
    
    # TODO: Initialize streaming platform
    print("1. Initializing StreamTech Entertainment Platform...")
    
    # TODO: Create experiment configuration
    print("2. Creating recommendation algorithm experiment...")
    
    # TODO: Set up experiment
    print("3. Setting up A/B test framework...")
    
    # TODO: Run simulation
    print("4. Running experiment simulation...")
    
    # TODO: Analyze results
    print("5. Analyzing experiment results...")
    
    # TODO: Generate recommendations
    print("6. Generating business recommendations...")
    
    print("\nExercise completed! Check the implementation above.")
    print("\nNext steps:")
    print("1. Implement the ExperimentConfig dataclass")
    print("2. Complete the TrafficSplitter for consistent user assignment")
    print("3. Add statistical analysis methods")
    print("4. Implement multi-armed bandit algorithms")
    print("5. Create the experiment management system")
    print("6. Add monitoring and early stopping logic")
    print("7. Run the full simulation and analyze results")

if __name__ == "__main__":
    main()