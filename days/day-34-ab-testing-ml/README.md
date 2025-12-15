# Day 34: A/B Testing for ML - Experimentation, Statistical Analysis

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Design and implement** A/B testing frameworks for ML model evaluation
- **Apply statistical methods** to measure significance and business impact
- **Handle multi-armed bandit problems** for dynamic model selection
- **Build experimentation platforms** for continuous model improvement
- **Analyze and interpret** A/B test results with proper statistical rigor

**Estimated Time**: 1 hour  
**Difficulty Level**: Advanced ⭐⭐⭐⭐

---

## 🎯 What is A/B Testing for ML?

A/B testing for machine learning is the practice of comparing different ML models, algorithms, or features in production to determine which performs better according to business metrics. Unlike traditional A/B testing that focuses on user interface changes, ML A/B testing evaluates model performance, prediction accuracy, and business impact.

### Key Differences from Traditional A/B Testing

**Traditional A/B Testing:**
- Tests user interface changes, features, or content
- Measures user behavior metrics (click-through rates, conversions)
- Relatively simple randomization and analysis

**ML A/B Testing:**
- Tests different models, algorithms, or prediction strategies
- Measures both technical metrics (accuracy, latency) and business metrics (revenue, engagement)
- Complex considerations: model drift, feature dependencies, temporal effects

---

## 🧪 A/B Testing Framework for ML

### 1. **Experimental Design**

```python
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from enum import Enum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    description: str
    
    # Model configurations
    control_model: Dict[str, Any]
    treatment_models: List[Dict[str, Any]]
    
    # Traffic allocation
    traffic_allocation: Dict[str, float]  # {"control": 0.5, "treatment_a": 0.3, "treatment_b": 0.2}
    
    # Success metrics
    primary_metric: str
    secondary_metrics: List[str]
    
    # Statistical parameters
    significance_level: float = 0.05
    power: float = 0.8
    minimum_detectable_effect: float = 0.05
    
    # Runtime parameters
    start_date: datetime
    end_date: Optional[datetime] = None
    min_sample_size: int = 1000
    
    # Guardrail metrics (metrics that shouldn't degrade)
    guardrail_metrics: Dict[str, Dict[str, float]] = None  # {"latency": {"max": 100}}

class ExperimentFramework:
    def __init__(self):
        self.experiments = {}
        self.assignment_logs = []
        
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new A/B test experiment"""
        
        # Validate configuration
        self._validate_config(config)
        
        # Store experiment
        self.experiments[config.experiment_id] = {
            'config': config,
            'status': ExperimentStatus.DRAFT,
            'results': {},
            'assignments': {},
            'metrics_history': []
        }
        
        return config.experiment_id
    
    def _validate_config(self, config: ExperimentConfig):
        """Validate experiment configuration"""
        
        # Check traffic allocation sums to 1
        total_traffic = sum(config.traffic_allocation.values())
        if abs(total_traffic - 1.0) > 0.001:
            raise ValueError(f"Traffic allocation must sum to 1.0, got {total_traffic}")
        
        # Check all variants are included
        variants = set(config.traffic_allocation.keys())
        expected_variants = {"control"} | {f"treatment_{i}" for i in range(len(config.treatment_models))}
        
        if variants != expected_variants:
            raise ValueError(f"Traffic allocation variants {variants} don't match expected {expected_variants}")
```

### 2. **Traffic Splitting and Assignment**

```python
import hashlib
import random

class TrafficSplitter:
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.variant_boundaries = self._calculate_boundaries()
        
    def _calculate_boundaries(self) -> Dict[str, tuple]:
        """Calculate hash boundaries for each variant"""
        boundaries = {}
        cumulative = 0.0
        
        for variant, allocation in self.config.traffic_allocation.items():
            start = cumulative
            end = cumulative + allocation
            boundaries[variant] = (start, end)
            cumulative = end
            
        return boundaries
    
    def assign_variant(self, user_id: str, experiment_id: str) -> str:
        """Assign user to experiment variant using consistent hashing"""
        
        # Create deterministic hash
        hash_input = f"{user_id}:{experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Convert to 0-1 range
        normalized_hash = (hash_value % 10000) / 10000.0
        
        # Find variant based on boundaries
        for variant, (start, end) in self.variant_boundaries.items():
            if start <= normalized_hash < end:
                return variant
        
        # Fallback to control
        return "control"
    
    def is_user_eligible(self, user_id: str, user_attributes: Dict[str, Any]) -> bool:
        """Check if user is eligible for experiment"""
        
        # Example eligibility criteria
        if 'country' in user_attributes:
            if user_attributes['country'] not in ['US', 'CA', 'UK']:
                return False
        
        if 'user_type' in user_attributes:
            if user_attributes['user_type'] == 'internal':
                return False
        
        return True

# Usage example
experiment_config = ExperimentConfig(
    experiment_id="fraud_model_comparison_v1",
    name="Fraud Detection Model A/B Test",
    description="Compare XGBoost vs LightGBM for fraud detection",
    control_model={"type": "xgboost", "version": "1.0"},
    treatment_models=[{"type": "lightgbm", "version": "1.0"}],
    traffic_allocation={"control": 0.7, "treatment_0": 0.3},
    primary_metric="fraud_detection_f1",
    secondary_metrics=["precision", "recall", "false_positive_rate"],
    significance_level=0.05,
    power=0.8,
    minimum_detectable_effect=0.02,
    start_date=datetime.now(),
    min_sample_size=10000
)

splitter = TrafficSplitter(experiment_config)
```

### 3. **Statistical Analysis Engine**

```python
import scipy.stats as stats
from scipy.stats import ttest_ind, chi2_contingency
import numpy as np

class StatisticalAnalyzer:
    def __init__(self, significance_level: float = 0.05):
        self.significance_level = significance_level
        
    def analyze_continuous_metric(self, control_data: np.array, treatment_data: np.array) -> Dict[str, Any]:
        """Analyze continuous metrics (e.g., revenue, latency)"""
        
        # Basic statistics
        control_mean = np.mean(control_data)
        treatment_mean = np.mean(treatment_data)
        control_std = np.std(control_data)
        treatment_std = np.std(treatment_data)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(((len(control_data) - 1) * control_std**2 + 
                             (len(treatment_data) - 1) * treatment_std**2) / 
                            (len(control_data) + len(treatment_data) - 2))
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        # Statistical test
        t_stat, p_value = ttest_ind(control_data, treatment_data)
        
        # Confidence interval for difference
        se_diff = pooled_std * np.sqrt(1/len(control_data) + 1/len(treatment_data))
        df = len(control_data) + len(treatment_data) - 2
        t_critical = stats.t.ppf(1 - self.significance_level/2, df)
        
        diff = treatment_mean - control_mean
        ci_lower = diff - t_critical * se_diff
        ci_upper = diff + t_critical * se_diff
        
        # Relative improvement
        relative_improvement = (treatment_mean - control_mean) / control_mean * 100
        
        return {
            'control_mean': control_mean,
            'treatment_mean': treatment_mean,
            'control_std': control_std,
            'treatment_std': treatment_std,
            'difference': diff,
            'relative_improvement_pct': relative_improvement,
            'cohens_d': cohens_d,
            't_statistic': t_stat,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_sizes': {'control': len(control_data), 'treatment': len(treatment_data)}
        }
    
    def analyze_binary_metric(self, control_successes: int, control_total: int,
                            treatment_successes: int, treatment_total: int) -> Dict[str, Any]:
        """Analyze binary metrics (e.g., conversion rate, click-through rate)"""
        
        # Conversion rates
        control_rate = control_successes / control_total
        treatment_rate = treatment_successes / treatment_total
        
        # Z-test for proportions
        pooled_rate = (control_successes + treatment_successes) / (control_total + treatment_total)
        se_pooled = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_total + 1/treatment_total))
        
        z_stat = (treatment_rate - control_rate) / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        # Confidence interval for difference
        se_diff = np.sqrt(control_rate * (1 - control_rate) / control_total + 
                         treatment_rate * (1 - treatment_rate) / treatment_total)
        z_critical = stats.norm.ppf(1 - self.significance_level/2)
        
        diff = treatment_rate - control_rate
        ci_lower = diff - z_critical * se_diff
        ci_upper = diff + z_critical * se_diff
        
        # Relative improvement
        relative_improvement = (treatment_rate - control_rate) / control_rate * 100
        
        return {
            'control_rate': control_rate,
            'treatment_rate': treatment_rate,
            'difference': diff,
            'relative_improvement_pct': relative_improvement,
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': p_value < self.significance_level,
            'confidence_interval': (ci_lower, ci_upper),
            'sample_sizes': {'control': control_total, 'treatment': treatment_total}
        }
    
    def calculate_sample_size(self, baseline_rate: float, minimum_detectable_effect: float,
                            power: float = 0.8, significance_level: float = 0.05) -> int:
        """Calculate required sample size for binary metrics"""
        
        # Effect size
        treatment_rate = baseline_rate * (1 + minimum_detectable_effect)
        
        # Z-scores
        z_alpha = stats.norm.ppf(1 - significance_level/2)
        z_beta = stats.norm.ppf(power)
        
        # Pooled variance
        pooled_rate = (baseline_rate + treatment_rate) / 2
        pooled_variance = pooled_rate * (1 - pooled_rate)
        
        # Sample size per group
        n = 2 * pooled_variance * (z_alpha + z_beta)**2 / (treatment_rate - baseline_rate)**2
        
        return int(np.ceil(n))
```

---

## 🎰 Multi-Armed Bandit Algorithms

### 1. **Epsilon-Greedy Algorithm**

```python
class EpsilonGreedyBandit:
    def __init__(self, n_arms: int, epsilon: float = 0.1):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        
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
    
    def get_arm_statistics(self) -> Dict[str, Any]:
        """Get current statistics for all arms"""
        
        return {
            'counts': self.counts.tolist(),
            'values': self.values.tolist(),
            'total_pulls': int(np.sum(self.counts)),
            'best_arm': int(np.argmax(self.values))
        }
```

### 2. **Upper Confidence Bound (UCB) Algorithm**

```python
class UCBBandit:
    def __init__(self, n_arms: int, confidence_level: float = 2.0):
        self.n_arms = n_arms
        self.confidence_level = confidence_level
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.total_counts = 0
        
    def select_arm(self) -> int:
        """Select arm using Upper Confidence Bound strategy"""
        
        # If any arm hasn't been tried, try it
        for arm in range(self.n_arms):
            if self.counts[arm] == 0:
                return arm
        
        # Calculate UCB for each arm
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            confidence_bonus = np.sqrt(
                (self.confidence_level * np.log(self.total_counts)) / self.counts[arm]
            )
            ucb_values[arm] = self.values[arm] + confidence_bonus
        
        return np.argmax(ucb_values)
    
    def update(self, arm: int, reward: float):
        """Update arm statistics with new reward"""
        
        self.counts[arm] += 1
        self.total_counts += 1
        n = self.counts[arm]
        
        # Incremental average update
        self.values[arm] = ((n - 1) / n) * self.values[arm] + (1 / n) * reward
```

### 3. **Thompson Sampling (Bayesian Bandit)**

```python
class ThompsonSamplingBandit:
    def __init__(self, n_arms: int):
        self.n_arms = n_arms
        # Beta distribution parameters (alpha, beta)
        self.alpha = np.ones(n_arms)  # Prior successes
        self.beta = np.ones(n_arms)   # Prior failures
        
    def select_arm(self) -> int:
        """Select arm using Thompson Sampling"""
        
        # Sample from Beta distribution for each arm
        sampled_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            sampled_values[arm] = np.random.beta(self.alpha[arm], self.beta[arm])
        
        return np.argmax(sampled_values)
    
    def update(self, arm: int, reward: float):
        """Update Beta distribution parameters"""
        
        if reward > 0:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1
    
    def get_arm_probabilities(self) -> np.array:
        """Get estimated success probability for each arm"""
        
        return self.alpha / (self.alpha + self.beta)
```

---

## 🏗️ Production Experimentation Platform

### 1. **Experiment Management System**

```python
class ExperimentManager:
    def __init__(self, storage_backend=None):
        self.storage = storage_backend or {}
        self.active_experiments = {}
        self.bandit_algorithms = {}
        
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an A/B test experiment"""
        
        if experiment_id not in self.storage:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment = self.storage[experiment_id]
        config = experiment['config']
        
        # Validate experiment can start
        if datetime.now() < config.start_date:
            raise ValueError("Experiment start date is in the future")
        
        if config.end_date and datetime.now() > config.end_date:
            raise ValueError("Experiment end date has passed")
        
        # Initialize traffic splitter
        splitter = TrafficSplitter(config)
        
        # Initialize bandit if configured
        bandit = None
        if hasattr(config, 'bandit_algorithm'):
            if config.bandit_algorithm == 'epsilon_greedy':
                bandit = EpsilonGreedyBandit(
                    n_arms=len(config.traffic_allocation),
                    epsilon=getattr(config, 'epsilon', 0.1)
                )
            elif config.bandit_algorithm == 'ucb':
                bandit = UCBBandit(
                    n_arms=len(config.traffic_allocation),
                    confidence_level=getattr(config, 'confidence_level', 2.0)
                )
            elif config.bandit_algorithm == 'thompson_sampling':
                bandit = ThompsonSamplingBandit(n_arms=len(config.traffic_allocation))
        
        # Store active experiment
        self.active_experiments[experiment_id] = {
            'config': config,
            'splitter': splitter,
            'bandit': bandit,
            'start_time': datetime.now(),
            'assignments': {},
            'results': []
        }
        
        # Update status
        experiment['status'] = ExperimentStatus.RUNNING
        
        return True
    
    def get_assignment(self, experiment_id: str, user_id: str, 
                      user_attributes: Dict[str, Any] = None) -> Optional[str]:
        """Get user assignment for experiment"""
        
        if experiment_id not in self.active_experiments:
            return None
        
        experiment = self.active_experiments[experiment_id]
        splitter = experiment['splitter']
        
        # Check eligibility
        if user_attributes and not splitter.is_user_eligible(user_id, user_attributes):
            return None
        
        # Check if user already assigned
        if user_id in experiment['assignments']:
            return experiment['assignments'][user_id]
        
        # Get assignment
        if experiment['bandit']:
            # Use bandit algorithm
            arm_index = experiment['bandit'].select_arm()
            variants = list(experiment['config'].traffic_allocation.keys())
            variant = variants[arm_index]
        else:
            # Use fixed traffic splitting
            variant = splitter.assign_variant(user_id, experiment_id)
        
        # Store assignment
        experiment['assignments'][user_id] = variant
        
        return variant
    
    def record_result(self, experiment_id: str, user_id: str, 
                     metric_name: str, metric_value: float):
        """Record experiment result"""
        
        if experiment_id not in self.active_experiments:
            return False
        
        experiment = self.active_experiments[experiment_id]
        
        # Get user's variant
        variant = experiment['assignments'].get(user_id)
        if not variant:
            return False
        
        # Record result
        result = {
            'timestamp': datetime.now(),
            'user_id': user_id,
            'variant': variant,
            'metric_name': metric_name,
            'metric_value': metric_value
        }
        
        experiment['results'].append(result)
        
        # Update bandit if applicable
        if experiment['bandit'] and metric_name == experiment['config'].primary_metric:
            variants = list(experiment['config'].traffic_allocation.keys())
            arm_index = variants.index(variant)
            experiment['bandit'].update(arm_index, metric_value)
        
        return True
```

### 2. **Real-time Monitoring and Analysis**

```python
class ExperimentMonitor:
    def __init__(self, experiment_manager: ExperimentManager):
        self.experiment_manager = experiment_manager
        self.analyzer = StatisticalAnalyzer()
        
    def get_experiment_status(self, experiment_id: str) -> Dict[str, Any]:
        """Get current experiment status and results"""
        
        if experiment_id not in self.experiment_manager.active_experiments:
            return {'error': 'Experiment not active'}
        
        experiment = self.experiment_manager.active_experiments[experiment_id]
        config = experiment['config']
        results = experiment['results']
        
        if not results:
            return {
                'experiment_id': experiment_id,
                'status': 'running',
                'sample_size': 0,
                'message': 'No results yet'
            }
        
        # Convert results to DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Analyze primary metric
        primary_results = self._analyze_primary_metric(df, config.primary_metric)
        
        # Check guardrail metrics
        guardrail_results = self._check_guardrails(df, config.guardrail_metrics or {})
        
        # Calculate statistical power
        power_analysis = self._calculate_power(df, config)
        
        return {
            'experiment_id': experiment_id,
            'status': 'running',
            'runtime_hours': (datetime.now() - experiment['start_time']).total_seconds() / 3600,
            'sample_size': len(df),
            'primary_metric_analysis': primary_results,
            'guardrail_analysis': guardrail_results,
            'power_analysis': power_analysis,
            'recommendation': self._get_recommendation(primary_results, guardrail_results, power_analysis)
        }
    
    def _analyze_primary_metric(self, df: pd.DataFrame, metric_name: str) -> Dict[str, Any]:
        """Analyze primary metric results"""
        
        metric_data = df[df['metric_name'] == metric_name]
        
        if len(metric_data) == 0:
            return {'error': f'No data for metric {metric_name}'}
        
        # Group by variant
        control_data = metric_data[metric_data['variant'] == 'control']['metric_value'].values
        
        results = {}
        for variant in metric_data['variant'].unique():
            if variant == 'control':
                continue
                
            treatment_data = metric_data[metric_data['variant'] == variant]['metric_value'].values
            
            if len(control_data) > 0 and len(treatment_data) > 0:
                analysis = self.analyzer.analyze_continuous_metric(control_data, treatment_data)
                results[variant] = analysis
        
        return results
    
    def _check_guardrails(self, df: pd.DataFrame, guardrail_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Check guardrail metrics"""
        
        guardrail_results = {}
        
        for metric_name, thresholds in guardrail_metrics.items():
            metric_data = df[df['metric_name'] == metric_name]
            
            if len(metric_data) == 0:
                continue
            
            for variant in metric_data['variant'].unique():
                variant_data = metric_data[metric_data['variant'] == variant]['metric_value']
                
                if len(variant_data) == 0:
                    continue
                
                mean_value = variant_data.mean()
                
                # Check thresholds
                violations = []
                if 'max' in thresholds and mean_value > thresholds['max']:
                    violations.append(f"Exceeds maximum: {mean_value:.3f} > {thresholds['max']}")
                
                if 'min' in thresholds and mean_value < thresholds['min']:
                    violations.append(f"Below minimum: {mean_value:.3f} < {thresholds['min']}")
                
                guardrail_results[f"{variant}_{metric_name}"] = {
                    'value': mean_value,
                    'violations': violations,
                    'is_healthy': len(violations) == 0
                }
        
        return guardrail_results
    
    def _calculate_power(self, df: pd.DataFrame, config: ExperimentConfig) -> Dict[str, Any]:
        """Calculate statistical power of current experiment"""
        
        primary_data = df[df['metric_name'] == config.primary_metric]
        
        if len(primary_data) == 0:
            return {'error': 'No primary metric data'}
        
        control_data = primary_data[primary_data['variant'] == 'control']['metric_value'].values
        
        if len(control_data) == 0:
            return {'error': 'No control data'}
        
        # Estimate effect size from current data
        observed_effects = {}
        for variant in primary_data['variant'].unique():
            if variant == 'control':
                continue
                
            treatment_data = primary_data[primary_data['variant'] == variant]['metric_value'].values
            
            if len(treatment_data) > 0:
                control_mean = np.mean(control_data)
                treatment_mean = np.mean(treatment_data)
                observed_effect = abs(treatment_mean - control_mean) / control_mean
                observed_effects[variant] = observed_effect
        
        # Calculate required sample size for desired power
        if observed_effects:
            max_effect = max(observed_effects.values())
            required_n = self.analyzer.calculate_sample_size(
                baseline_rate=np.mean(control_data),
                minimum_detectable_effect=max_effect,
                power=config.power,
                significance_level=config.significance_level
            )
            
            current_n = len(control_data)
            power_ratio = current_n / required_n
            
            return {
                'current_sample_size': current_n,
                'required_sample_size': required_n,
                'power_ratio': power_ratio,
                'is_adequately_powered': power_ratio >= 1.0,
                'observed_effects': observed_effects
            }
        
        return {'error': 'Cannot calculate power without treatment data'}
    
    def _get_recommendation(self, primary_results: Dict, guardrail_results: Dict, 
                          power_analysis: Dict) -> str:
        """Get recommendation based on analysis"""
        
        # Check guardrails first
        guardrail_violations = [
            result for result in guardrail_results.values() 
            if isinstance(result, dict) and not result.get('is_healthy', True)
        ]
        
        if guardrail_violations:
            return "STOP: Guardrail violations detected. Consider stopping experiment."
        
        # Check power
        if not power_analysis.get('is_adequately_powered', False):
            return "CONTINUE: Experiment needs more data to reach statistical power."
        
        # Check primary metric significance
        significant_improvements = []
        for variant, analysis in primary_results.items():
            if isinstance(analysis, dict) and analysis.get('is_significant', False):
                if analysis.get('relative_improvement_pct', 0) > 0:
                    significant_improvements.append(variant)
        
        if significant_improvements:
            return f"SUCCESS: Significant improvement detected in {', '.join(significant_improvements)}. Consider graduating experiment."
        
        return "CONTINUE: No significant results yet. Continue monitoring."
```

---

## 📊 Advanced Experimentation Techniques

### 1. **Sequential Testing and Early Stopping**

```python
class SequentialTester:
    def __init__(self, alpha: float = 0.05, beta: float = 0.2):
        self.alpha = alpha  # Type I error rate
        self.beta = beta    # Type II error rate
        
    def calculate_boundaries(self, max_n: int, effect_size: float) -> Dict[str, List[float]]:
        """Calculate sequential testing boundaries (O'Brien-Fleming)"""
        
        # Information fractions
        info_fractions = np.linspace(0.1, 1.0, 10)
        
        # O'Brien-Fleming boundaries
        z_alpha = stats.norm.ppf(1 - self.alpha/2)
        z_beta = stats.norm.ppf(1 - self.beta)
        
        upper_bounds = []
        lower_bounds = []
        
        for t in info_fractions:
            # Efficacy boundary (upper)
            upper_bound = z_alpha / np.sqrt(t)
            upper_bounds.append(upper_bound)
            
            # Futility boundary (lower)
            lower_bound = (z_beta - z_alpha) / np.sqrt(t) + effect_size * np.sqrt(max_n * t)
            lower_bounds.append(lower_bound)
        
        return {
            'information_fractions': info_fractions.tolist(),
            'upper_bounds': upper_bounds,
            'lower_bounds': lower_bounds
        }
    
    def check_early_stopping(self, current_z_score: float, current_n: int, 
                           max_n: int, boundaries: Dict) -> Dict[str, Any]:
        """Check if experiment should stop early"""
        
        info_fraction = current_n / max_n
        
        # Find appropriate boundary
        info_fractions = boundaries['information_fractions']
        upper_bounds = boundaries['upper_bounds']
        lower_bounds = boundaries['lower_bounds']
        
        # Interpolate boundary for current information fraction
        upper_bound = np.interp(info_fraction, info_fractions, upper_bounds)
        lower_bound = np.interp(info_fraction, info_fractions, lower_bounds)
        
        decision = "continue"
        reason = ""
        
        if current_z_score >= upper_bound:
            decision = "stop_for_efficacy"
            reason = f"Z-score {current_z_score:.3f} exceeds efficacy boundary {upper_bound:.3f}"
        elif current_z_score <= lower_bound:
            decision = "stop_for_futility"
            reason = f"Z-score {current_z_score:.3f} below futility boundary {lower_bound:.3f}"
        
        return {
            'decision': decision,
            'reason': reason,
            'current_z_score': current_z_score,
            'upper_bound': upper_bound,
            'lower_bound': lower_bound,
            'information_fraction': info_fraction
        }
```

### 2. **Stratified Randomization**

```python
class StratifiedRandomizer:
    def __init__(self, strata_columns: List[str], treatment_allocation: Dict[str, float]):
        self.strata_columns = strata_columns
        self.treatment_allocation = treatment_allocation
        self.strata_assignments = {}
        
    def assign_treatment(self, user_attributes: Dict[str, Any]) -> str:
        """Assign treatment using stratified randomization"""
        
        # Create stratum key
        stratum_key = tuple(user_attributes.get(col, 'unknown') for col in self.strata_columns)
        
        # Initialize stratum if not exists
        if stratum_key not in self.strata_assignments:
            self.strata_assignments[stratum_key] = {
                'counts': {treatment: 0 for treatment in self.treatment_allocation.keys()},
                'total': 0
            }
        
        stratum = self.strata_assignments[stratum_key]
        
        # Calculate current allocation ratios
        if stratum['total'] == 0:
            # First assignment in stratum - use random
            treatments = list(self.treatment_allocation.keys())
            weights = list(self.treatment_allocation.values())
            treatment = np.random.choice(treatments, p=weights)
        else:
            # Balance allocation within stratum
            current_ratios = {
                treatment: count / stratum['total'] 
                for treatment, count in stratum['counts'].items()
            }
            
            # Find treatment most under-represented
            deviations = {
                treatment: target_ratio - current_ratios[treatment]
                for treatment, target_ratio in self.treatment_allocation.items()
            }
            
            treatment = max(deviations.keys(), key=lambda x: deviations[x])
        
        # Update counts
        stratum['counts'][treatment] += 1
        stratum['total'] += 1
        
        return treatment
```

---

## 🔧 Hands-On Exercise

You'll build a complete A/B testing platform for comparing ML models in production:

### Exercise Scenario
**Company**: StreamTech Entertainment  
**Challenge**: Compare recommendation algorithms for video streaming platform
- **Control**: Collaborative filtering algorithm
- **Treatment A**: Deep learning recommendation model  
- **Treatment B**: Hybrid model (collaborative + content-based)

### Requirements
1. **Experiment Design**: Configure A/B test with proper statistical parameters
2. **Traffic Splitting**: Implement consistent user assignment
3. **Statistical Analysis**: Real-time significance testing
4. **Multi-Armed Bandit**: Dynamic traffic allocation based on performance
5. **Monitoring**: Guardrail metrics and early stopping rules

---

## 📚 Key Takeaways

- **Design experiments carefully** with proper statistical power and significance levels
- **Use consistent hashing** for deterministic user assignment across sessions
- **Monitor guardrail metrics** to prevent degradation of critical business metrics
- **Implement early stopping** to reduce experiment duration and risk
- **Consider multi-armed bandits** for dynamic optimization during experiments
- **Stratify randomization** when user segments have different baseline behaviors
- **Account for multiple comparisons** when testing multiple variants or metrics
- **Plan for practical significance** beyond just statistical significance

---

## 🔄 What's Next?

Tomorrow, we'll explore **Model Versioning with DVC** where you'll learn how to:
- Implement version control for ML models and datasets
- Build reproducible ML pipelines with data lineage
- Handle model rollbacks and deployment strategies
- Create automated model registry and artifact management

The A/B testing framework you build today will integrate with model versioning to enable safe, data-driven model deployments.

---

## 📖 Additional Resources

### Statistical Methods
- [A/B Testing Statistics Guide](https://www.evanmiller.org/ab-testing/)
- [Sequential Testing Methods](https://en.wikipedia.org/wiki/Sequential_analysis)
- [Multiple Comparisons Problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem)

### Multi-Armed Bandits
- [Bandit Algorithms Book](https://banditalgs.com/)
- [Thompson Sampling Tutorial](https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf)
- [UCB Algorithm Analysis](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf)

### Experimentation Platforms
- [Netflix Experimentation Platform](https://netflixtechblog.com/its-all-a-bout-testing-the-netflix-experimentation-platform-4e1ca458c15)
- [Uber's Experimentation Platform](https://eng.uber.com/experimentation-platform/)
- [Facebook's Planout Framework](https://facebook.github.io/planout/)

### Business Applications
- [ML A/B Testing Best Practices](https://blog.ml.cmu.edu/2020/08/31/5-a-b-testing/)
- [Experimentation in ML Systems](https://papers.nips.cc/paper/2019/file/5878a7ab84fb43402106c575658472fa-Paper.pdf)
- [Causal Inference in Experiments](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB)