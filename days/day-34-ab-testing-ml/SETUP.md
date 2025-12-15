# Day 34: A/B Testing for ML - Setup Guide

## 🚀 Quick Start (5 minutes)

### Prerequisites
- Python 3.9+
- Basic understanding of statistics and A/B testing
- Familiarity with ML model evaluation

### 1. Environment Setup
```bash
cd days/day-34-ab-testing-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Basic Testing
```bash
# Run the exercise (guided implementation)
python exercise.py

# Run the complete solution
python solution.py

# Run comprehensive tests
python test_ab_framework.py

# Start the API server
python api.py
```

### 3. API Testing
```bash
# Test API endpoints
curl http://localhost:8000/health

# Create experiment via API
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d @example_experiment.json
```

---

## 📋 Detailed Setup Instructions

### Statistical Libraries Setup

#### 1. Core Dependencies
```bash
# Essential packages
pip install numpy pandas scipy

# Statistical analysis
pip install statsmodels

# Visualization
pip install matplotlib seaborn plotly

# API framework
pip install fastapi uvicorn pydantic
```

#### 2. Verify Statistical Setup
```bash
python -c "
import numpy as np
import pandas as pd
import scipy.stats as stats
print('✅ Statistical libraries installed successfully')

# Test basic functionality
control = np.random.normal(100, 15, 1000)
treatment = np.random.normal(105, 15, 1000)
t_stat, p_value = stats.ttest_ind(control, treatment)
print(f'✅ Statistical test working: p-value = {p_value:.4f}')
"
```

### Framework Components

#### 1. A/B Testing Framework
```bash
# Test framework components
python -c "
from solution import ExperimentConfig, TrafficSplitter, StatisticalAnalyzer
from solution import EpsilonGreedyBandit, ExperimentManager

print('✅ A/B testing framework imported successfully')

# Test basic functionality
config = ExperimentConfig(
    experiment_id='test',
    name='Test',
    description='Test',
    variants={'control': {}, 'treatment': {}},
    traffic_allocation={'control': 0.5, 'treatment': 0.5},
    primary_metric='conversion_rate'
)
print('✅ Experiment configuration created')

splitter = TrafficSplitter(config)
variant = splitter.assign_variant('user_123', 'test')
print(f'✅ Traffic splitting working: user assigned to {variant}')
"
```

#### 2. Multi-Armed Bandits
```bash
# Test bandit algorithms
python -c "
from solution import EpsilonGreedyBandit, UCBBandit, ThompsonSamplingBandit
import numpy as np

# Test epsilon-greedy
bandit = EpsilonGreedyBandit(n_arms=3, epsilon=0.1)
for i in range(10):
    arm = bandit.select_arm()
    reward = np.random.random()
    bandit.update(arm, reward)

stats = bandit.get_statistics()
print(f'✅ Epsilon-greedy bandit working: {stats[\"total_pulls\"]} pulls')

# Test UCB
ucb_bandit = UCBBandit(n_arms=3)
arm = ucb_bandit.select_arm()
ucb_bandit.update(arm, 0.5)
print('✅ UCB bandit working')

# Test Thompson Sampling
ts_bandit = ThompsonSamplingBandit(n_arms=3)
arm = ts_bandit.select_arm()
ts_bandit.update(arm, 1.0)
print('✅ Thompson Sampling bandit working')
"
```

### API Server Setup

#### 1. Start API Server
```bash
# Start development server
python api.py

# Or use uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

#### 2. Test API Endpoints
```bash
# Health check
curl http://localhost:8000/health

# List experiments
curl http://localhost:8000/experiments

# Get API documentation
open http://localhost:8000/docs
```

#### 3. Create Sample Experiment
```bash
# Create experiment configuration file
cat > example_experiment.json << EOF
{
  "experiment_id": "rec_test_001",
  "name": "Recommendation Algorithm Test",
  "description": "Compare collaborative filtering vs deep learning",
  "variants": {
    "control": {"model": "collaborative_filtering"},
    "treatment": {"model": "deep_learning"}
  },
  "traffic_allocation": {
    "control": 0.5,
    "treatment": 0.5
  },
  "primary_metric": "click_through_rate",
  "secondary_metrics": ["watch_time", "user_satisfaction"],
  "significance_level": 0.05,
  "power": 0.8,
  "minimum_detectable_effect": 0.05,
  "min_sample_size": 1000
}
EOF

# Create experiment via API
curl -X POST http://localhost:8000/experiments \
  -H "Content-Type: application/json" \
  -d @example_experiment.json

# Start the experiment
curl -X POST http://localhost:8000/experiments/rec_test_001/start
```

---

## 🧪 Testing and Validation

### 1. Unit Testing
```bash
# Run comprehensive test suite
python test_ab_framework.py

# Run with pytest for detailed output
pytest test_ab_framework.py -v

# Run specific test categories
pytest test_ab_framework.py::TestStatisticalAnalyzer -v
pytest test_ab_framework.py::TestMultiArmedBandits -v
```

### 2. Integration Testing
```bash
# Test complete workflow
python -c "
from solution import StreamingPlatform, ExperimentConfig
from datetime import datetime

# Initialize platform
platform = StreamingPlatform()

# Create experiment
config = ExperimentConfig(
    experiment_id='integration_test',
    name='Integration Test',
    description='Test complete workflow',
    variants={'control': {}, 'treatment': {}},
    traffic_allocation={'control': 0.5, 'treatment': 0.5},
    primary_metric='conversion_rate'
)

# Run simulation
results = platform.run_experiment_simulation(config, n_users=100, n_days=1)
print(f'✅ Integration test completed: {results[\"sample_size\"]} samples')
"
```

### 3. Performance Testing
```bash
# Test traffic splitting performance
python -c "
from solution import TrafficSplitter, ExperimentConfig
import time

config = ExperimentConfig(
    experiment_id='perf_test',
    name='Performance Test',
    description='Test performance',
    variants={'control': {}, 'treatment': {}},
    traffic_allocation={'control': 0.5, 'treatment': 0.5},
    primary_metric='ctr'
)

splitter = TrafficSplitter(config)

start_time = time.time()
for i in range(10000):
    variant = splitter.assign_variant(f'user_{i}', 'perf_test')
end_time = time.time()

rate = 10000 / (end_time - start_time)
print(f'✅ Traffic splitting: {rate:.0f} assignments/second')
"
```

### 4. Statistical Validation
```bash
# Validate statistical methods
python -c "
from solution import StatisticalAnalyzer
import numpy as np

analyzer = StatisticalAnalyzer()

# Test with known effect
np.random.seed(42)
control = np.random.normal(100, 15, 1000)
treatment = np.random.normal(105, 15, 1000)  # 5% improvement

result = analyzer.analyze_continuous_metric(control, treatment)
print(f'✅ Statistical analysis: {result[\"relative_improvement_pct\"]:.1f}% improvement detected')
print(f'✅ Significance: {\"Yes\" if result[\"is_significant\"] else \"No\"} (p={result[\"p_value\"]:.4f})')
"
```

---

## 🔧 Configuration Options

### Experiment Configuration
```python
# Example comprehensive experiment configuration
config = ExperimentConfig(
    experiment_id="advanced_test",
    name="Advanced A/B Test",
    description="Comprehensive experiment with all features",
    
    # Variants
    variants={
        'control': {'model': 'baseline', 'version': '1.0'},
        'treatment_a': {'model': 'new_model', 'version': '2.0'},
        'treatment_b': {'model': 'hybrid', 'version': '1.5'}
    },
    
    # Traffic allocation
    traffic_allocation={
        'control': 0.4,
        'treatment_a': 0.3,
        'treatment_b': 0.3
    },
    
    # Metrics
    primary_metric='conversion_rate',
    secondary_metrics=['click_through_rate', 'engagement_score'],
    
    # Statistical parameters
    significance_level=0.05,
    power=0.8,
    minimum_detectable_effect=0.05,
    
    # Guardrails
    guardrail_metrics={
        'latency': {'max': 100},  # Max 100ms latency
        'error_rate': {'max': 0.01}  # Max 1% error rate
    },
    
    # Eligibility
    eligibility_criteria={
        'allowed_countries': ['US', 'CA', 'UK'],
        'min_account_age_days': 30,
        'allowed_subscription_types': ['premium', 'free']
    }
)
```

### Bandit Configuration
```python
# Epsilon-greedy bandit
bandit_config = {
    'use_bandit': True,
    'bandit_algorithm': 'epsilon_greedy',
    'bandit_params': {'epsilon': 0.1}
}

# UCB bandit
bandit_config = {
    'use_bandit': True,
    'bandit_algorithm': 'ucb',
    'bandit_params': {'confidence_level': 2.0}
}

# Thompson Sampling
bandit_config = {
    'use_bandit': True,
    'bandit_algorithm': 'thompson_sampling',
    'bandit_params': {}
}
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Statistical Library Issues
```bash
# Problem: scipy import error
# Solution: Install scipy
pip install scipy

# Problem: Statistical tests failing
# Check: Verify data types and sample sizes
python -c "
import numpy as np
data = np.array([1, 2, 3, 4, 5])
print(f'Data type: {data.dtype}')
print(f'Sample size: {len(data)}')
"
```

#### 2. Traffic Splitting Issues
```bash
# Problem: Inconsistent user assignments
# Check: Hash function consistency
python -c "
import hashlib
user_id = 'test_user'
exp_id = 'test_exp'
hash1 = hashlib.md5(f'{user_id}:{exp_id}'.encode()).hexdigest()
hash2 = hashlib.md5(f'{user_id}:{exp_id}'.encode()).hexdigest()
print(f'Hash consistency: {hash1 == hash2}')
"

# Problem: Traffic allocation not balanced
# Check: Large sample size for validation
python -c "
from solution import TrafficSplitter, ExperimentConfig
import pandas as pd

config = ExperimentConfig(
    experiment_id='balance_test',
    name='Balance Test',
    description='Test',
    variants={'control': {}, 'treatment': {}},
    traffic_allocation={'control': 0.5, 'treatment': 0.5},
    primary_metric='ctr'
)

splitter = TrafficSplitter(config)
assignments = [splitter.assign_variant(f'user_{i}', 'balance_test') for i in range(10000)]
distribution = pd.Series(assignments).value_counts(normalize=True)
print('Traffic distribution:')
print(distribution)
"
```

#### 3. API Issues
```bash
# Problem: API server won't start
# Check: Port availability
lsof -i :8000

# Problem: Import errors in API
# Check: Python path and imports
python -c "
try:
    from solution import ExperimentManager
    print('✅ Imports working')
except ImportError as e:
    print(f'❌ Import error: {e}')
"
```

#### 4. Statistical Analysis Issues
```bash
# Problem: P-values always 1.0 or 0.0
# Check: Data quality and sample sizes
python -c "
import numpy as np
from solution import StatisticalAnalyzer

# Generate test data
control = np.random.normal(100, 15, 100)
treatment = np.random.normal(100, 15, 100)  # No difference

analyzer = StatisticalAnalyzer()
result = analyzer.analyze_continuous_metric(control, treatment)
print(f'No difference test - p-value: {result[\"p_value\"]:.4f}')

# With difference
treatment_diff = np.random.normal(110, 15, 100)  # 10% difference
result_diff = analyzer.analyze_continuous_metric(control, treatment_diff)
print(f'With difference test - p-value: {result_diff[\"p_value\"]:.4f}')
"
```

### Performance Optimization

#### 1. Traffic Splitting Optimization
```python
# Use efficient hashing for high-volume assignments
# Cache variant boundaries for repeated use
# Consider using faster hash functions for extreme scale
```

#### 2. Statistical Analysis Optimization
```python
# Use vectorized operations with NumPy
# Cache statistical computations for repeated analysis
# Consider approximate methods for very large datasets
```

#### 3. Memory Management
```python
# Limit result storage for long-running experiments
# Use data streaming for large-scale analysis
# Implement result archiving for historical data
```

---

## 📊 Example Workflows

### 1. Basic A/B Test
```python
from solution import StreamingPlatform, ExperimentConfig

# Initialize platform
platform = StreamingPlatform()

# Create experiment
config = ExperimentConfig(
    experiment_id="basic_test",
    name="Basic A/B Test",
    description="Compare two recommendation models",
    variants={'control': {}, 'treatment': {}},
    traffic_allocation={'control': 0.5, 'treatment': 0.5},
    primary_metric='click_through_rate'
)

# Run experiment
exp_id = platform.experiment_manager.create_experiment(config)
platform.experiment_manager.start_experiment(exp_id)

# Get user assignment
variant = platform.experiment_manager.get_assignment(
    exp_id, "user_123", {'country': 'US'}
)

# Record result
platform.experiment_manager.record_result(
    exp_id, "user_123", "click_through_rate", 0.15
)

# Get analysis
status = platform.monitor.get_experiment_status(exp_id)
print(status['recommendation'])
```

### 2. Multi-Armed Bandit
```python
from solution import ExperimentConfig, StreamingPlatform

# Create bandit experiment
config = ExperimentConfig(
    experiment_id="bandit_test",
    name="Multi-Armed Bandit Test",
    description="Dynamic traffic allocation",
    variants={'control': {}, 'treatment_a': {}, 'treatment_b': {}},
    traffic_allocation={'control': 0.33, 'treatment_a': 0.33, 'treatment_b': 0.34},
    primary_metric='conversion_rate',
    use_bandit=True,
    bandit_algorithm='epsilon_greedy',
    bandit_params={'epsilon': 0.1}
)

platform = StreamingPlatform()
exp_id = platform.experiment_manager.create_experiment(config)
platform.experiment_manager.start_experiment(exp_id)

# Bandit will automatically adjust traffic allocation based on performance
```

### 3. Statistical Analysis
```python
from solution import StatisticalAnalyzer
import numpy as np

analyzer = StatisticalAnalyzer()

# Continuous metric analysis
control_data = np.random.normal(25.5, 5, 1000)  # Watch time
treatment_data = np.random.normal(27.2, 5, 1000)  # 6.7% improvement

result = analyzer.analyze_continuous_metric(control_data, treatment_data)
print(f"Improvement: {result['relative_improvement_pct']:.1f}%")
print(f"Significant: {result['is_significant']}")

# Binary metric analysis
result = analyzer.analyze_binary_metric(
    control_successes=120, control_total=1000,
    treatment_successes=144, treatment_total=1000
)
print(f"CTR improvement: {result['relative_improvement_pct']:.1f}%")
```

---

## ✅ Verification Checklist

After setup, verify:
- [ ] Python environment activated
- [ ] All dependencies installed (`pip list | grep -E "(numpy|pandas|scipy|fastapi)"`)
- [ ] Statistical libraries working (`python -c "import scipy.stats; print('OK')"`)
- [ ] Framework components importable (`python -c "from solution import *; print('OK')"`)
- [ ] Traffic splitting working (run basic test)
- [ ] Statistical analysis working (run test with known data)
- [ ] API server starts without errors
- [ ] API endpoints respond (`curl http://localhost:8000/health`)
- [ ] Test suite passes (`python test_ab_framework.py`)
- [ ] Example experiment creation works

**Setup Complete! 🎉**

You're ready to build production A/B testing systems for ML models. Start with the exercise for guided learning, then explore the complete solution for advanced patterns.