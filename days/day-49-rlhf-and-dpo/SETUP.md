# Day 49: RLHF and DPO - Human Feedback & Preference Learning - Setup Guide

## Overview

This guide helps you set up the environment for learning and implementing RLHF (Reinforcement Learning from Human Feedback) and DPO (Direct Preference Optimization) techniques. These methods are crucial for aligning AI systems with human values and preferences.

## Prerequisites

### Knowledge Requirements
- Understanding of transformer architectures and fine-tuning (Days 41-48)
- Familiarity with reinforcement learning concepts
- Basic knowledge of human-computer interaction principles
- Understanding of AI safety and alignment concepts

### Hardware Requirements

#### Minimum Requirements
- **GPU**: 16GB VRAM (RTX 4080, RTX 4090, or equivalent)
- **RAM**: 32GB system memory
- **Storage**: 50GB free space
- **CPU**: High-performance multi-core processor

#### Recommended Requirements
- **GPU**: 24GB+ VRAM (RTX 4090, A100, H100)
- **RAM**: 64GB+ system memory
- **Storage**: 100GB+ free space (for datasets and models)
- **CPU**: Server-grade multi-core processor

#### For Large-Scale Training
- **Multi-GPU**: 4x A100 or H100 GPUs
- **RAM**: 128GB+ system memory
- **Storage**: 500GB+ NVMe SSD
- **Network**: High-bandwidth for distributed training

## Installation

### 1. Create Virtual Environment

```bash
# Create conda environment
conda create -n rlhf-dpo python=3.10
conda activate rlhf-dpo

# Or using venv
python -m venv rlhf-dpo
source rlhf-dpo/bin/activate  # Linux/Mac
# rlhf-dpo\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install torch>=2.0.0 transformers>=4.35.0
pip install trl>=0.7.0 peft>=0.6.0
pip install datasets evaluate wandb
pip install detoxify perspective-api
```

### 3. Verify Installation

```python
import torch
import transformers
import trl
from transformers import AutoTokenizer, AutoModelForCausalLM

print(f"PyTorch version: {torch.__version__}")
print(f"Transformers version: {transformers.__version__}")
print(f"TRL version: {trl.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 4. Test Setup

```bash
python test_rlhf_dpo.py
```

## Core Concepts

### 1. AI Alignment Problem

AI alignment ensures that AI systems behave according to human values and intentions:

**Key Challenges:**
- **Specification Problem**: Difficulty defining what we want AI to do
- **Goodhart's Law**: Optimizing metrics can lead to unintended consequences
- **Distributional Shift**: Models behave differently in new contexts
- **Scalable Oversight**: Human evaluation doesn't scale to all outputs

### 2. RLHF (Reinforcement Learning from Human Feedback)

RLHF is a three-stage process:

```
Stage 1: Supervised Fine-tuning (SFT)
├── High-quality demonstrations
├── Instruction-following format
└── Foundation for alignment

Stage 2: Reward Model Training
├── Human preference data collection
├── Pairwise comparison training
└── Preference prediction model

Stage 3: Reinforcement Learning (PPO)
├── Policy optimization against reward model
├── KL divergence regularization
└── Aligned model output
```

**Benefits:**
- Captures complex human preferences
- Enables iterative improvement
- Scales beyond simple metrics

**Challenges:**
- Training instability (RL)
- Reward model accuracy
- Human annotation costs

### 3. DPO (Direct Preference Optimization)

DPO simplifies RLHF by directly optimizing preferences:

**Key Innovation:**
```
Traditional: Policy → Reward Model → RL Optimization
DPO: Policy ← Direct Preference Optimization
```

**Mathematical Foundation:**
```
DPO Loss = -log(σ(β(log π_θ(y_w|x) - log π_θ(y_l|x) - log π_ref(y_w|x) + log π_ref(y_l|x))))
```

Where:
- `π_θ`: Policy being optimized
- `π_ref`: Reference policy
- `y_w`: Preferred response
- `y_l`: Less preferred response
- `β`: Temperature parameter

**Advantages:**
- Simpler training pipeline
- More stable than RL
- Competitive performance
- Lower computational cost

### 4. Constitutional AI

Constitutional AI extends alignment through AI feedback:

**Process:**
1. **Critique**: AI evaluates responses against principles
2. **Revise**: AI improves responses based on critique
3. **Train**: Use revised responses as preference data

**Benefits:**
- Scalable oversight
- Consistent principles
- Reduced human annotation burden

## Configuration Guidelines

### 1. SFT (Supervised Fine-tuning) Setup

```python
sft_config = {
    "learning_rate": 1e-5,
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "num_epochs": 3,
    "warmup_ratio": 0.1,
    "max_length": 512,
    "data_format": "instruction_following"
}
```

### 2. Reward Model Configuration

```python
reward_model_config = {
    "learning_rate": 1e-5,
    "batch_size": 8,
    "num_epochs": 3,
    "margin": 0.5,
    "hidden_size": 768,
    "dropout": 0.1,
    "freeze_base_model": True
}
```

### 3. PPO Training Setup

```python
ppo_config = {
    "learning_rate": 1e-6,
    "batch_size": 32,
    "mini_batch_size": 4,
    "gradient_accumulation_steps": 8,
    "kl_coeff": 0.1,
    "clip_range": 0.2,
    "value_coeff": 0.5,
    "entropy_coeff": 0.01,
    "max_grad_norm": 1.0
}
```

### 4. DPO Configuration

```python
dpo_config = {
    "learning_rate": 1e-6,
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "beta": 0.1,
    "num_epochs": 3,
    "max_length": 512,
    "warmup_ratio": 0.1
}
```

## Human Preference Data Collection

### 1. Annotation Guidelines

**Quality Criteria:**
- **Helpfulness**: How well does the response address the user's request?
- **Harmlessness**: Is the response safe and non-toxic?
- **Honesty**: Is the response truthful and acknowledges uncertainty?
- **Clarity**: Is the response clear and well-structured?

**Annotation Format:**
```json
{
  "prompt": "User's question or request",
  "responses": [
    "Response option A",
    "Response option B"
  ],
  "preference": 0,  // Index of preferred response
  "explanation": "Why this response is better",
  "criteria_scores": {
    "helpfulness": 4,
    "harmlessness": 5,
    "honesty": 4,
    "clarity": 3
  }
}
```

### 2. Quality Control Measures

**Inter-Annotator Agreement:**
```python
def compute_agreement(annotations):
    agreements = []
    for i in range(len(annotations)):
        for j in range(i + 1, len(annotations)):
            agreement = (annotations[i] == annotations[j])
            agreements.append(agreement)
    return sum(agreements) / len(agreements)
```

**Validation Checks:**
- Annotation time (too fast = low quality)
- Consistency across similar examples
- Explanation quality and detail
- Agreement with expert annotations

### 3. Bias Mitigation

**Strategies:**
- Diverse annotator pool
- Blind annotation (hide model source)
- Multiple annotations per example
- Regular calibration sessions
- Bias detection algorithms

## Safety and Monitoring

### 1. Safety Classifiers

**Toxicity Detection:**
```python
from detoxify import Detoxify

detoxify = Detoxify('original')

def check_toxicity(text):
    results = detoxify.predict(text)
    return {
        'toxic': results['toxicity'] > 0.7,
        'score': results['toxicity']
    }
```

**Bias Detection:**
```python
def detect_bias(text):
    bias_indicators = [
        'always', 'never', 'all women', 'all men',
        'stereotype', 'typical', 'naturally'
    ]
    
    bias_count = sum(1 for indicator in bias_indicators 
                    if indicator in text.lower())
    
    return {
        'biased': bias_count > 2,
        'score': min(1.0, bias_count / 5.0)
    }
```

### 2. Real-time Monitoring

**Metrics to Track:**
- Safety score distribution
- Response quality metrics
- User satisfaction ratings
- Alignment drift indicators

**Alert Thresholds:**
```python
monitoring_thresholds = {
    'safety_score': 0.8,      # Minimum safety score
    'toxicity_rate': 0.05,    # Maximum toxicity rate
    'bias_rate': 0.1,         # Maximum bias rate
    'user_satisfaction': 0.7   # Minimum satisfaction
}
```

### 3. Deployment Safety

**Staged Rollout:**
1. **Canary**: 1% of traffic
2. **Limited**: 10% of traffic
3. **Gradual**: 50% of traffic
4. **Full**: 100% of traffic

**Rollback Conditions:**
- Safety score drops below threshold
- Error rate exceeds limit
- User complaints spike
- Manual intervention required

## Performance Optimization

### 1. Memory Optimization

**Gradient Checkpointing:**
```python
model.gradient_checkpointing_enable()
```

**Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    outputs = model(**batch)
    loss = outputs.loss

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**DeepSpeed Integration:**
```python
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    },
    "fp16": {"enabled": True},
    "gradient_clipping": 1.0
}
```

### 2. Data Loading Optimization

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True
)
```

### 3. Distributed Training

```python
# Multi-GPU training
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=12345 \
    train_rlhf.py

# DeepSpeed training
deepspeed train_rlhf.py \
    --deepspeed ds_config.json \
    --num_gpus=4
```

## Evaluation Metrics

### 1. Alignment Metrics

**Safety Metrics:**
- Toxicity rate
- Bias detection rate
- Harmful content rate
- Safety classifier scores

**Quality Metrics:**
- Helpfulness ratings
- Factual accuracy
- Response coherence
- Task completion rate

**Preference Metrics:**
- Human preference accuracy
- Reward model correlation
- Constitutional adherence
- Value alignment scores

### 2. Technical Metrics

**Training Metrics:**
- Loss convergence
- Gradient norms
- Learning rate schedules
- Training stability

**Inference Metrics:**
- Response latency
- Throughput (tokens/sec)
- Memory usage
- GPU utilization

### 3. Evaluation Framework

```python
class AlignmentEvaluator:
    def __init__(self):
        self.safety_classifiers = self.load_classifiers()
        self.human_evaluators = self.setup_evaluation_pool()
    
    def evaluate_model(self, model, test_data):
        results = {
            'safety_scores': self.evaluate_safety(model, test_data),
            'quality_scores': self.evaluate_quality(model, test_data),
            'preference_scores': self.evaluate_preferences(model, test_data),
            'technical_metrics': self.measure_performance(model, test_data)
        }
        return results
```

## Troubleshooting

### Common Issues

#### 1. Training Instability

**Symptoms:**
- Loss oscillations
- Gradient explosions
- Poor convergence

**Solutions:**
- Reduce learning rate
- Apply gradient clipping
- Use learning rate scheduling
- Check data quality

#### 2. Reward Model Overfitting

**Symptoms:**
- High training accuracy, low validation
- Poor generalization to new data
- Inconsistent preferences

**Solutions:**
- Increase regularization
- Use more diverse training data
- Apply early stopping
- Cross-validate annotations

#### 3. Alignment Drift

**Symptoms:**
- Degrading safety scores
- Increasing bias rates
- User complaint increases

**Solutions:**
- Implement continuous monitoring
- Regular model updates
- Feedback loop integration
- Safety filter updates

#### 4. Memory Issues

**Symptoms:**
- Out of memory errors
- Slow training speed
- GPU underutilization

**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Apply model parallelism

### Debugging Commands

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Profile memory usage
python -m memory_profiler train_rlhf.py

# Check model outputs
python debug_model.py --model_path ./checkpoints/model

# Validate data quality
python validate_preferences.py --data_path ./data/preferences.json
```

## Best Practices

### 1. Data Quality

- **Diverse Sources**: Collect preferences from diverse populations
- **Clear Guidelines**: Provide detailed annotation instructions
- **Quality Control**: Implement multiple validation layers
- **Iterative Improvement**: Continuously refine guidelines

### 2. Model Training

- **Staged Approach**: SFT → Reward Model → RL/DPO
- **Regularization**: Prevent overfitting and maintain capabilities
- **Monitoring**: Track alignment metrics throughout training
- **Checkpointing**: Save models at regular intervals

### 3. Safety Measures

- **Multi-layered Defense**: Combine multiple safety mechanisms
- **Real-time Monitoring**: Continuous safety assessment
- **Human Oversight**: Maintain human-in-the-loop systems
- **Rapid Response**: Quick rollback capabilities

### 4. Deployment Strategy

- **Gradual Rollout**: Staged deployment with monitoring
- **A/B Testing**: Compare aligned vs. baseline models
- **Feedback Collection**: Gather user feedback continuously
- **Continuous Improvement**: Regular model updates

## Resources

### Documentation
- [TRL Documentation](https://huggingface.co/docs/trl/index)
- [InstructGPT Paper](https://arxiv.org/abs/2203.02155)
- [DPO Paper](https://arxiv.org/abs/2305.18290)
- [Constitutional AI Paper](https://arxiv.org/abs/2212.08073)

### Datasets
- [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf)
- [OpenAI WebGPT](https://huggingface.co/datasets/openai/webgpt_comparisons)
- [Stanford SHP](https://huggingface.co/datasets/stanfordnlp/SHP)

### Tools and Libraries
- [Hugging Face TRL](https://github.com/huggingface/trl)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [Weights & Biases](https://wandb.ai/)
- [Detoxify](https://github.com/unitaryai/detoxify)

## Next Steps

After completing the setup:

1. **Run Exercises**: Complete all exercises in `exercise.py`
2. **Experiment**: Try different alignment techniques
3. **Collect Data**: Gather human preference data
4. **Train Models**: Implement RLHF and DPO pipelines
5. **Deploy Safely**: Use staged rollouts with monitoring

Remember: AI alignment is an active area of research with significant safety implications. Always prioritize safety and ethical considerations in your implementations!