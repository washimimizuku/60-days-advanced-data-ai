# Day 48: Fine-tuning Techniques - LoRA & QLoRA - Setup Guide

## Overview

This guide helps you set up the environment for learning and implementing LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) fine-tuning techniques. These methods enable efficient fine-tuning of large language models with minimal computational resources.

## Prerequisites

### Knowledge Requirements
- Understanding of transformer architectures (Day 41-42)
- Familiarity with PyTorch and neural networks
- Basic knowledge of fine-tuning concepts
- Understanding of quantization principles

### Hardware Requirements

#### Minimum Requirements
- **GPU**: 8GB VRAM (RTX 3070, RTX 4060 Ti, or equivalent)
- **RAM**: 16GB system memory
- **Storage**: 20GB free space
- **CPU**: Modern multi-core processor

#### Recommended Requirements
- **GPU**: 16GB+ VRAM (RTX 4080, RTX 4090, A100, or equivalent)
- **RAM**: 32GB+ system memory
- **Storage**: 50GB+ free space (for model downloads)
- **CPU**: High-performance multi-core processor

#### For Large Models (13B+ parameters)
- **GPU**: 24GB+ VRAM (RTX 4090, A100, H100)
- **RAM**: 64GB+ system memory
- **Storage**: 100GB+ free space

## Installation

### 1. Create Virtual Environment

```bash
# Create conda environment
conda create -n lora-qlora python=3.10
conda activate lora-qlora

# Or using venv
python -m venv lora-qlora
source lora-qlora/bin/activate  # Linux/Mac
# lora-qlora\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install manually
pip install torch>=2.0.0 transformers>=4.35.0
pip install peft>=0.6.0 bitsandbytes>=0.41.0
pip install accelerate datasets evaluate
pip install wandb tensorboard matplotlib
```

### 3. Verify GPU Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 4. Test Installation

```bash
python test_fine_tuning.py
```

## Core Concepts

### 1. LoRA (Low-Rank Adaptation)

LoRA decomposes weight updates into low-rank matrices:

```
W' = W + ΔW = W + BA
```

Where:
- `W`: Original frozen weights
- `B ∈ R^(d×r)`, `A ∈ R^(r×k)`: Trainable low-rank matrices
- `r << min(d,k)`: Rank (typically 1-64)

**Key Benefits:**
- **Parameter Efficiency**: Only 0.1-3% of original parameters
- **Memory Efficiency**: Reduced gradient and optimizer memory
- **Modularity**: Multiple adapters can share base model
- **Fast Switching**: Easy task switching in production

### 2. QLoRA (Quantized LoRA)

QLoRA combines LoRA with advanced quantization:

**Innovations:**
- **4-bit NormalFloat (NF4)**: Optimized for normal distributions
- **Double Quantization**: Quantizes quantization constants
- **Paged Optimizers**: Handles memory spikes via unified memory

**Memory Savings:**
- Base model: 4-bit (vs 16-bit)
- LoRA adapters: 16-bit
- Total reduction: 65-80% memory usage

### 3. Parameter-Efficient Fine-Tuning (PEFT)

**Comparison of Methods:**

| Method | Parameters | Memory | Performance | Use Case |
|--------|------------|---------|-------------|----------|
| Full FT | 100% | High | Best | Unlimited resources |
| LoRA | 0.1-3% | Medium | 95-99% | Balanced efficiency |
| QLoRA | 0.1-3% | Low | 90-98% | Limited hardware |
| Adapters | 1-5% | Medium | 90-95% | Task-specific |

## Configuration Guidelines

### 1. Rank Selection

**General Guidelines:**
- **Small tasks** (classification): rank = 8-16
- **Medium tasks** (summarization): rank = 16-32
- **Complex tasks** (instruction following): rank = 32-64
- **Code generation**: rank = 32-128

**Model Size Considerations:**
- **<7B parameters**: rank = 8-32
- **7B-30B parameters**: rank = 16-64
- **>30B parameters**: rank = 32-128

### 2. Alpha Scaling

**Best Practices:**
- **Standard**: alpha = 2 × rank
- **Conservative**: alpha = rank
- **Aggressive**: alpha = 4 × rank

### 3. Target Modules

**Attention Layers (Most Important):**
```python
target_modules = [
    "q_proj",    # Query projection
    "v_proj",    # Value projection
    "k_proj",    # Key projection (optional)
    "o_proj",    # Output projection (optional)
]
```

**MLP Layers (For Complex Tasks):**
```python
target_modules = [
    "q_proj", "v_proj",
    "gate_proj",  # MLP gate
    "up_proj",    # MLP up projection
    "down_proj"   # MLP down projection
]
```

### 4. Task-Specific Configurations

```python
# Instruction Following
config = LoRAConfig(
    rank=64, alpha=128,
    target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Code Generation
config = LoRAConfig(
    rank=32, alpha=64,
    target_modules=["q_proj", "v_proj", "gate_proj", "up_proj"]
)

# Summarization
config = LoRAConfig(
    rank=16, alpha=32,
    target_modules=["q_proj", "v_proj"]
)
```

## Memory Optimization Strategies

### 1. Gradient Checkpointing

```python
model.gradient_checkpointing_enable()
# Reduces activation memory by ~30%
# Increases training time by ~20%
```

### 2. Mixed Precision Training

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

### 3. Gradient Accumulation

```python
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    with autocast():
        outputs = model(**batch)
        loss = outputs.loss / accumulation_steps
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

### 4. Optimizer Offloading

```python
# Using DeepSpeed ZeRO
deepspeed_config = {
    "zero_optimization": {
        "stage": 2,
        "offload_optimizer": {"device": "cpu"}
    }
}
```

## Performance Optimization

### 1. Batch Size Optimization

```python
def find_optimal_batch_size(model, max_memory_gb=24):
    """Binary search for optimal batch size"""
    low, high = 1, 64
    optimal = 1
    
    while low <= high:
        mid = (low + high) // 2
        try:
            # Test memory usage with this batch size
            test_batch = create_test_batch(batch_size=mid)
            _ = model(**test_batch)
            torch.cuda.empty_cache()
            
            optimal = mid
            low = mid + 1
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
    
    return optimal
```

### 2. Learning Rate Scheduling

```python
from transformers import get_cosine_schedule_with_warmup

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)
```

### 3. Data Loading Optimization

```python
dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,
    pin_memory=True,
    prefetch_factor=2
)
```

## Monitoring and Evaluation

### 1. Memory Monitoring

```python
import psutil
import torch

def monitor_memory():
    # GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / 1e9
        gpu_cached = torch.cuda.memory_reserved() / 1e9
        print(f"GPU Memory: {gpu_memory:.1f}GB allocated, {gpu_cached:.1f}GB cached")
    
    # System memory
    system_memory = psutil.virtual_memory()
    print(f"System Memory: {system_memory.used / 1e9:.1f}GB / {system_memory.total / 1e9:.1f}GB")
```

### 2. Training Metrics

```python
import wandb

# Initialize tracking
wandb.init(project="lora-fine-tuning")

# Log metrics
wandb.log({
    "train_loss": loss.item(),
    "learning_rate": scheduler.get_last_lr()[0],
    "gpu_memory_gb": torch.cuda.memory_allocated() / 1e9,
    "tokens_per_second": tokens_per_second
})
```

### 3. Model Evaluation

```python
from evaluate import load

# Load metrics
bleu = load("bleu")
rouge = load("rouge")

def evaluate_model(model, eval_dataset):
    model.eval()
    predictions, references = [], []
    
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model.generate(**batch, max_length=128)
            predictions.extend(tokenizer.batch_decode(outputs))
            references.extend(batch["labels"])
    
    # Calculate metrics
    bleu_score = bleu.compute(predictions=predictions, references=references)
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    
    return {"bleu": bleu_score, "rouge": rouge_scores}
```

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

**Solutions:**
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation
- Apply mixed precision training
- Try QLoRA instead of LoRA

#### 2. Slow Training

**Solutions:**
- Increase batch size (if memory allows)
- Use multiple GPUs with DataParallel
- Optimize data loading (more workers, pin_memory)
- Use compiled models (torch.compile)

#### 3. Poor Performance

**Solutions:**
- Increase LoRA rank
- Add more target modules
- Adjust learning rate
- Use better data preprocessing
- Try different alpha scaling

#### 4. Convergence Issues

**Solutions:**
- Use learning rate scheduling
- Apply gradient clipping
- Check data quality
- Adjust warmup steps
- Monitor gradient norms

### Debugging Commands

```bash
# Check GPU utilization
nvidia-smi

# Monitor GPU memory
watch -n 1 nvidia-smi

# Profile memory usage
python -m memory_profiler exercise.py

# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

## Best Practices

### 1. Development Workflow

1. **Start Small**: Begin with small models and datasets
2. **Baseline First**: Establish baseline performance
3. **Iterative Improvement**: Gradually increase complexity
4. **Monitor Everything**: Track metrics, memory, and performance
5. **Save Checkpoints**: Regular checkpointing for long training

### 2. Production Deployment

1. **Adapter Management**: Organize adapters by task/version
2. **Dynamic Loading**: Enable runtime adapter switching
3. **Performance Testing**: Benchmark inference speed
4. **Memory Profiling**: Monitor production memory usage
5. **A/B Testing**: Compare adapter performance

### 3. Hyperparameter Tuning

1. **Grid Search**: Systematic exploration of rank/alpha
2. **Random Search**: Efficient exploration of hyperspace
3. **Bayesian Optimization**: Advanced hyperparameter optimization
4. **Early Stopping**: Prevent overfitting
5. **Cross Validation**: Robust performance estimation

## Resources

### Documentation
- [Hugging Face PEFT](https://huggingface.co/docs/peft/index)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)

### Tutorials
- [PEFT Tutorial](https://huggingface.co/docs/peft/tutorial/peft_model_config)
- [QLoRA Tutorial](https://huggingface.co/blog/4bit-transformers-bitsandbytes)
- [Memory Optimization Guide](https://huggingface.co/docs/transformers/perf_train_gpu_one)

### Model Repositories
- [Hugging Face Models](https://huggingface.co/models)
- [LoRA Adapters](https://huggingface.co/models?library=peft)
- [Quantized Models](https://huggingface.co/models?search=GPTQ)

## Next Steps

After completing the setup:

1. **Run Exercises**: Complete all exercises in `exercise.py`
2. **Experiment**: Try different configurations and models
3. **Build Projects**: Create your own fine-tuning projects
4. **Deploy**: Set up production inference systems
5. **Contribute**: Share your adapters and improvements

Remember: LoRA and QLoRA are powerful techniques that democratize fine-tuning of large models. Start with the exercises and gradually work up to larger, more complex scenarios!