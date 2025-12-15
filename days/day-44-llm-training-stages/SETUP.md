# Day 44: LLM Training Stages - Setup Guide

## 📋 Overview

This setup guide will help you prepare your environment for implementing and experimenting with large language model training stages including pre-training, fine-tuning, and alignment techniques.

## 🔧 Prerequisites

- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU training)
- **Memory**: 16GB+ RAM (32GB+ recommended)
- **Storage**: 50GB+ free space for models and datasets
- **GPU**: 8GB+ VRAM recommended (24GB+ for larger models)

## 📦 Installation

### 1. Create Virtual Environment

```bash
# Create virtual environment
python -m venv llm_training_env

# Activate environment
# On macOS/Linux:
source llm_training_env/bin/activate
# On Windows:
llm_training_env\Scripts\activate
```

### 2. Install Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt

# Optional: Install advanced libraries
# pip install flash-attn  # For memory-efficient attention
# pip install apex       # For mixed precision training
```

### 3. Verify Installation

```bash
# Run verification script
python -c "
import torch
import transformers
import accelerate
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
    print(f'CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'Transformers version: {transformers.__version__}')
print('✅ All dependencies installed successfully!')
"
```

## 🧠 LLM Training Fundamentals

### Three-Stage Training Pipeline

#### **Stage 1: Pre-training**
```
Objective: Learn general language understanding
Data: Massive text corpora (TB scale)
Method: Next-token prediction (autoregressive)
Duration: Weeks to months
Cost: $1M+ for large models
```

#### **Stage 2: Fine-tuning**
```
Objective: Task specialization and instruction following
Data: Task-specific datasets (GB scale)
Method: Supervised learning on curated data
Duration: Hours to days
Cost: $1K-$10K depending on model size
```

#### **Stage 3: Alignment**
```
Objective: Human preference alignment and safety
Data: Human preference comparisons (MB scale)
Method: RLHF, Constitutional AI, DPO
Duration: Days to weeks
Cost: $10K-$100K including human annotation
```

### Mathematical Foundations

#### Pre-training Loss
```
L_pretrain = -∑ log P(x_t | x_1, ..., x_{t-1}; θ)
```

#### Fine-tuning Loss
```
L_finetune = -∑ log P(y | x; θ) + λ * L_regularization
```

#### RLHF Objective
```
L_RLHF = E[r(x,y)] - β * KL(π_θ(y|x) || π_ref(y|x))
```

Where:
- `r(x,y)` is the reward model score
- `β` is the KL penalty coefficient
- `π_ref` is the reference model

## 🚀 Quick Start

### 1. Test Basic Model Implementation

```python
from exercise import TrainingConfig, SimpleLLM

# Create configuration
config = TrainingConfig(
    vocab_size=10000,
    hidden_size=512,
    num_layers=6,
    num_heads=8,
    max_seq_length=512
)

# Initialize model
model = SimpleLLM(config)

# Test forward pass
import torch
input_ids = torch.randint(0, config.vocab_size, (2, 10))
outputs = model(input_ids)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"Output shape: {outputs['logits'].shape}")
```

### 2. Test Parameter-Efficient Fine-tuning

```python
from exercise import LoRALayer, AdapterLayer

# Test LoRA
linear_layer = torch.nn.Linear(512, 512)
lora_layer = LoRALayer(linear_layer, rank=16, alpha=32)

# Count trainable parameters
total_params = sum(p.numel() for p in lora_layer.parameters())
trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)

print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")
print(f"Trainable ratio: {trainable_params/total_params:.2%}")

# Test Adapter
adapter = AdapterLayer(hidden_size=512, adapter_size=64)
test_input = torch.randn(2, 10, 512)
output = adapter(test_input)
print(f"Adapter output shape: {output.shape}")
```

### 3. Test Distributed Training Setup

```bash
# Single-node multi-GPU training
torchrun --nproc_per_node=2 exercise.py

# Multi-node training (on each node)
torchrun --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=12355 --nproc_per_node=4 exercise.py
```

## 🔍 Implementation Details

### 1. **Distributed Training Patterns**

#### Data Parallelism
```python
# Replicate model on each GPU, split data
model = torch.nn.parallel.DistributedDataParallel(model)
sampler = torch.utils.data.distributed.DistributedSampler(dataset)
```

#### Model Parallelism
```python
# Split model layers across GPUs
class ModelParallel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1000, 1000).to('cuda:0')
        self.layer2 = nn.Linear(1000, 1000).to('cuda:1')
    
    def forward(self, x):
        x = self.layer1(x.to('cuda:0'))
        x = self.layer2(x.to('cuda:1'))
        return x
```

#### Pipeline Parallelism
```python
# Process different micro-batches in pipeline
from torch.distributed.pipeline.sync import Pipe
model = Pipe(torch.nn.Sequential(*layers), balance=[2, 2, 2, 2])
```

### 2. **Memory Optimization Techniques**

#### Gradient Checkpointing
```python
# Trade computation for memory
from torch.utils.checkpoint import checkpoint

def forward_with_checkpointing(self, x):
    return checkpoint(self.expensive_layer, x)
```

#### Mixed Precision Training
```python
# Use FP16 for forward pass, FP32 for gradients
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, targets)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

#### ZeRO Optimizer
```python
# Partition optimizer states across devices
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer

optimizer = DeepSpeedZeroOptimizer(
    model.parameters(),
    lr=1e-4,
    partition_grads=True,
    contiguous_gradients=True
)
```

### 3. **Parameter-Efficient Fine-tuning**

#### LoRA Implementation
```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=16, alpha=32):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        
        # Original layer (frozen)
        self.linear = nn.Linear(in_features, out_features)
        self.linear.weight.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)
```

#### Adapter Layers
```python
class Adapter(nn.Module):
    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.up_proj(x)
        return residual + x
```

### 4. **RLHF Implementation**

#### Reward Model Training
```python
def train_reward_model(model, preference_data, epochs=3):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        for batch in preference_data:
            chosen_rewards = model(batch['chosen'])
            rejected_rewards = model(batch['rejected'])
            
            # Bradley-Terry loss
            loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

#### PPO Training
```python
def ppo_step(policy, ref_model, reward_model, batch, clip_eps=0.2):
    # Get current and old log probabilities
    current_logprobs = policy.get_log_probs(batch['responses'])
    old_logprobs = batch['old_logprobs']
    
    # Compute rewards and advantages
    rewards = reward_model(batch['responses'])
    kl_penalty = current_logprobs - ref_model.get_log_probs(batch['responses'])
    advantages = rewards - 0.1 * kl_penalty
    
    # PPO clipped objective
    ratio = torch.exp(current_logprobs - old_logprobs)
    clipped_ratio = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
    
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    
    return policy_loss
```

## 🧪 Testing Your Implementation

### Run Unit Tests

```bash
# Run all tests
python -m pytest test_llm_training.py -v

# Run specific test category
python -m pytest test_llm_training.py::TestDistributedTraining -v

# Run with coverage
python -m pytest test_llm_training.py --cov=exercise --cov-report=html
```

### Performance Testing

```bash
# Test distributed training
torchrun --nproc_per_node=2 exercise.py

# Profile memory usage
python -m memory_profiler exercise.py

# Benchmark training speed
python -c "
from exercise import *
import time

config = TrainingConfig(batch_size=8)
model = SimpleLLM(config)
data = torch.randint(0, config.vocab_size, (8, 512))

start_time = time.time()
for _ in range(100):
    outputs = model(data)
    loss = outputs['loss'] if 'loss' in outputs else torch.tensor(0.0)
    loss.backward()

print(f'Training speed: {100 / (time.time() - start_time):.2f} steps/sec')
"
```

## 🔧 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```python
   # Solutions:
   # - Reduce batch size
   # - Use gradient checkpointing
   # - Apply gradient accumulation
   
   # Gradient accumulation example
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       outputs = model(batch)
       loss = outputs.loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

2. **Distributed Training Hangs**
   ```bash
   # Check network connectivity
   export NCCL_DEBUG=INFO
   export NCCL_DEBUG_SUBSYS=ALL
   
   # Use different backend if needed
   torch.distributed.init_process_group(backend='gloo')  # CPU fallback
   ```

3. **Slow Training Speed**
   ```python
   # Optimizations:
   # - Use mixed precision
   # - Optimize data loading
   # - Use compiled models (PyTorch 2.0+)
   
   model = torch.compile(model)  # PyTorch 2.0+
   ```

4. **Gradient Explosion**
   ```python
   # Gradient clipping
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

### Performance Tips

1. **Data Loading Optimization**
   ```python
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       num_workers=4,           # Parallel data loading
       pin_memory=True,         # Faster GPU transfer
       persistent_workers=True  # Keep workers alive
   )
   ```

2. **Model Compilation**
   ```python
   # PyTorch 2.0+ compilation
   model = torch.compile(model, mode='max-autotune')
   ```

3. **Memory Management**
   ```python
   # Clear cache periodically
   if step % 100 == 0:
       torch.cuda.empty_cache()
   ```

## 📚 Additional Resources

### Research Papers
- "Language Models are Few-Shot Learners" (GPT-3)
- "Training language models to follow instructions with human feedback" (InstructGPT)
- "Constitutional AI: Harmlessness from AI Feedback" (Anthropic)
- "Direct Preference Optimization" (DPO)
- "LoRA: Low-Rank Adaptation of Large Language Models"

### Implementation References
- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [TRL (Transformer Reinforcement Learning)](https://github.com/huggingface/trl)
- [PEFT (Parameter-Efficient Fine-Tuning)](https://github.com/huggingface/peft)

### Datasets
- **Pre-training**: Common Crawl, C4, The Pile
- **Instruction Tuning**: Alpaca, Dolly, OpenAssistant
- **Preference Data**: Anthropic HH-RLHF, OpenAI WebGPT

## ✅ Success Criteria

After completing this setup and implementation, you should be able to:

- ✅ Implement distributed training for large language models
- ✅ Apply parameter-efficient fine-tuning methods (LoRA, Adapters)
- ✅ Build RLHF pipeline with reward model and PPO training
- ✅ Implement Constitutional AI for self-improvement
- ✅ Monitor and evaluate training progress comprehensively
- ✅ Optimize training for memory and computational efficiency
- ✅ Debug common training issues and bottlenecks

## 🎯 Next Steps

After mastering LLM training stages:

1. **Day 45**: Advanced prompt engineering with DSPy
2. **Day 46**: Prompt security and injection defense
3. **Day 47**: Project - Advanced prompting system
4. **Integration**: Apply training techniques to your own models

---

**Ready to master the complete LLM training pipeline?** 🚀

Start with the basic implementations and gradually work through distributed training, parameter-efficient methods, and alignment techniques. The comprehensive monitoring will help you understand the training dynamics at each step!