# Day 48: Fine-tuning Techniques - LoRA & QLoRA

## Learning Objectives
By the end of this session, you will be able to:
- Understand parameter-efficient fine-tuning (PEFT) techniques
- Implement LoRA (Low-Rank Adaptation) for efficient model adaptation
- Apply QLoRA (Quantized LoRA) for memory-efficient fine-tuning
- Design fine-tuning strategies for different use cases and constraints
- Optimize fine-tuning performance and resource utilization
- Deploy fine-tuned models in production environments

## Theory (30 minutes)

### What is Parameter-Efficient Fine-Tuning (PEFT)?

Parameter-Efficient Fine-Tuning refers to methods that adapt large pre-trained models to specific tasks while updating only a small subset of parameters. This approach addresses the computational and memory challenges of full fine-tuning while maintaining competitive performance.

**Key Benefits:**
- **Reduced Memory Requirements**: Only store and update a fraction of parameters
- **Faster Training**: Fewer parameters to optimize leads to faster convergence
- **Lower Storage Costs**: Multiple task-specific adapters can share the same base model
- **Reduced Catastrophic Forgetting**: Preserves pre-trained knowledge better
- **Easier Deployment**: Smaller adapter modules are easier to distribute and manage

### Traditional Fine-tuning Challenges

#### Full Fine-tuning Limitations
```python
# Traditional full fine-tuning approach
class FullFineTuning:
    def __init__(self, base_model):
        self.model = base_model
        # All parameters are trainable
        for param in self.model.parameters():
            param.requires_grad = True
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    # For a 7B parameter model: ~7 billion trainable parameters
    # Memory requirement: ~28GB (FP32) or ~14GB (FP16) just for parameters
    # Plus gradients, optimizer states, activations...
```

**Problems with Full Fine-tuning:**
- **Memory Explosion**: Requires storing gradients and optimizer states for all parameters
- **Catastrophic Forgetting**: May lose pre-trained capabilities
- **Overfitting**: Easy to overfit on small datasets
- **Storage Overhead**: Need separate model copies for each task
- **Computational Cost**: Expensive training and inference

### Low-Rank Adaptation (LoRA)

LoRA is based on the hypothesis that the weight updates during adaptation have a low "intrinsic rank." Instead of updating the full weight matrix, LoRA decomposes the update into two smaller matrices.

#### Mathematical Foundation

For a pre-trained weight matrix W ∈ R^(d×k), LoRA represents the update as:

```
W' = W + ΔW = W + BA
```

Where:
- B ∈ R^(d×r) and A ∈ R^(r×k) are trainable matrices
- r << min(d,k) is the rank (typically 1-64)
- W remains frozen during training

#### LoRA Implementation

```python
import torch
import torch.nn as nn
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, 
                 alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout)
        
        # Initialize A with random values, B with zeros
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        # LoRA forward pass: x @ (A^T @ B^T) * scaling
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features, 
            base_layer.out_features, 
            rank, 
            alpha
        )
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output
```

#### LoRA Configuration Strategies

```python
class LoRAConfig:
    def __init__(self):
        # Target modules to apply LoRA
        self.target_modules = [
            "q_proj",  # Query projection
            "v_proj",  # Value projection
            "k_proj",  # Key projection (optional)
            "o_proj",  # Output projection (optional)
            "gate_proj",  # MLP gate (for LLaMA-style models)
            "up_proj",    # MLP up projection
            "down_proj"   # MLP down projection
        ]
        
        # LoRA hyperparameters
        self.rank = 16  # Higher rank = more capacity, more parameters
        self.alpha = 32  # Scaling factor (typically 2x rank)
        self.dropout = 0.1
        
        # Task-specific configurations
        self.task_configs = {
            "instruction_following": {
                "rank": 64,
                "alpha": 128,
                "target_modules": ["q_proj", "v_proj", "o_proj"]
            },
            "code_generation": {
                "rank": 32,
                "alpha": 64,
                "target_modules": ["q_proj", "v_proj", "gate_proj", "up_proj"]
            },
            "summarization": {
                "rank": 16,
                "alpha": 32,
                "target_modules": ["q_proj", "v_proj"]
            }
        }
```

### Quantized LoRA (QLoRA)

QLoRA combines LoRA with quantization techniques to further reduce memory requirements while maintaining performance. It enables fine-tuning of very large models (65B+ parameters) on consumer hardware.

#### QLoRA Key Innovations

1. **4-bit NormalFloat (NF4)**: A new 4-bit data type optimized for normally distributed weights
2. **Double Quantization**: Quantizes the quantization constants themselves
3. **Paged Optimizers**: Uses NVIDIA unified memory to handle memory spikes

#### QLoRA Architecture

```python
import bitsandbytes as bnb
from transformers import BitsAndBytesConfig

class QLoRAConfig:
    def __init__(self):
        # Quantization configuration
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,  # Double quantization
        )
        
        # LoRA configuration for QLoRA
        self.lora_config = {
            "r": 64,  # Can use higher rank due to memory savings
            "lora_alpha": 128,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }

class QLoRAModel(nn.Module):
    def __init__(self, model_name: str, config: QLoRAConfig):
        super().__init__()
        
        # Load base model with quantization
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=config.quantization_config,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        
        # Apply LoRA adapters
        self.model = get_peft_model(self.base_model, config.lora_config)
        
    def get_memory_usage(self):
        """Calculate memory usage breakdown"""
        base_params = sum(p.numel() for p in self.base_model.parameters())
        lora_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "base_model_params": base_params,
            "trainable_params": lora_params,
            "trainable_percentage": (lora_params / base_params) * 100,
            "estimated_memory_gb": (base_params * 0.5 + lora_params * 2) / 1e9  # Rough estimate
        }
```

### Advanced Fine-tuning Strategies

#### 1. Multi-Task LoRA
Train multiple LoRA adapters for different tasks:

```python
class MultiTaskLoRA(nn.Module):
    def __init__(self, base_model, tasks: List[str], rank: int = 16):
        super().__init__()
        self.base_model = base_model
        self.tasks = tasks
        
        # Create separate LoRA adapters for each task
        self.task_adapters = nn.ModuleDict({
            task: self._create_lora_adapter(rank)
            for task in tasks
        })
        
        # Task classifier to route inputs
        self.task_classifier = nn.Linear(
            base_model.config.hidden_size, 
            len(tasks)
        )
    
    def _create_lora_adapter(self, rank):
        # Create LoRA adapter for specific layers
        adapter = {}
        for name, module in self.base_model.named_modules():
            if isinstance(module, nn.Linear) and any(
                target in name for target in ["q_proj", "v_proj"]
            ):
                adapter[name] = LoRALinear(module, rank)
        return nn.ModuleDict(adapter)
    
    def forward(self, x, task_id=None):
        if task_id is None:
            # Automatic task detection
            task_logits = self.task_classifier(x.mean(dim=1))
            task_id = torch.argmax(task_logits, dim=-1)
        
        # Apply task-specific LoRA adapter
        task_name = self.tasks[task_id]
        adapter = self.task_adapters[task_name]
        
        # Forward pass with task-specific adaptation
        return self._forward_with_adapter(x, adapter)
```

#### 2. Hierarchical LoRA
Use different ranks for different layers:

```python
class HierarchicalLoRA:
    def __init__(self, model, layer_configs):
        self.layer_configs = {
            "early_layers": {"rank": 8, "alpha": 16},    # Lower capacity
            "middle_layers": {"rank": 32, "alpha": 64},  # Higher capacity
            "late_layers": {"rank": 16, "alpha": 32}     # Medium capacity
        }
    
    def apply_hierarchical_lora(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                layer_depth = self._get_layer_depth(name)
                config = self._get_config_for_depth(layer_depth)
                
                # Apply LoRA with depth-specific configuration
                lora_module = LoRALinear(
                    module, 
                    rank=config["rank"], 
                    alpha=config["alpha"]
                )
                self._replace_module(model, name, lora_module)
```

#### 3. Adaptive LoRA
Dynamically adjust LoRA parameters during training:

```python
class AdaptiveLoRA(nn.Module):
    def __init__(self, base_layer, initial_rank=16, max_rank=64):
        super().__init__()
        self.base_layer = base_layer
        self.current_rank = initial_rank
        self.max_rank = max_rank
        
        # Initialize with maximum rank, use masking for current rank
        self.lora_A = nn.Parameter(torch.zeros(max_rank, base_layer.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base_layer.out_features, max_rank))
        self.rank_mask = nn.Parameter(torch.ones(max_rank), requires_grad=False)
        
    def adapt_rank(self, new_rank):
        """Dynamically change the effective rank"""
        self.current_rank = min(new_rank, self.max_rank)
        self.rank_mask.fill_(0)
        self.rank_mask[:self.current_rank] = 1
    
    def forward(self, x):
        # Apply rank masking
        masked_A = self.lora_A * self.rank_mask.unsqueeze(1)
        masked_B = self.lora_B * self.rank_mask.unsqueeze(0)
        
        base_output = self.base_layer(x)
        lora_output = x @ masked_A.T @ masked_B.T
        return base_output + lora_output
```

### Production Considerations

#### 1. Memory Optimization Techniques

```python
class MemoryOptimizedTraining:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
    def setup_gradient_checkpointing(self):
        """Enable gradient checkpointing to save memory"""
        self.model.gradient_checkpointing_enable()
    
    def setup_mixed_precision(self):
        """Configure automatic mixed precision"""
        from torch.cuda.amp import GradScaler, autocast
        
        self.scaler = GradScaler()
        self.use_amp = True
    
    def setup_deepspeed_zero(self):
        """Configure DeepSpeed ZeRO for distributed training"""
        deepspeed_config = {
            "zero_optimization": {
                "stage": 2,  # Partition optimizer states
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True
                }
            },
            "fp16": {"enabled": True},
            "gradient_clipping": 1.0
        }
        return deepspeed_config
    
    def calculate_optimal_batch_size(self, max_memory_gb=24):
        """Calculate optimal batch size based on available memory"""
        # Rough estimation based on model size and memory
        model_memory = self._estimate_model_memory()
        available_memory = max_memory_gb * 0.8  # Leave 20% buffer
        
        optimal_batch_size = int(
            (available_memory - model_memory) / self._estimate_per_sample_memory()
        )
        
        return max(1, optimal_batch_size)
```

#### 2. Training Optimization

```python
class LoRATrainer:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        
    def setup_optimizer(self):
        """Setup optimizer for LoRA parameters only"""
        # Only optimize LoRA parameters
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        return optimizer
    
    def setup_scheduler(self, optimizer, num_training_steps):
        """Setup learning rate scheduler"""
        from transformers import get_cosine_schedule_with_warmup
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * num_training_steps),
            num_training_steps=num_training_steps
        )
        
        return scheduler
    
    def train_step(self, batch, optimizer, scheduler):
        """Single training step with mixed precision"""
        self.model.train()
        
        with autocast():
            outputs = self.model(**batch)
            loss = outputs.loss
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        # Optimizer step
        self.scaler.step(optimizer)
        self.scaler.update()
        scheduler.step()
        optimizer.zero_grad()
        
        return loss.item()
```

#### 3. Evaluation and Monitoring

```python
class LoRAEvaluator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
    def evaluate_model_quality(self, eval_dataset):
        """Comprehensive model evaluation"""
        metrics = {}
        
        # Perplexity evaluation
        metrics['perplexity'] = self._calculate_perplexity(eval_dataset)
        
        # Task-specific metrics
        metrics['bleu_score'] = self._calculate_bleu(eval_dataset)
        metrics['rouge_scores'] = self._calculate_rouge(eval_dataset)
        
        # Efficiency metrics
        metrics['inference_speed'] = self._measure_inference_speed()
        metrics['memory_usage'] = self._measure_memory_usage()
        
        return metrics
    
    def compare_with_baseline(self, baseline_model, test_data):
        """Compare LoRA model with baseline"""
        lora_metrics = self.evaluate_model_quality(test_data)
        baseline_metrics = self._evaluate_baseline(baseline_model, test_data)
        
        comparison = {
            'performance_retention': lora_metrics['perplexity'] / baseline_metrics['perplexity'],
            'speed_improvement': baseline_metrics['inference_speed'] / lora_metrics['inference_speed'],
            'memory_reduction': 1 - (lora_metrics['memory_usage'] / baseline_metrics['memory_usage']),
            'parameter_reduction': self._calculate_parameter_reduction()
        }
        
        return comparison
    
    def analyze_adapter_importance(self):
        """Analyze which LoRA adapters contribute most to performance"""
        adapter_scores = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora'):
                # Calculate adapter magnitude
                adapter_norm = torch.norm(module.lora.lora_B @ module.lora.lora_A)
                adapter_scores[name] = adapter_norm.item()
        
        # Sort by importance
        sorted_adapters = sorted(
            adapter_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return sorted_adapters
```

### Best Practices for Production

#### 1. Hyperparameter Selection
- **Rank Selection**: Start with r=16, increase if underfitting
- **Alpha Scaling**: Typically set to 2×rank for balanced adaptation
- **Learning Rate**: 1e-4 to 5e-4, higher than full fine-tuning
- **Target Modules**: Focus on attention layers for most tasks

#### 2. Data Preparation
- **Quality over Quantity**: High-quality, task-specific data is crucial
- **Data Formatting**: Consistent prompt templates and formatting
- **Validation Split**: Hold out data for proper evaluation
- **Data Augmentation**: Paraphrasing and back-translation for robustness

#### 3. Training Strategies
- **Gradual Unfreezing**: Start with smaller rank, gradually increase
- **Curriculum Learning**: Start with easier examples, progress to harder ones
- **Early Stopping**: Monitor validation loss to prevent overfitting
- **Checkpoint Management**: Save adapters separately for easy deployment

#### 4. Deployment Considerations
- **Adapter Switching**: Enable dynamic task switching in production
- **Model Serving**: Use efficient serving frameworks like vLLM or TensorRT-LLM
- **Monitoring**: Track performance degradation and adaptation quality
- **A/B Testing**: Compare adapter performance against baselines

### Why LoRA and QLoRA Matter

These techniques are revolutionary because they:

- **Democratize Fine-tuning**: Enable fine-tuning of large models on consumer hardware
- **Reduce Costs**: Significantly lower computational and storage requirements
- **Improve Efficiency**: Faster training and deployment cycles
- **Enable Personalization**: Multiple task-specific adapters from single base model
- **Maintain Quality**: Competitive performance with full fine-tuning
- **Support Innovation**: Enable rapid experimentation and iteration

## Exercise (25 minutes)
Complete the hands-on exercises in `exercise.py` to practice LoRA and QLoRA implementation.

## Resources
- [LoRA Paper: "LoRA: Low-Rank Adaptation of Large Language Models"](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper: "QLoRA: Efficient Finetuning of Quantized LLMs"](https://arxiv.org/abs/2305.14314)
- [Hugging Face PEFT Library](https://github.com/huggingface/peft)
- [bitsandbytes Documentation](https://github.com/TimDettmers/bitsandbytes)
- [Parameter-Efficient Fine-Tuning Guide](https://huggingface.co/docs/peft/index)

## Next Steps
- Complete the exercises to practice LoRA and QLoRA implementation
- Take the quiz to test your understanding
- Experiment with different rank configurations and target modules
- Move to Day 49: RLHF and DPO - Human Feedback & Preference Learning
