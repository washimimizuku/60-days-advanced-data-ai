"""
Day 48: Fine-tuning Techniques - LoRA & QLoRA - Exercises

Complete the following exercises to practice parameter-efficient fine-tuning:
1. Basic LoRA implementation from scratch
2. LoRA configuration and target module selection
3. QLoRA setup with quantization
4. Multi-task LoRA adapter management
5. Memory optimization and performance analysis
6. Production deployment strategies
7. Evaluation and comparison frameworks

Run each exercise and observe the efficiency gains and performance characteristics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import json

# Mock transformers components for exercises (replace with actual transformers in production)
class MockTransformerConfig:
    def __init__(self):
        self.hidden_size = 768
        self.num_attention_heads = 12
        self.intermediate_size = 3072
        self.vocab_size = 50257

class MockLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None
    
    def forward(self, x):
        output = F.linear(x, self.weight, self.bias)
        return output

# Exercise 1: Basic LoRA Implementation
class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 4, 
                 alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Create LoRA matrices A and B
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize matrices properly
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
    def forward(self, x):
        # Implement LoRA forward pass
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0):
        super().__init__()
        self.base_layer = base_layer
        
        # Create LoRA layer
        self.lora = LoRALayer(
            base_layer.in_features, 
            base_layer.out_features, 
            rank, 
            alpha
        )
        
        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        # Combine base layer output with LoRA adaptation
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output

def exercise_1_basic_lora():
    """Exercise 1: Implement basic LoRA layer from scratch"""
    print("=== Exercise 1: Basic LoRA Implementation ===")
    
    # Create a mock linear layer
    base_layer = MockLinear(768, 768)
    
    # Create LoRA adaptation
    lora_layer = LoRALinear(base_layer, rank=16, alpha=32)
    
    # Test input
    x = torch.randn(32, 128, 768)  # (batch, seq_len, hidden_size)
    
    # Test forward pass
    output = lora_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Count parameters
    base_params = sum(p.numel() for p in base_layer.parameters())
    lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    print(f"Base parameters: {base_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Parameter reduction: {(1 - lora_params/base_params)*100:.1f}%")
    
    print("✅ LoRA implementation completed successfully!")
    print()

# Exercise 2: LoRA Configuration Management
@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"  # "none", "all", "lora_only"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

class LoRAConfigManager:
    def __init__(self):
        # Define task-specific configurations
        self.task_configs = {
            "instruction_following": LoRAConfig(
                rank=64, alpha=128, dropout=0.05,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj"]
            ),
            "code_generation": LoRAConfig(
                rank=32, alpha=64, dropout=0.1,
                target_modules=["q_proj", "v_proj", "gate_proj"]
            ),
            "summarization": LoRAConfig(
                rank=16, alpha=32, dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            )
        }
        
    def get_config_for_task(self, task: str) -> LoRAConfig:
        """Get optimized LoRA configuration for specific task"""
        if task in self.task_configs:
            return self.task_configs[task]
        return LoRAConfig(rank=16, alpha=32, target_modules=["q_proj", "v_proj"])
    
    def get_config_for_model_size(self, num_parameters: int) -> LoRAConfig:
        """Get configuration based on model size"""
        if num_parameters <= 7e9:
            return LoRAConfig(rank=8, alpha=16)
        elif num_parameters <= 30e9:
            return LoRAConfig(rank=16, alpha=32)
        else:
            return LoRAConfig(rank=32, alpha=64)
    
    def optimize_config_for_hardware(self, available_memory_gb: float) -> LoRAConfig:
        """Optimize configuration for available hardware"""
        if available_memory_gb < 16:
            return LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
        elif available_memory_gb < 32:
            return LoRAConfig(rank=16, alpha=32, target_modules=["q_proj", "v_proj"])
        else:
            return LoRAConfig(rank=32, alpha=64, target_modules=["q_proj", "v_proj", "o_proj"])

def exercise_2_lora_configuration():
    """Exercise 2: LoRA configuration and target module selection"""
    print("=== Exercise 2: LoRA Configuration Management ===")
    
    config_manager = LoRAConfigManager()
    
    # Test different configurations
    tasks = ["instruction_following", "code_generation", "summarization"]
    model_sizes = [7e9, 13e9, 70e9]  # 7B, 13B, 70B parameters
    
    print("Task-specific configurations:")
    for task in tasks:
        config = config_manager.get_config_for_task(task)
        print(f"{task}: rank={config.rank}, alpha={config.alpha}, targets={config.target_modules}")
    
    print("\nModel size-specific configurations:")
    for size in model_sizes:
        config = config_manager.get_config_for_model_size(size)
        print(f"{size/1e9:.0f}B params: rank={config.rank}, alpha={config.alpha}")
    
    print("\nHardware-optimized configurations:")
    for memory_gb in [8, 16, 32, 64]:
        config = config_manager.optimize_config_for_hardware(memory_gb)
        print(f"{memory_gb}GB: rank={config.rank}, alpha={config.alpha}, modules={len(config.target_modules)}")
    
    print("✅ Configuration management completed successfully!")
    print()

# Exercise 3: QLoRA Setup and Quantization
class MockQuantizationConfig:
    def __init__(self):
        self.load_in_4bit = True
        self.bnb_4bit_quant_type = "nf4"
        self.bnb_4bit_compute_dtype = torch.bfloat16
        self.bnb_4bit_use_double_quant = True

class QLoRASetup:
    def __init__(self):
        self.quantization_config = MockQuantizationConfig()
        
    def setup_quantized_model(self, model_name: str, lora_config: LoRAConfig):
        """Setup model with quantization and LoRA"""
        print(f"Setting up QLoRA for {model_name}")
        
        setup_info = {
            "model_name": model_name,
            "quantization": "4-bit NF4 with double quantization",
            "lora_rank": lora_config.rank,
            "lora_alpha": lora_config.alpha,
            "target_modules": lora_config.target_modules,
            "compute_dtype": "bfloat16"
        }
        
        return setup_info
    
    def calculate_memory_savings(self, base_model_params: int, rank: int, 
                                num_target_modules: int) -> Dict[str, float]:
        """Calculate memory savings from QLoRA"""
        # Full fine-tuning memory (FP16)
        full_ft_params = base_model_params
        full_ft_gradients = base_model_params * 2
        full_ft_optimizer = base_model_params * 8
        full_ft_total = (full_ft_params * 2 + full_ft_gradients + full_ft_optimizer) / 1e9
        
        # QLoRA memory (4-bit base + FP16 adapters)
        qlora_base = base_model_params * 0.5 / 1e9
        lora_params_per_module = rank * 768 * 2
        total_lora_params = num_target_modules * lora_params_per_module
        qlora_lora = total_lora_params * 2 / 1e9
        qlora_gradients = total_lora_params * 2 / 1e9
        qlora_optimizer = total_lora_params * 8 / 1e9
        qlora_total = qlora_base + qlora_lora + qlora_gradients + qlora_optimizer
        
        memory_analysis = {
            "full_finetuning_gb": full_ft_total,
            "qlora_total_gb": qlora_total,
            "memory_savings_gb": full_ft_total - qlora_total,
            "memory_reduction_percent": (1 - qlora_total / full_ft_total) * 100,
            "breakdown": {
                "base_model_gb": qlora_base,
                "lora_params_gb": qlora_lora,
                "total_lora_params": total_lora_params
            }
        }
        
        return memory_analysis

def exercise_3_qlora_setup():
    """Exercise 3: QLoRA setup with quantization"""
    print("=== Exercise 3: QLoRA Setup and Quantization ===")
    
    qlora_setup = QLoRASetup()
    
    # Model configurations to test
    model_configs = [
        {"name": "7B-model", "params": 7e9, "target_modules": 4},
        {"name": "13B-model", "params": 13e9, "target_modules": 6},
        {"name": "70B-model", "params": 70e9, "target_modules": 8},
    ]
    
    print("Memory analysis for different model sizes:")
    print(f"{'Model':<12} {'Full FT (GB)':<12} {'QLoRA (GB)':<11} {'Savings (%)':<12}")
    print("-" * 50)
    
    for config in model_configs:
        memory_analysis = qlora_setup.calculate_memory_savings(
            config["params"], rank=64, num_target_modules=config["target_modules"]
        )
        print(f"{config['name']:<12} "
              f"{memory_analysis['full_finetuning_gb']:<12.1f} "
              f"{memory_analysis['qlora_total_gb']:<11.1f} "
              f"{memory_analysis['memory_reduction_percent']:<12.1f}")
    
    # Detailed breakdown for one model
    print(f"\nDetailed breakdown for 13B model:")
    detailed = qlora_setup.calculate_memory_savings(13e9, rank=32, num_target_modules=6)
    breakdown = detailed["breakdown"]
    print(f"Base model (4-bit): {breakdown['base_model_gb']:.1f} GB")
    print(f"LoRA parameters: {breakdown['lora_params_gb']:.1f} GB")
    print(f"Total LoRA params: {breakdown['total_lora_params']:,}")
    
    print("✅ QLoRA setup completed successfully!")
    print()

# Exercise 4: Multi-Task LoRA Management
class MultiTaskLoRAManager:
    def __init__(self, base_model, tasks: List[str]):
        self.base_model = base_model
        self.tasks = tasks
        self.adapters = {}
        self.current_task = None
        self.adapter_metadata = {}
        
    def create_task_adapter(self, task: str, config: LoRAConfig):
        """Create LoRA adapter for specific task"""
        print(f"Creating adapter for task: {task}")
        
        adapter_info = {
            "task": task,
            "rank": config.rank,
            "alpha": config.alpha,
            "target_modules": config.target_modules,
            "parameters": self._calculate_adapter_params(config),
            "created_at": time.time()
        }
        
        self.adapters[task] = adapter_info
        return adapter_info
    
    def _calculate_adapter_params(self, config: LoRAConfig) -> int:
        """Calculate number of parameters in adapter"""
        params_per_module = config.rank * (768 + 768)
        total_params = len(config.target_modules) * params_per_module
        return total_params
    
    def switch_task(self, task: str):
        """Switch to different task adapter"""
        if task not in self.adapters:
            raise ValueError(f"Task '{task}' not found")
        
        previous_task = self.current_task
        self.current_task = task
        
        return {
            "previous_task": previous_task,
            "new_task": task,
            "adapter_info": self.adapters[task]
        }
    
    def merge_adapters(self, tasks: List[str], weights: List[float] = None):
        """Merge multiple task adapters"""
        if weights is None:
            weights = [1.0 / len(tasks)] * len(tasks)
        
        merged_rank = max(self.adapters[task]["rank"] for task in tasks)
        merged_params = sum(
            self.adapters[task]["parameters"] * weight 
            for task, weight in zip(tasks, weights)
        )
        
        merged_adapter = {
            "type": "merged",
            "source_tasks": tasks,
            "weights": weights,
            "rank": merged_rank,
            "parameters": int(merged_params)
        }
        
        merged_name = f"merged_{'_'.join(tasks)}"
        self.adapters[merged_name] = merged_adapter
        
        return merged_adapter
    
    def get_adapter_statistics(self) -> Dict[str, Any]:
        """Get statistics about all adapters"""
        if not self.adapters:
            return {"message": "No adapters created yet"}
        
        total_params = sum(adapter["parameters"] for adapter in self.adapters.values())
        avg_rank = sum(adapter["rank"] for adapter in self.adapters.values()) / len(self.adapters)
        
        stats = {
            "total_adapters": len(self.adapters),
            "total_parameters": total_params,
            "average_rank": avg_rank,
            "current_task": self.current_task,
            "adapter_list": list(self.adapters.keys())
        }
        
        return stats

def exercise_4_multitask_lora():
    """Exercise 4: Multi-task LoRA adapter management"""
    print("=== Exercise 4: Multi-Task LoRA Management ===")
    
    # Mock base model
    base_model = MockLinear(768, 768)
    
    tasks = ["qa", "summarization", "code_generation", "translation"]
    manager = MultiTaskLoRAManager(base_model, tasks)
    
    # Create adapters for each task
    print("Creating task-specific adapters:")
    for task in tasks:
        config = LoRAConfig(rank=16 if task != "code_generation" else 32)
        adapter_info = manager.create_task_adapter(task, config)
        print(f"  {task}: {adapter_info['parameters']:,} parameters, rank={adapter_info['rank']}")
    
    # Test task switching
    print(f"\nTesting task switching:")
    switch_result = manager.switch_task("qa")
    print(f"Switched to task: {manager.current_task}")
    
    switch_result = manager.switch_task("code_generation")
    print(f"Switched to task: {switch_result['new_task']}")
    
    # Test adapter merging
    print(f"\nTesting adapter merging:")
    merged_adapter = manager.merge_adapters(["qa", "summarization"], [0.7, 0.3])
    print(f"Merged adapter: {merged_adapter['source_tasks']} with weights {merged_adapter['weights']}")
    print(f"Merged parameters: {merged_adapter['parameters']:,}")
    
    # Get statistics
    print(f"\nAdapter statistics:")
    stats = manager.get_adapter_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("✅ Multi-task LoRA management completed successfully!")
    print()

# Exercise 5: Memory Optimization and Performance Analysis
class MemoryOptimizer:
    def __init__(self):
        self.optimization_strategies = {
            "gradient_checkpointing": {"memory_reduction": 0.3, "speed_penalty": 0.2},
            "mixed_precision": {"memory_reduction": 0.5, "speed_improvement": 0.1},
            "gradient_accumulation": {"memory_reduction": 0.6, "speed_penalty": 0.1},
            "optimizer_offloading": {"memory_reduction": 0.4, "speed_penalty": 0.3}
        }
    
    def analyze_memory_usage(self, model_params: int, batch_size: int, sequence_length: int) -> Dict[str, float]:
        """Analyze memory usage breakdown"""
        # Base model memory (4-bit quantization)
        base_model_memory = model_params * 0.5 / 1e9
        
        # LoRA parameters (assuming 4 adapters, rank 32)
        lora_params = 4 * 32 * 768 * 2
        lora_memory = lora_params * 2 / 1e9
        
        # Activation memory
        activation_memory = batch_size * sequence_length * 768 * 4 / 1e9
        
        # Gradient memory
        gradient_memory = lora_params * 2 / 1e9
        
        # Optimizer memory
        optimizer_memory = lora_params * 8 / 1e9
        
        total_memory = (base_model_memory + lora_memory + activation_memory + 
                       gradient_memory + optimizer_memory)
        
        memory_breakdown = {
            "base_model_gb": base_model_memory,
            "lora_parameters_gb": lora_memory,
            "activations_gb": activation_memory,
            "gradients_gb": gradient_memory,
            "optimizer_states_gb": optimizer_memory,
            "total_gb": total_memory
        }
        
        return memory_breakdown
    
    def optimize_batch_size(self, model_params: int, max_memory_gb: float) -> Dict[str, int]:
        """Find optimal batch size for given memory constraint"""
        low, high = 1, 128
        optimal_batch_size = 1
        
        while low <= high:
            mid = (low + high) // 2
            memory_usage = self.analyze_memory_usage(model_params, mid, 512)
            
            if memory_usage["total_gb"] <= max_memory_gb * 0.9:
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        target_effective_batch = 32
        accumulation_steps = max(1, target_effective_batch // optimal_batch_size)
        
        return {
            "optimal_batch_size": optimal_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": optimal_batch_size * accumulation_steps
        }
    
    def setup_gradient_accumulation(self, target_batch_size: int, max_batch_size: int) -> int:
        """Calculate gradient accumulation steps"""
        return max(1, target_batch_size // max_batch_size)

class PerformanceProfiler:
    def __init__(self):
        self.metrics_history = []
    
    def profile_training_step(self, model_params: int, batch_size: int, sequence_length: int):
        """Profile single training step"""
        start_time = time.time()
        
        # Simulate timing based on model characteristics
        base_forward_time = model_params / 1e9 * 0.1
        lora_overhead = 32 * 0.001  # LoRA computation overhead
        
        # Mock forward pass
        forward_time = base_forward_time + lora_overhead
        time.sleep(min(0.01, forward_time))
        
        # Mock backward pass
        backward_time = forward_time * 2
        time.sleep(min(0.01, backward_time))
        
        # Mock optimizer step
        optimizer_time = 32 * 0.0005
        time.sleep(min(0.005, optimizer_time))
        
        total_time = time.time() - start_time
        
        metrics = {
            "total_time": total_time,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
            "tokens_per_second": (batch_size * sequence_length) / total_time,
            "samples_per_second": batch_size / total_time
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def compare_lora_vs_full_finetuning(self, model_params: int, rank: int):
        """Compare LoRA vs full fine-tuning performance"""
        # LoRA metrics
        lora_metrics = self.profile_training_step(model_params, 16, 512)
        
        # Full fine-tuning simulation
        full_ft_forward_time = model_params / 1e9 * 0.15
        full_ft_backward_time = full_ft_forward_time * 3
        full_ft_optimizer_time = model_params / 1e9 * 0.05
        full_ft_total = full_ft_forward_time + full_ft_backward_time + full_ft_optimizer_time
        
        comparison = {
            "lora": {
                "total_time": lora_metrics["total_time"],
                "memory_gb": model_params * 0.5 / 1e9 + rank * 768 * 2 * 6 / 1e9,
                "trainable_params": rank * 768 * 2 * 4
            },
            "full_finetuning": {
                "total_time": full_ft_total,
                "memory_gb": model_params * 10 / 1e9,
                "trainable_params": model_params
            },
            "improvements": {
                "speed_improvement": full_ft_total / lora_metrics["total_time"],
                "memory_reduction": 1 - (lora_metrics["tokens_per_second"] / ((16 * 512) / full_ft_total)),
                "parameter_reduction": 1 - (rank * 768 * 2 * 4) / model_params
            }
        }
        
        return comparison

def exercise_5_memory_optimization():
    """Exercise 5: Memory optimization and performance analysis"""
    print("=== Exercise 5: Memory Optimization and Performance Analysis ===")
    
    optimizer = MemoryOptimizer()
    profiler = PerformanceProfiler()
    
    # Test memory analysis
    model_params = 7e9
    batch_size = 16
    sequence_length = 512
    
    print("Memory usage analysis:")
    memory_usage = optimizer.analyze_memory_usage(model_params, batch_size, sequence_length)
    for component, usage in memory_usage.items():
        print(f"  {component}: {usage:.2f} GB")
    
    print(f"\nBatch size optimization:")
    for memory_gb in [16, 24, 48]:
        optimization = optimizer.optimize_batch_size(model_params, memory_gb)
        print(f"  {memory_gb}GB: batch_size={optimization['optimal_batch_size']}, "
              f"accumulation={optimization['gradient_accumulation_steps']}, "
              f"effective={optimization['effective_batch_size']}")
    
    print(f"\nPerformance profiling:")
    performance = profiler.profile_training_step(model_params, batch_size, sequence_length)
    print(f"  Total time: {performance['total_time']:.4f}s")
    print(f"  Tokens/sec: {performance['tokens_per_second']:.1f}")
    print(f"  Samples/sec: {performance['samples_per_second']:.1f}")
    
    print(f"\nLoRA vs Full Fine-tuning comparison:")
    comparison = profiler.compare_lora_vs_full_finetuning(model_params, 32)
    improvements = comparison['improvements']
    print(f"  Speed improvement: {improvements['speed_improvement']:.1f}x")
    print(f"  Parameter reduction: {improvements['parameter_reduction']*100:.1f}%")
    
    print("✅ Memory optimization and performance analysis completed successfully!")
    print()

# Exercise 6: Production Deployment Strategies
class LoRADeploymentManager:
    def __init__(self):
        self.deployment_configs = {}
        self.adapter_registry = {}
        
    def prepare_adapter_for_deployment(self, adapter_path: str, base_model_path: str):
        """Prepare LoRA adapter for production deployment"""
        print(f"Preparing adapter {adapter_path} for deployment")
        
        deployment_info = {
            "adapter_path": adapter_path,
            "base_model_path": base_model_path,
            "validation_status": "passed",
            "optimization_applied": ["weight_merging", "quantization"],
            "deployment_ready": True,
            "estimated_latency_ms": 45.2,
            "memory_footprint_mb": 128.5
        }
        
        return deployment_info
    
    def setup_dynamic_adapter_loading(self, adapters: Dict[str, str]):
        """Setup system for dynamic adapter switching"""
        print(f"Setting up dynamic loading for {len(adapters)} adapters")
        
        loading_config = {
            "adapters": adapters,
            "cache_size": min(3, len(adapters)),
            "preload_popular": True,
            "switch_latency_ms": 12.3,
            "memory_overhead_mb": 64.2
        }
        
        self.deployment_configs["dynamic_loading"] = loading_config
        return loading_config
    
    def create_adapter_registry(self, adapters: List[Dict[str, Any]]):
        """Create registry for managing multiple adapters"""
        registry = {
            "total_adapters": len(adapters),
            "adapters_by_task": {},
            "adapters_by_version": {},
            "metadata": {}
        }
        
        for adapter in adapters:
            name = adapter["name"]
            task = adapter.get("task", "general")
            version = adapter.get("version", "1.0")
            
            if task not in registry["adapters_by_task"]:
                registry["adapters_by_task"][task] = []
            registry["adapters_by_task"][task].append(name)
            
            registry["adapters_by_version"][name] = version
            registry["metadata"][name] = {
                "path": adapter["path"],
                "size_mb": 45.2,
                "performance_score": 0.92
            }
        
        self.adapter_registry = registry
        return registry
    
    def benchmark_adapter_performance(self, adapter_path: str, test_data: List[str]) -> Dict[str, float]:
        """Benchmark adapter performance metrics"""
        print(f"Benchmarking adapter: {adapter_path}")
        
        # Simulate performance metrics
        metrics = {
            "inference_latency_ms": 42.5 + len(test_data) * 0.1,
            "throughput_samples_per_sec": 23.4,
            "memory_usage_mb": 156.7,
            "accuracy_score": 0.89,
            "bleu_score": 0.76,
            "rouge_l_score": 0.82,
            "perplexity": 12.3
        }
        
        return metrics

def exercise_6_deployment_strategies():
    """Exercise 6: Production deployment strategies"""
    print("=== Exercise 6: Production Deployment Strategies ===")
    
    deployment_manager = LoRADeploymentManager()
    
    # Mock adapter configurations
    adapters = [
        {"name": "customer_support", "path": "/models/adapters/support.bin", "version": "1.0", "task": "support"},
        {"name": "code_assistant", "path": "/models/adapters/code.bin", "version": "1.2", "task": "coding"},
        {"name": "content_writer", "path": "/models/adapters/content.bin", "version": "2.0", "task": "writing"},
    ]
    
    print("Creating adapter registry:")
    registry = deployment_manager.create_adapter_registry(adapters)
    print(f"  Total adapters: {registry['total_adapters']}")
    print(f"  Tasks: {list(registry['adapters_by_task'].keys())}")
    
    print(f"\nSetting up dynamic loading:")
    adapter_paths = {adapter["name"]: adapter["path"] for adapter in adapters}
    loading_config = deployment_manager.setup_dynamic_adapter_loading(adapter_paths)
    print(f"  Cache size: {loading_config['cache_size']}")
    print(f"  Switch latency: {loading_config['switch_latency_ms']:.1f}ms")
    
    print(f"\nBenchmarking adapter performance:")
    test_data = ["Hello, how can I help you?", "Write a Python function", "Create a blog post"]
    for adapter in adapters:
        metrics = deployment_manager.benchmark_adapter_performance(adapter["path"], test_data)
        print(f"  {adapter['name']}:")
        print(f"    Latency: {metrics['inference_latency_ms']:.1f}ms")
        print(f"    Accuracy: {metrics['accuracy_score']:.2f}")
        print(f"    Memory: {metrics['memory_usage_mb']:.1f}MB")
    
    print("✅ Deployment strategies completed successfully!")
    print()

# Exercise 7: Evaluation and Comparison Framework
class LoRAEvaluationFramework:
    def __init__(self):
        self.evaluation_metrics = [
            "perplexity", "bleu_score", "rouge_scores", 
            "inference_speed", "memory_usage", "parameter_count"
        ]
        self.evaluation_history = []
    
    def evaluate_adapter_quality(self, model_name: str, adapter_name: str, test_dataset_size: int) -> Dict[str, float]:
        """Comprehensive adapter quality evaluation"""
        print(f"Evaluating adapter '{adapter_name}' on {test_dataset_size} samples")
        
        # Simulate evaluation metrics
        metrics = {
            "perplexity": 15.2 + hash(adapter_name) % 10,
            "bleu_score": 0.75 + (hash(adapter_name) % 100) / 1000,
            "rouge_1": 0.68 + (hash(adapter_name) % 80) / 1000,
            "rouge_2": 0.45 + (hash(adapter_name) % 60) / 1000,
            "rouge_l": 0.62 + (hash(adapter_name) % 70) / 1000,
            "inference_speed_ms": 42.5 + (hash(adapter_name) % 20),
            "memory_usage_mb": 156.7 + (hash(adapter_name) % 50),
            "parameter_count": 32 * 768 * 2 * 4  # Assuming rank 32, 4 modules
        }
        
        self.evaluation_history.append({
            "adapter": adapter_name,
            "metrics": metrics,
            "timestamp": time.time()
        })
        
        return metrics
    
    def compare_adapters(self, base_model: str, adapters: List[str], test_data_size: int) -> Dict[str, Dict[str, float]]:
        """Compare multiple adapters on same test data"""
        print(f"Comparing {len(adapters)} adapters on {test_data_size} samples")
        
        comparison = {}
        for adapter in adapters:
            comparison[adapter] = self.evaluate_adapter_quality(base_model, adapter, test_data_size)
        
        # Add relative performance metrics
        best_bleu = max(comp["bleu_score"] for comp in comparison.values())
        best_speed = min(comp["inference_speed_ms"] for comp in comparison.values())
        
        for adapter, metrics in comparison.items():
            metrics["relative_bleu"] = metrics["bleu_score"] / best_bleu
            metrics["relative_speed"] = best_speed / metrics["inference_speed_ms"]
        
        return comparison
    
    def analyze_rank_sensitivity(self, model_name: str, ranks: List[int], test_data_size: int) -> Dict[int, Dict[str, float]]:
        """Analyze how rank affects performance"""
        print(f"Analyzing rank sensitivity for ranks: {ranks}")
        
        rank_analysis = {}
        
        for rank in ranks:
            # Simulate rank-dependent performance
            base_performance = 0.75
            rank_factor = min(1.0, rank / 32)  # Performance improves with rank up to 32
            diminishing_returns = 1 - (rank - 32) * 0.01 if rank > 32 else 1
            
            performance = base_performance * rank_factor * diminishing_returns
            
            rank_analysis[rank] = {
                "bleu_score": performance,
                "rouge_l": performance * 0.85,
                "perplexity": 20.0 / performance,
                "parameter_count": rank * 768 * 2 * 4,
                "memory_usage_mb": 100 + rank * 2.5,
                "training_time_hours": 2.0 + rank * 0.1
            }
        
        return rank_analysis
    
    def generate_evaluation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive evaluation report"""
        report = "\n" + "=" * 60 + "\n"
        report += "LoRA Adapter Evaluation Report\n"
        report += "=" * 60 + "\n\n"
        
        if "comparison" in results:
            report += "Adapter Comparison Results:\n"
            report += "-" * 30 + "\n"
            
            comparison = results["comparison"]
            for adapter, metrics in comparison.items():
                report += f"\n{adapter}:\n"
                report += f"  BLEU Score: {metrics['bleu_score']:.3f}\n"
                report += f"  ROUGE-L: {metrics['rouge_l']:.3f}\n"
                report += f"  Inference Speed: {metrics['inference_speed_ms']:.1f}ms\n"
                report += f"  Memory Usage: {metrics['memory_usage_mb']:.1f}MB\n"
        
        if "rank_analysis" in results:
            report += "\n\nRank Sensitivity Analysis:\n"
            report += "-" * 30 + "\n"
            
            rank_analysis = results["rank_analysis"]
            report += f"{'Rank':<6} {'BLEU':<8} {'Params':<10} {'Memory':<10} {'Time':<8}\n"
            report += "-" * 50 + "\n"
            
            for rank, metrics in sorted(rank_analysis.items()):
                report += f"{rank:<6} "
                report += f"{metrics['bleu_score']:<8.3f} "
                report += f"{metrics['parameter_count']:<10,} "
                report += f"{metrics['memory_usage_mb']:<10.1f} "
                report += f"{metrics['training_time_hours']:<8.1f}\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report

def exercise_7_evaluation_framework():
    """Exercise 7: Evaluation and comparison framework"""
    print("=== Exercise 7: Evaluation and Comparison Framework ===")
    
    evaluator = LoRAEvaluationFramework()
    
    # Mock evaluation scenarios
    adapters = ["customer_support_v1", "code_assistant_v2", "content_writer_v3"]
    ranks = [8, 16, 32, 64]
    
    print("Evaluating individual adapters:")
    for adapter in adapters:
        metrics = evaluator.evaluate_adapter_quality("llama-7b", adapter, 1000)
        print(f"  {adapter}: BLEU={metrics['bleu_score']:.3f}, Speed={metrics['inference_speed_ms']:.1f}ms")
    
    print(f"\nComparing adapters:")
    comparison = evaluator.compare_adapters("llama-7b", adapters, 1000)
    
    # Find best performer
    best_adapter = max(comparison.keys(), key=lambda x: comparison[x]["bleu_score"])
    print(f"  Best performer: {best_adapter} (BLEU: {comparison[best_adapter]['bleu_score']:.3f})")
    
    print(f"\nAnalyzing rank sensitivity:")
    rank_analysis = evaluator.analyze_rank_sensitivity("llama-7b", ranks, 1000)
    
    print(f"  Rank vs Performance:")
    for rank in ranks:
        metrics = rank_analysis[rank]
        print(f"    Rank {rank:2d}: BLEU={metrics['bleu_score']:.3f}, "
              f"Params={metrics['parameter_count']:,}, "
              f"Memory={metrics['memory_usage_mb']:.1f}MB")
    
    print(f"\nGenerating comprehensive report:")
    report = evaluator.generate_evaluation_report({
        "comparison": comparison, 
        "rank_analysis": rank_analysis
    })
    print(report)
    
    print("✅ Evaluation framework completed successfully!")
    print()

def main():
    """Run all LoRA and QLoRA exercises"""
    print("Day 48: Fine-tuning Techniques - LoRA & QLoRA - Exercises")
    print("=" * 70)
    print()
    
    # Run all exercises
    exercise_1_basic_lora()
    exercise_2_lora_configuration()
    exercise_3_qlora_setup()
    exercise_4_multitask_lora()
    exercise_5_memory_optimization()
    exercise_6_deployment_strategies()
    exercise_7_evaluation_framework()
    
    print("=" * 70)
    print("LoRA and QLoRA exercises completed! Check the solution.py file for complete implementations.")
    print()
    print("Next steps:")
    print("1. Implement the TODO sections in each exercise")
    print("2. Test with real transformer models using Hugging Face PEFT")
    print("3. Experiment with different rank configurations")
    print("4. Try QLoRA on large models (7B+ parameters)")
    print("5. Deploy adapters in production environments")

if __name__ == "__main__":
    main()
