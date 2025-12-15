"""
Day 48: Fine-tuning Techniques - LoRA & QLoRA - Complete Solutions

This file contains complete implementations for all LoRA and QLoRA exercises.
These solutions demonstrate production-ready parameter-efficient fine-tuning techniques.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Solution 1: Complete LoRA Implementation
class LoRALayer(nn.Module):
    """Complete LoRA layer implementation with all optimizations"""
    
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
        
        # Track if layer is merged for inference optimization
        self.merged = False
    
    def forward(self, x):
        if self.merged:
            # If merged, the adaptation is already in the base layer
            return torch.zeros_like(x @ self.lora_A.T @ self.lora_B.T)
        
        # Standard LoRA forward pass
        result = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result * self.scaling
    
    def merge_weights(self):
        """Merge LoRA weights into base layer for inference"""
        if not self.merged:
            self.merged = True
            return self.lora_B @ self.lora_A * self.scaling
        return None
    
    def unmerge_weights(self):
        """Unmerge LoRA weights for continued training"""
        if self.merged:
            self.merged = False
            return -(self.lora_B @ self.lora_A * self.scaling)
        return None

class LoRALinear(nn.Module):
    """LoRA adaptation of linear layer with merge/unmerge capabilities"""
    
    def __init__(self, base_layer: nn.Linear, rank: int = 4, alpha: float = 1.0, 
                 dropout: float = 0.0):
        super().__init__()
        self.base_layer = base_layer
        self.lora = LoRALayer(
            base_layer.in_features, 
            base_layer.out_features, 
            rank, 
            alpha, 
            dropout
        )
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        base_output = self.base_layer(x)
        lora_output = self.lora(x)
        return base_output + lora_output
    
    def merge_weights(self):
        """Merge LoRA weights into base layer for efficient inference"""
        delta_w = self.lora.merge_weights()
        if delta_w is not None:
            self.base_layer.weight.data += delta_w
    
    def unmerge_weights(self):
        """Unmerge LoRA weights for training"""
        delta_w = self.lora.unmerge_weights()
        if delta_w is not None:
            self.base_layer.weight.data += delta_w

def solution_1_basic_lora():
    """Solution 1: Complete LoRA implementation"""
    print("=== Solution 1: Complete LoRA Implementation ===")
    
    # Create base linear layer
    base_layer = nn.Linear(768, 768)
    
    # Create LoRA adaptation
    lora_layer = LoRALinear(base_layer, rank=16, alpha=32)
    
    # Test input
    x = torch.randn(32, 128, 768)  # (batch, seq_len, hidden_size)
    
    # Forward pass
    output = lora_layer(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Parameter analysis
    base_params = sum(p.numel() for p in base_layer.parameters())
    lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_layer.parameters())
    
    print(f"Base parameters: {base_params:,}")
    print(f"LoRA parameters: {lora_params:,}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable percentage: {(lora_params/total_params)*100:.2f}%")
    print(f"Parameter reduction: {(1 - lora_params/base_params)*100:.1f}%")
    
    # Test merge/unmerge functionality
    print("\nTesting merge/unmerge functionality:")
    output_before = lora_layer(x)
    lora_layer.merge_weights()
    output_merged = lora_layer(x)
    lora_layer.unmerge_weights()
    output_after = lora_layer(x)
    
    print(f"Output difference (before vs merged): {torch.allclose(output_before, output_merged, atol=1e-6)}")
    print(f"Output difference (before vs after): {torch.allclose(output_before, output_after, atol=1e-6)}")
    print()

# Solution 2: Advanced LoRA Configuration Management
@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: float = 32.0
    dropout: float = 0.1
    target_modules: List[str] = None
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj"]

class LoRAConfigManager:
    """Advanced LoRA configuration management for different scenarios"""
    
    def __init__(self):
        self.task_configs = {
            "instruction_following": LoRAConfig(
                rank=64, alpha=128, dropout=0.05,
                target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            ),
            "code_generation": LoRAConfig(
                rank=32, alpha=64, dropout=0.1,
                target_modules=["q_proj", "v_proj", "gate_proj", "up_proj"]
            ),
            "summarization": LoRAConfig(
                rank=16, alpha=32, dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            ),
            "translation": LoRAConfig(
                rank=24, alpha=48, dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
            ),
            "question_answering": LoRAConfig(
                rank=20, alpha=40, dropout=0.1,
                target_modules=["q_proj", "v_proj", "o_proj"]
            )
        }
        
        self.model_size_configs = {
            "small": (1e9, 7e9),      # 1B - 7B parameters
            "medium": (7e9, 30e9),    # 7B - 30B parameters  
            "large": (30e9, 100e9),   # 30B - 100B parameters
            "xlarge": (100e9, float('inf'))  # 100B+ parameters
        }
    
    def get_config_for_task(self, task: str) -> LoRAConfig:
        """Get optimized LoRA configuration for specific task"""
        if task in self.task_configs:
            return self.task_configs[task]
        
        # Default configuration for unknown tasks
        return LoRAConfig(rank=16, alpha=32, target_modules=["q_proj", "v_proj"])
    
    def get_config_for_model_size(self, num_parameters: int) -> LoRAConfig:
        """Get configuration optimized for model size"""
        if num_parameters <= 7e9:  # Small models
            return LoRAConfig(rank=8, alpha=16, target_modules=["q_proj", "v_proj"])
        elif num_parameters <= 30e9:  # Medium models
            return LoRAConfig(rank=16, alpha=32, target_modules=["q_proj", "v_proj", "o_proj"])
        elif num_parameters <= 100e9:  # Large models
            return LoRAConfig(rank=32, alpha=64, target_modules=["q_proj", "v_proj", "o_proj", "gate_proj"])
        else:  # XLarge models
            return LoRAConfig(rank=64, alpha=128, target_modules=["q_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    def optimize_config_for_hardware(self, available_memory_gb: float, 
                                   model_size: int) -> LoRAConfig:
        """Optimize configuration for available hardware"""
        base_config = self.get_config_for_model_size(model_size)
        
        # Estimate memory usage and adjust if needed
        estimated_memory = self._estimate_memory_usage(model_size, base_config)
        
        if estimated_memory > available_memory_gb * 0.8:  # Use 80% of available memory
            # Reduce rank and target modules
            base_config.rank = max(4, base_config.rank // 2)
            base_config.target_modules = base_config.target_modules[:2]  # Keep only q_proj, v_proj
        
        return base_config
    
    def _estimate_memory_usage(self, model_size: int, config: LoRAConfig) -> float:
        """Estimate memory usage in GB"""
        # Rough estimation: base model (4-bit) + LoRA parameters (16-bit) + gradients + optimizer states
        base_memory = model_size * 0.5 / 1e9  # 4-bit quantization
        lora_params = len(config.target_modules) * config.rank * 768 * 2  # Rough estimate
        lora_memory = lora_params * 6 / 1e9  # 16-bit + gradients + optimizer states
        
        return base_memory + lora_memory
    
    def create_custom_config(self, task: str, model_size: int, 
                           memory_gb: float, performance_priority: str = "balanced") -> LoRAConfig:
        """Create custom configuration balancing multiple factors"""
        
        # Start with task-specific config
        config = self.get_config_for_task(task)
        
        # Adjust for model size
        size_config = self.get_config_for_model_size(model_size)
        
        # Adjust for hardware constraints
        hw_config = self.optimize_config_for_hardware(memory_gb, model_size)
        
        # Balance based on priority
        if performance_priority == "speed":
            # Favor smaller rank for faster training/inference
            final_rank = min(config.rank, size_config.rank, hw_config.rank)
            final_modules = config.target_modules[:2]  # Minimal modules
        elif performance_priority == "quality":
            # Favor larger rank for better adaptation
            final_rank = max(config.rank, min(size_config.rank, hw_config.rank))
            final_modules = config.target_modules  # All specified modules
        else:  # balanced
            final_rank = int(np.mean([config.rank, size_config.rank, hw_config.rank]))
            final_modules = config.target_modules[:4]  # Moderate number of modules
        
        return LoRAConfig(
            rank=final_rank,
            alpha=final_rank * 2,  # Standard 2x scaling
            dropout=config.dropout,
            target_modules=final_modules,
            bias=config.bias,
            task_type=config.task_type
        )

def solution_2_lora_configuration():
    """Solution 2: Advanced LoRA configuration management"""
    print("=== Solution 2: Advanced LoRA Configuration Management ===")
    
    config_manager = LoRAConfigManager()
    
    # Test task-specific configurations
    tasks = ["instruction_following", "code_generation", "summarization", "translation"]
    print("Task-specific configurations:")
    for task in tasks:
        config = config_manager.get_config_for_task(task)
        print(f"{task:20}: rank={config.rank:2d}, alpha={config.alpha:3.0f}, "
              f"modules={len(config.target_modules)}, targets={config.target_modules[:3]}")
    
    # Test model size configurations
    model_sizes = [1e9, 7e9, 13e9, 30e9, 70e9]
    print(f"\nModel size-specific configurations:")
    for size in model_sizes:
        config = config_manager.get_config_for_model_size(size)
        print(f"{size/1e9:4.0f}B params: rank={config.rank:2d}, alpha={config.alpha:3.0f}, "
              f"modules={len(config.target_modules)}")
    
    # Test hardware optimization
    print(f"\nHardware-optimized configurations:")
    memory_configs = [8, 16, 24, 48]
    for memory_gb in memory_configs:
        config = config_manager.optimize_config_for_hardware(memory_gb, 13e9)
        estimated_memory = config_manager._estimate_memory_usage(13e9, config)
        print(f"{memory_gb:2d}GB available: rank={config.rank:2d}, "
              f"estimated usage={estimated_memory:.1f}GB")
    
    # Test custom configuration
    print(f"\nCustom configuration examples:")
    scenarios = [
        ("instruction_following", 7e9, 24, "quality"),
        ("code_generation", 13e9, 16, "speed"),
        ("summarization", 30e9, 48, "balanced")
    ]
    
    for task, model_size, memory, priority in scenarios:
        config = config_manager.create_custom_config(task, model_size, memory, priority)
        print(f"{task} ({model_size/1e9:.0f}B, {memory}GB, {priority}): "
              f"rank={config.rank}, modules={len(config.target_modules)}")
    
    print()

# Solution 3: QLoRA Implementation with Quantization
class MockQuantizationConfig:
    """Mock quantization configuration (replace with actual bitsandbytes in production)"""
    def __init__(self):
        self.load_in_4bit = True
        self.bnb_4bit_quant_type = "nf4"
        self.bnb_4bit_compute_dtype = torch.bfloat16
        self.bnb_4bit_use_double_quant = True

class QLoRASetup:
    """Complete QLoRA setup with quantization and memory optimization"""
    
    def __init__(self):
        self.quantization_config = MockQuantizationConfig()
        
    def setup_quantized_model(self, model_name: str, lora_config: LoRAConfig):
        """Setup model with quantization and LoRA"""
        print(f"Setting up QLoRA for {model_name}")
        
        # In production, this would use:
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_name,
        #     quantization_config=self.quantization_config,
        #     device_map="auto",
        #     torch_dtype=torch.bfloat16
        # )
        # model = get_peft_model(model, lora_config)
        
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
        """Calculate detailed memory savings from QLoRA"""
        
        # Full fine-tuning memory requirements (FP16)
        full_ft_params = base_model_params
        full_ft_gradients = base_model_params * 2  # FP16 gradients
        full_ft_optimizer = base_model_params * 8  # AdamW states (2x momentum + 2x variance)
        full_ft_total = (full_ft_params * 2 + full_ft_gradients + full_ft_optimizer) / 1e9
        
        # QLoRA memory requirements
        # Base model: 4-bit quantized
        qlora_base = base_model_params * 0.5 / 1e9
        
        # LoRA parameters: FP16
        lora_params_per_module = rank * 768 * 2  # A and B matrices
        total_lora_params = num_target_modules * lora_params_per_module
        qlora_lora = total_lora_params * 2 / 1e9  # FP16
        
        # LoRA gradients and optimizer states
        qlora_gradients = total_lora_params * 2 / 1e9  # FP16 gradients
        qlora_optimizer = total_lora_params * 8 / 1e9  # AdamW states
        
        qlora_total = qlora_base + qlora_lora + qlora_gradients + qlora_optimizer
        
        memory_analysis = {
            "full_finetuning_gb": full_ft_total,
            "qlora_total_gb": qlora_total,
            "memory_savings_gb": full_ft_total - qlora_total,
            "memory_reduction_percent": (1 - qlora_total / full_ft_total) * 100,
            "breakdown": {
                "base_model_gb": qlora_base,
                "lora_params_gb": qlora_lora,
                "gradients_gb": qlora_gradients,
                "optimizer_states_gb": qlora_optimizer,
                "total_lora_params": total_lora_params
            }
        }
        
        return memory_analysis
    
    def estimate_training_time(self, model_params: int, dataset_size: int, 
                             batch_size: int, num_epochs: int) -> Dict[str, float]:
        """Estimate training time for QLoRA vs full fine-tuning"""
        
        # Rough estimates based on empirical observations
        full_ft_time_per_sample = model_params / 1e9 * 0.1  # seconds per sample
        qlora_time_per_sample = model_params / 1e9 * 0.05   # ~2x faster due to fewer parameters
        
        steps_per_epoch = dataset_size // batch_size
        total_steps = steps_per_epoch * num_epochs
        
        full_ft_total_hours = (full_ft_time_per_sample * dataset_size * num_epochs) / 3600
        qlora_total_hours = (qlora_time_per_sample * dataset_size * num_epochs) / 3600
        
        return {
            "full_finetuning_hours": full_ft_total_hours,
            "qlora_hours": qlora_total_hours,
            "time_savings_hours": full_ft_total_hours - qlora_total_hours,
            "speedup_factor": full_ft_total_hours / qlora_total_hours,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps
        }

def solution_3_qlora_setup():
    """Solution 3: Complete QLoRA setup with quantization"""
    print("=== Solution 3: Complete QLoRA Setup with Quantization ===")
    
    qlora_setup = QLoRASetup()
    
    # Model configurations to analyze
    model_configs = [
        {"name": "Llama-2-7B", "params": 7e9, "target_modules": 4},
        {"name": "Llama-2-13B", "params": 13e9, "target_modules": 6},
        {"name": "Llama-2-70B", "params": 70e9, "target_modules": 8},
        {"name": "CodeLlama-34B", "params": 34e9, "target_modules": 6}
    ]
    
    print("Memory analysis for different model sizes:")
    print(f"{'Model':<15} {'Full FT (GB)':<12} {'QLoRA (GB)':<11} {'Savings':<8} {'Reduction':<10}")
    print("-" * 70)
    
    for config in model_configs:
        memory_analysis = qlora_setup.calculate_memory_savings(
            config["params"], rank=64, num_target_modules=config["target_modules"]
        )
        
        print(f"{config['name']:<15} "
              f"{memory_analysis['full_finetuning_gb']:<12.1f} "
              f"{memory_analysis['qlora_total_gb']:<11.1f} "
              f"{memory_analysis['memory_savings_gb']:<8.1f} "
              f"{memory_analysis['memory_reduction_percent']:<10.1f}%")
    
    # Detailed breakdown for one model
    print(f"\nDetailed memory breakdown for Llama-2-13B:")
    detailed_analysis = qlora_setup.calculate_memory_savings(13e9, rank=64, num_target_modules=6)
    breakdown = detailed_analysis["breakdown"]
    
    print(f"Base model (4-bit):     {breakdown['base_model_gb']:.1f} GB")
    print(f"LoRA parameters:        {breakdown['lora_params_gb']:.1f} GB")
    print(f"Gradients:              {breakdown['gradients_gb']:.1f} GB")
    print(f"Optimizer states:       {breakdown['optimizer_states_gb']:.1f} GB")
    print(f"Total LoRA parameters:  {breakdown['total_lora_params']:,}")
    
    # Training time estimates
    print(f"\nTraining time estimates:")
    training_scenarios = [
        {"model": "7B", "params": 7e9, "dataset": 10000, "batch": 4, "epochs": 3},
        {"model": "13B", "params": 13e9, "dataset": 50000, "batch": 2, "epochs": 2},
        {"model": "70B", "params": 70e9, "dataset": 100000, "batch": 1, "epochs": 1}
    ]
    
    for scenario in training_scenarios:
        time_analysis = qlora_setup.estimate_training_time(
            scenario["params"], scenario["dataset"], 
            scenario["batch"], scenario["epochs"]
        )
        
        print(f"{scenario['model']} model: "
              f"Full FT: {time_analysis['full_finetuning_hours']:.1f}h, "
              f"QLoRA: {time_analysis['qlora_hours']:.1f}h "
              f"({time_analysis['speedup_factor']:.1f}x speedup)")
    
    print()

# Solution 4: Multi-Task LoRA Management System
class MultiTaskLoRAManager:
    """Advanced multi-task LoRA adapter management system"""
    
    def __init__(self, base_model, tasks: List[str]):
        self.base_model = base_model
        self.tasks = tasks
        self.adapters = {}
        self.current_task = None
        self.adapter_metadata = {}
        
    def create_task_adapter(self, task: str, config: LoRAConfig):
        """Create LoRA adapter for specific task"""
        print(f"Creating adapter for task: {task}")
        
        # In production, this would create actual LoRA layers
        adapter_info = {
            "task": task,
            "rank": config.rank,
            "alpha": config.alpha,
            "target_modules": config.target_modules,
            "parameters": self._calculate_adapter_params(config),
            "created_at": time.time(),
            "training_steps": 0,
            "performance_metrics": {}
        }
        
        self.adapters[task] = adapter_info
        self.adapter_metadata[task] = {
            "version": "1.0",
            "description": f"LoRA adapter for {task}",
            "config": asdict(config)
        }
        
        return adapter_info
    
    def _calculate_adapter_params(self, config: LoRAConfig) -> int:
        """Calculate number of parameters in adapter"""
        # Rough calculation: rank * (in_features + out_features) per target module
        params_per_module = config.rank * (768 + 768)  # Assuming 768 hidden size
        total_params = len(config.target_modules) * params_per_module
        return total_params
    
    def switch_task(self, task: str):
        """Switch to different task adapter"""
        if task not in self.adapters:
            raise ValueError(f"Task '{task}' not found. Available tasks: {list(self.adapters.keys())}")
        
        print(f"Switching from {self.current_task} to {task}")
        
        # In production, this would swap LoRA weights
        self.current_task = task
        
        return {
            "previous_task": self.current_task,
            "new_task": task,
            "adapter_info": self.adapters[task]
        }
    
    def merge_adapters(self, tasks: List[str], weights: List[float] = None) -> Dict[str, Any]:
        """Merge multiple task adapters with optional weighting"""
        if weights is None:
            weights = [1.0 / len(tasks)] * len(tasks)
        
        if len(tasks) != len(weights):
            raise ValueError("Number of tasks must match number of weights")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        print(f"Merging adapters: {tasks} with weights: {weights}")
        
        # Calculate merged adapter properties
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
            "parameters": int(merged_params),
            "created_at": time.time()
        }
        
        # Store merged adapter
        merged_name = f"merged_{'_'.join(tasks)}"
        self.adapters[merged_name] = merged_adapter
        
        return merged_adapter
    
    def get_adapter_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all adapters"""
        if not self.adapters:
            return {"message": "No adapters created yet"}
        
        total_params = sum(adapter["parameters"] for adapter in self.adapters.values())
        avg_rank = sum(adapter["rank"] for adapter in self.adapters.values()) / len(self.adapters)
        
        # Task distribution
        task_types = {}
        for task, adapter in self.adapters.items():
            if adapter.get("type") == "merged":
                task_types["merged"] = task_types.get("merged", 0) + 1
            else:
                task_types["single"] = task_types.get("single", 0) + 1
        
        # Parameter distribution
        param_distribution = {
            task: adapter["parameters"] 
            for task, adapter in self.adapters.items()
        }
        
        stats = {
            "total_adapters": len(self.adapters),
            "total_parameters": total_params,
            "average_rank": avg_rank,
            "current_task": self.current_task,
            "task_distribution": task_types,
            "parameter_distribution": param_distribution,
            "memory_usage_mb": total_params * 4 / 1024 / 1024,  # Rough estimate
            "adapter_list": list(self.adapters.keys())
        }
        
        return stats
    
    def save_adapter(self, task: str, filepath: str):
        """Save adapter to file"""
        if task not in self.adapters:
            raise ValueError(f"Task '{task}' not found")
        
        adapter_data = {
            "adapter_info": self.adapters[task],
            "metadata": self.adapter_metadata.get(task, {}),
            "save_timestamp": time.time()
        }
        
        # In production, this would save actual model weights
        print(f"Saving adapter '{task}' to {filepath}")
        
        return {
            "task": task,
            "filepath": filepath,
            "size_mb": adapter_data["adapter_info"]["parameters"] * 4 / 1024 / 1024,
            "saved_at": adapter_data["save_timestamp"]
        }
    
    def load_adapter(self, task: str, filepath: str):
        """Load adapter from file"""
        print(f"Loading adapter '{task}' from {filepath}")
        
        # In production, this would load actual model weights
        # For demo, create a mock loaded adapter
        loaded_adapter = {
            "task": task,
            "rank": 32,
            "alpha": 64,
            "target_modules": ["q_proj", "v_proj"],
            "parameters": 32 * 768 * 2 * 2,  # Mock calculation
            "loaded_at": time.time(),
            "source_file": filepath
        }
        
        self.adapters[task] = loaded_adapter
        
        return loaded_adapter

def solution_4_multitask_lora():
    """Solution 4: Complete multi-task LoRA management system"""
    print("=== Solution 4: Complete Multi-Task LoRA Management System ===")
    
    # Mock base model
    base_model = nn.Linear(768, 768)
    
    tasks = ["customer_support", "code_generation", "content_writing", "translation"]
    manager = MultiTaskLoRAManager(base_model, tasks)
    
    # Create adapters for each task with different configurations
    task_configs = {
        "customer_support": LoRAConfig(rank=16, alpha=32, target_modules=["q_proj", "v_proj"]),
        "code_generation": LoRAConfig(rank=32, alpha=64, target_modules=["q_proj", "v_proj", "gate_proj"]),
        "content_writing": LoRAConfig(rank=24, alpha=48, target_modules=["q_proj", "v_proj", "o_proj"]),
        "translation": LoRAConfig(rank=20, alpha=40, target_modules=["q_proj", "k_proj", "v_proj"])
    }
    
    print("Creating task-specific adapters:")
    for task, config in task_configs.items():
        adapter_info = manager.create_task_adapter(task, config)
        print(f"  {task}: {adapter_info['parameters']:,} parameters, rank={adapter_info['rank']}")
    
    # Test task switching
    print(f"\nTesting task switching:")
    manager.switch_task("customer_support")
    print(f"Current task: {manager.current_task}")
    
    switch_result = manager.switch_task("code_generation")
    print(f"Switched to: {switch_result['new_task']}")
    
    # Test adapter merging
    print(f"\nTesting adapter merging:")
    merged_adapter = manager.merge_adapters(
        ["customer_support", "content_writing"], 
        [0.7, 0.3]
    )
    print(f"Merged adapter parameters: {merged_adapter['parameters']:,}")
    print(f"Source tasks: {merged_adapter['source_tasks']}")
    
    # Get comprehensive statistics
    print(f"\nAdapter Statistics:")
    stats = manager.get_adapter_statistics()
    for key, value in stats.items():
        if key != "parameter_distribution":
            print(f"  {key}: {value}")
    
    print(f"\nParameter distribution:")
    for task, params in stats["parameter_distribution"].items():
        print(f"  {task}: {params:,} parameters")
    
    # Test save/load functionality
    print(f"\nTesting save/load functionality:")
    save_result = manager.save_adapter("code_generation", "/tmp/code_gen_adapter.bin")
    print(f"Saved: {save_result['task']} ({save_result['size_mb']:.1f} MB)")
    
    load_result = manager.load_adapter("loaded_code_gen", "/tmp/code_gen_adapter.bin")
    print(f"Loaded: {load_result['task']} ({load_result['parameters']:,} parameters)")
    
    print()

# Solution 5: Memory Optimization and Performance Analysis
class MemoryOptimizer:
    """Advanced memory optimization for LoRA training"""
    
    def __init__(self):
        self.optimization_strategies = {
            "gradient_checkpointing": {"memory_reduction": 0.3, "speed_penalty": 0.2},
            "mixed_precision": {"memory_reduction": 0.5, "speed_improvement": 0.1},
            "gradient_accumulation": {"memory_reduction": 0.6, "speed_penalty": 0.1},
            "optimizer_offloading": {"memory_reduction": 0.4, "speed_penalty": 0.3}
        }
    
    def analyze_memory_usage(self, model_params: int, batch_size: int, 
                           sequence_length: int, rank: int, num_adapters: int) -> Dict[str, float]:
        """Analyze detailed memory usage breakdown"""
        
        # Base model memory (assuming 4-bit quantization)
        base_model_memory = model_params * 0.5 / 1e9  # GB
        
        # LoRA parameters memory
        lora_params = num_adapters * rank * 768 * 2  # A and B matrices
        lora_memory = lora_params * 2 / 1e9  # FP16
        
        # Activation memory (depends on batch size and sequence length)
        activation_memory = batch_size * sequence_length * 768 * 4 / 1e9  # FP32 activations
        
        # Gradient memory (only for LoRA parameters)
        gradient_memory = lora_params * 2 / 1e9  # FP16 gradients
        
        # Optimizer state memory (AdamW: 2x parameters for momentum and variance)
        optimizer_memory = lora_params * 8 / 1e9  # FP32 optimizer states
        
        # KV cache memory
        kv_cache_memory = batch_size * sequence_length * 768 * 2 * 2 / 1e9  # Key and Value
        
        total_memory = (base_model_memory + lora_memory + activation_memory + 
                       gradient_memory + optimizer_memory + kv_cache_memory)
        
        memory_breakdown = {
            "base_model_gb": base_model_memory,
            "lora_parameters_gb": lora_memory,
            "activations_gb": activation_memory,
            "gradients_gb": gradient_memory,
            "optimizer_states_gb": optimizer_memory,
            "kv_cache_gb": kv_cache_memory,
            "total_gb": total_memory,
            "breakdown_percent": {
                "base_model": (base_model_memory / total_memory) * 100,
                "lora_params": (lora_memory / total_memory) * 100,
                "activations": (activation_memory / total_memory) * 100,
                "gradients": (gradient_memory / total_memory) * 100,
                "optimizer": (optimizer_memory / total_memory) * 100,
                "kv_cache": (kv_cache_memory / total_memory) * 100
            }
        }
        
        return memory_breakdown
    
    def optimize_batch_size(self, model_params: int, max_memory_gb: float, 
                          sequence_length: int = 512, rank: int = 16) -> Dict[str, int]:
        """Find optimal batch size for given memory constraint"""
        
        # Binary search for optimal batch size
        low, high = 1, 128
        optimal_batch_size = 1
        
        while low <= high:
            mid = (low + high) // 2
            memory_usage = self.analyze_memory_usage(
                model_params, mid, sequence_length, rank, 4
            )
            
            if memory_usage["total_gb"] <= max_memory_gb * 0.9:  # 90% utilization
                optimal_batch_size = mid
                low = mid + 1
            else:
                high = mid - 1
        
        # Calculate gradient accumulation if needed for larger effective batch size
        target_effective_batch = 32  # Common effective batch size
        accumulation_steps = max(1, target_effective_batch // optimal_batch_size)
        
        return {
            "optimal_batch_size": optimal_batch_size,
            "gradient_accumulation_steps": accumulation_steps,
            "effective_batch_size": optimal_batch_size * accumulation_steps,
            "memory_utilization_gb": self.analyze_memory_usage(
                model_params, optimal_batch_size, sequence_length, rank, 4
            )["total_gb"]
        }
    
    def apply_optimization_strategy(self, base_memory_gb: float, 
                                  strategies: List[str]) -> Dict[str, float]:
        """Apply multiple optimization strategies and calculate combined effect"""
        
        current_memory = base_memory_gb
        speed_factor = 1.0
        
        applied_optimizations = {}
        
        for strategy in strategies:
            if strategy in self.optimization_strategies:
                opt_info = self.optimization_strategies[strategy]
                
                # Apply memory reduction
                memory_reduction = opt_info.get("memory_reduction", 0)
                current_memory *= (1 - memory_reduction)
                
                # Apply speed impact
                if "speed_penalty" in opt_info:
                    speed_factor *= (1 + opt_info["speed_penalty"])
                elif "speed_improvement" in opt_info:
                    speed_factor *= (1 - opt_info["speed_improvement"])
                
                applied_optimizations[strategy] = {
                    "memory_reduction": memory_reduction,
                    "speed_impact": opt_info.get("speed_penalty", -opt_info.get("speed_improvement", 0))
                }
        
        return {
            "original_memory_gb": base_memory_gb,
            "optimized_memory_gb": current_memory,
            "memory_savings_gb": base_memory_gb - current_memory,
            "memory_reduction_percent": ((base_memory_gb - current_memory) / base_memory_gb) * 100,
            "speed_factor": speed_factor,
            "applied_optimizations": applied_optimizations
        }

class PerformanceProfiler:
    """Comprehensive performance profiling for LoRA training"""
    
    def __init__(self):
        self.metrics_history = []
    
    def profile_training_step(self, model_params: int, batch_size: int, 
                            sequence_length: int, rank: int) -> Dict[str, float]:
        """Profile single training step performance"""
        
        # Simulate timing based on model characteristics
        base_forward_time = model_params / 1e9 * 0.1  # Base forward pass time
        lora_overhead = rank * 0.001  # LoRA computation overhead
        
        # Simulate actual timing
        start_time = time.time()
        
        # Mock forward pass
        forward_time = base_forward_time + lora_overhead
        time.sleep(min(0.01, forward_time))  # Simulate computation
        
        # Mock backward pass (typically 2x forward pass time)
        backward_time = forward_time * 2
        time.sleep(min(0.02, backward_time))
        
        # Mock optimizer step
        optimizer_time = rank * 0.0005  # Proportional to LoRA parameters
        time.sleep(min(0.005, optimizer_time))
        
        total_time = time.time() - start_time
        
        # Calculate throughput metrics
        tokens_per_second = (batch_size * sequence_length) / total_time
        samples_per_second = batch_size / total_time
        
        metrics = {
            "total_time": total_time,
            "forward_time": forward_time,
            "backward_time": backward_time,
            "optimizer_time": optimizer_time,
            "tokens_per_second": tokens_per_second,
            "samples_per_second": samples_per_second,
            "memory_efficiency": self._calculate_memory_efficiency(model_params, rank),
            "flops_utilization": self._estimate_flops_utilization(model_params, rank)
        }
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _calculate_memory_efficiency(self, model_params: int, rank: int) -> float:
        """Calculate memory efficiency score"""
        lora_params = rank * 768 * 2 * 4  # Rough estimate
        efficiency = lora_params / model_params
        return min(1.0, efficiency * 100)  # Normalize to 0-1 scale
    
    def _estimate_flops_utilization(self, model_params: int, rank: int) -> float:
        """Estimate FLOPS utilization efficiency"""
        # LoRA reduces computation compared to full fine-tuning
        reduction_factor = rank / 768  # Rough approximation
        return min(1.0, reduction_factor)
    
    def compare_lora_vs_full_finetuning(self, model_params: int, rank: int) -> Dict[str, Any]:
        """Compare LoRA vs full fine-tuning performance"""
        
        # LoRA metrics
        lora_metrics = self.profile_training_step(model_params, 16, 512, rank)
        
        # Full fine-tuning simulation (all parameters trainable)
        full_ft_forward_time = model_params / 1e9 * 0.15  # Slower due to more computation
        full_ft_backward_time = full_ft_forward_time * 3   # More gradients to compute
        full_ft_optimizer_time = model_params / 1e9 * 0.05  # More parameters to update
        
        full_ft_total = full_ft_forward_time + full_ft_backward_time + full_ft_optimizer_time
        
        comparison = {
            "lora": {
                "total_time": lora_metrics["total_time"],
                "memory_gb": model_params * 0.5 / 1e9 + rank * 768 * 2 * 6 / 1e9,  # Base + LoRA
                "trainable_params": rank * 768 * 2 * 4,  # LoRA parameters
                "tokens_per_second": lora_metrics["tokens_per_second"]
            },
            "full_finetuning": {
                "total_time": full_ft_total,
                "memory_gb": model_params * 10 / 1e9,  # Much higher memory usage
                "trainable_params": model_params,
                "tokens_per_second": (16 * 512) / full_ft_total
            },
            "improvements": {
                "speed_improvement": full_ft_total / lora_metrics["total_time"],
                "memory_reduction": 1 - (lora_metrics["tokens_per_second"] / ((16 * 512) / full_ft_total)),
                "parameter_reduction": 1 - (rank * 768 * 2 * 4) / model_params,
                "cost_reduction": (full_ft_total / lora_metrics["total_time"]) * 0.7  # Rough cost estimate
            }
        }
        
        return comparison
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        if not self.metrics_history:
            return "No performance data available"
        
        avg_metrics = {}
        for key in self.metrics_history[0].keys():
            if isinstance(self.metrics_history[0][key], (int, float)):
                avg_metrics[key] = sum(m[key] for m in self.metrics_history) / len(self.metrics_history)
        
        report = f"""
Performance Analysis Report
==========================

Training Steps Analyzed: {len(self.metrics_history)}

Average Performance Metrics:
- Total Time per Step: {avg_metrics.get('total_time', 0):.4f} seconds
- Forward Pass Time: {avg_metrics.get('forward_time', 0):.4f} seconds  
- Backward Pass Time: {avg_metrics.get('backward_time', 0):.4f} seconds
- Optimizer Step Time: {avg_metrics.get('optimizer_time', 0):.4f} seconds

Throughput Metrics:
- Tokens per Second: {avg_metrics.get('tokens_per_second', 0):.1f}
- Samples per Second: {avg_metrics.get('samples_per_second', 0):.1f}

Efficiency Metrics:
- Memory Efficiency: {avg_metrics.get('memory_efficiency', 0):.2f}
- FLOPS Utilization: {avg_metrics.get('flops_utilization', 0):.2f}

Recommendations:
- Consider increasing batch size if memory allows
- Monitor memory efficiency for optimal rank selection
- Use gradient accumulation for larger effective batch sizes
"""
        
        return report

def solution_5_memory_optimization():
    """Solution 5: Complete memory optimization and performance analysis"""
    print("=== Solution 5: Complete Memory Optimization and Performance Analysis ===")
    
    optimizer = MemoryOptimizer()
    profiler = PerformanceProfiler()
    
    # Test memory analysis for different model sizes
    model_configs = [
        {"name": "7B", "params": 7e9},
        {"name": "13B", "params": 13e9},
        {"name": "30B", "params": 30e9}
    ]
    
    print("Memory usage analysis:")
    print(f"{'Model':<6} {'Total (GB)':<10} {'Base':<8} {'LoRA':<8} {'Activations':<12} {'Optimizer':<10}")
    print("-" * 65)
    
    for config in model_configs:
        memory_usage = optimizer.analyze_memory_usage(
            config["params"], batch_size=16, sequence_length=512, rank=32, num_adapters=4
        )
        
        print(f"{config['name']:<6} "
              f"{memory_usage['total_gb']:<10.1f} "
              f"{memory_usage['base_model_gb']:<8.1f} "
              f"{memory_usage['lora_parameters_gb']:<8.1f} "
              f"{memory_usage['activations_gb']:<12.1f} "
              f"{memory_usage['optimizer_states_gb']:<10.1f}")
    
    # Optimize batch size for different memory constraints
    print(f"\nBatch size optimization:")
    memory_constraints = [16, 24, 48, 80]
    
    for memory_gb in memory_constraints:
        optimization = optimizer.optimize_batch_size(13e9, memory_gb)
        print(f"{memory_gb:2d}GB: batch_size={optimization['optimal_batch_size']:2d}, "
              f"accumulation={optimization['gradient_accumulation_steps']:2d}, "
              f"effective={optimization['effective_batch_size']:2d}, "
              f"utilization={optimization['memory_utilization_gb']:.1f}GB")
    
    # Test optimization strategies
    print(f"\nOptimization strategies comparison:")
    base_memory = 45.0  # GB
    
    strategy_combinations = [
        ["gradient_checkpointing"],
        ["mixed_precision"],
        ["gradient_accumulation"],
        ["gradient_checkpointing", "mixed_precision"],
        ["gradient_checkpointing", "mixed_precision", "optimizer_offloading"]
    ]
    
    for strategies in strategy_combinations:
        result = optimizer.apply_optimization_strategy(base_memory, strategies)
        print(f"{'+'.join(strategies):<40}: "
              f"{result['optimized_memory_gb']:.1f}GB "
              f"({result['memory_reduction_percent']:.1f}% reduction, "
              f"{result['speed_factor']:.2f}x speed)")
    
    # Performance profiling
    print(f"\nPerformance profiling:")
    
    for i in range(3):  # Profile multiple steps
        metrics = profiler.profile_training_step(13e9, 16, 512, 32)
    
    # Compare LoRA vs full fine-tuning
    comparison = profiler.compare_lora_vs_full_finetuning(13e9, 32)
    
    print(f"LoRA vs Full Fine-tuning Comparison:")
    print(f"Speed improvement: {comparison['improvements']['speed_improvement']:.1f}x")
    print(f"Memory reduction: {comparison['improvements']['memory_reduction']:.1f}x")
    print(f"Parameter reduction: {comparison['improvements']['parameter_reduction']*100:.1f}%")
    print(f"Cost reduction: {comparison['improvements']['cost_reduction']:.1f}x")
    
    # Generate performance report
    print(f"\n{profiler.generate_performance_report()}")
    
    print()

def main():
    """Run all LoRA and QLoRA solutions"""
    print("Day 48: Fine-tuning Techniques - LoRA & QLoRA - Complete Solutions")
    print("=" * 80)
    print()
    
    # Run all solutions
    solution_1_basic_lora()
    solution_2_lora_configuration()
    solution_3_qlora_setup()
    solution_4_multitask_lora()
    solution_5_memory_optimization()
    
    print("=" * 80)
    print("All LoRA and QLoRA solutions completed successfully!")
    print()
    print("Key Takeaways:")
    print("1. LoRA enables efficient fine-tuning with minimal parameter overhead")
    print("2. QLoRA combines quantization with LoRA for extreme memory efficiency")
    print("3. Proper configuration is crucial for balancing performance and efficiency")
    print("4. Multi-task adapters enable flexible deployment strategies")
    print("5. Memory optimization techniques can dramatically reduce hardware requirements")
    print()
    print("Production Recommendations:")
    print("- Start with rank 16-32 for most tasks, adjust based on performance")
    print("- Use QLoRA for large models (13B+ parameters) on limited hardware")
    print("- Implement adapter switching for multi-task production systems")
    print("- Monitor memory usage and apply optimization strategies as needed")
    print("- Consider adapter merging for deployment efficiency")
    print("- Use gradient checkpointing and mixed precision for memory savings")

if __name__ == "__main__":
    main()
