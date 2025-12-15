"""
Day 48: Fine-tuning Techniques - LoRA & QLoRA - Comprehensive Test Suite

This test suite validates all LoRA and QLoRA implementations including:
- Basic LoRA layer functionality
- Configuration management
- QLoRA setup and quantization
- Multi-task adapter management
- Memory optimization
- Performance profiling
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solution import (
    LoRALayer, LoRALinear, LoRAConfig, LoRAConfigManager,
    QLoRASetup, MultiTaskLoRAManager, MemoryOptimizer, PerformanceProfiler
)

class TestLoRAImplementation:
    """Test basic LoRA layer implementation"""
    
    def test_lora_layer_initialization(self):
        """Test LoRA layer proper initialization"""
        layer = LoRALayer(768, 768, rank=16, alpha=32)
        
        assert layer.rank == 16
        assert layer.alpha == 32
        assert layer.scaling == 2.0  # alpha / rank
        assert layer.lora_A.shape == (16, 768)
        assert layer.lora_B.shape == (768, 16)
        assert not layer.merged
    
    def test_lora_layer_forward(self):
        """Test LoRA layer forward pass"""
        layer = LoRALayer(768, 768, rank=16, alpha=32)
        x = torch.randn(32, 128, 768)
        
        output = layer(x)
        
        assert output.shape == x.shape
        assert not torch.allclose(output, torch.zeros_like(output))
    
    def test_lora_linear_integration(self):
        """Test LoRA integration with linear layer"""
        base_layer = nn.Linear(768, 768)
        lora_layer = LoRALinear(base_layer, rank=16, alpha=32)
        
        x = torch.randn(32, 128, 768)
        
        # Test forward pass
        output = lora_layer(x)
        assert output.shape == x.shape
        
        # Test that base layer is frozen
        for param in lora_layer.base_layer.parameters():
            assert not param.requires_grad
        
        # Test that LoRA parameters are trainable
        lora_params = [p for p in lora_layer.parameters() if p.requires_grad]
        assert len(lora_params) > 0
    
    def test_lora_merge_unmerge(self):
        """Test LoRA weight merging and unmerging"""
        base_layer = nn.Linear(768, 768)
        lora_layer = LoRALinear(base_layer, rank=16, alpha=32)
        
        x = torch.randn(16, 768)
        
        # Get output before merging
        output_before = lora_layer(x)
        
        # Merge weights
        lora_layer.merge_weights()
        output_merged = lora_layer(x)
        
        # Unmerge weights
        lora_layer.unmerge_weights()
        output_after = lora_layer(x)
        
        # Outputs should be approximately equal
        assert torch.allclose(output_before, output_merged, atol=1e-5)
        assert torch.allclose(output_before, output_after, atol=1e-5)
    
    def test_parameter_efficiency(self):
        """Test parameter efficiency of LoRA"""
        base_layer = nn.Linear(768, 768)
        lora_layer = LoRALinear(base_layer, rank=16, alpha=32)
        
        base_params = sum(p.numel() for p in base_layer.parameters())
        lora_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
        
        # LoRA should use significantly fewer parameters
        assert lora_params < base_params * 0.1  # Less than 10% of base parameters
        assert lora_params == 16 * 768 * 2  # rank * (in_features + out_features)

class TestLoRAConfiguration:
    """Test LoRA configuration management"""
    
    def test_config_initialization(self):
        """Test LoRA config initialization"""
        config = LoRAConfig(rank=32, alpha=64)
        
        assert config.rank == 32
        assert config.alpha == 64
        assert config.target_modules == ["q_proj", "v_proj"]  # Default
    
    def test_config_manager_task_configs(self):
        """Test task-specific configurations"""
        manager = LoRAConfigManager()
        
        # Test known task
        config = manager.get_config_for_task("instruction_following")
        assert config.rank == 64
        assert config.alpha == 128
        
        # Test unknown task (should return default)
        config = manager.get_config_for_task("unknown_task")
        assert config.rank == 16
        assert config.alpha == 32
    
    def test_config_manager_model_size(self):
        """Test model size-based configurations"""
        manager = LoRAConfigManager()
        
        # Small model
        config = manager.get_config_for_model_size(3e9)
        assert config.rank == 8
        
        # Large model
        config = manager.get_config_for_model_size(70e9)
        assert config.rank == 32
        
        # XLarge model
        config = manager.get_config_for_model_size(150e9)
        assert config.rank == 64
    
    def test_hardware_optimization(self):
        """Test hardware-constrained optimization"""
        manager = LoRAConfigManager()
        
        # Low memory constraint
        config = manager.optimize_config_for_hardware(8.0, 13e9)
        
        # Should reduce rank and target modules for memory constraints
        assert config.rank <= 16
        assert len(config.target_modules) <= 2
    
    def test_custom_config_creation(self):
        """Test custom configuration creation"""
        manager = LoRAConfigManager()
        
        config = manager.create_custom_config(
            task="code_generation",
            model_size=13e9,
            memory_gb=24,
            performance_priority="quality"
        )
        
        assert isinstance(config, LoRAConfig)
        assert config.rank > 0
        assert len(config.target_modules) > 0

class TestQLoRASetup:
    """Test QLoRA setup and quantization"""
    
    def test_qlora_initialization(self):
        """Test QLoRA setup initialization"""
        qlora = QLoRASetup()
        
        assert qlora.quantization_config.load_in_4bit
        assert qlora.quantization_config.bnb_4bit_quant_type == "nf4"
        assert qlora.quantization_config.bnb_4bit_use_double_quant
    
    def test_memory_savings_calculation(self):
        """Test memory savings calculation"""
        qlora = QLoRASetup()
        
        analysis = qlora.calculate_memory_savings(
            base_model_params=7e9,
            rank=32,
            num_target_modules=4
        )
        
        assert "full_finetuning_gb" in analysis
        assert "qlora_total_gb" in analysis
        assert "memory_savings_gb" in analysis
        assert "memory_reduction_percent" in analysis
        
        # QLoRA should use significantly less memory
        assert analysis["qlora_total_gb"] < analysis["full_finetuning_gb"]
        assert analysis["memory_reduction_percent"] > 50  # At least 50% reduction
    
    def test_training_time_estimation(self):
        """Test training time estimation"""
        qlora = QLoRASetup()
        
        time_analysis = qlora.estimate_training_time(
            model_params=7e9,
            dataset_size=10000,
            batch_size=4,
            num_epochs=3
        )
        
        assert "full_finetuning_hours" in time_analysis
        assert "qlora_hours" in time_analysis
        assert "speedup_factor" in time_analysis
        
        # QLoRA should be faster
        assert time_analysis["qlora_hours"] < time_analysis["full_finetuning_hours"]
        assert time_analysis["speedup_factor"] > 1.0

class TestMultiTaskLoRA:
    """Test multi-task LoRA management"""
    
    def test_manager_initialization(self):
        """Test multi-task manager initialization"""
        base_model = nn.Linear(768, 768)
        tasks = ["task1", "task2", "task3"]
        manager = MultiTaskLoRAManager(base_model, tasks)
        
        assert manager.base_model == base_model
        assert manager.tasks == tasks
        assert len(manager.adapters) == 0
        assert manager.current_task is None
    
    def test_adapter_creation(self):
        """Test task adapter creation"""
        base_model = nn.Linear(768, 768)
        manager = MultiTaskLoRAManager(base_model, ["task1"])
        
        config = LoRAConfig(rank=16, alpha=32)
        adapter_info = manager.create_task_adapter("task1", config)
        
        assert "task1" in manager.adapters
        assert adapter_info["task"] == "task1"
        assert adapter_info["rank"] == 16
        assert adapter_info["parameters"] > 0
    
    def test_task_switching(self):
        """Test task switching functionality"""
        base_model = nn.Linear(768, 768)
        manager = MultiTaskLoRAManager(base_model, ["task1", "task2"])
        
        # Create adapters
        config = LoRAConfig(rank=16, alpha=32)
        manager.create_task_adapter("task1", config)
        manager.create_task_adapter("task2", config)
        
        # Test switching
        result = manager.switch_task("task1")
        assert manager.current_task == "task1"
        assert result["new_task"] == "task1"
        
        # Switch to another task
        result = manager.switch_task("task2")
        assert manager.current_task == "task2"
    
    def test_adapter_merging(self):
        """Test adapter merging functionality"""
        base_model = nn.Linear(768, 768)
        manager = MultiTaskLoRAManager(base_model, ["task1", "task2"])
        
        # Create adapters
        config = LoRAConfig(rank=16, alpha=32)
        manager.create_task_adapter("task1", config)
        manager.create_task_adapter("task2", config)
        
        # Test merging
        merged = manager.merge_adapters(["task1", "task2"], [0.6, 0.4])
        
        assert merged["type"] == "merged"
        assert merged["source_tasks"] == ["task1", "task2"]
        assert merged["weights"] == [0.6, 0.4]
        assert merged["parameters"] > 0
    
    def test_adapter_statistics(self):
        """Test adapter statistics generation"""
        base_model = nn.Linear(768, 768)
        manager = MultiTaskLoRAManager(base_model, ["task1", "task2"])
        
        # Create adapters
        config = LoRAConfig(rank=16, alpha=32)
        manager.create_task_adapter("task1", config)
        manager.create_task_adapter("task2", config)
        
        stats = manager.get_adapter_statistics()
        
        assert stats["total_adapters"] == 2
        assert stats["total_parameters"] > 0
        assert stats["average_rank"] == 16
        assert "parameter_distribution" in stats
    
    def test_save_load_adapter(self):
        """Test adapter save/load functionality"""
        base_model = nn.Linear(768, 768)
        manager = MultiTaskLoRAManager(base_model, ["task1"])
        
        # Create adapter
        config = LoRAConfig(rank=16, alpha=32)
        manager.create_task_adapter("task1", config)
        
        # Test save
        save_result = manager.save_adapter("task1", "/tmp/test_adapter.bin")
        assert save_result["task"] == "task1"
        assert save_result["size_mb"] > 0
        
        # Test load
        load_result = manager.load_adapter("loaded_task", "/tmp/test_adapter.bin")
        assert load_result["task"] == "loaded_task"
        assert "loaded_task" in manager.adapters

class TestMemoryOptimization:
    """Test memory optimization functionality"""
    
    def test_memory_analyzer_initialization(self):
        """Test memory optimizer initialization"""
        optimizer = MemoryOptimizer()
        
        assert len(optimizer.optimization_strategies) > 0
        assert "gradient_checkpointing" in optimizer.optimization_strategies
        assert "mixed_precision" in optimizer.optimization_strategies
    
    def test_memory_usage_analysis(self):
        """Test memory usage analysis"""
        optimizer = MemoryOptimizer()
        
        analysis = optimizer.analyze_memory_usage(
            model_params=7e9,
            batch_size=16,
            sequence_length=512,
            rank=32,
            num_adapters=4
        )
        
        required_keys = [
            "base_model_gb", "lora_parameters_gb", "activations_gb",
            "gradients_gb", "optimizer_states_gb", "total_gb"
        ]
        
        for key in required_keys:
            assert key in analysis
            assert analysis[key] > 0
        
        # Total should be sum of components
        components_sum = (
            analysis["base_model_gb"] + analysis["lora_parameters_gb"] +
            analysis["activations_gb"] + analysis["gradients_gb"] +
            analysis["optimizer_states_gb"] + analysis["kv_cache_gb"]
        )
        assert abs(analysis["total_gb"] - components_sum) < 0.1
    
    def test_batch_size_optimization(self):
        """Test batch size optimization"""
        optimizer = MemoryOptimizer()
        
        result = optimizer.optimize_batch_size(
            model_params=7e9,
            max_memory_gb=24
        )
        
        assert "optimal_batch_size" in result
        assert "gradient_accumulation_steps" in result
        assert "effective_batch_size" in result
        
        assert result["optimal_batch_size"] > 0
        assert result["gradient_accumulation_steps"] > 0
        assert result["effective_batch_size"] > 0
    
    def test_optimization_strategies(self):
        """Test optimization strategy application"""
        optimizer = MemoryOptimizer()
        
        result = optimizer.apply_optimization_strategy(
            base_memory_gb=50.0,
            strategies=["gradient_checkpointing", "mixed_precision"]
        )
        
        assert "original_memory_gb" in result
        assert "optimized_memory_gb" in result
        assert "memory_savings_gb" in result
        assert "memory_reduction_percent" in result
        
        # Should reduce memory usage
        assert result["optimized_memory_gb"] < result["original_memory_gb"]
        assert result["memory_reduction_percent"] > 0

class TestPerformanceProfiler:
    """Test performance profiling functionality"""
    
    def test_profiler_initialization(self):
        """Test performance profiler initialization"""
        profiler = PerformanceProfiler()
        
        assert len(profiler.evaluation_metrics) > 0
        assert len(profiler.metrics_history) == 0
    
    def test_training_step_profiling(self):
        """Test training step profiling"""
        profiler = PerformanceProfiler()
        
        metrics = profiler.profile_training_step(
            model_params=7e9,
            batch_size=16,
            sequence_length=512,
            rank=32
        )
        
        required_metrics = [
            "total_time", "forward_time", "backward_time", "optimizer_time",
            "tokens_per_second", "samples_per_second"
        ]
        
        for metric in required_metrics:
            assert metric in metrics
            assert metrics[metric] > 0
        
        # Check that metrics are added to history
        assert len(profiler.metrics_history) == 1
    
    def test_lora_vs_full_finetuning_comparison(self):
        """Test LoRA vs full fine-tuning comparison"""
        profiler = PerformanceProfiler()
        
        comparison = profiler.compare_lora_vs_full_finetuning(
            model_params=7e9,
            rank=32
        )
        
        assert "lora" in comparison
        assert "full_finetuning" in comparison
        assert "improvements" in comparison
        
        # LoRA should be more efficient
        lora_time = comparison["lora"]["total_time"]
        full_ft_time = comparison["full_finetuning"]["total_time"]
        assert lora_time < full_ft_time
        
        # Check improvement metrics
        improvements = comparison["improvements"]
        assert improvements["speed_improvement"] > 1.0
        assert improvements["parameter_reduction"] > 0.9  # >90% reduction
    
    def test_performance_report_generation(self):
        """Test performance report generation"""
        profiler = PerformanceProfiler()
        
        # Generate some metrics
        for _ in range(3):
            profiler.profile_training_step(7e9, 16, 512, 32)
        
        report = profiler.generate_performance_report()
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Performance Analysis Report" in report
        assert "Training Steps Analyzed: 3" in report

class TestIntegration:
    """Integration tests for complete workflow"""
    
    def test_end_to_end_workflow(self):
        """Test complete LoRA workflow"""
        # 1. Create configuration
        config_manager = LoRAConfigManager()
        config = config_manager.get_config_for_task("instruction_following")
        
        # 2. Setup QLoRA
        qlora_setup = QLoRASetup()
        memory_analysis = qlora_setup.calculate_memory_savings(7e9, config.rank, 4)
        
        # 3. Create multi-task manager
        base_model = nn.Linear(768, 768)
        manager = MultiTaskLoRAManager(base_model, ["task1"])
        adapter_info = manager.create_task_adapter("task1", config)
        
        # 4. Optimize memory
        optimizer = MemoryOptimizer()
        batch_optimization = optimizer.optimize_batch_size(7e9, 24)
        
        # 5. Profile performance
        profiler = PerformanceProfiler()
        metrics = profiler.profile_training_step(7e9, 16, 512, config.rank)
        
        # Verify all components work together
        assert config.rank > 0
        assert memory_analysis["memory_reduction_percent"] > 0
        assert adapter_info["parameters"] > 0
        assert batch_optimization["optimal_batch_size"] > 0
        assert metrics["total_time"] > 0
    
    def test_configuration_consistency(self):
        """Test configuration consistency across components"""
        config_manager = LoRAConfigManager()
        
        # Test that configurations are consistent
        for task in ["instruction_following", "code_generation", "summarization"]:
            config = config_manager.get_config_for_task(task)
            
            assert config.rank > 0
            assert config.alpha > 0
            assert len(config.target_modules) > 0
            assert config.alpha >= config.rank  # Common best practice

def run_tests():
    """Run all tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])

if __name__ == "__main__":
    run_tests()