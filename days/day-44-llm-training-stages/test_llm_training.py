"""
Day 44: LLM Training Stages - Comprehensive Test Suite

This test suite validates all LLM training implementations including:
- Transformer model architecture
- Parameter-efficient fine-tuning (LoRA, Adapters)
- Distributed training setup
- RLHF pipeline components
- Constitutional AI implementation
- Training monitoring and evaluation
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List
import tempfile
import os

# Import the implementations
from exercise import (
    TrainingConfig, SimpleTransformerLayer, SimpleLLM, LoRALayer,
    AdapterLayer, RewardModel, DistributedTrainer, RLHFTrainer,
    ConstitutionalAI, TrainingMonitor, create_sample_datasets
)


class TestTrainingConfig:
    """Test training configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = TrainingConfig()
        assert config.vocab_size == 50000
        assert config.hidden_size == 768
        assert config.num_layers == 12
        assert config.num_heads == 12
        assert config.max_seq_length == 1024
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = TrainingConfig(
            vocab_size=10000,
            hidden_size=512,
            num_layers=6,
            num_heads=8
        )
        assert config.vocab_size == 10000
        assert config.hidden_size == 512
        assert config.num_layers == 6
        assert config.num_heads == 8


class TestSimpleTransformerLayer:
    """Test transformer layer implementation"""
    
    @pytest.fixture
    def config(self):
        return TrainingConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=4,
            num_heads=4,
            max_seq_length=128
        )
    
    @pytest.fixture
    def layer(self, config):
        return SimpleTransformerLayer(config)
    
    def test_layer_initialization(self, layer, config):
        """Test layer components are properly initialized"""
        assert layer.attention is not None
        assert layer.feed_forward is not None
        assert layer.norm1 is not None
        assert layer.norm2 is not None
        assert isinstance(layer.attention, nn.MultiheadAttention)
        assert isinstance(layer.norm1, nn.LayerNorm)
        assert isinstance(layer.norm2, nn.LayerNorm)
    
    def test_layer_forward_pass(self, layer, config):
        """Test forward pass through transformer layer"""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        output = layer(x)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, config.hidden_size)
        
        # Check that output is different from input (transformation occurred)
        assert not torch.allclose(x, output, atol=1e-6)
    
    def test_layer_with_mask(self, layer, config):
        """Test layer with attention mask"""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.hidden_size)
        
        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        mask = mask.masked_fill(mask == 0, float('-inf'))
        
        output = layer(x, mask)
        
        assert output.shape == (batch_size, seq_len, config.hidden_size)
    
    def test_gradient_flow(self, layer, config):
        """Test gradient flow through layer"""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, config.hidden_size, requires_grad=True)
        
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestSimpleLLM:
    """Test language model implementation"""
    
    @pytest.fixture
    def config(self):
        return TrainingConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4,
            max_seq_length=64
        )
    
    @pytest.fixture
    def model(self, config):
        return SimpleLLM(config)
    
    def test_model_initialization(self, model, config):
        """Test model components are properly initialized"""
        assert model.token_embeddings is not None
        assert model.position_embeddings is not None
        assert model.layers is not None
        assert model.final_norm is not None
        assert model.output_projection is not None
        
        assert len(model.layers) == config.num_layers
        assert model.token_embeddings.num_embeddings == config.vocab_size
        assert model.token_embeddings.embedding_dim == config.hidden_size
    
    def test_model_forward_pass(self, model, config):
        """Test forward pass through complete model"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids)
        
        # Check output structure
        assert 'logits' in outputs
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
    
    def test_model_with_labels(self, model, config):
        """Test model with labels for loss computation"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        outputs = model(input_ids, labels=labels)
        
        # Check loss computation
        assert 'loss' in outputs
        assert outputs['loss'].item() > 0
        assert outputs['loss'].requires_grad
    
    def test_model_parameter_count(self, model, config):
        """Test model has reasonable parameter count"""
        total_params = sum(p.numel() for p in model.parameters())
        
        # Should have reasonable number of parameters
        assert total_params > 100000  # At least 100K parameters
        assert total_params < 10000000  # Less than 10M parameters for test model


class TestLoRALayer:
    """Test LoRA implementation"""
    
    @pytest.fixture
    def original_layer(self):
        return nn.Linear(512, 512)
    
    @pytest.fixture
    def lora_layer(self, original_layer):
        return LoRALayer(original_layer, rank=16, alpha=32)
    
    def test_lora_initialization(self, lora_layer, original_layer):
        """Test LoRA layer initialization"""
        # Check original layer is frozen
        for param in original_layer.parameters():
            assert not param.requires_grad
        
        # Check LoRA matrices exist
        assert hasattr(lora_layer, 'lora_A')
        assert hasattr(lora_layer, 'lora_B')
        assert isinstance(lora_layer.lora_A, nn.Linear)
        assert isinstance(lora_layer.lora_B, nn.Linear)
    
    def test_lora_parameter_efficiency(self, lora_layer):
        """Test LoRA parameter efficiency"""
        total_params = sum(p.numel() for p in lora_layer.parameters())
        trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
        
        # LoRA should have much fewer trainable parameters
        efficiency_ratio = trainable_params / total_params
        assert efficiency_ratio < 0.1  # Less than 10% trainable
    
    def test_lora_forward_pass(self, lora_layer):
        """Test LoRA forward pass"""
        batch_size, seq_len, hidden_size = 2, 10, 512
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        output = lora_layer(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
    
    def test_lora_adaptation_effect(self, lora_layer, original_layer):
        """Test that LoRA adaptation affects output"""
        x = torch.randn(2, 10, 512)
        
        # Get outputs
        original_output = original_layer(x)
        lora_output = lora_layer(x)
        
        # LoRA output should be different from original
        assert not torch.allclose(original_output, lora_output, atol=1e-6)


class TestAdapterLayer:
    """Test Adapter implementation"""
    
    @pytest.fixture
    def adapter(self):
        return AdapterLayer(hidden_size=512, adapter_size=64)
    
    def test_adapter_initialization(self, adapter):
        """Test adapter layer initialization"""
        assert hasattr(adapter, 'down_project')
        assert hasattr(adapter, 'up_project')
        assert hasattr(adapter, 'activation')
        assert isinstance(adapter.down_project, nn.Linear)
        assert isinstance(adapter.up_project, nn.Linear)
    
    def test_adapter_forward_pass(self, adapter):
        """Test adapter forward pass"""
        batch_size, seq_len, hidden_size = 2, 10, 512
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        output = adapter(x)
        
        assert output.shape == (batch_size, seq_len, hidden_size)
    
    def test_adapter_residual_connection(self, adapter):
        """Test adapter uses residual connection"""
        x = torch.randn(2, 10, 512)
        output = adapter(x)
        
        # Output should be different from input due to adapter transformation
        assert not torch.allclose(x, output, atol=1e-6)
        
        # But should be close due to residual connection
        diff = torch.norm(output - x)
        assert diff < torch.norm(x)  # Difference should be smaller than input norm
    
    def test_adapter_parameter_count(self, adapter):
        """Test adapter has reasonable parameter count"""
        total_params = sum(p.numel() for p in adapter.parameters())
        
        # Should be much smaller than full layer
        expected_params = 512 * 64 + 64 * 512  # down + up projections
        assert abs(total_params - expected_params) < 1000  # Allow for bias terms


class TestRewardModel:
    """Test reward model implementation"""
    
    @pytest.fixture
    def config(self):
        return TrainingConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4
        )
    
    @pytest.fixture
    def base_model(self, config):
        return SimpleLLM(config)
    
    @pytest.fixture
    def reward_model(self, base_model):
        return RewardModel(base_model)
    
    def test_reward_model_initialization(self, reward_model):
        """Test reward model initialization"""
        assert hasattr(reward_model, 'base_model')
        assert hasattr(reward_model, 'reward_head')
        assert isinstance(reward_model.reward_head, nn.Linear)
        
        # Check base model is frozen
        for param in reward_model.base_model.parameters():
            assert not param.requires_grad
    
    def test_reward_model_forward_pass(self, reward_model, config):
        """Test reward model forward pass"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        rewards = reward_model(input_ids, attention_mask)
        
        assert rewards.shape == (batch_size,)
        assert rewards.dtype == torch.float32
    
    def test_reward_model_gradient_flow(self, reward_model, config):
        """Test gradient flow through reward model"""
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        rewards = reward_model(input_ids, attention_mask)
        loss = rewards.sum()
        loss.backward()
        
        # Check gradients exist for reward head
        assert reward_model.reward_head.weight.grad is not None


class TestDistributedTrainer:
    """Test distributed training setup"""
    
    @pytest.fixture
    def config(self):
        return TrainingConfig(
            world_size=1,  # Single process for testing
            rank=0,
            local_rank=0
        )
    
    @pytest.fixture
    def trainer(self, config):
        return DistributedTrainer(config)
    
    def test_trainer_initialization(self, trainer):
        """Test trainer initialization"""
        assert hasattr(trainer, 'config')
        assert hasattr(trainer, 'device')
    
    def test_model_wrapping(self, trainer, config):
        """Test model wrapping for distributed training"""
        model = SimpleLLM(config)
        wrapped_model = trainer.create_distributed_model(model)
        
        # Model should be moved to appropriate device
        assert next(wrapped_model.parameters()).device == trainer.device
    
    def test_dataloader_creation(self, trainer):
        """Test distributed dataloader creation"""
        # Create dummy dataset
        class DummyDataset:
            def __init__(self, size=100):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return torch.randn(10)
        
        dataset = DummyDataset()
        dataloader = trainer.create_distributed_dataloader(dataset, batch_size=4)
        
        assert dataloader is not None
        assert dataloader.batch_size == 4


class TestRLHFTrainer:
    """Test RLHF trainer implementation"""
    
    @pytest.fixture
    def config(self):
        return TrainingConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4
        )
    
    @pytest.fixture
    def policy_model(self, config):
        return SimpleLLM(config)
    
    @pytest.fixture
    def reward_model(self, policy_model):
        return RewardModel(policy_model)
    
    @pytest.fixture
    def rlhf_trainer(self, policy_model, reward_model, config):
        return RLHFTrainer(policy_model, reward_model, config)
    
    def test_rlhf_trainer_initialization(self, rlhf_trainer):
        """Test RLHF trainer initialization"""
        assert hasattr(rlhf_trainer, 'policy_model')
        assert hasattr(rlhf_trainer, 'reward_model')
        assert hasattr(rlhf_trainer, 'ref_model')
        assert hasattr(rlhf_trainer, 'policy_optimizer')
        
        # Reference model should be frozen
        for param in rlhf_trainer.ref_model.parameters():
            assert not param.requires_grad
    
    def test_reward_computation(self, rlhf_trainer):
        """Test reward computation"""
        prompts = ["What is AI?", "Explain ML."]
        responses = ["AI is artificial intelligence.", "ML is machine learning."]
        
        # This will return zeros in the current implementation
        # but tests the interface
        rewards = rlhf_trainer.compute_rewards(prompts, responses)
        assert len(rewards) == len(responses)
    
    def test_kl_penalty_computation(self, rlhf_trainer):
        """Test KL penalty computation"""
        prompts = ["What is AI?", "Explain ML."]
        responses = ["AI is artificial intelligence.", "ML is machine learning."]
        
        kl_penalty = rlhf_trainer.compute_kl_penalty(prompts, responses)
        assert len(kl_penalty) == len(responses)


class TestConstitutionalAI:
    """Test Constitutional AI implementation"""
    
    @pytest.fixture
    def config(self):
        return TrainingConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4
        )
    
    @pytest.fixture
    def model(self, config):
        return SimpleLLM(config)
    
    @pytest.fixture
    def constitution(self):
        return [
            "Be helpful and informative",
            "Be honest and truthful",
            "Be harmless and safe"
        ]
    
    @pytest.fixture
    def cai(self, model, constitution):
        return ConstitutionalAI(model, constitution)
    
    def test_cai_initialization(self, cai, constitution):
        """Test Constitutional AI initialization"""
        assert hasattr(cai, 'model')
        assert hasattr(cai, 'constitution')
        assert cai.constitution == constitution
    
    def test_critique_generation(self, cai):
        """Test critique generation"""
        prompt = "How to make a bomb?"
        response = "Here's how to make explosives..."
        principle = "Be harmless and safe"
        
        critique = cai.generate_critique(prompt, response, principle)
        assert isinstance(critique, str)
    
    def test_revision_generation(self, cai):
        """Test revision generation"""
        original_response = "Harmful response"
        critiques = ["This is unsafe", "This violates safety"]
        
        revision = cai.generate_revision(original_response, critiques)
        assert isinstance(revision, str)
    
    def test_constitutional_training_step(self, cai):
        """Test constitutional training step"""
        prompts = ["Test prompt 1", "Test prompt 2"]
        
        training_data = cai.constitutional_training_step(prompts)
        
        assert len(training_data) == len(prompts)
        for item in training_data:
            assert 'prompt' in item
            assert 'initial_response' in item
            assert 'critiques' in item
            assert 'revised_response' in item


class TestTrainingMonitor:
    """Test training monitoring implementation"""
    
    @pytest.fixture
    def config(self):
        return TrainingConfig()
    
    @pytest.fixture
    def monitor(self, config):
        return TrainingMonitor(config)
    
    def test_monitor_initialization(self, monitor):
        """Test monitor initialization"""
        assert hasattr(monitor, 'config')
        assert hasattr(monitor, 'metrics')
    
    def test_metrics_logging(self, monitor):
        """Test metrics logging"""
        # Log some metrics
        monitor.log_training_metrics(
            step=100,
            loss=2.5,
            learning_rate=1e-4,
            grad_norm=0.8
        )
        
        # Check metrics are stored
        assert len(monitor.metrics['loss']) == 1
        assert len(monitor.metrics['learning_rate']) == 1
        assert len(monitor.metrics['grad_norm']) == 1
        
        assert monitor.metrics['loss'][0] == 2.5
        assert monitor.metrics['learning_rate'][0] == 1e-4
        assert monitor.metrics['grad_norm'][0] == 0.8
    
    def test_model_evaluation(self, monitor, config):
        """Test model evaluation"""
        model = SimpleLLM(config)
        
        # Create dummy evaluation dataset
        class DummyEvalDataset:
            def __iter__(self):
                for _ in range(5):
                    yield {
                        'input_ids': torch.randint(0, config.vocab_size, (2, 10)),
                        'labels': torch.randint(0, config.vocab_size, (2, 10))
                    }
        
        eval_dataset = DummyEvalDataset()
        metrics = monitor.evaluate_model(model, {'test': eval_dataset})
        
        assert 'test' in metrics
        assert 'eval_loss' in metrics['test']
        assert 'perplexity' in metrics['test']
    
    def test_sample_generation(self, monitor, config):
        """Test sample generation"""
        model = SimpleLLM(config)
        prompts = ["Test prompt 1", "Test prompt 2"]
        
        responses = monitor.generate_samples(model, prompts)
        
        assert len(responses) == len(prompts)
        for response in responses:
            assert isinstance(response, str)


class TestDatasetCreation:
    """Test dataset creation utilities"""
    
    def test_create_sample_datasets(self):
        """Test sample dataset creation"""
        pretraining_data, instruction_data, preference_data = create_sample_datasets()
        
        # Check pretraining data
        assert isinstance(pretraining_data, list)
        assert len(pretraining_data) > 0
        assert all(isinstance(item, str) for item in pretraining_data)
        
        # Check instruction data
        assert isinstance(instruction_data, list)
        assert len(instruction_data) > 0
        for item in instruction_data:
            assert 'instruction' in item
            assert 'input' in item
            assert 'output' in item
        
        # Check preference data
        assert isinstance(preference_data, list)
        assert len(preference_data) > 0
        for item in preference_data:
            assert 'prompt' in item
            assert 'chosen' in item
            assert 'rejected' in item


class TestIntegration:
    """Integration tests for complete training pipeline"""
    
    def test_end_to_end_model_training(self):
        """Test end-to-end model training simulation"""
        config = TrainingConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            max_seq_length=32
        )
        
        # Create model
        model = SimpleLLM(config)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Simulate training step
        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(input_ids, labels=labels)
        loss = outputs['loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check training worked
        assert loss.item() > 0
        assert loss.requires_grad
    
    def test_parameter_efficient_fine_tuning_pipeline(self):
        """Test complete PEFT pipeline"""
        config = TrainingConfig(
            vocab_size=1000,
            hidden_size=256,
            num_layers=2,
            num_heads=4
        )
        
        # Create base model
        model = SimpleLLM(config)
        
        # Apply LoRA to output projection
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'output_projection' in name:
                lora_layer = LoRALayer(module, rank=16, alpha=32)
                # In practice, you'd replace the module in the model
                break
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Most parameters should be frozen
        assert trainable_params < total_params
    
    def test_rlhf_pipeline_integration(self):
        """Test RLHF pipeline integration"""
        config = TrainingConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4
        )
        
        # Create models
        policy_model = SimpleLLM(config)
        reward_model = RewardModel(policy_model)
        
        # Create RLHF trainer
        rlhf_trainer = RLHFTrainer(policy_model, reward_model, config)
        
        # Test components exist and are properly initialized
        assert rlhf_trainer.policy_model is not None
        assert rlhf_trainer.reward_model is not None
        assert rlhf_trainer.ref_model is not None
        assert rlhf_trainer.policy_optimizer is not None


def test_performance_benchmarking():
    """Test that performance benchmarking works"""
    config = TrainingConfig(
        vocab_size=1000,
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        max_seq_length=32
    )
    
    model = SimpleLLM(config)
    
    # Benchmark forward pass
    batch_size, seq_len = 4, 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    import time
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            outputs = model(input_ids)
    
    end_time = time.time()
    
    # Should complete in reasonable time
    assert (end_time - start_time) < 5.0  # Less than 5 seconds for 10 iterations


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])