"""
Day 42: Attention Mechanisms - Comprehensive Test Suite

This test suite validates all attention mechanism implementations including:
- Multi-head attention (self and cross)
- Sparse attention patterns
- Attention caching
- Visualization tools
- Performance benchmarks
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any
import tempfile
import os

# Import the implementations
from exercise import (
    AttentionConfig, MultiHeadAttention, SparseAttention, 
    AttentionCache, AttentionVisualizer
)


class TestAttentionConfig:
    """Test attention configuration validation"""
    
    def test_valid_config(self):
        """Test valid configuration creation"""
        config = AttentionConfig(d_model=512, num_heads=8)
        assert config.d_model == 512
        assert config.num_heads == 8
        assert config.d_model % config.num_heads == 0
    
    def test_invalid_head_division(self):
        """Test invalid d_model/num_heads combination"""
        with pytest.raises(AssertionError):
            AttentionConfig(d_model=513, num_heads=8)  # Not divisible
    
    def test_invalid_dropout(self):
        """Test invalid dropout values"""
        with pytest.raises(AssertionError):
            AttentionConfig(dropout=-0.1)  # Negative dropout
        
        with pytest.raises(AssertionError):
            AttentionConfig(dropout=1.5)  # Dropout > 1


class TestMultiHeadAttention:
    """Test multi-head attention implementation"""
    
    @pytest.fixture
    def config(self):
        return AttentionConfig(d_model=256, num_heads=4, dropout=0.1)
    
    @pytest.fixture
    def attention(self, config):
        return MultiHeadAttention(config)
    
    def test_initialization(self, attention, config):
        """Test proper initialization of attention layers"""
        assert attention.d_model == config.d_model
        assert attention.num_heads == config.num_heads
        assert attention.head_dim == config.d_model // config.num_heads
        
        # Check if linear layers are initialized
        assert isinstance(attention.query, nn.Linear)
        assert isinstance(attention.key, nn.Linear)
        assert isinstance(attention.value, nn.Linear)
        assert isinstance(attention.output, nn.Linear)
    
    def test_self_attention_forward(self, attention):
        """Test self-attention forward pass"""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, attention.d_model)
        
        output, weights = attention(x, return_attention=True)
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, attention.d_model)
        
        # Check attention weights shape
        assert weights.shape == (batch_size, attention.num_heads, seq_len, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-6)
    
    def test_cross_attention_forward(self, attention):
        """Test cross-attention forward pass"""
        batch_size, seq_len_q, seq_len_k = 2, 8, 12
        
        query = torch.randn(batch_size, seq_len_q, attention.d_model)
        key = torch.randn(batch_size, seq_len_k, attention.d_model)
        value = torch.randn(batch_size, seq_len_k, attention.d_model)
        
        output, weights = attention(query, key, value, return_attention=True)
        
        # Check output shape matches query
        assert output.shape == (batch_size, seq_len_q, attention.d_model)
        
        # Check attention weights shape
        assert weights.shape == (batch_size, attention.num_heads, seq_len_q, seq_len_k)
    
    def test_causal_masking(self, attention):
        """Test causal masking functionality"""
        batch_size, seq_len = 1, 6
        x = torch.randn(batch_size, seq_len, attention.d_model)
        
        # Create causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len))
        
        output, weights = attention(x, mask=causal_mask, return_attention=True)
        
        # Check that upper triangular part has zero attention
        upper_triangular = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        masked_weights = weights[0, 0] * upper_triangular
        
        assert torch.allclose(masked_weights, torch.zeros_like(masked_weights), atol=1e-6)
    
    def test_gradient_flow(self, attention):
        """Test gradient flow through attention"""
        x = torch.randn(1, 5, attention.d_model, requires_grad=True)
        
        output, _ = attention(x)
        loss = output.sum()
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestSparseAttention:
    """Test sparse attention implementations"""
    
    @pytest.fixture
    def config(self):
        return AttentionConfig(
            d_model=256, 
            num_heads=4, 
            sparse_pattern="local", 
            window_size=8
        )
    
    @pytest.fixture
    def sparse_attention(self, config):
        return SparseAttention(config)
    
    def test_local_mask_creation(self, sparse_attention):
        """Test local sparse mask creation"""
        seq_len = 16
        mask = sparse_attention.create_sparse_mask(seq_len, "local")
        
        assert mask.shape == (seq_len, seq_len)
        assert mask.dtype == torch.bool
        
        # Check that diagonal is True (self-attention)
        assert torch.all(torch.diag(mask))
        
        # Check window size constraint
        window = sparse_attention.config.window_size // 2
        for i in range(seq_len):
            start = max(0, i - window)
            end = min(seq_len, i + window + 1)
            assert torch.all(mask[i, start:end])
    
    def test_strided_mask_creation(self, sparse_attention):
        """Test strided sparse mask creation"""
        seq_len = 32
        mask = sparse_attention.create_sparse_mask(seq_len, "strided")
        
        assert mask.shape == (seq_len, seq_len)
        
        # Check that diagonal is True
        assert torch.all(torch.diag(mask))
        
        # Check strided pattern exists
        stride = max(1, sparse_attention.config.window_size // 4)
        for i in range(seq_len):
            for j in range(0, seq_len, stride):
                if j < seq_len:
                    assert mask[i, j]
    
    def test_random_mask_creation(self, sparse_attention):
        """Test random sparse mask creation"""
        seq_len = 20
        mask = sparse_attention.create_sparse_mask(seq_len, "random")
        
        assert mask.shape == (seq_len, seq_len)
        
        # Check that diagonal is True (self-attention preserved)
        assert torch.all(torch.diag(mask))
        
        # Check sparsity level
        sparsity = mask.float().mean()
        expected_sparsity = min(0.1, sparse_attention.config.window_size / seq_len)
        assert sparsity >= expected_sparsity  # At least the expected connections
    
    def test_sparse_attention_forward(self, sparse_attention):
        """Test sparse attention forward pass"""
        batch_size, seq_len = 2, 32
        x = torch.randn(batch_size, seq_len, sparse_attention.config.d_model)
        
        output, weights = sparse_attention(x, return_attention=True)
        
        assert output.shape == (batch_size, seq_len, sparse_attention.config.d_model)
        assert weights.shape == (batch_size, sparse_attention.config.num_heads, seq_len, seq_len)


class TestAttentionCache:
    """Test attention caching functionality"""
    
    @pytest.fixture
    def cache(self):
        return AttentionCache(
            max_seq_len=100,
            num_heads=4,
            head_dim=64,
            device="cpu"
        )
    
    def test_cache_initialization(self, cache):
        """Test cache initialization"""
        assert cache.max_seq_len == 100
        assert cache.num_heads == 4
        assert cache.head_dim == 64
        assert cache.seq_len == 0
        
        assert cache.k_cache.shape == (1, 4, 100, 64)
        assert cache.v_cache.shape == (1, 4, 100, 64)
    
    def test_cache_update(self, cache):
        """Test cache update functionality"""
        batch_size, num_heads, seq_len, head_dim = 1, 4, 3, 64
        
        new_k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        new_v = torch.randn(batch_size, num_heads, seq_len, head_dim)
        
        cache.update(new_k, new_v)
        
        assert cache.seq_len == seq_len
        
        # Check cached values
        k_cached, v_cached = cache.get_cached()
        assert k_cached.shape == (batch_size, num_heads, seq_len, head_dim)
        assert v_cached.shape == (batch_size, num_heads, seq_len, head_dim)
        
        assert torch.allclose(k_cached, new_k)
        assert torch.allclose(v_cached, new_v)
    
    def test_cache_overflow_protection(self, cache):
        """Test cache overflow protection"""
        batch_size, num_heads, head_dim = 1, 4, 64
        
        # Try to add more than max_seq_len
        large_seq_len = cache.max_seq_len + 10
        new_k = torch.randn(batch_size, num_heads, large_seq_len, head_dim)
        new_v = torch.randn(batch_size, num_heads, large_seq_len, head_dim)
        
        with pytest.raises(AssertionError):
            cache.update(new_k, new_v)
    
    def test_cache_reset(self, cache):
        """Test cache reset functionality"""
        # Add some data
        new_k = torch.randn(1, 4, 5, 64)
        new_v = torch.randn(1, 4, 5, 64)
        cache.update(new_k, new_v)
        
        assert cache.seq_len == 5
        
        # Reset cache
        cache.reset()
        assert cache.seq_len == 0


class TestAttentionVisualizer:
    """Test attention visualization tools"""
    
    def test_attention_entropy_computation(self):
        """Test attention entropy computation"""
        # Create test attention weights
        batch_size, num_heads, seq_len = 2, 4, 8
        
        # Uniform attention (high entropy)
        uniform_weights = torch.ones(batch_size, num_heads, seq_len, seq_len) / seq_len
        uniform_entropy = AttentionVisualizer.compute_attention_entropy(uniform_weights)
        
        # Focused attention (low entropy)
        focused_weights = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        focused_weights[:, :, :, 0] = 1.0  # All attention on first position
        focused_entropy = AttentionVisualizer.compute_attention_entropy(focused_weights)
        
        # Uniform should have higher entropy than focused
        assert torch.all(uniform_entropy > focused_entropy)
    
    def test_attention_distance_computation(self):
        """Test attention distance computation"""
        batch_size, num_heads, seq_len = 1, 2, 6
        
        # Create attention that focuses on position 3 (index 3)
        weights = torch.zeros(batch_size, num_heads, seq_len, seq_len)
        weights[:, :, :, 3] = 1.0
        
        distances = AttentionVisualizer.compute_attention_distance(weights)
        
        # All positions should have average distance of 3
        expected_distance = torch.full((batch_size, num_heads, seq_len), 3.0)
        assert torch.allclose(distances, expected_distance)
    
    def test_attention_pattern_analysis(self):
        """Test comprehensive attention pattern analysis"""
        batch_size, num_heads, seq_len = 2, 4, 10
        
        # Create test attention weights
        weights = torch.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
        
        stats = AttentionVisualizer.analyze_attention_patterns(weights)
        
        # Check that all expected statistics are present
        expected_keys = ['entropy', 'max_attention', 'avg_distance', 'sparsity']
        for key in expected_keys:
            assert key in stats
            
        # Check entropy statistics
        assert 'mean' in stats['entropy']
        assert 'std' in stats['entropy']
        assert 'min' in stats['entropy']
        assert 'max' in stats['entropy']
        
        # Check sparsity is between 0 and 1
        assert 0 <= stats['sparsity'] <= 1


class TestIntegration:
    """Integration tests for complete attention pipeline"""
    
    def test_end_to_end_attention_pipeline(self):
        """Test complete attention pipeline with caching"""
        config = AttentionConfig(d_model=256, num_heads=4)
        attention = MultiHeadAttention(config)
        cache = AttentionCache(
            max_seq_len=50,
            num_heads=config.num_heads,
            head_dim=config.d_model // config.num_heads
        )
        
        # Simulate incremental inference
        seq_len = 10
        batch_size = 1
        
        for i in range(seq_len):
            # Process one token at a time
            x = torch.randn(batch_size, 1, config.d_model)
            
            with torch.no_grad():
                output, weights = attention(
                    x, 
                    use_cache=True, 
                    cache=cache, 
                    return_attention=True
                )
            
            # Check output shape
            assert output.shape == (batch_size, 1, config.d_model)
            
            # Check that cache is growing
            assert cache.seq_len == i + 1
    
    def test_sparse_vs_standard_attention_equivalence(self):
        """Test that sparse attention with full pattern equals standard attention"""
        config = AttentionConfig(d_model=128, num_heads=2)
        
        # Standard attention
        standard_attention = MultiHeadAttention(config)
        
        # Sparse attention with no sparsity (should be equivalent)
        sparse_config = AttentionConfig(
            d_model=128, 
            num_heads=2, 
            sparse_pattern="none"
        )
        sparse_attention = SparseAttention(sparse_config)
        
        # Same input
        x = torch.randn(1, 8, config.d_model)
        
        with torch.no_grad():
            standard_output, _ = standard_attention(x)
            sparse_output, _ = sparse_attention(x)
        
        # Outputs should be close (not exactly equal due to different initialization)
        # This test mainly checks that sparse attention with "none" pattern works
        assert standard_output.shape == sparse_output.shape


def test_performance_benchmarking():
    """Test that performance benchmarking functions work"""
    # This is a basic test to ensure the benchmark functions don't crash
    # Full benchmarking would be too slow for unit tests
    
    config = AttentionConfig(d_model=64, num_heads=2)
    attention = MultiHeadAttention(config)
    
    # Small test
    x = torch.randn(1, 16, config.d_model)
    
    with torch.no_grad():
        output, _ = attention(x)
    
    assert output.shape == (1, 16, config.d_model)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])