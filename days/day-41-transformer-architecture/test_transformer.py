#!/usr/bin/env python3
"""
Day 41: Transformer Architecture - Test Suite

Tests for transformer components implementation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from exercise import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    PositionalEncoding,
    TransformerEncoderLayer,
    SimpleTransformer
)

class TestScaledDotProductAttention:
    """Test scaled dot-product attention implementation"""
    
    def test_attention_output_shape(self):
        """Test attention output has correct shape"""
        d_k = 64
        batch_size, seq_len = 2, 10
        
        attention = ScaledDotProductAttention(d_k)
        
        Q = torch.randn(batch_size, seq_len, d_k)
        K = torch.randn(batch_size, seq_len, d_k)
        V = torch.randn(batch_size, seq_len, d_k)
        
        output, weights = attention(Q, K, V)
        
        assert output.shape == (batch_size, seq_len, d_k)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test attention weights sum to 1 across sequence dimension"""
        d_k = 32
        attention = ScaledDotProductAttention(d_k)
        
        Q = torch.randn(1, 5, d_k)
        K = torch.randn(1, 5, d_k)
        V = torch.randn(1, 5, d_k)
        
        _, weights = attention(Q, K, V)
        
        # Check weights sum to 1 along last dimension
        weight_sums = weights.sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6)

class TestMultiHeadAttention:
    """Test multi-head attention implementation"""
    
    def test_multihead_initialization(self):
        """Test multi-head attention initializes correctly"""
        d_model, num_heads = 512, 8
        mha = MultiHeadAttention(d_model, num_heads)
        
        assert mha.d_model == d_model
        assert mha.num_heads == num_heads
        assert mha.d_k == d_model // num_heads
        
        # Check linear layers are initialized
        assert isinstance(mha.w_q, nn.Linear)
        assert isinstance(mha.w_k, nn.Linear)
        assert isinstance(mha.w_v, nn.Linear)
        assert isinstance(mha.w_o, nn.Linear)
    
    def test_multihead_output_shape(self):
        """Test multi-head attention output shape"""
        d_model, num_heads = 256, 4
        batch_size, seq_len = 2, 8
        
        mha = MultiHeadAttention(d_model, num_heads)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, weights = mha(x, x, x)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert weights.shape == (batch_size, num_heads, seq_len, seq_len)

class TestPositionalEncoding:
    """Test positional encoding implementation"""
    
    def test_positional_encoding_shape(self):
        """Test positional encoding output shape"""
        d_model, max_seq_len = 128, 100
        pos_enc = PositionalEncoding(d_model, max_seq_len)
        
        seq_len, batch_size = 20, 3
        x = torch.randn(seq_len, batch_size, d_model)
        
        output = pos_enc(x)
        assert output.shape == (seq_len, batch_size, d_model)
    
    def test_positional_encoding_deterministic(self):
        """Test positional encoding is deterministic"""
        d_model = 64
        pos_enc = PositionalEncoding(d_model)
        
        x = torch.randn(10, 2, d_model)
        
        output1 = pos_enc(x)
        output2 = pos_enc(x)
        
        assert torch.allclose(output1, output2)

class TestTransformerEncoderLayer:
    """Test transformer encoder layer implementation"""
    
    def test_encoder_layer_initialization(self):
        """Test encoder layer initializes correctly"""
        d_model, num_heads, d_ff = 256, 8, 1024
        
        layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
        
        assert isinstance(layer.self_attn, MultiHeadAttention)
        assert isinstance(layer.feed_forward, nn.Sequential)
        assert isinstance(layer.norm1, nn.LayerNorm)
        assert isinstance(layer.norm2, nn.LayerNorm)
    
    def test_encoder_layer_output_shape(self):
        """Test encoder layer maintains input shape"""
        d_model, num_heads, d_ff = 128, 4, 512
        batch_size, seq_len = 2, 10
        
        layer = TransformerEncoderLayer(d_model, num_heads, d_ff)
        
        x = torch.randn(batch_size, seq_len, d_model)
        output = layer(x)
        
        assert output.shape == (batch_size, seq_len, d_model)

class TestSimpleTransformer:
    """Test complete transformer model implementation"""
    
    def test_transformer_initialization(self):
        """Test transformer model initializes correctly"""
        vocab_size = 1000
        d_model, num_heads, num_layers = 256, 8, 6
        d_ff, max_seq_len, num_classes = 1024, 100, 2
        
        model = SimpleTransformer(
            vocab_size, d_model, num_heads, num_layers,
            d_ff, max_seq_len, num_classes
        )
        
        assert isinstance(model.embedding, nn.Embedding)
        assert isinstance(model.pos_encoding, PositionalEncoding)
        assert len(model.encoder_layers) == num_layers
        assert isinstance(model.classifier, nn.Linear)
    
    def test_transformer_forward_pass(self):
        """Test transformer forward pass works"""
        vocab_size = 500
        d_model, num_heads, num_layers = 128, 4, 2
        d_ff, max_seq_len, num_classes = 512, 50, 3
        batch_size, seq_len = 2, 15
        
        model = SimpleTransformer(
            vocab_size, d_model, num_heads, num_layers,
            d_ff, max_seq_len, num_classes
        )
        
        # Test with random token IDs
        token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        logits = model(token_ids)
        
        assert logits.shape == (batch_size, num_classes)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()
    
    def test_transformer_gradient_flow(self):
        """Test gradients flow through the model"""
        model = SimpleTransformer(100, 64, 2, 1, 256, 20, 2)
        
        token_ids = torch.randint(0, 100, (1, 10))
        target = torch.tensor([1])
        
        logits = model(token_ids)
        loss = nn.CrossEntropyLoss()(logits, target)
        loss.backward()
        
        # Check that gradients exist
        for param in model.parameters():
            assert param.grad is not None

class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_attention_mechanism_consistency(self):
        """Test attention mechanism produces consistent results"""
        torch.manual_seed(42)
        
        d_model, num_heads = 64, 4
        mha = MultiHeadAttention(d_model, num_heads)
        
        x = torch.randn(1, 8, d_model)
        
        # Run multiple times with same input
        output1, _ = mha(x, x, x)
        output2, _ = mha(x, x, x)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_model_training_step(self):
        """Test model can perform a training step"""
        model = SimpleTransformer(50, 32, 2, 1, 128, 10, 2)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        
        # Create dummy batch
        batch_size = 4
        seq_len = 8
        token_ids = torch.randint(0, 50, (batch_size, seq_len))
        labels = torch.randint(0, 2, (batch_size,))
        
        # Training step
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert not torch.isnan(loss)

def run_performance_tests():
    """Run performance benchmarks"""
    print("🚀 Running Performance Tests...")
    
    import time
    
    # Test attention performance
    d_model, num_heads = 512, 8
    batch_size, seq_len = 16, 128
    
    mha = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    
    start_time = time.time()
    for _ in range(10):
        output, _ = mha(x, x, x)
    attention_time = (time.time() - start_time) / 10
    
    print(f"✅ Multi-head attention: {attention_time:.4f} seconds per forward pass")
    
    # Test full model performance
    model = SimpleTransformer(1000, 256, 8, 4, 1024, 100, 2)
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    start_time = time.time()
    for _ in range(5):
        logits = model(token_ids)
    model_time = (time.time() - start_time) / 5
    
    print(f"✅ Full transformer: {model_time:.4f} seconds per forward pass")
    
    # Performance assertions
    assert attention_time < 0.1, "Attention too slow"
    assert model_time < 0.5, "Model too slow"
    
    print("🎉 All performance tests passed!")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run performance tests
    run_performance_tests()