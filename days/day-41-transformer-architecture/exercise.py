#!/usr/bin/env python3
"""
Day 41: Transformer Architecture - Exercise

This exercise provides hands-on implementation of Transformer components from scratch.
You'll build the core attention mechanisms, positional encoding, and transformer layers
to understand how these revolutionary models work under the hood.

Learning Objectives:
- Implement scaled dot-product attention from scratch
- Build multi-head attention mechanisms
- Create positional encoding for sequence information
- Construct complete transformer encoder layers
- Train a small transformer on a real NLP task

Time: 40 minutes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class ScaledDotProductAttention(nn.Module):
    """
    Implement the core scaled dot-product attention mechanism.
    
    This is the fundamental building block of Transformers.
    Formula: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) @ V
    """
    
    def __init__(self, d_k: int, dropout: float = 0.1):
        super().__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Implement scaled dot-product attention
        
        Args:
            query: Query tensor of shape (batch_size, seq_len, d_k)
            key: Key tensor of shape (batch_size, seq_len, d_k)  
            value: Value tensor of shape (batch_size, seq_len, d_v)
            mask: Optional mask tensor to prevent attention to certain positions
            
        Returns:
            output: Attention output of shape (batch_size, seq_len, d_v)
            attention_weights: Attention weights of shape (batch_size, seq_len, seq_len)
            
        HINT: 
        1. Compute attention scores: Q @ K^T
        2. Scale by sqrt(d_k)
        3. Apply mask if provided (set masked positions to -inf)
        4. Apply softmax to get attention weights
        5. Apply dropout to attention weights
        6. Multiply by values: attention_weights @ V
        """
        
        # Your implementation here
        batch_size, seq_len, d_k = query.size()
        
        # Step 1: Compute attention scores
        # Compute QK^T
        scores = torch.matmul(query, key.transpose(-2, -1))
        
        # Step 2: Scale by sqrt(d_k)
        # TODO: Apply scaling factor
        
        # Step 3: Apply mask if provided
        if mask is not None:
            # TODO: Set masked positions to -inf
            pass
        
        # Step 4: Apply softmax
        # Compute attention weights using softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # Step 5: Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Step 6: Apply attention to values
        # Compute final output
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights

class MultiHeadAttention(nn.Module):
    """
    Implement multi-head attention mechanism.
    
    This allows the model to jointly attend to information from different 
    representation subspaces at different positions.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize linear projections for Q, K, V and output
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention(self.d_k, dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TODO: Implement multi-head attention
        
        Args:
            query, key, value: Input tensors of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            output: Multi-head attention output
            attention_weights: Averaged attention weights across heads
            
        HINT:
        1. Apply linear projections to get Q, K, V
        2. Reshape to separate heads: (batch_size, num_heads, seq_len, d_k)
        3. Apply scaled dot-product attention for each head
        4. Concatenate heads and apply output projection
        """
        
        batch_size, seq_len, d_model = query.size()
        
        # Step 1: Apply linear projections
        # Compute Q, K, V using the linear layers
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)
        
        # Step 2: Reshape for multi-head attention
        # Reshape to (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Step 3: Apply attention
        # Apply scaled dot-product attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Step 4: Concatenate heads
        # Reshape back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        # Apply output projection
        output = self.w_o(attn_output)
        
        return output, attn_weights

class PositionalEncoding(nn.Module):
    """
    Implement sinusoidal positional encoding.
    
    Since attention is permutation-invariant, we need to inject 
    position information into the model.
    """
    
    def __init__(self, d_model: int, max_seq_length: int = 5000):
        super().__init__()
        
        # TODO: Implement sinusoidal positional encoding
        # HINT: Use the formulas:
        # PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        # Compute the positional encoding
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        # TODO: pe[:, 0::2] = ?
        
        # Apply cos to odd indices  
        # TODO: pe[:, 1::2] = ?
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input embeddings of shape (seq_len, batch_size, d_model)
            
        Returns:
            x + positional encoding
        """
        # Add positional encoding to input
        return x + self.pe[:x.size(0), :]

class TransformerEncoderLayer(nn.Module):
    """
    Implement a single Transformer encoder layer.
    
    Architecture:
    1. Multi-head self-attention
    2. Add & Norm (residual connection + layer normalization)
    3. Feed-forward network
    4. Add & Norm (residual connection + layer normalization)
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # Initialize components
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        TODO: Implement transformer encoder layer forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Output tensor of same shape as input
            
        HINT: Follow the "Add & Norm" pattern:
        1. attn_output = self_attention(x) + x  # Residual connection
        2. x = layer_norm(attn_output)          # Layer normalization
        3. ff_output = feed_forward(x) + x      # Residual connection  
        4. x = layer_norm(ff_output)            # Layer normalization
        """
        
        # Step 1: Self-attention with residual connection
        # TODO: Apply self-attention and add residual connection
        attn_output, _ = None, None  # Replace with attention computation
        x = None  # Replace with residual connection and normalization
        
        # Step 2: Feed-forward with residual connection
        # TODO: Apply feed-forward network and add residual connection
        ff_output = None  # Replace with feed-forward computation
        x = None  # Replace with residual connection and normalization
        
        return x

class SimpleTransformer(nn.Module):
    """
    A simple Transformer model for text classification.
    
    This demonstrates how to combine all the components into a working model.
    """
    
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, 
                 num_layers: int, d_ff: int, max_seq_length: int, 
                 num_classes: int, dropout: float = 0.1):
        super().__init__()
        
        # Initialize all components
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        # Create a stack of encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Add classification head
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        TODO: Implement full transformer forward pass
        
        Args:
            x: Input token ids of shape (batch_size, seq_len)
            mask: Optional attention mask
            
        Returns:
            Classification logits of shape (batch_size, num_classes)
        """
        
        # Step 1: Embedding and positional encoding
        # Convert tokens to embeddings and add positional encoding
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # Step 2: Apply transformer layers
        # Pass through all encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask)
        
        # Step 3: Classification
        # Pool the sequence (use mean pooling) and classify
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        
        return logits

def create_padding_mask(sequences: torch.Tensor, pad_token: int = 0) -> torch.Tensor:
    """
    TODO: Create padding mask to ignore padded tokens in attention.
    
    Args:
        sequences: Token sequences of shape (batch_size, seq_len)
        pad_token: Token ID used for padding
        
    Returns:
        Mask tensor of shape (batch_size, 1, 1, seq_len) where True = ignore
        
    HINT: Padded positions should be True, non-padded positions should be False
    """
    # Your implementation here
    return None

def visualize_attention(attention_weights: torch.Tensor, tokens: list, head_idx: int = 0):
    """
    TODO: Visualize attention weights as a heatmap
    
    Args:
        attention_weights: Attention weights of shape (batch_size, num_heads, seq_len, seq_len)
        tokens: List of token strings
        head_idx: Which attention head to visualize
        
    HINT: Use matplotlib/seaborn to create a heatmap
    """
    # Extract attention weights for specified head
    attn = attention_weights[0, head_idx].detach().cpu().numpy()
    
    # TODO: Create heatmap visualization
    # Your implementation here
    pass

def train_simple_example():
    """
    TODO: Train a simple transformer on a toy classification task
    
    This function demonstrates how to use the transformer for a real task.
    Create a simple binary classification task (e.g., positive/negative sentiment).
    """
    
    # TODO: Create toy dataset
    # HINT: You can create simple sequences like:
    # Positive: ["good", "great", "excellent", "amazing"]
    # Negative: ["bad", "terrible", "awful", "horrible"]
    
    # TODO: Initialize model
    vocab_size = 1000  # Adjust based on your vocabulary
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=128,
        num_heads=8,
        num_layers=2,
        d_ff=512,
        max_seq_length=50,
        num_classes=2,
        dropout=0.1
    )
    
    # TODO: Training loop
    # 1. Create data loader
    # 2. Define loss function and optimizer
    # 3. Train for a few epochs
    # 4. Evaluate performance
    
    print("Training completed! (Implement the training loop)")
    
    return model

def main():
    """
    Main function to test all implemented components
    """
    print("🤖 Transformer Architecture Implementation")
    print("=" * 50)
    
    # Test 1: Scaled Dot-Product Attention
    print("\n🔍 Testing Scaled Dot-Product Attention...")
    d_k = 64
    seq_len = 10
    batch_size = 2
    
    attention = ScaledDotProductAttention(d_k)
    
    # Create random Q, K, V tensors
    Q = torch.randn(batch_size, seq_len, d_k)
    K = torch.randn(batch_size, seq_len, d_k)
    V = torch.randn(batch_size, seq_len, d_k)
    
    output, weights = attention(Q, K, V)
    print(f"✅ Attention output shape: {output.shape}")
    print(f"✅ Attention weights shape: {weights.shape}")
    
    # Test 2: Multi-Head Attention
    print("\n🔍 Testing Multi-Head Attention...")
    d_model = 512
    num_heads = 8
    
    mha = MultiHeadAttention(d_model, num_heads)
    
    x = torch.randn(batch_size, seq_len, d_model)
    output, weights = mha(x, x, x)
    print(f"✅ Multi-head output shape: {output.shape}")
    
    # Test 3: Positional Encoding
    print("\n🔍 Testing Positional Encoding...")
    pos_enc = PositionalEncoding(d_model)
    
    # Test with sequence length dimension first (seq_len, batch_size, d_model)
    x_pos = torch.randn(seq_len, batch_size, d_model)
    encoded = pos_enc(x_pos)
    print(f"✅ Positional encoding shape: {encoded.shape}")
    
    # Test 4: Transformer Encoder Layer
    print("\n🔍 Testing Transformer Encoder Layer...")
    encoder_layer = TransformerEncoderLayer(d_model, num_heads, d_ff=2048)
    
    x_enc = torch.randn(batch_size, seq_len, d_model)
    output = encoder_layer(x_enc)
    print(f"✅ Encoder layer output shape: {output.shape}")
    
    # Test 5: Full Transformer Model
    print("\n🔍 Testing Full Transformer Model...")
    model = SimpleTransformer(
        vocab_size=1000,
        d_model=256,
        num_heads=8,
        num_layers=4,
        d_ff=1024,
        max_seq_length=100,
        num_classes=2
    )
    
    # Test with token sequences
    token_ids = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(token_ids)
    print(f"✅ Model output shape: {logits.shape}")
    
    # Test 6: Training Example
    print("\n🔍 Running Training Example...")
    trained_model = train_simple_example()
    
    print(f"\n🎯 All tests completed!")
    print(f"You've successfully implemented the core components of a Transformer!")
    
    # Bonus: Attention Visualization
    print(f"\n📊 Bonus: Try implementing attention visualization")
    print(f"Use the visualize_attention function to see what the model learns")
    
    return {
        "attention": attention,
        "multi_head_attention": mha,
        "positional_encoding": pos_enc,
        "encoder_layer": encoder_layer,
        "full_model": model,
        "trained_model": trained_model
    }

if __name__ == "__main__":
    # Run all tests
    results = main()
    
    print(f"\n" + "=" * 50)
    print(f"🚀 Transformer Implementation Complete!")
    print(f"=" * 50)
    
    print(f"\n🎯 What you've accomplished:")
    print(f"  • Implemented scaled dot-product attention from scratch")
    print(f"  • Built multi-head attention mechanisms")
    print(f"  • Created sinusoidal positional encoding")
    print(f"  • Constructed complete transformer encoder layers")
    print(f"  • Built a full transformer model for classification")
    
    print(f"\n🔬 Key insights:")
    print(f"  • Attention allows parallel processing of sequences")
    print(f"  • Multi-head attention captures different relationship types")
    print(f"  • Positional encoding preserves sequence order information")
    print(f"  • Residual connections enable training of deep networks")
    print(f"  • Layer normalization stabilizes training")
    
    print(f"\n🚀 Ready for Day 42: Attention Mechanisms Deep Dive!")

"""
EXERCISE COMPLETION CHECKLIST:

Core Components (Required):
□ Implement ScaledDotProductAttention.forward()
□ Implement MultiHeadAttention.__init__() and forward()
□ Implement PositionalEncoding.__init__() and forward()
□ Implement TransformerEncoderLayer.__init__() and forward()
□ Implement SimpleTransformer.__init__() and forward()

Helper Functions (Required):
□ Implement create_padding_mask()
□ Implement basic training loop in train_simple_example()

Bonus Challenges (Optional):
□ Implement visualize_attention() with matplotlib
□ Add causal masking for decoder-style attention
□ Implement layer-wise learning rate decay
□ Add gradient clipping and warmup scheduling
□ Create attention pattern analysis tools

Testing (Required):
□ All components pass shape tests
□ Model can forward pass without errors
□ Training loop runs without crashes
□ Attention weights sum to 1.0 across sequence dimension

Advanced Extensions (Optional):
□ Implement sparse attention patterns
□ Add relative positional encoding
□ Create attention head pruning
□ Implement knowledge distillation
□ Add mixed precision training support

SCORING RUBRIC:
- Core Implementation (70 points): All required components working
- Code Quality (15 points): Clean, well-documented, efficient code
- Understanding (10 points): Correct mathematical implementation
- Bonus Features (5 points): Additional optimizations or visualizations

TOTAL: 100 points

This exercise provides hands-on experience with the fundamental building blocks
of modern AI systems. Understanding these components deeply will prepare you
for working with large language models and advanced GenAI applications.
"""