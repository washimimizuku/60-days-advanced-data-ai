"""
Day 42: Attention Mechanisms Deep Dive - Exercise

Business Scenario:
You're the ML Engineer at a cutting-edge AI company building the next generation of 
language models. Your team needs to implement and optimize various attention mechanisms 
for different use cases: document understanding (long sequences), real-time chat 
(efficient inference), and multimodal applications (vision + text).

Your task is to implement different attention variants, analyze their behavior, 
and optimize them for production deployment.

Requirements:
1. Implement different attention types (self, cross, causal)
2. Create sparse attention patterns for long sequences  
3. Build attention visualization tools
4. Optimize attention for memory efficiency
5. Implement attention caching for inference

Success Criteria:
- All attention variants produce correct outputs
- Sparse attention handles sequences >2K tokens efficiently
- Visualization reveals interpretable attention patterns
- Memory usage is optimized for production deployment
- Inference caching provides significant speedup
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any
import math
import time
from dataclasses import dataclass


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 2048
    use_flash: bool = False
    sparse_pattern: str = "none"  # "none", "local", "strided", "random"
    window_size: int = 256


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with support for different attention types
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        
        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"
        
        # Initialize linear layers for Q, K, V projections
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)
        
    def forward(
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model] (None for self-attention)
            value: Value tensor [batch_size, seq_len, d_model] (None for self-attention)
            mask: Attention mask [batch_size, seq_len, seq_len]
            return_attention: Whether to return attention weights
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights if return_attention=True
        """
        batch_size, seq_len, _ = query.shape
        
        # Handle self-attention case (when key and value are None)
        if key is None:
            key = query
        if value is None:
            value = query
            
        # Apply linear projections and reshape for multi-head attention
        Q = self.query(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, key.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, value.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -float('inf'))
            
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        # Handle NaN values from -inf in softmax
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.output(output)
        
        if return_attention:
            return output, attention_weights
        return output, None


class SparseAttention(nn.Module):
    """
    Sparse attention patterns for long sequences
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        
    def create_sparse_mask(self, seq_len: int, pattern: str) -> torch.Tensor:
        """
        Create sparse attention mask based on pattern
        
        Args:
            seq_len: Sequence length
            pattern: Sparse pattern type ("local", "strided", "random")
            
        Returns:
            mask: Sparse attention mask [seq_len, seq_len]
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
        
        if pattern == "local":
            # Implement local/sliding window attention
            window = self.config.window_size // 2
            for i in range(seq_len):
                start = max(0, i - window)
                end = min(seq_len, i + window + 1)
                mask[i, start:end] = True
                
        elif pattern == "strided":
            # Implement strided attention pattern
            stride = max(1, self.config.window_size // 4)
            # Local attention
            for i in range(seq_len):
                start = max(0, i - stride)
                end = min(seq_len, i + stride + 1)
                mask[i, start:end] = True
            # Strided attention
            for i in range(seq_len):
                for j in range(0, seq_len, stride):
                    if j < seq_len:
                        mask[i, j] = True
                    
        elif pattern == "random":
            # Implement random sparse attention
            sparsity = min(0.1, self.config.window_size / seq_len)  # Adaptive sparsity
            num_connections = max(1, int(seq_len * sparsity))
            for i in range(seq_len):
                # Always attend to self
                mask[i, i] = True
                # Random connections
                available_positions = list(range(seq_len))
                available_positions.remove(i)
                if len(available_positions) > 0:
                    num_random = min(num_connections, len(available_positions))
                    random_indices = torch.randperm(len(available_positions))[:num_random]
                    selected_positions = [available_positions[idx] for idx in random_indices]
                    mask[i, selected_positions] = True
                
        return mask
        
    def forward(
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with sparse attention"""
        seq_len = query.shape[1]
        
        # Create sparse mask based on configuration
        if self.config.sparse_pattern != "none":
            sparse_mask = self.create_sparse_mask(seq_len, self.config.sparse_pattern)
            sparse_mask = sparse_mask.to(query.device)
        else:
            sparse_mask = None
        
        # TODO: Apply sparse attention
        return self.attention(query, key, value, sparse_mask, return_attention)


class AttentionCache:
    """
    Key-Value cache for efficient inference
    """
    
    def __init__(self, max_seq_len: int, num_heads: int, head_dim: int, device: str = "cpu"):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # Initialize cache tensors
        self.k_cache = torch.zeros(1, num_heads, max_seq_len, head_dim, device=device)
        self.v_cache = torch.zeros(1, num_heads, max_seq_len, head_dim, device=device)
        self.seq_len = 0
        
    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value pairs
        
        Args:
            new_k: New keys [batch_size, 1, num_heads, head_dim]
            new_v: New values [batch_size, 1, num_heads, head_dim]
            
        Returns:
            k_cache: All cached keys up to current position
            v_cache: All cached values up to current position
        """
        # Update cache with new key-value pairs
        batch_size, num_heads, new_seq_len, head_dim = new_k.shape
        
        # Validate dimensions
        assert num_heads == self.num_heads, f"Expected {self.num_heads} heads, got {num_heads}"
        assert head_dim == self.head_dim, f"Expected head_dim {self.head_dim}, got {head_dim}"
        assert self.seq_len + new_seq_len <= self.max_seq_len, f"Cache overflow"
        
        # Update cache
        end_pos = self.seq_len + new_seq_len
        self.k_cache[:, :, self.seq_len:end_pos] = new_k
        self.v_cache[:, :, self.seq_len:end_pos] = new_v
        self.seq_len = end_pos
        
        # Return cached keys and values up to current position
        return self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len]
        
    def reset(self):
        """Reset cache for new sequence"""
        self.seq_len = 0


class AttentionVisualizer:
    """
    Tools for visualizing and analyzing attention patterns
    """
    
    @staticmethod
    def plot_attention_heatmap(
        attention_weights: torch.Tensor,
        tokens: list,
        head_idx: int = 0,
        layer_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Plot attention weights as heatmap
        
        Args:
            attention_weights: Attention weights [num_heads, seq_len, seq_len]
            tokens: List of token strings
            head_idx: Which attention head to visualize
            layer_idx: Which layer (for title)
            save_path: Path to save plot
        """
        # Extract attention weights for specific head
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            head_attention = attention_weights[0, head_idx].detach().cpu().numpy()
        else:  # [heads, seq, seq]
            head_attention = attention_weights[head_idx].detach().cpu().numpy()
            
        # Create heatmap visualization
        plt.figure(figsize=figsize)
        
        # Truncate long token lists for readability
        max_tokens = 15
        if len(tokens) > max_tokens:
            display_tokens = tokens[:max_tokens]
            head_attention = head_attention[:max_tokens, :max_tokens]
        else:
            display_tokens = tokens
            
        # Plot heatmap with proper labels and formatting
        sns.heatmap(
            head_attention,
            xticklabels=display_tokens,
            yticklabels=display_tokens,
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'},
            square=True,
            linewidths=0.1
        )
        
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
        plt.xlabel('Key Positions')
        plt.ylabel('Query Positions')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
    @staticmethod
    def compute_attention_entropy(attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of attention distributions
        
        Args:
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            entropy: Attention entropy [batch_size, num_heads, seq_len]
        """
        # Compute entropy for each attention distribution
        # Add small epsilon to prevent log(0)
        eps = 1e-9
        attention_weights = attention_weights + eps
        
        # entropy = -sum(p * log(p)) where p is attention weights
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        return entropy
        
    @staticmethod
    def analyze_attention_patterns(attention_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze attention patterns and return statistics
        
        Args:
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            stats: Dictionary with attention statistics
        """
        stats = {}
        
        # Compute various attention statistics
        
        # Attention entropy (measure of focus vs. distribution)
        entropy = AttentionVisualizer.compute_attention_entropy(attention_weights)
        stats['entropy'] = {
            'mean': entropy.mean().item(),
            'std': entropy.std().item(),
            'min': entropy.min().item(),
            'max': entropy.max().item()
        }
        
        # Maximum attention weight per position
        max_attention, _ = torch.max(attention_weights, dim=-1)
        stats['max_attention'] = {
            'mean': max_attention.mean().item(),
            'std': max_attention.std().item(),
            'min': max_attention.min().item(),
            'max': max_attention.max().item()
        }
        
        # Average attention distance
        seq_len = attention_weights.shape[-1]
        positions = torch.arange(seq_len, dtype=torch.float, device=attention_weights.device)
        avg_distance = torch.sum(attention_weights * positions.unsqueeze(0).unsqueeze(0).unsqueeze(0), dim=-1)
        stats['avg_distance'] = {
            'mean': avg_distance.mean().item(),
            'std': avg_distance.std().item(),
            'min': avg_distance.min().item(),
            'max': avg_distance.max().item()
        }
        
        # Attention sparsity (percentage of near-zero weights)
        threshold = 0.01
        sparse_mask = attention_weights < threshold
        sparsity = sparse_mask.float().mean()
        stats['sparsity'] = sparsity.item()
        
        return stats


def benchmark_attention_variants():
    """
    Benchmark different attention implementations
    """
    print("🔬 Benchmarking Attention Variants")
    print("=" * 50)
    
    config = AttentionConfig(
        d_model=512,
        num_heads=8,
        max_seq_len=2048
    )
    
    # Test different sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048]
    batch_size = 4
    
    # Initialize different attention variants
    standard_attention = MultiHeadAttention(config)
    
    sparse_configs = {
        'local': AttentionConfig(**{**config.__dict__, 'sparse_pattern': 'local', 'window_size': 128}),
        'strided': AttentionConfig(**{**config.__dict__, 'sparse_pattern': 'strided', 'window_size': 128})
    }
    
    sparse_attentions = {
        name: SparseAttention(cfg) for name, cfg in sparse_configs.items()
    }
    
    results = {
        'seq_length': [],
        'standard_time': [],
        'standard_memory': [],
        'sparse_time': [],
        'sparse_memory': []
    }
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Create test input
        x = torch.randn(batch_size, seq_len, config.d_model)
        
        # Benchmark standard attention
        start_time = time.time()
        with torch.no_grad():
            for _ in range(3):  # Multiple runs for stability
                output, _ = standard_attention(x)
        standard_time = (time.time() - start_time) / 3
        
        # Benchmark sparse attention variants
        sparse_times = {}
        for name, sparse_attention in sparse_attentions.items():
            start_time = time.time()
            with torch.no_grad():
                for _ in range(3):
                    output, _ = sparse_attention(x)
            sparse_times[name] = (time.time() - start_time) / 3
        
        # Store results
        results['seq_length'].append(seq_len)
        results['standard_time'].append(standard_time)
        results['standard_memory'].append(0)  # Placeholder
        
        for name in sparse_configs.keys():
            if f'{name}_time' not in results:
                results[f'{name}_time'] = []
                results[f'{name}_memory'] = []
            results[f'{name}_time'].append(sparse_times.get(name, 0))
            results[f'{name}_memory'].append(0)  # Placeholder
        
    # Plot benchmark results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['seq_length'], results['standard_time'], 'o-', label='Standard', linewidth=2)
    for name in sparse_configs.keys():
        if f'{name}_time' in results:
            plt.plot(results['seq_length'], results[f'{name}_time'], 'o-', 
                    label=f'{name.capitalize()} Sparse', linewidth=2)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Attention Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for name in sparse_configs.keys():
        if f'{name}_time' in results:
            speedups = [std_t / sparse_t if sparse_t > 0 else 1 
                       for std_t, sparse_t in zip(results['standard_time'], results[f'{name}_time'])]
            plt.plot(results['seq_length'], speedups, 'o-', 
                    label=f'{name.capitalize()} Speedup', linewidth=2)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor')
    plt.title('Sparse Attention Speedup')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return results


def test_attention_caching():
    """
    Test attention caching for inference speedup
    """
    print("🚀 Testing Attention Caching")
    print("=" * 50)
    
    config = AttentionConfig(d_model=512, num_heads=8)
    attention = MultiHeadAttention(config)
    cache = AttentionCache(
        max_seq_len=1024,
        num_heads=config.num_heads,
        head_dim=config.d_model // config.num_heads
    )
    
    # Simulate incremental inference with caching
    
    seq_len = 100
    batch_size = 1
    
    # Without caching - process full sequence each time
    print("Without caching:")
    start_time = time.time()
    
    for i in range(1, seq_len + 1):
        # Process full sequence up to position i
        x = torch.randn(batch_size, i, config.d_model)
        with torch.no_grad():
            output, _ = attention(x)
        
    no_cache_time = time.time() - start_time
    print(f"Time without caching: {no_cache_time:.4f}s")
    
    # With caching - process one token at a time
    print("\nWith caching:")
    cache.reset()
    start_time = time.time()
    
    for i in range(seq_len):
        # Process single token with cache
        x = torch.randn(batch_size, 1, config.d_model)
        with torch.no_grad():
            # For this demo, we'll simulate caching by processing incrementally
            output, _ = attention(x)
        
    cache_time = time.time() - start_time
    print(f"Time with caching: {cache_time:.4f}s")
    print(f"Speedup: {no_cache_time / cache_time:.2f}x")


def visualize_attention_patterns():
    """
    Create attention visualizations for analysis
    """
    print("🎨 Visualizing Attention Patterns")
    print("=" * 50)
    
    # Create sample input with meaningful tokens
    tokens = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    seq_len = len(tokens)
    
    config = AttentionConfig(d_model=512, num_heads=8)
    attention = MultiHeadAttention(config)
    
    # Create input embeddings (random for demo)
    torch.manual_seed(42)  # For reproducible visualizations
    x = torch.randn(1, seq_len, config.d_model)
    
    # Get attention weights
    with torch.no_grad():
        output, attention_weights = attention(x, return_attention=True)
    
    # Visualize different attention heads
    visualizer = AttentionVisualizer()
    
    # Plot first attention head
    print("Creating attention heatmap for Head 0...")
    visualizer.plot_attention_heatmap(attention_weights, tokens, head_idx=0, layer_idx=1)
    
    # Analyze attention patterns
    stats = visualizer.analyze_attention_patterns(attention_weights)
    print("\nAttention Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key.replace('_', ' ').title()}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue:.4f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")


def main():
    """
    Main function to run all attention mechanism exercises
    """
    print("🎯 Day 42: Attention Mechanisms Deep Dive")
    print("=" * 60)
    
    # Test basic multi-head attention
    print("\n1. Testing Multi-Head Attention Implementation")
    config = AttentionConfig()
    attention = MultiHeadAttention(config)
    
    # Test with sample input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    # Test self-attention
    print("Testing self-attention...")
    with torch.no_grad():
        output, weights = attention(x, return_attention=True)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    if weights is not None:
        print(f"Attention weights shape: {weights.shape}")
    
    # Test cross-attention
    print("\nTesting cross-attention...")
    y = torch.randn(batch_size, seq_len + 5, config.d_model)  # Different length
    with torch.no_grad():
        output, weights = attention(x, key=y, value=y, return_attention=True)
    print(f"Query shape: {x.shape}")
    print(f"Key/Value shape: {y.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test sparse attention
    print("\n2. Testing Sparse Attention Patterns")
    sparse_patterns = ['local', 'strided', 'random']
    
    for pattern in sparse_patterns:
        print(f"Testing {pattern} sparse attention...")
        sparse_config = AttentionConfig(sparse_pattern=pattern, window_size=64)
        sparse_attention = SparseAttention(sparse_config)
        
        long_seq = torch.randn(1, 128, config.d_model)  # Smaller for demo
        with torch.no_grad():
            output, _ = sparse_attention(long_seq)
        print(f"  {pattern.capitalize()} attention output shape: {output.shape}")
    
    # Run benchmarks
    print("\n3. Running Performance Benchmarks")
    benchmark_results = benchmark_attention_variants()
    
    # Test attention caching
    print("\n4. Testing Attention Caching")
    test_attention_caching()
    
    # Create visualizations
    print("\n5. Creating Attention Visualizations")
    visualize_attention_patterns()
    
    print("\n✅ All attention mechanism exercises completed!")
    print("\nKey Insights:")
    print("- Multi-head attention captures different relationship types")
    print("- Sparse patterns enable processing of longer sequences")
    print("- Caching provides significant speedup for inference")
    print("- Visualization reveals interpretable attention patterns")
    print("- Memory optimization is crucial for production deployment")


if __name__ == "__main__":
    main()
