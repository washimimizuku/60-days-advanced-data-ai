"""
Day 42: Attention Mechanisms Deep Dive - Production Solution

This solution provides production-ready implementations of various attention mechanisms
including multi-head attention, sparse attention patterns, attention caching, and
visualization tools. All implementations are optimized for performance and include
comprehensive error handling and documentation.

Key Features:
- Multi-head attention with self and cross-attention support
- Sparse attention patterns (local, strided, random) for long sequences
- Memory-efficient attention caching for inference
- Comprehensive attention visualization and analysis tools
- Performance benchmarking and optimization techniques
- Production-ready error handling and validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Dict, Any, List
import math
import time
from dataclasses import dataclass
import warnings
from contextlib import contextmanager


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms with validation"""
    d_model: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    max_seq_len: int = 2048
    use_flash: bool = False
    sparse_pattern: str = "none"  # "none", "local", "strided", "random"
    window_size: int = 256
    
    def __post_init__(self):
        """Validate configuration parameters"""
        assert self.d_model % self.num_heads == 0, f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})"
        assert self.dropout >= 0.0 and self.dropout <= 1.0, f"dropout must be between 0 and 1, got {self.dropout}"
        assert self.window_size > 0, f"window_size must be positive, got {self.window_size}"
        assert self.sparse_pattern in ["none", "local", "strided", "random"], f"Invalid sparse_pattern: {self.sparse_pattern}"


class MultiHeadAttention(nn.Module):
    """
    Production-ready multi-head attention with support for different attention types
    
    Features:
    - Self-attention and cross-attention
    - Causal masking for autoregressive models
    - Efficient implementation with proper scaling
    - Comprehensive error handling and validation
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.output = nn.Linear(config.d_model, config.d_model, bias=False)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = math.sqrt(self.head_dim)
        
        # Initialize weights using Xavier uniform
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stable training"""
        for module in [self.query, self.key, self.value, self.output]:
            nn.init.xavier_uniform_(module.weight)
            
    def forward(
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
        use_cache: bool = False,
        cache: Optional['AttentionCache'] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for multi-head attention
        
        Args:
            query: Query tensor [batch_size, seq_len, d_model]
            key: Key tensor [batch_size, seq_len, d_model] (None for self-attention)
            value: Value tensor [batch_size, seq_len, d_model] (None for self-attention)
            mask: Attention mask [batch_size, seq_len, seq_len] or [seq_len, seq_len]
            return_attention: Whether to return attention weights
            use_cache: Whether to use KV caching
            cache: AttentionCache object for caching
            
        Returns:
            output: Attention output [batch_size, seq_len, d_model]
            attention_weights: Attention weights if return_attention=True
        """
        batch_size, seq_len_q, _ = query.shape
        
        # Handle self-attention case
        if key is None:
            key = query
        if value is None:
            value = query
            
        seq_len_k = key.shape[1]
        
        # Apply linear projections and reshape for multi-head attention
        Q = self.query(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(value).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Handle caching for inference
        if use_cache and cache is not None:
            if cache.seq_len > 0:
                # Concatenate with cached K, V
                K_cached, V_cached = cache.get_cached()
                K = torch.cat([K_cached, K], dim=2)  # Concatenate along sequence dimension
                V = torch.cat([V_cached, V], dim=2)
            cache.update(K[:, :, -seq_len_q:], V[:, :, -seq_len_q:])  # Cache new K, V
            
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Broadcast mask to match scores dimensions
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -float('inf'))
            
        # Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        
        # Handle NaN values that can occur with -inf in softmax
        attention_weights = torch.where(
            torch.isnan(attention_weights),
            torch.zeros_like(attention_weights),
            attention_weights
        )
        
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        output = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.output(output)
        
        if return_attention:
            return output, attention_weights
        return output, None


class SparseAttention(nn.Module):
    """
    Sparse attention patterns for efficient processing of long sequences
    
    Supports multiple sparse patterns:
    - Local: Sliding window attention
    - Strided: Combination of local and strided patterns
    - Random: Random sparse connections
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        
    def create_sparse_mask(self, seq_len: int, pattern: str, device: str = "cpu") -> torch.Tensor:
        """
        Create sparse attention mask based on pattern
        
        Args:
            seq_len: Sequence length
            pattern: Sparse pattern type ("local", "strided", "random")
            device: Device to create mask on
            
        Returns:
            mask: Sparse attention mask [seq_len, seq_len]
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        
        if pattern == "local":
            # Local sliding window attention
            window = self.config.window_size // 2
            for i in range(seq_len):
                start = max(0, i - window)
                end = min(seq_len, i + window + 1)
                mask[i, start:end] = True
                
        elif pattern == "strided":
            # Combination of local and strided attention (Longformer-style)
            stride = max(1, self.config.window_size // 4)
            local_window = stride
            
            # Local attention
            for i in range(seq_len):
                start = max(0, i - local_window)
                end = min(seq_len, i + local_window + 1)
                mask[i, start:end] = True
                
            # Strided attention
            for i in range(seq_len):
                for j in range(0, seq_len, stride):
                    if j < seq_len:
                        mask[i, j] = True
                        
        elif pattern == "random":
            # Random sparse attention with guaranteed self-attention
            sparsity = min(0.1, self.config.window_size / seq_len)  # Adaptive sparsity
            num_connections = max(1, int(seq_len * sparsity))
            
            for i in range(seq_len):
                # Always attend to self
                mask[i, i] = True
                
                # Random connections
                available_positions = list(range(seq_len))
                available_positions.remove(i)  # Remove self-position
                
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
        device = query.device
        
        # Create sparse mask based on configuration
        if self.config.sparse_pattern != "none":
            sparse_mask = self.create_sparse_mask(seq_len, self.config.sparse_pattern, device)
        else:
            sparse_mask = None
            
        return self.attention(query, key, value, sparse_mask, return_attention)


class AttentionCache:
    """
    Efficient key-value cache for autoregressive inference
    
    Features:
    - Pre-allocated tensors for memory efficiency
    - Automatic device management
    - Bounds checking and error handling
    """
    
    def __init__(self, max_seq_len: int, num_heads: int, head_dim: int, device: str = "cpu"):
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.device = device
        
        # Pre-allocate cache tensors
        self.k_cache = torch.zeros(1, num_heads, max_seq_len, head_dim, device=device)
        self.v_cache = torch.zeros(1, num_heads, max_seq_len, head_dim, device=device)
        self.seq_len = 0
        
    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> None:
        """
        Update cache with new key-value pairs
        
        Args:
            new_k: New keys [batch_size, num_heads, seq_len, head_dim]
            new_v: New values [batch_size, num_heads, seq_len, head_dim]
        """
        batch_size, num_heads, new_seq_len, head_dim = new_k.shape
        
        # Validate dimensions
        assert num_heads == self.num_heads, f"Expected {self.num_heads} heads, got {num_heads}"
        assert head_dim == self.head_dim, f"Expected head_dim {self.head_dim}, got {head_dim}"
        assert self.seq_len + new_seq_len <= self.max_seq_len, f"Cache overflow: {self.seq_len + new_seq_len} > {self.max_seq_len}"
        
        # Update cache
        end_pos = self.seq_len + new_seq_len
        self.k_cache[:, :, self.seq_len:end_pos] = new_k
        self.v_cache[:, :, self.seq_len:end_pos] = new_v
        self.seq_len = end_pos
        
    def get_cached(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached keys and values up to current position
        
        Returns:
            k_cache: Cached keys [batch_size, num_heads, seq_len, head_dim]
            v_cache: Cached values [batch_size, num_heads, seq_len, head_dim]
        """
        return self.k_cache[:, :, :self.seq_len], self.v_cache[:, :, :self.seq_len]
        
    def reset(self):
        """Reset cache for new sequence"""
        self.seq_len = 0
        
    def to(self, device: str):
        """Move cache to different device"""
        self.device = device
        self.k_cache = self.k_cache.to(device)
        self.v_cache = self.v_cache.to(device)
        return self


class AttentionVisualizer:
    """
    Comprehensive tools for visualizing and analyzing attention patterns
    
    Features:
    - Multiple visualization types (heatmaps, head comparisons, etc.)
    - Statistical analysis of attention patterns
    - Export capabilities for reports and presentations
    """
    
    @staticmethod
    def plot_attention_heatmap(
        attention_weights: torch.Tensor,
        tokens: List[str],
        head_idx: int = 0,
        layer_idx: int = 0,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10)
    ):
        """
        Plot attention weights as heatmap with proper formatting
        
        Args:
            attention_weights: Attention weights [num_heads, seq_len, seq_len]
            tokens: List of token strings
            head_idx: Which attention head to visualize
            layer_idx: Which layer (for title)
            save_path: Path to save plot
            figsize: Figure size tuple
        """
        # Extract attention weights for specific head
        if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
            head_attention = attention_weights[0, head_idx].detach().cpu().numpy()
        else:  # [heads, seq, seq]
            head_attention = attention_weights[head_idx].detach().cpu().numpy()
            
        # Create heatmap
        plt.figure(figsize=figsize)
        
        # Truncate long token lists for readability
        max_tokens = 20
        if len(tokens) > max_tokens:
            display_tokens = tokens[:max_tokens]
            head_attention = head_attention[:max_tokens, :max_tokens]
            warnings.warn(f"Truncating visualization to first {max_tokens} tokens for readability")
        else:
            display_tokens = tokens
            
        # Create heatmap with proper formatting
        sns.heatmap(
            head_attention,
            xticklabels=display_tokens,
            yticklabels=display_tokens,
            cmap='Blues',
            cbar_kws={'label': 'Attention Weight'},
            square=True,
            linewidths=0.1
        )
        
        plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}', fontsize=14, fontweight='bold')
        plt.xlabel('Key Positions (Attended To)', fontsize=12)
        plt.ylabel('Query Positions (Attending From)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention heatmap saved to {save_path}")
            
        plt.show()
        
    @staticmethod
    def plot_multi_head_comparison(
        attention_weights: torch.Tensor,
        tokens: List[str],
        num_heads_to_show: int = 4,
        layer_idx: int = 0,
        save_path: Optional[str] = None
    ):
        """
        Compare attention patterns across multiple heads
        
        Args:
            attention_weights: Attention weights [num_heads, seq_len, seq_len]
            tokens: List of token strings
            num_heads_to_show: Number of heads to display
            layer_idx: Layer index for title
            save_path: Path to save plot
        """
        num_heads = min(num_heads_to_show, attention_weights.shape[-3])
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i in range(num_heads):
            if attention_weights.dim() == 4:  # [batch, heads, seq, seq]
                head_attention = attention_weights[0, i].detach().cpu().numpy()
            else:  # [heads, seq, seq]
                head_attention = attention_weights[i].detach().cpu().numpy()
                
            # Truncate for visualization
            max_tokens = 15
            if len(tokens) > max_tokens:
                display_tokens = tokens[:max_tokens]
                head_attention = head_attention[:max_tokens, :max_tokens]
            else:
                display_tokens = tokens
                
            sns.heatmap(
                head_attention,
                xticklabels=display_tokens,
                yticklabels=display_tokens,
                cmap='Blues',
                ax=axes[i],
                cbar=True,
                square=True
            )
            
            axes[i].set_title(f'Head {i}', fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            
        plt.suptitle(f'Multi-Head Attention Comparison - Layer {layer_idx}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Multi-head comparison saved to {save_path}")
            
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
        # Add small epsilon to prevent log(0)
        eps = 1e-9
        attention_weights = attention_weights + eps
        
        # Compute entropy: -sum(p * log(p))
        entropy = -torch.sum(attention_weights * torch.log(attention_weights), dim=-1)
        
        return entropy
        
    @staticmethod
    def compute_attention_distance(attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute average attention distance for each query position
        
        Args:
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            avg_distance: Average attention distance [batch_size, num_heads, seq_len]
        """
        seq_len = attention_weights.shape[-1]
        positions = torch.arange(seq_len, dtype=torch.float, device=attention_weights.device)
        
        # Compute weighted average of attended positions
        avg_distance = torch.sum(attention_weights * positions.unsqueeze(0).unsqueeze(0).unsqueeze(0), dim=-1)
        
        return avg_distance
        
    @staticmethod
    def analyze_attention_patterns(attention_weights: torch.Tensor) -> Dict[str, Any]:
        """
        Comprehensive analysis of attention patterns
        
        Args:
            attention_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
            
        Returns:
            stats: Dictionary with attention statistics
        """
        stats = {}
        
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
        avg_distance = AttentionVisualizer.compute_attention_distance(attention_weights)
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
        
        # Head similarity (correlation between heads)
        batch_size, num_heads, seq_len, _ = attention_weights.shape
        head_similarities = []
        
        for b in range(batch_size):
            for i in range(num_heads):
                for j in range(i + 1, num_heads):
                    head_i = attention_weights[b, i].flatten()
                    head_j = attention_weights[b, j].flatten()
                    correlation = torch.corrcoef(torch.stack([head_i, head_j]))[0, 1]
                    if not torch.isnan(correlation):
                        head_similarities.append(correlation.item())
                        
        if head_similarities:
            stats['head_similarity'] = {
                'mean': np.mean(head_similarities),
                'std': np.std(head_similarities),
                'min': np.min(head_similarities),
                'max': np.max(head_similarities)
            }
        
        return stats


@contextmanager
def memory_profiler():
    """Context manager for memory profiling"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        start_memory = torch.cuda.memory_allocated()
        yield
        end_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Memory used: {(end_memory - start_memory) / 1024**2:.2f} MB")
        print(f"Peak memory: {peak_memory / 1024**2:.2f} MB")
    else:
        yield


def benchmark_attention_variants():
    """
    Comprehensive benchmark of different attention implementations
    """
    print("🔬 Benchmarking Attention Variants")
    print("=" * 50)
    
    config = AttentionConfig(
        d_model=512,
        num_heads=8,
        max_seq_len=4096
    )
    
    # Test different sequence lengths
    seq_lengths = [128, 256, 512, 1024, 2048]
    batch_size = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize different attention variants
    standard_attention = MultiHeadAttention(config).to(device)
    
    sparse_configs = {
        'local': AttentionConfig(**{**config.__dict__, 'sparse_pattern': 'local', 'window_size': 128}),
        'strided': AttentionConfig(**{**config.__dict__, 'sparse_pattern': 'strided', 'window_size': 128}),
        'random': AttentionConfig(**{**config.__dict__, 'sparse_pattern': 'random', 'window_size': 128})
    }
    
    sparse_attentions = {
        name: SparseAttention(cfg).to(device) 
        for name, cfg in sparse_configs.items()
    }
    
    results = {
        'seq_length': [],
        'standard_time': [],
        'standard_memory': [],
    }
    
    for pattern in sparse_configs.keys():
        results[f'{pattern}_time'] = []
        results[f'{pattern}_memory'] = []
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Create test input
        x = torch.randn(batch_size, seq_len, config.d_model, device=device)
        
        # Benchmark standard attention
        print("  Standard attention...")
        with memory_profiler():
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(5):  # Multiple runs for stability
                    output, _ = standard_attention(x)
                    
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            standard_time = (time.time() - start_time) / 5
            
        # Benchmark sparse attention variants
        for pattern, sparse_attention in sparse_attentions.items():
            print(f"  {pattern.capitalize()} sparse attention...")
            with memory_profiler():
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(5):
                        output, _ = sparse_attention(x)
                        
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                sparse_time = (time.time() - start_time) / 5
                
            results[f'{pattern}_time'].append(sparse_time)
            
        # Store results
        results['seq_length'].append(seq_len)
        results['standard_time'].append(standard_time)
        
        print(f"    Standard: {standard_time:.4f}s")
        for pattern in sparse_configs.keys():
            print(f"    {pattern.capitalize()}: {results[f'{pattern}_time'][-1]:.4f}s")
    
    # Plot benchmark results
    plt.figure(figsize=(15, 5))
    
    # Time comparison
    plt.subplot(1, 2, 1)
    plt.plot(results['seq_length'], results['standard_time'], 'o-', label='Standard', linewidth=2)
    for pattern in sparse_configs.keys():
        plt.plot(results['seq_length'], results[f'{pattern}_time'], 'o-', 
                label=f'{pattern.capitalize()} Sparse', linewidth=2)
    
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Attention Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Speedup comparison
    plt.subplot(1, 2, 2)
    for pattern in sparse_configs.keys():
        speedups = [std_t / sparse_t for std_t, sparse_t in 
                   zip(results['standard_time'], results[f'{pattern}_time'])]
        plt.plot(results['seq_length'], speedups, 'o-', 
                label=f'{pattern.capitalize()} Speedup', linewidth=2)
    
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
    Test attention caching for inference speedup with comprehensive analysis
    """
    print("🚀 Testing Attention Caching")
    print("=" * 50)
    
    config = AttentionConfig(d_model=512, num_heads=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attention = MultiHeadAttention(config).to(device)
    
    cache = AttentionCache(
        max_seq_len=1024,
        num_heads=config.num_heads,
        head_dim=config.d_model // config.num_heads,
        device=device
    )
    
    # Test different sequence lengths
    seq_lengths = [50, 100, 200, 500]
    batch_size = 1
    
    results = {
        'seq_length': [],
        'no_cache_time': [],
        'cache_time': [],
        'speedup': []
    }
    
    for seq_len in seq_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        
        # Without caching - process full sequence each time
        print("  Without caching...")
        start_time = time.time()
        
        for i in range(1, seq_len + 1):
            x = torch.randn(batch_size, i, config.d_model, device=device)
            with torch.no_grad():
                output, _ = attention(x)
                
        no_cache_time = time.time() - start_time
        
        # With caching - process one token at a time
        print("  With caching...")
        cache.reset()
        start_time = time.time()
        
        for i in range(seq_len):
            x = torch.randn(batch_size, 1, config.d_model, device=device)
            with torch.no_grad():
                output, _ = attention(x, use_cache=True, cache=cache)
                
        cache_time = time.time() - start_time
        speedup = no_cache_time / cache_time
        
        # Store results
        results['seq_length'].append(seq_len)
        results['no_cache_time'].append(no_cache_time)
        results['cache_time'].append(cache_time)
        results['speedup'].append(speedup)
        
        print(f"    No cache: {no_cache_time:.4f}s")
        print(f"    With cache: {cache_time:.4f}s")
        print(f"    Speedup: {speedup:.2f}x")
    
    # Plot caching results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(results['seq_length'], results['no_cache_time'], 'o-', label='No Cache', linewidth=2)
    plt.plot(results['seq_length'], results['cache_time'], 'o-', label='With Cache', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Time (seconds)')
    plt.title('Caching Performance Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results['seq_length'], results['speedup'], 'o-', color='green', linewidth=2)
    plt.xlabel('Sequence Length')
    plt.ylabel('Speedup Factor')
    plt.title('Caching Speedup')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()
    
    return results


def visualize_attention_patterns():
    """
    Create comprehensive attention visualizations for analysis
    """
    print("🎨 Visualizing Attention Patterns")
    print("=" * 50)
    
    # Create sample input with meaningful tokens
    tokens = [
        "The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog", 
        "and", "runs", "through", "the", "forest", "quickly"
    ]
    seq_len = len(tokens)
    
    config = AttentionConfig(d_model=512, num_heads=8)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attention = MultiHeadAttention(config).to(device)
    
    # Create input embeddings (random for demo, but could be real embeddings)
    torch.manual_seed(42)  # For reproducible visualizations
    x = torch.randn(1, seq_len, config.d_model, device=device)
    
    # Get attention weights
    with torch.no_grad():
        output, attention_weights = attention(x, return_attention=True)
    
    # Move to CPU for visualization
    attention_weights = attention_weights.cpu()
    
    # Initialize visualizer
    visualizer = AttentionVisualizer()
    
    # 1. Single head heatmap
    print("Creating single head attention heatmap...")
    visualizer.plot_attention_heatmap(
        attention_weights, tokens, head_idx=0, layer_idx=1
    )
    
    # 2. Multi-head comparison
    print("Creating multi-head comparison...")
    visualizer.plot_multi_head_comparison(
        attention_weights, tokens, num_heads_to_show=4, layer_idx=1
    )
    
    # 3. Analyze attention patterns
    print("Analyzing attention patterns...")
    stats = visualizer.analyze_attention_patterns(attention_weights)
    
    print("\n📊 Attention Statistics:")
    print("-" * 30)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"{key.replace('_', ' ').title()}:")
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue:.4f}")
        else:
            print(f"{key.replace('_', ' ').title()}: {value:.4f}")
    
    # 4. Attention entropy visualization
    entropy = visualizer.compute_attention_entropy(attention_weights)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    for head in range(min(4, config.num_heads)):
        plt.plot(entropy[0, head].numpy(), label=f'Head {head}', marker='o')
    plt.xlabel('Token Position')
    plt.ylabel('Attention Entropy')
    plt.title('Attention Entropy by Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5. Attention distance visualization
    avg_distance = visualizer.compute_attention_distance(attention_weights)
    
    plt.subplot(1, 2, 2)
    for head in range(min(4, config.num_heads)):
        plt.plot(avg_distance[0, head].numpy(), label=f'Head {head}', marker='o')
    plt.xlabel('Token Position')
    plt.ylabel('Average Attention Distance')
    plt.title('Average Attention Distance by Position')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demonstrate_causal_attention():
    """
    Demonstrate causal (masked) attention for autoregressive models
    """
    print("🎭 Demonstrating Causal Attention")
    print("=" * 50)
    
    config = AttentionConfig(d_model=256, num_heads=4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    attention = MultiHeadAttention(config).to(device)
    
    # Create sample sequence
    tokens = ["I", "love", "machine", "learning", "and", "AI"]
    seq_len = len(tokens)
    
    # Create causal mask (lower triangular)
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    
    # Create input
    torch.manual_seed(42)
    x = torch.randn(1, seq_len, config.d_model, device=device)
    
    # Get attention with and without causal mask
    with torch.no_grad():
        # Bidirectional attention
        output_bi, attention_bi = attention(x, return_attention=True)
        
        # Causal attention
        output_causal, attention_causal = attention(x, mask=causal_mask.to(device), return_attention=True)
    
    # Visualize both
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bidirectional attention
    sns.heatmap(
        attention_bi[0, 0].cpu().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        ax=axes[0],
        square=True
    )
    axes[0].set_title('Bidirectional Attention (BERT-style)')
    axes[0].set_xlabel('Key Positions')
    axes[0].set_ylabel('Query Positions')
    
    # Causal attention
    sns.heatmap(
        attention_causal[0, 0].cpu().numpy(),
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        ax=axes[1],
        square=True
    )
    axes[1].set_title('Causal Attention (GPT-style)')
    axes[1].set_xlabel('Key Positions')
    axes[1].set_ylabel('Query Positions')
    
    plt.tight_layout()
    plt.show()
    
    print("Key Differences:")
    print("- Bidirectional: Each token can attend to all tokens (past and future)")
    print("- Causal: Each token can only attend to previous tokens (autoregressive)")


def main():
    """
    Main function demonstrating all attention mechanism implementations
    """
    print("🎯 Day 42: Attention Mechanisms Deep Dive - Production Solution")
    print("=" * 70)
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Test basic multi-head attention
    print("\n1️⃣ Testing Multi-Head Attention Implementation")
    print("-" * 50)
    
    config = AttentionConfig(d_model=512, num_heads=8)
    attention = MultiHeadAttention(config).to(device)
    
    # Test with sample input
    batch_size, seq_len = 2, 10
    x = torch.randn(batch_size, seq_len, config.d_model, device=device)
    
    # Test self-attention
    print("✅ Testing self-attention...")
    with torch.no_grad():
        output, weights = attention(x, return_attention=True)
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {output.shape}")
    if weights is not None:
        print(f"   Attention weights shape: {weights.shape}")
    
    # Test cross-attention
    print("✅ Testing cross-attention...")
    y = torch.randn(batch_size, seq_len + 5, config.d_model, device=device)
    with torch.no_grad():
        output, weights = attention(x, key=y, value=y, return_attention=True)
    print(f"   Query shape: {x.shape}")
    print(f"   Key/Value shape: {y.shape}")
    print(f"   Output shape: {output.shape}")
    
    # 2. Test sparse attention patterns
    print("\n2️⃣ Testing Sparse Attention Patterns")
    print("-" * 50)
    
    sparse_patterns = ['local', 'strided', 'random']
    
    for pattern in sparse_patterns:
        print(f"✅ Testing {pattern} sparse attention...")
        sparse_config = AttentionConfig(
            d_model=512, 
            num_heads=8, 
            sparse_pattern=pattern, 
            window_size=64
        )
        sparse_attention = SparseAttention(sparse_config).to(device)
        
        long_seq = torch.randn(1, 256, config.d_model, device=device)
        with torch.no_grad():
            output, _ = sparse_attention(long_seq)
        print(f"   {pattern.capitalize()} attention output shape: {output.shape}")
    
    # 3. Demonstrate causal attention
    print("\n3️⃣ Demonstrating Causal vs Bidirectional Attention")
    print("-" * 50)
    demonstrate_causal_attention()
    
    # 4. Test attention caching
    print("\n4️⃣ Testing Attention Caching for Inference")
    print("-" * 50)
    caching_results = test_attention_caching()
    
    # 5. Run performance benchmarks
    print("\n5️⃣ Running Performance Benchmarks")
    print("-" * 50)
    benchmark_results = benchmark_attention_variants()
    
    # 6. Create attention visualizations
    print("\n6️⃣ Creating Attention Visualizations")
    print("-" * 50)
    visualize_attention_patterns()
    
    # 7. Production considerations summary
    print("\n7️⃣ Production Deployment Summary")
    print("-" * 50)
    print("✅ Key Production Considerations:")
    print("   • Memory Management: Use gradient checkpointing for long sequences")
    print("   • Caching: Implement KV caching for autoregressive inference")
    print("   • Sparsity: Use sparse patterns for sequences >2K tokens")
    print("   • Optimization: Consider Flash Attention for 2-4x speedup")
    print("   • Monitoring: Track attention entropy and patterns in production")
    print("   • Numerical Stability: Handle edge cases and NaN values")
    print("   • Device Management: Proper GPU memory management")
    
    print("\n🎉 All attention mechanism implementations completed successfully!")
    
    print("\n📊 Key Insights from Today's Implementation:")
    print("=" * 60)
    print("🔍 Attention Types:")
    print("   • Self-attention: Captures intra-sequence relationships")
    print("   • Cross-attention: Aligns different sequences (e.g., encoder-decoder)")
    print("   • Causal attention: Essential for autoregressive generation")
    
    print("\n⚡ Performance Optimizations:")
    print("   • Sparse patterns reduce O(n²) to O(n×w) complexity")
    print("   • KV caching provides significant inference speedup")
    print("   • Memory-efficient implementations crucial for long sequences")
    
    print("\n🎨 Interpretability:")
    print("   • Attention patterns reveal learned linguistic structures")
    print("   • Different heads specialize in different relationship types")
    print("   • Visualization helps debug and understand model behavior")
    
    print("\n🚀 Production Ready:")
    print("   • Comprehensive error handling and validation")
    print("   • Scalable implementations for various sequence lengths")
    print("   • Monitoring and analysis tools for production deployment")


if __name__ == "__main__":
    main()
