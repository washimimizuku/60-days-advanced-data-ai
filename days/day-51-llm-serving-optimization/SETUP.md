# Day 51: LLM Serving & Optimization Setup Guide

## Overview
This guide helps you set up the environment for Day 51: LLM Serving & Optimization - vLLM, TensorRT, Inference. You'll learn to implement high-performance LLM serving with advanced optimization techniques.

## Prerequisites
- Completed Days 1-50 of the bootcamp
- Python 3.8+ installed
- CUDA-capable GPU (recommended: A100, H100, or V100)
- Basic understanding of LLM inference and serving
- Familiarity with async programming concepts

## Installation

### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv llm_serving_env

# Activate environment
# On macOS/Linux:
source llm_serving_env/bin/activate
# On Windows:
llm_serving_env\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install core requirements
pip install -r requirements.txt

# For GPU support (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install vLLM (requires CUDA)
pip install vllm

# For TensorRT (if available)
pip install tensorrt tensorrt-llm

# For development
pip install pytest pytest-asyncio black flake8
```

### 3. Verify Installation
```bash
# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test vLLM installation
python -c "import vllm; print('vLLM installed successfully')"

# Test async support
python -c "import asyncio; print('Asyncio available')"
```

## LLM Serving Fundamentals

### Understanding LLM Inference Phases

#### 1. Prefill Phase
- **Characteristics**: Compute-bound, parallelizable
- **Process**: Processes input prompt tokens in parallel
- **Optimization**: Focus on throughput and batch processing

#### 2. Decode Phase  
- **Characteristics**: Memory-bound, sequential
- **Process**: Generates output tokens one by one
- **Optimization**: Focus on KV-cache efficiency and latency

### Key Performance Metrics

#### Time to First Token (TTFT)
```python
# Measure TTFT
import time

start_time = time.time()
# Process input prompt
first_token_time = time.time()
ttft = (first_token_time - start_time) * 1000  # ms
```

#### Time Per Output Token (TPOT)
```python
# Measure TPOT
token_times = []
for token in output_tokens:
    token_start = time.time()
    # Generate token
    token_end = time.time()
    token_times.append((token_end - token_start) * 1000)

tpot = sum(token_times) / len(token_times)  # Average ms per token
```

## vLLM Setup and Configuration

### Basic vLLM Setup
```python
from vllm import LLM, SamplingParams

# Initialize vLLM engine
llm = LLM(
    model="microsoft/DialoGPT-medium",  # Start with smaller model
    tensor_parallel_size=1,
    dtype="float16",
    max_model_len=2048,
    gpu_memory_utilization=0.8
)

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.9,
    max_tokens=256
)

# Generate text
prompts = ["Hello, how are you?"]
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Generated: {output.outputs[0].text}")
```

### Advanced vLLM Configuration
```python
# Production configuration
production_config = {
    # Model settings
    "model": "your-model-path",
    "tensor_parallel_size": 2,  # Multi-GPU
    "dtype": "float16",
    
    # Memory management
    "gpu_memory_utilization": 0.85,
    "max_model_len": 4096,
    "block_size": 16,  # PagedAttention block size
    
    # Performance tuning
    "max_num_seqs": 256,
    "max_num_batched_tokens": 8192,
    "enforce_eager": False,  # Enable CUDA graphs
    
    # Quantization (if supported)
    "quantization": "awq",  # or "gptq"
}
```

### PagedAttention Benefits
- **Memory Efficiency**: Up to 4x reduction in memory waste
- **Dynamic Allocation**: Non-contiguous memory blocks
- **Reduced Fragmentation**: Better memory utilization
- **Flexible Batching**: Support for variable sequence lengths

## TensorRT Optimization

### TensorRT-LLM Setup
```bash
# Install TensorRT-LLM (requires NVIDIA drivers)
pip install tensorrt-llm

# Build TensorRT engine (example)
python build.py \
    --model_dir ./model \
    --dtype float16 \
    --use_gpt_attention_plugin \
    --use_gemm_plugin \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_output_len 512
```

### Optimization Strategies
```python
# TensorRT optimization configuration
tensorrt_config = {
    # Precision settings
    "precision": "fp16",  # or "int8" for more aggressive optimization
    
    # Batch configuration
    "max_batch_size": 8,
    "opt_batch_size": 4,
    
    # Sequence length settings
    "max_input_len": 2048,
    "max_output_len": 512,
    
    # Performance optimizations
    "use_cuda_graphs": True,
    "use_gpt_attention_plugin": True,
    "use_gemm_plugin": True,
    "enable_context_fmha": True,  # Flash Attention
    
    # Memory optimization
    "use_paged_kv_cache": True,
    "kv_cache_free_gpu_mem_fraction": 0.9
}
```

## KV-Cache Optimization

### Understanding KV-Cache Memory
```python
def calculate_kv_cache_memory(batch_size, seq_len, num_layers, hidden_size):
    """Calculate KV-cache memory requirements"""
    
    # Each layer stores key and value
    # Shape: [batch_size, num_heads, seq_len, head_dim]
    bytes_per_element = 2  # FP16
    
    kv_cache_size = (
        2 *  # key and value
        num_layers *
        batch_size *
        seq_len *
        hidden_size *
        bytes_per_element
    )
    
    return {
        'size_bytes': kv_cache_size,
        'size_mb': kv_cache_size / (1024**2),
        'size_gb': kv_cache_size / (1024**3)
    }

# Example calculation
memory_req = calculate_kv_cache_memory(
    batch_size=8,
    seq_len=2048,
    num_layers=32,
    hidden_size=4096
)
print(f"KV-cache memory: {memory_req['size_gb']:.2f} GB")
```

### PagedAttention Implementation
```python
class PagedAttentionConfig:
    def __init__(self):
        self.block_size = 16  # Tokens per block
        self.max_blocks = 1000  # Total memory blocks
        self.block_tables = {}  # Logical to physical mapping
        
    def allocate_blocks(self, sequence_id, num_blocks):
        """Allocate memory blocks for sequence"""
        # Implementation details in solution.py
        pass
        
    def deallocate_blocks(self, sequence_id):
        """Free memory blocks for completed sequence"""
        # Implementation details in solution.py
        pass
```

## Continuous Batching

### Basic Continuous Batching
```python
import asyncio
from typing import List, Dict

class SimpleContinuousBatching:
    def __init__(self, max_batch_size=32):
        self.max_batch_size = max_batch_size
        self.active_requests = {}
        self.request_queue = asyncio.Queue()
    
    async def add_request(self, request_id, prompt, max_tokens):
        """Add new request to processing queue"""
        request = {
            'id': request_id,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'generated_tokens': 0,
            'status': 'queued'
        }
        await self.request_queue.put(request)
    
    async def process_batch(self):
        """Main processing loop"""
        while True:
            # Form dynamic batch
            batch = await self.form_batch()
            
            if batch:
                # Process generation step
                completed = await self.generation_step(batch)
                
                # Handle completed requests
                for request in completed:
                    print(f"Completed: {request['id']}")
            
            await asyncio.sleep(0.01)  # Small delay
```

### Benefits of Continuous Batching
- **Higher Throughput**: Better GPU utilization
- **Lower Latency**: Requests don't wait for entire batch
- **Dynamic Scaling**: Adapts to varying request patterns
- **Memory Efficiency**: Optimal batch formation

## Speculative Decoding

### Basic Speculative Decoding Setup
```python
class SpeculativeDecoding:
    def __init__(self, target_model, draft_model):
        self.target_model = target_model  # Large, accurate model
        self.draft_model = draft_model    # Small, fast model
    
    def generate_with_speculation(self, input_ids, num_candidates=4):
        """Generate tokens using speculative decoding"""
        
        # Step 1: Draft model generates candidates quickly
        candidates = self.draft_model.generate(
            input_ids, 
            max_new_tokens=num_candidates,
            do_sample=True
        )
        
        # Step 2: Target model verifies candidates in parallel
        verification_input = torch.cat([input_ids, candidates], dim=-1)
        target_logits = self.target_model(verification_input).logits
        
        # Step 3: Accept/reject based on probability ratios
        accepted_tokens = self.verify_candidates(candidates, target_logits)
        
        return accepted_tokens
```

### Optimization Tips
- **Draft Model Selection**: 5-10x smaller than target model
- **Candidate Count**: Typically 2-6 tokens for optimal speedup
- **Acceptance Rate**: Aim for 60-80% acceptance rate
- **Temperature Tuning**: Lower temperature for better acceptance

## Production Deployment

### Load Balancing Setup
```python
from fastapi import FastAPI
import httpx

app = FastAPI()

class LLMLoadBalancer:
    def __init__(self):
        self.instances = [
            {"url": "http://llm-1:8000", "load": 0},
            {"url": "http://llm-2:8000", "load": 0},
            {"url": "http://llm-3:8000", "load": 0}
        ]
    
    def select_instance(self):
        """Select least loaded instance"""
        return min(self.instances, key=lambda x: x["load"])

load_balancer = LLMLoadBalancer()

@app.post("/generate")
async def generate_text(request: dict):
    # Route to best instance
    instance = load_balancer.select_instance()
    
    # Forward request
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{instance['url']}/generate",
            json=request
        )
    
    return response.json()
```

### Monitoring Setup
```python
import psutil
import GPUtil

class PerformanceMonitor:
    def __init__(self):
        self.metrics = []
    
    def collect_metrics(self):
        """Collect system and GPU metrics"""
        
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpus = GPUtil.getGPUs()
        gpu_metrics = []
        
        for gpu in gpus:
            gpu_metrics.append({
                'id': gpu.id,
                'utilization': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            })
        
        metrics = {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'gpu_metrics': gpu_metrics
        }
        
        self.metrics.append(metrics)
        return metrics
```

## Performance Optimization Tips

### 1. Memory Optimization
```python
# Optimize GPU memory usage
optimization_tips = {
    'gpu_memory_utilization': 0.85,  # Leave some headroom
    'use_fp16': True,  # Half precision
    'gradient_checkpointing': True,  # Trade compute for memory
    'max_batch_size': 'auto',  # Let vLLM optimize
}
```

### 2. Latency Optimization
```python
# Minimize latency
latency_config = {
    'batch_size': 1,  # Single request processing
    'use_cuda_graphs': True,  # Reduce kernel launch overhead
    'enforce_eager': False,  # Enable optimizations
    'preemption_mode': 'swap',  # Handle memory pressure
}
```

### 3. Throughput Optimization
```python
# Maximize throughput
throughput_config = {
    'max_batch_size': 64,  # Large batches
    'continuous_batching': True,  # Dynamic batching
    'tensor_parallel_size': 4,  # Multi-GPU
    'pipeline_parallel_size': 2,  # Pipeline parallelism
}
```

## Common Issues and Solutions

### Issue 1: Out of Memory (OOM)
**Symptoms**: CUDA OOM errors during inference
**Solutions**:
```python
# Reduce memory usage
config = {
    'gpu_memory_utilization': 0.7,  # Reduce from 0.9
    'max_model_len': 2048,  # Reduce from 4096
    'max_num_seqs': 64,  # Reduce batch size
    'block_size': 8,  # Smaller blocks
}
```

### Issue 2: Low Throughput
**Symptoms**: Low requests per second
**Solutions**:
```python
# Increase throughput
config = {
    'max_num_seqs': 256,  # Increase batch size
    'max_num_batched_tokens': 16384,  # More tokens per batch
    'continuous_batching': True,  # Enable dynamic batching
}
```

### Issue 3: High Latency
**Symptoms**: Slow response times
**Solutions**:
```python
# Reduce latency
config = {
    'enforce_eager': False,  # Enable CUDA graphs
    'use_v2_block_manager': True,  # Faster block management
    'preemption_mode': 'recompute',  # Faster preemption
}
```

### Issue 4: Model Loading Errors
**Symptoms**: Failed to load model
**Solutions**:
```bash
# Check model format and path
ls -la /path/to/model/

# Verify model compatibility
python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('model-path')"

# Check available GPU memory
nvidia-smi
```

## Testing Your Setup

### Basic Functionality Test
```python
# test_setup.py
import asyncio
from solution import vLLMServingEngine

async def test_basic_serving():
    """Test basic vLLM serving functionality"""
    
    config = {
        'tensor_parallel_size': 1,
        'max_model_len': 1024,
        'gpu_memory_utilization': 0.7
    }
    
    engine = vLLMServingEngine("gpt2", config)  # Use small model for testing
    engine.initialize_engine()
    
    # Test generation
    prompts = ["Hello, how are you?"]
    sampling_params = engine.create_sampling_params(max_tokens=50)
    
    results = await engine.generate_async(prompts, sampling_params)
    
    print(f"Generated: {results[0]}")
    print("✓ Basic serving test passed!")

if __name__ == "__main__":
    asyncio.run(test_basic_serving())
```

### Performance Benchmark
```python
# benchmark.py
import time
import numpy as np
from solution import vLLMServingEngine

def benchmark_performance():
    """Benchmark serving performance"""
    
    config = {'tensor_parallel_size': 1, 'max_model_len': 1024}
    engine = vLLMServingEngine("gpt2", config)
    
    # Benchmark with different batch sizes
    batch_sizes = [1, 4, 8, 16]
    
    for batch_size in batch_sizes:
        prompts = [f"Test prompt {i}" for i in range(batch_size)]
        
        # Measure performance
        results = engine.benchmark_performance(prompts, num_runs=5)
        
        print(f"Batch size {batch_size}:")
        print(f"  TTFT: {results['avg_ttft_ms']:.1f}ms")
        print(f"  Throughput: {results['avg_throughput_tps']:.1f} TPS")

if __name__ == "__main__":
    benchmark_performance()
```

## Performance Expectations

### Typical Performance Metrics

#### 7B Model (A100 40GB)
- **TTFT**: 20-50ms
- **TPOT**: 15-25ms  
- **Throughput**: 100-200 tokens/sec
- **Memory Usage**: 14-20GB

#### 13B Model (A100 80GB)
- **TTFT**: 30-70ms
- **TPOT**: 25-40ms
- **Throughput**: 60-120 tokens/sec  
- **Memory Usage**: 26-35GB

#### 70B Model (4x A100 80GB)
- **TTFT**: 50-100ms
- **TPOT**: 40-80ms
- **Throughput**: 30-80 tokens/sec
- **Memory Usage**: 140-200GB

### Optimization Impact
- **vLLM vs Transformers**: 2-4x throughput improvement
- **TensorRT optimization**: 1.5-3x speedup
- **Continuous batching**: 2-5x throughput increase
- **Speculative decoding**: 1.5-2.5x speedup
- **Quantization (INT8)**: 1.5-2x speedup, 50% memory reduction

## Next Steps

1. **Complete the exercises** in `exercise.py`
2. **Run the test suite** to verify implementations
3. **Experiment with different models** and configurations
4. **Deploy serving infrastructure** with monitoring
5. **Optimize for your specific use case** and hardware

## Resources

### Documentation
- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Continuous Batching Blog](https://www.anyscale.com/blog/continuous-batching-llm-inference)

### Tools and Frameworks
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [Ray Serve](https://docs.ray.io/en/latest/serve/)
- [Triton Inference Server](https://github.com/triton-inference-server/server)

### Hardware Guides
- [NVIDIA A100 Optimization](https://docs.nvidia.com/deeplearning/performance/index.html)
- [Multi-GPU Serving](https://docs.vllm.ai/en/latest/getting_started/distributed_serving.html)
- [Memory Optimization](https://huggingface.co/docs/transformers/perf_infer_gpu_one)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review GPU memory and CUDA setup
3. Consult the official documentation
4. Ask in the course community forums

Happy serving! 🚀