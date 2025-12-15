# Day 51: LLM Serving & Optimization - vLLM, TensorRT, Inference

## Learning Objectives
By the end of this session, you will be able to:
- Understand LLM serving challenges and optimization strategies
- Implement high-performance inference with vLLM and TensorRT
- Apply advanced optimization techniques: KV-cache, continuous batching, speculative decoding
- Deploy scalable LLM serving infrastructure with load balancing and auto-scaling
- Monitor and optimize LLM serving performance in production
- Build cost-effective serving solutions for different use cases

## Theory (30 minutes)

### LLM Serving Challenges

Large Language Models present unique serving challenges that differ significantly from traditional ML models:

**Key Challenges:**
- **Memory Requirements**: Models can be 7B-175B+ parameters (14GB-350GB+ memory)
- **Sequential Generation**: Autoregressive nature requires multiple forward passes
- **Variable Input/Output Lengths**: Unpredictable compute and memory requirements
- **Latency Sensitivity**: Real-time applications require low time-to-first-token (TTFT)
- **Throughput Demands**: High concurrent user loads
- **Cost Optimization**: GPU costs can be $1000s/month per instance

### LLM Inference Fundamentals

#### Autoregressive Generation Process

```python
# Simplified LLM inference process
def generate_text(model, tokenizer, prompt, max_length=100):
    """
    Autoregressive text generation process
    """
    # Tokenize input
    input_ids = tokenizer.encode(prompt)
    
    # Initialize generation
    generated_ids = input_ids.copy()
    
    for step in range(max_length):
        # Forward pass through model
        with torch.no_grad():
            outputs = model(torch.tensor([generated_ids]))
            logits = outputs.logits[0, -1, :]  # Last token logits
        
        # Sample next token
        next_token_id = torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        
        # Add to sequence
        generated_ids.append(next_token_id)
        
        # Check for end token
        if next_token_id == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated_ids)

# Performance characteristics
inference_metrics = {
    'prefill_phase': {
        'description': 'Processing input prompt',
        'compute_bound': True,
        'parallelizable': True,
        'memory_pattern': 'Linear with sequence length'
    },
    'decode_phase': {
        'description': 'Generating output tokens',
        'memory_bound': True,
        'sequential': True,
        'memory_pattern': 'Constant per token'
    }
}
```

#### Memory and Compute Patterns

```python
class LLMInferenceAnalyzer:
    """Analyze LLM inference patterns"""
    
    def __init__(self, model_size_gb, sequence_length):
        self.model_size_gb = model_size_gb
        self.sequence_length = sequence_length
        
    def estimate_memory_requirements(self):
        """Estimate memory requirements for inference"""
        
        # Model weights
        model_memory = self.model_size_gb
        
        # KV-cache memory (key-value pairs for attention)
        # Approximation: 2 * num_layers * hidden_size * sequence_length * batch_size
        kv_cache_per_token = self.model_size_gb * 0.1  # Rough estimate
        kv_cache_memory = kv_cache_per_token * self.sequence_length
        
        # Activation memory
        activation_memory = self.model_size_gb * 0.2  # Rough estimate
        
        total_memory = model_memory + kv_cache_memory + activation_memory
        
        return {
            'model_weights_gb': model_memory,
            'kv_cache_gb': kv_cache_memory,
            'activations_gb': activation_memory,
            'total_memory_gb': total_memory,
            'recommended_gpu_memory_gb': total_memory * 1.2  # 20% buffer
        }
    
    def estimate_latency_components(self, batch_size=1):
        """Estimate latency components"""
        
        # Time to first token (TTFT) - prefill phase
        ttft_ms = self.sequence_length * 0.1  # Simplified estimate
        
        # Time per output token (TPOT) - decode phase
        tpot_ms = 10 + (batch_size * 2)  # Increases with batch size
        
        # Total latency for 100 token generation
        output_tokens = 100
        total_latency_ms = ttft_ms + (output_tokens * tpot_ms)
        
        return {
            'time_to_first_token_ms': ttft_ms,
            'time_per_output_token_ms': tpot_ms,
            'total_latency_100_tokens_ms': total_latency_ms,
            'throughput_tokens_per_second': 1000 / tpot_ms
        }
```

### vLLM: High-Performance LLM Serving

vLLM is a fast and easy-to-use library for LLM inference and serving, featuring:

- **PagedAttention**: Efficient memory management for KV-cache
- **Continuous Batching**: Dynamic batching for improved throughput
- **Optimized CUDA Kernels**: Custom kernels for attention computation
- **Multiple Sampling Methods**: Support for various decoding strategies

#### vLLM Architecture and Implementation

```python
# vLLM serving implementation
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
import asyncio
from typing import List, Dict, Any

class vLLMServer:
    """High-performance LLM serving with vLLM"""
    
    def __init__(self, model_name: str, tensor_parallel_size: int = 1):
        self.model_name = model_name
        self.tensor_parallel_size = tensor_parallel_size
        self.engine = None
        
    async def initialize_engine(self):
        """Initialize vLLM async engine"""
        
        engine_args = AsyncEngineArgs(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            dtype="float16",  # Use FP16 for memory efficiency
            max_model_len=4096,  # Maximum sequence length
            gpu_memory_utilization=0.9,  # Use 90% of GPU memory
            enforce_eager=False,  # Enable CUDA graphs
            disable_log_stats=False,  # Enable performance logging
        )
        
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print(f"vLLM engine initialized for {self.model_name}")
    
    async def generate_text(self, prompts: List[str], 
                           sampling_params: SamplingParams) -> List[str]:
        """Generate text using vLLM engine"""
        
        if not self.engine:
            await self.initialize_engine()
        
        # Add requests to engine
        request_ids = []
        for i, prompt in enumerate(prompts):
            request_id = f"request_{i}"
            await self.engine.add_request(request_id, prompt, sampling_params)
            request_ids.append(request_id)
        
        # Collect results
        results = []
        completed_requests = set()
        
        while len(completed_requests) < len(request_ids):
            # Process engine step
            request_outputs = await self.engine.step_async()
            
            for request_output in request_outputs:
                if request_output.finished:
                    completed_requests.add(request_output.request_id)
                    # Extract generated text
                    generated_text = request_output.outputs[0].text
                    results.append(generated_text)
        
        return results
    
    def create_sampling_params(self, **kwargs) -> SamplingParams:
        """Create sampling parameters for generation"""
        
        default_params = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'max_tokens': 256,
            'repetition_penalty': 1.1,
            'stop': ['</s>', '<|endoftext|>']
        }
        
        # Override with provided parameters
        default_params.update(kwargs)
        
        return SamplingParams(**default_params)

# Advanced vLLM configuration
class AdvancedvLLMConfig:
    """Advanced vLLM configuration for production"""
    
    @staticmethod
    def get_production_config(model_size: str, hardware: str) -> Dict[str, Any]:
        """Get production configuration based on model size and hardware"""
        
        configs = {
            '7b': {
                'a100_40gb': {
                    'tensor_parallel_size': 1,
                    'gpu_memory_utilization': 0.85,
                    'max_model_len': 4096,
                    'max_num_seqs': 256,
                    'max_num_batched_tokens': 8192
                },
                'a100_80gb': {
                    'tensor_parallel_size': 1,
                    'gpu_memory_utilization': 0.9,
                    'max_model_len': 8192,
                    'max_num_seqs': 512,
                    'max_num_batched_tokens': 16384
                }
            },
            '13b': {
                'a100_40gb': {
                    'tensor_parallel_size': 2,
                    'gpu_memory_utilization': 0.85,
                    'max_model_len': 2048,
                    'max_num_seqs': 128,
                    'max_num_batched_tokens': 4096
                },
                'a100_80gb': {
                    'tensor_parallel_size': 1,
                    'gpu_memory_utilization': 0.9,
                    'max_model_len': 4096,
                    'max_num_seqs': 256,
                    'max_num_batched_tokens': 8192
                }
            },
            '70b': {
                'a100_80gb': {
                    'tensor_parallel_size': 4,
                    'gpu_memory_utilization': 0.9,
                    'max_model_len': 2048,
                    'max_num_seqs': 64,
                    'max_num_batched_tokens': 2048
                }
            }
        }
        
        return configs.get(model_size, {}).get(hardware, {})
```

### TensorRT Optimization

NVIDIA TensorRT provides deep learning inference optimization through:

- **Layer Fusion**: Combines operations to reduce memory bandwidth
- **Precision Calibration**: INT8 and FP16 optimizations
- **Kernel Auto-tuning**: Selects optimal kernels for hardware
- **Dynamic Shapes**: Optimizes for variable input sizes

#### TensorRT-LLM Implementation

```python
import tensorrt as trt
import tensorrt_llm
from tensorrt_llm.runtime import ModelRunner
from tensorrt_llm.builder import Builder
import numpy as np

class TensorRTLLMOptimizer:
    """TensorRT optimization for LLM inference"""
    
    def __init__(self, model_path: str, max_batch_size: int = 8):
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        self.engine = None
        
    def build_tensorrt_engine(self, precision: str = 'fp16'):
        """Build TensorRT engine from model"""
        
        # TensorRT-LLM builder configuration
        builder_config = {
            'max_batch_size': self.max_batch_size,
            'max_input_len': 2048,
            'max_output_len': 512,
            'max_beam_width': 1,
            'precision': precision,
            'use_gpt_attention_plugin': True,
            'use_gemm_plugin': True,
            'use_layernorm_plugin': True,
            'enable_context_fmha': True,  # Flash Attention
            'enable_remove_input_padding': True
        }
        
        print(f"Building TensorRT engine with {precision} precision...")
        
        # Mock engine building process (actual implementation would use TensorRT-LLM APIs)
        engine_info = {
            'precision': precision,
            'max_batch_size': self.max_batch_size,
            'optimization_level': 5,
            'memory_pool_size_gb': 2,
            'estimated_speedup': '2-4x',
            'build_time_minutes': 15
        }
        
        print(f"TensorRT engine built successfully: {engine_info}")
        return engine_info
    
    def optimize_for_deployment(self, target_latency_ms: int = 100):
        """Optimize engine for specific deployment requirements"""
        
        optimization_strategies = {
            'ultra_low_latency': {
                'target_latency_ms': 50,
                'batch_size': 1,
                'precision': 'fp16',
                'use_cuda_graphs': True,
                'kv_cache_optimization': 'aggressive'
            },
            'balanced': {
                'target_latency_ms': 100,
                'batch_size': 4,
                'precision': 'fp16',
                'use_cuda_graphs': True,
                'kv_cache_optimization': 'moderate'
            },
            'high_throughput': {
                'target_latency_ms': 200,
                'batch_size': 16,
                'precision': 'int8',
                'use_cuda_graphs': False,
                'kv_cache_optimization': 'memory_efficient'
            }
        }
        
        # Select strategy based on target latency
        if target_latency_ms <= 50:
            strategy = optimization_strategies['ultra_low_latency']
        elif target_latency_ms <= 100:
            strategy = optimization_strategies['balanced']
        else:
            strategy = optimization_strategies['high_throughput']
        
        print(f"Selected optimization strategy: {strategy}")
        return strategy

class TensorRTInferenceEngine:
    """TensorRT inference engine for LLM"""
    
    def __init__(self, engine_path: str):
        self.engine_path = engine_path
        self.runner = None
        
    def load_engine(self):
        """Load TensorRT engine"""
        
        # Mock engine loading (actual implementation would load TensorRT engine)
        engine_config = {
            'engine_path': self.engine_path,
            'max_batch_size': 8,
            'max_input_len': 2048,
            'max_output_len': 512,
            'precision': 'fp16',
            'memory_usage_gb': 12
        }
        
        print(f"TensorRT engine loaded: {engine_config}")
        self.runner = engine_config  # Mock runner
        
    def generate_batch(self, input_texts: List[str], 
                      generation_config: Dict[str, Any]) -> List[str]:
        """Generate text for batch of inputs"""
        
        if not self.runner:
            self.load_engine()
        
        # Mock batch generation
        batch_size = len(input_texts)
        
        # Simulate optimized inference
        inference_stats = {
            'batch_size': batch_size,
            'avg_input_length': np.mean([len(text.split()) for text in input_texts]),
            'estimated_latency_ms': batch_size * 20,  # 20ms per sample
            'memory_usage_gb': 8 + (batch_size * 0.5),
            'throughput_tokens_per_second': 150 * batch_size
        }
        
        # Generate mock outputs
        outputs = [f"Generated response for: {text[:50]}..." for text in input_texts]
        
        print(f"Batch inference completed: {inference_stats}")
        return outputs
```

### Advanced Optimization Techniques

#### 1. KV-Cache Optimization

```python
class KVCacheOptimizer:
    """Optimize Key-Value cache for attention computation"""
    
    def __init__(self, max_sequence_length: int, num_layers: int, hidden_size: int):
        self.max_sequence_length = max_sequence_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
    def calculate_kv_cache_size(self, batch_size: int, sequence_length: int) -> Dict[str, float]:
        """Calculate KV-cache memory requirements"""
        
        # Each layer has key and value caches
        # Shape: [batch_size, num_heads, sequence_length, head_dim]
        bytes_per_element = 2  # FP16
        
        kv_cache_size_bytes = (
            2 *  # key and value
            self.num_layers *
            batch_size *
            sequence_length *
            self.hidden_size *
            bytes_per_element
        )
        
        return {
            'kv_cache_size_mb': kv_cache_size_bytes / (1024 ** 2),
            'kv_cache_size_gb': kv_cache_size_bytes / (1024 ** 3),
            'memory_per_token_kb': (kv_cache_size_bytes / sequence_length) / 1024,
            'max_batch_size_8gb': int(8 * 1024 ** 3 / (kv_cache_size_bytes / batch_size))
        }
    
    def implement_paged_attention(self, block_size: int = 16):
        """Implement PagedAttention for efficient KV-cache management"""
        
        paged_attention_config = {
            'block_size': block_size,
            'memory_efficiency': 'Up to 4x reduction in memory waste',
            'benefits': [
                'Non-contiguous memory allocation',
                'Dynamic memory management',
                'Reduced memory fragmentation',
                'Better GPU utilization'
            ],
            'implementation': {
                'logical_blocks': 'Sequence divided into fixed-size blocks',
                'physical_blocks': 'GPU memory allocated as needed',
                'block_mapping': 'Logical to physical block mapping table',
                'attention_computation': 'Block-wise attention computation'
            }
        }
        
        print(f"PagedAttention configuration: {paged_attention_config}")
        return paged_attention_config

#### 2. Continuous Batching

class ContinuousBatchingEngine:
    """Implement continuous batching for improved throughput"""
    
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.active_requests = {}
        self.request_queue = []
        
    def add_request(self, request_id: str, prompt: str, max_tokens: int):
        """Add new request to the system"""
        
        request = {
            'id': request_id,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'generated_tokens': 0,
            'status': 'queued',
            'start_time': time.time()
        }
        
        self.request_queue.append(request)
        print(f"Added request {request_id} to queue")
    
    def form_dynamic_batch(self) -> List[Dict[str, Any]]:
        """Form dynamic batch from active requests and queue"""
        
        # Get active requests that need more tokens
        active_batch = [
            req for req in self.active_requests.values()
            if req['generated_tokens'] < req['max_tokens']
        ]
        
        # Add new requests from queue to fill batch
        available_slots = self.max_batch_size - len(active_batch)
        new_requests = self.request_queue[:available_slots]
        
        # Move new requests to active
        for req in new_requests:
            req['status'] = 'active'
            self.active_requests[req['id']] = req
        
        self.request_queue = self.request_queue[available_slots:]
        
        current_batch = active_batch + new_requests
        
        print(f"Formed batch with {len(current_batch)} requests")
        return current_batch
    
    def process_batch_step(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process one generation step for the batch"""
        
        completed_requests = []
        
        for request in batch:
            # Simulate token generation
            request['generated_tokens'] += 1
            
            # Check if request is complete
            if (request['generated_tokens'] >= request['max_tokens'] or 
                self._should_stop_generation(request)):
                
                request['status'] = 'completed'
                request['end_time'] = time.time()
                request['total_time'] = request['end_time'] - request['start_time']
                
                completed_requests.append(request)
                del self.active_requests[request['id']]
        
        return completed_requests
    
    def _should_stop_generation(self, request: Dict[str, Any]) -> bool:
        """Check if generation should stop (EOS token, etc.)"""
        # Mock stopping condition
        return request['generated_tokens'] > 50 and np.random.random() < 0.1

#### 3. Speculative Decoding

class SpeculativeDecoding:
    """Implement speculative decoding for faster generation"""
    
    def __init__(self, target_model, draft_model):
        self.target_model = target_model  # Large, accurate model
        self.draft_model = draft_model    # Small, fast model
        
    def speculative_generate(self, input_ids: torch.Tensor, 
                           num_speculative_tokens: int = 4) -> torch.Tensor:
        """Generate tokens using speculative decoding"""
        
        generated_ids = input_ids.clone()
        
        while True:
            # Step 1: Draft model generates multiple tokens quickly
            draft_tokens = self._draft_generation(
                generated_ids, num_speculative_tokens
            )
            
            # Step 2: Target model verifies draft tokens in parallel
            verification_results = self._verify_tokens(
                generated_ids, draft_tokens
            )
            
            # Step 3: Accept verified tokens and reject rest
            accepted_tokens = verification_results['accepted_tokens']
            generated_ids = torch.cat([generated_ids, accepted_tokens], dim=-1)
            
            # Step 4: If not all tokens accepted, generate one more with target model
            if len(accepted_tokens) < num_speculative_tokens:
                next_token = self._target_generation(generated_ids)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Check stopping condition
            if self._should_stop(generated_ids):
                break
        
        return generated_ids
    
    def _draft_generation(self, input_ids: torch.Tensor, 
                         num_tokens: int) -> torch.Tensor:
        """Generate tokens with draft model"""
        
        # Mock draft generation (fast but potentially inaccurate)
        draft_tokens = torch.randint(0, 1000, (num_tokens,))
        
        return draft_tokens
    
    def _verify_tokens(self, input_ids: torch.Tensor, 
                      draft_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Verify draft tokens with target model"""
        
        # Mock verification process
        # In practice, run target model on input + draft tokens
        # and compare probability distributions
        
        num_accepted = np.random.randint(1, len(draft_tokens) + 1)
        accepted_tokens = draft_tokens[:num_accepted]
        
        return {
            'accepted_tokens': accepted_tokens,
            'acceptance_rate': num_accepted / len(draft_tokens)
        }
    
    def _target_generation(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate one token with target model"""
        
        # Mock target model generation
        next_token = torch.randint(0, 1000, (1,))
        
        return next_token
    
    def _should_stop(self, generated_ids: torch.Tensor) -> bool:
        """Check if generation should stop"""
        
        return len(generated_ids) > 100  # Mock stopping condition
```

### Production Deployment Architecture

#### Load Balancing and Auto-scaling

```python
class LLMLoadBalancer:
    """Load balancer for LLM serving instances"""
    
    def __init__(self):
        self.instances = []
        self.health_check_interval = 30  # seconds
        
    def add_instance(self, instance_id: str, endpoint: str, capacity: int):
        """Add serving instance to load balancer"""
        
        instance = {
            'id': instance_id,
            'endpoint': endpoint,
            'capacity': capacity,
            'current_load': 0,
            'health_status': 'healthy',
            'last_health_check': time.time(),
            'total_requests': 0,
            'avg_latency_ms': 0
        }
        
        self.instances.append(instance)
        print(f"Added instance {instance_id} with capacity {capacity}")
    
    def route_request(self, request: Dict[str, Any]) -> str:
        """Route request to best available instance"""
        
        # Filter healthy instances
        healthy_instances = [
            inst for inst in self.instances 
            if inst['health_status'] == 'healthy'
        ]
        
        if not healthy_instances:
            raise Exception("No healthy instances available")
        
        # Select instance with lowest load
        best_instance = min(
            healthy_instances, 
            key=lambda x: x['current_load'] / x['capacity']
        )
        
        # Update load
        best_instance['current_load'] += 1
        best_instance['total_requests'] += 1
        
        print(f"Routed request to instance {best_instance['id']}")
        return best_instance['endpoint']
    
    def complete_request(self, instance_id: str, latency_ms: float):
        """Mark request as completed and update metrics"""
        
        for instance in self.instances:
            if instance['id'] == instance_id:
                instance['current_load'] -= 1
                
                # Update average latency (exponential moving average)
                alpha = 0.1
                instance['avg_latency_ms'] = (
                    alpha * latency_ms + 
                    (1 - alpha) * instance['avg_latency_ms']
                )
                break
    
    def auto_scale_decision(self) -> Dict[str, Any]:
        """Make auto-scaling decisions based on load"""
        
        total_capacity = sum(inst['capacity'] for inst in self.instances)
        total_load = sum(inst['current_load'] for inst in self.instances)
        
        utilization = total_load / total_capacity if total_capacity > 0 else 0
        avg_latency = np.mean([inst['avg_latency_ms'] for inst in self.instances])
        
        scaling_decision = {
            'current_utilization': utilization,
            'avg_latency_ms': avg_latency,
            'action': 'none',
            'reason': ''
        }
        
        # Scale up conditions
        if utilization > 0.8:
            scaling_decision['action'] = 'scale_up'
            scaling_decision['reason'] = 'High utilization (>80%)'
        elif avg_latency > 200:
            scaling_decision['action'] = 'scale_up'
            scaling_decision['reason'] = 'High latency (>200ms)'
        
        # Scale down conditions
        elif utilization < 0.3 and len(self.instances) > 1:
            scaling_decision['action'] = 'scale_down'
            scaling_decision['reason'] = 'Low utilization (<30%)'
        
        return scaling_decision

class LLMServingCluster:
    """Manage cluster of LLM serving instances"""
    
    def __init__(self, cluster_config: Dict[str, Any]):
        self.config = cluster_config
        self.load_balancer = LLMLoadBalancer()
        self.monitoring = LLMMonitoring()
        
    def deploy_cluster(self):
        """Deploy LLM serving cluster"""
        
        deployment_config = {
            'model_name': self.config['model_name'],
            'instance_type': self.config['instance_type'],
            'min_instances': self.config['min_instances'],
            'max_instances': self.config['max_instances'],
            'target_utilization': self.config.get('target_utilization', 0.7),
            'health_check_path': '/health',
            'metrics_endpoint': '/metrics'
        }
        
        # Deploy initial instances
        for i in range(deployment_config['min_instances']):
            instance_id = f"llm-instance-{i}"
            endpoint = f"http://llm-{i}.cluster.local:8000"
            capacity = 10  # requests per second
            
            self.load_balancer.add_instance(instance_id, endpoint, capacity)
        
        print(f"Deployed LLM cluster: {deployment_config}")
        return deployment_config
    
    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming request through cluster"""
        
        start_time = time.time()
        
        try:
            # Route request
            endpoint = self.load_balancer.route_request(request)
            
            # Process request (mock)
            processing_time = np.random.normal(100, 20)  # Mock latency
            time.sleep(processing_time / 1000)  # Convert to seconds
            
            # Generate response
            response = {
                'request_id': request.get('id', 'unknown'),
                'generated_text': f"Response to: {request.get('prompt', '')[:50]}...",
                'endpoint': endpoint,
                'processing_time_ms': processing_time
            }
            
            # Update metrics
            instance_id = endpoint.split('/')[-1]  # Extract instance ID
            self.load_balancer.complete_request(instance_id, processing_time)
            
            return response
            
        except Exception as e:
            return {
                'error': str(e),
                'request_id': request.get('id', 'unknown')
            }
```

### Monitoring and Optimization

```python
class LLMMonitoring:
    """Comprehensive monitoring for LLM serving"""
    
    def __init__(self):
        self.metrics = {
            'requests_per_second': [],
            'latency_p50': [],
            'latency_p95': [],
            'latency_p99': [],
            'throughput_tokens_per_second': [],
            'gpu_utilization': [],
            'memory_utilization': [],
            'error_rate': [],
            'cost_per_1k_tokens': []
        }
        
    def collect_metrics(self, instance_metrics: Dict[str, Any]):
        """Collect metrics from serving instances"""
        
        # Request metrics
        self.metrics['requests_per_second'].append(
            instance_metrics.get('rps', 0)
        )
        
        # Latency metrics
        latencies = instance_metrics.get('latencies', [])
        if latencies:
            self.metrics['latency_p50'].append(np.percentile(latencies, 50))
            self.metrics['latency_p95'].append(np.percentile(latencies, 95))
            self.metrics['latency_p99'].append(np.percentile(latencies, 99))
        
        # Throughput metrics
        self.metrics['throughput_tokens_per_second'].append(
            instance_metrics.get('tokens_per_second', 0)
        )
        
        # Resource metrics
        self.metrics['gpu_utilization'].append(
            instance_metrics.get('gpu_util', 0)
        )
        self.metrics['memory_utilization'].append(
            instance_metrics.get('memory_util', 0)
        )
        
        # Error metrics
        self.metrics['error_rate'].append(
            instance_metrics.get('error_rate', 0)
        )
        
        # Cost metrics
        self.metrics['cost_per_1k_tokens'].append(
            instance_metrics.get('cost_per_1k_tokens', 0)
        )
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        if not any(self.metrics.values()):
            return {'error': 'No metrics available'}
        
        report = {
            'performance_summary': {
                'avg_rps': np.mean(self.metrics['requests_per_second']),
                'avg_latency_p50_ms': np.mean(self.metrics['latency_p50']),
                'avg_latency_p95_ms': np.mean(self.metrics['latency_p95']),
                'avg_latency_p99_ms': np.mean(self.metrics['latency_p99']),
                'avg_throughput_tps': np.mean(self.metrics['throughput_tokens_per_second']),
            },
            'resource_utilization': {
                'avg_gpu_utilization': np.mean(self.metrics['gpu_utilization']),
                'avg_memory_utilization': np.mean(self.metrics['memory_utilization']),
                'peak_gpu_utilization': np.max(self.metrics['gpu_utilization']),
                'peak_memory_utilization': np.max(self.metrics['memory_utilization'])
            },
            'reliability': {
                'avg_error_rate': np.mean(self.metrics['error_rate']),
                'uptime_percentage': 100 - (np.mean(self.metrics['error_rate']) * 100)
            },
            'cost_analysis': {
                'avg_cost_per_1k_tokens': np.mean(self.metrics['cost_per_1k_tokens']),
                'estimated_monthly_cost': self._estimate_monthly_cost()
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _estimate_monthly_cost(self) -> float:
        """Estimate monthly serving cost"""
        
        avg_rps = np.mean(self.metrics['requests_per_second'])
        avg_tokens_per_request = 100  # Assumption
        avg_cost_per_1k_tokens = np.mean(self.metrics['cost_per_1k_tokens'])
        
        # Calculate monthly cost
        seconds_per_month = 30 * 24 * 3600
        monthly_requests = avg_rps * seconds_per_month
        monthly_tokens = monthly_requests * avg_tokens_per_request
        monthly_cost = (monthly_tokens / 1000) * avg_cost_per_1k_tokens
        
        return monthly_cost
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        
        recommendations = []
        
        # Latency recommendations
        avg_p95_latency = np.mean(self.metrics['latency_p95'])
        if avg_p95_latency > 200:
            recommendations.append(
                "High P95 latency detected. Consider scaling up or optimizing model."
            )
        
        # Utilization recommendations
        avg_gpu_util = np.mean(self.metrics['gpu_utilization'])
        if avg_gpu_util < 50:
            recommendations.append(
                "Low GPU utilization. Consider increasing batch size or scaling down."
            )
        elif avg_gpu_util > 90:
            recommendations.append(
                "High GPU utilization. Consider scaling up to prevent bottlenecks."
            )
        
        # Cost recommendations
        avg_cost = np.mean(self.metrics['cost_per_1k_tokens'])
        if avg_cost > 0.01:  # $0.01 per 1k tokens
            recommendations.append(
                "High serving cost. Consider model optimization or instance type changes."
            )
        
        return recommendations
```

### Cost Optimization Strategies

```python
class LLMCostOptimizer:
    """Optimize LLM serving costs"""
    
    def __init__(self):
        self.cost_models = {
            'gpu_instance': {
                'a100_40gb': {'cost_per_hour': 3.20, 'memory_gb': 40},
                'a100_80gb': {'cost_per_hour': 6.40, 'memory_gb': 80},
                'h100_80gb': {'cost_per_hour': 8.00, 'memory_gb': 80},
                'v100_32gb': {'cost_per_hour': 2.40, 'memory_gb': 32}
            }
        }
    
    def calculate_serving_cost(self, model_size_gb: float, 
                             requests_per_hour: int,
                             avg_tokens_per_request: int) -> Dict[str, Any]:
        """Calculate serving cost for different configurations"""
        
        cost_analysis = {}
        
        for instance_type, specs in self.cost_models['gpu_instance'].items():
            # Check if model fits in memory
            if model_size_gb > specs['memory_gb'] * 0.8:  # 80% utilization
                continue
            
            # Calculate costs
            hourly_cost = specs['cost_per_hour']
            daily_cost = hourly_cost * 24
            monthly_cost = daily_cost * 30
            
            # Calculate cost per token
            tokens_per_hour = requests_per_hour * avg_tokens_per_request
            cost_per_1k_tokens = (hourly_cost / tokens_per_hour) * 1000 if tokens_per_hour > 0 else 0
            
            cost_analysis[instance_type] = {
                'hourly_cost': hourly_cost,
                'daily_cost': daily_cost,
                'monthly_cost': monthly_cost,
                'cost_per_1k_tokens': cost_per_1k_tokens,
                'max_model_size_gb': specs['memory_gb'] * 0.8,
                'estimated_throughput': self._estimate_throughput(instance_type)
            }
        
        return cost_analysis
    
    def _estimate_throughput(self, instance_type: str) -> Dict[str, float]:
        """Estimate throughput for instance type"""
        
        # Mock throughput estimates (tokens per second)
        throughput_estimates = {
            'a100_40gb': {'7b_model': 150, '13b_model': 80, '30b_model': 30},
            'a100_80gb': {'7b_model': 200, '13b_model': 120, '30b_model': 50, '70b_model': 20},
            'h100_80gb': {'7b_model': 300, '13b_model': 180, '30b_model': 80, '70b_model': 35},
            'v100_32gb': {'7b_model': 100, '13b_model': 50}
        }
        
        return throughput_estimates.get(instance_type, {})
    
    def recommend_optimal_configuration(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal serving configuration"""
        
        model_size_gb = requirements['model_size_gb']
        target_latency_ms = requirements.get('target_latency_ms', 100)
        expected_rps = requirements.get('requests_per_second', 10)
        budget_per_month = requirements.get('budget_per_month', 1000)
        
        # Calculate costs for all viable configurations
        cost_analysis = self.calculate_serving_cost(
            model_size_gb, 
            expected_rps * 3600,  # Convert to requests per hour
            requirements.get('avg_tokens_per_request', 100)
        )
        
        # Filter by budget
        viable_configs = {
            k: v for k, v in cost_analysis.items()
            if v['monthly_cost'] <= budget_per_month
        }
        
        if not viable_configs:
            return {
                'error': 'No configurations within budget',
                'min_budget_required': min(v['monthly_cost'] for v in cost_analysis.values())
            }
        
        # Select best configuration (lowest cost per token)
        best_config = min(
            viable_configs.items(),
            key=lambda x: x[1]['cost_per_1k_tokens']
        )
        
        recommendation = {
            'recommended_instance': best_config[0],
            'configuration': best_config[1],
            'optimization_strategies': [
                'Use quantization to reduce memory requirements',
                'Implement request batching for higher throughput',
                'Consider spot instances for cost savings',
                'Monitor and auto-scale based on demand'
            ],
            'estimated_savings': self._calculate_savings(cost_analysis, best_config[0])
        }
        
        return recommendation
    
    def _calculate_savings(self, cost_analysis: Dict[str, Any], 
                          selected_config: str) -> Dict[str, float]:
        """Calculate potential savings from optimization"""
        
        selected_cost = cost_analysis[selected_config]['monthly_cost']
        
        # Compare with most expensive viable option
        max_cost = max(v['monthly_cost'] for v in cost_analysis.values())
        
        savings = {
            'absolute_savings_per_month': max_cost - selected_cost,
            'percentage_savings': ((max_cost - selected_cost) / max_cost) * 100,
            'annual_savings': (max_cost - selected_cost) * 12
        }
        
        return savings
```

### Why LLM Serving Optimization Matters

LLM serving optimization is critical for production AI systems because:

- **Cost Efficiency**: GPU costs can exceed $10,000/month per model
- **User Experience**: Latency directly impacts user satisfaction and adoption
- **Scalability**: Efficient serving enables handling millions of requests
- **Resource Utilization**: Optimization maximizes hardware ROI
- **Competitive Advantage**: Faster, cheaper serving enables better products
- **Environmental Impact**: Efficient serving reduces energy consumption

## Exercise (25 minutes)
Complete the hands-on exercises in `exercise.py` to practice LLM serving optimization.

## Resources
- [vLLM Documentation](https://docs.vllm.ai/)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
- [PagedAttention Paper](https://arxiv.org/abs/2309.06180)
- [Continuous Batching Blog](https://www.anyscale.com/blog/continuous-batching-llm-inference)
- [Speculative Decoding Paper](https://arxiv.org/abs/2211.17192)
- [LLM Inference Optimization Guide](https://huggingface.co/docs/transformers/llm_tutorial_optimization)

## Next Steps
- Complete the exercises and experiment with different optimization techniques
- Set up vLLM or TensorRT for your models
- Take the quiz to test your understanding
- Move to Day 52: Advanced RAG - Retrieval-Augmented Generation Systems
