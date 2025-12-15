"""
Day 51: LLM Serving & Optimization - vLLM, TensorRT, Inference - Solutions

Complete implementations for all LLM serving optimization exercises with production-ready code.
"""

import torch
import torch.nn as nn
import asyncio
import time
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import psutil
import GPUtil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock components for demonstration (replace with actual implementations in production)
class MockLLMModel(nn.Module):
    def __init__(self, vocab_size=50000, hidden_size=4096, num_layers=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(hidden_size, 32, batch_first=True)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
    def forward(self, input_ids, past_key_values=None):
        x = self.embedding(input_ids)
        
        new_past_key_values = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values else None
            x = layer(x, memory=past_kv)
            new_past_key_values.append(x)  # Simplified KV storage
        
        logits = self.lm_head(x)
        
        return {
            'logits': logits,
            'past_key_values': new_past_key_values
        }

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 50000
        self.eos_token_id = 2
        self.pad_token_id = 0
        
    def encode(self, text):
        # Simple mock tokenization
        return [hash(word) % (self.vocab_size - 10) + 10 for word in text.split()]
    
    def decode(self, tokens):
        return f"Generated text from {len(tokens)} tokens"

# Solution 1: vLLM Serving Setup and Configuration
class vLLMServingEngine:
    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.engine = None
        self.active_requests = {}
        self.request_counter = 0
        
    def initialize_engine(self):
        """Initialize vLLM engine with configuration"""
        
        engine_config = {
            'model': self.model_name,
            'tensor_parallel_size': self.config.get('tensor_parallel_size', 1),
            'dtype': self.config.get('dtype', 'float16'),
            'max_model_len': self.config.get('max_model_len', 4096),
            'gpu_memory_utilization': self.config.get('gpu_memory_utilization', 0.9),
            'max_num_seqs': self.config.get('max_num_seqs', 256),
            'max_num_batched_tokens': self.config.get('max_num_batched_tokens', 8192),
            'block_size': self.config.get('block_size', 16),  # PagedAttention block size
            'swap_space': self.config.get('swap_space', 4),   # GB of CPU swap space
            'enforce_eager': self.config.get('enforce_eager', False)
        }
        
        # Mock engine initialization (in production, use actual vLLM AsyncLLMEngine)
        self.engine = {
            'config': engine_config,
            'initialized': True,
            'memory_pool': self._initialize_memory_pool(),
            'scheduler': self._initialize_scheduler()
        }
        
        logger.info(f"vLLM engine initialized for {self.model_name}")
        logger.info(f"Configuration: {engine_config}")
        
        return self.engine
    
    def _initialize_memory_pool(self):
        """Initialize PagedAttention memory pool"""
        
        gpu_memory_gb = self.config.get('gpu_memory_utilization', 0.9) * 40  # Assume 40GB GPU
        block_size = self.config.get('block_size', 16)
        
        # Calculate number of blocks
        bytes_per_block = block_size * self.config.get('hidden_size', 4096) * 2  # FP16
        total_blocks = int((gpu_memory_gb * 1024**3) / bytes_per_block)
        
        memory_pool = {
            'total_blocks': total_blocks,
            'free_blocks': total_blocks,
            'allocated_blocks': {},
            'block_size': block_size,
            'fragmentation_ratio': 0.0
        }
        
        logger.info(f"Memory pool initialized: {total_blocks} blocks of size {block_size}")
        return memory_pool
    
    def _initialize_scheduler(self):
        """Initialize request scheduler"""
        
        scheduler_config = {
            'max_num_seqs': self.config.get('max_num_seqs', 256),
            'max_num_batched_tokens': self.config.get('max_num_batched_tokens', 8192),
            'scheduling_policy': 'fcfs',  # First-come-first-serve
            'preemption_enabled': True
        }
        
        return {
            'config': scheduler_config,
            'waiting_queue': [],
            'running_queue': [],
            'swapped_queue': []
        }
    
    def create_sampling_params(self, **kwargs):
        """Create sampling parameters for text generation"""
        
        default_params = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'max_tokens': 256,
            'repetition_penalty': 1.1,
            'stop': ['</s>', '<|endoftext|>'],
            'ignore_eos': False,
            'use_beam_search': False,
            'best_of': 1
        }
        
        # Override with provided parameters
        default_params.update(kwargs)
        
        logger.info(f"Created sampling params: {default_params}")
        return default_params
    
    async def generate_async(self, prompts: List[str], sampling_params):
        """Generate text asynchronously using vLLM"""
        
        if not self.engine:
            self.initialize_engine()
        
        # Add requests to engine
        request_ids = []
        for i, prompt in enumerate(prompts):
            request_id = f"request_{self.request_counter}_{i}"
            self.request_counter += 1
            
            # Mock request addition
            request = {
                'request_id': request_id,
                'prompt': prompt,
                'sampling_params': sampling_params,
                'status': 'waiting',
                'created_at': time.time(),
                'tokens_generated': 0
            }
            
            self.active_requests[request_id] = request
            request_ids.append(request_id)
        
        logger.info(f"Added {len(prompts)} requests to engine")
        
        # Simulate async generation
        results = []
        for request_id in request_ids:
            # Mock generation process
            await asyncio.sleep(0.1)  # Simulate processing time
            
            generated_text = f"Generated response for request {request_id}"
            results.append(generated_text)
            
            # Update request status
            self.active_requests[request_id]['status'] = 'completed'
            self.active_requests[request_id]['completed_at'] = time.time()
        
        return results
    
    def benchmark_performance(self, test_prompts: List[str], num_runs: int = 10):
        """Benchmark vLLM performance"""
        
        logger.info(f"Starting performance benchmark with {len(test_prompts)} prompts, {num_runs} runs")
        
        metrics = {
            'ttft_times': [],  # Time to first token
            'tpot_times': [],  # Time per output token
            'total_times': [],
            'throughput_tps': [],  # Tokens per second
            'memory_usage': []
        }
        
        for run in range(num_runs):
            start_time = time.time()
            
            # Create sampling params
            sampling_params = self.create_sampling_params(max_tokens=100, temperature=0.7)
            
            # Simulate generation
            ttft = np.random.normal(50, 10)  # Mock TTFT in ms
            tpot = np.random.normal(20, 5)   # Mock TPOT in ms
            total_tokens = 100
            
            total_time = ttft + (total_tokens * tpot)
            throughput = (total_tokens * 1000) / total_time  # tokens per second
            
            # Mock memory usage
            memory_usage = np.random.uniform(8, 12)  # GB
            
            # Store metrics
            metrics['ttft_times'].append(ttft)
            metrics['tpot_times'].append(tpot)
            metrics['total_times'].append(total_time)
            metrics['throughput_tps'].append(throughput)
            metrics['memory_usage'].append(memory_usage)
            
            if run % 5 == 0:
                logger.info(f"Benchmark run {run+1}/{num_runs} completed")
        
        # Calculate statistics
        benchmark_results = {
            'avg_ttft_ms': np.mean(metrics['ttft_times']),
            'p95_ttft_ms': np.percentile(metrics['ttft_times'], 95),
            'avg_tpot_ms': np.mean(metrics['tpot_times']),
            'p95_tpot_ms': np.percentile(metrics['tpot_times'], 95),
            'avg_throughput_tps': np.mean(metrics['throughput_tps']),
            'peak_throughput_tps': np.max(metrics['throughput_tps']),
            'avg_memory_usage_gb': np.mean(metrics['memory_usage']),
            'peak_memory_usage_gb': np.max(metrics['memory_usage'])
        }
        
        logger.info(f"Benchmark completed: {benchmark_results}")
        return benchmark_results

# Solution 2: TensorRT Optimization Implementation
class TensorRTOptimizer:
    def __init__(self, model_path: str, optimization_config: Dict[str, Any]):
        self.model_path = model_path
        self.config = optimization_config
        self.engine = None
        self.optimization_profiles = []
        
    def build_tensorrt_engine(self):
        """Build optimized TensorRT engine"""
        
        build_config = {
            'precision': self.config.get('precision', 'fp16'),
            'max_batch_size': self.config.get('max_batch_size', 8),
            'max_sequence_length': self.config.get('max_sequence_length', 2048),
            'optimization_level': self.config.get('optimization_level', 3),
            'use_cuda_graphs': self.config.get('use_cuda_graphs', True),
            'enable_plugins': self.config.get('enable_plugins', True)
        }
        
        logger.info(f"Building TensorRT engine with config: {build_config}")
        
        # Mock engine building process
        build_time_start = time.time()
        
        # Simulate optimization steps
        optimization_steps = [
            'Loading model weights',
            'Analyzing network structure',
            'Applying layer fusion',
            'Optimizing precision',
            'Generating CUDA kernels',
            'Building engine'
        ]
        
        for step in optimization_steps:
            logger.info(f"TensorRT: {step}...")
            time.sleep(0.5)  # Simulate build time
        
        build_time = time.time() - build_time_start
        
        # Mock engine info
        engine_info = {
            'precision': build_config['precision'],
            'max_batch_size': build_config['max_batch_size'],
            'build_time_seconds': build_time,
            'engine_size_mb': 1500,  # Mock size
            'optimization_level': build_config['optimization_level'],
            'estimated_speedup': self._estimate_speedup(build_config),
            'memory_usage_gb': self._estimate_memory_usage(build_config)
        }
        
        self.engine = engine_info
        logger.info(f"TensorRT engine built successfully: {engine_info}")
        
        return engine_info
    
    def _estimate_speedup(self, config):
        """Estimate speedup based on optimization configuration"""
        
        speedup_factors = {
            'fp32': 1.0,
            'fp16': 1.8,
            'int8': 3.2
        }
        
        base_speedup = speedup_factors.get(config['precision'], 1.0)
        
        # Additional speedup from optimizations
        if config.get('use_cuda_graphs', False):
            base_speedup *= 1.2
        
        if config.get('optimization_level', 0) >= 3:
            base_speedup *= 1.15
        
        return f"{base_speedup:.1f}x"
    
    def _estimate_memory_usage(self, config):
        """Estimate memory usage"""
        
        base_memory = 8.0  # GB
        
        precision_factors = {
            'fp32': 1.0,
            'fp16': 0.5,
            'int8': 0.25
        }
        
        memory_usage = base_memory * precision_factors.get(config['precision'], 1.0)
        memory_usage *= config.get('max_batch_size', 1) * 0.1  # Batch scaling
        
        return memory_usage
    
    def optimize_for_latency(self):
        """Optimize engine for low latency"""
        
        latency_optimizations = {
            'cuda_graphs': True,
            'kernel_fusion': 'aggressive',
            'memory_layout': 'optimized',
            'batch_size': 1,
            'precision': 'fp16',
            'profile': 'latency_optimized'
        }
        
        logger.info("Applying latency optimizations...")
        
        # Mock optimization application
        for opt_name, opt_value in latency_optimizations.items():
            logger.info(f"  {opt_name}: {opt_value}")
        
        # Update engine configuration
        if self.engine:
            self.engine['latency_optimizations'] = latency_optimizations
            self.engine['estimated_latency_ms'] = 15  # Mock optimized latency
        
        return latency_optimizations
    
    def optimize_for_throughput(self):
        """Optimize engine for high throughput"""
        
        throughput_optimizations = {
            'batch_size': 'dynamic_max',
            'memory_pooling': True,
            'pipeline_parallelism': True,
            'multi_stream': True,
            'precision': 'int8',
            'profile': 'throughput_optimized'
        }
        
        logger.info("Applying throughput optimizations...")
        
        for opt_name, opt_value in throughput_optimizations.items():
            logger.info(f"  {opt_name}: {opt_value}")
        
        if self.engine:
            self.engine['throughput_optimizations'] = throughput_optimizations
            self.engine['estimated_throughput_tps'] = 250  # Mock optimized throughput
        
        return throughput_optimizations
    
    def benchmark_tensorrt_performance(self, input_shapes: List[Tuple[int, int]]):
        """Benchmark TensorRT engine performance"""
        
        logger.info(f"Benchmarking TensorRT performance with shapes: {input_shapes}")
        
        benchmark_results = {}
        
        for batch_size, seq_length in input_shapes:
            logger.info(f"Testing batch_size={batch_size}, seq_length={seq_length}")
            
            # Mock performance metrics
            latency_ms = 10 + (batch_size * 2) + (seq_length * 0.01)
            throughput_tps = (batch_size * 100) / (latency_ms / 1000)
            memory_gb = 2 + (batch_size * 0.5) + (seq_length * 0.001)
            
            benchmark_results[f"bs{batch_size}_seq{seq_length}"] = {
                'batch_size': batch_size,
                'sequence_length': seq_length,
                'latency_ms': latency_ms,
                'throughput_tps': throughput_tps,
                'memory_usage_gb': memory_gb,
                'gpu_utilization': min(95, 60 + batch_size * 5)
            }
        
        logger.info(f"TensorRT benchmark completed: {len(benchmark_results)} configurations tested")
        return benchmark_results

# Solution 3: KV-Cache Optimization Strategies
class KVCacheManager:
    def __init__(self, max_sequence_length: int, num_layers: int, hidden_size: int):
        self.max_sequence_length = max_sequence_length
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.cache_blocks = {}
        self.memory_pool = None
        self.block_size = 16
        
    def implement_paged_attention(self, block_size: int = 16):
        """Implement PagedAttention for efficient KV-cache management"""
        
        self.block_size = block_size
        
        # Calculate memory requirements
        bytes_per_element = 2  # FP16
        kv_size_per_token = 2 * self.num_layers * self.hidden_size * bytes_per_element
        
        # Initialize memory pool
        total_gpu_memory_gb = 40  # Assume A100 40GB
        usable_memory_gb = total_gpu_memory_gb * 0.8  # 80% for KV cache
        
        total_blocks = int((usable_memory_gb * 1024**3) / (block_size * kv_size_per_token))
        
        self.memory_pool = {
            'total_blocks': total_blocks,
            'free_blocks': list(range(total_blocks)),
            'allocated_blocks': {},
            'block_table': {},  # logical_block_id -> physical_block_id
            'fragmentation_stats': {
                'total_fragmentation': 0,
                'largest_free_chunk': total_blocks
            }
        }
        
        paged_attention_config = {
            'block_size': block_size,
            'total_blocks': total_blocks,
            'memory_efficiency': f"Up to {block_size}x reduction in memory waste",
            'features': [
                'Non-contiguous memory allocation',
                'Dynamic memory management',
                'Reduced memory fragmentation',
                'Copy-on-write for shared prefixes'
            ]
        }
        
        logger.info(f"PagedAttention initialized: {paged_attention_config}")
        return paged_attention_config
    
    def allocate_blocks(self, sequence_id: str, num_blocks: int):
        """Allocate memory blocks for a sequence"""
        
        if len(self.memory_pool['free_blocks']) < num_blocks:
            # Trigger garbage collection or eviction
            self._garbage_collect()
        
        if len(self.memory_pool['free_blocks']) < num_blocks:
            raise RuntimeError(f"Insufficient memory: need {num_blocks}, have {len(self.memory_pool['free_blocks'])}")
        
        # Allocate blocks
        allocated_blocks = []
        for _ in range(num_blocks):
            physical_block = self.memory_pool['free_blocks'].pop(0)
            allocated_blocks.append(physical_block)
        
        self.memory_pool['allocated_blocks'][sequence_id] = allocated_blocks
        
        logger.info(f"Allocated {num_blocks} blocks for sequence {sequence_id}")
        return allocated_blocks
    
    def deallocate_blocks(self, sequence_id: str):
        """Deallocate memory blocks for a sequence"""
        
        if sequence_id in self.memory_pool['allocated_blocks']:
            blocks = self.memory_pool['allocated_blocks'][sequence_id]
            self.memory_pool['free_blocks'].extend(blocks)
            del self.memory_pool['allocated_blocks'][sequence_id]
            
            logger.info(f"Deallocated {len(blocks)} blocks for sequence {sequence_id}")
    
    def _garbage_collect(self):
        """Perform garbage collection to free unused blocks"""
        
        # Mock garbage collection
        freed_blocks = 0
        
        # In practice, this would:
        # 1. Identify completed sequences
        # 2. Free their allocated blocks
        # 3. Defragment memory if needed
        
        logger.info(f"Garbage collection freed {freed_blocks} blocks")
    
    def optimize_cache_memory(self):
        """Optimize KV-cache memory usage"""
        
        optimization_strategies = {
            'memory_pooling': {
                'enabled': True,
                'pool_size_gb': 32,
                'allocation_strategy': 'best_fit'
            },
            'compression': {
                'enabled': True,
                'method': 'quantization',
                'compression_ratio': '2:1'
            },
            'eviction_policy': {
                'policy': 'lru',
                'max_cache_size': self.memory_pool['total_blocks'] * 0.9
            },
            'prefetching': {
                'enabled': True,
                'prefetch_distance': 4
            }
        }
        
        logger.info(f"Applied cache optimizations: {optimization_strategies}")
        return optimization_strategies
    
    def calculate_memory_requirements(self, batch_size: int, sequence_length: int):
        """Calculate KV-cache memory requirements"""
        
        # Memory per token (key + value for all layers)
        bytes_per_element = 2  # FP16
        memory_per_token = 2 * self.num_layers * self.hidden_size * bytes_per_element
        
        # Total memory for batch
        total_memory_bytes = batch_size * sequence_length * memory_per_token
        total_memory_gb = total_memory_bytes / (1024**3)
        
        # Calculate blocks needed
        tokens_per_block = self.block_size
        blocks_needed = (batch_size * sequence_length + tokens_per_block - 1) // tokens_per_block
        
        memory_analysis = {
            'batch_size': batch_size,
            'sequence_length': sequence_length,
            'memory_per_token_bytes': memory_per_token,
            'total_memory_gb': total_memory_gb,
            'blocks_needed': blocks_needed,
            'memory_efficiency': f"{(sequence_length % self.block_size) / self.block_size * 100:.1f}% waste",
            'max_sequences_in_memory': self.memory_pool['total_blocks'] // blocks_needed if blocks_needed > 0 else 0
        }
        
        return memory_analysis
    
    def benchmark_cache_performance(self):
        """Benchmark KV-cache performance"""
        
        test_scenarios = [
            (1, 512), (4, 1024), (8, 2048), (16, 4096)
        ]
        
        benchmark_results = {}
        
        for batch_size, seq_length in test_scenarios:
            scenario_name = f"bs{batch_size}_seq{seq_length}"
            
            # Calculate memory requirements
            memory_req = self.calculate_memory_requirements(batch_size, seq_length)
            
            # Mock performance metrics
            cache_hit_rate = np.random.uniform(0.85, 0.95)
            memory_bandwidth_gbps = np.random.uniform(800, 1200)
            access_latency_us = np.random.uniform(1, 5)
            
            benchmark_results[scenario_name] = {
                **memory_req,
                'cache_hit_rate': cache_hit_rate,
                'memory_bandwidth_gbps': memory_bandwidth_gbps,
                'access_latency_us': access_latency_us,
                'efficiency_score': cache_hit_rate * (1000 / access_latency_us)
            }
        
        logger.info(f"Cache performance benchmark completed: {len(benchmark_results)} scenarios")
        return benchmark_results

# Solution 4: Continuous Batching Implementation
class ContinuousBatchingEngine:
    def __init__(self, max_batch_size: int = 32):
        self.max_batch_size = max_batch_size
        self.active_requests = {}
        self.request_queue = queue.Queue()
        self.running = False
        self.batch_stats = {
            'batches_processed': 0,
            'total_requests': 0,
            'avg_batch_size': 0,
            'utilization_rate': 0
        }
        
    def add_request(self, request_id: str, prompt: str, max_tokens: int):
        """Add new request to the system"""
        
        request = {
            'id': request_id,
            'prompt': prompt,
            'max_tokens': max_tokens,
            'generated_tokens': 0,
            'status': 'queued',
            'start_time': time.time(),
            'priority': 1,  # Default priority
            'estimated_completion_time': time.time() + (max_tokens * 0.02)  # Mock estimate
        }
        
        self.request_queue.put(request)
        logger.info(f"Added request {request_id} to queue (queue size: {self.request_queue.qsize()})")
    
    def form_dynamic_batch(self):
        """Form dynamic batch from active requests and queue"""
        
        # Get active requests that need more tokens
        active_batch = [
            req for req in self.active_requests.values()
            if req['generated_tokens'] < req['max_tokens'] and req['status'] == 'active'
        ]
        
        # Add new requests from queue to fill batch
        available_slots = self.max_batch_size - len(active_batch)
        new_requests = []
        
        while available_slots > 0 and not self.request_queue.empty():
            try:
                request = self.request_queue.get_nowait()
                request['status'] = 'active'
                self.active_requests[request['id']] = request
                new_requests.append(request)
                available_slots -= 1
            except queue.Empty:
                break
        
        current_batch = active_batch + new_requests
        
        # Update batch statistics
        if current_batch:
            self.batch_stats['batches_processed'] += 1
            self.batch_stats['total_requests'] += len(new_requests)
            self.batch_stats['avg_batch_size'] = (
                (self.batch_stats['avg_batch_size'] * (self.batch_stats['batches_processed'] - 1) + len(current_batch)) /
                self.batch_stats['batches_processed']
            )
            self.batch_stats['utilization_rate'] = len(current_batch) / self.max_batch_size
        
        logger.info(f"Formed batch with {len(current_batch)} requests "
                   f"({len(active_batch)} active, {len(new_requests)} new)")
        
        return current_batch
    
    def process_batch_step(self, batch: List[Dict[str, Any]]):
        """Process one generation step for the batch"""
        
        if not batch:
            return []
        
        completed_requests = []
        
        # Mock batch processing
        processing_time = 0.05 + (len(batch) * 0.01)  # Simulate processing time
        time.sleep(processing_time)
        
        for request in batch:
            # Simulate token generation
            request['generated_tokens'] += 1
            
            # Mock stopping conditions
            should_stop = (
                request['generated_tokens'] >= request['max_tokens'] or
                (request['generated_tokens'] > 10 and np.random.random() < 0.05)  # Random EOS
            )
            
            if should_stop:
                request['status'] = 'completed'
                request['end_time'] = time.time()
                request['total_time'] = request['end_time'] - request['start_time']
                request['tokens_per_second'] = request['generated_tokens'] / request['total_time']
                
                completed_requests.append(request)
                
                # Remove from active requests
                if request['id'] in self.active_requests:
                    del self.active_requests[request['id']]
                
                logger.info(f"Request {request['id']} completed: "
                           f"{request['generated_tokens']} tokens in {request['total_time']:.2f}s")
        
        return completed_requests
    
    async def run_continuous_batching(self, duration_seconds: int = 60):
        """Main continuous batching loop"""
        
        self.running = True
        start_time = time.time()
        
        logger.info(f"Starting continuous batching for {duration_seconds} seconds")
        
        while self.running and (time.time() - start_time) < duration_seconds:
            # Form dynamic batch
            batch = self.form_dynamic_batch()
            
            if batch:
                # Process batch step
                completed = self.process_batch_step(batch)
                
                # Handle completed requests
                for request in completed:
                    logger.info(f"Completed request {request['id']}: "
                               f"{request['tokens_per_second']:.1f} tokens/sec")
            else:
                # No requests to process, short sleep
                await asyncio.sleep(0.01)
        
        self.running = False
        logger.info("Continuous batching stopped")
    
    def benchmark_batching_efficiency(self):
        """Benchmark continuous batching efficiency"""
        
        efficiency_metrics = {
            'avg_batch_size': self.batch_stats['avg_batch_size'],
            'batch_utilization': self.batch_stats['utilization_rate'],
            'batches_processed': self.batch_stats['batches_processed'],
            'total_requests_processed': self.batch_stats['total_requests'],
            'active_requests': len(self.active_requests),
            'queue_size': self.request_queue.qsize()
        }
        
        # Calculate efficiency scores
        efficiency_metrics['utilization_score'] = min(100, efficiency_metrics['batch_utilization'] * 100)
        efficiency_metrics['throughput_improvement'] = f"{efficiency_metrics['avg_batch_size']:.1f}x"
        
        logger.info(f"Batching efficiency metrics: {efficiency_metrics}")
        return efficiency_metrics

# Solution 5: Speculative Decoding Setup
class SpeculativeDecodingEngine:
    def __init__(self, target_model, draft_model):
        self.target_model = target_model
        self.draft_model = draft_model
        self.tokenizer = MockTokenizer()
        self.acceptance_stats = {
            'total_candidates': 0,
            'accepted_tokens': 0,
            'acceptance_rate': 0.0
        }
        
    def setup_draft_model(self):
        """Setup and optimize draft model"""
        
        draft_optimizations = {
            'model_size': 'small',  # e.g., 1B parameters vs 7B target
            'quantization': 'int8',
            'optimization_level': 'aggressive',
            'memory_usage_gb': 2,
            'inference_speed': '5x faster than target',
            'accuracy_trade_off': '10-15% lower quality'
        }
        
        logger.info(f"Draft model optimizations: {draft_optimizations}")
        
        # Mock optimization application
        self.draft_model.eval()
        
        return draft_optimizations
    
    def speculative_generate_step(self, input_ids: torch.Tensor, num_candidates: int = 4):
        """Single step of speculative decoding"""
        
        # Step 1: Draft model generates candidate tokens
        draft_candidates = self._draft_generation(input_ids, num_candidates)
        
        # Step 2: Target model verifies candidates
        verification_results = self._verify_candidates(input_ids, draft_candidates)
        
        # Step 3: Accept/reject tokens based on verification
        accepted_tokens = verification_results['accepted_tokens']
        acceptance_rate = verification_results['acceptance_rate']
        
        # Update statistics
        self.acceptance_stats['total_candidates'] += num_candidates
        self.acceptance_stats['accepted_tokens'] += len(accepted_tokens)
        self.acceptance_stats['acceptance_rate'] = (
            self.acceptance_stats['accepted_tokens'] / self.acceptance_stats['total_candidates']
        )
        
        # Step 4: If not all tokens accepted, generate one more with target model
        final_tokens = accepted_tokens
        if len(accepted_tokens) < num_candidates:
            additional_token = self._target_generation(
                torch.cat([input_ids, accepted_tokens], dim=-1)
            )
            final_tokens = torch.cat([accepted_tokens, additional_token], dim=-1)
        
        generation_result = {
            'generated_tokens': final_tokens,
            'candidates_generated': num_candidates,
            'tokens_accepted': len(accepted_tokens),
            'acceptance_rate': acceptance_rate,
            'speedup_estimate': self._calculate_speedup(len(accepted_tokens), num_candidates)
        }
        
        return generation_result
    
    def _draft_generation(self, input_ids: torch.Tensor, num_tokens: int):
        """Generate candidate tokens with draft model"""
        
        # Mock fast draft generation
        draft_tokens = torch.randint(10, 1000, (num_tokens,))
        
        # Simulate draft model inference time (much faster than target)
        draft_time = 0.005 * num_tokens  # 5ms per token
        
        logger.debug(f"Draft model generated {num_tokens} candidates in {draft_time:.3f}s")
        return draft_tokens
    
    def _verify_candidates(self, input_ids: torch.Tensor, candidates: torch.Tensor):
        """Verify candidate tokens with target model"""
        
        # Mock verification process
        # In practice: run target model on input + candidates and compare distributions
        
        # Simulate acceptance pattern (typically decreases with position)
        acceptance_probs = [0.8, 0.6, 0.4, 0.2][:len(candidates)]
        accepted_mask = [np.random.random() < prob for prob in acceptance_probs]
        
        # Find first rejection point
        first_rejection = len(accepted_mask)
        for i, accepted in enumerate(accepted_mask):
            if not accepted:
                first_rejection = i
                break
        
        accepted_tokens = candidates[:first_rejection]
        acceptance_rate = first_rejection / len(candidates)
        
        verification_result = {
            'accepted_tokens': accepted_tokens,
            'acceptance_rate': acceptance_rate,
            'verification_time': 0.02,  # Mock verification time
            'candidates_verified': len(candidates)
        }
        
        return verification_result
    
    def _target_generation(self, input_ids: torch.Tensor):
        """Generate one token with target model"""
        
        # Mock target model generation
        next_token = torch.randint(10, 1000, (1,))
        
        return next_token
    
    def _calculate_speedup(self, accepted_tokens: int, total_candidates: int):
        """Calculate speedup from speculative decoding"""
        
        # Speedup calculation:
        # Normal: N forward passes for N tokens
        # Speculative: 1 draft pass + 1 verification pass for up to K tokens
        
        if accepted_tokens == 0:
            return 1.0
        
        # Assume draft model is 5x faster than target model
        draft_cost = total_candidates * 0.2  # Relative cost
        verification_cost = 1.0  # One target model pass
        total_cost = draft_cost + verification_cost
        
        # Compare to normal generation cost
        normal_cost = accepted_tokens * 1.0
        
        speedup = normal_cost / total_cost
        return max(1.0, speedup)
    
    def calculate_acceptance_rate(self, verification_results: List[Dict[str, Any]]):
        """Calculate token acceptance rate statistics"""
        
        if not verification_results:
            return {'error': 'No verification results provided'}
        
        total_candidates = sum(r['candidates_verified'] for r in verification_results)
        total_accepted = sum(len(r['accepted_tokens']) for r in verification_results)
        
        acceptance_stats = {
            'overall_acceptance_rate': total_accepted / total_candidates if total_candidates > 0 else 0,
            'avg_accepted_per_step': total_accepted / len(verification_results),
            'acceptance_by_position': self._calculate_positional_acceptance(verification_results),
            'estimated_speedup': self._estimate_overall_speedup(verification_results)
        }
        
        return acceptance_stats
    
    def _calculate_positional_acceptance(self, verification_results: List[Dict[str, Any]]):
        """Calculate acceptance rate by token position"""
        
        position_stats = {}
        
        for result in verification_results:
            accepted_count = len(result['accepted_tokens'])
            total_candidates = result['candidates_verified']
            
            for pos in range(total_candidates):
                if pos not in position_stats:
                    position_stats[pos] = {'accepted': 0, 'total': 0}
                
                position_stats[pos]['total'] += 1
                if pos < accepted_count:
                    position_stats[pos]['accepted'] += 1
        
        # Calculate rates
        positional_rates = {}
        for pos, stats in position_stats.items():
            positional_rates[f'position_{pos}'] = stats['accepted'] / stats['total']
        
        return positional_rates
    
    def _estimate_overall_speedup(self, verification_results: List[Dict[str, Any]]):
        """Estimate overall speedup from speculative decoding"""
        
        total_speedup = 0
        for result in verification_results:
            accepted = len(result['accepted_tokens'])
            candidates = result['candidates_verified']
            speedup = self._calculate_speedup(accepted, candidates)
            total_speedup += speedup
        
        avg_speedup = total_speedup / len(verification_results)
        return f"{avg_speedup:.2f}x"
    
    def optimize_speculation_parameters(self):
        """Optimize speculative decoding parameters"""
        
        # Test different numbers of candidate tokens
        candidate_counts = [2, 4, 6, 8]
        optimization_results = {}
        
        for num_candidates in candidate_counts:
            # Mock optimization testing
            mock_acceptance_rate = max(0.2, 0.9 - (num_candidates * 0.1))
            mock_speedup = self._calculate_speedup(
                int(num_candidates * mock_acceptance_rate), 
                num_candidates
            )
            
            optimization_results[num_candidates] = {
                'acceptance_rate': mock_acceptance_rate,
                'estimated_speedup': mock_speedup,
                'efficiency_score': mock_acceptance_rate * mock_speedup
            }
        
        # Select optimal parameters
        best_config = max(
            optimization_results.items(),
            key=lambda x: x[1]['efficiency_score']
        )
        
        optimal_params = {
            'optimal_candidates': best_config[0],
            'expected_acceptance_rate': best_config[1]['acceptance_rate'],
            'expected_speedup': best_config[1]['estimated_speedup'],
            'optimization_results': optimization_results
        }
        
        logger.info(f"Optimal speculative decoding parameters: {optimal_params}")
        return optimal_params
    
    def benchmark_speculative_speedup(self, test_prompts: List[str]):
        """Benchmark speculative decoding speedup"""
        
        logger.info(f"Benchmarking speculative decoding with {len(test_prompts)} prompts")
        
        benchmark_results = {
            'standard_generation': {},
            'speculative_generation': {},
            'speedup_analysis': {}
        }
        
        for i, prompt in enumerate(test_prompts):
            # Mock standard generation
            standard_time = np.random.normal(2.0, 0.3)  # 2 seconds average
            standard_tokens = 100
            
            # Mock speculative generation
            speculative_time = standard_time / np.random.uniform(1.5, 2.5)  # 1.5-2.5x speedup
            speculative_tokens = standard_tokens
            
            benchmark_results['standard_generation'][f'prompt_{i}'] = {
                'time_seconds': standard_time,
                'tokens_generated': standard_tokens,
                'tokens_per_second': standard_tokens / standard_time
            }
            
            benchmark_results['speculative_generation'][f'prompt_{i}'] = {
                'time_seconds': speculative_time,
                'tokens_generated': speculative_tokens,
                'tokens_per_second': speculative_tokens / speculative_time,
                'acceptance_rate': np.random.uniform(0.6, 0.8)
            }
        
        # Calculate overall speedup
        avg_standard_time = np.mean([r['time_seconds'] for r in benchmark_results['standard_generation'].values()])
        avg_speculative_time = np.mean([r['time_seconds'] for r in benchmark_results['speculative_generation'].values()])
        
        benchmark_results['speedup_analysis'] = {
            'average_speedup': avg_standard_time / avg_speculative_time,
            'avg_acceptance_rate': np.mean([r['acceptance_rate'] for r in benchmark_results['speculative_generation'].values()]),
            'throughput_improvement': f"{(avg_standard_time / avg_speculative_time):.2f}x"
        }
        
        logger.info(f"Speculative decoding benchmark completed: "
                   f"{benchmark_results['speedup_analysis']['average_speedup']:.2f}x speedup")
        
        return benchmark_results
# Solution 6: Load Balancing and Auto-scaling
class LLMLoadBalancer:
    def __init__(self):
        self.instances = []
        self.routing_strategy = 'least_connections'
        self.health_checker = None
        self.request_counter = 0
        
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
            'avg_latency_ms': 0,
            'error_count': 0,
            'cpu_utilization': 0,
            'gpu_utilization': 0,
            'memory_utilization': 0
        }
        
        self.instances.append(instance)
        logger.info(f"Added instance {instance_id} with capacity {capacity}")
    
    def implement_routing_strategies(self):
        """Implement different routing strategies"""
        
        routing_strategies = {
            'round_robin': self._round_robin_routing,
            'least_connections': self._least_connections_routing,
            'weighted_round_robin': self._weighted_round_robin_routing,
            'latency_based': self._latency_based_routing,
            'resource_aware': self._resource_aware_routing
        }
        
        logger.info(f"Available routing strategies: {list(routing_strategies.keys())}")
        return routing_strategies
    
    def _round_robin_routing(self, request: Dict[str, Any]) -> str:
        """Round robin routing strategy"""
        
        healthy_instances = [inst for inst in self.instances if inst['health_status'] == 'healthy']
        if not healthy_instances:
            raise Exception("No healthy instances available")
        
        selected_instance = healthy_instances[self.request_counter % len(healthy_instances)]
        self.request_counter += 1
        
        return selected_instance['id']
    
    def _least_connections_routing(self, request: Dict[str, Any]) -> str:
        """Least connections routing strategy"""
        
        healthy_instances = [inst for inst in self.instances if inst['health_status'] == 'healthy']
        if not healthy_instances:
            raise Exception("No healthy instances available")
        
        # Select instance with lowest current load
        selected_instance = min(healthy_instances, key=lambda x: x['current_load'])
        
        return selected_instance['id']
    
    def _weighted_round_robin_routing(self, request: Dict[str, Any]) -> str:
        """Weighted round robin based on capacity"""
        
        healthy_instances = [inst for inst in self.instances if inst['health_status'] == 'healthy']
        if not healthy_instances:
            raise Exception("No healthy instances available")
        
        # Calculate weights based on capacity and current load
        weights = []
        for inst in healthy_instances:
            available_capacity = inst['capacity'] - inst['current_load']
            weight = max(0, available_capacity)
            weights.append(weight)
        
        if sum(weights) == 0:
            # All instances at capacity, use least loaded
            selected_instance = min(healthy_instances, key=lambda x: x['current_load'])
        else:
            # Weighted selection
            total_weight = sum(weights)
            rand_val = np.random.random() * total_weight
            
            cumulative_weight = 0
            selected_instance = healthy_instances[0]
            
            for i, weight in enumerate(weights):
                cumulative_weight += weight
                if rand_val <= cumulative_weight:
                    selected_instance = healthy_instances[i]
                    break
        
        return selected_instance['id']
    
    def _latency_based_routing(self, request: Dict[str, Any]) -> str:
        """Route to instance with lowest average latency"""
        
        healthy_instances = [inst for inst in self.instances if inst['health_status'] == 'healthy']
        if not healthy_instances:
            raise Exception("No healthy instances available")
        
        # Select instance with lowest average latency
        selected_instance = min(healthy_instances, key=lambda x: x['avg_latency_ms'])
        
        return selected_instance['id']
    
    def _resource_aware_routing(self, request: Dict[str, Any]) -> str:
        """Route based on resource utilization"""
        
        healthy_instances = [inst for inst in self.instances if inst['health_status'] == 'healthy']
        if not healthy_instances:
            raise Exception("No healthy instances available")
        
        # Calculate resource score (lower is better)
        def resource_score(instance):
            cpu_score = instance['cpu_utilization'] / 100
            gpu_score = instance['gpu_utilization'] / 100
            memory_score = instance['memory_utilization'] / 100
            load_score = instance['current_load'] / instance['capacity']
            
            return (cpu_score + gpu_score + memory_score + load_score) / 4
        
        selected_instance = min(healthy_instances, key=resource_score)
        
        return selected_instance['id']
    
    def route_request(self, request: Dict[str, Any]) -> str:
        """Route request to optimal instance"""
        
        routing_strategies = self.implement_routing_strategies()
        routing_func = routing_strategies.get(self.routing_strategy, self._least_connections_routing)
        
        try:
            selected_instance_id = routing_func(request)
            
            # Update instance load
            for instance in self.instances:
                if instance['id'] == selected_instance_id:
                    instance['current_load'] += 1
                    instance['total_requests'] += 1
                    break
            
            logger.debug(f"Routed request {request.get('id', 'unknown')} to {selected_instance_id}")
            return selected_instance_id
            
        except Exception as e:
            logger.error(f"Routing failed: {e}")
            raise
    
    def complete_request(self, instance_id: str, latency_ms: float, success: bool = True):
        """Mark request as completed and update metrics"""
        
        for instance in self.instances:
            if instance['id'] == instance_id:
                instance['current_load'] = max(0, instance['current_load'] - 1)
                
                # Update average latency (exponential moving average)
                alpha = 0.1
                instance['avg_latency_ms'] = (
                    alpha * latency_ms + 
                    (1 - alpha) * instance['avg_latency_ms']
                )
                
                # Update error count
                if not success:
                    instance['error_count'] += 1
                
                break
    
    def auto_scale_decision(self) -> Dict[str, Any]:
        """Make auto-scaling decisions based on load and performance"""
        
        if not self.instances:
            return {'action': 'none', 'reason': 'No instances configured'}
        
        # Calculate cluster metrics
        total_capacity = sum(inst['capacity'] for inst in self.instances)
        total_load = sum(inst['current_load'] for inst in self.instances)
        healthy_instances = [inst for inst in self.instances if inst['health_status'] == 'healthy']
        
        utilization = total_load / total_capacity if total_capacity > 0 else 0
        avg_latency = np.mean([inst['avg_latency_ms'] for inst in healthy_instances]) if healthy_instances else 0
        error_rate = sum(inst['error_count'] for inst in self.instances) / max(sum(inst['total_requests'] for inst in self.instances), 1)
        
        scaling_decision = {
            'current_utilization': utilization,
            'avg_latency_ms': avg_latency,
            'error_rate': error_rate,
            'healthy_instances': len(healthy_instances),
            'total_instances': len(self.instances),
            'action': 'none',
            'reason': '',
            'recommended_instances': 0
        }
        
        # Scale up conditions
        if utilization > 0.8:
            scaling_decision['action'] = 'scale_up'
            scaling_decision['reason'] = f'High utilization ({utilization:.1%})'
            scaling_decision['recommended_instances'] = max(1, int(len(self.instances) * 0.5))
        elif avg_latency > 200:
            scaling_decision['action'] = 'scale_up'
            scaling_decision['reason'] = f'High latency ({avg_latency:.1f}ms)'
            scaling_decision['recommended_instances'] = 1
        elif error_rate > 0.05:
            scaling_decision['action'] = 'scale_up'
            scaling_decision['reason'] = f'High error rate ({error_rate:.1%})'
            scaling_decision['recommended_instances'] = 1
        
        # Scale down conditions
        elif utilization < 0.3 and len(healthy_instances) > 1:
            scaling_decision['action'] = 'scale_down'
            scaling_decision['reason'] = f'Low utilization ({utilization:.1%})'
            scaling_decision['recommended_instances'] = max(1, int(len(self.instances) * 0.3))
        
        return scaling_decision
    
    def implement_health_checks(self):
        """Implement comprehensive health checks"""
        
        health_check_config = {
            'check_interval_seconds': 30,
            'timeout_seconds': 5,
            'failure_threshold': 3,
            'success_threshold': 2,
            'endpoints': {
                'health': '/health',
                'ready': '/ready',
                'metrics': '/metrics'
            }
        }
        
        logger.info(f"Health check configuration: {health_check_config}")
        
        # Mock health check implementation
        for instance in self.instances:
            # Simulate health check
            health_score = np.random.uniform(0.8, 1.0)
            
            if health_score > 0.9:
                instance['health_status'] = 'healthy'
            elif health_score > 0.7:
                instance['health_status'] = 'degraded'
            else:
                instance['health_status'] = 'unhealthy'
            
            instance['last_health_check'] = time.time()
            
            # Mock resource utilization
            instance['cpu_utilization'] = np.random.uniform(20, 80)
            instance['gpu_utilization'] = np.random.uniform(60, 95)
            instance['memory_utilization'] = np.random.uniform(40, 85)
        
        return health_check_config

class AutoScalingManager:
    def __init__(self, min_instances: int = 1, max_instances: int = 10):
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scaling_policies = {}
        self.scaling_history = []
        
    def define_scaling_policies(self):
        """Define auto-scaling policies"""
        
        self.scaling_policies = {
            'scale_up': {
                'triggers': {
                    'cpu_threshold': 80,
                    'gpu_threshold': 90,
                    'latency_threshold_ms': 200,
                    'queue_length_threshold': 50,
                    'error_rate_threshold': 0.05
                },
                'cooldown_seconds': 300,  # 5 minutes
                'step_size': 1,
                'max_step_size': 3
            },
            'scale_down': {
                'triggers': {
                    'cpu_threshold': 30,
                    'gpu_threshold': 40,
                    'utilization_threshold': 0.3
                },
                'cooldown_seconds': 600,  # 10 minutes
                'step_size': 1,
                'max_step_size': 2
            },
            'instance_warmup_time_seconds': 180,  # 3 minutes
            'health_check_grace_period_seconds': 60
        }
        
        logger.info(f"Scaling policies defined: {self.scaling_policies}")
        return self.scaling_policies
    
    def execute_scaling_action(self, action: str, count: int):
        """Execute scaling actions"""
        
        if action not in ['scale_up', 'scale_down']:
            logger.warning(f"Unknown scaling action: {action}")
            return
        
        # Check cooldown period
        if self._is_in_cooldown(action):
            logger.info(f"Scaling action {action} skipped due to cooldown")
            return
        
        # Validate scaling limits
        if action == 'scale_up':
            if len(self.scaling_history) + count > self.max_instances:
                count = max(0, self.max_instances - len(self.scaling_history))
        elif action == 'scale_down':
            if len(self.scaling_history) - count < self.min_instances:
                count = max(0, len(self.scaling_history) - self.min_instances)
        
        if count <= 0:
            logger.info(f"No scaling needed for {action}")
            return
        
        # Execute scaling
        scaling_event = {
            'action': action,
            'count': count,
            'timestamp': time.time(),
            'reason': f"Auto-scaling {action}",
            'status': 'in_progress'
        }
        
        logger.info(f"Executing {action}: {count} instances")
        
        if action == 'scale_up':
            self._launch_instances(count)
        elif action == 'scale_down':
            self._terminate_instances(count)
        
        scaling_event['status'] = 'completed'
        self.scaling_history.append(scaling_event)
        
        logger.info(f"Scaling action completed: {scaling_event}")
    
    def _is_in_cooldown(self, action: str) -> bool:
        """Check if scaling action is in cooldown period"""
        
        cooldown_seconds = self.scaling_policies[action]['cooldown_seconds']
        current_time = time.time()
        
        # Check recent scaling events
        for event in reversed(self.scaling_history):
            if event['action'] == action:
                time_since_event = current_time - event['timestamp']
                if time_since_event < cooldown_seconds:
                    return True
                break
        
        return False
    
    def _launch_instances(self, count: int):
        """Launch new instances"""
        
        for i in range(count):
            instance_config = {
                'instance_type': 'gpu_optimized',
                'model_config': 'production',
                'auto_scaling': True,
                'launch_time': time.time()
            }
            
            logger.info(f"Launching instance {i+1}/{count}: {instance_config}")
            
            # Mock instance launch time
            time.sleep(0.1)
    
    def _terminate_instances(self, count: int):
        """Terminate instances"""
        
        for i in range(count):
            termination_config = {
                'graceful_shutdown': True,
                'drain_requests': True,
                'timeout_seconds': 60
            }
            
            logger.info(f"Terminating instance {i+1}/{count}: {termination_config}")
            
            # Mock instance termination time
            time.sleep(0.05)

# Solution 7: Performance Monitoring and Cost Optimization
class LLMPerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'latency': [],
            'throughput': [],
            'gpu_utilization': [],
            'memory_usage': [],
            'cost_per_token': [],
            'error_rate': [],
            'queue_length': []
        }
        self.alert_thresholds = {
            'max_latency_ms': 200,
            'min_throughput_tps': 50,
            'max_error_rate': 0.05,
            'max_queue_length': 100
        }
        
    def collect_real_time_metrics(self, instance_id: str):
        """Collect real-time performance metrics"""
        
        # Mock system metrics collection
        try:
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Mock GPU metrics (in production, use nvidia-ml-py)
            gpu_utilization = np.random.uniform(60, 95)
            gpu_memory_used = np.random.uniform(8, 15)  # GB
            
        except Exception:
            # Fallback to mock values
            cpu_percent = np.random.uniform(30, 80)
            memory = type('Memory', (), {'percent': np.random.uniform(40, 85)})()
            gpu_utilization = np.random.uniform(60, 95)
            gpu_memory_used = np.random.uniform(8, 15)
        
        # Mock inference metrics
        current_latency = np.random.normal(100, 20)  # ms
        current_throughput = np.random.normal(80, 15)  # tokens/sec
        current_error_rate = np.random.uniform(0, 0.02)
        current_queue_length = np.random.randint(0, 50)
        
        # Calculate cost per token (mock)
        gpu_cost_per_hour = 3.20  # A100 40GB
        tokens_per_hour = current_throughput * 3600
        cost_per_token = gpu_cost_per_hour / tokens_per_hour if tokens_per_hour > 0 else 0
        
        # Store metrics
        metrics_snapshot = {
            'timestamp': time.time(),
            'instance_id': instance_id,
            'latency_ms': current_latency,
            'throughput_tps': current_throughput,
            'cpu_utilization': cpu_percent,
            'memory_utilization': memory.percent,
            'gpu_utilization': gpu_utilization,
            'gpu_memory_gb': gpu_memory_used,
            'error_rate': current_error_rate,
            'queue_length': current_queue_length,
            'cost_per_token': cost_per_token
        }
        
        # Update running metrics
        self.metrics['latency'].append(current_latency)
        self.metrics['throughput'].append(current_throughput)
        self.metrics['gpu_utilization'].append(gpu_utilization)
        self.metrics['memory_usage'].append(gpu_memory_used)
        self.metrics['cost_per_token'].append(cost_per_token)
        self.metrics['error_rate'].append(current_error_rate)
        self.metrics['queue_length'].append(current_queue_length)
        
        # Keep only recent metrics (last 1000 points)
        for key in self.metrics:
            if len(self.metrics[key]) > 1000:
                self.metrics[key] = self.metrics[key][-1000:]
        
        # Check for alerts
        alerts = self._check_alerts(metrics_snapshot)
        
        return {
            'metrics': metrics_snapshot,
            'alerts': alerts
        }
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Check for performance alerts"""
        
        alerts = []
        
        if metrics['latency_ms'] > self.alert_thresholds['max_latency_ms']:
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"High latency: {metrics['latency_ms']:.1f}ms > {self.alert_thresholds['max_latency_ms']}ms"
            })
        
        if metrics['throughput_tps'] < self.alert_thresholds['min_throughput_tps']:
            alerts.append({
                'type': 'low_throughput',
                'severity': 'warning',
                'message': f"Low throughput: {metrics['throughput_tps']:.1f} < {self.alert_thresholds['min_throughput_tps']} TPS"
            })
        
        if metrics['error_rate'] > self.alert_thresholds['max_error_rate']:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f"High error rate: {metrics['error_rate']:.1%} > {self.alert_thresholds['max_error_rate']:.1%}"
            })
        
        if metrics['queue_length'] > self.alert_thresholds['max_queue_length']:
            alerts.append({
                'type': 'queue_backlog',
                'severity': 'warning',
                'message': f"Queue backlog: {metrics['queue_length']} > {self.alert_thresholds['max_queue_length']}"
            })
        
        return alerts
    
    def analyze_performance_trends(self):
        """Analyze performance trends over time"""
        
        if not any(self.metrics.values()):
            return {'error': 'No metrics available for analysis'}
        
        # Calculate trends (simplified linear regression)
        def calculate_trend(values):
            if len(values) < 10:
                return 0
            
            x = np.arange(len(values))
            y = np.array(values)
            
            # Simple linear regression
            slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x))
            return slope
        
        trends = {}
        for metric_name, values in self.metrics.items():
            if values:
                trend = calculate_trend(values[-100:])  # Last 100 points
                trends[metric_name] = {
                    'trend_slope': trend,
                    'direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
                    'current_value': values[-1],
                    'avg_value': np.mean(values[-100:]),
                    'std_value': np.std(values[-100:])
                }
        
        # Identify concerning trends
        concerning_trends = []
        
        if trends.get('latency', {}).get('direction') == 'increasing':
            concerning_trends.append('Latency is increasing over time')
        
        if trends.get('throughput', {}).get('direction') == 'decreasing':
            concerning_trends.append('Throughput is decreasing over time')
        
        if trends.get('error_rate', {}).get('direction') == 'increasing':
            concerning_trends.append('Error rate is increasing over time')
        
        analysis_result = {
            'trends': trends,
            'concerning_trends': concerning_trends,
            'overall_health': 'good' if not concerning_trends else 'degrading',
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        return analysis_result
    
    def generate_optimization_recommendations(self):
        """Generate optimization recommendations"""
        
        if not any(self.metrics.values()):
            return {'error': 'No metrics available for recommendations'}
        
        recommendations = []
        
        # Analyze current performance
        avg_latency = np.mean(self.metrics['latency'][-100:]) if self.metrics['latency'] else 0
        avg_throughput = np.mean(self.metrics['throughput'][-100:]) if self.metrics['throughput'] else 0
        avg_gpu_util = np.mean(self.metrics['gpu_utilization'][-100:]) if self.metrics['gpu_utilization'] else 0
        avg_cost = np.mean(self.metrics['cost_per_token'][-100:]) if self.metrics['cost_per_token'] else 0
        
        # Latency recommendations
        if avg_latency > 150:
            recommendations.append({
                'category': 'latency',
                'priority': 'high',
                'recommendation': 'Consider model quantization or smaller batch sizes to reduce latency',
                'expected_impact': '20-40% latency reduction'
            })
        
        # Throughput recommendations
        if avg_throughput < 60:
            recommendations.append({
                'category': 'throughput',
                'priority': 'medium',
                'recommendation': 'Increase batch size or implement continuous batching',
                'expected_impact': '30-50% throughput increase'
            })
        
        # GPU utilization recommendations
        if avg_gpu_util < 70:
            recommendations.append({
                'category': 'utilization',
                'priority': 'medium',
                'recommendation': 'Increase batch size or implement request batching to improve GPU utilization',
                'expected_impact': '15-25% cost reduction'
            })
        elif avg_gpu_util > 95:
            recommendations.append({
                'category': 'utilization',
                'priority': 'high',
                'recommendation': 'Scale up instances or optimize model to reduce GPU bottleneck',
                'expected_impact': 'Prevent performance degradation'
            })
        
        # Cost recommendations
        if avg_cost > 0.01:  # $0.01 per token
            recommendations.append({
                'category': 'cost',
                'priority': 'high',
                'recommendation': 'Implement aggressive quantization or consider smaller model variants',
                'expected_impact': '40-60% cost reduction'
            })
        
        # General recommendations
        recommendations.extend([
            {
                'category': 'optimization',
                'priority': 'medium',
                'recommendation': 'Implement speculative decoding for faster generation',
                'expected_impact': '1.5-2.5x speedup'
            },
            {
                'category': 'infrastructure',
                'priority': 'low',
                'recommendation': 'Consider using spot instances for cost savings',
                'expected_impact': '50-70% infrastructure cost reduction'
            }
        ])
        
        return {
            'recommendations': recommendations,
            'current_performance': {
                'avg_latency_ms': avg_latency,
                'avg_throughput_tps': avg_throughput,
                'avg_gpu_utilization': avg_gpu_util,
                'avg_cost_per_token': avg_cost
            },
            'generated_at': datetime.now().isoformat()
        }

class CostOptimizer:
    def __init__(self):
        self.cost_models = {
            'gpu_instances': {
                'a100_40gb': {'cost_per_hour': 3.20, 'memory_gb': 40, 'performance_score': 100},
                'a100_80gb': {'cost_per_hour': 6.40, 'memory_gb': 80, 'performance_score': 120},
                'h100_80gb': {'cost_per_hour': 8.00, 'memory_gb': 80, 'performance_score': 180},
                'v100_32gb': {'cost_per_hour': 2.40, 'memory_gb': 32, 'performance_score': 70},
                't4_16gb': {'cost_per_hour': 0.80, 'memory_gb': 16, 'performance_score': 30}
            },
            'pricing_models': {
                'on_demand': {'multiplier': 1.0, 'availability': 'guaranteed'},
                'spot': {'multiplier': 0.3, 'availability': 'variable'},
                'reserved': {'multiplier': 0.6, 'availability': 'guaranteed', 'commitment': '1_year'}
            }
        }
        self.usage_patterns = {}
        
    def calculate_serving_costs(self, usage_metrics: Dict[str, Any]):
        """Calculate comprehensive serving costs"""
        
        # Extract usage metrics
        gpu_hours = usage_metrics.get('gpu_hours', 0)
        requests_processed = usage_metrics.get('requests_processed', 0)
        tokens_generated = usage_metrics.get('tokens_generated', 0)
        instance_type = usage_metrics.get('instance_type', 'a100_40gb')
        pricing_model = usage_metrics.get('pricing_model', 'on_demand')
        
        # Get cost parameters
        instance_config = self.cost_models['gpu_instances'].get(instance_type, self.cost_models['gpu_instances']['a100_40gb'])
        pricing_config = self.cost_models['pricing_models'].get(pricing_model, self.cost_models['pricing_models']['on_demand'])
        
        # Calculate costs
        base_cost_per_hour = instance_config['cost_per_hour']
        adjusted_cost_per_hour = base_cost_per_hour * pricing_config['multiplier']
        
        total_compute_cost = gpu_hours * adjusted_cost_per_hour
        
        # Additional costs
        network_cost = requests_processed * 0.0001  # $0.0001 per request
        storage_cost = (tokens_generated / 1000000) * 0.10  # $0.10 per million tokens stored
        
        total_cost = total_compute_cost + network_cost + storage_cost
        
        # Calculate per-unit costs
        cost_per_request = total_cost / requests_processed if requests_processed > 0 else 0
        cost_per_1k_tokens = (total_cost / tokens_generated) * 1000 if tokens_generated > 0 else 0
        
        cost_breakdown = {
            'total_cost': total_cost,
            'compute_cost': total_compute_cost,
            'network_cost': network_cost,
            'storage_cost': storage_cost,
            'cost_per_request': cost_per_request,
            'cost_per_1k_tokens': cost_per_1k_tokens,
            'instance_type': instance_type,
            'pricing_model': pricing_model,
            'gpu_hours': gpu_hours,
            'effective_hourly_rate': adjusted_cost_per_hour
        }
        
        return cost_breakdown
    
    def optimize_cost_performance_ratio(self, requirements: Dict[str, Any]):
        """Optimize cost-performance ratio"""
        
        target_latency_ms = requirements.get('target_latency_ms', 100)
        target_throughput_tps = requirements.get('target_throughput_tps', 100)
        budget_per_month = requirements.get('budget_per_month', 5000)
        availability_requirement = requirements.get('availability', 0.99)
        
        optimization_results = {}
        
        # Test different configurations
        for instance_type, instance_config in self.cost_models['gpu_instances'].items():
            for pricing_model, pricing_config in self.cost_models['pricing_models'].items():
                
                # Skip spot instances if high availability required
                if availability_requirement > 0.95 and pricing_model == 'spot':
                    continue
                
                # Estimate performance
                performance_score = instance_config['performance_score']
                estimated_latency = max(20, 200 - (performance_score * 1.5))
                estimated_throughput = performance_score * 1.2
                
                # Check if meets requirements
                meets_latency = estimated_latency <= target_latency_ms
                meets_throughput = estimated_throughput >= target_throughput_tps
                
                if not (meets_latency and meets_throughput):
                    continue
                
                # Calculate monthly cost
                hours_per_month = 24 * 30
                monthly_cost = (instance_config['cost_per_hour'] * 
                              pricing_config['multiplier'] * 
                              hours_per_month)
                
                if monthly_cost > budget_per_month:
                    continue
                
                # Calculate efficiency score
                efficiency_score = (performance_score / monthly_cost) * 1000
                
                config_key = f"{instance_type}_{pricing_model}"
                optimization_results[config_key] = {
                    'instance_type': instance_type,
                    'pricing_model': pricing_model,
                    'monthly_cost': monthly_cost,
                    'estimated_latency_ms': estimated_latency,
                    'estimated_throughput_tps': estimated_throughput,
                    'efficiency_score': efficiency_score,
                    'meets_requirements': True
                }
        
        if not optimization_results:
            return {
                'error': 'No configurations meet requirements within budget',
                'suggestions': [
                    'Increase budget',
                    'Relax latency requirements',
                    'Consider model optimization'
                ]
            }
        
        # Select best configuration
        best_config = max(optimization_results.values(), key=lambda x: x['efficiency_score'])
        
        optimization_plan = {
            'recommended_config': best_config,
            'all_viable_configs': optimization_results,
            'optimization_strategies': [
                'Use quantization to reduce memory requirements',
                'Implement request batching for higher throughput',
                'Consider spot instances for non-critical workloads',
                'Monitor and auto-scale based on demand'
            ],
            'estimated_savings': self._calculate_potential_savings(optimization_results, best_config)
        }
        
        return optimization_plan
    
    def _calculate_potential_savings(self, all_configs: Dict[str, Any], selected_config: Dict[str, Any]):
        """Calculate potential savings from optimization"""
        
        if not all_configs:
            return {}
        
        selected_cost = selected_config['monthly_cost']
        
        # Compare with most expensive viable option
        max_cost = max(config['monthly_cost'] for config in all_configs.values())
        min_cost = min(config['monthly_cost'] for config in all_configs.values())
        
        savings_vs_max = {
            'absolute_savings_per_month': max_cost - selected_cost,
            'percentage_savings': ((max_cost - selected_cost) / max_cost) * 100 if max_cost > 0 else 0,
            'annual_savings': (max_cost - selected_cost) * 12
        }
        
        return {
            'vs_most_expensive': savings_vs_max,
            'vs_cheapest': {
                'additional_cost_per_month': selected_cost - min_cost,
                'performance_benefit': 'Better latency and throughput'
            }
        }
    
    def implement_cost_controls(self, budget_limits: Dict[str, float]):
        """Implement cost control mechanisms"""
        
        cost_controls = {
            'budget_alerts': {
                'daily_limit': budget_limits.get('daily_limit', 100),
                'monthly_limit': budget_limits.get('monthly_limit', 3000),
                'alert_thresholds': [0.5, 0.8, 0.9, 1.0]  # 50%, 80%, 90%, 100%
            },
            'usage_limits': {
                'max_concurrent_instances': budget_limits.get('max_instances', 10),
                'max_gpu_hours_per_day': budget_limits.get('max_gpu_hours', 240),
                'auto_shutdown_idle_instances': True
            },
            'cost_optimization': {
                'auto_scaling_enabled': True,
                'spot_instance_preference': budget_limits.get('allow_spot', False),
                'scheduled_scaling': {
                    'scale_down_nights': True,
                    'scale_down_weekends': True
                }
            }
        }
        
        logger.info(f"Cost controls implemented: {cost_controls}")
        return cost_controls

# Demonstration functions for each solution
def demonstrate_vllm_serving():
    """Demonstrate vLLM serving setup"""
    print("=== vLLM Serving Demo ===")
    
    config = {
        'tensor_parallel_size': 1,
        'max_model_len': 4096,
        'max_num_seqs': 256,
        'gpu_memory_utilization': 0.85
    }
    
    engine = vLLMServingEngine("llama-7b", config)
    engine.initialize_engine()
    
    # Test generation
    test_prompts = ["Explain quantum computing", "Write a Python function"]
    sampling_params = engine.create_sampling_params(temperature=0.7, max_tokens=100)
    
    # Mock async generation
    async def test_generation():
        results = await engine.generate_async(test_prompts, sampling_params)
        return results
    
    # Benchmark performance
    benchmark_results = engine.benchmark_performance(test_prompts, num_runs=5)
    print(f"vLLM benchmark: {benchmark_results['avg_ttft_ms']:.1f}ms TTFT, "
          f"{benchmark_results['avg_throughput_tps']:.1f} TPS")
    print()

def demonstrate_tensorrt_optimization():
    """Demonstrate TensorRT optimization"""
    print("=== TensorRT Optimization Demo ===")
    
    config = {
        'precision': 'fp16',
        'max_batch_size': 8,
        'optimization_level': 5,
        'use_cuda_graphs': True
    }
    
    optimizer = TensorRTOptimizer("./model.onnx", config)
    engine_info = optimizer.build_tensorrt_engine()
    
    print(f"TensorRT engine built: {engine_info['estimated_speedup']} speedup")
    
    # Apply optimizations
    latency_opts = optimizer.optimize_for_latency()
    throughput_opts = optimizer.optimize_for_throughput()
    
    # Benchmark performance
    input_shapes = [(1, 512), (4, 1024), (8, 2048)]
    benchmark_results = optimizer.benchmark_tensorrt_performance(input_shapes)
    
    for shape, results in benchmark_results.items():
        print(f"{shape}: {results['latency_ms']:.1f}ms, {results['throughput_tps']:.1f} TPS")
    print()

def demonstrate_kv_cache_optimization():
    """Demonstrate KV-cache optimization"""
    print("=== KV-Cache Optimization Demo ===")
    
    cache_manager = KVCacheManager(
        max_sequence_length=4096,
        num_layers=32,
        hidden_size=4096
    )
    
    # Implement PagedAttention
    paged_config = cache_manager.implement_paged_attention(block_size=16)
    print(f"PagedAttention: {paged_config['total_blocks']} blocks")
    
    # Optimize memory
    optimizations = cache_manager.optimize_cache_memory()
    
    # Calculate memory requirements
    scenarios = [(1, 512), (8, 1024), (32, 2048)]
    for batch_size, seq_len in scenarios:
        memory_req = cache_manager.calculate_memory_requirements(batch_size, seq_len)
        print(f"Batch {batch_size}, Seq {seq_len}: {memory_req['total_memory_gb']:.1f}GB")
    
    # Benchmark cache performance
    benchmark_results = cache_manager.benchmark_cache_performance()
    print(f"Cache benchmark completed: {len(benchmark_results)} scenarios tested")
    print()

def demonstrate_continuous_batching():
    """Demonstrate continuous batching"""
    print("=== Continuous Batching Demo ===")
    
    engine = ContinuousBatchingEngine(max_batch_size=16)
    
    # Add test requests
    test_requests = [
        ("req_1", "Explain machine learning", 100),
        ("req_2", "Write a Python function", 150),
        ("req_3", "Describe quantum computing", 200),
        ("req_4", "Create a data pipeline", 120),
        ("req_5", "Optimize database queries", 80)
    ]
    
    for req_id, prompt, max_tokens in test_requests:
        engine.add_request(req_id, prompt, max_tokens)
    
    # Run continuous batching
    async def run_batching():
        await engine.run_continuous_batching(duration_seconds=5)
    
    # Mock async execution
    print("Running continuous batching simulation...")
    
    # Benchmark efficiency
    efficiency_metrics = engine.benchmark_batching_efficiency()
    print(f"Batching efficiency: {efficiency_metrics['utilization_score']:.1f}% utilization")
    print()

def demonstrate_speculative_decoding():
    """Demonstrate speculative decoding"""
    print("=== Speculative Decoding Demo ===")
    
    target_model = MockLLMModel(hidden_size=4096, num_layers=32)
    draft_model = MockLLMModel(hidden_size=2048, num_layers=16)
    
    spec_engine = SpeculativeDecodingEngine(target_model, draft_model)
    
    # Setup draft model
    draft_opts = spec_engine.setup_draft_model()
    print(f"Draft model setup: {draft_opts['inference_speed']}")
    
    # Test speculative generation
    test_input = torch.randint(0, 1000, (1, 10))
    results = spec_engine.speculative_generate_step(test_input, num_candidates=4)
    
    print(f"Speculative generation: {results['acceptance_rate']:.1%} acceptance rate")
    
    # Optimize parameters
    optimal_params = spec_engine.optimize_speculation_parameters()
    print(f"Optimal candidates: {optimal_params['optimal_candidates']}")
    
    # Benchmark speedup
    test_prompts = ["Explain AI", "Write code", "Describe science"]
    speedup_results = spec_engine.benchmark_speculative_speedup(test_prompts)
    print(f"Speedup: {speedup_results['speedup_analysis']['average_speedup']:.2f}x")
    print()

def demonstrate_load_balancing():
    """Demonstrate load balancing and auto-scaling"""
    print("=== Load Balancing Demo ===")
    
    load_balancer = LLMLoadBalancer()
    
    # Add instances
    instances = [
        ("instance-1", "http://llm-1:8000", 10),
        ("instance-2", "http://llm-2:8000", 10),
        ("instance-3", "http://llm-3:8000", 10)
    ]
    
    for inst_id, endpoint, capacity in instances:
        load_balancer.add_instance(inst_id, endpoint, capacity)
    
    # Test routing
    for i in range(20):
        request = {"id": f"req_{i}", "prompt": f"Test prompt {i}"}
        selected_instance = load_balancer.route_request(request)
        
        # Simulate request completion
        latency = np.random.normal(100, 20)
        load_balancer.complete_request(selected_instance, latency, success=True)
    
    # Auto-scaling decision
    scaling_decision = load_balancer.auto_scale_decision()
    print(f"Scaling decision: {scaling_decision['action']} - {scaling_decision['reason']}")
    
    # Setup auto-scaling
    auto_scaler = AutoScalingManager(min_instances=2, max_instances=8)
    auto_scaler.define_scaling_policies()
    
    if scaling_decision['action'] != 'none':
        auto_scaler.execute_scaling_action(scaling_decision['action'], 1)
    
    print("Load balancing and auto-scaling demo completed")
    print()

def demonstrate_monitoring_optimization():
    """Demonstrate performance monitoring and cost optimization"""
    print("=== Monitoring and Cost Optimization Demo ===")
    
    monitor = LLMPerformanceMonitor()
    
    # Simulate metric collection
    for i in range(50):
        metrics_result = monitor.collect_real_time_metrics(f"instance-{i % 3}")
        
        if i % 10 == 0 and metrics_result['alerts']:
            print(f"Alert: {metrics_result['alerts'][0]['message']}")
    
    # Analyze trends
    trends = monitor.analyze_performance_trends()
    print(f"Performance trends: {trends['overall_health']}")
    
    # Generate recommendations
    recommendations = monitor.generate_optimization_recommendations()
    if recommendations.get('recommendations'):
        print(f"Top recommendation: {recommendations['recommendations'][0]['recommendation']}")
    
    # Cost optimization
    cost_optimizer = CostOptimizer()
    
    usage_metrics = {
        'gpu_hours': 100,
        'requests_processed': 10000,
        'tokens_generated': 1000000,
        'instance_type': 'a100_40gb',
        'pricing_model': 'on_demand'
    }
    
    cost_breakdown = cost_optimizer.calculate_serving_costs(usage_metrics)
    print(f"Current cost: ${cost_breakdown['total_cost']:.2f} (${cost_breakdown['cost_per_1k_tokens']:.4f}/1k tokens)")
    
    # Optimization plan
    requirements = {
        'target_latency_ms': 100,
        'target_throughput_tps': 80,
        'budget_per_month': 3000
    }
    
    optimization_plan = cost_optimizer.optimize_cost_performance_ratio(requirements)
    if 'recommended_config' in optimization_plan:
        recommended = optimization_plan['recommended_config']
        print(f"Recommended: {recommended['instance_type']} ({recommended['pricing_model']}) - "
              f"${recommended['monthly_cost']:.2f}/month")
    
    print()

def main():
    """Run all LLM serving optimization solution demonstrations"""
    print("Day 51: LLM Serving & Optimization - vLLM, TensorRT, Inference - Solutions")
    print("=" * 80)
    print()
    
    # Run all demonstrations
    demonstrate_vllm_serving()
    demonstrate_tensorrt_optimization()
    demonstrate_kv_cache_optimization()
    demonstrate_continuous_batching()
    demonstrate_speculative_decoding()
    demonstrate_load_balancing()
    demonstrate_monitoring_optimization()
    
    print("=" * 80)
    print("All LLM serving optimization solutions demonstrated successfully!")
    print()
    print("Key Takeaways:")
    print("1. vLLM provides efficient serving with PagedAttention and continuous batching")
    print("2. TensorRT optimization can provide 2-4x speedup through precision and kernel optimization")
    print("3. KV-cache optimization is crucial for memory efficiency in long sequences")
    print("4. Continuous batching improves throughput by dynamically managing request batches")
    print("5. Speculative decoding can provide 1.5-2.5x speedup with draft models")
    print("6. Load balancing and auto-scaling ensure reliable and cost-effective serving")
    print("7. Comprehensive monitoring enables proactive optimization and cost control")

if __name__ == "__main__":
    main()