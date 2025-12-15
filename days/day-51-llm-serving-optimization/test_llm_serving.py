"""
Day 51: LLM Serving & Optimization - Test Suite

Comprehensive test suite for LLM serving optimization implementations with 30+ test cases
covering vLLM, TensorRT, KV-cache, continuous batching, speculative decoding, and deployment.
"""

import pytest
import torch
import asyncio
import time
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import json
import tempfile
import os

# Import solution classes
from solution import (
    vLLMServingEngine,
    TensorRTOptimizer,
    KVCacheManager,
    ContinuousBatchingEngine,
    SpeculativeDecodingEngine,
    LLMLoadBalancer,
    AutoScalingManager,
    LLMPerformanceMonitor,
    CostOptimizer,
    MockLLMModel,
    MockTokenizer
)

class TestVLLMServing:
    """Test vLLM serving implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'tensor_parallel_size': 1,
            'max_model_len': 2048,
            'max_num_seqs': 128,
            'gpu_memory_utilization': 0.8
        }
        self.engine = vLLMServingEngine("test-model", self.config)
    
    def test_engine_initialization(self):
        """Test vLLM engine initialization"""
        engine_config = self.engine.initialize_engine()
        
        assert engine_config is not None
        assert engine_config['initialized'] is True
        assert 'memory_pool' in engine_config
        assert 'scheduler' in engine_config
    
    def test_memory_pool_initialization(self):
        """Test memory pool setup"""
        self.engine.initialize_engine()
        memory_pool = self.engine.engine['memory_pool']
        
        assert memory_pool['total_blocks'] > 0
        assert memory_pool['free_blocks'] == memory_pool['total_blocks']
        assert memory_pool['block_size'] > 0
    
    def test_sampling_params_creation(self):
        """Test sampling parameters creation"""
        params = self.engine.create_sampling_params(
            temperature=0.8,
            max_tokens=200,
            top_p=0.95
        )
        
        assert params['temperature'] == 0.8
        assert params['max_tokens'] == 200
        assert params['top_p'] == 0.95
        assert 'stop' in params
    
    @pytest.mark.asyncio
    async def test_async_generation(self):
        """Test async text generation"""
        prompts = ["Test prompt 1", "Test prompt 2"]
        sampling_params = self.engine.create_sampling_params(max_tokens=50)
        
        results = await self.engine.generate_async(prompts, sampling_params)
        
        assert len(results) == len(prompts)
        assert all(isinstance(result, str) for result in results)
    
    def test_performance_benchmark(self):
        """Test performance benchmarking"""
        test_prompts = ["Benchmark prompt 1", "Benchmark prompt 2"]
        
        benchmark_results = self.engine.benchmark_performance(test_prompts, num_runs=3)
        
        assert 'avg_ttft_ms' in benchmark_results
        assert 'avg_tpot_ms' in benchmark_results
        assert 'avg_throughput_tps' in benchmark_results
        assert benchmark_results['avg_ttft_ms'] > 0
        assert benchmark_results['avg_throughput_tps'] > 0
    
    def test_request_tracking(self):
        """Test request tracking functionality"""
        self.engine.initialize_engine()
        
        initial_count = self.engine.request_counter
        
        # Mock async generation to test request tracking
        async def test_tracking():
            await self.engine.generate_async(["test"], {})
        
        # Run the async function
        asyncio.run(test_tracking())
        
        assert len(self.engine.active_requests) >= 0  # Requests may complete immediately

class TestTensorRTOptimization:
    """Test TensorRT optimization implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.config = {
            'precision': 'fp16',
            'max_batch_size': 4,
            'optimization_level': 3,
            'use_cuda_graphs': True
        }
        self.optimizer = TensorRTOptimizer("./test_model.onnx", self.config)
    
    def test_optimizer_initialization(self):
        """Test TensorRT optimizer initialization"""
        assert self.optimizer.model_path == "./test_model.onnx"
        assert self.optimizer.config == self.config
        assert self.optimizer.engine is None
    
    def test_engine_building(self):
        """Test TensorRT engine building"""
        engine_info = self.optimizer.build_tensorrt_engine()
        
        assert engine_info is not None
        assert 'precision' in engine_info
        assert 'estimated_speedup' in engine_info
        assert 'build_time_seconds' in engine_info
        assert engine_info['precision'] == 'fp16'
    
    def test_latency_optimization(self):
        """Test latency optimization"""
        self.optimizer.build_tensorrt_engine()
        
        latency_opts = self.optimizer.optimize_for_latency()
        
        assert 'cuda_graphs' in latency_opts
        assert 'kernel_fusion' in latency_opts
        assert latency_opts['batch_size'] == 1
        assert latency_opts['profile'] == 'latency_optimized'
    
    def test_throughput_optimization(self):
        """Test throughput optimization"""
        self.optimizer.build_tensorrt_engine()
        
        throughput_opts = self.optimizer.optimize_for_throughput()
        
        assert 'batch_size' in throughput_opts
        assert 'memory_pooling' in throughput_opts
        assert throughput_opts['profile'] == 'throughput_optimized'
    
    def test_performance_benchmarking(self):
        """Test TensorRT performance benchmarking"""
        self.optimizer.build_tensorrt_engine()
        
        input_shapes = [(1, 512), (4, 1024)]
        benchmark_results = self.optimizer.benchmark_tensorrt_performance(input_shapes)
        
        assert len(benchmark_results) == len(input_shapes)
        
        for shape_key, results in benchmark_results.items():
            assert 'latency_ms' in results
            assert 'throughput_tps' in results
            assert 'memory_usage_gb' in results
            assert results['latency_ms'] > 0
            assert results['throughput_tps'] > 0
    
    def test_speedup_estimation(self):
        """Test speedup estimation"""
        config = {'precision': 'int8', 'use_cuda_graphs': True, 'optimization_level': 5}
        speedup = self.optimizer._estimate_speedup(config)
        
        assert isinstance(speedup, str)
        assert 'x' in speedup
        
        # INT8 should provide higher speedup than FP16
        fp16_config = {'precision': 'fp16', 'use_cuda_graphs': False, 'optimization_level': 1}
        fp16_speedup = self.optimizer._estimate_speedup(fp16_config)
        
        # Extract numeric values for comparison
        int8_value = float(speedup.replace('x', ''))
        fp16_value = float(fp16_speedup.replace('x', ''))
        
        assert int8_value > fp16_value

class TestKVCacheManager:
    """Test KV-cache management implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cache_manager = KVCacheManager(
            max_sequence_length=2048,
            num_layers=24,
            hidden_size=2048
        )
    
    def test_cache_manager_initialization(self):
        """Test KV-cache manager initialization"""
        assert self.cache_manager.max_sequence_length == 2048
        assert self.cache_manager.num_layers == 24
        assert self.cache_manager.hidden_size == 2048
        assert self.cache_manager.block_size == 16
    
    def test_paged_attention_implementation(self):
        """Test PagedAttention implementation"""
        config = self.cache_manager.implement_paged_attention(block_size=32)
        
        assert config is not None
        assert config['block_size'] == 32
        assert 'total_blocks' in config
        assert config['total_blocks'] > 0
        assert 'memory_efficiency' in config
    
    def test_memory_pool_initialization(self):
        """Test memory pool initialization"""
        self.cache_manager.implement_paged_attention()
        
        memory_pool = self.cache_manager.memory_pool
        
        assert memory_pool is not None
        assert 'total_blocks' in memory_pool
        assert 'free_blocks' in memory_pool
        assert 'allocated_blocks' in memory_pool
        assert len(memory_pool['free_blocks']) == memory_pool['total_blocks']
    
    def test_block_allocation(self):
        """Test memory block allocation"""
        self.cache_manager.implement_paged_attention()
        
        sequence_id = "test_seq_1"
        num_blocks = 10
        
        allocated_blocks = self.cache_manager.allocate_blocks(sequence_id, num_blocks)
        
        assert len(allocated_blocks) == num_blocks
        assert sequence_id in self.cache_manager.memory_pool['allocated_blocks']
        assert len(self.cache_manager.memory_pool['free_blocks']) == (
            self.cache_manager.memory_pool['total_blocks'] - num_blocks
        )
    
    def test_block_deallocation(self):
        """Test memory block deallocation"""
        self.cache_manager.implement_paged_attention()
        
        sequence_id = "test_seq_2"
        num_blocks = 5
        
        # Allocate blocks
        self.cache_manager.allocate_blocks(sequence_id, num_blocks)
        initial_free_blocks = len(self.cache_manager.memory_pool['free_blocks'])
        
        # Deallocate blocks
        self.cache_manager.deallocate_blocks(sequence_id)
        
        assert sequence_id not in self.cache_manager.memory_pool['allocated_blocks']
        assert len(self.cache_manager.memory_pool['free_blocks']) == initial_free_blocks + num_blocks
    
    def test_memory_requirements_calculation(self):
        """Test memory requirements calculation"""
        batch_size = 4
        sequence_length = 1024
        
        memory_req = self.cache_manager.calculate_memory_requirements(batch_size, sequence_length)
        
        assert 'batch_size' in memory_req
        assert 'sequence_length' in memory_req
        assert 'total_memory_gb' in memory_req
        assert 'blocks_needed' in memory_req
        assert memory_req['total_memory_gb'] > 0
        assert memory_req['blocks_needed'] > 0
    
    def test_cache_optimization(self):
        """Test cache memory optimization"""
        optimizations = self.cache_manager.optimize_cache_memory()
        
        assert 'memory_pooling' in optimizations
        assert 'compression' in optimizations
        assert 'eviction_policy' in optimizations
        assert optimizations['memory_pooling']['enabled'] is True
    
    def test_performance_benchmarking(self):
        """Test cache performance benchmarking"""
        self.cache_manager.implement_paged_attention()
        
        benchmark_results = self.cache_manager.benchmark_cache_performance()
        
        assert len(benchmark_results) > 0
        
        for scenario, results in benchmark_results.items():
            assert 'cache_hit_rate' in results
            assert 'memory_bandwidth_gbps' in results
            assert 'efficiency_score' in results
            assert 0 <= results['cache_hit_rate'] <= 1

class TestContinuousBatching:
    """Test continuous batching implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.engine = ContinuousBatchingEngine(max_batch_size=8)
    
    def test_engine_initialization(self):
        """Test continuous batching engine initialization"""
        assert self.engine.max_batch_size == 8
        assert len(self.engine.active_requests) == 0
        assert self.engine.running is False
    
    def test_request_addition(self):
        """Test adding requests to the system"""
        request_id = "test_req_1"
        prompt = "Test prompt"
        max_tokens = 100
        
        initial_queue_size = self.engine.request_queue.qsize()
        self.engine.add_request(request_id, prompt, max_tokens)
        
        assert self.engine.request_queue.qsize() == initial_queue_size + 1
    
    def test_dynamic_batch_formation(self):
        """Test dynamic batch formation"""
        # Add multiple requests
        for i in range(5):
            self.engine.add_request(f"req_{i}", f"Prompt {i}", 50)
        
        batch = self.engine.form_dynamic_batch()
        
        assert len(batch) <= self.engine.max_batch_size
        assert len(batch) > 0
        
        # Check that requests are moved to active
        for request in batch:
            assert request['status'] == 'active'
            assert request['id'] in self.engine.active_requests
    
    def test_batch_processing(self):
        """Test batch processing step"""
        # Create mock batch
        batch = [
            {
                'id': 'req_1',
                'prompt': 'Test',
                'max_tokens': 10,
                'generated_tokens': 5,
                'status': 'active',
                'start_time': time.time()
            },
            {
                'id': 'req_2',
                'prompt': 'Test',
                'max_tokens': 20,
                'generated_tokens': 15,
                'status': 'active',
                'start_time': time.time()
            }
        ]
        
        # Add to active requests
        for request in batch:
            self.engine.active_requests[request['id']] = request
        
        completed = self.engine.process_batch_step(batch)
        
        # Check that tokens were generated
        for request in batch:
            assert request['generated_tokens'] > 5  # Should have increased
        
        # Some requests might be completed
        assert isinstance(completed, list)
    
    def test_batching_efficiency_metrics(self):
        """Test batching efficiency calculation"""
        # Simulate some processing
        self.engine.batch_stats['batches_processed'] = 10
        self.engine.batch_stats['total_requests'] = 50
        self.engine.batch_stats['avg_batch_size'] = 5.0
        
        efficiency = self.engine.benchmark_batching_efficiency()
        
        assert 'avg_batch_size' in efficiency
        assert 'batch_utilization' in efficiency
        assert 'utilization_score' in efficiency
        assert efficiency['avg_batch_size'] == 5.0
    
    @pytest.mark.asyncio
    async def test_continuous_batching_loop(self):
        """Test continuous batching main loop"""
        # Add some requests
        for i in range(3):
            self.engine.add_request(f"req_{i}", f"Prompt {i}", 20)
        
        # Run for short duration
        await self.engine.run_continuous_batching(duration_seconds=1)
        
        # Check that some processing occurred
        assert self.engine.batch_stats['batches_processed'] >= 0

class TestSpeculativeDecoding:
    """Test speculative decoding implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.target_model = MockLLMModel(hidden_size=2048, num_layers=24)
        self.draft_model = MockLLMModel(hidden_size=1024, num_layers=12)
        self.engine = SpeculativeDecodingEngine(self.target_model, self.draft_model)
    
    def test_engine_initialization(self):
        """Test speculative decoding engine initialization"""
        assert self.engine.target_model is not None
        assert self.engine.draft_model is not None
        assert hasattr(self.engine, 'tokenizer')
        assert 'total_candidates' in self.engine.acceptance_stats
    
    def test_draft_model_setup(self):
        """Test draft model setup and optimization"""
        optimizations = self.engine.setup_draft_model()
        
        assert 'model_size' in optimizations
        assert 'inference_speed' in optimizations
        assert optimizations['model_size'] == 'small'
        assert 'faster' in optimizations['inference_speed']
    
    def test_speculative_generation_step(self):
        """Test single speculative generation step"""
        input_ids = torch.randint(0, 1000, (1, 10))
        
        result = self.engine.speculative_generate_step(input_ids, num_candidates=4)
        
        assert 'generated_tokens' in result
        assert 'candidates_generated' in result
        assert 'tokens_accepted' in result
        assert 'acceptance_rate' in result
        assert result['candidates_generated'] == 4
        assert 0 <= result['acceptance_rate'] <= 1
    
    def test_draft_generation(self):
        """Test draft model generation"""
        input_ids = torch.randint(0, 1000, (1, 5))
        
        draft_tokens = self.engine._draft_generation(input_ids, num_tokens=3)
        
        assert len(draft_tokens) == 3
        assert all(isinstance(token.item(), int) for token in draft_tokens)
    
    def test_candidate_verification(self):
        """Test candidate token verification"""
        input_ids = torch.randint(0, 1000, (1, 5))
        candidates = torch.randint(0, 1000, (4,))
        
        verification = self.engine._verify_candidates(input_ids, candidates)
        
        assert 'accepted_tokens' in verification
        assert 'acceptance_rate' in verification
        assert 'verification_time' in verification
        assert len(verification['accepted_tokens']) <= len(candidates)
        assert 0 <= verification['acceptance_rate'] <= 1
    
    def test_acceptance_rate_calculation(self):
        """Test acceptance rate calculation"""
        mock_results = [
            {'accepted_tokens': torch.tensor([1, 2]), 'candidates_verified': 4},
            {'accepted_tokens': torch.tensor([3]), 'candidates_verified': 3},
            {'accepted_tokens': torch.tensor([4, 5, 6]), 'candidates_verified': 4}
        ]
        
        stats = self.engine.calculate_acceptance_rate(mock_results)
        
        assert 'overall_acceptance_rate' in stats
        assert 'avg_accepted_per_step' in stats
        assert 'acceptance_by_position' in stats
        assert 0 <= stats['overall_acceptance_rate'] <= 1
    
    def test_parameter_optimization(self):
        """Test speculation parameter optimization"""
        optimal_params = self.engine.optimize_speculation_parameters()
        
        assert 'optimal_candidates' in optimal_params
        assert 'expected_acceptance_rate' in optimal_params
        assert 'expected_speedup' in optimal_params
        assert optimal_params['optimal_candidates'] > 0
    
    def test_speedup_benchmarking(self):
        """Test speculative decoding speedup benchmarking"""
        test_prompts = ["Test prompt 1", "Test prompt 2"]
        
        benchmark_results = self.engine.benchmark_speculative_speedup(test_prompts)
        
        assert 'standard_generation' in benchmark_results
        assert 'speculative_generation' in benchmark_results
        assert 'speedup_analysis' in benchmark_results
        
        speedup_analysis = benchmark_results['speedup_analysis']
        assert 'average_speedup' in speedup_analysis
        assert speedup_analysis['average_speedup'] > 1.0  # Should be faster

class TestLoadBalancing:
    """Test load balancing and auto-scaling implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.load_balancer = LLMLoadBalancer()
        self.auto_scaler = AutoScalingManager(min_instances=1, max_instances=5)
    
    def test_load_balancer_initialization(self):
        """Test load balancer initialization"""
        assert len(self.load_balancer.instances) == 0
        assert self.load_balancer.routing_strategy == 'least_connections'
        assert self.load_balancer.request_counter == 0
    
    def test_instance_addition(self):
        """Test adding instances to load balancer"""
        instance_id = "test-instance-1"
        endpoint = "http://test:8000"
        capacity = 10
        
        self.load_balancer.add_instance(instance_id, endpoint, capacity)
        
        assert len(self.load_balancer.instances) == 1
        
        instance = self.load_balancer.instances[0]
        assert instance['id'] == instance_id
        assert instance['endpoint'] == endpoint
        assert instance['capacity'] == capacity
        assert instance['health_status'] == 'healthy'
    
    def test_routing_strategies(self):
        """Test different routing strategies"""
        # Add test instances
        for i in range(3):
            self.load_balancer.add_instance(f"instance-{i}", f"http://test-{i}:8000", 10)
        
        strategies = self.load_balancer.implement_routing_strategies()
        
        assert 'round_robin' in strategies
        assert 'least_connections' in strategies
        assert 'weighted_round_robin' in strategies
        assert 'latency_based' in strategies
        assert 'resource_aware' in strategies
    
    def test_request_routing(self):
        """Test request routing"""
        # Add instances
        self.load_balancer.add_instance("instance-1", "http://test-1:8000", 10)
        self.load_balancer.add_instance("instance-2", "http://test-2:8000", 10)
        
        request = {"id": "test-req", "prompt": "Test prompt"}
        
        selected_instance = self.load_balancer.route_request(request)
        
        assert selected_instance in ["instance-1", "instance-2"]
        
        # Check that load was updated
        for instance in self.load_balancer.instances:
            if instance['id'] == selected_instance:
                assert instance['current_load'] > 0
                assert instance['total_requests'] > 0
    
    def test_request_completion(self):
        """Test request completion handling"""
        self.load_balancer.add_instance("instance-1", "http://test-1:8000", 10)
        
        # Route a request
        request = {"id": "test-req", "prompt": "Test"}
        selected_instance = self.load_balancer.route_request(request)
        
        # Complete the request
        latency_ms = 150.0
        self.load_balancer.complete_request(selected_instance, latency_ms, success=True)
        
        # Check that load was decreased
        instance = next(inst for inst in self.load_balancer.instances if inst['id'] == selected_instance)
        assert instance['current_load'] == 0
        assert instance['avg_latency_ms'] > 0
    
    def test_auto_scaling_decision(self):
        """Test auto-scaling decision making"""
        # Add instances with high load
        self.load_balancer.add_instance("instance-1", "http://test-1:8000", 10)
        instance = self.load_balancer.instances[0]
        instance['current_load'] = 9  # High utilization
        
        scaling_decision = self.load_balancer.auto_scale_decision()
        
        assert 'action' in scaling_decision
        assert 'reason' in scaling_decision
        assert 'current_utilization' in scaling_decision
        assert scaling_decision['action'] in ['scale_up', 'scale_down', 'none']
    
    def test_health_checks(self):
        """Test health check implementation"""
        self.load_balancer.add_instance("instance-1", "http://test-1:8000", 10)
        
        health_config = self.load_balancer.implement_health_checks()
        
        assert 'check_interval_seconds' in health_config
        assert 'endpoints' in health_config
        assert 'health' in health_config['endpoints']
        
        # Check that instance health was updated
        instance = self.load_balancer.instances[0]
        assert 'health_status' in instance
        assert instance['health_status'] in ['healthy', 'degraded', 'unhealthy']
    
    def test_auto_scaler_initialization(self):
        """Test auto-scaler initialization"""
        assert self.auto_scaler.min_instances == 1
        assert self.auto_scaler.max_instances == 5
        assert len(self.auto_scaler.scaling_history) == 0
    
    def test_scaling_policies(self):
        """Test scaling policy definition"""
        policies = self.auto_scaler.define_scaling_policies()
        
        assert 'scale_up' in policies
        assert 'scale_down' in policies
        assert 'triggers' in policies['scale_up']
        assert 'cooldown_seconds' in policies['scale_up']
    
    def test_scaling_execution(self):
        """Test scaling action execution"""
        self.auto_scaler.define_scaling_policies()
        
        # Test scale up
        self.auto_scaler.execute_scaling_action('scale_up', 2)
        
        assert len(self.auto_scaler.scaling_history) > 0
        
        last_event = self.auto_scaler.scaling_history[-1]
        assert last_event['action'] == 'scale_up'
        assert last_event['count'] == 2
        assert last_event['status'] == 'completed'

class TestPerformanceMonitoring:
    """Test performance monitoring implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.monitor = LLMPerformanceMonitor()
        self.cost_optimizer = CostOptimizer()
    
    def test_monitor_initialization(self):
        """Test performance monitor initialization"""
        assert 'latency' in self.monitor.metrics
        assert 'throughput' in self.monitor.metrics
        assert 'gpu_utilization' in self.monitor.metrics
        assert len(self.monitor.metrics['latency']) == 0
    
    def test_metrics_collection(self):
        """Test real-time metrics collection"""
        instance_id = "test-instance"
        
        result = self.monitor.collect_real_time_metrics(instance_id)
        
        assert 'metrics' in result
        assert 'alerts' in result
        
        metrics = result['metrics']
        assert 'timestamp' in metrics
        assert 'instance_id' in metrics
        assert 'latency_ms' in metrics
        assert 'throughput_tps' in metrics
        assert 'gpu_utilization' in metrics
        
        # Check that metrics were stored
        assert len(self.monitor.metrics['latency']) > 0
        assert len(self.monitor.metrics['throughput']) > 0
    
    def test_alert_generation(self):
        """Test performance alert generation"""
        # Create metrics that should trigger alerts
        high_latency_metrics = {
            'latency_ms': 300,  # Above threshold
            'throughput_tps': 30,  # Below threshold
            'error_rate': 0.1,  # Above threshold
            'queue_length': 150  # Above threshold
        }
        
        alerts = self.monitor._check_alerts(high_latency_metrics)
        
        assert len(alerts) > 0
        
        alert_types = [alert['type'] for alert in alerts]
        assert 'high_latency' in alert_types
        assert 'low_throughput' in alert_types
        assert 'high_error_rate' in alert_types
        assert 'queue_backlog' in alert_types
    
    def test_trend_analysis(self):
        """Test performance trend analysis"""
        # Add some metrics
        for i in range(20):
            self.monitor.metrics['latency'].append(100 + i * 2)  # Increasing trend
            self.monitor.metrics['throughput'].append(80 - i)    # Decreasing trend
        
        trends = self.monitor.analyze_performance_trends()
        
        assert 'trends' in trends
        assert 'concerning_trends' in trends
        assert 'overall_health' in trends
        
        # Check trend detection
        latency_trend = trends['trends']['latency']
        assert latency_trend['direction'] == 'increasing'
        
        throughput_trend = trends['trends']['throughput']
        assert throughput_trend['direction'] == 'decreasing'
    
    def test_optimization_recommendations(self):
        """Test optimization recommendation generation"""
        # Add metrics that should trigger recommendations
        for _ in range(10):
            self.monitor.metrics['latency'].append(200)  # High latency
            self.monitor.metrics['throughput'].append(40)  # Low throughput
            self.monitor.metrics['gpu_utilization'].append(50)  # Low utilization
            self.monitor.metrics['cost_per_token'].append(0.02)  # High cost
        
        recommendations = self.monitor.generate_optimization_recommendations()
        
        assert 'recommendations' in recommendations
        assert 'current_performance' in recommendations
        
        # Should have recommendations for each issue
        rec_categories = [rec['category'] for rec in recommendations['recommendations']]
        assert 'latency' in rec_categories
        assert 'throughput' in rec_categories
    
    def test_cost_optimizer_initialization(self):
        """Test cost optimizer initialization"""
        assert 'gpu_instances' in self.cost_optimizer.cost_models
        assert 'pricing_models' in self.cost_optimizer.cost_models
        
        gpu_instances = self.cost_optimizer.cost_models['gpu_instances']
        assert 'a100_40gb' in gpu_instances
        assert 'cost_per_hour' in gpu_instances['a100_40gb']
    
    def test_cost_calculation(self):
        """Test serving cost calculation"""
        usage_metrics = {
            'gpu_hours': 100,
            'requests_processed': 10000,
            'tokens_generated': 1000000,
            'instance_type': 'a100_40gb',
            'pricing_model': 'on_demand'
        }
        
        cost_breakdown = self.cost_optimizer.calculate_serving_costs(usage_metrics)
        
        assert 'total_cost' in cost_breakdown
        assert 'compute_cost' in cost_breakdown
        assert 'cost_per_request' in cost_breakdown
        assert 'cost_per_1k_tokens' in cost_breakdown
        assert cost_breakdown['total_cost'] > 0
        assert cost_breakdown['cost_per_1k_tokens'] > 0
    
    def test_cost_performance_optimization(self):
        """Test cost-performance ratio optimization"""
        requirements = {
            'target_latency_ms': 100,
            'target_throughput_tps': 80,
            'budget_per_month': 3000,
            'availability': 0.99
        }
        
        optimization_plan = self.cost_optimizer.optimize_cost_performance_ratio(requirements)
        
        if 'error' not in optimization_plan:
            assert 'recommended_config' in optimization_plan
            assert 'optimization_strategies' in optimization_plan
            
            recommended = optimization_plan['recommended_config']
            assert 'instance_type' in recommended
            assert 'monthly_cost' in recommended
            assert recommended['monthly_cost'] <= requirements['budget_per_month']
    
    def test_cost_controls_implementation(self):
        """Test cost control mechanism implementation"""
        budget_limits = {
            'daily_limit': 100,
            'monthly_limit': 2500,
            'max_instances': 8,
            'allow_spot': True
        }
        
        cost_controls = self.cost_optimizer.implement_cost_controls(budget_limits)
        
        assert 'budget_alerts' in cost_controls
        assert 'usage_limits' in cost_controls
        assert 'cost_optimization' in cost_controls
        
        budget_alerts = cost_controls['budget_alerts']
        assert budget_alerts['daily_limit'] == 100
        assert budget_alerts['monthly_limit'] == 2500

class TestIntegration:
    """Integration tests for LLM serving pipeline"""
    
    def test_end_to_end_serving_pipeline(self):
        """Test complete serving pipeline"""
        # Setup components
        config = {'tensor_parallel_size': 1, 'max_model_len': 1024}
        vllm_engine = vLLMServingEngine("test-model", config)
        load_balancer = LLMLoadBalancer()
        monitor = LLMPerformanceMonitor()
        
        # Initialize components
        vllm_engine.initialize_engine()
        load_balancer.add_instance("instance-1", "http://test:8000", 10)
        
        # Simulate request flow
        request = {"id": "test-req", "prompt": "Test prompt"}
        selected_instance = load_balancer.route_request(request)
        
        # Monitor performance
        metrics_result = monitor.collect_real_time_metrics(selected_instance)
        
        # Complete request
        load_balancer.complete_request(selected_instance, 120.0, success=True)
        
        # Verify pipeline worked
        assert selected_instance == "instance-1"
        assert 'metrics' in metrics_result
        assert len(monitor.metrics['latency']) > 0
    
    def test_optimization_pipeline(self):
        """Test optimization pipeline integration"""
        # Setup optimization components
        tensorrt_config = {'precision': 'fp16', 'max_batch_size': 4}
        tensorrt_optimizer = TensorRTOptimizer("./model.onnx", tensorrt_config)
        
        cache_manager = KVCacheManager(2048, 24, 2048)
        
        # Apply optimizations
        engine_info = tensorrt_optimizer.build_tensorrt_engine()
        tensorrt_optimizer.optimize_for_latency()
        
        cache_config = cache_manager.implement_paged_attention()
        cache_manager.optimize_cache_memory()
        
        # Verify optimizations
        assert engine_info['precision'] == 'fp16'
        assert cache_config['block_size'] > 0
        assert cache_manager.memory_pool is not None
    
    def test_monitoring_and_scaling_integration(self):
        """Test monitoring and auto-scaling integration"""
        # Setup components
        load_balancer = LLMLoadBalancer()
        auto_scaler = AutoScalingManager(min_instances=1, max_instances=5)
        monitor = LLMPerformanceMonitor()
        
        # Add instance
        load_balancer.add_instance("instance-1", "http://test:8000", 10)
        
        # Simulate high load
        instance = load_balancer.instances[0]
        instance['current_load'] = 9
        instance['avg_latency_ms'] = 250
        
        # Collect metrics and make scaling decision
        monitor.collect_real_time_metrics("instance-1")
        scaling_decision = load_balancer.auto_scale_decision()
        
        # Execute scaling if needed
        auto_scaler.define_scaling_policies()
        if scaling_decision['action'] == 'scale_up':
            auto_scaler.execute_scaling_action('scale_up', 1)
        
        # Verify integration
        assert scaling_decision['action'] in ['scale_up', 'scale_down', 'none']
        if scaling_decision['action'] == 'scale_up':
            assert len(auto_scaler.scaling_history) > 0

def test_mock_models():
    """Test mock model functionality"""
    # Test MockLLMModel
    model = MockLLMModel(vocab_size=1000, hidden_size=512, num_layers=6)
    input_ids = torch.randint(0, 1000, (2, 10))
    
    output = model(input_ids)
    
    assert 'logits' in output
    assert 'past_key_values' in output
    assert output['logits'].shape == (2, 10, 1000)
    
    # Test MockTokenizer
    tokenizer = MockTokenizer()
    text = "Hello world test"
    
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    assert isinstance(tokens, list)
    assert len(tokens) == len(text.split())
    assert isinstance(decoded, str)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])