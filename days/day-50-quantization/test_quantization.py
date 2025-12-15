"""
Day 50: Quantization - Model Compression & Optimization - Test Suite

Comprehensive test suite for quantization implementations with 30+ test cases
covering PTQ, QAT, GPTQ, AWQ, GGUF, hardware optimization, and deployment.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
import json
import time

# Import solution classes
from solution import (
    PostTrainingQuantizer,
    QuantizationAwareTrainer,
    GPTQQuantizer,
    AWQQuantizer,
    GGUFConverter,
    HardwareOptimizer,
    QuantizedModelDeployment,
    MockCNNModel,
    MockTransformerModel
)

class TestPostTrainingQuantization:
    """Test Post-Training Quantization implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = MockCNNModel()
        self.calibration_data = [torch.randn(4, 3, 32, 32) for _ in range(5)]
        self.quantizer = PostTrainingQuantizer(self.model, self.calibration_data)
    
    def test_quantizer_initialization(self):
        """Test quantizer initialization"""
        assert self.quantizer.model is not None
        assert len(self.quantizer.calibration_data) == 5
        assert isinstance(self.quantizer.model, MockCNNModel)
    
    def test_prepare_model_for_quantization(self):
        """Test model preparation for quantization"""
        prepared_model = self.quantizer.prepare_model_for_quantization()
        
        # Check model is in eval mode
        assert not prepared_model.training
        
        # Check qconfig is set
        assert hasattr(prepared_model, 'qconfig')
        assert prepared_model.qconfig is not None
    
    def test_calibrate_model(self):
        """Test model calibration"""
        self.quantizer.prepare_model_for_quantization()
        
        # Should not raise exception
        self.quantizer.calibrate_model()
    
    def test_convert_to_quantized(self):
        """Test conversion to quantized model"""
        self.quantizer.prepare_model_for_quantization()
        self.quantizer.calibrate_model()
        
        quantized_model = self.quantizer.convert_to_quantized()
        assert quantized_model is not None
    
    def test_compare_model_sizes(self):
        """Test model size comparison"""
        original_model = MockCNNModel()
        quantized_model = MockCNNModel()  # Mock quantized model
        
        comparison = self.quantizer.compare_model_sizes(original_model, quantized_model)
        
        assert 'original_size_mb' in comparison
        assert 'quantized_size_mb' in comparison
        assert 'compression_ratio' in comparison
        assert 'size_reduction_percent' in comparison
        assert comparison['compression_ratio'] > 0
    
    def test_full_ptq_pipeline(self):
        """Test complete PTQ pipeline"""
        quantized_model = self.quantizer.full_ptq_pipeline()
        assert quantized_model is not None
    
    def test_empty_calibration_data(self):
        """Test handling of empty calibration data"""
        empty_quantizer = PostTrainingQuantizer(self.model, [])
        empty_quantizer.prepare_model_for_quantization()
        
        # Should handle empty data gracefully
        empty_quantizer.calibrate_model()

class TestQuantizationAwareTraining:
    """Test Quantization-Aware Training implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = MockCNNModel()
        self.train_data = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(10)]
        self.val_data = [(torch.randn(4, 3, 32, 32), torch.randint(0, 10, (4,))) for _ in range(3)]
        self.trainer = QuantizationAwareTrainer(self.model, self.train_data, self.val_data)
    
    def test_trainer_initialization(self):
        """Test trainer initialization"""
        assert self.trainer.model is not None
        assert len(self.trainer.train_loader) == 10
        assert len(self.trainer.val_loader) == 3
    
    def test_prepare_qat_model(self):
        """Test QAT model preparation"""
        prepared_model = self.trainer.prepare_qat_model()
        
        # Check qconfig is set
        assert hasattr(prepared_model, 'qconfig')
        assert prepared_model.qconfig is not None
    
    def test_validate_qat_model(self):
        """Test QAT model validation"""
        self.trainer.prepare_qat_model()
        accuracy = self.trainer.validate_qat_model()
        
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
    
    def test_train_with_quantization_simulation(self):
        """Test training with quantization simulation"""
        self.trainer.prepare_qat_model()
        
        # Should not raise exception
        self.trainer.train_with_quantization_simulation(epochs=1)
    
    def test_convert_qat_to_quantized(self):
        """Test QAT to quantized conversion"""
        self.trainer.prepare_qat_model()
        quantized_model = self.trainer.convert_qat_to_quantized()
        
        assert quantized_model is not None

class TestGPTQQuantization:
    """Test GPTQ quantization implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = MockTransformerModel()
        self.calibration_data = [torch.randint(0, 1000, (2, 16)) for _ in range(5)]
        self.gptq = GPTQQuantizer(self.model, bits=4, group_size=128)
    
    def test_gptq_initialization(self):
        """Test GPTQ quantizer initialization"""
        assert self.gptq.model is not None
        assert self.gptq.bits == 4
        assert self.gptq.group_size == 128
    
    def test_collect_layer_statistics(self):
        """Test layer statistics collection"""
        # Get first linear layer
        linear_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                linear_layer = module
                break
        
        assert linear_layer is not None
        
        inputs, outputs = self.gptq.collect_layer_statistics(linear_layer, self.calibration_data)
        
        # May be None if no data collected, but should not raise exception
        if inputs is not None:
            assert inputs.dim() >= 2
        if outputs is not None:
            assert outputs.dim() >= 2
    
    def test_compute_hessian_approximation(self):
        """Test Hessian approximation computation"""
        inputs = torch.randn(10, 256)
        hessian = self.gptq.compute_hessian_approximation(inputs)
        
        assert hessian is not None
        assert hessian.shape == (256, 256)
        assert torch.allclose(hessian, hessian.T, atol=1e-6)  # Should be symmetric
    
    def test_compute_quantization_params(self):
        """Test quantization parameter computation"""
        weights = torch.randn(64, 128)
        scale, zero_point = self.gptq._compute_quantization_params(weights)
        
        assert isinstance(scale, (float, torch.Tensor))
        assert isinstance(zero_point, (float, torch.Tensor))
        assert scale > 0
    
    def test_gptq_quantize_weights(self):
        """Test GPTQ weight quantization"""
        weights = torch.randn(64, 128)
        hessian = torch.eye(128) + torch.randn(128, 128) * 0.1
        
        quantized_weights = self.gptq.gptq_quantize_weights(weights, hessian)
        
        assert quantized_weights.shape == weights.shape
        assert not torch.equal(quantized_weights, weights)  # Should be different
    
    def test_quantize_model_with_gptq(self):
        """Test full GPTQ model quantization"""
        original_weights = {}
        for name, param in self.model.named_parameters():
            original_weights[name] = param.data.clone()
        
        quantized_model = self.gptq.quantize_model_with_gptq(self.calibration_data)
        
        # Check that some weights have changed
        weights_changed = False
        for name, param in quantized_model.named_parameters():
            if not torch.equal(param.data, original_weights[name]):
                weights_changed = True
                break
        
        assert weights_changed

class TestAWQQuantization:
    """Test AWQ quantization implementation"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = MockTransformerModel()
        self.calibration_data = [torch.randint(0, 1000, (2, 16)) for _ in range(5)]
        self.awq = AWQQuantizer(self.model, bits=4, group_size=128)
    
    def test_awq_initialization(self):
        """Test AWQ quantizer initialization"""
        assert self.awq.model is not None
        assert self.awq.bits == 4
        assert self.awq.group_size == 128
    
    def test_compute_activation_scales(self):
        """Test activation scale computation"""
        # Get first linear layer
        linear_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                linear_layer = module
                break
        
        assert linear_layer is not None
        
        activation_scales = self.awq.compute_activation_scales(linear_layer, self.calibration_data)
        
        # May be None if no activations collected
        if activation_scales is not None:
            assert activation_scales.dim() == 1
            assert (activation_scales >= 0).all()  # Should be non-negative
    
    def test_compute_weight_importance(self):
        """Test weight importance computation"""
        weights = torch.randn(64, 128)
        activation_scales = torch.rand(64)
        
        importance = self.awq.compute_weight_importance(weights, activation_scales)
        
        assert importance.shape == weights.shape
        assert (importance >= 0).all()  # Should be non-negative
    
    def test_protect_important_weights(self):
        """Test important weight protection"""
        weights = torch.randn(64, 128)
        importance = torch.rand(64, 128)
        
        protection_mask = self.awq.protect_important_weights(weights, importance, protection_ratio=0.1)
        
        assert protection_mask.shape == weights.shape
        assert protection_mask.dtype == torch.bool
        
        # Should protect approximately 10% of weights
        protection_ratio = protection_mask.float().mean().item()
        assert 0.05 <= protection_ratio <= 0.15  # Allow some variance
    
    def test_quantize_with_protection(self):
        """Test quantization with weight protection"""
        weights = torch.randn(64, 128)
        protection_mask = torch.rand(64, 128) > 0.9  # Protect ~10% of weights
        
        quantized_weights = self.awq.quantize_with_protection(weights, protection_mask)
        
        assert quantized_weights.shape == weights.shape
        assert not torch.equal(quantized_weights, weights)  # Should be different
    
    def test_quantize_layer_with_awq(self):
        """Test AWQ layer quantization"""
        # Get first linear layer
        linear_layer = None
        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                linear_layer = module
                break
        
        assert linear_layer is not None
        
        original_weights = linear_layer.weight.data.clone()
        quantized_layer = self.awq.quantize_layer_with_awq(linear_layer, self.calibration_data)
        
        assert quantized_layer is not None
        # Weights may or may not change depending on activation collection

class TestGGUFConverter:
    """Test GGUF format conversion"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model.bin")
        self.output_path = os.path.join(self.temp_dir, "test_model.gguf")
        
        # Create mock model file
        with open(self.model_path, 'wb') as f:
            f.write(b"mock model data")
        
        self.converter = GGUFConverter(self.model_path, self.output_path)
    
    def teardown_method(self):
        """Cleanup test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_converter_initialization(self):
        """Test GGUF converter initialization"""
        assert self.converter.model_path == self.model_path
        assert self.converter.output_path == self.output_path
        assert len(self.converter.supported_quantizations) > 10
    
    def test_analyze_model_architecture(self):
        """Test model architecture analysis"""
        analysis = self.converter.analyze_model_architecture()
        
        assert 'compatible' in analysis
        assert 'architecture' in analysis
        assert 'total_parameters' in analysis
        assert 'layer_types' in analysis
        assert 'issues' in analysis
    
    def test_prepare_model_weights(self):
        """Test model weight preparation"""
        weight_info = self.converter.prepare_model_weights()
        
        assert 'tensors' in weight_info
        assert 'metadata' in weight_info
        assert len(weight_info['tensors']) > 0
        assert 'general.architecture' in weight_info['metadata']
    
    def test_apply_gguf_quantization(self):
        """Test GGUF quantization application"""
        weight_info = self.converter.prepare_model_weights()
        
        for quant_type in ['q4_0', 'q8_0', 'f16']:
            quantized_info = self.converter.apply_gguf_quantization(weight_info, quant_type)
            
            assert 'quantization_type' in quantized_info
            assert 'tensors' in quantized_info
            assert 'total_size_original' in quantized_info
            assert 'total_size_quantized' in quantized_info
            assert quantized_info['quantization_type'] == quant_type
    
    def test_write_gguf_file(self):
        """Test GGUF file writing"""
        weight_info = self.converter.prepare_model_weights()
        quantized_info = self.converter.apply_gguf_quantization(weight_info, 'q4_0')
        
        gguf_structure = self.converter.write_gguf_file(quantized_info, weight_info['metadata'])
        
        assert 'header' in gguf_structure
        assert 'metadata' in gguf_structure
        assert 'tensor_info' in gguf_structure
        assert 'compression_ratio' in gguf_structure
        
        # Check info file was created
        assert os.path.exists(self.output_path + '.info')
    
    def test_validate_gguf_file(self):
        """Test GGUF file validation"""
        # First create a file
        weight_info = self.converter.prepare_model_weights()
        quantized_info = self.converter.apply_gguf_quantization(weight_info, 'q4_0')
        self.converter.write_gguf_file(quantized_info, weight_info['metadata'])
        
        validation = self.converter.validate_gguf_file()
        
        assert 'valid' in validation
        assert 'issues' in validation
        assert 'file_size_mb' in validation
        assert 'tensor_count' in validation
    
    def test_convert_to_gguf(self):
        """Test complete GGUF conversion"""
        result = self.converter.convert_to_gguf('q4_0')
        
        assert 'success' in result
        assert 'output_path' in result
        assert 'quantization_type' in result
        assert 'compression_ratio' in result
        assert result['quantization_type'] == 'q4_0'
    
    def test_unsupported_quantization_type(self):
        """Test handling of unsupported quantization type"""
        weight_info = self.converter.prepare_model_weights()
        
        with pytest.raises(ValueError):
            self.converter.apply_gguf_quantization(weight_info, 'unsupported_type')

class TestHardwareOptimizer:
    """Test hardware-specific optimization"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = MockCNNModel()
        self.input_data = torch.randn(1, 3, 32, 32)
    
    def test_cpu_optimization(self):
        """Test CPU optimization"""
        optimizer = HardwareOptimizer(self.model, 'cpu')
        optimized_model = optimizer.optimize_for_cpu()
        
        assert optimized_model is not None
        assert 'use_mkldnn' in optimizer.optimization_config
        assert 'use_openmp' in optimizer.optimization_config
    
    def test_gpu_optimization(self):
        """Test GPU optimization"""
        optimizer = HardwareOptimizer(self.model, 'gpu')
        optimized_model = optimizer.optimize_for_gpu()
        
        assert optimized_model is not None
        assert 'use_cuda_graphs' in optimizer.optimization_config
    
    def test_mobile_optimization(self):
        """Test mobile optimization"""
        optimizer = HardwareOptimizer(self.model, 'mobile')
        optimized_model = optimizer.optimize_for_mobile()
        
        assert optimized_model is not None
        assert 'target_platform' in optimizer.optimization_config
        assert 'max_memory_mb' in optimizer.optimization_config
    
    def test_benchmark_performance(self):
        """Test performance benchmarking"""
        optimizer = HardwareOptimizer(self.model, 'cpu')
        optimizer.optimize_for_cpu()
        
        benchmark = optimizer.benchmark_performance(self.input_data, num_runs=10)
        
        assert 'hardware' in benchmark
        assert 'avg_inference_time_ms' in benchmark
        assert 'throughput_fps' in benchmark
        assert 'memory_usage_mb' in benchmark
        assert benchmark['avg_inference_time_ms'] > 0
        assert benchmark['throughput_fps'] > 0

class TestQuantizedModelDeployment:
    """Test quantized model deployment and monitoring"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.model = MockCNNModel()
        self.deployment = QuantizedModelDeployment(self.model, "test_model")
        self.input_data = torch.randn(1, 3, 32, 32)
        self.ground_truth = torch.randint(0, 10, (1,))
    
    def test_deployment_initialization(self):
        """Test deployment initialization"""
        assert self.deployment.model is not None
        assert self.deployment.model_name == "test_model"
        assert 'inference_times' in self.deployment.metrics
        assert 'max_latency_ms' in self.deployment.alert_thresholds
    
    def test_validate_quantized_model(self):
        """Test model validation"""
        validation = self.deployment.validate_quantized_model()
        
        assert 'valid' in validation
        assert 'issues' in validation
        assert 'metrics' in validation
        
        if validation['valid']:
            assert 'model_size_mb' in validation['metrics']
            assert 'avg_inference_time_ms' in validation['metrics']
    
    def test_setup_monitoring(self):
        """Test monitoring setup"""
        config = self.deployment.setup_monitoring()
        
        assert 'metrics_to_track' in config
        assert 'alert_thresholds' in config
        assert 'monitoring_frequency' in config
        assert len(config['metrics_to_track']) > 0
    
    def test_monitor_inference(self):
        """Test inference monitoring"""
        result = self.deployment.monitor_inference(self.input_data, self.ground_truth)
        
        assert 'inference_time_ms' in result
        assert 'accuracy' in result
        assert 'memory_usage_mb' in result
        assert 'alerts' in result
        assert 'timestamp' in result
        
        assert result['inference_time_ms'] > 0
        assert 0 <= result['accuracy'] <= 1
    
    def test_compute_accuracy(self):
        """Test accuracy computation"""
        predictions = torch.tensor([[0.1, 0.9, 0.0]])  # Class 1
        ground_truth = torch.tensor([1])
        
        accuracy = self.deployment._compute_accuracy(predictions, ground_truth)
        assert accuracy == 1.0
        
        # Test incorrect prediction
        ground_truth = torch.tensor([0])
        accuracy = self.deployment._compute_accuracy(predictions, ground_truth)
        assert accuracy == 0.0
    
    def test_get_memory_usage(self):
        """Test memory usage measurement"""
        memory_usage = self.deployment._get_memory_usage()
        
        assert isinstance(memory_usage, (int, float))
        assert memory_usage > 0
    
    def test_check_performance_alerts(self):
        """Test performance alert checking"""
        # Test high latency alert
        alerts = self.deployment._check_performance_alerts(200, 0.9, 100)  # 200ms latency
        assert len(alerts) > 0
        assert any(alert['type'] == 'high_latency' for alert in alerts)
        
        # Test low accuracy alert
        alerts = self.deployment._check_performance_alerts(50, 0.7, 100)  # 70% accuracy
        assert any(alert['type'] == 'low_accuracy' for alert in alerts)
        
        # Test normal performance
        alerts = self.deployment._check_performance_alerts(50, 0.9, 100)
        latency_alerts = [alert for alert in alerts if alert['type'] == 'high_latency']
        accuracy_alerts = [alert for alert in alerts if alert['type'] == 'low_accuracy']
        assert len(latency_alerts) == 0
        assert len(accuracy_alerts) == 0
    
    def test_generate_performance_report(self):
        """Test performance report generation"""
        # Add some metrics
        for _ in range(10):
            self.deployment.monitor_inference(self.input_data, self.ground_truth)
        
        report = self.deployment.generate_performance_report()
        
        assert 'model_name' in report
        assert 'performance_summary' in report
        assert 'accuracy_summary' in report
        assert 'recommendations' in report
        assert 'generated_at' in report
        
        perf_summary = report['performance_summary']
        assert 'avg_inference_time_ms' in perf_summary
        assert 'p95_inference_time_ms' in perf_summary
        assert 'avg_throughput_fps' in perf_summary
    
    def test_handle_performance_degradation(self):
        """Test performance degradation handling"""
        # Simulate degradation by adding high latency metrics
        for _ in range(10):
            self.deployment.metrics['inference_times'].append(0.2)  # 200ms
            self.deployment.metrics['accuracy_scores'].append(0.7)  # 70% accuracy
        
        result = self.deployment.handle_performance_degradation()
        
        assert 'degradation_detected' in result
        if result['degradation_detected']:
            assert 'issues' in result
            assert 'fallback_actions' in result
            assert 'timestamp' in result
    
    def test_generate_recommendations(self):
        """Test recommendation generation"""
        # Add metrics that should trigger recommendations
        self.deployment.metrics['inference_times'] = [0.15] * 10  # High latency
        self.deployment.metrics['accuracy_scores'] = [0.8] * 10   # Low accuracy
        
        recommendations = self.deployment._generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        assert any('latency' in rec.lower() for rec in recommendations)

class TestIntegration:
    """Integration tests for quantization pipeline"""
    
    def test_ptq_to_deployment_pipeline(self):
        """Test complete PTQ to deployment pipeline"""
        # Step 1: Create and quantize model
        model = MockCNNModel()
        calibration_data = [torch.randn(4, 3, 32, 32) for _ in range(5)]
        
        quantizer = PostTrainingQuantizer(model, calibration_data)
        quantized_model = quantizer.full_ptq_pipeline()
        
        # Step 2: Deploy quantized model
        deployment = QuantizedModelDeployment(quantized_model, "ptq_model")
        validation = deployment.validate_quantized_model()
        
        assert validation['valid']
        
        # Step 3: Monitor performance
        input_data = torch.randn(1, 3, 32, 32)
        ground_truth = torch.randint(0, 10, (1,))
        
        result = deployment.monitor_inference(input_data, ground_truth)
        assert 'inference_time_ms' in result
        assert result['inference_time_ms'] > 0
    
    def test_hardware_optimization_pipeline(self):
        """Test hardware optimization pipeline"""
        model = MockCNNModel()
        input_data = torch.randn(1, 3, 32, 32)
        
        # Test different hardware targets
        for hardware in ['cpu', 'gpu', 'mobile']:
            optimizer = HardwareOptimizer(model, hardware)
            
            if hardware == 'cpu':
                optimized_model = optimizer.optimize_for_cpu()
            elif hardware == 'gpu':
                optimized_model = optimizer.optimize_for_gpu()
            else:
                optimized_model = optimizer.optimize_for_mobile()
            
            # Benchmark performance
            benchmark = optimizer.benchmark_performance(input_data, num_runs=5)
            
            assert benchmark['hardware'] == hardware
            assert benchmark['avg_inference_time_ms'] > 0
    
    def test_quantization_accuracy_preservation(self):
        """Test that quantization preserves reasonable accuracy"""
        model = MockCNNModel()
        input_data = torch.randn(10, 3, 32, 32)
        
        # Get original outputs
        model.eval()
        with torch.no_grad():
            original_outputs = model(input_data)
        
        # Apply PTQ
        calibration_data = [torch.randn(4, 3, 32, 32) for _ in range(5)]
        quantizer = PostTrainingQuantizer(model, calibration_data)
        quantized_model = quantizer.full_ptq_pipeline()
        
        # Get quantized outputs
        with torch.no_grad():
            quantized_outputs = quantized_model(input_data)
        
        # Check outputs are similar (allowing for quantization error)
        assert original_outputs.shape == quantized_outputs.shape
        
        # Outputs should be correlated (not identical due to quantization)
        correlation = torch.corrcoef(torch.stack([
            original_outputs.flatten(),
            quantized_outputs.flatten()
        ]))[0, 1]
        
        assert correlation > 0.5  # Should maintain reasonable correlation

def test_mock_models():
    """Test mock model functionality"""
    # Test CNN model
    cnn_model = MockCNNModel()
    cnn_input = torch.randn(2, 3, 32, 32)
    cnn_output = cnn_model(cnn_input)
    
    assert cnn_output.shape == (2, 10)  # Batch size 2, 10 classes
    
    # Test Transformer model
    transformer_model = MockTransformerModel()
    transformer_input = torch.randint(0, 1000, (2, 16))
    transformer_output = transformer_model(transformer_input)
    
    assert transformer_output.shape == (2, 16, 1000)  # Batch, sequence, vocab

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])