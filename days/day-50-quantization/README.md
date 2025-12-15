# Day 50: Quantization - Model Compression & Optimization

## Learning Objectives
By the end of this session, you will be able to:
- Understand quantization fundamentals and their impact on model performance
- Implement post-training quantization (PTQ) and quantization-aware training (QAT)
- Apply advanced quantization methods: GPTQ, AWQ, and GGUF formats
- Optimize models for different hardware targets (CPU, GPU, mobile)
- Deploy quantized models in production with proper monitoring
- Balance model size, speed, and accuracy trade-offs

## Theory (30 minutes)

### What is Model Quantization?

Model quantization is a technique that reduces the precision of model weights and activations from higher precision (typically FP32) to lower precision (INT8, INT4, or even binary), significantly reducing model size and inference time while maintaining acceptable accuracy.

**Key Benefits:**
- **Memory Reduction**: 2-8x smaller model size
- **Speed Improvement**: 2-4x faster inference
- **Energy Efficiency**: Lower power consumption
- **Hardware Compatibility**: Enables deployment on resource-constrained devices
- **Cost Reduction**: Lower cloud computing costs

### Quantization Fundamentals

#### Precision Formats

```python
# Different precision formats and their characteristics
precision_formats = {
    'FP32': {
        'bits': 32,
        'range': '±3.4 × 10^38',
        'precision': '7 decimal digits',
        'memory': '4 bytes per parameter'
    },
    'FP16': {
        'bits': 16,
        'range': '±65,504',
        'precision': '3-4 decimal digits',
        'memory': '2 bytes per parameter'
    },
    'INT8': {
        'bits': 8,
        'range': '-128 to 127',
        'precision': 'Integer only',
        'memory': '1 byte per parameter'
    },
    'INT4': {
        'bits': 4,
        'range': '-8 to 7',
        'precision': 'Integer only',
        'memory': '0.5 bytes per parameter'
    }
}
```

#### Quantization Process

The quantization process involves mapping floating-point values to integers:

```
quantized_value = round((float_value - zero_point) / scale)
dequantized_value = scale * quantized_value + zero_point
```

Where:
- **Scale**: Controls the step size between quantized values
- **Zero Point**: Ensures that zero in floating-point maps to zero in quantized space

### Types of Quantization

#### 1. Post-Training Quantization (PTQ)

PTQ quantizes a pre-trained model without additional training:

```python
import torch
import torch.quantization as quant

class PostTrainingQuantization:
    """Post-training quantization implementation"""
    
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        
    def prepare_model(self):
        """Prepare model for quantization"""
        # Set model to evaluation mode
        self.model.eval()
        
        # Fuse operations (Conv + BN + ReLU)
        self.model = torch.quantization.fuse_modules(
            self.model, 
            [['conv', 'bn', 'relu']]  # Example fusion
        )
        
        # Set quantization configuration
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare model
        torch.quantization.prepare(self.model, inplace=True)
        
        return self.model
    
    def calibrate_model(self):
        """Calibrate model with representative data"""
        self.model.eval()
        
        with torch.no_grad():
            for batch in self.calibration_data:
                # Forward pass to collect statistics
                _ = self.model(batch)
    
    def quantize_model(self):
        """Convert model to quantized version"""
        # Convert to quantized model
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        
        return quantized_model
    
    def full_quantization_pipeline(self):
        """Complete PTQ pipeline"""
        # Step 1: Prepare model
        self.prepare_model()
        
        # Step 2: Calibrate with data
        self.calibrate_model()
        
        # Step 3: Convert to quantized model
        quantized_model = self.quantize_model()
        
        return quantized_model

# Example usage
def demonstrate_ptq():
    """Demonstrate post-training quantization"""
    
    # Mock model and data
    model = torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )
    
    calibration_data = [torch.randn(32, 784) for _ in range(10)]
    
    # Apply PTQ
    ptq = PostTrainingQuantization(model, calibration_data)
    quantized_model = ptq.full_quantization_pipeline()
    
    # Compare model sizes
    original_size = sum(p.numel() * 4 for p in model.parameters())  # FP32
    quantized_size = sum(p.numel() for p in quantized_model.parameters())  # INT8
    
    print(f"Original model size: {original_size / 1024:.2f} KB")
    print(f"Quantized model size: {quantized_size / 1024:.2f} KB")
    print(f"Compression ratio: {original_size / quantized_size:.2f}x")
```

#### 2. Quantization-Aware Training (QAT)

QAT simulates quantization during training to maintain accuracy:

```python
class QuantizationAwareTraining:
    """Quantization-aware training implementation"""
    
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
    def prepare_qat_model(self):
        """Prepare model for QAT"""
        # Set model to training mode
        self.model.train()
        
        # Fuse modules
        self.model = torch.quantization.fuse_modules(
            self.model,
            [['conv', 'bn', 'relu']]
        )
        
        # Set QAT configuration
        self.model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
        
        # Prepare for QAT
        torch.quantization.prepare_qat(self.model, inplace=True)
        
        return self.model
    
    def train_qat_model(self, epochs=5, lr=0.001):
        """Train model with quantization simulation"""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                optimizer.zero_grad()
                
                # Forward pass with quantization simulation
                output = self.model(data)
                loss = criterion(output, target)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
            
            # Validation
            val_accuracy = self.validate_model()
            print(f'Epoch {epoch}, Validation Accuracy: {val_accuracy:.4f}')
    
    def validate_model(self):
        """Validate quantized model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        return correct / total
    
    def convert_to_quantized(self):
        """Convert QAT model to fully quantized model"""
        self.model.eval()
        quantized_model = torch.quantization.convert(self.model, inplace=False)
        return quantized_model
```

### Advanced Quantization Methods

#### 1. GPTQ (Gradient-Free Post-Training Quantization)

GPTQ is an advanced post-training quantization method specifically designed for large language models:

```python
class GPTQQuantizer:
    """GPTQ quantization implementation"""
    
    def __init__(self, model, bits=4, group_size=128):
        self.model = model
        self.bits = bits
        self.group_size = group_size
        
    def quantize_layer(self, layer, calibration_data):
        """Quantize a single layer using GPTQ algorithm"""
        
        # Collect layer inputs and outputs
        inputs = []
        outputs = []
        
        def hook_fn(module, input, output):
            inputs.append(input[0].detach())
            outputs.append(output.detach())
        
        handle = layer.register_forward_hook(hook_fn)
        
        # Forward pass to collect data
        with torch.no_grad():
            for batch in calibration_data:
                _ = self.model(batch)
        
        handle.remove()
        
        # Stack collected data
        X = torch.cat(inputs, dim=0)
        Y = torch.cat(outputs, dim=0)
        
        # GPTQ quantization algorithm
        W = layer.weight.data.clone()
        H = X.T @ X  # Hessian approximation
        
        # Quantize weights group by group
        quantized_weights = self._gptq_quantize_weights(W, H)
        
        # Update layer weights
        layer.weight.data = quantized_weights
        
        return layer
    
    def _gptq_quantize_weights(self, weights, hessian):
        """Core GPTQ quantization algorithm"""
        
        # Initialize quantized weights
        quantized_weights = weights.clone()
        
        # Process weights in groups
        for i in range(0, weights.shape[1], self.group_size):
            end_idx = min(i + self.group_size, weights.shape[1])
            
            # Extract weight group
            weight_group = weights[:, i:end_idx]
            
            # Compute quantization parameters
            scale, zero_point = self._compute_quantization_params(weight_group)
            
            # Quantize weights
            quantized_group = self._quantize_weights(weight_group, scale, zero_point)
            
            # Update quantized weights
            quantized_weights[:, i:end_idx] = quantized_group
        
        return quantized_weights
    
    def _compute_quantization_params(self, weights):
        """Compute scale and zero point for quantization"""
        
        # Compute min and max values
        w_min = weights.min()
        w_max = weights.max()
        
        # Compute scale and zero point
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1
        
        scale = (w_max - w_min) / (qmax - qmin)
        zero_point = qmin - w_min / scale
        
        return scale, zero_point
    
    def _quantize_weights(self, weights, scale, zero_point):
        """Quantize weights using scale and zero point"""
        
        # Quantize
        quantized = torch.round(weights / scale + zero_point)
        
        # Clamp to valid range
        qmin = -(2 ** (self.bits - 1))
        qmax = 2 ** (self.bits - 1) - 1
        quantized = torch.clamp(quantized, qmin, qmax)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized
```

#### 2. AWQ (Activation-aware Weight Quantization)

AWQ protects important weights based on activation patterns:

```python
class AWQQuantizer:
    """AWQ quantization implementation"""
    
    def __init__(self, model, bits=4, group_size=128):
        self.model = model
        self.bits = bits
        self.group_size = group_size
        
    def compute_activation_scales(self, layer, calibration_data):
        """Compute activation-based importance scores"""
        
        activations = []
        
        def hook_fn(module, input, output):
            activations.append(input[0].detach())
        
        handle = layer.register_forward_hook(hook_fn)
        
        # Collect activations
        with torch.no_grad():
            for batch in calibration_data:
                _ = self.model(batch)
        
        handle.remove()
        
        # Compute activation statistics
        all_activations = torch.cat(activations, dim=0)
        
        # Compute channel-wise activation magnitudes
        activation_scales = all_activations.abs().mean(dim=(0, 2, 3))  # For conv layers
        
        return activation_scales
    
    def quantize_with_awq(self, layer, calibration_data):
        """Quantize layer using AWQ method"""
        
        # Compute activation scales
        activation_scales = self.compute_activation_scales(layer, calibration_data)
        
        # Compute weight importance based on activations
        weights = layer.weight.data
        weight_importance = self._compute_weight_importance(weights, activation_scales)
        
        # Protect important weights
        protected_weights = self._protect_important_weights(weights, weight_importance)
        
        # Quantize remaining weights
        quantized_weights = self._quantize_unprotected_weights(protected_weights, weight_importance)
        
        # Update layer
        layer.weight.data = quantized_weights
        
        return layer
    
    def _compute_weight_importance(self, weights, activation_scales):
        """Compute weight importance based on activations"""
        
        # Weight importance = |weight| * activation_scale
        importance = weights.abs() * activation_scales.view(-1, 1, 1, 1)
        
        return importance
    
    def _protect_important_weights(self, weights, importance, protection_ratio=0.1):
        """Protect most important weights from quantization"""
        
        # Find top important weights
        flat_importance = importance.flatten()
        threshold = torch.quantile(flat_importance, 1 - protection_ratio)
        
        # Create protection mask
        protection_mask = importance > threshold
        
        return weights, protection_mask
    
    def _quantize_unprotected_weights(self, weights_and_mask, importance):
        """Quantize weights that are not protected"""
        
        weights, protection_mask = weights_and_mask
        quantized_weights = weights.clone()
        
        # Quantize unprotected weights
        unprotected_mask = ~protection_mask
        unprotected_weights = weights[unprotected_mask]
        
        if unprotected_weights.numel() > 0:
            # Compute quantization parameters
            scale, zero_point = self._compute_quantization_params(unprotected_weights)
            
            # Quantize
            quantized_unprotected = self._quantize_weights(unprotected_weights, scale, zero_point)
            
            # Update quantized weights
            quantized_weights[unprotected_mask] = quantized_unprotected
        
        return quantized_weights
```

#### 3. GGUF Format and llama.cpp Integration

GGUF (GPT-Generated Unified Format) is optimized for CPU inference:

```python
class GGUFConverter:
    """Convert models to GGUF format for llama.cpp"""
    
    def __init__(self, model_path, output_path):
        self.model_path = model_path
        self.output_path = output_path
        
    def convert_to_gguf(self, quantization_type='q4_0'):
        """Convert model to GGUF format"""
        
        # Supported quantization types
        quant_types = {
            'f32': 'Full precision (32-bit float)',
            'f16': 'Half precision (16-bit float)',
            'q8_0': '8-bit quantization',
            'q4_0': '4-bit quantization (legacy)',
            'q4_1': '4-bit quantization (improved)',
            'q5_0': '5-bit quantization',
            'q5_1': '5-bit quantization (improved)',
            'q2_k': '2-bit quantization (k-quant)',
            'q3_k': '3-bit quantization (k-quant)',
            'q4_k': '4-bit quantization (k-quant)',
            'q5_k': '5-bit quantization (k-quant)',
            'q6_k': '6-bit quantization (k-quant)',
            'q8_k': '8-bit quantization (k-quant)'
        }
        
        print(f"Converting to GGUF format with {quantization_type} quantization")
        print(f"Description: {quant_types.get(quantization_type, 'Unknown')}")
        
        # Mock conversion process (in production, use actual llama.cpp tools)
        conversion_config = {
            'input_model': self.model_path,
            'output_path': self.output_path,
            'quantization': quantization_type,
            'metadata': {
                'converted_by': 'GGUF Converter',
                'quantization_version': '1.0',
                'target_architecture': 'llama'
            }
        }
        
        # Simulate conversion steps
        self._prepare_model_for_conversion()
        self._apply_quantization(quantization_type)
        self._write_gguf_file(conversion_config)
        
        return conversion_config
    
    def _prepare_model_for_conversion(self):
        """Prepare model for GGUF conversion"""
        print("Preparing model for conversion...")
        
        # Load model weights
        # Normalize weight formats
        # Validate model architecture
        
        print("Model preparation complete")
    
    def _apply_quantization(self, quant_type):
        """Apply specified quantization"""
        print(f"Applying {quant_type} quantization...")
        
        # Different quantization strategies
        if quant_type.startswith('q2'):
            self._apply_2bit_quantization()
        elif quant_type.startswith('q4'):
            self._apply_4bit_quantization()
        elif quant_type.startswith('q8'):
            self._apply_8bit_quantization()
        
        print("Quantization complete")
    
    def _apply_4bit_quantization(self):
        """Apply 4-bit quantization optimized for CPU"""
        
        # 4-bit quantization parameters
        config = {
            'bits': 4,
            'block_size': 32,  # Process weights in blocks
            'use_zero_point': True,
            'symmetric': False
        }
        
        # Quantization process
        print(f"Applying 4-bit quantization with config: {config}")
    
    def _write_gguf_file(self, config):
        """Write model in GGUF format"""
        print(f"Writing GGUF file to {self.output_path}")
        
        # GGUF file structure
        gguf_structure = {
            'header': {
                'magic': 'GGUF',
                'version': 3,
                'tensor_count': 0,
                'metadata_kv_count': 0
            },
            'metadata': config['metadata'],
            'tensor_info': [],
            'tensor_data': b''
        }
        
        print("GGUF file written successfully")
        return gguf_structure

# Usage example
def demonstrate_gguf_conversion():
    """Demonstrate GGUF conversion process"""
    
    converter = GGUFConverter(
        model_path='./models/llama-7b.bin',
        output_path='./models/llama-7b-q4_0.gguf'
    )
    
    # Convert with different quantization levels
    quantization_types = ['q4_0', 'q5_1', 'q8_0']
    
    for quant_type in quantization_types:
        config = converter.convert_to_gguf(quant_type)
        print(f"Conversion complete: {config['output_path']}")
```

### Hardware-Specific Optimizations

#### CPU Optimization

```python
class CPUQuantizationOptimizer:
    """Optimize quantized models for CPU inference"""
    
    def __init__(self, model):
        self.model = model
        
    def optimize_for_cpu(self):
        """Apply CPU-specific optimizations"""
        
        # Enable CPU-specific optimizations
        optimizations = {
            'use_mkldnn': True,          # Intel MKL-DNN
            'use_openmp': True,          # OpenMP parallelization
            'optimize_memory': True,     # Memory layout optimization
            'fuse_operations': True,     # Operation fusion
            'vectorization': True        # SIMD vectorization
        }
        
        # Apply optimizations
        optimized_model = self._apply_cpu_optimizations(optimizations)
        
        return optimized_model
    
    def _apply_cpu_optimizations(self, optimizations):
        """Apply specific CPU optimizations"""
        
        if optimizations['use_mkldnn']:
            # Enable Intel MKL-DNN backend
            torch.backends.mkldnn.enabled = True
            
        if optimizations['fuse_operations']:
            # Fuse consecutive operations
            self.model = torch.jit.optimize_for_inference(torch.jit.script(self.model))
        
        return self.model
    
    def benchmark_cpu_performance(self, input_data, num_runs=100):
        """Benchmark CPU inference performance"""
        
        import time
        
        # Warmup
        for _ in range(10):
            _ = self.model(input_data)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            _ = self.model(input_data)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = 1.0 / avg_time
        
        return {
            'avg_inference_time': avg_time,
            'throughput_fps': throughput,
            'total_time': end_time - start_time
        }
```

#### Mobile Optimization

```python
class MobileQuantizationOptimizer:
    """Optimize quantized models for mobile deployment"""
    
    def __init__(self, model):
        self.model = model
        
    def optimize_for_mobile(self):
        """Apply mobile-specific optimizations"""
        
        # Mobile optimization configuration
        mobile_config = {
            'target_platform': 'android',  # or 'ios'
            'max_memory_mb': 100,          # Memory constraint
            'target_latency_ms': 50,       # Latency constraint
            'battery_optimization': True,   # Optimize for battery life
            'thermal_management': True      # Prevent overheating
        }
        
        # Apply optimizations
        optimized_model = self._apply_mobile_optimizations(mobile_config)
        
        return optimized_model, mobile_config
    
    def _apply_mobile_optimizations(self, config):
        """Apply mobile-specific optimizations"""
        
        # Convert to mobile-optimized format
        mobile_model = torch.jit.optimize_for_mobile(
            torch.jit.script(self.model),
            optimization_blocklist={'quantized::linear_dynamic'}
        )
        
        return mobile_model
    
    def estimate_mobile_performance(self, input_shape):
        """Estimate mobile performance metrics"""
        
        # Model analysis
        model_size = sum(p.numel() for p in self.model.parameters()) * 4  # bytes
        
        # Estimated metrics (simplified)
        estimated_metrics = {
            'model_size_mb': model_size / (1024 * 1024),
            'estimated_latency_ms': model_size / 1000000 * 10,  # Rough estimate
            'memory_usage_mb': model_size / (1024 * 1024) * 1.5,
            'battery_impact': 'low' if model_size < 50 * 1024 * 1024 else 'medium'
        }
        
        return estimated_metrics
```

### Production Deployment and Monitoring

```python
class QuantizedModelDeployment:
    """Deploy and monitor quantized models in production"""
    
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.metrics = {
            'inference_times': [],
            'accuracy_scores': [],
            'memory_usage': [],
            'throughput': []
        }
        
    def deploy_model(self, deployment_config):
        """Deploy quantized model to production"""
        
        deployment_info = {
            'model_name': self.model_name,
            'quantization_type': deployment_config.get('quantization_type', 'int8'),
            'target_hardware': deployment_config.get('hardware', 'cpu'),
            'optimization_level': deployment_config.get('optimization', 'standard'),
            'deployment_timestamp': time.time()
        }
        
        # Validate model before deployment
        validation_results = self._validate_quantized_model()
        
        if validation_results['valid']:
            print(f"Deploying {self.model_name} to production...")
            self._setup_monitoring()
            print("Deployment successful!")
        else:
            print(f"Deployment failed: {validation_results['issues']}")
        
        return deployment_info
    
    def _validate_quantized_model(self):
        """Validate quantized model before deployment"""
        
        validation_results = {
            'valid': True,
            'issues': [],
            'metrics': {}
        }
        
        try:
            # Test inference
            dummy_input = torch.randn(1, 3, 224, 224)  # Example input
            output = self.model(dummy_input)
            
            # Check output validity
            if torch.isnan(output).any():
                validation_results['valid'] = False
                validation_results['issues'].append('Model produces NaN outputs')
            
            # Check model size
            model_size = sum(p.numel() for p in self.model.parameters())
            validation_results['metrics']['model_size'] = model_size
            
        except Exception as e:
            validation_results['valid'] = False
            validation_results['issues'].append(f'Inference error: {str(e)}')
        
        return validation_results
    
    def _setup_monitoring(self):
        """Setup monitoring for quantized model"""
        
        monitoring_config = {
            'metrics_to_track': [
                'inference_latency',
                'throughput',
                'memory_usage',
                'accuracy_drift',
                'error_rate'
            ],
            'alert_thresholds': {
                'max_latency_ms': 100,
                'min_accuracy': 0.85,
                'max_error_rate': 0.05
            },
            'monitoring_frequency': 'real_time'
        }
        
        print(f"Monitoring setup complete: {monitoring_config}")
        return monitoring_config
    
    def monitor_performance(self, input_data, ground_truth=None):
        """Monitor quantized model performance"""
        
        import time
        
        # Measure inference time
        start_time = time.time()
        output = self.model(input_data)
        inference_time = time.time() - start_time
        
        # Track metrics
        self.metrics['inference_times'].append(inference_time)
        
        # Compute accuracy if ground truth available
        if ground_truth is not None:
            accuracy = self._compute_accuracy(output, ground_truth)
            self.metrics['accuracy_scores'].append(accuracy)
        
        # Monitor memory usage
        memory_usage = self._get_memory_usage()
        self.metrics['memory_usage'].append(memory_usage)
        
        # Check for alerts
        alerts = self._check_performance_alerts(inference_time, accuracy if ground_truth else None)
        
        return {
            'inference_time': inference_time,
            'accuracy': accuracy if ground_truth else None,
            'memory_usage': memory_usage,
            'alerts': alerts
        }
    
    def _compute_accuracy(self, predictions, ground_truth):
        """Compute model accuracy"""
        
        if predictions.dim() > 1:
            predicted_classes = torch.argmax(predictions, dim=1)
        else:
            predicted_classes = predictions
        
        accuracy = (predicted_classes == ground_truth).float().mean().item()
        return accuracy
    
    def _get_memory_usage(self):
        """Get current memory usage"""
        
        if torch.cuda.is_available():
            memory_usage = torch.cuda.memory_allocated() / 1024**2  # MB
        else:
            # For CPU, use a simplified estimation
            memory_usage = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024**2
        
        return memory_usage
    
    def _check_performance_alerts(self, inference_time, accuracy):
        """Check for performance alerts"""
        
        alerts = []
        
        # Latency alert
        if inference_time > 0.1:  # 100ms threshold
            alerts.append(f"High latency detected: {inference_time:.3f}s")
        
        # Accuracy alert
        if accuracy is not None and accuracy < 0.85:
            alerts.append(f"Low accuracy detected: {accuracy:.3f}")
        
        return alerts
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        
        if not self.metrics['inference_times']:
            return {'error': 'No performance data available'}
        
        import numpy as np
        
        report = {
            'model_name': self.model_name,
            'performance_summary': {
                'avg_inference_time': np.mean(self.metrics['inference_times']),
                'p95_inference_time': np.percentile(self.metrics['inference_times'], 95),
                'avg_throughput': 1.0 / np.mean(self.metrics['inference_times']),
                'avg_memory_usage': np.mean(self.metrics['memory_usage']) if self.metrics['memory_usage'] else 0
            },
            'accuracy_summary': {
                'avg_accuracy': np.mean(self.metrics['accuracy_scores']) if self.metrics['accuracy_scores'] else None,
                'min_accuracy': np.min(self.metrics['accuracy_scores']) if self.metrics['accuracy_scores'] else None,
                'accuracy_std': np.std(self.metrics['accuracy_scores']) if self.metrics['accuracy_scores'] else None
            },
            'total_inferences': len(self.metrics['inference_times']),
            'report_timestamp': time.time()
        }
        
        return report
```

### Best Practices for Production Quantization

#### 1. Model Selection and Preparation
- **Choose appropriate quantization method** based on hardware constraints
- **Validate model architecture** compatibility with quantization
- **Prepare high-quality calibration data** representative of production data
- **Establish baseline metrics** before quantization

#### 2. Quantization Strategy
- **Start with PTQ** for quick deployment, move to QAT if accuracy drops
- **Use mixed precision** to balance accuracy and performance
- **Protect critical layers** from aggressive quantization
- **Consider hardware-specific optimizations**

#### 3. Validation and Testing
- **Comprehensive accuracy testing** across diverse datasets
- **Performance benchmarking** on target hardware
- **Stress testing** under production load conditions
- **A/B testing** against full-precision models

#### 4. Deployment and Monitoring
- **Gradual rollout** with careful monitoring
- **Real-time performance tracking** and alerting
- **Accuracy drift detection** and model refresh triggers
- **Fallback mechanisms** to full-precision models if needed

### Why Quantization Matters in Production

Quantization is essential for modern AI deployment because:

- **Cost Efficiency**: Reduces cloud computing costs by 2-4x
- **Edge Deployment**: Enables AI on mobile and IoT devices
- **Real-time Performance**: Achieves low-latency inference requirements
- **Environmental Impact**: Reduces energy consumption and carbon footprint
- **Scalability**: Allows serving more users with same infrastructure
- **Accessibility**: Makes AI accessible on resource-constrained devices

## Exercise (25 minutes)
Complete the hands-on exercises in `exercise.py` to practice quantization techniques.

## Resources
- [PyTorch Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [GPTQ Paper: "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers"](https://arxiv.org/abs/2210.17323)
- [AWQ Paper: "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"](https://arxiv.org/abs/2306.00978)
- [GGML/GGUF Documentation](https://github.com/ggerganov/ggml)
- [Intel Neural Compressor](https://github.com/intel/neural-compressor)
- [NVIDIA TensorRT Quantization](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#working-with-int8)

## Next Steps
- Complete the exercises and explore different quantization methods
- Experiment with hardware-specific optimizations
- Take the quiz to test your understanding
- Move to Day 51: LLM Serving & Optimization - vLLM, TensorRT, Inference
