# Day 50: Quantization Setup Guide

## Overview
This guide helps you set up the environment for Day 50: Quantization - Model Compression & Optimization. You'll learn to implement and deploy quantized models using various techniques including PTQ, QAT, GPTQ, AWQ, and GGUF formats.

## Prerequisites
- Completed Days 1-49 of the bootcamp
- Python 3.8+ installed
- Basic understanding of neural networks and PyTorch
- Familiarity with model deployment concepts

## Installation

### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv quantization_env

# Activate environment
# On macOS/Linux:
source quantization_env/bin/activate
# On Windows:
quantization_env\Scripts\activate
```

### 2. Install Dependencies
```bash
# Install core requirements
pip install -r requirements.txt

# For GPU support (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For Intel CPU optimizations (optional)
pip install intel-extension-for-pytorch

# For NVIDIA optimizations (if available)
pip install tensorrt
```

### 3. Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Quantization Fundamentals

### What is Quantization?
Quantization reduces the precision of model weights and activations from higher precision (FP32) to lower precision (INT8, INT4), significantly reducing model size and inference time.

### Key Benefits:
- **Memory Reduction**: 2-8x smaller model size
- **Speed Improvement**: 2-4x faster inference
- **Energy Efficiency**: Lower power consumption
- **Cost Reduction**: Lower cloud computing costs

### Quantization Types:

#### 1. Post-Training Quantization (PTQ)
- Quantizes pre-trained model without retraining
- Quick to apply but may have accuracy loss
- Best for: Quick deployment, proof of concepts

#### 2. Quantization-Aware Training (QAT)
- Simulates quantization during training
- Better accuracy preservation
- Best for: Production deployments requiring high accuracy

#### 3. Advanced Methods
- **GPTQ**: Gradient-free quantization for large models
- **AWQ**: Activation-aware weight quantization
- **GGUF**: CPU-optimized format for inference

## Hardware Considerations

### CPU Optimization
```python
# Enable Intel MKL-DNN
torch.backends.mkldnn.enabled = True

# Set optimal thread count
torch.set_num_threads(torch.get_num_threads())
```

### GPU Optimization
```python
# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Use mixed precision
model = model.half()  # FP16
```

### Mobile Optimization
```python
# Convert to mobile-optimized format
mobile_model = torch.jit.optimize_for_mobile(
    torch.jit.script(model)
)
```

## Quick Start Examples

### 1. Post-Training Quantization
```python
import torch
import torch.quantization as quant
from solution import PostTrainingQuantizer, MockCNNModel

# Create model and calibration data
model = MockCNNModel()
calibration_data = [torch.randn(8, 3, 32, 32) for _ in range(10)]

# Apply PTQ
quantizer = PostTrainingQuantizer(model, calibration_data)
quantized_model = quantizer.full_ptq_pipeline()

# Compare sizes
comparison = quantizer.compare_model_sizes(model, quantized_model)
print(f"Compression ratio: {comparison['compression_ratio']:.2f}x")
```

### 2. Quantization-Aware Training
```python
from solution import QuantizationAwareTrainer

# Create trainer
trainer = QuantizationAwareTrainer(model, train_loader, val_loader)

# Apply QAT
trainer.prepare_qat_model()
trainer.train_with_quantization_simulation(epochs=5)
quantized_model = trainer.convert_qat_to_quantized()
```

### 3. GPTQ for Large Models
```python
from solution import GPTQQuantizer, MockTransformerModel

# Create transformer model
model = MockTransformerModel()
calibration_data = [torch.randint(0, 1000, (4, 32)) for _ in range(10)]

# Apply GPTQ
gptq = GPTQQuantizer(model, bits=4, group_size=128)
quantized_model = gptq.quantize_model_with_gptq(calibration_data)
```

### 4. AWQ Activation-Aware Quantization
```python
from solution import AWQQuantizer

# Apply AWQ
awq = AWQQuantizer(model, bits=4, group_size=128)

# Quantize specific layer
for name, layer in model.named_modules():
    if isinstance(layer, torch.nn.Linear):
        awq.quantize_layer_with_awq(layer, calibration_data)
        break
```

### 5. GGUF Format Conversion
```python
from solution import GGUFConverter

# Convert to GGUF format
converter = GGUFConverter("./model.bin", "./model_q4_0.gguf")
result = converter.convert_to_gguf(quantization_type='q4_0')

print(f"Conversion successful: {result['success']}")
print(f"Compression ratio: {result['compression_ratio']:.2f}x")
```

## Performance Benchmarking

### Basic Benchmarking
```python
import time
import torch

def benchmark_model(model, input_data, num_runs=100):
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_data)
    
    # Benchmark
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_data)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    throughput = 1.0 / avg_time
    
    return {
        'avg_inference_time_ms': avg_time * 1000,
        'throughput_fps': throughput
    }

# Usage
input_data = torch.randn(1, 3, 224, 224)
results = benchmark_model(quantized_model, input_data)
print(f"Latency: {results['avg_inference_time_ms']:.2f}ms")
print(f"Throughput: {results['throughput_fps']:.1f} FPS")
```

### Hardware-Specific Optimization
```python
from solution import HardwareOptimizer

# Optimize for different hardware
optimizer = HardwareOptimizer(model, target_hardware='cpu')
optimized_model = optimizer.optimize_for_cpu()

# Benchmark performance
benchmark = optimizer.benchmark_performance(input_data)
print(f"Optimized latency: {benchmark['avg_inference_time_ms']:.2f}ms")
```

## Production Deployment

### Model Validation
```python
from solution import QuantizedModelDeployment

# Create deployment manager
deployment = QuantizedModelDeployment(quantized_model, "my_quantized_model")

# Validate model
validation = deployment.validate_quantized_model()
if validation['valid']:
    print("Model validation passed")
else:
    print(f"Validation issues: {validation['issues']}")
```

### Monitoring Setup
```python
# Setup monitoring
monitoring_config = deployment.setup_monitoring()

# Monitor inference
input_data = torch.randn(1, 3, 32, 32)
ground_truth = torch.randint(0, 10, (1,))

result = deployment.monitor_inference(input_data, ground_truth)
print(f"Inference time: {result['inference_time_ms']:.2f}ms")
print(f"Accuracy: {result['accuracy']:.3f}")

# Generate performance report
report = deployment.generate_performance_report()
print(f"Average latency: {report['performance_summary']['avg_inference_time_ms']:.2f}ms")
```

## Common Issues and Solutions

### Issue 1: Quantization Accuracy Loss
**Problem**: Significant accuracy drop after quantization
**Solutions**:
- Use QAT instead of PTQ
- Increase calibration data quality and quantity
- Use mixed precision (protect sensitive layers)
- Try AWQ to protect important weights

### Issue 2: Slow Quantized Inference
**Problem**: Quantized model not faster than original
**Solutions**:
- Enable hardware-specific optimizations
- Use appropriate quantization backend (fbgemm for CPU, qnnpack for mobile)
- Check if quantization is actually applied
- Optimize batch size and input preprocessing

### Issue 3: Memory Issues During Quantization
**Problem**: Out of memory during quantization process
**Solutions**:
- Reduce calibration batch size
- Process layers sequentially
- Use gradient checkpointing
- Quantize on CPU if GPU memory limited

### Issue 4: GGUF Conversion Errors
**Problem**: Model not compatible with GGUF format
**Solutions**:
- Check model architecture compatibility
- Ensure proper weight format
- Use supported quantization types
- Validate tensor shapes and names

## Testing Your Setup

### Run Basic Tests
```bash
# Run all tests
python -m pytest test_quantization.py -v

# Run specific test categories
python -m pytest test_quantization.py::TestPostTrainingQuantization -v
python -m pytest test_quantization.py::TestGPTQQuantization -v
```

### Verify Quantization Works
```python
# Quick verification script
from solution import *

def verify_setup():
    print("Testing quantization setup...")
    
    # Test PTQ
    model = MockCNNModel()
    calibration_data = [torch.randn(4, 3, 32, 32) for _ in range(5)]
    quantizer = PostTrainingQuantizer(model, calibration_data)
    quantized_model = quantizer.full_ptq_pipeline()
    print("✓ PTQ working")
    
    # Test inference
    input_data = torch.randn(1, 3, 32, 32)
    output = quantized_model(input_data)
    print(f"✓ Quantized inference working: {output.shape}")
    
    # Test deployment
    deployment = QuantizedModelDeployment(quantized_model, "test")
    validation = deployment.validate_quantized_model()
    print(f"✓ Deployment validation: {validation['valid']}")
    
    print("All tests passed! Setup is ready.")

if __name__ == "__main__":
    verify_setup()
```

## Performance Expectations

### Typical Compression Ratios:
- **INT8 PTQ**: 4x size reduction, 2-3x speed improvement
- **INT4 GPTQ**: 8x size reduction, 3-4x speed improvement
- **AWQ**: Similar to GPTQ with better accuracy preservation
- **GGUF q4_0**: 4x size reduction, optimized for CPU inference

### Accuracy Preservation:
- **PTQ**: 1-5% accuracy drop (model dependent)
- **QAT**: 0.5-2% accuracy drop
- **GPTQ/AWQ**: 0.5-3% accuracy drop for 4-bit quantization

## Next Steps

1. **Complete the exercises** in `exercise.py`
2. **Run the test suite** to verify implementations
3. **Experiment with different quantization methods** on your own models
4. **Benchmark performance** on your target hardware
5. **Deploy quantized models** with proper monitoring

## Resources

### Documentation
- [PyTorch Quantization Tutorial](https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html)
- [Transformers Quantization Guide](https://huggingface.co/docs/transformers/quantization)
- [GPTQ Paper](https://arxiv.org/abs/2210.17323)
- [AWQ Paper](https://arxiv.org/abs/2306.00978)

### Tools and Libraries
- [Auto-GPTQ](https://github.com/PanQiWei/AutoGPTQ)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Optimum](https://github.com/huggingface/optimum)
- [Intel Neural Compressor](https://github.com/intel/neural-compressor)

### Hardware-Specific Guides
- [Intel CPU Optimization](https://intel.github.io/intel-extension-for-pytorch/)
- [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/)
- [Apple Core ML](https://developer.apple.com/documentation/coreml)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the test suite for examples
3. Consult the official documentation
4. Ask in the course community forums

Happy quantizing! 🚀