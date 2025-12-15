# Day 50 Quiz: Quantization - Model Compression & Optimization

## Instructions
Choose the best answer for each question. Each question has only one correct answer.

---

1. **What is the primary benefit of model quantization?**
   - A) Improved model accuracy
   - B) Reduced model size and faster inference with minimal accuracy loss
   - C) Better gradient flow during training
   - D) Enhanced model interpretability

2. **What is the difference between Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)?**
   - A) PTQ is only for CNNs, QAT is for transformers
   - B) PTQ quantizes after training without retraining, QAT simulates quantization during training
   - C) PTQ uses INT8, QAT uses INT4
   - D) PTQ is faster but less accurate than QAT

3. **In the quantization formula `quantized_value = round((float_value - zero_point) / scale)`, what does the scale parameter control?**
   - A) The range of quantized values
   - B) The step size between quantized values
   - C) The precision of the original model
   - D) The compression ratio

4. **What is the key innovation of GPTQ (Gradient-Free Post-Training Quantization)?**
   - A) It uses gradients to optimize quantization
   - B) It processes weights layer by layer using Hessian information for optimal quantization
   - C) It only works with transformer models
   - D) It requires retraining the entire model

5. **How does AWQ (Activation-aware Weight Quantization) differ from standard quantization methods?**
   - A) It quantizes activations instead of weights
   - B) It protects important weights based on activation patterns from aggressive quantization
   - C) It only works on the attention layers
   - D) It uses a different number format

6. **What is the GGUF format primarily optimized for?**
   - A) GPU inference with CUDA
   - B) CPU inference with llama.cpp and similar engines
   - C) Mobile deployment on Android
   - D) Cloud-based serving

7. **Which quantization precision typically provides the best balance between model size and accuracy for large language models?**
   - A) INT8 (8-bit)
   - B) INT4 (4-bit)
   - C) INT2 (2-bit)
   - D) FP16 (16-bit float)

8. **What is a critical consideration when selecting calibration data for post-training quantization?**
   - A) Use as much data as possible regardless of quality
   - B) Use data that is representative of the production distribution
   - C) Use only the training data
   - D) Use randomly generated synthetic data

9. **Which hardware-specific optimization is most important for mobile deployment of quantized models?**
   - A) Memory bandwidth optimization
   - B) Battery life and thermal management
   - C) Multi-GPU parallelization
   - D) High-precision arithmetic units

10. **What should you monitor when deploying quantized models in production?**
    - A) Only inference latency
    - B) Only model accuracy
    - C) Inference latency, accuracy drift, memory usage, and error rates
    - D) Only memory usage

---

## Answer Key

**1. B) Reduced model size and faster inference with minimal accuracy loss**
- Explanation: The primary benefit of quantization is reducing model size (typically 2-8x smaller) and improving inference speed (2-4x faster) while maintaining acceptable accuracy. This enables deployment on resource-constrained devices and reduces computational costs.

**2. B) PTQ quantizes after training without retraining, QAT simulates quantization during training**
- Explanation: Post-Training Quantization (PTQ) applies quantization to an already trained model without additional training, while Quantization-Aware Training (QAT) incorporates quantization simulation during the training process to maintain better accuracy.

**3. B) The step size between quantized values**
- Explanation: The scale parameter determines the granularity of quantization by controlling how much each quantized integer step represents in the original floating-point space. A smaller scale means finer quantization granularity.

**4. B) It processes weights layer by layer using Hessian information for optimal quantization**
- Explanation: GPTQ's key innovation is using second-order information (Hessian) to optimally quantize weights layer by layer, minimizing the impact on model outputs. This allows for aggressive quantization (like 4-bit) while maintaining model quality.

**5. B) It protects important weights based on activation patterns from aggressive quantization**
- Explanation: AWQ analyzes activation patterns to identify which weights are most important for model performance and protects these critical weights from aggressive quantization, while quantizing less important weights more aggressively.

**6. B) CPU inference with llama.cpp and similar engines**
- Explanation: GGUF (GPT-Generated Unified Format) is specifically designed for efficient CPU inference, particularly with llama.cpp and similar engines. It includes optimizations for CPU architectures and memory layouts.

**7. B) INT4 (4-bit)**
- Explanation: INT4 quantization typically provides the best balance for large language models, offering significant size reduction (4x smaller than FP16) while maintaining acceptable quality. INT8 is more conservative, while INT2 often degrades quality too much.

**8. B) Use data that is representative of the production distribution**
- Explanation: Calibration data should be representative of the actual data the model will see in production. This ensures that the quantization parameters (scale and zero-point) are optimized for the real-world distribution, maintaining accuracy.

**9. B) Battery life and thermal management**
- Explanation: For mobile deployment, battery life and thermal management are critical. Quantized models help reduce power consumption and heat generation, which are major constraints on mobile devices compared to server environments.

**10. C) Inference latency, accuracy drift, memory usage, and error rates**
- Explanation: Comprehensive monitoring of quantized models should include multiple metrics: latency (performance), accuracy drift (quality degradation over time), memory usage (resource consumption), and error rates (reliability). This ensures the quantized model maintains production requirements.
