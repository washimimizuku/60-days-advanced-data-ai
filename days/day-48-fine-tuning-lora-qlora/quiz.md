# Day 48 Quiz: Fine-tuning Techniques - LoRA & QLoRA

## Instructions
Choose the best answer for each question. Each question has only one correct answer.

---

1. **What is the main principle behind LoRA (Low-Rank Adaptation)?**
   - A) It freezes all model parameters during fine-tuning
   - B) It decomposes weight updates into two low-rank matrices
   - C) It quantizes the model weights to 4-bit precision
   - D) It uses gradient checkpointing to save memory

2. **In LoRA, what does the rank parameter (r) control?**
   - A) The learning rate for the adaptation
   - B) The number of training epochs
   - C) The dimensionality of the low-rank decomposition
   - D) The batch size for training

3. **What is the typical relationship between LoRA's alpha and rank parameters?**
   - A) Alpha should be equal to rank
   - B) Alpha should be approximately 2 times the rank
   - C) Alpha should be much smaller than rank
   - D) Alpha and rank are independent parameters

4. **What is the primary advantage of QLoRA over standard LoRA?**
   - A) Better model performance on downstream tasks
   - B) Faster training convergence
   - C) Significantly reduced memory requirements through quantization
   - D) Support for larger batch sizes

5. **Which quantization technique does QLoRA use for optimal performance?**
   - A) 8-bit integer quantization
   - B) 4-bit NormalFloat (NF4) quantization
   - C) Binary quantization
   - D) Dynamic quantization

6. **What is "double quantization" in QLoRA?**
   - A) Quantizing both weights and activations
   - B) Applying quantization twice to the same weights
   - C) Quantizing the quantization constants themselves
   - D) Using two different quantization methods simultaneously

7. **Which layers are typically targeted for LoRA adaptation in transformer models?**
   - A) Only the embedding layers
   - B) Only the final classification layer
   - C) Attention projection layers (q_proj, v_proj, etc.)
   - D) Only the normalization layers

8. **What is a key benefit of parameter-efficient fine-tuning methods like LoRA?**
   - A) They always achieve better performance than full fine-tuning
   - B) They require no training data
   - C) They can share the same base model across multiple tasks
   - D) They eliminate the need for validation data

9. **In production, what is an important consideration when choosing the LoRA rank?**
   - A) Higher rank always leads to better performance
   - B) Rank should be balanced between model capacity and efficiency
   - C) Rank must be a power of 2
   - D) Rank should equal the model's hidden dimension

10. **What is the main advantage of using paged optimizers in QLoRA?**
    - A) Faster gradient computation
    - B) Better convergence properties
    - C) Handling memory spikes through unified memory
    - D) Reduced communication overhead in distributed training

---

## Answer Key

**1. B) It decomposes weight updates into two low-rank matrices**
- Explanation: LoRA's core innovation is representing weight updates ΔW as the product of two low-rank matrices B and A, where ΔW = BA. This dramatically reduces the number of trainable parameters while maintaining adaptation capability.

**2. C) The dimensionality of the low-rank decomposition**
- Explanation: The rank parameter r determines the dimensionality of the low-rank matrices A (r×k) and B (d×r). Higher rank provides more capacity for adaptation but increases the number of trainable parameters.

**3. B) Alpha should be approximately 2 times the rank**
- Explanation: The alpha parameter controls the scaling of LoRA updates. A common best practice is to set alpha ≈ 2×rank, which provides a good balance between adaptation strength and stability during training.

**4. C) Significantly reduced memory requirements through quantization**
- Explanation: QLoRA's main advantage is combining LoRA with 4-bit quantization, enabling fine-tuning of very large models (65B+ parameters) on consumer hardware by dramatically reducing memory requirements.

**5. B) 4-bit NormalFloat (NF4) quantization**
- Explanation: QLoRA uses NF4, a novel 4-bit data type specifically designed for normally distributed weights common in neural networks. This provides better quantization quality than standard 4-bit integer quantization.

**6. C) Quantizing the quantization constants themselves**
- Explanation: Double quantization in QLoRA means that even the quantization constants (scaling factors) are quantized to save additional memory, further reducing the memory footprint without significant performance loss.

**7. C) Attention projection layers (q_proj, v_proj, etc.)**
- Explanation: LoRA is typically applied to attention projection layers (query, key, value, output projections) and sometimes MLP layers, as these are the most impactful for model adaptation while being computationally efficient.

**8. C) They can share the same base model across multiple tasks**
- Explanation: A key advantage of PEFT methods is that multiple task-specific adapters can share the same frozen base model, dramatically reducing storage requirements and enabling efficient multi-task deployment.

**9. B) Rank should be balanced between model capacity and efficiency**
- Explanation: In production, rank selection involves balancing adaptation capacity (higher rank = more expressive) with efficiency (lower rank = fewer parameters, faster training/inference). The optimal rank depends on task complexity and resource constraints.

**10. C) Handling memory spikes through unified memory**
- Explanation: Paged optimizers in QLoRA use NVIDIA's unified memory to automatically handle memory spikes by moving data between GPU and CPU memory as needed, preventing out-of-memory errors during training.
