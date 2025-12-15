# Day 41 Quiz: Transformer Architecture - Attention, Encoders, Decoders

## Instructions
This quiz tests your understanding of Transformer architecture fundamentals, attention mechanisms, and their applications in modern AI systems. Each question builds on the concepts covered in today's lesson.

---

### 1. **What is the key innovation that makes Transformers different from RNNs and CNNs?**
   - a) They use deeper networks with more parameters
   - b) They process sequences in parallel using self-attention mechanisms
   - c) They require less training data to achieve good performance
   - d) They can only be used for natural language processing tasks

### 2. **In the scaled dot-product attention formula Attention(Q,K,V) = softmax(QK^T/√d_k)V, why is the scaling factor √d_k used?**
   - a) To make the computation faster by reducing the matrix size
   - b) To prevent the softmax function from saturating when d_k is large
   - c) To ensure the attention weights sum to exactly 1.0
   - d) To make the gradients exactly zero during backpropagation

### 3. **What is the primary purpose of positional encoding in Transformers?**
   - a) To reduce the computational complexity of attention mechanisms
   - b) To inject information about token positions since attention is permutation-invariant
   - c) To normalize the input embeddings for better training stability
   - d) To compress the sequence length for memory efficiency

### 4. **In multi-head attention, what is the main advantage of using multiple attention heads?**
   - a) It reduces the total number of parameters in the model
   - b) It allows the model to attend to different types of relationships simultaneously
   - c) It makes the model train faster by parallelizing computations
   - d) It prevents overfitting by adding regularization

### 5. **What is the key difference between encoder and decoder layers in a Transformer?**
   - a) Encoders use self-attention while decoders use cross-attention only
   - b) Encoders have more parameters than decoders
   - c) Decoders use masked self-attention to prevent looking at future tokens
   - d) Encoders process input tokens while decoders only generate outputs

### 6. **Which of the following best describes the computational complexity of self-attention?**
   - a) O(n) where n is the sequence length
   - b) O(n log n) where n is the sequence length
   - c) O(n²) where n is the sequence length
   - d) O(n³) where n is the sequence length

### 7. **What is the main purpose of residual connections (skip connections) in Transformer layers?**
   - a) To reduce the number of parameters and make the model smaller
   - b) To allow gradients to flow directly and stabilize training of deep networks
   - c) To implement the attention mechanism more efficiently
   - d) To prevent the model from overfitting to the training data

### 8. **In the context of modern LLMs, what type of Transformer architecture does GPT use?**
   - a) Encoder-only architecture like BERT
   - b) Decoder-only architecture with causal masking
   - c) Full encoder-decoder architecture like T5
   - d) A hybrid architecture combining CNN and Transformer layers

### 9. **What is the primary challenge when scaling Transformers to very long sequences?**
   - a) The model becomes too deep and suffers from vanishing gradients
   - b) The quadratic memory and computational complexity of attention
   - c) The positional encodings become inaccurate for long sequences
   - d) The model loses the ability to capture local patterns

### 10. **Which technique is commonly used to optimize Transformer inference in production?**
   - a) Increasing the batch size to maximize GPU utilization
   - b) Using model quantization to reduce precision and memory usage
   - c) Adding more attention heads to improve parallel processing
   - d) Removing positional encodings to reduce computational overhead

---

## Answer Key

### 1. **What is the key innovation that makes Transformers different from RNNs and CNNs?**
**Answer: b) They process sequences in parallel using self-attention mechanisms**

**Explanation:** The key innovation of Transformers is the self-attention mechanism that allows all positions in a sequence to be processed in parallel, unlike RNNs which process sequentially. This parallel processing enables much faster training and better capture of long-range dependencies. While Transformers can be large, their size isn't the key differentiator, and they're used beyond NLP (e.g., Vision Transformers).

### 2. **In the scaled dot-product attention formula, why is the scaling factor √d_k used?**
**Answer: b) To prevent the softmax function from saturating when d_k is large**

**Explanation:** When d_k (the dimension of the key vectors) is large, the dot products QK^T can become very large in magnitude. This pushes the softmax function into regions where it has extremely small gradients, making training difficult. Dividing by √d_k keeps the dot products in a reasonable range, ensuring the softmax doesn't saturate and gradients remain stable during training.

### 3. **What is the primary purpose of positional encoding in Transformers?**
**Answer: b) To inject information about token positions since attention is permutation-invariant**

**Explanation:** Self-attention is inherently permutation-invariant, meaning it treats the input as a set rather than a sequence. Without positional encoding, the model couldn't distinguish between "The cat sat on the mat" and "Mat the on sat cat the." Positional encoding adds position-specific information to each token embedding, allowing the model to understand sequence order.

### 4. **In multi-head attention, what is the main advantage of using multiple attention heads?**
**Answer: b) It allows the model to attend to different types of relationships simultaneously**

**Explanation:** Multiple attention heads allow the model to learn different types of relationships in parallel. For example, one head might focus on syntactic relationships (subject-verb agreement), another on semantic relationships (word meanings), and another on long-range dependencies. This increases the model's representational capacity without dramatically increasing parameters.

### 5. **What is the key difference between encoder and decoder layers in a Transformer?**
**Answer: c) Decoders use masked self-attention to prevent looking at future tokens**

**Explanation:** The key difference is that decoder layers use masked (causal) self-attention, which prevents tokens from attending to future positions in the sequence. This is essential for autoregressive generation where the model should only use information from previous tokens to predict the next token. Encoders can attend to all positions since they process the entire input simultaneously.

### 6. **Which of the following best describes the computational complexity of self-attention?**
**Answer: c) O(n²) where n is the sequence length**

**Explanation:** Self-attention has quadratic complexity O(n²) because every token needs to compute attention weights with every other token in the sequence. For a sequence of length n, this requires n² attention computations. This quadratic scaling is one of the main limitations of Transformers for very long sequences, leading to research on sparse and linear attention mechanisms.

### 7. **What is the main purpose of residual connections in Transformer layers?**
**Answer: b) To allow gradients to flow directly and stabilize training of deep networks**

**Explanation:** Residual connections (skip connections) allow gradients to flow directly from later layers to earlier layers during backpropagation, preventing the vanishing gradient problem in deep networks. They also help with training stability by ensuring that each layer learns incremental changes rather than completely new representations, making it easier to train very deep Transformer models.

### 8. **In the context of modern LLMs, what type of Transformer architecture does GPT use?**
**Answer: b) Decoder-only architecture with causal masking**

**Explanation:** GPT (Generative Pre-trained Transformer) uses a decoder-only architecture where each layer has masked self-attention (causal masking) to ensure autoregressive generation. Unlike BERT (encoder-only) or T5 (encoder-decoder), GPT doesn't have separate encoder layers. The causal masking ensures that when predicting token i, the model can only see tokens 1 through i-1.

### 9. **What is the primary challenge when scaling Transformers to very long sequences?**
**Answer: b) The quadratic memory and computational complexity of attention**

**Explanation:** The main challenge is the O(n²) complexity of attention mechanisms. For a sequence of length n, the attention matrix requires n² memory and computation, which becomes prohibitive for very long sequences. For example, a sequence of 10,000 tokens requires 100 million attention computations. This has led to research on sparse attention patterns and linear attention approximations.

### 10. **Which technique is commonly used to optimize Transformer inference in production?**
**Answer: b) Using model quantization to reduce precision and memory usage**

**Explanation:** Model quantization, which reduces the precision of model weights and activations (e.g., from FP32 to INT8 or INT4), is a common optimization technique. It significantly reduces memory usage and can speed up inference while maintaining acceptable performance. Other techniques include pruning, knowledge distillation, and caching, but quantization is one of the most widely adopted approaches.

---

## Scoring Guide

- **9-10 correct**: Excellent understanding of Transformer architecture - Ready for advanced GenAI topics
- **7-8 correct**: Good grasp of fundamentals - Review specific concepts before proceeding
- **5-6 correct**: Basic understanding - Recommend reviewing attention mechanisms and architecture details
- **3-4 correct**: Foundational gaps - Significant review of Transformer concepts needed
- **0-2 correct**: Comprehensive review required - Consider additional study materials

## Areas for Review Based on Incorrect Answers

- **Questions 1, 8**: Review Transformer variants and their differences (encoder-only, decoder-only, encoder-decoder)
- **Questions 2, 6**: Review attention mechanism mathematics and computational complexity
- **Questions 3, 5**: Review positional encoding and masked attention concepts
- **Questions 4, 7**: Review multi-head attention and residual connections
- **Questions 9, 10**: Review scaling challenges and production optimization techniques

Take time to review any areas where you scored incorrectly. A strong understanding of Transformer fundamentals is essential for the advanced GenAI topics in the coming days.
