# Day 42 Quiz: Attention Mechanisms Deep Dive

## Instructions
Answer all 10 questions. Each question has 4 options (a, b, c, d). Choose the best answer.

---

### 1. What is the key difference between self-attention and cross-attention?
- a) Self-attention uses different weight matrices for Q, K, V
- b) Cross-attention allows positions in one sequence to attend to positions in another sequence
- c) Self-attention is only used in encoders, cross-attention only in decoders
- d) Cross-attention requires positional encoding, self-attention does not

**Answer: b) Cross-attention allows positions in one sequence to attend to positions in another sequence**

**Explanation:** Cross-attention enables alignment between different sequences (e.g., source and target in translation), where queries come from one sequence and keys/values from another. Self-attention operates within a single sequence where all positions can attend to each other. This distinction is crucial for tasks like machine translation, image captioning, and multimodal applications.

---

### 2. Why is causal masking essential in autoregressive language models like GPT?
- a) It reduces computational complexity from O(n²) to O(n)
- b) It prevents the model from attending to future tokens during training and inference
- c) It improves the quality of attention weights by focusing on relevant tokens
- d) It enables parallel processing of all tokens simultaneously

**Answer: b) It prevents the model from attending to future tokens during training and inference**

**Explanation:** Causal masking (lower triangular mask) ensures that during training, the model cannot "cheat" by looking at future tokens when predicting the next token. This maintains the autoregressive property where each token is predicted based only on previous context, making training consistent with inference where tokens are generated sequentially.

---

### 3. What is the primary advantage of sparse attention mechanisms like Longformer?
- a) They provide better attention quality than full attention
- b) They reduce memory and computational complexity for long sequences
- c) They eliminate the need for positional encoding
- d) They work better with small datasets

**Answer: b) They reduce memory and computational complexity for long sequences**

**Explanation:** Sparse attention patterns (sliding window + global tokens) reduce complexity from O(n²) to O(n×w) where w is the window size. This enables processing of much longer sequences (e.g., 4K+ tokens) that would be computationally prohibitive with full attention, while maintaining most of the modeling capability.

---

### 4. In multi-head attention, what is the purpose of having multiple attention heads?
- a) To increase the model's parameter count for better capacity
- b) To capture different types of relationships and patterns in parallel
- c) To reduce overfitting by averaging multiple attention patterns
- d) To enable faster parallel computation on multiple GPUs

**Answer: b) To capture different types of relationships and patterns in parallel**

**Explanation:** Different attention heads learn to focus on different aspects: some capture syntactic relationships (subject-verb), others semantic similarity, positional patterns, or coreference. This parallel specialization allows the model to simultaneously process multiple types of linguistic and semantic relationships, making it more expressive than single-head attention.

---

### 5. What does Flash Attention optimize compared to standard attention implementation?
- a) It approximates attention with linear complexity
- b) It uses sparse attention patterns to reduce computation
- c) It optimizes memory access patterns while maintaining exact attention computation
- d) It quantizes attention weights to reduce memory usage

**Answer: c) It optimizes memory access patterns while maintaining exact attention computation**

**Explanation:** Flash Attention uses block-wise computation and tiling to optimize GPU memory hierarchy (SRAM vs HBM), achieving 2-4x speedup without any approximation. Unlike sparse or linear attention methods, it computes exact attention but does so more efficiently by minimizing memory transfers and using online softmax computation.

---

### 6. What is attention entropy and what does it measure?
- a) The randomness in attention weight initialization
- b) How focused or distributed the attention pattern is
- c) The information content of the input sequence
- d) The computational cost of attention computation

**Answer: b) How focused or distributed the attention pattern is**

**Explanation:** Attention entropy measures the concentration of attention weights: low entropy indicates focused attention (peaked distribution on few tokens), while high entropy indicates distributed attention (uniform across many tokens). This metric helps analyze model behavior and can indicate whether the model is attending to specific relevant tokens or spreading attention broadly.

---

### 7. In Vision Transformers (ViTs), how are images processed through attention mechanisms?
- a) Pixels are treated as individual tokens in a sequence
- b) Images are divided into patches that are treated as tokens
- c) Convolutional features are used as attention keys and values
- d) Each color channel is processed with separate attention heads

**Answer: b) Images are divided into patches that are treated as tokens**

**Explanation:** ViTs divide images into fixed-size patches (e.g., 16×16 pixels), flatten and embed each patch as a token, add positional encoding, and apply standard Transformer attention. This allows the model to capture long-range spatial dependencies between different image regions, unlike CNNs which have limited receptive fields.

---

### 8. What is the key innovation of Performer (Linear Attention)?
- a) It uses sparse attention patterns to reduce complexity
- b) It approximates attention using kernel methods with feature maps φ(Q) and φ(K)
- c) It caches attention weights to speed up inference
- d) It uses quantized attention weights to reduce memory

**Answer: b) It approximates attention using kernel methods with feature maps φ(Q) and φ(K)**

**Explanation:** Performer approximates the attention softmax kernel using random feature maps: Attention(Q,K,V) ≈ φ(Q)(φ(K)ᵀV), where φ maps to a higher-dimensional space. This reformulation enables linear O(n) complexity while providing an unbiased approximation that converges to true attention, making it suitable for very long sequences.

---

### 9. What is the purpose of attention visualization and what insights can it provide?
- a) To debug computational errors in attention implementation
- b) To understand what linguistic and semantic patterns the model has learned
- c) To optimize attention weights for better performance
- d) To compress attention matrices for deployment

**Answer: b) To understand what linguistic and semantic patterns the model has learned**

**Explanation:** Attention visualization reveals interpretable patterns: syntactic heads focusing on grammatical relationships, semantic heads capturing meaning similarity, positional heads tracking relative positions. This helps researchers understand model behavior, debug issues, improve architectures, and build trust in model decisions for critical applications.

---

### 10. In production deployment, what is a key consideration for attention-based models with long sequences?
- a) Using higher learning rates to speed up training
- b) Implementing memory-efficient attention variants and caching strategies
- c) Increasing the number of attention heads for better quality
- d) Using larger batch sizes to improve throughput

**Answer: b) Implementing memory-efficient attention variants and caching strategies**

**Explanation:** Long sequences cause quadratic memory growth in standard attention, leading to out-of-memory errors. Production systems need: memory-efficient implementations (Flash Attention), KV caching for inference, gradient checkpointing, sequence parallelism, and potentially sparse attention patterns. These optimizations enable handling longer contexts while maintaining reasonable resource usage and latency.

---

## Answer Key
1. b) Cross-attention allows positions in one sequence to attend to positions in another sequence
2. b) It prevents the model from attending to future tokens during training and inference  
3. b) They reduce memory and computational complexity for long sequences
4. b) To capture different types of relationships and patterns in parallel
5. c) It optimizes memory access patterns while maintaining exact attention computation
6. b) How focused or distributed the attention pattern is
7. b) Images are divided into patches that are treated as tokens
8. b) It approximates attention using kernel methods with feature maps φ(Q) and φ(K)
9. b) To understand what linguistic and semantic patterns the model has learned
10. b) Implementing memory-efficient attention variants and caching strategies

## Scoring
- 9-10 correct: Excellent understanding of attention mechanisms
- 7-8 correct: Good grasp with minor gaps
- 5-6 correct: Basic understanding, review key concepts
- Below 5: Revisit the material and practice implementations
