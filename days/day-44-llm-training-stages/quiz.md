# Day 44 Quiz: LLM Training Stages

## Instructions
Answer all 10 questions. Each question has 4 options (a, b, c, d). Choose the best answer.

---

### 1. What is the primary objective of the pre-training stage in LLM development?
- a) To align the model with human preferences and values
- b) To learn general language understanding through next-token prediction on massive corpora
- c) To fine-tune the model for specific downstream tasks
- d) To reduce the model size for efficient deployment

**Answer: b) To learn general language understanding through next-token prediction on massive corpora**

**Explanation:** Pre-training is the foundational stage where models learn general language capabilities by predicting the next token in sequences from massive text corpora (trillions of tokens). This unsupervised learning develops broad language understanding, grammar, facts, and reasoning abilities that serve as the foundation for all subsequent training stages.

---

### 2. What is the key difference between LoRA (Low-Rank Adaptation) and full fine-tuning?
- a) LoRA requires more computational resources than full fine-tuning
- b) LoRA only updates a small number of additional parameters while freezing the original model weights
- c) LoRA can only be used during pre-training, not fine-tuning
- d) LoRA produces lower quality results than full fine-tuning

**Answer: b) LoRA only updates a small number of additional parameters while freezing the original model weights**

**Explanation:** LoRA is a parameter-efficient fine-tuning method that freezes the original pre-trained weights and adds small, trainable low-rank matrices. This dramatically reduces the number of trainable parameters (often by 99%+) while maintaining comparable performance to full fine-tuning, making it much more memory and compute efficient.

---

### 3. In Reinforcement Learning from Human Feedback (RLHF), what is the purpose of the reward model?
- a) To generate new training data for the language model
- b) To predict human preferences and provide reward signals for policy optimization
- c) To compress the model size for faster inference
- d) To tokenize input text more efficiently

**Answer: b) To predict human preferences and provide reward signals for policy optimization**

**Explanation:** The reward model in RLHF is trained on human preference data to predict which responses humans would prefer. It then provides reward signals during reinforcement learning (typically PPO) to guide the language model toward generating responses that align with human preferences and values.

---

### 4. What is the main advantage of Constitutional AI (CAI) over traditional RLHF?
- a) CAI requires less computational resources
- b) CAI uses the model's own self-critique capabilities to improve alignment without extensive human feedback
- c) CAI only works with smaller models
- d) CAI eliminates the need for any human supervision

**Answer: b) CAI uses the model's own self-critique capabilities to improve alignment without extensive human feedback**

**Explanation:** Constitutional AI leverages the model's own reasoning abilities to critique and revise its responses according to a set of principles (constitution). This reduces dependence on extensive human feedback collection while still achieving alignment, as the model learns to self-correct based on constitutional principles.

---

### 5. Which distributed training strategy is most effective for training very large language models (100B+ parameters)?
- a) Data parallelism only
- b) Model parallelism only
- c) A combination of data parallelism, model parallelism, and pipeline parallelism
- d) Single-GPU training with gradient accumulation

**Answer: c) A combination of data parallelism, model parallelism, and pipeline parallelism**

**Explanation:** Very large models require hybrid parallelism strategies: model parallelism splits the model across devices, pipeline parallelism processes different stages concurrently, and data parallelism distributes batches. This combination, often called 3D parallelism, is essential for efficiently training models that don't fit on single devices.

---

### 6. What is the primary purpose of gradient checkpointing in LLM training?
- a) To save model weights during training
- b) To trade computation for memory by recomputing activations during backpropagation
- c) To improve model accuracy
- d) To speed up the forward pass

**Answer: b) To trade computation for memory by recomputing activations during backpropagation**

**Explanation:** Gradient checkpointing reduces memory usage by storing only selected intermediate activations during the forward pass and recomputing others during backpropagation. This memory-computation tradeoff allows training larger models or using larger batch sizes on the same hardware.

---

### 7. In instruction tuning, why are only the output tokens typically used for loss computation?
- a) Input tokens are not important for learning
- b) To focus learning on generating appropriate responses rather than memorizing instructions
- c) Output tokens contain more information than input tokens
- d) It reduces computational requirements significantly

**Answer: b) To focus learning on generating appropriate responses rather than memorizing instructions**

**Explanation:** By masking instruction and input tokens in the loss computation (setting their labels to -100), the model learns to generate appropriate outputs given instructions without wasting capacity on predicting the instruction text itself. This focuses learning on the desired behavior: following instructions to produce helpful responses.

---

### 8. What is the main benefit of Direct Preference Optimization (DPO) compared to RLHF?
- a) DPO requires more human feedback data
- b) DPO eliminates the need for a separate reward model while achieving similar alignment results
- c) DPO only works with smaller models
- d) DPO is more computationally expensive

**Answer: b) DPO eliminates the need for a separate reward model while achieving similar alignment results**

**Explanation:** DPO directly optimizes the language model using preference data without requiring a separate reward model training phase. This simplifies the alignment pipeline while achieving comparable results to RLHF, reducing complexity and potential instabilities from the reward model.

---

### 9. Why is mixed precision training (FP16/FP32) commonly used in LLM training?
- a) It improves model accuracy by using higher precision
- b) It reduces memory usage and increases training speed while maintaining numerical stability
- c) It is required for distributed training
- d) It eliminates the need for gradient clipping

**Answer: b) It reduces memory usage and increases training speed while maintaining numerical stability**

**Explanation:** Mixed precision uses FP16 for forward pass computations (reducing memory and increasing speed) while keeping FP32 for gradients and optimizer states (maintaining numerical stability). This provides significant memory savings and speedup with minimal impact on training stability or final model quality.

---

### 10. What is the typical learning rate adjustment when transitioning from pre-training to fine-tuning?
- a) Use the same learning rate as pre-training
- b) Increase the learning rate by 10x
- c) Decrease the learning rate by 5-10x compared to pre-training
- d) Set the learning rate to zero

**Answer: c) Decrease the learning rate by 5-10x compared to pre-training**

**Explanation:** Fine-tuning typically uses a much lower learning rate (5-10x smaller) than pre-training because the model has already learned general language capabilities and needs only gentle adjustments for specific tasks. Higher learning rates could disrupt the valuable pre-trained representations and lead to catastrophic forgetting.

---

## Answer Key
1. b) To learn general language understanding through next-token prediction on massive corpora
2. b) LoRA only updates a small number of additional parameters while freezing the original model weights
3. b) To predict human preferences and provide reward signals for policy optimization
4. b) CAI uses the model's own self-critique capabilities to improve alignment without extensive human feedback
5. c) A combination of data parallelism, model parallelism, and pipeline parallelism
6. b) To trade computation for memory by recomputing activations during backpropagation
7. b) To focus learning on generating appropriate responses rather than memorizing instructions
8. b) DPO eliminates the need for a separate reward model while achieving similar alignment results
9. b) It reduces memory usage and increases training speed while maintaining numerical stability
10. c) Decrease the learning rate by 5-10x compared to pre-training

## Scoring
- 9-10 correct: Excellent understanding of LLM training stages
- 7-8 correct: Good grasp with minor gaps
- 5-6 correct: Basic understanding, review key concepts
- Below 5: Revisit the material and practice implementations
