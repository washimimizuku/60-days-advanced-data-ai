# Day 49 Quiz: RLHF and DPO - Human Feedback & Preference Learning

## Instructions
Choose the best answer for each question. Each question has only one correct answer.

---

1. **What are the three main stages of the RLHF pipeline?**
   - A) Pre-training, Fine-tuning, Evaluation
   - B) Supervised Fine-tuning, Reward Model Training, Reinforcement Learning
   - C) Data Collection, Model Training, Deployment
   - D) Tokenization, Embedding, Generation

2. **What is the primary purpose of the reward model in RLHF?**
   - A) To generate text responses
   - B) To predict human preferences and score model outputs
   - C) To tokenize input text
   - D) To compress the model size

3. **How does DPO differ from traditional RLHF?**
   - A) DPO requires more computational resources
   - B) DPO directly optimizes the policy using preference data without a separate reward model
   - C) DPO only works with smaller models
   - D) DPO requires more human annotators

4. **What is the main advantage of Constitutional AI over standard RLHF?**
   - A) It requires less computational power
   - B) It enables more scalable oversight by using AI feedback instead of only human feedback
   - C) It works only with specific model architectures
   - D) It eliminates the need for any human involvement

5. **In DPO, what does the beta parameter control?**
   - A) The learning rate
   - B) The temperature parameter that controls how much the policy can deviate from the reference model
   - C) The batch size
   - D) The number of training epochs

6. **What is a key challenge in collecting high-quality human preference data?**
   - A) Finding enough computational resources
   - B) Ensuring inter-annotator agreement and consistent annotation quality
   - C) Implementing the training algorithms
   - D) Choosing the right model architecture

7. **What is the purpose of the KL divergence penalty in PPO for RLHF?**
   - A) To speed up training
   - B) To prevent the policy from deviating too far from the reference model
   - C) To reduce memory usage
   - D) To improve text generation quality

8. **Which of the following is NOT a common alignment criterion for human feedback?**
   - A) Helpfulness
   - B) Harmlessness
   - C) Model size
   - D) Honesty

9. **What is alignment drift in the context of deployed AI systems?**
   - A) Changes in model architecture over time
   - B) Degradation of alignment properties as the model encounters new data distributions
   - C) Improvements in model performance
   - D) Changes in hardware requirements

10. **What is a key advantage of DPO over RLHF in terms of training stability?**
    - A) DPO requires fewer GPUs
    - B) DPO avoids the instability issues often associated with reinforcement learning
    - C) DPO works with smaller datasets
    - D) DPO has faster inference speed

---

## Answer Key

**1. B) Supervised Fine-tuning, Reward Model Training, Reinforcement Learning**
- Explanation: The RLHF pipeline consists of three main stages: (1) Supervised Fine-tuning on high-quality demonstrations, (2) Training a reward model on human preference data, and (3) Using reinforcement learning (typically PPO) to optimize the policy against the reward model.

**2. B) To predict human preferences and score model outputs**
- Explanation: The reward model is trained on human preference data to learn what humans consider good vs. bad responses. It then provides scalar reward scores that guide the reinforcement learning process to align the model with human preferences.

**3. B) DPO directly optimizes the policy using preference data without a separate reward model**
- Explanation: DPO's key innovation is that it directly optimizes the policy using preference data by reparameterizing the reward function in terms of the optimal policy, eliminating the need for a separate reward model training stage.

**4. B) It enables more scalable oversight by using AI feedback instead of only human feedback**
- Explanation: Constitutional AI uses AI systems to provide feedback based on a set of principles (constitution), which allows for more scalable oversight since AI can evaluate many more outputs than humans can practically review.

**5. B) The temperature parameter that controls how much the policy can deviate from the reference model**
- Explanation: In DPO, the beta parameter acts as a temperature that controls the strength of the KL regularization, determining how much the optimized policy is allowed to deviate from the reference model during training.

**6. B) Ensuring inter-annotator agreement and consistent annotation quality**
- Explanation: Collecting high-quality preference data requires ensuring that different human annotators agree on preferences and that annotations are consistent and reliable. This involves careful guideline design, quality control, and validation processes.

**7. B) To prevent the policy from deviating too far from the reference model**
- Explanation: The KL divergence penalty in PPO acts as a regularization term that prevents the policy being optimized from straying too far from the reference model, which helps maintain the model's capabilities while improving alignment.

**8. C) Model size**
- Explanation: Model size is a technical specification, not an alignment criterion. The common alignment criteria are helpfulness (how well the model assists users), harmlessness (avoiding harmful outputs), and honesty (providing truthful information).

**9. B) Degradation of alignment properties as the model encounters new data distributions**
- Explanation: Alignment drift refers to the phenomenon where a model's alignment properties may degrade over time as it encounters new data distributions or contexts that differ from its training data, potentially leading to misaligned behavior.

**10. B) DPO avoids the instability issues often associated with reinforcement learning**
- Explanation: DPO provides better training stability compared to RLHF because it directly optimizes the policy using a supervised learning approach rather than using reinforcement learning, which can be unstable and difficult to tune properly.
