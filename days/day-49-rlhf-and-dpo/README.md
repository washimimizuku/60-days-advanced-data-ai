# Day 49: RLHF and DPO - Human Feedback & Preference Learning

## Learning Objectives
By the end of this session, you will be able to:
- Understand Reinforcement Learning from Human Feedback (RLHF) methodology
- Implement Direct Preference Optimization (DPO) as an alternative to RLHF
- Design human preference collection and annotation systems
- Build reward models for alignment training
- Apply constitutional AI principles for scalable oversight
- Deploy aligned models in production environments

## Theory (30 minutes)

### What is AI Alignment?

AI Alignment refers to ensuring that AI systems behave in accordance with human values, intentions, and preferences. As language models become more capable, alignment becomes crucial for safe and beneficial deployment.

**Key Alignment Challenges:**
- **Specification Problem**: Difficulty in precisely defining what we want AI to do
- **Goodhart's Law**: When a measure becomes a target, it ceases to be a good measure
- **Distributional Shift**: Models may behave differently in new contexts
- **Scalable Oversight**: Human evaluation doesn't scale to all model outputs
- **Value Learning**: Learning complex human preferences from limited feedback

### Reinforcement Learning from Human Feedback (RLHF)

RLHF is a technique that uses human preferences to train AI systems to produce outputs that are more aligned with human values and intentions.

#### RLHF Pipeline Overview

```
1. Supervised Fine-tuning (SFT)
   ↓
2. Reward Model Training
   ↓  
3. Reinforcement Learning with PPO
   ↓
4. Aligned Model
```

#### Stage 1: Supervised Fine-tuning (SFT)

Start with a pre-trained language model and fine-tune it on high-quality demonstrations:

```python
class SupervisedFineTuning:
    def __init__(self, base_model, tokenizer):
        self.model = base_model
        self.tokenizer = tokenizer
        
    def prepare_sft_data(self, demonstrations):
        """Prepare demonstration data for supervised fine-tuning"""
        formatted_data = []
        
        for demo in demonstrations:
            # Format as instruction-following examples
            prompt = demo['instruction']
            response = demo['response']
            
            # Create input-output pairs
            input_text = f"Human: {prompt}\n\nAssistant: {response}"
            formatted_data.append({
                'input_ids': self.tokenizer.encode(input_text),
                'labels': self.tokenizer.encode(response)
            })
        
        return formatted_data
    
    def train_sft_model(self, sft_data, epochs=3, lr=1e-5):
        """Train supervised fine-tuned model"""
        
        # Training configuration
        training_config = {
            'learning_rate': lr,
            'num_epochs': epochs,
            'batch_size': 16,
            'gradient_accumulation_steps': 4,
            'warmup_steps': 100,
            'logging_steps': 10
        }
        
        # In production, use actual training loop
        print(f"Training SFT model with {len(sft_data)} demonstrations")
        print(f"Config: {training_config}")
        
        return self.model  # Return fine-tuned model
```

#### Stage 2: Reward Model Training

Train a reward model to predict human preferences:

```python
import torch
import torch.nn as nn

class RewardModel(nn.Module):
    """Reward model that scores model outputs based on human preferences"""
    
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)  # Single reward score
        )
    
    def forward(self, input_ids, attention_mask=None):
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        
        # Use last token representation for reward
        last_hidden_state = outputs.last_hidden_state
        sequence_lengths = attention_mask.sum(dim=1) - 1
        
        # Get last token embeddings
        batch_size = last_hidden_state.size(0)
        last_tokens = last_hidden_state[range(batch_size), sequence_lengths]
        
        # Compute reward score
        reward = self.reward_head(last_tokens)
        return reward

class RewardModelTrainer:
    def __init__(self, reward_model, tokenizer):
        self.reward_model = reward_model
        self.tokenizer = tokenizer
        
    def prepare_preference_data(self, preference_pairs):
        """Prepare human preference data for reward model training"""
        
        training_data = []
        
        for pair in preference_pairs:
            prompt = pair['prompt']
            chosen = pair['chosen']
            rejected = pair['rejected']
            
            # Tokenize chosen and rejected responses
            chosen_text = f"{prompt}\n{chosen}"
            rejected_text = f"{prompt}\n{rejected}"
            
            chosen_tokens = self.tokenizer.encode(chosen_text, return_tensors='pt')
            rejected_tokens = self.tokenizer.encode(rejected_text, return_tensors='pt')
            
            training_data.append({
                'chosen_input_ids': chosen_tokens,
                'rejected_input_ids': rejected_tokens,
                'prompt': prompt
            })
        
        return training_data
    
    def compute_reward_loss(self, chosen_rewards, rejected_rewards, margin=0.5):
        """Compute ranking loss for reward model"""
        
        # Preference loss: chosen should have higher reward than rejected
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards - margin))
        return loss.mean()
    
    def train_reward_model(self, preference_data, epochs=3):
        """Train reward model on human preference data"""
        
        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=1e-5)
        
        for epoch in range(epochs):
            total_loss = 0
            
            for batch in preference_data:
                # Get rewards for chosen and rejected responses
                chosen_rewards = self.reward_model(batch['chosen_input_ids'])
                rejected_rewards = self.reward_model(batch['rejected_input_ids'])
                
                # Compute loss
                loss = self.compute_reward_loss(chosen_rewards, rejected_rewards)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(preference_data)
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        return self.reward_model
```

#### Stage 3: Reinforcement Learning with PPO

Use Proximal Policy Optimization (PPO) to optimize the model against the reward model:

```python
class PPOTrainer:
    """PPO trainer for RLHF alignment"""
    
    def __init__(self, policy_model, reward_model, ref_model, tokenizer):
        self.policy_model = policy_model  # Model being optimized
        self.reward_model = reward_model  # Trained reward model
        self.ref_model = ref_model       # Reference model (SFT model)
        self.tokenizer = tokenizer
        
        # PPO hyperparameters
        self.kl_coeff = 0.1              # KL divergence coefficient
        self.clip_range = 0.2            # PPO clipping range
        self.value_coeff = 0.5           # Value function coefficient
        self.entropy_coeff = 0.01        # Entropy bonus coefficient
        
    def compute_kl_penalty(self, policy_logprobs, ref_logprobs):
        """Compute KL divergence penalty to prevent drift from reference model"""
        kl_div = policy_logprobs - ref_logprobs
        return self.kl_coeff * kl_div.mean()
    
    def compute_ppo_loss(self, old_logprobs, new_logprobs, advantages, returns, values):
        """Compute PPO loss with clipping"""
        
        # Probability ratio
        ratio = torch.exp(new_logprobs - old_logprobs)
        
        # Clipped surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value function loss
        value_loss = nn.MSELoss()(values, returns)
        
        # Entropy bonus
        entropy = -(new_logprobs * torch.exp(new_logprobs)).mean()
        
        # Total loss
        total_loss = (policy_loss + 
                     self.value_coeff * value_loss - 
                     self.entropy_coeff * entropy)
        
        return total_loss, policy_loss, value_loss, entropy
    
    def generate_and_evaluate(self, prompts, max_length=256):
        """Generate responses and evaluate with reward model"""
        
        responses = []
        rewards = []
        
        for prompt in prompts:
            # Generate response
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                output = self.policy_model.generate(
                    input_ids,
                    max_length=max_length,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            responses.append(response)
            
            # Get reward
            with torch.no_grad():
                reward = self.reward_model(output)
                rewards.append(reward.item())
        
        return responses, rewards
    
    def train_ppo_step(self, prompts, epochs=4):
        """Single PPO training step"""
        
        # Generate responses and get rewards
        responses, rewards = self.generate_and_evaluate(prompts)
        
        # Convert to tensors
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Compute advantages (simplified - in practice, use GAE)
        advantages = reward_tensor - reward_tensor.mean()
        returns = reward_tensor
        
        # PPO training loop
        for epoch in range(epochs):
            # Get current policy logprobs
            # ... (implementation details)
            pass
        
        return {
            'mean_reward': reward_tensor.mean().item(),
            'reward_std': reward_tensor.std().item(),
            'responses': responses[:3]  # Sample responses
        }
```

### Direct Preference Optimization (DPO)

DPO is a simpler alternative to RLHF that directly optimizes the policy using preference data without requiring a separate reward model.

#### DPO Mathematical Foundation

DPO reparameterizes the reward function in terms of the optimal policy:

```
r(x, y) = β log π*(y|x) - β log π_ref(y|x) + β log Z(x)
```

Where:
- π*(y|x) is the optimal policy
- π_ref(y|x) is the reference policy  
- β is the temperature parameter
- Z(x) is the partition function

#### DPO Implementation

```python
class DPOTrainer:
    """Direct Preference Optimization trainer"""
    
    def __init__(self, model, ref_model, tokenizer, beta=0.1):
        self.model = model           # Policy model being optimized
        self.ref_model = ref_model   # Reference model (frozen)
        self.tokenizer = tokenizer
        self.beta = beta             # Temperature parameter
        
        # Freeze reference model
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def compute_dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
                        reference_chosen_logps, reference_rejected_logps):
        """Compute DPO loss"""
        
        # Compute log ratios
        policy_logratios = policy_chosen_logps - policy_rejected_logps
        reference_logratios = reference_chosen_logps - reference_rejected_logps
        
        # DPO loss
        logits = self.beta * (policy_logratios - reference_logratios)
        loss = -torch.log(torch.sigmoid(logits)).mean()
        
        # Compute accuracy (how often chosen > rejected)
        accuracy = (logits > 0).float().mean()
        
        return loss, accuracy
    
    def get_batch_logps(self, model, input_ids, labels):
        """Get log probabilities for a batch"""
        
        outputs = model(input_ids, labels=labels)
        logits = outputs.logits
        
        # Shift logits and labels for next token prediction
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Compute log probabilities
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        
        # Gather log probabilities for actual tokens
        gathered_log_probs = torch.gather(
            log_probs, 
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask padding tokens
        mask = (shift_labels != -100).float()
        masked_log_probs = gathered_log_probs * mask
        
        # Sum log probabilities for each sequence
        sequence_log_probs = masked_log_probs.sum(dim=-1)
        
        return sequence_log_probs
    
    def train_step(self, batch):
        """Single DPO training step"""
        
        chosen_input_ids = batch['chosen_input_ids']
        chosen_labels = batch['chosen_labels']
        rejected_input_ids = batch['rejected_input_ids']
        rejected_labels = batch['rejected_labels']
        
        # Get log probabilities from policy model
        policy_chosen_logps = self.get_batch_logps(
            self.model, chosen_input_ids, chosen_labels
        )
        policy_rejected_logps = self.get_batch_logps(
            self.model, rejected_input_ids, rejected_labels
        )
        
        # Get log probabilities from reference model
        with torch.no_grad():
            reference_chosen_logps = self.get_batch_logps(
                self.ref_model, chosen_input_ids, chosen_labels
            )
            reference_rejected_logps = self.get_batch_logps(
                self.ref_model, rejected_input_ids, rejected_labels
            )
        
        # Compute DPO loss
        loss, accuracy = self.compute_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'policy_chosen_logps': policy_chosen_logps.mean(),
            'policy_rejected_logps': policy_rejected_logps.mean()
        }
    
    def train(self, dataloader, epochs=3, lr=1e-6):
        """Train DPO model"""
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            total_accuracy = 0
            
            for batch_idx, batch in enumerate(dataloader):
                # Training step
                metrics = self.train_step(batch)
                loss = metrics['loss']
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                total_accuracy += metrics['accuracy'].item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}, "
                          f"Loss: {loss.item():.4f}, "
                          f"Accuracy: {metrics['accuracy'].item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            avg_accuracy = total_accuracy / len(dataloader)
            
            print(f"Epoch {epoch+1} Summary - "
                  f"Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        return self.model
```

### Constitutional AI and Advanced Alignment

Constitutional AI extends RLHF by using AI systems to provide feedback, enabling more scalable oversight:

```python
class ConstitutionalAI:
    """Constitutional AI implementation for scalable oversight"""
    
    def __init__(self, model, tokenizer, constitution):
        self.model = model
        self.tokenizer = tokenizer
        self.constitution = constitution  # List of principles/rules
        
    def critique_response(self, prompt, response):
        """Generate critique based on constitutional principles"""
        
        critique_prompt = f"""
Please critique the following response according to these principles:
{self.constitution}

Prompt: {prompt}
Response: {response}

Critique:"""
        
        # Generate critique
        critique_input = self.tokenizer.encode(critique_prompt, return_tensors='pt')
        
        with torch.no_grad():
            critique_output = self.model.generate(
                critique_input,
                max_length=critique_input.shape[1] + 200,
                temperature=0.7,
                do_sample=True
            )
        
        critique = self.tokenizer.decode(
            critique_output[0][critique_input.shape[1]:], 
            skip_special_tokens=True
        )
        
        return critique.strip()
    
    def revise_response(self, prompt, response, critique):
        """Revise response based on critique"""
        
        revision_prompt = f"""
Please revise the following response based on the critique:

Original Prompt: {prompt}
Original Response: {response}
Critique: {critique}

Revised Response:"""
        
        revision_input = self.tokenizer.encode(revision_prompt, return_tensors='pt')
        
        with torch.no_grad():
            revision_output = self.model.generate(
                revision_input,
                max_length=revision_input.shape[1] + 300,
                temperature=0.7,
                do_sample=True
            )
        
        revised_response = self.tokenizer.decode(
            revision_output[0][revision_input.shape[1]:], 
            skip_special_tokens=True
        )
        
        return revised_response.strip()
    
    def constitutional_training_step(self, prompts):
        """Single constitutional AI training step"""
        
        training_data = []
        
        for prompt in prompts:
            # Generate initial response
            initial_response = self.generate_response(prompt)
            
            # Critique the response
            critique = self.critique_response(prompt, initial_response)
            
            # Revise based on critique
            revised_response = self.revise_response(prompt, initial_response, critique)
            
            # Create preference pair (revised > initial)
            training_data.append({
                'prompt': prompt,
                'chosen': revised_response,
                'rejected': initial_response,
                'critique': critique
            })
        
        return training_data
```

### Human Preference Collection

Effective alignment requires high-quality human preference data:

```python
class PreferenceCollectionSystem:
    """System for collecting and managing human preferences"""
    
    def __init__(self):
        self.annotation_guidelines = {
            'helpfulness': 'Rate how well the response addresses the user\'s request',
            'harmlessness': 'Rate how safe and non-harmful the response is',
            'honesty': 'Rate how truthful and accurate the response is',
            'clarity': 'Rate how clear and well-structured the response is'
        }
        
        self.quality_checks = [
            'inter_annotator_agreement',
            'annotation_time_validation',
            'consistency_checks',
            'expert_review'
        ]
    
    def create_annotation_task(self, prompt, responses, criteria=None):
        """Create annotation task for human evaluators"""
        
        if criteria is None:
            criteria = list(self.annotation_guidelines.keys())
        
        task = {
            'task_id': self.generate_task_id(),
            'prompt': prompt,
            'responses': responses,
            'criteria': criteria,
            'guidelines': {k: self.annotation_guidelines[k] for k in criteria},
            'created_at': time.time(),
            'status': 'pending'
        }
        
        return task
    
    def validate_annotations(self, annotations):
        """Validate annotation quality"""
        
        validation_results = {
            'valid': True,
            'issues': [],
            'confidence_score': 1.0
        }
        
        # Check annotation time (too fast might indicate low quality)
        if annotations.get('annotation_time', 0) < 30:  # seconds
            validation_results['issues'].append('Annotation completed too quickly')
            validation_results['confidence_score'] *= 0.8
        
        # Check for consistent preferences
        preferences = annotations.get('preferences', [])
        if len(set(preferences)) == 1:  # All same preference
            validation_results['issues'].append('No preference variation')
            validation_results['confidence_score'] *= 0.9
        
        # Check for explanation quality
        explanations = annotations.get('explanations', [])
        if any(len(exp.split()) < 5 for exp in explanations):
            validation_results['issues'].append('Insufficient explanation detail')
            validation_results['confidence_score'] *= 0.7
        
        if validation_results['confidence_score'] < 0.7:
            validation_results['valid'] = False
        
        return validation_results
    
    def compute_inter_annotator_agreement(self, annotations_list):
        """Compute agreement between multiple annotators"""
        
        if len(annotations_list) < 2:
            return {'agreement': 1.0, 'method': 'single_annotator'}
        
        # Simple agreement calculation (in practice, use Cohen's kappa or similar)
        agreements = []
        
        for i in range(len(annotations_list)):
            for j in range(i + 1, len(annotations_list)):
                ann1 = annotations_list[i]['preferences']
                ann2 = annotations_list[j]['preferences']
                
                agreement = sum(1 for a, b in zip(ann1, ann2) if a == b) / len(ann1)
                agreements.append(agreement)
        
        avg_agreement = sum(agreements) / len(agreements)
        
        return {
            'agreement': avg_agreement,
            'method': 'pairwise_agreement',
            'num_pairs': len(agreements)
        }
```

### Production Deployment and Monitoring

```python
class AlignmentMonitor:
    """Monitor aligned models in production"""
    
    def __init__(self):
        self.safety_classifiers = {
            'toxicity': self.load_toxicity_classifier(),
            'bias': self.load_bias_classifier(),
            'factuality': self.load_factuality_classifier()
        }
        
        self.alignment_metrics = {
            'helpfulness_score': [],
            'harmlessness_score': [],
            'honesty_score': [],
            'user_satisfaction': []
        }
    
    def load_toxicity_classifier(self):
        """Load toxicity detection model"""
        # In production, load actual classifier
        return lambda text: {'toxic': False, 'score': 0.1}
    
    def load_bias_classifier(self):
        """Load bias detection model"""
        return lambda text: {'biased': False, 'score': 0.05}
    
    def load_factuality_classifier(self):
        """Load factuality checking model"""
        return lambda text: {'factual': True, 'confidence': 0.9}
    
    def evaluate_response_safety(self, response):
        """Evaluate response safety across multiple dimensions"""
        
        safety_results = {}
        
        for classifier_name, classifier in self.safety_classifiers.items():
            result = classifier(response)
            safety_results[classifier_name] = result
        
        # Compute overall safety score
        safety_score = 1.0
        if safety_results['toxicity']['toxic']:
            safety_score *= 0.1
        if safety_results['bias']['biased']:
            safety_score *= 0.5
        if not safety_results['factuality']['factual']:
            safety_score *= 0.7
        
        return {
            'safety_score': safety_score,
            'details': safety_results,
            'safe': safety_score > 0.8
        }
    
    def monitor_alignment_drift(self, recent_responses, baseline_metrics):
        """Monitor for alignment drift over time"""
        
        current_metrics = self.compute_alignment_metrics(recent_responses)
        
        drift_analysis = {}
        
        for metric, baseline_value in baseline_metrics.items():
            current_value = current_metrics.get(metric, 0)
            drift = abs(current_value - baseline_value) / baseline_value
            
            drift_analysis[metric] = {
                'baseline': baseline_value,
                'current': current_value,
                'drift_percentage': drift * 100,
                'significant_drift': drift > 0.1  # 10% threshold
            }
        
        return drift_analysis
    
    def compute_alignment_metrics(self, responses):
        """Compute alignment metrics for a set of responses"""
        
        metrics = {
            'avg_safety_score': 0,
            'toxicity_rate': 0,
            'bias_rate': 0,
            'factuality_rate': 0
        }
        
        if not responses:
            return metrics
        
        safety_scores = []
        toxicity_count = 0
        bias_count = 0
        factual_count = 0
        
        for response in responses:
            safety_eval = self.evaluate_response_safety(response)
            safety_scores.append(safety_eval['safety_score'])
            
            if safety_eval['details']['toxicity']['toxic']:
                toxicity_count += 1
            if safety_eval['details']['bias']['biased']:
                bias_count += 1
            if safety_eval['details']['factuality']['factual']:
                factual_count += 1
        
        metrics['avg_safety_score'] = sum(safety_scores) / len(safety_scores)
        metrics['toxicity_rate'] = toxicity_count / len(responses)
        metrics['bias_rate'] = bias_count / len(responses)
        metrics['factuality_rate'] = factual_count / len(responses)
        
        return metrics
```

### Best Practices for Production Alignment

#### 1. Data Quality and Diversity
- **Diverse Annotators**: Include annotators from different backgrounds and perspectives
- **Clear Guidelines**: Provide comprehensive annotation guidelines with examples
- **Quality Control**: Implement multiple validation layers and inter-annotator agreement checks
- **Iterative Improvement**: Continuously refine guidelines based on edge cases

#### 2. Model Training and Evaluation
- **Staged Training**: Use SFT → Reward Model → RL/DPO pipeline
- **Regularization**: Prevent overfitting to specific preference patterns
- **Evaluation Diversity**: Test on diverse prompts and edge cases
- **Safety Testing**: Comprehensive red-teaming and adversarial evaluation

#### 3. Deployment and Monitoring
- **Gradual Rollout**: Deploy aligned models gradually with careful monitoring
- **Real-time Safety**: Implement real-time safety filters and monitoring
- **Feedback Loops**: Collect user feedback to identify alignment issues
- **Drift Detection**: Monitor for alignment drift over time

#### 4. Ethical Considerations
- **Value Pluralism**: Acknowledge that human values are diverse and sometimes conflicting
- **Transparency**: Be transparent about alignment methods and limitations
- **Bias Mitigation**: Actively work to identify and mitigate biases
- **Stakeholder Involvement**: Include diverse stakeholders in alignment decisions

### RLHF vs DPO Comparison

| Aspect | RLHF | DPO |
|--------|------|-----|
| **Complexity** | High (3-stage pipeline) | Lower (direct optimization) |
| **Training Stability** | Can be unstable (RL) | More stable |
| **Computational Cost** | Higher (separate reward model) | Lower |
| **Interpretability** | Reward model provides insights | Less interpretable |
| **Performance** | Often slightly better | Competitive performance |
| **Implementation** | More complex | Simpler to implement |

### Why Alignment Matters

Alignment is crucial because:

- **Safety**: Prevents harmful or dangerous model outputs
- **Trust**: Builds user confidence in AI systems
- **Reliability**: Ensures consistent behavior across contexts
- **Ethics**: Aligns AI behavior with human values and social norms
- **Regulation**: Meets emerging regulatory requirements for AI safety
- **Business Value**: Reduces risks and improves user satisfaction

## Exercise (25 minutes)
Complete the hands-on exercises in `exercise.py` to practice RLHF and DPO implementation.

## Resources
- [InstructGPT Paper: "Training language models to follow instructions with human feedback"](https://arxiv.org/abs/2203.02155)
- [DPO Paper: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"](https://arxiv.org/abs/2305.18290)
- [Constitutional AI Paper: "Constitutional AI: Harmlessness from AI Feedback"](https://arxiv.org/abs/2212.08073)
- [Anthropic's RLHF Blog Post](https://www.anthropic.com/index/core-views-on-ai-safety)
- [OpenAI Alignment Research](https://openai.com/research/alignment)

## Next Steps
- Complete the exercises to practice alignment techniques
- Take the quiz to test your understanding
- Experiment with preference collection and reward modeling
- Move to Day 50: Quantization - Model Compression & Optimization