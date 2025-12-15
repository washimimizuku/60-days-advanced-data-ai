# Day 44: LLM Training Stages - Pre-training, Fine-tuning & Alignment

## 📖 Learning Objectives
By the end of today, you will be able to:
- **Master the three-stage LLM training pipeline** from pre-training through alignment
- **Implement fine-tuning strategies** including supervised fine-tuning and parameter-efficient methods
- **Understand alignment techniques** such as RLHF, Constitutional AI, and preference learning
- **Design training infrastructure** for large-scale model training and distributed computing
- **Apply production best practices** for model training, evaluation, and deployment

**Estimated Time**: 1 hour  
**Difficulty Level**: Advanced ⭐⭐⭐⭐⭐

---

## 🔍 Building on Yesterday's Foundation

Yesterday, you mastered tokenization strategies that convert raw text into the numerical representations that models can process. Today, we explore how these tokens are used to train large language models through multiple sophisticated stages.

**Key Concepts from Previous Days**:
- Transformer architecture provides the model foundation
- Attention mechanisms enable long-range dependencies
- Tokenization converts text to model inputs
- Each training stage builds upon the previous one

**Today's Focus**: We'll dive deep into the multi-stage training process that creates powerful language models, from initial pre-training on massive corpora to fine-tuning for specific tasks and aligning with human preferences.

---

## 🎯 The Three-Stage LLM Training Pipeline

### **Stage 1: Pre-training (Foundation Learning)**
**Objective**: Learn general language understanding and generation capabilities

**Stage 2: Fine-tuning (Task Specialization)**  
**Objective**: Adapt the model for specific tasks and domains

**Stage 3: Alignment (Human Preference Learning)**
**Objective**: Align model behavior with human values and preferences

---

## 🏗️ Stage 1: Pre-training - Building the Foundation

### **1. Pre-training Fundamentals**

#### What is Pre-training?
**Definition**: Unsupervised learning on massive text corpora to develop general language capabilities.

**Core Objective**: Learn to predict the next token in a sequence using self-supervised learning.

**Mathematical Foundation**:
```
L_pretrain = -∑(log P(x_t | x_1, x_2, ..., x_{t-1}; θ))
```

Where:
- `x_t` is the token at position t
- `θ` represents model parameters
- The model learns to maximize likelihood of the training data

#### Pre-training Data Requirements

**Scale**: Modern LLMs are trained on trillions of tokens
- **GPT-3**: ~300B tokens
- **PaLM**: ~780B tokens  
- **GPT-4**: Estimated 1-10T tokens
- **Llama 2**: ~2T tokens

**Data Sources**:
- **Web Crawls**: Common Crawl, web pages, forums
- **Books**: Project Gutenberg, published literature
- **Academic Papers**: arXiv, research publications
- **Code Repositories**: GitHub, programming languages
- **Reference Materials**: Wikipedia, encyclopedias

**Data Quality Considerations**:
```python
# Data filtering pipeline
def filter_pretraining_data(text):
    # Language detection
    if not is_target_language(text):
        return False
    
    # Quality filters
    if len(text) < MIN_LENGTH or len(text) > MAX_LENGTH:
        return False
    
    # Deduplication
    if is_duplicate(text):
        return False
    
    # Content filtering
    if contains_harmful_content(text):
        return False
    
    # Perplexity filtering (remove low-quality text)
    if calculate_perplexity(text) > PERPLEXITY_THRESHOLD:
        return False
    
    return True
```

### **2. Pre-training Objectives**

#### Causal Language Modeling (CLM)
**Used by**: GPT family, LLaMA, PaLM

**Objective**: Predict next token given previous context
```
P(x_t | x_1, x_2, ..., x_{t-1})
```

**Training Process**:
1. **Input**: Sequence of tokens [x₁, x₂, ..., xₙ]
2. **Prediction**: Model predicts x₂ given x₁, x₃ given x₁,x₂, etc.
3. **Loss**: Cross-entropy between predictions and actual tokens
4. **Optimization**: Gradient descent to minimize loss

#### Masked Language Modeling (MLM)
**Used by**: BERT, RoBERTa, DeBERTa

**Objective**: Predict masked tokens using bidirectional context
```
P(x_masked | x_context)
```

**Training Process**:
1. **Masking**: Randomly mask 15% of tokens
2. **Prediction**: Predict masked tokens using surrounding context
3. **Bidirectional**: Can use both left and right context
4. **Applications**: Better for understanding tasks

#### Prefix Language Modeling
**Used by**: T5, UL2, PaLM-2

**Objective**: Predict suffix given prefix
```
P(suffix | prefix)
```

**Advantages**:
- **Flexible**: Can handle various input-output formats
- **Unified**: Same model for different tasks
- **Efficient**: Better sample efficiency than CLM

### **3. Pre-training Infrastructure**

#### Distributed Training Strategies

**Data Parallelism**:
```python
# Distribute batches across multiple GPUs
def data_parallel_training(model, data_loader, num_gpus):
    # Split batch across GPUs
    batch_size_per_gpu = batch_size // num_gpus
    
    for batch in data_loader:
        # Distribute batch
        sub_batches = split_batch(batch, num_gpus)
        
        # Forward pass on each GPU
        losses = []
        for gpu_id, sub_batch in enumerate(sub_batches):
            with torch.cuda.device(gpu_id):
                loss = model(sub_batch)
                losses.append(loss)
        
        # Aggregate gradients
        total_loss = sum(losses) / len(losses)
        total_loss.backward()
        
        # Synchronize gradients across GPUs
        synchronize_gradients(model)
        optimizer.step()
```

**Model Parallelism**:
```python
# Split model layers across GPUs
class ModelParallelTransformer:
    def __init__(self, config, num_gpus):
        self.layers_per_gpu = config.num_layers // num_gpus
        
        # Distribute layers across GPUs
        for gpu_id in range(num_gpus):
            start_layer = gpu_id * self.layers_per_gpu
            end_layer = (gpu_id + 1) * self.layers_per_gpu
            
            with torch.cuda.device(gpu_id):
                self.layer_groups[gpu_id] = TransformerLayers(
                    start_layer, end_layer, config
                )
    
    def forward(self, x):
        # Sequential execution across GPUs
        for gpu_id in range(self.num_gpus):
            with torch.cuda.device(gpu_id):
                x = self.layer_groups[gpu_id](x)
                if gpu_id < self.num_gpus - 1:
                    x = x.to(f'cuda:{gpu_id + 1}')
        return x
```

**Pipeline Parallelism**:
```python
# Pipeline different stages of the model
class PipelineParallelTraining:
    def __init__(self, model_stages, num_microbatches):
        self.stages = model_stages
        self.num_microbatches = num_microbatches
    
    def forward_backward_pass(self, batch):
        # Split batch into microbatches
        microbatches = split_into_microbatches(batch, self.num_microbatches)
        
        # Pipeline execution
        for step in range(len(self.stages) + self.num_microbatches - 1):
            for stage_id, stage in enumerate(self.stages):
                microbatch_id = step - stage_id
                
                if 0 <= microbatch_id < self.num_microbatches:
                    if stage_id == 0:
                        # Forward pass
                        output = stage(microbatches[microbatch_id])
                    else:
                        # Backward pass
                        stage.backward(output)
```

#### Memory Optimization Techniques

**Gradient Checkpointing**:
```python
def gradient_checkpointing_forward(model, x):
    """Trade computation for memory by recomputing activations"""
    
    # Store only checkpoint activations
    checkpoints = []
    
    for i, layer in enumerate(model.layers):
        if i % CHECKPOINT_INTERVAL == 0:
            x = checkpoint(layer, x)  # Store activation
            checkpoints.append(x)
        else:
            x = layer(x)  # Don't store activation
    
    return x, checkpoints
```

**Mixed Precision Training**:
```python
# Use FP16 for forward pass, FP32 for gradients
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    # Forward pass in FP16
    outputs = model(inputs)
    loss = criterion(outputs, targets)

# Backward pass with gradient scaling
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**ZeRO (Zero Redundancy Optimizer)**:
```python
# Partition optimizer states, gradients, and parameters
class ZeROOptimizer:
    def __init__(self, model, optimizer, partition_level=3):
        self.partition_level = partition_level
        
        if partition_level >= 1:
            # Partition optimizer states
            self.partition_optimizer_states()
        
        if partition_level >= 2:
            # Partition gradients
            self.partition_gradients()
        
        if partition_level >= 3:
            # Partition model parameters
            self.partition_parameters()
    
    def partition_parameters(self):
        """Partition model parameters across devices"""
        for param in self.model.parameters():
            # Keep only local partition
            param.data = param.data[self.local_start:self.local_end]
```

---

## 🎯 Stage 2: Fine-tuning - Task Specialization

### **1. Supervised Fine-tuning (SFT)**

#### Full Fine-tuning
**Approach**: Update all model parameters for specific tasks.

**Process**:
1. **Initialize**: Start with pre-trained model weights
2. **Task Data**: Prepare task-specific training data
3. **Learning Rate**: Use lower learning rate than pre-training
4. **Training**: Fine-tune on task data with supervised learning

**Implementation**:
```python
def supervised_fine_tuning(pretrained_model, task_data, config):
    # Load pre-trained weights
    model = load_pretrained_model(pretrained_model)
    
    # Prepare task-specific data
    train_loader = prepare_task_data(task_data, config.batch_size)
    
    # Use lower learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate * 0.1,  # 10x lower than pre-training
        weight_decay=config.weight_decay
    )
    
    # Fine-tuning loop
    for epoch in range(config.num_epochs):
        for batch in train_loader:
            # Forward pass
            outputs = model(batch['input_ids'], labels=batch['labels'])
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
```

#### Parameter-Efficient Fine-tuning (PEFT)

**LoRA (Low-Rank Adaptation)**:
```python
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=16, alpha=32):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Add low-rank adaptation matrices
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # Original computation
        original_output = self.original_layer(x)
        
        # LoRA adaptation
        lora_output = self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)
        
        return original_output + lora_output
```

**Adapter Layers**:
```python
class AdapterLayer(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Residual connection around adapter
        adapter_output = self.up_project(
            self.dropout(self.activation(self.down_project(x)))
        )
        return x + adapter_output
```

**Prefix Tuning**:
```python
class PrefixTuning(nn.Module):
    def __init__(self, config, prefix_length=10):
        super().__init__()
        self.prefix_length = prefix_length
        self.num_layers = config.num_layers
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        
        # Learnable prefix parameters
        self.prefix_embeddings = nn.Parameter(
            torch.randn(prefix_length, config.hidden_size)
        )
        
        # Transform to key-value pairs for each layer
        self.prefix_mlp = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 2 * self.num_layers * config.hidden_size)
        )
    
    def get_prefix_states(self, batch_size):
        # Generate prefix key-value pairs
        prefix_states = self.prefix_mlp(self.prefix_embeddings)
        prefix_states = prefix_states.view(
            self.prefix_length, 2 * self.num_layers, self.num_heads, self.head_dim
        )
        
        # Expand for batch
        prefix_states = prefix_states.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        
        return prefix_states
```

### **2. Instruction Tuning**

#### Instruction Following
**Objective**: Train models to follow natural language instructions.

**Data Format**:
```json
{
  "instruction": "Translate the following English text to French:",
  "input": "Hello, how are you today?",
  "output": "Bonjour, comment allez-vous aujourd'hui?"
}
```

**Training Process**:
```python
def instruction_tuning_loss(model, batch):
    # Format instruction + input + output
    formatted_inputs = []
    for item in batch:
        formatted_input = f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}"
        formatted_inputs.append(formatted_input)
    
    # Tokenize
    tokenized = tokenizer(formatted_inputs, return_tensors='pt', padding=True)
    
    # Only compute loss on output tokens
    labels = tokenized['input_ids'].clone()
    
    # Mask instruction and input tokens
    for i, item in enumerate(batch):
        instruction_length = len(tokenizer(f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: ")['input_ids'])
        labels[i, :instruction_length] = -100  # Ignore in loss computation
    
    # Forward pass
    outputs = model(tokenized['input_ids'], labels=labels)
    return outputs.loss
```

#### Multi-task Fine-tuning
**Approach**: Train on multiple tasks simultaneously to improve generalization.

```python
class MultiTaskTrainer:
    def __init__(self, model, task_datasets, task_weights=None):
        self.model = model
        self.task_datasets = task_datasets
        self.task_weights = task_weights or {task: 1.0 for task in task_datasets}
    
    def train_step(self):
        total_loss = 0
        
        for task_name, dataset in self.task_datasets.items():
            # Sample batch from task
            batch = dataset.sample_batch()
            
            # Compute task-specific loss
            task_loss = self.compute_task_loss(task_name, batch)
            
            # Weight the loss
            weighted_loss = task_loss * self.task_weights[task_name]
            total_loss += weighted_loss
        
        # Backward pass on combined loss
        total_loss.backward()
        return total_loss
```

---

## 🎯 Stage 3: Alignment - Human Preference Learning

### **1. Reinforcement Learning from Human Feedback (RLHF)**

#### The RLHF Pipeline

**Step 1: Collect Human Preferences**
```python
def collect_preference_data(model, prompts):
    preference_data = []
    
    for prompt in prompts:
        # Generate multiple responses
        responses = model.generate(prompt, num_return_sequences=4)
        
        # Human annotators rank responses
        rankings = get_human_rankings(prompt, responses)
        
        # Convert to pairwise preferences
        for i in range(len(responses)):
            for j in range(i + 1, len(responses)):
                if rankings[i] > rankings[j]:
                    preference_data.append({
                        'prompt': prompt,
                        'chosen': responses[i],
                        'rejected': responses[j]
                    })
    
    return preference_data
```

**Step 2: Train Reward Model**
```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        # Get last hidden state
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        
        # Get reward for last token
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden_state[range(len(sequence_lengths)), sequence_lengths]
        
        reward = self.reward_head(last_token_hidden)
        return reward

def train_reward_model(reward_model, preference_data):
    for batch in preference_data:
        # Get rewards for chosen and rejected responses
        chosen_rewards = reward_model(batch['chosen_ids'], batch['chosen_mask'])
        rejected_rewards = reward_model(batch['rejected_ids'], batch['rejected_mask'])
        
        # Preference loss (Bradley-Terry model)
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)).mean()
        
        loss.backward()
        optimizer.step()
```

**Step 3: PPO Training**
```python
class PPOTrainer:
    def __init__(self, policy_model, reward_model, ref_model):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.ref_model = ref_model
        self.kl_coeff = 0.1
    
    def compute_ppo_loss(self, prompts, responses, old_log_probs):
        # Get current policy log probabilities
        new_log_probs = self.policy_model.get_log_probs(prompts, responses)
        
        # Get reference model log probabilities
        ref_log_probs = self.ref_model.get_log_probs(prompts, responses)
        
        # Get rewards from reward model
        rewards = self.reward_model(prompts + responses)
        
        # KL penalty
        kl_penalty = self.kl_coeff * (new_log_probs - ref_log_probs)
        
        # Compute advantages
        advantages = rewards - kl_penalty
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
        
        policy_loss = -torch.min(
            ratio * advantages,
            clipped_ratio * advantages
        ).mean()
        
        return policy_loss
```

### **2. Constitutional AI (CAI)**

#### Self-Critique and Revision
```python
class ConstitutionalAI:
    def __init__(self, model, constitution):
        self.model = model
        self.constitution = constitution  # List of principles
    
    def constitutional_training_step(self, prompt):
        # Generate initial response
        initial_response = self.model.generate(prompt)
        
        # Self-critique against each principle
        critiques = []
        for principle in self.constitution:
            critique_prompt = f"""
            Principle: {principle}
            
            Human: {prompt}
            Assistant: {initial_response}
            
            Does the assistant's response violate the principle? If so, how?
            """
            
            critique = self.model.generate(critique_prompt)
            critiques.append(critique)
        
        # Generate revised response
        revision_prompt = f"""
        Original response: {initial_response}
        Critiques: {' '.join(critiques)}
        
        Please provide a revised response that addresses the critiques:
        """
        
        revised_response = self.model.generate(revision_prompt)
        
        return {
            'original': initial_response,
            'critiques': critiques,
            'revised': revised_response
        }
```

### **3. Direct Preference Optimization (DPO)**

#### Simplified Alignment Training
```python
def dpo_loss(policy_model, ref_model, batch, beta=0.1):
    """
    Direct Preference Optimization loss
    Simpler alternative to RLHF that doesn't require reward model
    """
    
    # Get log probabilities from policy and reference models
    policy_chosen_logps = policy_model.get_log_probs(batch['chosen'])
    policy_rejected_logps = policy_model.get_log_probs(batch['rejected'])
    
    ref_chosen_logps = ref_model.get_log_probs(batch['chosen'])
    ref_rejected_logps = ref_model.get_log_probs(batch['rejected'])
    
    # Compute preference probabilities
    policy_ratio_chosen = policy_chosen_logps - ref_chosen_logps
    policy_ratio_rejected = policy_rejected_logps - ref_rejected_logps
    
    # DPO loss
    loss = -torch.log(torch.sigmoid(
        beta * (policy_ratio_chosen - policy_ratio_rejected)
    )).mean()
    
    return loss
```

---

## 🏭 Production Training Infrastructure

### **1. Training Pipeline Architecture**

#### Distributed Training Setup
```python
class LLMTrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.setup_distributed_training()
        self.setup_data_pipeline()
        self.setup_model_and_optimizer()
        self.setup_monitoring()
    
    def setup_distributed_training(self):
        # Initialize distributed process group
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.config.world_size,
            rank=self.config.rank
        )
        
        # Set device
        torch.cuda.set_device(self.config.local_rank)
        self.device = torch.device(f'cuda:{self.config.local_rank}')
    
    def setup_data_pipeline(self):
        # Distributed data loading
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(
            self.train_dataset,
            num_replicas=self.config.world_size,
            rank=self.config.rank
        )
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=self.train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
```

#### Checkpointing and Recovery
```python
class TrainingCheckpoint:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
    
    def save_checkpoint(self, model, optimizer, scheduler, step, loss):
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'step': step,
            'loss': loss,
            'timestamp': time.time()
        }
        
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_step_{step}.pt'
        )
        
        torch.save(checkpoint, checkpoint_path)
        
        # Keep only recent checkpoints
        self.cleanup_old_checkpoints()
    
    def load_checkpoint(self, model, optimizer, scheduler):
        # Find latest checkpoint
        checkpoint_files = glob.glob(
            os.path.join(self.checkpoint_dir, 'checkpoint_step_*.pt')
        )
        
        if not checkpoint_files:
            return 0, float('inf')
        
        latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
        checkpoint = torch.load(latest_checkpoint)
        
        # Restore states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return checkpoint['step'], checkpoint['loss']
```

### **2. Training Monitoring and Evaluation**

#### Comprehensive Metrics Tracking
```python
class TrainingMonitor:
    def __init__(self, config):
        self.config = config
        self.metrics = defaultdict(list)
        self.setup_wandb()
    
    def log_training_metrics(self, step, loss, learning_rate, grad_norm):
        # Core training metrics
        self.metrics['train_loss'].append(loss)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['grad_norm'].append(grad_norm)
        
        # Log to wandb
        wandb.log({
            'train/loss': loss,
            'train/learning_rate': learning_rate,
            'train/grad_norm': grad_norm,
            'train/step': step
        })
    
    def evaluate_model(self, model, eval_datasets):
        model.eval()
        eval_results = {}
        
        with torch.no_grad():
            for dataset_name, dataset in eval_datasets.items():
                total_loss = 0
                num_batches = 0
                
                for batch in dataset:
                    outputs = model(**batch)
                    total_loss += outputs.loss.item()
                    num_batches += 1
                
                avg_loss = total_loss / num_batches
                perplexity = math.exp(avg_loss)
                
                eval_results[dataset_name] = {
                    'loss': avg_loss,
                    'perplexity': perplexity
                }
        
        model.train()
        return eval_results
```

#### Model Quality Assessment
```python
def evaluate_model_quality(model, tokenizer, test_prompts):
    """Comprehensive model quality evaluation"""
    
    results = {
        'generation_quality': [],
        'instruction_following': [],
        'safety_scores': [],
        'factual_accuracy': []
    }
    
    for prompt in test_prompts:
        # Generate response
        response = model.generate(
            prompt, 
            max_length=512,
            temperature=0.7,
            do_sample=True
        )
        
        # Evaluate different aspects
        results['generation_quality'].append(
            evaluate_generation_quality(prompt, response)
        )
        
        results['instruction_following'].append(
            evaluate_instruction_following(prompt, response)
        )
        
        results['safety_scores'].append(
            evaluate_safety(response)
        )
        
        results['factual_accuracy'].append(
            evaluate_factual_accuracy(prompt, response)
        )
    
    # Aggregate results
    return {
        metric: np.mean(scores) 
        for metric, scores in results.items()
    }
```

---

## 🔧 Advanced Training Techniques

### **1. Curriculum Learning**

#### Progressive Training Strategy
```python
class CurriculumLearning:
    def __init__(self, datasets, difficulty_scorer):
        self.datasets = datasets
        self.difficulty_scorer = difficulty_scorer
        self.current_difficulty = 0.0
    
    def get_curriculum_batch(self, step, total_steps):
        # Gradually increase difficulty
        progress = step / total_steps
        target_difficulty = progress * 1.0  # Scale from 0 to 1
        
        # Filter examples by difficulty
        suitable_examples = []
        for example in self.datasets:
            difficulty = self.difficulty_scorer(example)
            if difficulty <= target_difficulty + 0.1:  # Small tolerance
                suitable_examples.append(example)
        
        # Sample batch from suitable examples
        return random.sample(suitable_examples, self.batch_size)
```

### **2. Data Mixing Strategies**

#### Multi-source Data Blending
```python
class DataMixer:
    def __init__(self, data_sources, mixing_ratios):
        self.data_sources = data_sources
        self.mixing_ratios = mixing_ratios
        self.validate_ratios()
    
    def create_mixed_batch(self, batch_size):
        mixed_batch = []
        
        for source_name, ratio in self.mixing_ratios.items():
            source_batch_size = int(batch_size * ratio)
            source_data = self.data_sources[source_name]
            
            # Sample from this source
            source_samples = source_data.sample(source_batch_size)
            mixed_batch.extend(source_samples)
        
        # Shuffle the mixed batch
        random.shuffle(mixed_batch)
        return mixed_batch
```

### **3. Adaptive Training Strategies**

#### Dynamic Learning Rate Scheduling
```python
class AdaptiveLRScheduler:
    def __init__(self, optimizer, patience=5, factor=0.5):
        self.optimizer = optimizer
        self.patience = patience
        self.factor = factor
        self.best_loss = float('inf')
        self.wait_count = 0
    
    def step(self, current_loss):
        if current_loss < self.best_loss:
            self.best_loss = current_loss
            self.wait_count = 0
        else:
            self.wait_count += 1
            
            if self.wait_count >= self.patience:
                # Reduce learning rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= self.factor
                
                self.wait_count = 0
                print(f"Reduced learning rate to {param_group['lr']}")
```

---

## 📊 Training Stage Comparison

### **Performance Characteristics**

| Stage | Compute Requirements | Data Requirements | Training Time | Model Capabilities |
|-------|---------------------|-------------------|---------------|-------------------|
| **Pre-training** | Very High (1000s of GPUs) | Massive (TB of text) | Weeks/Months | General language understanding |
| **Fine-tuning** | Moderate (10s of GPUs) | Task-specific (GB) | Hours/Days | Task specialization |
| **Alignment** | Moderate (10s of GPUs) | Human preferences (MB) | Days/Weeks | Human-aligned behavior |

### **Cost Analysis**

```python
def estimate_training_costs(stage, model_size, data_size, hardware_config):
    """Estimate training costs for different stages"""
    
    costs = {
        'pre_training': {
            'compute_hours': model_size * data_size * 0.001,  # Rough estimate
            'gpu_cost_per_hour': 8.0,  # A100 cost
            'data_preparation': data_size * 0.1,
            'storage': data_size * 0.05
        },
        'fine_tuning': {
            'compute_hours': model_size * data_size * 0.0001,
            'gpu_cost_per_hour': 8.0,
            'data_preparation': data_size * 0.5,
            'storage': data_size * 0.02
        },
        'alignment': {
            'compute_hours': model_size * data_size * 0.0005,
            'gpu_cost_per_hour': 8.0,
            'human_annotation': data_size * 10.0,  # Human labeling cost
            'storage': data_size * 0.01
        }
    }
    
    stage_costs = costs[stage]
    total_cost = (
        stage_costs['compute_hours'] * stage_costs['gpu_cost_per_hour'] +
        stage_costs['data_preparation'] +
        stage_costs['storage'] +
        stage_costs.get('human_annotation', 0)
    )
    
    return total_cost
```

---

## 🎯 Key Takeaways

### Training Pipeline Understanding
1. **Three-Stage Process**: Pre-training builds foundation, fine-tuning adds specialization, alignment ensures safety
2. **Scale Requirements**: Each stage has different compute, data, and time requirements
3. **Progressive Capability**: Each stage builds upon the previous one's capabilities
4. **Cost Considerations**: Pre-training is most expensive, alignment requires human input
5. **Infrastructure Needs**: Distributed training essential for large models

### Technical Mastery
1. **Distributed Training**: Understand data, model, and pipeline parallelism strategies
2. **Memory Optimization**: Apply gradient checkpointing, mixed precision, and ZeRO
3. **Parameter Efficiency**: Use LoRA, adapters, and prefix tuning for efficient fine-tuning
4. **Alignment Methods**: Implement RLHF, Constitutional AI, and DPO for human alignment
5. **Monitoring**: Track comprehensive metrics throughout training process

### Production Readiness
1. **Checkpointing**: Implement robust checkpoint and recovery systems
2. **Evaluation**: Develop comprehensive model quality assessment frameworks
3. **Cost Management**: Optimize training efficiency and resource utilization
4. **Safety**: Ensure alignment and safety throughout the training process
5. **Scalability**: Design systems that can handle increasing model and data sizes

---

## 🚀 What's Next?

### Tomorrow: Day 45 - Prompt Engineering with DSPy
- **Advanced Prompting Frameworks**: Systematic prompt optimization and composition
- **DSPy Programming**: Declarative self-improving language programs
- **Prompt Optimization**: Automatic prompt tuning and evaluation
- **Production Prompting**: Scalable prompt management and deployment

### This Week's Journey
- **Day 46**: Prompt Security - Injection attacks and defense mechanisms
- **Day 47**: Project - Advanced Prompting System (Integration Project)
- **Day 48**: Fine-tuning Techniques - LoRA, QLoRA, and parameter-efficient methods

### Building Towards
By mastering LLM training stages, you're building the foundation for:
- **Understanding Model Behavior**: How training affects model capabilities and limitations
- **Optimizing Training Processes**: Efficient and effective model development
- **Implementing Safety Measures**: Ensuring models are aligned with human values
- **Scaling Training Systems**: Building infrastructure for large-scale model development

---

## 🎉 Ready to Master LLM Training?

Today's comprehensive exploration of LLM training stages will give you the expertise to understand, implement, and optimize the complete pipeline from raw text to aligned, capable language models. You'll gain insights into the sophisticated multi-stage process that creates the AI systems powering modern applications.

**Your journey from tokens to intelligent, aligned models starts now!** 🚀
