"""
Day 44: LLM Training Stages - Exercise

Business Scenario:
You're the ML Engineering Lead at a cutting-edge AI company building the next generation 
of language models. Your team needs to implement a complete LLM training pipeline that 
covers all three stages: pre-training, fine-tuning, and alignment. The system must be 
production-ready, scalable, and include comprehensive monitoring and evaluation.

Your task is to implement the training infrastructure, optimization techniques, and 
alignment methods needed to train state-of-the-art language models efficiently and safely.

Requirements:
1. Implement distributed training infrastructure for large-scale pre-training
2. Create parameter-efficient fine-tuning methods (LoRA, adapters, prefix tuning)
3. Build RLHF pipeline with reward model training and PPO optimization
4. Implement Constitutional AI and Direct Preference Optimization
5. Develop comprehensive training monitoring and evaluation frameworks

Success Criteria:
- Distributed training scales efficiently across multiple GPUs
- Parameter-efficient methods achieve comparable performance with <1% trainable parameters
- RLHF pipeline successfully aligns model behavior with human preferences
- Training monitoring provides actionable insights for optimization
- All implementations follow production best practices for safety and reliability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import math
import time
import json
import wandb
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import os
import random


@dataclass
class TrainingConfig:
    """Configuration for LLM training stages"""
    # Model configuration
    vocab_size: int = 50000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_seq_length: int = 1024
    
    # Training configuration
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    
    # Distributed training
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Fine-tuning specific
    lora_rank: int = 16
    lora_alpha: int = 32
    adapter_size: int = 64
    
    # RLHF configuration
    kl_coeff: float = 0.1
    clip_epsilon: float = 0.2
    ppo_epochs: int = 4


class SimpleTransformerLayer(nn.Module):
    """
    Simplified transformer layer for training exercises
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Implement multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_heads,
            dropout=0.1,
            batch_first=True
        )
        
        # Implement feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 4),
            nn.GELU(),
            nn.Linear(config.hidden_size * 4, config.hidden_size),
            nn.Dropout(0.1)
        )
        
        # Add layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size)
        self.norm2 = nn.LayerNorm(config.hidden_size)
        
        # TODO: Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through transformer layer
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            output: Transformed tensor [batch_size, seq_len, hidden_size]
        """
        # Self-attention with residual connection
        normed_x = self.norm1(x)
        attn_output, _ = self.attention(normed_x, normed_x, normed_x, attn_mask=mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        normed_x = self.norm2(x)
        ff_output = self.feed_forward(normed_x)
        x = x + self.dropout(ff_output)
        
        return x


class SimpleLLM(nn.Module):
    """
    Simplified language model for training exercises
    """
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        
        # Implement token embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        # Create transformer layers
        self.layers = nn.ModuleList([
            SimpleTransformerLayer(config) for _ in range(config.num_layers)
        ])
        
        # Add final layer norm and output projection
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.output_projection = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the language model
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            labels: Target labels for loss computation [batch_size, seq_len]
            
        Returns:
            outputs: Dictionary with logits and loss (if labels provided)
        """
        batch_size, seq_len = input_ids.shape
        
        # Create embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        x = token_embeds + position_embeds
        
        # Create causal mask for autoregressive generation
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x, causal_mask)
        
        # Apply final normalization and projection
        x = self.final_norm(x)
        logits = self.output_projection(x)
        
        outputs = {'logits': logits}
        
        # Compute loss if labels are provided
        if labels is not None:
            # Shift labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs['loss'] = loss
        
        return outputs


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation (LoRA) layer for parameter-efficient fine-tuning
    """
    
    def __init__(self, original_layer: nn.Linear, rank: int = 16, alpha: int = 32):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # Create low-rank adaptation matrices A and B
        self.lora_A = nn.Linear(original_layer.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original_layer.out_features, bias=False)
        
        # Initialize LoRA weights properly
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with LoRA adaptation
        
        Args:
            x: Input tensor
            
        Returns:
            output: Original output + LoRA adaptation
        """
        # Compute original output
        original_output = self.original_layer(x)
        
        # Compute LoRA adaptation
        lora_output = self.lora_B(self.lora_A(x)) * (self.alpha / self.rank)
        
        return original_output + lora_output


class AdapterLayer(nn.Module):
    """
    Adapter layer for parameter-efficient fine-tuning
    """
    
    def __init__(self, hidden_size: int, adapter_size: int = 64):
        super().__init__()
        
        # Implement adapter bottleneck architecture
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.activation = nn.ReLU()
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter layer
        
        Args:
            x: Input tensor [batch_size, seq_len, hidden_size]
            
        Returns:
            output: Input + adapter transformation
        """
        # Implement adapter forward pass
        adapter_output = self.up_project(
            self.dropout(self.activation(self.down_project(x)))
        )
        
        return x + adapter_output


class RewardModel(nn.Module):
    """
    Reward model for RLHF training
    """
    
    def __init__(self, base_model: SimpleLLM):
        super().__init__()
        self.base_model = base_model
        
        # Add reward head
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)
        
        # Freeze base model parameters (optional)
        for param in self.base_model.parameters():
            param.requires_grad = False
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute reward scores
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            rewards: Reward scores [batch_size]
        """
        # Get hidden states from base model
        outputs = self.base_model(input_ids)
        hidden_states = outputs['logits']  # Use logits as hidden states for simplicity
        
        # Extract representation for reward computation (use last token)
        sequence_lengths = attention_mask.sum(dim=1) - 1
        last_hidden = hidden_states[range(len(sequence_lengths)), sequence_lengths]
        
        # Compute reward score
        rewards = self.reward_head(last_hidden).squeeze(-1)
        
        return rewards


class DistributedTrainer:
    """
    Distributed training infrastructure for large-scale LLM training
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.setup_distributed()
        
    def setup_distributed(self):
        """
        Initialize distributed training
        """
        # Initialize distributed process group
        if self.config.world_size > 1:
            dist.init_process_group(
                backend='nccl' if torch.cuda.is_available() else 'gloo',
                world_size=self.config.world_size,
                rank=self.config.rank
            )
        
        # Set device for current process
        if torch.cuda.is_available():
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f'cuda:{self.config.local_rank}')
        else:
            self.device = torch.device('cpu')
        
        print(f"Initialized distributed training: rank {self.config.rank}/{self.config.world_size}")
    
    def create_distributed_model(self, model: nn.Module) -> nn.Module:
        """
        Wrap model for distributed training
        
        Args:
            model: Model to distribute
            
        Returns:
            distributed_model: DDP-wrapped model
        """
        # Move model to appropriate device
        model = model.to(self.device)
        
        # Wrap with DistributedDataParallel
        if self.config.world_size > 1:
            model = DDP(model, device_ids=[self.config.local_rank])
        
        return model
    
    def create_distributed_dataloader(self, dataset, batch_size: int) -> DataLoader:
        """
        Create distributed data loader
        
        Args:
            dataset: Training dataset
            batch_size: Batch size per process
            
        Returns:
            dataloader: Distributed data loader
        """
        # Create distributed sampler
        if self.config.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.config.world_size,
                rank=self.config.rank
            )
        else:
            sampler = None
        
        # Create data loader with distributed sampler
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=2,
            pin_memory=True
        )
        
        return dataloader


class RLHFTrainer:
    """
    Reinforcement Learning from Human Feedback trainer
    """
    
    def __init__(self, policy_model: SimpleLLM, reward_model: RewardModel, config: TrainingConfig):
        self.policy_model = policy_model
        self.reward_model = reward_model
        self.config = config
        
        # Create reference model (copy of initial policy)
        import copy
        self.ref_model = copy.deepcopy(policy_model)
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Setup optimizers
        self.policy_optimizer = torch.optim.AdamW(
            policy_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
    def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute rewards for prompt-response pairs
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            
        Returns:
            rewards: Reward scores for each response
        """
        # TODO: Tokenize prompts and responses
        # HINT: Concatenate prompts and responses, tokenize
        
        # TODO: Get rewards from reward model
        # HINT: Use reward model to score complete sequences
        
        return torch.zeros(len(responses))  # TODO: Return actual rewards
    
    def compute_kl_penalty(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """
        Compute KL divergence penalty between policy and reference model
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            
        Returns:
            kl_penalty: KL divergence penalty
        """
        # TODO: Get log probabilities from policy and reference models
        # HINT: Use both models to get log probs for responses
        
        # TODO: Compute KL divergence
        # HINT: KL = sum(policy_logprobs - ref_logprobs)
        
        return torch.zeros(len(responses))  # TODO: Return actual KL penalty
    
    def ppo_step(self, prompts: List[str], responses: List[str], old_log_probs: torch.Tensor):
        """
        Perform PPO training step
        
        Args:
            prompts: List of prompt strings
            responses: List of response strings
            old_log_probs: Log probabilities from previous policy
        """
        # TODO: Compute rewards and KL penalty
        rewards = self.compute_rewards(prompts, responses)
        kl_penalty = self.compute_kl_penalty(prompts, responses)
        
        # TODO: Compute advantages
        # HINT: advantages = rewards - kl_penalty
        
        # TODO: Get new log probabilities from current policy
        # HINT: Forward pass through current policy model
        
        # TODO: Compute PPO loss
        # HINT: Use clipped surrogate objective
        
        # TODO: Backward pass and optimization step
        # HINT: loss.backward(), optimizer.step()
        
        print("PPO step completed")


class ConstitutionalAI:
    """
    Constitutional AI implementation for self-improvement
    """
    
    def __init__(self, model: SimpleLLM, constitution: List[str]):
        self.model = model
        self.constitution = constitution
        
    def generate_critique(self, prompt: str, response: str, principle: str) -> str:
        """
        Generate critique of response based on constitutional principle
        
        Args:
            prompt: Original prompt
            response: Model response to critique
            principle: Constitutional principle to apply
            
        Returns:
            critique: Critique of the response
        """
        # TODO: Create critique prompt
        critique_prompt = f"""
        Principle: {principle}
        
        Human: {prompt}
        Assistant: {response}
        
        Does the assistant's response violate the principle? If so, how can it be improved?
        """
        
        # TODO: Generate critique using the model
        # HINT: Use model.generate() or similar method
        
        return "Generated critique"  # TODO: Return actual critique
    
    def generate_revision(self, original_response: str, critiques: List[str]) -> str:
        """
        Generate revised response based on critiques
        
        Args:
            original_response: Original model response
            critiques: List of critiques to address
            
        Returns:
            revised_response: Improved response
        """
        # TODO: Create revision prompt
        revision_prompt = f"""
        Original response: {original_response}
        
        Critiques to address:
        {chr(10).join(f"- {critique}" for critique in critiques)}
        
        Please provide a revised response that addresses these critiques:
        """
        
        # TODO: Generate revised response
        # HINT: Use model to generate improved response
        
        return "Revised response"  # TODO: Return actual revision
    
    def constitutional_training_step(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """
        Perform constitutional AI training step
        
        Args:
            prompts: List of training prompts
            
        Returns:
            training_data: List of training examples with critiques and revisions
        """
        training_data = []
        
        for prompt in prompts:
            # TODO: Generate initial response
            initial_response = "Initial response"  # TODO: Generate actual response
            
            # TODO: Generate critiques for each principle
            critiques = []
            for principle in self.constitution:
                critique = self.generate_critique(prompt, initial_response, principle)
                critiques.append(critique)
            
            # TODO: Generate revised response
            revised_response = self.generate_revision(initial_response, critiques)
            
            training_data.append({
                'prompt': prompt,
                'initial_response': initial_response,
                'critiques': critiques,
                'revised_response': revised_response
            })
        
        return training_data


class TrainingMonitor:
    """
    Comprehensive training monitoring and evaluation
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.metrics = defaultdict(list)
        
        # Initialize wandb for experiment tracking
        try:
            wandb.init(
                project="llm-training-stages",
                config=vars(config),
                mode="disabled"  # Disable for testing
            )
        except Exception:
            print("Warning: wandb initialization failed, continuing without logging")
        
    def log_training_metrics(self, step: int, loss: float, learning_rate: float, grad_norm: float):
        """
        Log training metrics
        
        Args:
            step: Training step
            loss: Training loss
            learning_rate: Current learning rate
            grad_norm: Gradient norm
        """
        # TODO: Store metrics locally
        self.metrics['loss'].append(loss)
        self.metrics['learning_rate'].append(learning_rate)
        self.metrics['grad_norm'].append(grad_norm)
        
        # Log to wandb
        try:
            wandb.log({
                'train/loss': loss,
                'train/learning_rate': learning_rate,
                'train/grad_norm': grad_norm,
                'train/step': step
            })
        except Exception:
            pass  # Continue if wandb logging fails
        
        if step % 100 == 0:
            print(f"Step {step}: Loss={loss:.4f}, LR={learning_rate:.2e}, GradNorm={grad_norm:.4f}")
    
    def evaluate_model(self, model: SimpleLLM, eval_dataset) -> Dict[str, float]:
        """
        Evaluate model on validation dataset
        
        Args:
            model: Model to evaluate
            eval_dataset: Evaluation dataset
            
        Returns:
            metrics: Evaluation metrics
        """
        model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            # TODO: Iterate through evaluation dataset
            # HINT: Compute loss on evaluation data
            
            # TODO: Calculate perplexity
            # HINT: perplexity = exp(average_loss)
            
            pass
        
        model.train()
        
        metrics = {
            'eval_loss': 0.0,  # TODO: Add actual loss
            'perplexity': 0.0   # TODO: Add actual perplexity
        }
        
        return metrics
    
    def generate_samples(self, model: SimpleLLM, prompts: List[str]) -> List[str]:
        """
        Generate sample responses for qualitative evaluation
        
        Args:
            model: Model to use for generation
            prompts: List of prompts
            
        Returns:
            responses: Generated responses
        """
        model.eval()
        responses = []
        
        with torch.no_grad():
            for prompt in prompts:
                # TODO: Tokenize prompt
                # HINT: Convert prompt to token IDs
                
                # TODO: Generate response
                # HINT: Use model to generate continuation
                
                # TODO: Decode response
                # HINT: Convert token IDs back to text
                
                responses.append("Generated response")  # TODO: Add actual response
        
        model.train()
        return responses


def create_sample_datasets():
    """
    Create sample datasets for training exercises
    """
    # Create sample pre-training data
    pretraining_data = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Transformers have revolutionized natural language processing.",
        "Large language models can generate human-like text."
    ]
    
    # TODO: Create sample instruction tuning data
    instruction_data = [
        {
            'instruction': 'Translate the following English text to French:',
            'input': 'Hello, how are you?',
            'output': 'Bonjour, comment allez-vous?'
        },
        {
            'instruction': 'Summarize the following text:',
            'input': 'Machine learning is a method of data analysis that automates analytical model building.',
            'output': 'ML automates analytical model building through data analysis.'
        },
        {
            'instruction': 'Answer the following question:',
            'input': 'What is machine learning?',
            'output': 'Machine learning is a method of data analysis that automates analytical model building.'
        }
    ]
    
    # TODO: Create sample preference data for RLHF
    preference_data = [
        {
            'prompt': 'What is the capital of France?',
            'chosen': 'The capital of France is Paris.',
            'rejected': 'I think it might be Lyon or maybe Marseille.'
        },
        {
            'prompt': 'Explain quantum computing.',
            'chosen': 'Quantum computing uses quantum mechanical phenomena to process information.',
            'rejected': 'Quantum computing is just regular computing but faster.'
        }
    ]
    
    return pretraining_data, instruction_data, preference_data


def test_distributed_training():
    """
    Test distributed training setup
    """
    print("🔧 Testing Distributed Training Setup")
    print("-" * 40)
    
    config = TrainingConfig(
        world_size=1,  # Single process for testing
        rank=0,
        local_rank=0
    )
    
    # Initialize distributed trainer
    trainer = DistributedTrainer(config)
    
    # Create model and wrap for distributed training
    model = SimpleLLM(config)
    distributed_model = trainer.create_distributed_model(model)
    
    print("✅ Distributed training setup completed")


def test_parameter_efficient_fine_tuning():
    """
    Test parameter-efficient fine-tuning methods
    """
    print("🎯 Testing Parameter-Efficient Fine-Tuning")
    print("-" * 40)
    
    config = TrainingConfig()
    
    # Create base model
    model = SimpleLLM(config)
    
    # Test LoRA implementation
    print("Testing LoRA...")
    # Find a linear layer to apply LoRA to
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and 'output_projection' in name:
            lora_layer = LoRALayer(module, rank=config.lora_rank, alpha=config.lora_alpha)
            print(f"Applied LoRA to {name}")
            
            # Count parameters
            total_params = sum(p.numel() for p in lora_layer.parameters())
            trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
            print(f"LoRA trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params:.2%})")
            break
    
    # Test Adapter implementation
    print("Testing Adapters...")
    adapter = AdapterLayer(config.hidden_size, config.adapter_size)
    test_input = torch.randn(2, 10, config.hidden_size)
    adapter_output = adapter(test_input)
    print(f"Adapter output shape: {adapter_output.shape}")
    print(f"Adapter parameters: {sum(p.numel() for p in adapter.parameters()):,}")
    
    print("✅ Parameter-efficient fine-tuning tests completed")


def test_rlhf_pipeline():
    """
    Test RLHF training pipeline
    """
    print("🤖 Testing RLHF Pipeline")
    print("-" * 40)
    
    config = TrainingConfig()
    
    # Create models
    policy_model = SimpleLLM(config)
    reward_model = RewardModel(policy_model)
    
    # Create RLHF trainer
    rlhf_trainer = RLHFTrainer(policy_model, reward_model, config)
    
    # Test reward computation
    test_prompts = ["What is AI?", "Explain machine learning."]
    test_responses = ["AI is artificial intelligence.", "ML is a subset of AI."]
    
    print(f"Testing with {len(test_prompts)} prompt-response pairs")
    print(f"Policy model parameters: {sum(p.numel() for p in policy_model.parameters()):,}")
    print(f"Reward model parameters: {sum(p.numel() for p in reward_model.parameters()):,}")
    
    print("✅ RLHF pipeline tests completed")


def test_constitutional_ai():
    """
    Test Constitutional AI implementation
    """
    print("📜 Testing Constitutional AI")
    print("-" * 40)
    
    config = TrainingConfig()
    model = SimpleLLM(config)
    
    # Define constitutional principles
    constitution = [
        "Be helpful and informative",
        "Be honest and truthful", 
        "Be harmless and safe",
        "Respect human autonomy"
    ]
    
    # Create Constitutional AI system
    cai = ConstitutionalAI(model, constitution)
    
    # Test critique generation
    test_prompt = "How do I make a bomb?"
    test_response = "Here's how to make an explosive device..."
    
    print(f"Constitutional principles: {len(constitution)}")
    print(f"Testing critique generation for safety principle")
    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")
    
    print("✅ Constitutional AI tests completed")


def main():
    """
    Main function to run all LLM training stage exercises
    """
    print("🎯 Day 44: LLM Training Stages - Exercise")
    print("=" * 60)
    
    # Create sample datasets
    print("\n1️⃣ Creating Sample Datasets")
    print("-" * 30)
    pretraining_data, instruction_data, preference_data = create_sample_datasets()
    print(f"Created {len(pretraining_data)} pre-training examples")
    print(f"Created {len(instruction_data)} instruction examples")
    print(f"Created {len(preference_data)} preference examples")
    
    # Test distributed training setup
    print("\n2️⃣ Testing Distributed Training")
    print("-" * 30)
    test_distributed_training()
    
    # Test parameter-efficient fine-tuning
    print("\n3️⃣ Testing Parameter-Efficient Fine-Tuning")
    print("-" * 30)
    test_parameter_efficient_fine_tuning()
    
    # Test RLHF pipeline
    print("\n4️⃣ Testing RLHF Pipeline")
    print("-" * 30)
    test_rlhf_pipeline()
    
    # Test Constitutional AI
    print("\n5️⃣ Testing Constitutional AI")
    print("-" * 30)
    test_constitutional_ai()
    
    # Test training monitoring
    print("\n6️⃣ Testing Training Monitoring")
    print("-" * 30)
    
    config = TrainingConfig()
    monitor = TrainingMonitor(config)
    
    # Simulate training metrics
    for step in range(0, 500, 100):
        loss = 4.0 - step * 0.005  # Decreasing loss
        lr = 1e-4 * (1 - step / 1000)  # Decreasing learning rate
        grad_norm = 1.0 + 0.1 * np.random.randn()  # Random grad norm
        
        monitor.log_training_metrics(step, loss, lr, grad_norm)
    
    print("✅ All LLM training stage exercises completed!")
    
    print("\n📊 Key Insights:")
    print("=" * 50)
    print("🏗️ Training Infrastructure:")
    print("   • Distributed training enables scaling to large models")
    print("   • Memory optimization techniques are crucial for efficiency")
    print("   • Proper checkpointing ensures training resilience")
    
    print("\n🎯 Fine-Tuning Methods:")
    print("   • LoRA provides efficient adaptation with minimal parameters")
    print("   • Adapters offer modular fine-tuning capabilities")
    print("   • Parameter efficiency is key for practical deployment")
    
    print("\n🤖 Alignment Techniques:")
    print("   • RLHF aligns models with human preferences")
    print("   • Constitutional AI enables self-improvement")
    print("   • Multiple alignment methods can be combined effectively")
    
    print("\n📈 Monitoring & Evaluation:")
    print("   • Comprehensive metrics tracking is essential")
    print("   • Qualitative evaluation complements quantitative metrics")
    print("   • Continuous monitoring enables optimization")


if __name__ == "__main__":
    main()
