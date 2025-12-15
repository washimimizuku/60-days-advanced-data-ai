"""
Day 49: RLHF and DPO - Human Feedback & Preference Learning - Exercises

Complete the following exercises to practice alignment techniques:
1. Supervised Fine-tuning (SFT) data preparation
2. Reward model implementation and training
3. DPO trainer implementation
4. Human preference collection system
5. Constitutional AI implementation
6. Alignment monitoring and evaluation
7. Production deployment strategies

Run each exercise and observe the alignment training process.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import json
import random
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

# Mock components for exercises (replace with actual models in production)
class MockLanguageModel(nn.Module):
    def __init__(self, vocab_size=50000, hidden_size=768):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hidden_size, 8, batch_first=True),
            num_layers=6
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        embeddings = self.embedding(input_ids)
        hidden_states = self.transformer(embeddings)
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        return type('Output', (), {
            'logits': logits,
            'loss': loss,
            'last_hidden_state': hidden_states
        })()

class MockTokenizer:
    def __init__(self):
        self.vocab_size = 50000
        self.eos_token_id = 2
        self.pad_token_id = 0
        
    def encode(self, text, return_tensors=None):
        # Simple mock tokenization
        tokens = [hash(word) % self.vocab_size for word in text.split()]
        if return_tensors == 'pt':
            return torch.tensor([tokens])
        return tokens
    
    def decode(self, tokens, skip_special_tokens=False):
        # Simple mock detokenization
        return f"Generated text from {len(tokens)} tokens"

# Exercise 1: Supervised Fine-tuning Data Preparation
class SFTDataPreparator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def prepare_instruction_data(self, demonstrations: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Prepare demonstration data for supervised fine-tuning"""
        formatted_data = []
        
        for demo in demonstrations:
            instruction = demo['instruction']
            response = demo['response']
            
            # Format as instruction-following example
            full_text = f"Human: {instruction}\n\nAssistant: {response}"
            
            # Tokenize
            input_ids = self.tokenizer.encode(full_text, return_tensors='pt', max_length=512, truncation=True)
            
            # Create labels (only train on assistant response)
            instruction_part = f"Human: {instruction}\n\nAssistant: "
            instruction_tokens = self.tokenizer.encode(instruction_part, max_length=512, truncation=True)
            
            # Labels: -100 for instruction tokens, actual tokens for response
            labels = input_ids.clone()
            labels[0, :len(instruction_tokens)] = -100
            
            formatted_data.append({
                'input_ids': input_ids.squeeze(0),
                'labels': labels.squeeze(0),
                'instruction': instruction,
                'response': response
            })
        
        return formatted_data
    
    def create_conversation_format(self, conversations: List[List[Dict[str, str]]]) -> List[Dict[str, Any]]:
        """Format multi-turn conversations for training"""
        formatted_conversations = []
        
        for conversation in conversations:
            # Build conversation text
            conversation_text = ""
            for turn in conversation:
                role = turn['role'].title()
                content = turn['content']
                conversation_text += f"{role}: {content}\n\n"
            
            # Tokenize full conversation
            input_ids = self.tokenizer.encode(conversation_text, return_tensors='pt', max_length=512, truncation=True)
            
            # Create labels (train on assistant responses only)
            labels = input_ids.clone()
            
            # Find assistant response positions and mask human parts
            text_parts = conversation_text.split("Assistant: ")
            if len(text_parts) > 1:
                human_part = text_parts[0]
                human_tokens = len(self.tokenizer.encode(human_part))
                labels[0, :human_tokens] = -100
            
            formatted_conversations.append({
                'input_ids': input_ids.squeeze(0),
                'labels': labels.squeeze(0),
                'conversation': conversation
            })
        
        return formatted_conversations

def exercise_1_sft_preparation():
    """Exercise 1: Supervised Fine-tuning data preparation"""
    print("=== Exercise 1: SFT Data Preparation ===")
    
    tokenizer = MockTokenizer()
    preparator = SFTDataPreparator(tokenizer)
    
    # Sample demonstration data
    demonstrations = [
        {
            "instruction": "Explain what machine learning is",
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."
        },
        {
            "instruction": "Write a Python function to calculate factorial",
            "response": "Here's a Python function to calculate factorial:\n\ndef factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)"
        }
    ]
    
    # Prepare SFT data
    sft_data = preparator.prepare_instruction_data(demonstrations)
    print(f"Prepared {len(sft_data)} SFT examples")
    
    # Sample conversation data
    conversations = [
        [
            {"role": "human", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "human", "content": "What's the population?"},
            {"role": "assistant", "content": "Paris has a population of approximately 2.1 million people in the city proper."}
        ]
    ]
    
    # Prepare conversation data
    conversation_data = preparator.create_conversation_format(conversations)
    print(f"Prepared {len(conversation_data)} conversation examples")
    
    # Display sample data structure
    if sft_data:
        sample = sft_data[0]
        print(f"Sample SFT data structure:")
        print(f"  Input IDs shape: {sample['input_ids'].shape}")
        print(f"  Labels shape: {sample['labels'].shape}")
        print(f"  Instruction: {sample['instruction'][:50]}...")
    
    print("✅ SFT data preparation completed successfully!")
    print()

# Exercise 2: Reward Model Implementation
class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size=768):
        super().__init__()
        self.base_model = base_model
        
        # Freeze base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
        
        # Create reward head
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, input_ids, attention_mask=None):
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        
        # Use last token representation for reward
        last_hidden_state = outputs.last_hidden_state
        
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
        else:
            sequence_lengths = torch.full((input_ids.size(0),), input_ids.size(1) - 1, device=input_ids.device)
        
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
        
    def prepare_preference_data(self, preference_pairs: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Prepare human preference data for reward model training"""
        training_data = []
        
        for pair in preference_pairs:
            prompt = pair['prompt']
            chosen = pair['chosen']
            rejected = pair['rejected']
            
            # Create full texts
            chosen_text = f"{prompt}\n{chosen}"
            rejected_text = f"{prompt}\n{rejected}"
            
            # Tokenize
            chosen_tokens = self.tokenizer.encode(chosen_text, return_tensors='pt', max_length=512, truncation=True)
            rejected_tokens = self.tokenizer.encode(rejected_text, return_tensors='pt', max_length=512, truncation=True)
            
            training_data.append({
                'chosen_input_ids': chosen_tokens.squeeze(0),
                'rejected_input_ids': rejected_tokens.squeeze(0),
                'prompt': prompt,
                'chosen': chosen,
                'rejected': rejected
            })
        
        return training_data
    
    def compute_reward_loss(self, chosen_rewards: torch.Tensor, 
                           rejected_rewards: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
        """Compute ranking loss for reward model"""
        # Preference loss: chosen should have higher reward than rejected
        loss = -torch.log(torch.sigmoid(chosen_rewards - rejected_rewards - margin))
        return loss.mean()
    
    def train_reward_model(self, preference_data: List[Dict[str, Any]], epochs: int = 3):
        """Train reward model on human preference data"""
        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=1e-5)
        
        for epoch in range(epochs):
            total_loss = 0
            correct_preferences = 0
            total_pairs = 0
            
            for batch in preference_data:
                # Get rewards for chosen and rejected responses
                chosen_rewards = self.reward_model(batch['chosen_input_ids'].unsqueeze(0))
                rejected_rewards = self.reward_model(batch['rejected_input_ids'].unsqueeze(0))
                
                # Compute loss
                loss = self.compute_reward_loss(chosen_rewards, rejected_rewards)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Track metrics
                total_loss += loss.item()
                if chosen_rewards.item() > rejected_rewards.item():
                    correct_preferences += 1
                total_pairs += 1
            
            avg_loss = total_loss / len(preference_data)
            accuracy = correct_preferences / total_pairs
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return self.reward_model

def exercise_2_reward_model():
    """Exercise 2: Reward model implementation and training"""
    print("=== Exercise 2: Reward Model Implementation ===")
    
    # Create mock components
    base_model = MockLanguageModel()
    tokenizer = MockTokenizer()
    
    # Create reward model
    reward_model = RewardModel(base_model)
    trainer = RewardModelTrainer(reward_model, tokenizer)
    
    # Sample preference data
    preference_pairs = [
        {
            "prompt": "Explain quantum computing",
            "chosen": "Quantum computing uses quantum mechanical phenomena like superposition and entanglement to process information in ways that classical computers cannot.",
            "rejected": "Quantum computing is just faster computers."
        },
        {
            "prompt": "Write a haiku about spring",
            "chosen": "Cherry blossoms bloom\nGentle breeze carries petals\nSpring awakens life",
            "rejected": "Spring is nice and warm\nFlowers bloom everywhere now\nI like the season"
        }
    ]
    
    # Prepare and train reward model
    training_data = trainer.prepare_preference_data(preference_pairs)
    print(f"Prepared {len(training_data)} preference pairs")
    
    # Test reward model forward pass
    sample_input = training_data[0]['chosen_input_ids'].unsqueeze(0)
    reward_score = reward_model(sample_input)
    print(f"Sample reward score: {reward_score.item():.4f}")
    
    # Train reward model
    print("Training reward model...")
    trained_model = trainer.train_reward_model(training_data, epochs=2)
    
    print("✅ Reward model implementation and training completed successfully!")
    print()

# Exercise 3: DPO Implementation
class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, beta: float = 0.1):
        self.model = model           # Policy model being optimized
        self.ref_model = ref_model   # Reference model (frozen)
        self.tokenizer = tokenizer
        self.beta = beta             # Temperature parameter
        
        # Freeze reference model parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
    def compute_dpo_loss(self, policy_chosen_logps: torch.Tensor, 
                        policy_rejected_logps: torch.Tensor,
                        reference_chosen_logps: torch.Tensor, 
                        reference_rejected_logps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def get_batch_logps(self, model, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Get log probabilities for a batch of sequences"""
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
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single DPO training step"""
        chosen_input_ids = batch['chosen_input_ids'].unsqueeze(0)
        chosen_labels = batch['chosen_labels'].unsqueeze(0)
        rejected_input_ids = batch['rejected_input_ids'].unsqueeze(0)
        rejected_labels = batch['rejected_labels'].unsqueeze(0)
        
        # Get log probabilities from policy model
        policy_chosen_logps = self.get_batch_logps(self.model, chosen_input_ids, chosen_labels)
        policy_rejected_logps = self.get_batch_logps(self.model, rejected_input_ids, rejected_labels)
        
        # Get log probabilities from reference model
        with torch.no_grad():
            reference_chosen_logps = self.get_batch_logps(self.ref_model, chosen_input_ids, chosen_labels)
            reference_rejected_logps = self.get_batch_logps(self.ref_model, rejected_input_ids, rejected_labels)
        
        # Compute DPO loss
        loss, accuracy = self.compute_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'policy_chosen_logps': policy_chosen_logps.mean().item(),
            'policy_rejected_logps': policy_rejected_logps.mean().item()
        }
    
    def prepare_dpo_data(self, preference_pairs: List[Dict[str, str]]) -> List[Dict[str, torch.Tensor]]:
        """Prepare preference data for DPO training"""
        dpo_data = []
        
        for pair in preference_pairs:
            prompt = pair['prompt']
            chosen = pair['chosen']
            rejected = pair['rejected']
            
            # Create full texts
            chosen_text = f"{prompt}\n{chosen}"
            rejected_text = f"{prompt}\n{rejected}"
            
            # Tokenize
            chosen_tokens = self.tokenizer.encode(chosen_text, return_tensors='pt', max_length=512, truncation=True)
            rejected_tokens = self.tokenizer.encode(rejected_text, return_tensors='pt', max_length=512, truncation=True)
            
            # Create labels (same as input_ids for DPO)
            chosen_labels = chosen_tokens.clone()
            rejected_labels = rejected_tokens.clone()
            
            dpo_data.append({
                'chosen_input_ids': chosen_tokens.squeeze(0),
                'chosen_labels': chosen_labels.squeeze(0),
                'rejected_input_ids': rejected_tokens.squeeze(0),
                'rejected_labels': rejected_labels.squeeze(0)
            })
        
        return dpo_data
    
    def train(self, preference_pairs: List[Dict[str, str]], epochs: int = 3, lr: float = 1e-6):
        """Train DPO model"""
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        
        # Prepare data
        dpo_data = self.prepare_dpo_data(preference_pairs)
        
        for epoch in range(epochs):
            total_loss = 0
            total_accuracy = 0
            
            for batch_idx, batch in enumerate(dpo_data):
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
            
            avg_loss = total_loss / len(dpo_data)
            avg_accuracy = total_accuracy / len(dpo_data)
            
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}")
        
        return self.model

def exercise_3_dpo_implementation():
    """Exercise 3: DPO trainer implementation"""
    print("=== Exercise 3: DPO Implementation ===")
    
    # Create mock components
    model = MockLanguageModel()
    ref_model = MockLanguageModel()
    tokenizer = MockTokenizer()
    
    # Create DPO trainer
    dpo_trainer = DPOTrainer(model, ref_model, tokenizer, beta=0.1)
    
    # Mock training data
    preference_data = [
        {
            "prompt": "Explain artificial intelligence",
            "chosen": "AI is a field of computer science focused on creating systems that can perform tasks typically requiring human intelligence.",
            "rejected": "AI is robots that will take over the world."
        },
        {
            "prompt": "What is machine learning?",
            "chosen": "Machine learning is a subset of AI that enables computers to learn from data without explicit programming.",
            "rejected": "Machine learning is magic."
        }
    ]
    
    # Prepare DPO data
    dpo_data = dpo_trainer.prepare_dpo_data(preference_data)
    print(f"Prepared {len(dpo_data)} DPO training examples")
    
    # Test DPO loss computation
    sample_batch = dpo_data[0]
    metrics = dpo_trainer.train_step(sample_batch)
    print(f"Sample DPO metrics: Loss={metrics['loss'].item():.4f}, Accuracy={metrics['accuracy'].item():.4f}")
    
    # Train DPO model
    print("Training DPO model...")
    trained_model = dpo_trainer.train(preference_data, epochs=2)
    
    print("✅ DPO implementation completed successfully!")
    print()

# Exercise 4: Human Preference Collection System
class PreferenceCollectionSystem:
    def __init__(self):
        self.annotation_guidelines = {
            'helpfulness': 'Rate how well the response addresses the user\'s request and provides useful information',
            'harmlessness': 'Rate how safe and non-harmful the response is, avoiding toxic or dangerous content',
            'honesty': 'Rate how truthful and accurate the response is, including acknowledgment of uncertainty',
            'clarity': 'Rate how clear, well-structured, and easy to understand the response is'
        }
        
        self.quality_checks = [
            'inter_annotator_agreement',
            'annotation_time_validation',
            'consistency_checks',
            'expert_review'
        ]
        
    def generate_task_id(self) -> str:
        """Generate unique task ID"""
        import uuid
        return str(uuid.uuid4())
        
    def create_annotation_task(self, prompt: str, responses: List[str], 
                             criteria: List[str] = None) -> Dict[str, Any]:
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
            'status': 'pending',
            'num_responses': len(responses)
        }
        
        return task
    
    def validate_annotations(self, annotations: Dict[str, Any]) -> Dict[str, Any]:
        """Validate annotation quality"""
        validation_results = {
            'valid': True,
            'issues': [],
            'confidence_score': 1.0
        }
        
        # Check annotation time (too fast might indicate low quality)
        annotation_time = annotations.get('annotation_time', 0)
        if annotation_time < 30:  # seconds
            validation_results['issues'].append('Annotation completed too quickly')
            validation_results['confidence_score'] *= 0.8
        
        # Check for consistent preferences
        preferences = annotations.get('preferences', [])
        if len(set(preferences)) == 1 and len(preferences) > 1:
            validation_results['issues'].append('No preference variation - all responses rated the same')
            validation_results['confidence_score'] *= 0.9
        
        # Check for explanation quality
        explanations = annotations.get('explanations', [])
        if explanations:
            short_explanations = [exp for exp in explanations if len(exp.split()) < 5]
            if short_explanations:
                validation_results['issues'].append('Insufficient explanation detail')
                validation_results['confidence_score'] *= 0.7
        
        # Check for extreme ratings without justification
        ratings = annotations.get('ratings', [])
        if ratings:
            extreme_ratings = [r for r in ratings if r in [1, 5]]  # Assuming 1-5 scale
            if len(extreme_ratings) > len(ratings) * 0.8:  # More than 80% extreme
                validation_results['issues'].append('Too many extreme ratings')
                validation_results['confidence_score'] *= 0.8
        
        if validation_results['confidence_score'] < 0.7:
            validation_results['valid'] = False
        
        return validation_results
    
    def compute_inter_annotator_agreement(self, annotations_list: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute agreement between multiple annotators"""
        if len(annotations_list) < 2:
            return {'agreement': 1.0, 'method': 'single_annotator'}
        
        # Simple pairwise agreement calculation
        agreements = []
        
        for i in range(len(annotations_list)):
            for j in range(i + 1, len(annotations_list)):
                ann1_prefs = annotations_list[i].get('preferences', [])
                ann2_prefs = annotations_list[j].get('preferences', [])
                
                if len(ann1_prefs) == len(ann2_prefs) and len(ann1_prefs) > 0:
                    agreement = sum(1 for a, b in zip(ann1_prefs, ann2_prefs) if a == b) / len(ann1_prefs)
                    agreements.append(agreement)
        
        avg_agreement = sum(agreements) / len(agreements) if agreements else 0.0
        
        return {
            'agreement': avg_agreement,
            'method': 'pairwise_agreement',
            'num_pairs': len(agreements),
            'agreement_threshold': 0.7,
            'acceptable': avg_agreement >= 0.7
        }
    
    def generate_annotation_report(self, annotations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive annotation quality report"""
        if not annotations:
            return {'error': 'No annotations provided'}
        
        # Compute basic statistics
        total_annotations = len(annotations)
        valid_annotations = sum(1 for ann in annotations if self.validate_annotations(ann)['valid'])
        
        # Compute inter-annotator agreement
        agreement_metrics = self.compute_inter_annotator_agreement(annotations)
        
        # Analyze annotation patterns
        all_preferences = []
        all_ratings = []
        annotation_times = []
        
        for ann in annotations:
            all_preferences.extend(ann.get('preferences', []))
            all_ratings.extend(ann.get('ratings', []))
            if 'annotation_time' in ann:
                annotation_times.append(ann['annotation_time'])
        
        report = {
            'summary': {
                'total_annotations': total_annotations,
                'valid_annotations': valid_annotations,
                'validity_rate': valid_annotations / total_annotations if total_annotations > 0 else 0,
                'average_annotation_time': np.mean(annotation_times) if annotation_times else 0
            },
            'agreement_metrics': agreement_metrics,
            'preference_distribution': {
                'mean_preference': np.mean(all_preferences) if all_preferences else 0,
                'preference_variance': np.var(all_preferences) if all_preferences else 0
            },
            'quality_issues': [],
            'recommendations': []
        }
        
        # Add quality issues and recommendations
        if agreement_metrics['agreement'] < 0.7:
            report['quality_issues'].append('Low inter-annotator agreement')
            report['recommendations'].append('Review annotation guidelines and provide additional training')
        
        if report['summary']['validity_rate'] < 0.8:
            report['quality_issues'].append('High rate of invalid annotations')
            report['recommendations'].append('Implement stricter quality control measures')
        
        return report

def exercise_4_preference_collection():
    """Exercise 4: Human preference collection system"""
    print("=== Exercise 4: Human Preference Collection System ===")
    
    collection_system = PreferenceCollectionSystem()
    
    # Sample annotation scenario
    prompt = "Write a professional email declining a job offer"
    responses = [
        "Thank you for the offer. I must decline due to other commitments. Best regards.",
        "Thanks for the job offer! Unfortunately, I can't accept it right now because I have other plans. Hope we can work together in the future!"
    ]
    
    # Create annotation task
    task = collection_system.create_annotation_task(prompt, responses, ['professionalism', 'clarity'])
    print(f"Created annotation task: {task['task_id']}")
    print(f"Task criteria: {task['criteria']}")
    
    # Mock annotations from multiple annotators
    mock_annotations = [
        {
            'preferences': [1, 0], 
            'explanations': ['More professional tone', 'Better structure'],
            'annotation_time': 120,
            'ratings': [4, 2]
        },
        {
            'preferences': [1, 0], 
            'explanations': ['Appropriate formality', 'Clear message'],
            'annotation_time': 95,
            'ratings': [5, 3]
        },
        {
            'preferences': [0, 1], 
            'explanations': ['Too formal', 'More friendly'],
            'annotation_time': 80,
            'ratings': [2, 4]
        }
    ]
    
    # Validate annotations and compute agreement
    print("\nValidating annotations:")
    for i, annotation in enumerate(mock_annotations):
        validation = collection_system.validate_annotations(annotation)
        print(f"Annotation {i+1}: Valid={validation['valid']}, Score={validation['confidence_score']:.2f}")
        if validation['issues']:
            print(f"  Issues: {validation['issues']}")
    
    agreement = collection_system.compute_inter_annotator_agreement(mock_annotations)
    print(f"\nInter-annotator agreement: {agreement['agreement']:.2f} ({agreement['method']})")
    print(f"Agreement acceptable: {agreement['acceptable']}")
    
    # Generate annotation report
    report = collection_system.generate_annotation_report(mock_annotations)
    print(f"\nAnnotation Report Summary:")
    print(f"  Total annotations: {report['summary']['total_annotations']}")
    print(f"  Valid annotations: {report['summary']['valid_annotations']}")
    print(f"  Validity rate: {report['summary']['validity_rate']:.2f}")
    
    if report['recommendations']:
        print(f"  Recommendations: {report['recommendations']}")
    
    print("✅ Preference collection system completed successfully!")
    print()

# Exercise 5: Constitutional AI Implementation
class ConstitutionalAI:
    def __init__(self, model, tokenizer, constitution: List[str]):
        self.model = model
        self.tokenizer = tokenizer
        self.constitution = constitution
        
    def critique_response(self, prompt: str, response: str) -> str:
        """Generate critique based on constitutional principles"""
        constitution_text = "\n".join([f"- {principle}" for principle in self.constitution])
        
        critique_prompt = f"""Please critique the following response according to these principles:

Constitutional Principles:
{constitution_text}

Prompt: {prompt}
Response: {response}

Please provide a detailed critique focusing on:
1. Adherence to constitutional principles
2. Potential issues or concerns
3. Suggestions for improvement

Critique:"""
        
        # Generate critique (mock implementation)
        critique_input = self.tokenizer.encode(critique_prompt, return_tensors='pt', max_length=512, truncation=True)
        
        # Mock critique generation
        critique = f"The response addresses the prompt but could be improved by being more helpful and providing specific examples. It adheres to safety principles."
        
        return critique.strip()
    
    def revise_response(self, prompt: str, response: str, critique: str) -> str:
        """Revise response based on critique"""
        revision_prompt = f"""Please revise the following response based on the critique provided:

Original Prompt: {prompt}
Original Response: {response}
Critique: {critique}

Please provide a revised response that addresses the concerns raised in the critique while maintaining helpfulness and accuracy.

Revised Response:"""
        
        # Mock revision generation
        revised_response = f"Here's an improved response that addresses your question with specific examples and clear explanations while following safety guidelines."
        
        return revised_response.strip()
    
    def generate_response(self, prompt: str) -> str:
        """Generate initial response to prompt"""
        input_ids = self.tokenizer.encode(f"Human: {prompt}\n\nAssistant:", return_tensors='pt')
        
        # Mock response generation
        response = f"This is a helpful response to: {prompt[:50]}..."
        
        return response.strip()
    
    def constitutional_training_step(self, prompts: List[str]) -> List[Dict[str, str]]:
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
                'critique': critique,
                'improvement_type': 'constitutional_revision'
            })
        
        return training_data
    
    def evaluate_constitutional_adherence(self, responses: List[str]) -> Dict[str, float]:
        """Evaluate how well responses adhere to constitutional principles"""
        adherence_scores = {}
        
        for i, principle in enumerate(self.constitution):
            principle_scores = []
            
            for response in responses:
                # Simple scoring based on principle keywords
                score = self._score_principle_adherence(response, principle)
                principle_scores.append(score)
            
            adherence_scores[f"principle_{i+1}"] = np.mean(principle_scores)
            adherence_scores[f"principle_{i+1}_name"] = principle
        
        # Overall adherence score
        adherence_scores['overall_adherence'] = np.mean([
            score for key, score in adherence_scores.items() 
            if key.startswith('principle_') and not key.endswith('_name')
        ])
        
        return adherence_scores
    
    def _score_principle_adherence(self, response: str, principle: str) -> float:
        """Score how well a response adheres to a specific principle"""
        response_lower = response.lower()
        
        if "helpful" in principle.lower():
            helpful_indicators = ["here's", "i can help", "let me", "solution", "answer"]
            score = sum(1 for indicator in helpful_indicators if indicator in response_lower) / len(helpful_indicators)
        elif "harmful" in principle.lower() or "safe" in principle.lower():
            harmful_indicators = ["dangerous", "illegal", "harmful", "toxic"]
            score = 1.0 - (sum(1 for indicator in harmful_indicators if indicator in response_lower) / len(harmful_indicators))
        elif "honest" in principle.lower():
            honesty_indicators = ["i don't know", "uncertain", "not sure", "accurate", "factual"]
            score = sum(1 for indicator in honesty_indicators if indicator in response_lower) / len(honesty_indicators)
        else:
            score = 0.8  # Default good adherence
        
        return min(1.0, max(0.0, score))

def exercise_5_constitutional_ai():
    """Exercise 5: Constitutional AI implementation"""
    print("=== Exercise 5: Constitutional AI Implementation ===")
    
    model = MockLanguageModel()
    tokenizer = MockTokenizer()
    
    # Sample constitution (principles)
    constitution = [
        "Be helpful and informative",
        "Avoid harmful or offensive content",
        "Be honest and acknowledge uncertainty",
        "Respect privacy and confidentiality",
        "Promote fairness and avoid bias"
    ]
    
    # Create Constitutional AI system
    constitutional_ai = ConstitutionalAI(model, tokenizer, constitution)
    
    print(f"Constitutional AI initialized with {len(constitution)} principles")
    
    # Test prompts
    test_prompts = [
        "How can I improve my public speaking skills?",
        "What are some healthy meal ideas?",
        "Explain the basics of investing"
    ]
    
    # Run constitutional training step
    training_data = constitutional_ai.constitutional_training_step(test_prompts)
    print(f"\nGenerated {len(training_data)} constitutional training examples")
    
    # Display sample training data
    if training_data:
        sample = training_data[0]
        print(f"\nSample constitutional training example:")
        print(f"  Prompt: {sample['prompt'][:50]}...")
        print(f"  Chosen: {sample['chosen'][:50]}...")
        print(f"  Rejected: {sample['rejected'][:50]}...")
    
    # Evaluate constitutional adherence
    sample_responses = [
        "I can help you with that task. Here's a step-by-step approach...",
        "I'm not certain about that specific detail, but I can share what I do know...",
        "That's an interesting question. Let me provide accurate information..."
    ]
    
    adherence_scores = constitutional_ai.evaluate_constitutional_adherence(sample_responses)
    print(f"\nConstitutional adherence evaluation:")
    print(f"  Overall adherence: {adherence_scores['overall_adherence']:.2f}")
    
    for key, value in adherence_scores.items():
        if key.startswith('principle_') and not key.endswith('_name'):
            principle_name = adherence_scores.get(f"{key}_name", "Unknown")
            print(f"  {principle_name}: {value:.2f}")
    
    print("✅ Constitutional AI implementation completed successfully!")
    print()

# Exercise 6: Alignment Monitoring and Evaluation
class AlignmentMonitor:
    def __init__(self):
        self.safety_classifiers = {}
        self.alignment_metrics = defaultdict(list)
        self.load_safety_classifiers()
        
    def load_safety_classifiers(self):
        """Load safety classification models"""
        # Mock classifiers (replace with actual models in production)
        self.safety_classifiers = {
            'toxicity': self._mock_toxicity_classifier,
            'bias': self._mock_bias_classifier,
            'factuality': self._mock_factuality_classifier,
            'helpfulness': self._mock_helpfulness_classifier
        }
    
    def _mock_toxicity_classifier(self, text: str) -> Dict[str, Any]:
        """Mock toxicity classifier"""
        toxic_words = ['hate', 'kill', 'stupid', 'idiot', 'toxic']
        toxic_count = sum(1 for word in toxic_words if word in text.lower())
        score = min(1.0, toxic_count / 10.0)
        return {'toxic': score > 0.3, 'score': score}
    
    def _mock_bias_classifier(self, text: str) -> Dict[str, Any]:
        """Mock bias classifier"""
        bias_indicators = ['always', 'never', 'all women', 'all men', 'stereotype']
        bias_count = sum(1 for indicator in bias_indicators if indicator in text.lower())
        score = min(1.0, bias_count / 5.0)
        return {'biased': score > 0.4, 'score': score}
    
    def _mock_factuality_classifier(self, text: str) -> Dict[str, Any]:
        """Mock factuality classifier"""
        uncertainty_indicators = ['i think', 'maybe', 'possibly', 'not sure']
        factual_indicators = ['according to', 'research shows', 'studies indicate']
        
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in text.lower())
        factual_count = sum(1 for indicator in factual_indicators if indicator in text.lower())
        
        confidence = max(0.5, 0.9 - uncertainty_count * 0.1 + factual_count * 0.1)
        return {'factual': confidence > 0.7, 'confidence': confidence}
    
    def _mock_helpfulness_classifier(self, text: str) -> Dict[str, Any]:
        """Mock helpfulness classifier"""
        helpful_indicators = ['here is', 'i can help', 'solution', 'steps', 'example']
        unhelpful_indicators = ['i cannot', 'impossible', 'no way']
        
        helpful_count = sum(1 for indicator in helpful_indicators if indicator in text.lower())
        unhelpful_count = sum(1 for indicator in unhelpful_indicators if indicator in text.lower())
        
        score = max(0.0, min(1.0, (helpful_count - unhelpful_count) / 5.0 + 0.5))
        return {'helpful': score > 0.6, 'score': score}
    
    def evaluate_response_safety(self, response: str) -> Dict[str, Any]:
        """Evaluate response safety across multiple dimensions"""
        safety_results = {}
        
        for classifier_name, classifier in self.safety_classifiers.items():
            result = classifier(response)
            safety_results[classifier_name] = result
        
        # Compute overall safety score
        safety_score = 1.0
        
        if safety_results['toxicity']['toxic']:
            safety_score *= (1.0 - safety_results['toxicity']['score'])
        
        if safety_results['bias']['biased']:
            safety_score *= (1.0 - safety_results['bias']['score'] * 0.5)
        
        if not safety_results['factuality']['factual']:
            safety_score *= safety_results['factuality']['confidence']
        
        if not safety_results['helpfulness']['helpful']:
            safety_score *= safety_results['helpfulness']['score']
        
        return {
            'safety_score': safety_score,
            'details': safety_results,
            'safe': safety_score > 0.8,
            'timestamp': time.time()
        }
    
    def monitor_alignment_drift(self, recent_responses: List[str], 
                              baseline_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Monitor for alignment drift over time"""
        current_metrics = self.compute_alignment_metrics(recent_responses)
        
        drift_analysis = {}
        
        for metric, baseline_value in baseline_metrics.items():
            current_value = current_metrics.get(metric, 0)
            
            if baseline_value > 0:
                drift = abs(current_value - baseline_value) / baseline_value
            else:
                drift = abs(current_value - baseline_value)
            
            drift_analysis[metric] = {
                'baseline': baseline_value,
                'current': current_value,
                'drift_percentage': drift * 100,
                'significant_drift': drift > 0.1,
                'direction': 'improvement' if current_value > baseline_value else 'degradation'
            }
        
        # Overall drift assessment
        significant_drifts = sum(1 for analysis in drift_analysis.values() if analysis['significant_drift'])
        drift_analysis['overall_assessment'] = {
            'metrics_with_significant_drift': significant_drifts,
            'total_metrics': len(drift_analysis),
            'drift_rate': significant_drifts / len(drift_analysis) if drift_analysis else 0,
            'requires_attention': significant_drifts > len(drift_analysis) * 0.3
        }
        
        return drift_analysis
    
    def compute_alignment_metrics(self, responses: List[str]) -> Dict[str, float]:
        """Compute comprehensive alignment metrics"""
        if not responses:
            return {}
        
        metrics = {
            'avg_safety_score': 0,
            'toxicity_rate': 0,
            'bias_rate': 0,
            'factuality_rate': 0,
            'helpfulness_rate': 0
        }
        
        safety_scores = []
        toxicity_count = 0
        bias_count = 0
        factual_count = 0
        helpful_count = 0
        
        for response in responses:
            safety_eval = self.evaluate_response_safety(response)
            safety_scores.append(safety_eval['safety_score'])
            
            details = safety_eval['details']
            if details['toxicity']['toxic']:
                toxicity_count += 1
            if details['bias']['biased']:
                bias_count += 1
            if details['factuality']['factual']:
                factual_count += 1
            if details['helpfulness']['helpful']:
                helpful_count += 1
        
        metrics['avg_safety_score'] = np.mean(safety_scores)
        metrics['toxicity_rate'] = toxicity_count / len(responses)
        metrics['bias_rate'] = bias_count / len(responses)
        metrics['factuality_rate'] = factual_count / len(responses)
        metrics['helpfulness_rate'] = helpful_count / len(responses)
        
        return metrics
    
    def generate_alignment_report(self, responses: List[str], 
                                time_period: str = "24h") -> Dict[str, Any]:
        """Generate comprehensive alignment report"""
        if not responses:
            return {'error': 'No responses provided'}
        
        # Compute current metrics
        current_metrics = self.compute_alignment_metrics(responses)
        
        # Analyze individual responses
        response_analyses = []
        for i, response in enumerate(responses):
            analysis = self.evaluate_response_safety(response)
            analysis['response_id'] = i
            response_analyses.append(analysis)
        
        # Identify problematic responses
        problematic_responses = [
            analysis for analysis in response_analyses 
            if not analysis['safe']
        ]
        
        # Generate recommendations
        recommendations = []
        
        if current_metrics['toxicity_rate'] > 0.05:
            recommendations.append("High toxicity rate detected. Review content filtering.")
        
        if current_metrics['bias_rate'] > 0.1:
            recommendations.append("Elevated bias rate. Consider bias mitigation training.")
        
        if current_metrics['factuality_rate'] < 0.8:
            recommendations.append("Low factuality rate. Improve fact-checking mechanisms.")
        
        if current_metrics['helpfulness_rate'] < 0.7:
            recommendations.append("Low helpfulness rate. Review instruction following training.")
        
        report = {
            'summary': {
                'time_period': time_period,
                'total_responses': len(responses),
                'safe_responses': len(responses) - len(problematic_responses),
                'problematic_responses': len(problematic_responses),
                'overall_safety_rate': (len(responses) - len(problematic_responses)) / len(responses)
            },
            'metrics': current_metrics,
            'problematic_responses': problematic_responses[:5],  # Top 5 issues
            'recommendations': recommendations,
            'generated_at': time.time()
        }
        
        return report

def exercise_6_alignment_monitoring():
    """Exercise 6: Alignment monitoring and evaluation"""
    print("=== Exercise 6: Alignment Monitoring and Evaluation ===")
    
    monitor = AlignmentMonitor()
    
    # Sample responses to monitor
    sample_responses = [
        "I'd be happy to help you with that task. Here's a step-by-step approach...",
        "I cannot provide information on that topic as it could be harmful.",
        "I'm not certain about that specific detail, but I can share what I do know...",
        "That's an interesting question. Let me break it down for you..."
    ]
    
    print(f"Monitoring {len(sample_responses)} responses...")
    
    # Evaluate response safety
    print("\nSafety evaluation results:")
    for i, response in enumerate(sample_responses[:2]):  # Test first 2
        safety_eval = monitor.evaluate_response_safety(response)
        print(f"Response {i+1}:")
        print(f"  Safety score: {safety_eval['safety_score']:.2f}")
        print(f"  Safe: {safety_eval['safe']}")
        print(f"  Toxicity: {safety_eval['details']['toxicity']['score']:.2f}")
        print(f"  Helpfulness: {safety_eval['details']['helpfulness']['score']:.2f}")
    
    # Compute alignment metrics
    current_metrics = monitor.compute_alignment_metrics(sample_responses)
    print(f"\nCurrent alignment metrics:")
    for metric, value in current_metrics.items():
        print(f"  {metric}: {value:.3f}")
    
    # Monitor alignment drift
    baseline_metrics = {
        'avg_safety_score': 0.9, 
        'helpfulness_rate': 0.85, 
        'toxicity_rate': 0.02
    }
    
    drift_analysis = monitor.monitor_alignment_drift(sample_responses, baseline_metrics)
    print(f"\nAlignment drift analysis:")
    print(f"  Requires attention: {drift_analysis['overall_assessment']['requires_attention']}")
    print(f"  Metrics with significant drift: {drift_analysis['overall_assessment']['metrics_with_significant_drift']}")
    
    for metric in ['avg_safety_score', 'helpfulness_rate']:
        if metric in drift_analysis:
            analysis = drift_analysis[metric]
            print(f"  {metric}: {analysis['direction']} ({analysis['drift_percentage']:.1f}% drift)")
    
    # Generate alignment report
    report = monitor.generate_alignment_report(sample_responses)
    print(f"\nAlignment Report Summary:")
    print(f"  Total responses: {report['summary']['total_responses']}")
    print(f"  Safe responses: {report['summary']['safe_responses']}")
    print(f"  Overall safety rate: {report['summary']['overall_safety_rate']:.2f}")
    
    if report['recommendations']:
        print(f"  Recommendations: {len(report['recommendations'])} items")
        for rec in report['recommendations'][:2]:
            print(f"    - {rec}")
    
    print("✅ Alignment monitoring and evaluation completed successfully!")
    print()

# Exercise 7: Production Deployment Strategies
class AlignmentDeploymentManager:
    def __init__(self):
        self.deployment_configs = {}
        self.monitoring_systems = []
        self.rollout_history = []
        
    def generate_id(self) -> str:
        """Generate unique ID"""
        import uuid
        return str(uuid.uuid4())
        
    def create_staged_rollout_plan(self, model_versions: List[str], 
                                 rollout_percentages: List[float]) -> Dict[str, Any]:
        """Create staged rollout plan for aligned models"""
        if len(model_versions) != len(rollout_percentages):
            raise ValueError("Number of model versions must match rollout percentages")
        
        rollout_plan = {
            'plan_id': self.generate_id(),
            'created_at': time.time(),
            'stages': [],
            'monitoring_config': {
                'safety_threshold': 0.8,
                'performance_threshold': 0.85,
                'rollback_conditions': [
                    'safety_score < 0.8',
                    'error_rate > 0.05',
                    'user_satisfaction < 0.7'
                ]
            },
            'rollback_procedures': {
                'automatic_rollback': True,
                'rollback_delay': 300,  # 5 minutes
                'notification_channels': ['email', 'slack', 'pagerduty']
            }
        }
        
        for i, (version, percentage) in enumerate(zip(model_versions, rollout_percentages)):
            stage = {
                'stage_number': i + 1,
                'model_version': version,
                'traffic_percentage': percentage,
                'duration_hours': 24 if percentage < 100 else None,
                'success_criteria': {
                    'min_safety_score': 0.8,
                    'max_error_rate': 0.02,
                    'min_user_satisfaction': 0.75
                },
                'monitoring_frequency': 'every_5_minutes' if percentage < 50 else 'every_minute'
            }
            rollout_plan['stages'].append(stage)
        
        return rollout_plan
    
    def setup_realtime_monitoring(self, monitoring_config: Dict[str, Any]):
        """Setup real-time alignment monitoring"""
        monitoring_system = {
            'system_id': self.generate_id(),
            'config': monitoring_config,
            'alerts': {
                'safety_threshold': monitoring_config.get('safety_threshold', 0.8),
                'response_time_limit': monitoring_config.get('response_time_limit', 2.0),
                'error_rate_threshold': monitoring_config.get('error_rate_threshold', 0.05)
            },
            'notification_channels': monitoring_config.get('alert_channels', ['email']),
            'monitoring_frequency': monitoring_config.get('frequency', 60),  # seconds
            'active': True,
            'created_at': time.time()
        }
        
        self.monitoring_systems.append(monitoring_system)
        
        print(f"Real-time monitoring system {monitoring_system['system_id']} configured")
        return monitoring_system
    
    def implement_safety_filters(self, filter_configs: List[Dict[str, Any]]):
        """Implement production safety filters"""
        safety_filters = []
        
        for config in filter_configs:
            filter_system = {
                'filter_id': self.generate_id(),
                'type': config['type'],
                'threshold': config['threshold'],
                'action': config.get('action', 'block'),  # block, flag, or warn
                'bypass_conditions': config.get('bypass_conditions', []),
                'active': True,
                'created_at': time.time()
            }
            
            # Configure specific filter logic
            if config['type'] == 'toxicity':
                filter_system['classifier'] = 'toxicity_classifier_v2'
                filter_system['block_threshold'] = config['threshold']
            elif config['type'] == 'bias':
                filter_system['classifier'] = 'bias_detector_v1'
                filter_system['flag_threshold'] = config['threshold']
            elif config['type'] == 'factuality':
                filter_system['classifier'] = 'fact_checker_v1'
                filter_system['confidence_threshold'] = config['threshold']
            
            safety_filters.append(filter_system)
            print(f"Safety filter {filter_system['filter_id']} implemented for {config['type']}")
        
        return safety_filters
    
    def create_feedback_collection_system(self) -> Dict[str, Any]:
        """Create system for collecting user feedback on alignment"""
        feedback_system = {
            'system_id': self.generate_id(),
            'collection_methods': {
                'thumbs_up_down': {
                    'enabled': True,
                    'weight': 1.0,
                    'required_explanation': False
                },
                'detailed_rating': {
                    'enabled': True,
                    'criteria': ['helpfulness', 'accuracy', 'safety', 'clarity'],
                    'scale': '1-5',
                    'weight': 2.0
                },
                'free_text_feedback': {
                    'enabled': True,
                    'max_length': 500,
                    'weight': 1.5,
                    'sentiment_analysis': True
                }
            },
            'sampling_strategy': {
                'sample_rate': 0.1,  # 10% of interactions
                'prioritize_edge_cases': True,
                'balance_positive_negative': True
            },
            'analysis_pipeline': {
                'real_time_processing': True,
                'batch_analysis_frequency': 'daily',
                'trend_detection': True,
                'alert_thresholds': {
                    'negative_feedback_spike': 0.3,
                    'safety_concern_rate': 0.05
                }
            },
            'privacy_protection': {
                'anonymize_feedback': True,
                'data_retention_days': 90,
                'consent_required': True
            },
            'created_at': time.time()
        }
        
        print(f"Feedback collection system {feedback_system['system_id']} created")
        return feedback_system
    
    def execute_rollout_stage(self, rollout_plan: Dict[str, Any], stage_number: int) -> Dict[str, Any]:
        """Execute a specific stage of the rollout plan"""
        if stage_number > len(rollout_plan['stages']):
            raise ValueError(f"Stage {stage_number} does not exist in rollout plan")
        
        stage = rollout_plan['stages'][stage_number - 1]
        
        execution_result = {
            'stage_number': stage_number,
            'model_version': stage['model_version'],
            'traffic_percentage': stage['traffic_percentage'],
            'started_at': time.time(),
            'status': 'in_progress',
            'metrics': {
                'safety_score': 0.0,
                'error_rate': 0.0,
                'user_satisfaction': 0.0,
                'response_time': 0.0
            },
            'issues': [],
            'rollback_triggered': False
        }
        
        # Simulate rollout execution
        print(f"Executing rollout stage {stage_number}: {stage['model_version']} at {stage['traffic_percentage']}% traffic")
        
        # Mock metrics (in production, these would come from real monitoring)
        execution_result['metrics'] = {
            'safety_score': random.uniform(0.75, 0.95),
            'error_rate': random.uniform(0.001, 0.03),
            'user_satisfaction': random.uniform(0.7, 0.9),
            'response_time': random.uniform(0.5, 2.5)
        }
        
        # Check success criteria
        success_criteria = stage['success_criteria']
        criteria_met = (
            execution_result['metrics']['safety_score'] >= success_criteria['min_safety_score'] and
            execution_result['metrics']['error_rate'] <= success_criteria['max_error_rate'] and
            execution_result['metrics']['user_satisfaction'] >= success_criteria['min_user_satisfaction']
        )
        
        if criteria_met:
            execution_result['status'] = 'success'
            print(f"Stage {stage_number} completed successfully")
        else:
            execution_result['status'] = 'failed'
            execution_result['rollback_triggered'] = True
            print(f"Stage {stage_number} failed success criteria, triggering rollback")
        
        self.rollout_history.append(execution_result)
        return execution_result

def exercise_7_deployment_strategies():
    """Exercise 7: Production deployment strategies"""
    print("=== Exercise 7: Production Deployment Strategies ===")
    
    deployment_manager = AlignmentDeploymentManager()
    
    # Create staged rollout plan
    model_versions = ['aligned_v1.0', 'aligned_v1.1', 'aligned_v1.2']
    rollout_percentages = [10, 50, 100]
    rollout_plan = deployment_manager.create_staged_rollout_plan(model_versions, rollout_percentages)
    print(f"Created rollout plan with {len(rollout_plan['stages'])} stages")
    print(f"Plan ID: {rollout_plan['plan_id']}")
    
    # Display rollout stages
    for stage in rollout_plan['stages']:
        print(f"  Stage {stage['stage_number']}: {stage['model_version']} ({stage['traffic_percentage']}% traffic)")
    
    # Setup monitoring
    monitoring_config = {
        'safety_threshold': 0.8,
        'response_time_limit': 2.0,
        'alert_channels': ['email', 'slack'],
        'frequency': 30
    }
    monitoring_system = deployment_manager.setup_realtime_monitoring(monitoring_config)
    print(f"\nMonitoring system configured: {monitoring_system['system_id']}")
    
    # Implement safety filters
    filter_configs = [
        {'type': 'toxicity', 'threshold': 0.7, 'action': 'block'},
        {'type': 'bias', 'threshold': 0.6, 'action': 'flag'},
        {'type': 'factuality', 'threshold': 0.8, 'action': 'warn'}
    ]
    safety_filters = deployment_manager.implement_safety_filters(filter_configs)
    print(f"\nImplemented {len(safety_filters)} safety filters")
    
    # Create feedback system
    feedback_system = deployment_manager.create_feedback_collection_system()
    print(f"\nFeedback collection system created: {feedback_system['system_id']}")
    print(f"Collection methods: {list(feedback_system['collection_methods'].keys())}")
    print(f"Sampling rate: {feedback_system['sampling_strategy']['sample_rate']*100}%")
    
    # Execute first rollout stage
    print(f"\nExecuting rollout stage 1...")
    execution_result = deployment_manager.execute_rollout_stage(rollout_plan, 1)
    print(f"Stage 1 status: {execution_result['status']}")
    print(f"Safety score: {execution_result['metrics']['safety_score']:.2f}")
    print(f"Error rate: {execution_result['metrics']['error_rate']:.3f}")
    
    if execution_result['rollback_triggered']:
        print("⚠️  Rollback triggered due to failed success criteria")
    else:
        print("✅ Stage 1 completed successfully")
    
    print("✅ Deployment strategies completed successfully!")
    print()

def main():
    """Run all RLHF and DPO exercises"""
    print("Day 49: RLHF and DPO - Human Feedback & Preference Learning - Exercises")
    print("=" * 80)
    print()
    
    # Run all exercises
    exercise_1_sft_preparation()
    exercise_2_reward_model()
    exercise_3_dpo_implementation()
    exercise_4_preference_collection()
    exercise_5_constitutional_ai()
    exercise_6_alignment_monitoring()
    exercise_7_deployment_strategies()
    
    print("=" * 80)
    print("RLHF and DPO exercises completed! Check the solution.py file for complete implementations.")
    print()
    print("Next steps:")
    print("1. Implement the TODO sections in each exercise")
    print("2. Test with real language models and preference data")
    print("3. Experiment with different alignment techniques")
    print("4. Deploy aligned models with proper monitoring")
    print("5. Collect and analyze human feedback for continuous improvement")

if __name__ == "__main__":
    main()
