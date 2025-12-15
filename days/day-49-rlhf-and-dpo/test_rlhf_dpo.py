"""
Day 49: RLHF and DPO - Human Feedback & Preference Learning - Comprehensive Test Suite

This test suite validates all RLHF and DPO implementations including:
- SFT data preparation
- Reward model training
- DPO implementation
- Human preference collection
- Constitutional AI
- Alignment monitoring
- Production deployment
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import time
from unittest.mock import Mock, patch
import sys
import os

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from solution import (
    SFTDataPreparator, RewardModel, RewardModelTrainer,
    DPOTrainer, PreferenceCollectionSystem, ConstitutionalAI,
    AlignmentMonitor, AlignmentDeploymentManager,
    MockLanguageModel, MockTokenizer
)

class TestSFTDataPreparation:
    """Test SFT data preparation functionality"""
    
    def test_sft_preparator_initialization(self):
        """Test SFT data preparator initialization"""
        tokenizer = MockTokenizer()
        preparator = SFTDataPreparator(tokenizer)
        
        assert preparator.tokenizer == tokenizer
    
    def test_instruction_data_preparation(self):
        """Test instruction data preparation"""
        tokenizer = MockTokenizer()
        preparator = SFTDataPreparator(tokenizer)
        
        demonstrations = [
            {
                "instruction": "What is AI?",
                "response": "AI is artificial intelligence."
            },
            {
                "instruction": "Explain machine learning",
                "response": "Machine learning is a subset of AI."
            }
        ]
        
        formatted_data = preparator.prepare_instruction_data(demonstrations)
        
        assert len(formatted_data) == 2
        for item in formatted_data:
            assert 'input_ids' in item
            assert 'labels' in item
            assert 'instruction' in item
            assert 'response' in item
            assert isinstance(item['input_ids'], torch.Tensor)
            assert isinstance(item['labels'], torch.Tensor)
    
    def test_conversation_format_creation(self):
        """Test conversation format creation"""
        tokenizer = MockTokenizer()
        preparator = SFTDataPreparator(tokenizer)
        
        conversations = [
            [
                {"role": "human", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "human", "content": "How are you?"},
                {"role": "assistant", "content": "I'm doing well, thank you!"}
            ]
        ]
        
        formatted_conversations = preparator.create_conversation_format(conversations)
        
        assert len(formatted_conversations) == 1
        item = formatted_conversations[0]
        assert 'input_ids' in item
        assert 'labels' in item
        assert 'conversation' in item
        assert isinstance(item['input_ids'], torch.Tensor)
        assert isinstance(item['labels'], torch.Tensor)

class TestRewardModel:
    """Test reward model implementation"""
    
    def test_reward_model_initialization(self):
        """Test reward model initialization"""
        base_model = MockLanguageModel()
        reward_model = RewardModel(base_model)
        
        assert reward_model.base_model == base_model
        assert hasattr(reward_model, 'reward_head')
        
        # Check that base model parameters are frozen
        for param in reward_model.base_model.parameters():
            assert not param.requires_grad
    
    def test_reward_model_forward(self):
        """Test reward model forward pass"""
        base_model = MockLanguageModel()
        reward_model = RewardModel(base_model)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        rewards = reward_model(input_ids, attention_mask)
        
        assert rewards.shape == (batch_size, 1)
        assert isinstance(rewards, torch.Tensor)
    
    def test_reward_model_trainer_initialization(self):
        """Test reward model trainer initialization"""
        base_model = MockLanguageModel()
        reward_model = RewardModel(base_model)
        tokenizer = MockTokenizer()
        
        trainer = RewardModelTrainer(reward_model, tokenizer)
        
        assert trainer.reward_model == reward_model
        assert trainer.tokenizer == tokenizer
    
    def test_preference_data_preparation(self):
        """Test preference data preparation"""
        base_model = MockLanguageModel()
        reward_model = RewardModel(base_model)
        tokenizer = MockTokenizer()
        trainer = RewardModelTrainer(reward_model, tokenizer)
        
        preference_pairs = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence, a field of computer science.",
                "rejected": "AI is just computers."
            }
        ]
        
        training_data = trainer.prepare_preference_data(preference_pairs)
        
        assert len(training_data) == 1
        item = training_data[0]
        assert 'chosen_input_ids' in item
        assert 'rejected_input_ids' in item
        assert 'prompt' in item
        assert isinstance(item['chosen_input_ids'], torch.Tensor)
        assert isinstance(item['rejected_input_ids'], torch.Tensor)
    
    def test_reward_loss_computation(self):
        """Test reward loss computation"""
        base_model = MockLanguageModel()
        reward_model = RewardModel(base_model)
        tokenizer = MockTokenizer()
        trainer = RewardModelTrainer(reward_model, tokenizer)
        
        chosen_rewards = torch.tensor([[0.8], [0.9]])
        rejected_rewards = torch.tensor([[0.3], [0.4]])
        
        loss = trainer.compute_reward_loss(chosen_rewards, rejected_rewards)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert loss.item() >= 0  # Loss should be non-negative

class TestDPOTrainer:
    """Test DPO trainer implementation"""
    
    def test_dpo_trainer_initialization(self):
        """Test DPO trainer initialization"""
        model = MockLanguageModel()
        ref_model = MockLanguageModel()
        tokenizer = MockTokenizer()
        
        dpo_trainer = DPOTrainer(model, ref_model, tokenizer, beta=0.1)
        
        assert dpo_trainer.model == model
        assert dpo_trainer.ref_model == ref_model
        assert dpo_trainer.tokenizer == tokenizer
        assert dpo_trainer.beta == 0.1
        
        # Check that reference model parameters are frozen
        for param in dpo_trainer.ref_model.parameters():
            assert not param.requires_grad
    
    def test_dpo_loss_computation(self):
        """Test DPO loss computation"""
        model = MockLanguageModel()
        ref_model = MockLanguageModel()
        tokenizer = MockTokenizer()
        dpo_trainer = DPOTrainer(model, ref_model, tokenizer)
        
        policy_chosen_logps = torch.tensor([2.0, 1.8])
        policy_rejected_logps = torch.tensor([1.0, 0.8])
        reference_chosen_logps = torch.tensor([1.5, 1.3])
        reference_rejected_logps = torch.tensor([1.2, 1.0])
        
        loss, accuracy = dpo_trainer.compute_dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            reference_chosen_logps, reference_rejected_logps
        )
        
        assert isinstance(loss, torch.Tensor)
        assert isinstance(accuracy, torch.Tensor)
        assert loss.dim() == 0
        assert 0 <= accuracy.item() <= 1
    
    def test_batch_logps_computation(self):
        """Test batch log probabilities computation"""
        model = MockLanguageModel()
        ref_model = MockLanguageModel()
        tokenizer = MockTokenizer()
        dpo_trainer = DPOTrainer(model, ref_model, tokenizer)
        
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        labels = input_ids.clone()
        
        logps = dpo_trainer.get_batch_logps(model, input_ids, labels)
        
        assert logps.shape == (batch_size,)
        assert isinstance(logps, torch.Tensor)
    
    def test_dpo_data_preparation(self):
        """Test DPO data preparation"""
        model = MockLanguageModel()
        ref_model = MockLanguageModel()
        tokenizer = MockTokenizer()
        dpo_trainer = DPOTrainer(model, ref_model, tokenizer)
        
        preference_pairs = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "AI is just computers."
            }
        ]
        
        dpo_data = dpo_trainer.prepare_dpo_data(preference_pairs)
        
        assert len(dpo_data) == 1
        item = dpo_data[0]
        assert 'chosen_input_ids' in item
        assert 'chosen_labels' in item
        assert 'rejected_input_ids' in item
        assert 'rejected_labels' in item

class TestPreferenceCollection:
    """Test human preference collection system"""
    
    def test_preference_system_initialization(self):
        """Test preference collection system initialization"""
        system = PreferenceCollectionSystem()
        
        assert hasattr(system, 'annotation_guidelines')
        assert hasattr(system, 'quality_checks')
        assert len(system.annotation_guidelines) > 0
        assert len(system.quality_checks) > 0
    
    def test_annotation_task_creation(self):
        """Test annotation task creation"""
        system = PreferenceCollectionSystem()
        
        prompt = "What is machine learning?"
        responses = ["ML is AI", "ML is a subset of AI"]
        criteria = ["accuracy", "clarity"]
        
        task = system.create_annotation_task(prompt, responses, criteria)
        
        assert 'task_id' in task
        assert task['prompt'] == prompt
        assert task['responses'] == responses
        assert task['criteria'] == criteria
        assert 'guidelines' in task
        assert task['status'] == 'pending'
    
    def test_annotation_validation(self):
        """Test annotation validation"""
        system = PreferenceCollectionSystem()
        
        # Valid annotation
        valid_annotation = {
            'preferences': [1, 0, 1],
            'explanations': ['Good explanation', 'Clear reasoning', 'Helpful response'],
            'annotation_time': 120,
            'ratings': [4, 2, 5]
        }
        
        validation = system.validate_annotations(valid_annotation)
        assert validation['valid'] == True
        assert validation['confidence_score'] > 0.7
        
        # Invalid annotation (too fast)
        invalid_annotation = {
            'preferences': [1, 1, 1],
            'explanations': ['ok', 'good', 'fine'],
            'annotation_time': 10,
            'ratings': [5, 5, 5]
        }
        
        validation = system.validate_annotations(invalid_annotation)
        assert validation['valid'] == False
        assert len(validation['issues']) > 0
    
    def test_inter_annotator_agreement(self):
        """Test inter-annotator agreement computation"""
        system = PreferenceCollectionSystem()
        
        annotations = [
            {'preferences': [1, 0, 1]},
            {'preferences': [1, 0, 1]},
            {'preferences': [1, 1, 0]}
        ]
        
        agreement = system.compute_inter_annotator_agreement(annotations)
        
        assert 'agreement' in agreement
        assert 'method' in agreement
        assert 0 <= agreement['agreement'] <= 1
    
    def test_annotation_report_generation(self):
        """Test annotation report generation"""
        system = PreferenceCollectionSystem()
        
        annotations = [
            {
                'preferences': [1, 0],
                'ratings': [4, 2],
                'annotation_time': 120
            },
            {
                'preferences': [1, 1],
                'ratings': [5, 4],
                'annotation_time': 90
            }
        ]
        
        report = system.generate_annotation_report(annotations)
        
        assert 'summary' in report
        assert 'agreement_metrics' in report
        assert report['summary']['total_annotations'] == 2

class TestConstitutionalAI:
    """Test Constitutional AI implementation"""
    
    def test_constitutional_ai_initialization(self):
        """Test Constitutional AI initialization"""
        model = MockLanguageModel()
        tokenizer = MockTokenizer()
        constitution = ["Be helpful", "Be harmless", "Be honest"]
        
        constitutional_ai = ConstitutionalAI(model, tokenizer, constitution)
        
        assert constitutional_ai.model == model
        assert constitutional_ai.tokenizer == tokenizer
        assert constitutional_ai.constitution == constitution
    
    def test_response_critique(self):
        """Test response critique generation"""
        model = MockLanguageModel()
        tokenizer = MockTokenizer()
        constitution = ["Be helpful", "Be harmless"]
        
        constitutional_ai = ConstitutionalAI(model, tokenizer, constitution)
        
        prompt = "How to cook pasta?"
        response = "Boil water and add pasta."
        
        critique = constitutional_ai.critique_response(prompt, response)
        
        assert isinstance(critique, str)
        assert len(critique) > 0
    
    def test_response_revision(self):
        """Test response revision"""
        model = MockLanguageModel()
        tokenizer = MockTokenizer()
        constitution = ["Be helpful", "Be harmless"]
        
        constitutional_ai = ConstitutionalAI(model, tokenizer, constitution)
        
        prompt = "How to cook pasta?"
        response = "Boil water."
        critique = "Response is too brief."
        
        revised_response = constitutional_ai.revise_response(prompt, response, critique)
        
        assert isinstance(revised_response, str)
        assert len(revised_response) > 0
    
    def test_constitutional_training_step(self):
        """Test constitutional training step"""
        model = MockLanguageModel()
        tokenizer = MockTokenizer()
        constitution = ["Be helpful", "Be harmless"]
        
        constitutional_ai = ConstitutionalAI(model, tokenizer, constitution)
        
        prompts = ["What is AI?", "How to learn programming?"]
        
        training_data = constitutional_ai.constitutional_training_step(prompts)
        
        assert len(training_data) == 2
        for item in training_data:
            assert 'prompt' in item
            assert 'chosen' in item
            assert 'rejected' in item
            assert 'critique' in item
    
    def test_constitutional_adherence_evaluation(self):
        """Test constitutional adherence evaluation"""
        model = MockLanguageModel()
        tokenizer = MockTokenizer()
        constitution = ["Be helpful", "Be harmless", "Be honest"]
        
        constitutional_ai = ConstitutionalAI(model, tokenizer, constitution)
        
        responses = [
            "I can help you with that task.",
            "I'm not sure about that information.",
            "Here's a safe approach to your question."
        ]
        
        adherence_scores = constitutional_ai.evaluate_constitutional_adherence(responses)
        
        assert 'overall_adherence' in adherence_scores
        assert 0 <= adherence_scores['overall_adherence'] <= 1

class TestAlignmentMonitoring:
    """Test alignment monitoring functionality"""
    
    def test_alignment_monitor_initialization(self):
        """Test alignment monitor initialization"""
        monitor = AlignmentMonitor()
        
        assert hasattr(monitor, 'safety_classifiers')
        assert hasattr(monitor, 'alignment_metrics')
        assert len(monitor.safety_classifiers) > 0
    
    def test_safety_classifiers_loading(self):
        """Test safety classifiers loading"""
        monitor = AlignmentMonitor()
        
        # Test each classifier
        test_text = "This is a test response."
        
        for classifier_name, classifier in monitor.safety_classifiers.items():
            result = classifier(test_text)
            assert isinstance(result, dict)
            assert 'score' in result or 'confidence' in result
    
    def test_response_safety_evaluation(self):
        """Test response safety evaluation"""
        monitor = AlignmentMonitor()
        
        response = "I'd be happy to help you with that task."
        
        safety_eval = monitor.evaluate_response_safety(response)
        
        assert 'safety_score' in safety_eval
        assert 'details' in safety_eval
        assert 'safe' in safety_eval
        assert 'timestamp' in safety_eval
        assert 0 <= safety_eval['safety_score'] <= 1
        assert isinstance(safety_eval['safe'], bool)
    
    def test_alignment_drift_monitoring(self):
        """Test alignment drift monitoring"""
        monitor = AlignmentMonitor()
        
        recent_responses = [
            "I can help with that.",
            "Here's the information you need.",
            "I'm not sure about that."
        ]
        
        baseline_metrics = {
            'avg_safety_score': 0.9,
            'helpfulness_rate': 0.8,
            'toxicity_rate': 0.05
        }
        
        drift_analysis = monitor.monitor_alignment_drift(recent_responses, baseline_metrics)
        
        assert 'overall_assessment' in drift_analysis
        for metric in baseline_metrics.keys():
            assert metric in drift_analysis
            assert 'baseline' in drift_analysis[metric]
            assert 'current' in drift_analysis[metric]
            assert 'drift_percentage' in drift_analysis[metric]
    
    def test_alignment_metrics_computation(self):
        """Test alignment metrics computation"""
        monitor = AlignmentMonitor()
        
        responses = [
            "I can help you with that.",
            "That's a great question.",
            "I'm not certain about that."
        ]
        
        metrics = monitor.compute_alignment_metrics(responses)
        
        expected_metrics = [
            'avg_safety_score', 'toxicity_rate', 'bias_rate',
            'factuality_rate', 'helpfulness_rate'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert 0 <= metrics[metric] <= 1
    
    def test_alignment_report_generation(self):
        """Test alignment report generation"""
        monitor = AlignmentMonitor()
        
        responses = [
            "I can help you with that.",
            "Here's some information.",
            "I'm not sure about that."
        ]
        
        report = monitor.generate_alignment_report(responses)
        
        assert 'summary' in report
        assert 'metrics' in report
        assert 'recommendations' in report
        assert 'generated_at' in report
        assert report['summary']['total_responses'] == 3

class TestDeploymentStrategies:
    """Test deployment strategies functionality"""
    
    def test_deployment_manager_initialization(self):
        """Test deployment manager initialization"""
        manager = AlignmentDeploymentManager()
        
        assert hasattr(manager, 'deployment_configs')
        assert hasattr(manager, 'monitoring_systems')
        assert hasattr(manager, 'rollout_history')
    
    def test_staged_rollout_plan_creation(self):
        """Test staged rollout plan creation"""
        manager = AlignmentDeploymentManager()
        
        model_versions = ['v1.0', 'v1.1', 'v1.2']
        rollout_percentages = [10, 50, 100]
        
        rollout_plan = manager.create_staged_rollout_plan(model_versions, rollout_percentages)
        
        assert 'plan_id' in rollout_plan
        assert 'stages' in rollout_plan
        assert 'monitoring_config' in rollout_plan
        assert len(rollout_plan['stages']) == 3
        
        for i, stage in enumerate(rollout_plan['stages']):
            assert stage['model_version'] == model_versions[i]
            assert stage['traffic_percentage'] == rollout_percentages[i]
    
    def test_realtime_monitoring_setup(self):
        """Test real-time monitoring setup"""
        manager = AlignmentDeploymentManager()
        
        monitoring_config = {
            'safety_threshold': 0.8,
            'response_time_limit': 2.0,
            'alert_channels': ['email', 'slack']
        }
        
        monitoring_system = manager.setup_realtime_monitoring(monitoring_config)
        
        assert 'system_id' in monitoring_system
        assert 'config' in monitoring_system
        assert 'alerts' in monitoring_system
        assert monitoring_system['active'] == True
    
    def test_safety_filters_implementation(self):
        """Test safety filters implementation"""
        manager = AlignmentDeploymentManager()
        
        filter_configs = [
            {'type': 'toxicity', 'threshold': 0.7},
            {'type': 'bias', 'threshold': 0.6}
        ]
        
        safety_filters = manager.implement_safety_filters(filter_configs)
        
        assert len(safety_filters) == 2
        for filter_system in safety_filters:
            assert 'filter_id' in filter_system
            assert 'type' in filter_system
            assert 'threshold' in filter_system
            assert filter_system['active'] == True
    
    def test_feedback_collection_system_creation(self):
        """Test feedback collection system creation"""
        manager = AlignmentDeploymentManager()
        
        feedback_system = manager.create_feedback_collection_system()
        
        assert 'system_id' in feedback_system
        assert 'collection_methods' in feedback_system
        assert 'sampling_strategy' in feedback_system
        assert 'analysis_pipeline' in feedback_system
        assert 'privacy_protection' in feedback_system
    
    def test_rollout_stage_execution(self):
        """Test rollout stage execution"""
        manager = AlignmentDeploymentManager()
        
        model_versions = ['v1.0', 'v1.1']
        rollout_percentages = [50, 100]
        rollout_plan = manager.create_staged_rollout_plan(model_versions, rollout_percentages)
        
        execution_result = manager.execute_rollout_stage(rollout_plan, 1)
        
        assert 'stage_number' in execution_result
        assert 'model_version' in execution_result
        assert 'status' in execution_result
        assert 'metrics' in execution_result
        assert execution_result['stage_number'] == 1
        assert execution_result['model_version'] == 'v1.0'

class TestIntegration:
    """Integration tests for complete RLHF/DPO workflow"""
    
    def test_end_to_end_rlhf_workflow(self):
        """Test complete RLHF workflow"""
        # 1. Prepare SFT data
        tokenizer = MockTokenizer()
        preparator = SFTDataPreparator(tokenizer)
        
        demonstrations = [
            {"instruction": "What is AI?", "response": "AI is artificial intelligence."}
        ]
        sft_data = preparator.prepare_instruction_data(demonstrations)
        
        # 2. Train reward model
        base_model = MockLanguageModel()
        reward_model = RewardModel(base_model)
        trainer = RewardModelTrainer(reward_model, tokenizer)
        
        preference_pairs = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "AI is computers."
            }
        ]
        training_data = trainer.prepare_preference_data(preference_pairs)
        
        # 3. Monitor alignment
        monitor = AlignmentMonitor()
        responses = ["AI is artificial intelligence."]
        metrics = monitor.compute_alignment_metrics(responses)
        
        # Verify all components work together
        assert len(sft_data) > 0
        assert len(training_data) > 0
        assert len(metrics) > 0
    
    def test_end_to_end_dpo_workflow(self):
        """Test complete DPO workflow"""
        # 1. Setup DPO trainer
        model = MockLanguageModel()
        ref_model = MockLanguageModel()
        tokenizer = MockTokenizer()
        dpo_trainer = DPOTrainer(model, ref_model, tokenizer)
        
        # 2. Prepare preference data
        preference_pairs = [
            {
                "prompt": "What is AI?",
                "chosen": "AI is artificial intelligence.",
                "rejected": "AI is computers."
            }
        ]
        dpo_data = dpo_trainer.prepare_dpo_data(preference_pairs)
        
        # 3. Constitutional AI
        constitution = ["Be helpful", "Be harmless"]
        constitutional_ai = ConstitutionalAI(model, tokenizer, constitution)
        
        # 4. Deployment
        deployment_manager = AlignmentDeploymentManager()
        rollout_plan = deployment_manager.create_staged_rollout_plan(['v1.0'], [100])
        
        # Verify all components work together
        assert len(dpo_data) > 0
        assert len(constitutional_ai.constitution) > 0
        assert len(rollout_plan['stages']) > 0

def run_tests():
    """Run all tests with pytest"""
    pytest.main([__file__, "-v", "--tb=short"])

if __name__ == "__main__":
    run_tests()