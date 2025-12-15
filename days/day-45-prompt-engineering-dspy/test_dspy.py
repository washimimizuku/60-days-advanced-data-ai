"""
Day 45: Prompt Engineering with DSPy - Comprehensive Test Suite

Tests for DSPy implementations including signatures, modules, optimizers,
and production patterns.
"""

import pytest
import dspy
from unittest.mock import Mock, patch
import time
import hashlib
from typing import Dict, Any

# Import solutions for testing
from solution import (
    SimpleQAModule, SentimentAnalyzer, MultiStageQA, SelfCorrectingQA,
    EnsembleQA, ProductionQA, accuracy_metric, semantic_similarity_metric,
    comprehensive_metric
)

class TestMockLM:
    """Test the mock language model setup"""
    
    def test_mock_lm_initialization(self):
        """Test mock LM can be initialized"""
        from solution import MockLM
        lm = MockLM()
        assert lm.call_count == 0
        assert isinstance(lm.responses, dict)
    
    def test_mock_lm_responses(self):
        """Test mock LM returns appropriate responses"""
        from solution import MockLM
        lm = MockLM()
        
        response = lm("What is the capital of France?")
        assert "Paris" in response
        assert lm.call_count == 1

class TestBasicSignatures:
    """Test basic DSPy signature implementations"""
    
    def test_basic_qa_signature(self):
        """Test BasicQA signature structure"""
        from solution import BasicQA
        
        # Check signature has required fields
        assert hasattr(BasicQA, 'context')
        assert hasattr(BasicQA, 'question') 
        assert hasattr(BasicQA, 'answer')
    
    def test_sentiment_analysis_signature(self):
        """Test SentimentAnalysis signature structure"""
        from solution import SentimentAnalysis
        
        # Check signature has required fields
        assert hasattr(SentimentAnalysis, 'text')
        assert hasattr(SentimentAnalysis, 'sentiment')
        assert hasattr(SentimentAnalysis, 'confidence')
        assert hasattr(SentimentAnalysis, 'reasoning')

class TestSimpleQAModule:
    """Test SimpleQAModule implementation"""
    
    def test_module_initialization(self):
        """Test SimpleQAModule can be initialized"""
        module = SimpleQAModule()
        assert hasattr(module, 'generate_answer')
    
    def test_module_forward(self):
        """Test SimpleQAModule forward method"""
        module = SimpleQAModule()
        
        context = "France is in Europe. Paris is its capital."
        question = "What is the capital of France?"
        
        result = module(context=context, question=question)
        assert hasattr(result, 'answer')
        assert isinstance(result.answer, str)

class TestSentimentAnalyzer:
    """Test SentimentAnalyzer implementation"""
    
    def test_analyzer_initialization(self):
        """Test SentimentAnalyzer can be initialized"""
        analyzer = SentimentAnalyzer()
        assert hasattr(analyzer, 'analyze')
    
    def test_analyzer_forward(self):
        """Test SentimentAnalyzer forward method"""
        analyzer = SentimentAnalyzer()
        
        text = "I love this product!"
        result = analyzer(text=text)
        
        assert hasattr(result, 'sentiment')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'reasoning')

class TestMultiStageQA:
    """Test MultiStageQA implementation"""
    
    def test_multistage_initialization(self):
        """Test MultiStageQA can be initialized"""
        module = MultiStageQA()
        assert hasattr(module, 'extract_facts')
        assert hasattr(module, 'analyze_question')
        assert hasattr(module, 'reason')
        assert hasattr(module, 'generate_answer')
    
    def test_multistage_forward(self):
        """Test MultiStageQA forward method"""
        module = MultiStageQA()
        
        context = "The solar system has eight planets. Mercury is closest to the Sun."
        question = "Which planet is closest to the Sun?"
        
        result = module(context=context, question=question)
        
        assert hasattr(result, 'facts')
        assert hasattr(result, 'analysis')
        assert hasattr(result, 'reasoning')
        assert hasattr(result, 'answer')

class TestSelfCorrectingQA:
    """Test SelfCorrectingQA implementation"""
    
    def test_self_correcting_initialization(self):
        """Test SelfCorrectingQA can be initialized"""
        module = SelfCorrectingQA()
        assert hasattr(module, 'generate')
        assert hasattr(module, 'validate')
        assert hasattr(module, 'correct')
    
    def test_self_correcting_forward(self):
        """Test SelfCorrectingQA forward method"""
        module = SelfCorrectingQA()
        
        context = "Water boils at 100 degrees Celsius."
        question = "At what temperature does water boil?"
        
        result = module(context=context, question=question, max_attempts=2)
        
        assert hasattr(result, 'answer')
        assert hasattr(result, 'attempts')
        assert hasattr(result, 'final_validation')

class TestEnsembleQA:
    """Test EnsembleQA implementation"""
    
    def test_ensemble_initialization(self):
        """Test EnsembleQA can be initialized"""
        module = EnsembleQA()
        assert hasattr(module, 'direct_qa')
        assert hasattr(module, 'cot_qa')
        assert hasattr(module, 'multistage_qa')
        assert hasattr(module, 'aggregate')
    
    def test_ensemble_forward(self):
        """Test EnsembleQA forward method"""
        module = EnsembleQA()
        
        context = "Python was created by Guido van Rossum in 1991."
        question = "Who created Python?"
        
        result = module(context=context, question=question)
        
        assert hasattr(result, 'direct_answer')
        assert hasattr(result, 'cot_answer')
        assert hasattr(result, 'multistage_answer')
        assert hasattr(result, 'final_answer')
        assert hasattr(result, 'reasoning')

class TestProductionQA:
    """Test ProductionQA implementation"""
    
    def test_production_initialization(self):
        """Test ProductionQA can be initialized"""
        module = ProductionQA()
        assert hasattr(module, 'primary_qa')
        assert hasattr(module, 'fallback_qa')
        assert hasattr(module, 'cache')
        assert isinstance(module.cache, dict)
    
    def test_cache_key_generation(self):
        """Test cache key generation"""
        module = ProductionQA()
        
        context = "Test context"
        question = "Test question"
        
        key1 = module._cache_key(context, question)
        key2 = module._cache_key(context, question)
        key3 = module._cache_key("Different context", question)
        
        assert key1 == key2  # Same inputs should generate same key
        assert key1 != key3  # Different inputs should generate different keys
        assert len(key1) == 32  # MD5 hash length
    
    def test_caching_functionality(self):
        """Test caching works correctly"""
        module = ProductionQA()
        
        context = "Test context"
        question = "Test question"
        
        # First call should not hit cache
        result1 = module(context=context, question=question)
        assert len(module.cache) == 1
        
        # Second call should hit cache
        result2 = module(context=context, question=question)
        assert result1.answer == result2.answer
    
    def test_error_handling(self):
        """Test error handling in production system"""
        module = ProductionQA()
        
        # Test with empty context (should trigger error handling)
        result = module(context="", question="What is the meaning of life?")
        assert hasattr(result, 'answer')
        assert "apologize" in result.answer.lower() or "unable" in result.answer.lower()
    
    def test_statistics_tracking(self):
        """Test statistics tracking"""
        module = ProductionQA()
        
        # Make some requests
        module(context="Test", question="Test question 1")
        module(context="Test", question="Test question 2")
        
        stats = module.get_stats()
        
        assert 'total_requests' in stats
        assert 'successful_requests' in stats
        assert 'failed_requests' in stats
        assert 'success_rate' in stats
        assert 'cache_size' in stats
        assert 'average_response_time' in stats
        
        assert stats['total_requests'] >= 2

class TestCustomMetrics:
    """Test custom evaluation metrics"""
    
    def test_accuracy_metric_exact_match(self):
        """Test accuracy metric with exact match"""
        example = {'expected': 'Paris'}
        prediction = Mock()
        prediction.answer = 'Paris'
        
        score = accuracy_metric(example, prediction)
        assert score == 1.0
    
    def test_accuracy_metric_partial_match(self):
        """Test accuracy metric with partial match"""
        example = {'expected': 'Paris'}
        prediction = Mock()
        prediction.answer = 'Paris is the capital'
        
        score = accuracy_metric(example, prediction)
        assert 0.5 < score < 1.0
    
    def test_accuracy_metric_no_match(self):
        """Test accuracy metric with no match"""
        example = {'expected': 'Paris'}
        prediction = Mock()
        prediction.answer = 'London'
        
        score = accuracy_metric(example, prediction)
        assert score < 0.5
    
    def test_semantic_similarity_metric(self):
        """Test semantic similarity metric"""
        example = {'expected': 'Paris France'}
        prediction = Mock()
        prediction.answer = 'France Paris'
        
        score = semantic_similarity_metric(example, prediction)
        assert score > 0.5  # Should have high similarity due to word overlap
    
    def test_comprehensive_metric(self):
        """Test comprehensive metric combination"""
        example = {'expected': 'Paris'}
        prediction = Mock()
        prediction.answer = 'Paris is the capital of France'
        
        score = comprehensive_metric(example, prediction)
        assert 0.0 <= score <= 1.0

class TestPerformanceAndScaling:
    """Test performance and scaling aspects"""
    
    def test_response_time_tracking(self):
        """Test response time is tracked"""
        module = ProductionQA()
        
        start_time = time.time()
        module(context="Test context", question="Test question")
        end_time = time.time()
        
        stats = module.get_stats()
        assert stats['average_response_time'] > 0
        assert stats['average_response_time'] < (end_time - start_time) + 0.1  # Allow some overhead
    
    def test_cache_performance(self):
        """Test caching improves performance"""
        module = ProductionQA()
        
        context = "Test context"
        question = "Test question"
        
        # First call (no cache)
        start1 = time.time()
        module(context=context, question=question)
        time1 = time.time() - start1
        
        # Second call (with cache)
        start2 = time.time()
        module(context=context, question=question)
        time2 = time.time() - start2
        
        # Cached call should be faster (though with mock LM, difference might be minimal)
        assert time2 <= time1 + 0.01  # Allow small margin for timing variations

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_input_handling(self):
        """Test handling of empty inputs"""
        module = SimpleQAModule()
        
        result = module(context="", question="")
        assert hasattr(result, 'answer')
    
    def test_none_input_handling(self):
        """Test handling of None inputs"""
        module = ProductionQA()
        
        # Should handle gracefully without crashing
        try:
            result = module(context=None, question=None)
            assert hasattr(result, 'answer')
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            assert "answer" in str(e).lower() or "error" in str(e).lower()
    
    def test_very_long_input_handling(self):
        """Test handling of very long inputs"""
        module = ProductionQA()
        
        long_context = "This is a test. " * 1000  # Very long context
        question = "What is this about?"
        
        result = module(context=long_context, question=question)
        assert hasattr(result, 'answer')

class TestIntegration:
    """Integration tests for complete workflows"""
    
    def test_end_to_end_qa_workflow(self):
        """Test complete QA workflow"""
        # Test simple QA
        simple_qa = SimpleQAModule()
        result = simple_qa(
            context="The sky is blue due to light scattering.",
            question="Why is the sky blue?"
        )
        assert hasattr(result, 'answer')
        
        # Test production QA with same input
        production_qa = ProductionQA()
        result2 = production_qa(
            context="The sky is blue due to light scattering.",
            question="Why is the sky blue?"
        )
        assert hasattr(result2, 'answer')
    
    def test_ensemble_vs_simple_comparison(self):
        """Test ensemble provides different results than simple QA"""
        context = "Machine learning is a subset of artificial intelligence."
        question = "What is machine learning?"
        
        simple_qa = SimpleQAModule()
        ensemble_qa = EnsembleQA()
        
        simple_result = simple_qa(context=context, question=question)
        ensemble_result = ensemble_qa(context=context, question=question)
        
        assert hasattr(simple_result, 'answer')
        assert hasattr(ensemble_result, 'final_answer')
        # Results might be different due to ensemble aggregation
    
    def test_metrics_evaluation_workflow(self):
        """Test complete metrics evaluation workflow"""
        examples = [
            {'expected': 'Paris', 'question': 'Capital of France?'},
            {'expected': '4', 'question': 'What is 2+2?'}
        ]
        
        module = SimpleQAModule()
        
        total_score = 0
        for example in examples:
            prediction = module(
                context=f"Context for {example['question']}",
                question=example['question']
            )
            score = accuracy_metric(example, prediction)
            total_score += score
        
        avg_score = total_score / len(examples)
        assert 0 <= avg_score <= 1

def run_performance_benchmark():
    """Run performance benchmark for production system"""
    print("\n=== Performance Benchmark ===")
    
    module = ProductionQA()
    
    # Test multiple requests
    test_cases = [
        ("France is in Europe.", "Where is France?"),
        ("Python is a programming language.", "What is Python?"),
        ("Water boils at 100C.", "When does water boil?"),
    ] * 10  # Repeat for more data
    
    start_time = time.time()
    
    for context, question in test_cases:
        module(context=context, question=question)
    
    end_time = time.time()
    
    stats = module.get_stats()
    
    print(f"Total requests: {stats['total_requests']}")
    print(f"Success rate: {stats['success_rate']:.2%}")
    print(f"Cache size: {stats['cache_size']}")
    print(f"Average response time: {stats['average_response_time']:.3f}s")
    print(f"Total benchmark time: {end_time - start_time:.3f}s")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
    
    # Run performance benchmark
    run_performance_benchmark()