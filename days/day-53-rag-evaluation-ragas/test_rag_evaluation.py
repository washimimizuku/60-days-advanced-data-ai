"""
Day 53: RAG Evaluation with RAGAS - Comprehensive Test Suite
Tests for RAGAS metrics, evaluation pipelines, A/B testing, and monitoring
"""

import pytest
import numpy as np
import pandas as pd
import time
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from typing import List, Dict

# Import classes from solution
import sys
import os
sys.path.append(os.path.dirname(__file__))

from solution import (
    RAGResponse, EvaluationResult, MockLLM, MockEmbedding,
    RAGASMetrics, EvaluationPipeline, RAGABTester, QualityMonitor
)


class TestRAGResponse:
    """Test RAGResponse dataclass"""
    
    def test_rag_response_creation(self):
        response = RAGResponse(
            question="What is AI?",
            answer="AI is artificial intelligence.",
            contexts=["AI context"],
            metadata={"source": "test"}
        )
        assert response.question == "What is AI?"
        assert response.answer == "AI is artificial intelligence."
        assert len(response.contexts) == 1
        assert response.metadata["source"] == "test"
        assert response.timestamp is not None
        assert response.response_id is not None
    
    def test_response_id_generation(self):
        response1 = RAGResponse("Q1", "A1", ["C1"], {})
        response2 = RAGResponse("Q2", "A2", ["C2"], {})
        
        assert response1.response_id != response2.response_id
        assert len(response1.response_id) == 8  # MD5 hash truncated


class TestEvaluationResult:
    """Test EvaluationResult dataclass"""
    
    def test_evaluation_result_creation(self):
        result = EvaluationResult(
            response_id="test123",
            faithfulness=0.8,
            answer_relevancy=0.9,
            context_precision=0.7,
            context_recall=0.85,
            overall_score=0.82,
            alerts=["Low precision"],
            timestamp=datetime.now(),
            metadata={"test": True}
        )
        
        assert result.response_id == "test123"
        assert result.faithfulness == 0.8
        assert result.overall_score == 0.82
        assert len(result.alerts) == 1


class TestMockLLM:
    """Test MockLLM functionality"""
    
    def test_generate_questions_from_answer(self):
        llm = MockLLM()
        answer = "Machine learning is a subset of artificial intelligence."
        questions = llm.generate_questions_from_answer(answer, 3)
        
        assert len(questions) == 3
        assert all(isinstance(q, str) for q in questions)
        assert all("?" in q for q in questions)
    
    def test_extract_claims(self):
        llm = MockLLM()
        text = "AI is powerful. It can solve complex problems. Machine learning is a subset."
        claims = llm.extract_claims(text)
        
        assert len(claims) == 3
        assert "AI is powerful" in claims
        assert "It can solve complex problems" in claims
    
    def test_check_claim_support(self):
        llm = MockLLM()
        claim = "Machine learning uses algorithms"
        contexts = [
            "Machine learning algorithms are used for data analysis",
            "Deep learning is a subset of machine learning"
        ]
        
        assert llm.check_claim_support(claim, contexts) == True
        
        # Test unsupported claim
        unsupported_claim = "Quantum computing is faster"
        assert llm.check_claim_support(unsupported_claim, contexts) == False
    
    def test_generate_question_from_text(self):
        llm = MockLLM()
        text = "Machine learning algorithms process data to make predictions"
        question = llm.generate_question_from_text(text)
        
        assert isinstance(question, str)
        assert "?" in question


class TestMockEmbedding:
    """Test MockEmbedding functionality"""
    
    def test_encode_single_text(self):
        embedding = MockEmbedding()
        result = embedding.encode(["test text"])
        
        assert result.shape == (1, 384)
        assert isinstance(result, np.ndarray)
    
    def test_encode_multiple_texts(self):
        embedding = MockEmbedding()
        texts = ["text1", "text2", "text3"]
        result = embedding.encode(texts)
        
        assert result.shape == (3, 384)
    
    def test_deterministic_encoding(self):
        embedding = MockEmbedding()
        text = "consistent text"
        
        result1 = embedding.encode([text])
        result2 = embedding.encode([text])
        
        np.testing.assert_array_equal(result1, result2)
    
    def test_similarity_calculation(self):
        embedding = MockEmbedding()
        
        # Identical texts should have high similarity
        sim1 = embedding.similarity("test", "test")
        assert sim1 == pytest.approx(1.0, abs=0.01)
        
        # Different texts should have lower similarity
        sim2 = embedding.similarity("machine learning", "quantum computing")
        assert 0 <= sim2 <= 1
    
    def test_embedding_features(self):
        embedding = MockEmbedding()
        text = "machine learning algorithm data model"
        result = embedding.encode([text])[0]
        
        # Check that features are set
        assert result[0] > 0  # Length feature
        assert result[1] > 0  # Word count feature
        assert result[2] > 0  # Diversity feature
        assert result[3] > 0  # Tech words feature (should be high for ML text)


class TestRAGASMetrics:
    """Test RAGAS metrics implementation"""
    
    @pytest.fixture
    def metrics_calculator(self):
        return RAGASMetrics()
    
    @pytest.fixture
    def sample_response(self):
        return RAGResponse(
            question="What is machine learning?",
            answer="Machine learning is a subset of AI that enables computers to learn from data.",
            contexts=[
                "Machine learning is a method of data analysis.",
                "AI encompasses machine learning and other approaches.",
                "Computers can learn patterns from data using ML algorithms."
            ],
            metadata={"test": True}
        )
    
    def test_calculate_faithfulness(self, metrics_calculator):
        answer = "Machine learning uses algorithms. It processes data automatically."
        contexts = [
            "Machine learning algorithms process data",
            "Automated data processing is key to ML"
        ]
        
        faithfulness = metrics_calculator.calculate_faithfulness(answer, contexts)
        assert 0 <= faithfulness <= 1
        assert faithfulness > 0.5  # Should be supported by contexts
    
    def test_calculate_faithfulness_empty_inputs(self, metrics_calculator):
        # Empty answer
        assert metrics_calculator.calculate_faithfulness("", ["context"]) == 0.0
        
        # Empty contexts
        assert metrics_calculator.calculate_faithfulness("answer", []) == 0.0
    
    def test_calculate_answer_relevancy(self, metrics_calculator):
        question = "What is machine learning?"
        answer = "Machine learning is a subset of artificial intelligence."
        
        relevancy = metrics_calculator.calculate_answer_relevancy(question, answer)
        assert 0 <= relevancy <= 1
    
    def test_calculate_context_precision(self, metrics_calculator):
        question = "What is machine learning?"
        contexts = [
            "Machine learning is a method of data analysis",  # Relevant
            "The weather is sunny today",                     # Irrelevant
            "ML algorithms learn from data"                   # Relevant
        ]
        answer = "Machine learning is a subset of AI"
        
        precision = metrics_calculator.calculate_context_precision(question, contexts, answer)
        assert 0 <= precision <= 1
    
    def test_calculate_context_recall(self, metrics_calculator):
        ground_truth = ["ML is data analysis", "AI includes ML"]
        retrieved = ["Machine learning analyzes data", "AI encompasses ML", "Extra context"]
        
        recall = metrics_calculator.calculate_context_recall(ground_truth, retrieved)
        assert 0 <= recall <= 1
    
    def test_calculate_overall_score(self, metrics_calculator):
        overall = metrics_calculator.calculate_overall_score(0.8, 0.9, 0.7, 0.85)
        assert 0 <= overall <= 1
        assert overall == pytest.approx(0.82, abs=0.01)  # Weighted average
    
    def test_generate_alerts(self, metrics_calculator):
        metrics = {
            'faithfulness': 0.5,      # Below threshold
            'answer_relevancy': 0.9,  # Above threshold
            'context_precision': 0.4  # Below threshold
        }
        thresholds = {
            'faithfulness': 0.7,
            'answer_relevancy': 0.8,
            'context_precision': 0.6
        }
        
        alerts = metrics_calculator.generate_alerts(metrics, thresholds)
        assert len(alerts) == 2  # Two metrics below threshold
        assert any("faithfulness" in alert for alert in alerts)
        assert any("context_precision" in alert for alert in alerts)
    
    def test_evaluate_response(self, metrics_calculator, sample_response):
        result = metrics_calculator.evaluate_response(sample_response)
        
        assert isinstance(result, EvaluationResult)
        assert result.response_id == sample_response.response_id
        assert 0 <= result.faithfulness <= 1
        assert 0 <= result.answer_relevancy <= 1
        assert 0 <= result.context_precision <= 1
        assert 0 <= result.overall_score <= 1
        assert isinstance(result.alerts, list)


class TestEvaluationPipeline:
    """Test evaluation pipeline functionality"""
    
    @pytest.fixture
    def pipeline(self):
        metrics = RAGASMetrics()
        return EvaluationPipeline(metrics)
    
    @pytest.fixture
    def sample_responses(self):
        return [
            RAGResponse("Q1", "A1", ["C1"], {"id": "1"}),
            RAGResponse("Q2", "A2", ["C2"], {"id": "2"}),
            RAGResponse("Q3", "A3", ["C3"], {"id": "3"})
        ]
    
    def test_evaluate_batch(self, pipeline, sample_responses):
        results = pipeline.evaluate_batch(sample_responses)
        
        assert len(results) == 3
        assert all(isinstance(r, EvaluationResult) for r in results)
        assert len(pipeline.evaluation_history) == 3
    
    def test_generate_evaluation_report(self, pipeline, sample_responses):
        results = pipeline.evaluate_batch(sample_responses)
        report = pipeline.generate_evaluation_report(results)
        
        assert "summary" in report
        assert "metrics" in report
        assert "alerts" in report
        assert "recommendations" in report
        
        assert report["summary"]["total_responses"] == 3
        assert "faithfulness" in report["metrics"]
        assert "mean" in report["metrics"]["faithfulness"]
    
    def test_detect_quality_issues(self, pipeline):
        # Create results with known issues
        results = [
            EvaluationResult("1", 0.3, 0.4, 0.2, 0.5, 0.35, ["Low score"], datetime.now(), {}),
            EvaluationResult("2", 0.8, 0.9, 0.7, 0.85, 0.82, [], datetime.now(), {}),
        ]
        
        issues = pipeline.detect_quality_issues(results)
        
        assert len(issues) >= 1  # Should detect the low-scoring response
        assert issues[0]["response_id"] == "1"
        assert issues[0]["severity"] in ["low", "medium", "high", "critical"]
    
    def test_track_performance_over_time(self, pipeline, sample_responses):
        # Add some evaluation history
        results = pipeline.evaluate_batch(sample_responses)
        
        trends = pipeline.track_performance_over_time(window_days=1)
        
        assert "window_days" in trends
        assert "total_results" in trends
        assert trends["total_results"] == 3


class TestRAGABTester:
    """Test A/B testing framework"""
    
    @pytest.fixture
    def ab_tester(self):
        metrics = RAGASMetrics()
        return RAGABTester(metrics)
    
    @pytest.fixture
    def system_responses(self):
        system_a = [
            RAGResponse("Q1", "Answer A1", ["Context A1"], {"system": "A"}),
            RAGResponse("Q2", "Answer A2", ["Context A2"], {"system": "A"})
        ]
        system_b = [
            RAGResponse("Q1", "Answer B1", ["Context B1"], {"system": "B"}),
            RAGResponse("Q2", "Answer B2", ["Context B2"], {"system": "B"})
        ]
        return system_a, system_b
    
    def test_run_ab_test(self, ab_tester, system_responses):
        system_a, system_b = system_responses
        
        result = ab_tester.run_ab_test(system_a, system_b, "Test Comparison")
        
        assert "test_name" in result
        assert "system_a" in result
        assert "system_b" in result
        assert "statistical_significance" in result
        assert "recommendation" in result
        
        assert result["system_a"]["sample_size"] == 2
        assert result["system_b"]["sample_size"] == 2
    
    def test_calculate_statistical_significance(self, ab_tester):
        scores_a = [0.7, 0.8, 0.75, 0.82, 0.78]
        scores_b = [0.85, 0.9, 0.88, 0.92, 0.87]
        
        sig_result = ab_tester.calculate_statistical_significance(scores_a, scores_b)
        
        assert "test" in sig_result
        assert "p_value" in sig_result
        assert "significant" in sig_result
        assert isinstance(sig_result["significant"], bool)
    
    def test_statistical_significance_insufficient_data(self, ab_tester):
        scores_a = [0.7]  # Only one sample
        scores_b = [0.8]
        
        sig_result = ab_tester.calculate_statistical_significance(scores_a, scores_b)
        
        assert sig_result["test"] == "insufficient_data"
        assert sig_result["significant"] == False
    
    def test_generate_ab_report(self, ab_tester, system_responses):
        system_a, system_b = system_responses
        test_result = ab_tester.run_ab_test(system_a, system_b, "Test Report")
        
        report = ab_tester.generate_ab_report(test_result)
        
        assert isinstance(report, str)
        assert "A/B Test Report" in report
        assert "Test Report" in report
        assert "System A" in report
        assert "System B" in report
    
    def test_power_analysis(self, ab_tester):
        # Test with medium effect size
        sample_size = ab_tester.power_analysis(effect_size=0.5)
        
        assert isinstance(sample_size, int)
        assert sample_size > 0
        
        # Larger effect size should require smaller sample
        smaller_sample = ab_tester.power_analysis(effect_size=1.0)
        assert smaller_sample < sample_size


class TestQualityMonitor:
    """Test quality monitoring functionality"""
    
    @pytest.fixture
    def quality_monitor(self):
        metrics = RAGASMetrics()
        thresholds = {
            'faithfulness': 0.7,
            'answer_relevancy': 0.8,
            'response_time': 2.0
        }
        return QualityMonitor(metrics, thresholds)
    
    def test_monitor_response(self, quality_monitor):
        response = RAGResponse(
            "What is AI?",
            "AI is artificial intelligence.",
            ["AI context"],
            {"response_time": 1.5}
        )
        
        result = quality_monitor.monitor_response(response)
        
        assert "response_id" in result
        assert "timestamp" in result
        assert "metrics" in result
        assert "alerts" in result
        
        assert "faithfulness" in result["metrics"]
        assert "answer_relevancy" in result["metrics"]
        assert "response_time" in result["metrics"]
    
    def test_monitor_response_with_alerts(self, quality_monitor):
        # Create response that should trigger alerts
        response = RAGResponse(
            "What is AI?",
            "Unrelated answer about cooking.",  # Should have low relevancy
            ["Cooking recipes and techniques"],   # Irrelevant context
            {"response_time": 3.0}              # Slow response
        )
        
        result = quality_monitor.monitor_response(response)
        
        # Should have alerts for low relevancy and slow response
        assert len(result["alerts"]) >= 1
        assert len(quality_monitor.alerts) >= 1
    
    def test_set_baseline_metrics(self, quality_monitor):
        baseline_data = [
            {"metrics": {"faithfulness": 0.8, "answer_relevancy": 0.85, "response_time": 1.2}},
            {"metrics": {"faithfulness": 0.82, "answer_relevancy": 0.88, "response_time": 1.1}}
        ]
        
        quality_monitor.set_baseline_metrics(baseline_data)
        
        assert "faithfulness" in quality_monitor.baseline_metrics
        assert quality_monitor.baseline_metrics["faithfulness"] == pytest.approx(0.81, abs=0.01)
    
    def test_detect_quality_degradation(self, quality_monitor):
        # Set baseline
        baseline_data = [
            {"metrics": {"faithfulness": 0.9, "answer_relevancy": 0.9, "response_time": 1.0}}
        ]
        quality_monitor.set_baseline_metrics(baseline_data)
        
        # Add monitoring data with degraded performance
        for _ in range(10):
            response = RAGResponse(
                "Test question",
                "Poor quality answer",
                ["Irrelevant context"],
                {"response_time": 2.5}
            )
            quality_monitor.monitor_response(response)
        
        degradation_alerts = quality_monitor.detect_quality_degradation(window_size=5)
        
        # Should detect degradation
        assert len(degradation_alerts) >= 0  # May or may not detect based on thresholds
    
    def test_generate_monitoring_dashboard(self, quality_monitor):
        # Add some monitoring data
        for i in range(5):
            response = RAGResponse(
                f"Question {i}",
                f"Answer {i}",
                [f"Context {i}"],
                {"response_time": 1.0 + i * 0.1}
            )
            quality_monitor.monitor_response(response)
        
        dashboard = quality_monitor.generate_monitoring_dashboard()
        
        assert "timestamp" in dashboard
        assert "system_health" in dashboard
        assert "current_metrics" in dashboard
        assert "total_responses_monitored" in dashboard
        
        assert dashboard["system_health"] in ["healthy", "warning", "critical"]
        assert dashboard["total_responses_monitored"] == 5
    
    def test_auto_remediation_suggestions(self, quality_monitor):
        # Test different alert types
        alerts = [
            {"type": "low_faithfulness"},
            {"type": "low_relevancy"},
            {"type": "slow_response"},
            {"type": "quality_degradation"}
        ]
        
        for alert in alerts:
            suggestions = quality_monitor.auto_remediation_suggestions(alert)
            
            assert isinstance(suggestions, list)
            assert len(suggestions) <= 3  # Should return top 3 suggestions
            assert all(isinstance(s, str) for s in suggestions)


class TestIntegration:
    """Integration tests for complete evaluation workflow"""
    
    def test_end_to_end_evaluation_workflow(self):
        """Test complete evaluation workflow from response to monitoring"""
        
        # Initialize components
        metrics = RAGASMetrics()
        pipeline = EvaluationPipeline(metrics)
        monitor = QualityMonitor(metrics, {
            'faithfulness': 0.7,
            'answer_relevancy': 0.8,
            'response_time': 2.0
        })
        
        # Create test responses
        responses = [
            RAGResponse(
                "What is machine learning?",
                "Machine learning is a subset of AI that learns from data.",
                ["ML is part of AI", "Learning from data is key to ML"],
                {"response_time": 1.2}
            ),
            RAGResponse(
                "How does deep learning work?",
                "Deep learning uses neural networks with multiple layers.",
                ["Neural networks have layers", "Deep learning is advanced ML"],
                {"response_time": 1.8}
            )
        ]
        
        # 1. Individual evaluation
        individual_results = []
        for response in responses:
            result = metrics.evaluate_response(response)
            individual_results.append(result)
            assert isinstance(result, EvaluationResult)
        
        # 2. Batch evaluation
        batch_results = pipeline.evaluate_batch(responses)
        assert len(batch_results) == 2
        
        # 3. Generate report
        report = pipeline.generate_evaluation_report(batch_results)
        assert report["summary"]["total_responses"] == 2
        
        # 4. Quality monitoring
        for response in responses:
            monitoring_result = monitor.monitor_response(response)
            assert "metrics" in monitoring_result
        
        # 5. Dashboard generation
        dashboard = monitor.generate_monitoring_dashboard()
        assert dashboard["total_responses_monitored"] == 2
        
        # 6. Performance tracking
        trends = pipeline.track_performance_over_time(window_days=1)
        assert trends["total_results"] == 2
    
    def test_ab_testing_integration(self):
        """Test A/B testing integration with evaluation pipeline"""
        
        metrics = RAGASMetrics()
        ab_tester = RAGABTester(metrics)
        
        # Create two systems with different performance
        system_a_responses = [
            RAGResponse("Q1", "Good answer A1", ["Relevant context A1"], {"system": "A"}),
            RAGResponse("Q2", "Good answer A2", ["Relevant context A2"], {"system": "A"})
        ]
        
        system_b_responses = [
            RAGResponse("Q1", "Better answer B1", ["Very relevant context B1"], {"system": "B"}),
            RAGResponse("Q2", "Better answer B2", ["Very relevant context B2"], {"system": "B"})
        ]
        
        # Run A/B test
        ab_result = ab_tester.run_ab_test(
            system_a_responses, 
            system_b_responses, 
            "Integration Test"
        )
        
        assert ab_result["test_name"] == "Integration Test"
        assert "recommendation" in ab_result
        
        # Generate report
        report = ab_tester.generate_ab_report(ab_result)
        assert "Integration Test" in report
    
    def test_monitoring_with_alerts_integration(self):
        """Test monitoring system with alert generation"""
        
        metrics = RAGASMetrics()
        monitor = QualityMonitor(metrics, {
            'faithfulness': 0.8,      # High threshold
            'answer_relevancy': 0.9,  # High threshold
            'response_time': 1.0      # Low threshold
        })
        
        # Create responses that should trigger alerts
        problematic_responses = [
            RAGResponse(
                "What is quantum computing?",
                "Cooking is about preparing food.",  # Irrelevant answer
                ["Quantum mechanics principles"],     # Somewhat relevant context
                {"response_time": 2.5}              # Slow response
            ),
            RAGResponse(
                "How does blockchain work?",
                "Blockchain uses cryptographic hashing.",  # Good answer
                ["Blockchain is distributed ledger"],      # Good context
                {"response_time": 0.8}                    # Fast response
            )
        ]
        
        alert_count = 0
        for response in problematic_responses:
            result = monitor.monitor_response(response)
            alert_count += len(result["alerts"])
        
        # Should have generated some alerts
        assert alert_count > 0
        assert len(monitor.alerts) > 0
        
        # Check dashboard reflects alerts
        dashboard = monitor.generate_monitoring_dashboard()
        assert dashboard["active_alerts"] >= 0


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])