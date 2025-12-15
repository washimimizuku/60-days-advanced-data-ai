"""
Day 53: RAG Evaluation with RAGAS - Metrics, Quality Assessment & Optimization
Exercises for comprehensive RAG evaluation using RAGAS framework
"""

import numpy as np
import pandas as pd
import time
import json
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import random
from collections import defaultdict
import asyncio


@dataclass
class RAGResponse:
    """RAG system response structure"""
    question: str
    answer: str
    contexts: List[str]
    metadata: Dict
    timestamp: Optional[datetime] = None


@dataclass
class EvaluationResult:
    """Evaluation result structure"""
    response_id: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    overall_score: float
    alerts: List[str]


class MockLLM:
    """Mock LLM for evaluation exercises"""
    
    def generate_questions_from_answer(self, answer: str, num_questions: int = 3) -> List[str]:
        """Generate questions that the answer could address"""
        # Simple mock implementation
        question_templates = [
            f"What is {answer.split()[0] if answer.split() else 'this'}?",
            f"How does {answer.split()[0] if answer.split() else 'this'} work?",
            f"Why is {answer.split()[0] if answer.split() else 'this'} important?"
        ]
        return question_templates[:num_questions]
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Simple sentence splitting as mock claim extraction
        sentences = text.split('.')
        return [s.strip() for s in sentences if s.strip()]
    
    def check_claim_support(self, claim: str, contexts: List[str]) -> bool:
        """Check if claim is supported by contexts"""
        # Simple keyword overlap check
        claim_words = set(claim.lower().split())
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(claim_words.intersection(context_words))
            if overlap >= len(claim_words) * 0.3:  # 30% overlap threshold
                return True
        return False


class MockEmbedding:
    """Mock embedding model for similarity calculations"""
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings"""
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text
            words = text.lower().split()
            embedding = np.random.RandomState(hash(text) % 2**32).rand(384)
            
            # Add some semantic features
            embedding[0] = len(text) / 1000
            embedding[1] = len(words) / 100
            embedding[2] = len(set(words)) / len(words) if words else 0
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        emb1 = self.encode([text1])[0]
        emb2 = self.encode([text2])[0]
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))


# Exercise 1: Implement Core RAGAS Metrics
def exercise_1_core_metrics():
    """
    Exercise 1: Implement the four core RAGAS metrics
    
    TODO: Complete the RAGASMetrics class
    """
    print("=== Exercise 1: Core RAGAS Metrics Implementation ===")
    
    class RAGASMetrics:
        def __init__(self):
            self.llm = MockLLM()
            self.embedding = MockEmbedding()
        
        def calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
            """Calculate faithfulness: ratio of claims supported by contexts"""
            if not answer.strip() or not contexts:
                return 0.0
            
            claims = self.llm.extract_claims(answer)
            if not claims:
                return 1.0  # No claims to verify
            
            supported_claims = 0
            for claim in claims:
                if self.llm.check_claim_support(claim, contexts):
                    supported_claims += 1
            
            return supported_claims / len(claims)
        
        def calculate_answer_relevancy(self, question: str, answer: str) -> float:
            """Calculate answer relevancy using question generation"""
            if not question.strip() or not answer.strip():
                return 0.0
            
            # Generate questions from answer
            generated_questions = self.llm.generate_questions_from_answer(answer, 3)
            
            # Calculate similarities
            similarities = []
            for gen_question in generated_questions:
                similarity = self.embedding.similarity(question, gen_question)
                similarities.append(similarity)
            
            return np.mean(similarities) if similarities else 0.0
        
        def calculate_context_precision(self, question: str, contexts: List[str], answer: str) -> float:
            """Calculate context precision: ratio of relevant contexts"""
            if not contexts:
                return 0.0
            
            relevant_contexts = 0
            for context in contexts:
                # Check if context is relevant to question
                question_context_sim = self.embedding.similarity(question, context)
                
                # Also check if context contributes to answer
                answer_context_sim = self.embedding.similarity(answer, context)
                
                # Context is relevant if it's similar to question OR contributes to answer
                if question_context_sim >= 0.6 or answer_context_sim >= 0.6:
                    relevant_contexts += 1
            
            return relevant_contexts / len(contexts)
        
        def calculate_context_recall(self, ground_truth_contexts: List[str], 
                                   retrieved_contexts: List[str]) -> float:
            """Calculate context recall: coverage of ground truth contexts"""
            if not ground_truth_contexts:
                return 1.0  # Perfect recall if no ground truth needed
            
            if not retrieved_contexts:
                return 0.0
            
            covered_contexts = 0
            for gt_context in ground_truth_contexts:
                # Check if this ground truth context is covered by any retrieved context
                for ret_context in retrieved_contexts:
                    similarity = self.embedding.similarity(gt_context, ret_context)
                    if similarity >= 0.7:
                        covered_contexts += 1
                        break
            
            return covered_contexts / len(ground_truth_contexts)
        
        def evaluate_response(self, response: RAGResponse, 
                            ground_truth_contexts: Optional[List[str]] = None) -> EvaluationResult:
            """Complete evaluation of a single response"""
            thresholds = {
                'faithfulness': 0.7,
                'answer_relevancy': 0.8,
                'context_precision': 0.6,
                'overall_score': 0.7
            }
            
            # Calculate core metrics
            faithfulness = self.calculate_faithfulness(response.answer, response.contexts)
            answer_relevancy = self.calculate_answer_relevancy(response.question, response.answer)
            context_precision = self.calculate_context_precision(
                response.question, response.contexts, response.answer)
            
            # Context recall requires ground truth
            if ground_truth_contexts:
                context_recall = self.calculate_context_recall(ground_truth_contexts, response.contexts)
            else:
                context_recall = 1.0  # Assume perfect if no ground truth
            
            # Calculate overall score (weighted average)
            overall_score = (
                0.3 * faithfulness +
                0.3 * answer_relevancy +
                0.2 * context_precision +
                0.2 * context_recall
            )
            
            # Generate alerts
            alerts = []
            if faithfulness < thresholds['faithfulness']:
                alerts.append(f"Low faithfulness: {faithfulness:.3f}")
            if answer_relevancy < thresholds['answer_relevancy']:
                alerts.append(f"Low answer relevancy: {answer_relevancy:.3f}")
            if context_precision < thresholds['context_precision']:
                alerts.append(f"Low context precision: {context_precision:.3f}")
            if overall_score < thresholds['overall_score']:
                alerts.append(f"Low overall score: {overall_score:.3f}")
            
            return EvaluationResult(
                response_id=response.response_id,
                faithfulness=faithfulness,
                answer_relevancy=answer_relevancy,
                context_precision=context_precision,
                context_recall=context_recall,
                overall_score=overall_score,
                alerts=alerts,
                timestamp=time.time(),
                metadata=response.metadata
            )
    
    # Test data
    test_response = RAGResponse(
        question="What is machine learning?",
        answer="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
        contexts=[
            "Machine learning is a method of data analysis that automates analytical model building.",
            "Artificial intelligence encompasses machine learning and other computational approaches.",
            "Programming languages like Python are commonly used in machine learning."
        ],
        metadata={"source": "test"}
    )
    
    metrics = RAGASMetrics()
    
    print(f"Question: {test_response.question}")
    print(f"Answer: {test_response.answer}")
    print("\n--- RAGAS Metrics Results ---")
    result = metrics.evaluate_response(test_response)
    print(f"Faithfulness: {result.faithfulness:.3f}")
    print(f"Answer Relevancy: {result.answer_relevancy:.3f}")
    print(f"Context Precision: {result.context_precision:.3f}")
    print(f"Overall Score: {result.overall_score:.3f}")
    if result.alerts:
        print(f"Alerts: {', '.join(result.alerts)}")


# Exercise 2: Build Evaluation Pipeline
def exercise_2_evaluation_pipeline():
    """
    Exercise 2: Create an automated evaluation pipeline
    
    TODO: Complete the EvaluationPipeline class
    """
    print("\n=== Exercise 2: Automated Evaluation Pipeline ===")
    
    class EvaluationPipeline:
        def __init__(self, metrics_calculator):
            self.metrics = metrics_calculator
            self.evaluation_history = []
            self.thresholds = {
                'faithfulness': 0.7,
                'answer_relevancy': 0.8,
                'context_precision': 0.6,
                'overall_score': 0.7
            }
        
        def evaluate_batch(self, responses: List[RAGResponse]) -> List[EvaluationResult]:
            """Evaluate a batch of responses"""
            results = []
            
            for response in responses:
                result = self.metrics.evaluate_response(response)
                results.append(result)
            
            # Store in history
            self.evaluation_history.extend(results)
            
            return results
        
        def generate_evaluation_report(self, results: List[EvaluationResult]) -> Dict:
            """Generate comprehensive evaluation report"""
            if not results:
                return {"error": "No evaluation results provided"}
            
            # Extract metric values
            faithfulness_scores = [r.faithfulness for r in results]
            relevancy_scores = [r.answer_relevancy for r in results]
            precision_scores = [r.context_precision for r in results]
            overall_scores = [r.overall_score for r in results]
            
            # Calculate statistics
            def calc_stats(scores):
                return {
                    'mean': np.mean(scores),
                    'std': np.std(scores),
                    'min': np.min(scores),
                    'max': np.max(scores)
                }
            
            # Count alerts
            total_alerts = sum(len(r.alerts) for r in results)
            responses_with_alerts = sum(1 for r in results if r.alerts)
            
            report = {
                'summary': {
                    'total_responses': len(results),
                    'total_alerts': total_alerts,
                    'responses_with_alerts': responses_with_alerts,
                    'alert_rate': responses_with_alerts / len(results)
                },
                'metrics': {
                    'faithfulness': calc_stats(faithfulness_scores),
                    'answer_relevancy': calc_stats(relevancy_scores),
                    'context_precision': calc_stats(precision_scores),
                    'overall_score': calc_stats(overall_scores)
                },
                'recommendations': self._generate_recommendations(results)
            }
            
            return report
        
        def _generate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
            """Generate improvement recommendations"""
            recommendations = []
            
            avg_faithfulness = np.mean([r.faithfulness for r in results])
            avg_relevancy = np.mean([r.answer_relevancy for r in results])
            avg_precision = np.mean([r.context_precision for r in results])
            
            if avg_faithfulness < 0.7:
                recommendations.append("Improve context quality and fact-checking mechanisms")
            if avg_relevancy < 0.8:
                recommendations.append("Review answer generation prompts for better relevancy")
            if avg_precision < 0.6:
                recommendations.append("Enhance retrieval system for more relevant contexts")
            
            return recommendations
        
        def detect_quality_issues(self, results: List[EvaluationResult]) -> List[Dict]:
            """Detect quality issues and anomalies"""
            issues = []
            
            for result in results:
                issue = {
                    'response_id': result.response_id,
                    'timestamp': result.timestamp,
                    'issues': [],
                    'severity': 'low'
                }
                
                # Check for multiple low scores
                low_scores = []
                if result.faithfulness < 0.5:
                    low_scores.append('faithfulness')
                if result.answer_relevancy < 0.5:
                    low_scores.append('answer_relevancy')
                if result.context_precision < 0.4:
                    low_scores.append('context_precision')
                
                if len(low_scores) >= 2:
                    issue['issues'].append(f"Multiple low scores: {', '.join(low_scores)}")
                    issue['severity'] = 'high'
                
                # Check for extremely low overall score
                if result.overall_score < 0.3:
                    issue['issues'].append(f"Extremely low overall score: {result.overall_score:.3f}")
                    issue['severity'] = 'critical'
                
                # Add alerts
                if result.alerts:
                    issue['issues'].extend(result.alerts)
                    if issue['severity'] == 'low':
                        issue['severity'] = 'medium'
                
                if issue['issues']:
                    issues.append(issue)
            
            return issues
        
        def track_performance_over_time(self, window_days: int = 7) -> Dict:
            """Track performance trends over time"""
            if not self.evaluation_history:
                return {"error": "No evaluation history available"}
            
            # For this mock implementation, use all available history
            recent_results = self.evaluation_history
            
            if not recent_results:
                return {"error": f"No results in the last {window_days} days"}
            
            # Calculate averages
            avg_metrics = {
                'faithfulness': np.mean([r.faithfulness for r in recent_results]),
                'answer_relevancy': np.mean([r.answer_relevancy for r in recent_results]),
                'context_precision': np.mean([r.context_precision for r in recent_results]),
                'overall_score': np.mean([r.overall_score for r in recent_results])
            }
            
            return {
                'window_days': window_days,
                'total_results': len(recent_results),
                'average_metrics': avg_metrics,
                'trends': {}  # Simplified for mock implementation
            }
    
    # Test the evaluation pipeline
    test_responses = [
        RAGResponse(
            question="What is deep learning?",
            answer="Deep learning is a subset of machine learning using neural networks with multiple layers.",
            contexts=["Deep learning uses artificial neural networks.", "Neural networks have multiple layers."],
            metadata={"id": "1"}
        ),
        RAGResponse(
            question="How does reinforcement learning work?",
            answer="Reinforcement learning trains agents through rewards and penalties.",
            contexts=["RL uses reward signals.", "Agents learn through trial and error."],
            metadata={"id": "2"}
        )
    ]
    
    pipeline = EvaluationPipeline(RAGASMetrics())
    
    print("Testing evaluation pipeline...")
    print("\n--- Batch Evaluation Results ---")
    results = pipeline.evaluate_batch(test_responses)
    report = pipeline.generate_evaluation_report(results)
    issues = pipeline.detect_quality_issues(results)
    print(f"Processed {len(results)} responses")
    print(f"Average overall score: {report['metrics']['overall_score']['mean']:.3f}")
    print(f"Alert rate: {report['summary']['alert_rate']:.2%}")
    print(f"Quality issues detected: {len(issues)}")


# Exercise 3: A/B Testing Framework
def exercise_3_ab_testing():
    """
    Exercise 3: Implement A/B testing for RAG systems
    
    TODO: Complete the RAGABTester class
    """
    print("\n=== Exercise 3: A/B Testing Framework ===")
    
    class RAGABTester:
        def __init__(self, evaluator):
            self.evaluator = evaluator
            self.test_results = []
        
        def run_ab_test(self, system_a_responses: List[RAGResponse], 
                       system_b_responses: List[RAGResponse],
                       test_name: str) -> Dict:
            """Run A/B test between two RAG systems"""
            
            # Evaluate both systems
            results_a = [self.evaluator.evaluate_response(r) for r in system_a_responses]
            results_b = [self.evaluator.evaluate_response(r) for r in system_b_responses]
            
            # Extract metric scores
            scores_a = [r.overall_score for r in results_a]
            scores_b = [r.overall_score for r in results_b]
            
            # Calculate statistical significance
            significance = self.calculate_statistical_significance(scores_a, scores_b)
            
            # Generate recommendation
            mean_a = np.mean(scores_a)
            mean_b = np.mean(scores_b)
            
            if significance.get('significant', False) and mean_b > mean_a:
                recommendation = "system_b"
                confidence = "high"
            elif significance.get('significant', False) and mean_a > mean_b:
                recommendation = "system_a"
                confidence = "high"
            else:
                recommendation = "inconclusive"
                confidence = "low"
            
            test_result = {
                'test_name': test_name,
                'timestamp': time.time(),
                'system_a': {
                    'sample_size': len(results_a),
                    'mean_score': mean_a,
                    'alert_rate': sum(1 for r in results_a if r.alerts) / len(results_a)
                },
                'system_b': {
                    'sample_size': len(results_b),
                    'mean_score': mean_b,
                    'alert_rate': sum(1 for r in results_b if r.alerts) / len(results_b)
                },
                'statistical_significance': significance,
                'recommendation': {
                    'recommended_system': recommendation,
                    'confidence': confidence
                }
            }
            
            self.test_results.append(test_result)
            return test_result
        
        def calculate_statistical_significance(self, scores_a: List[float], 
                                            scores_b: List[float]) -> Dict:
            """Calculate statistical significance of differences"""
            if len(scores_a) < 2 or len(scores_b) < 2:
                return {
                    'test': 'insufficient_data',
                    'p_value': None,
                    'significant': False
                }
            
            # Simple t-test approximation
            mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
            std_a, std_b = np.std(scores_a), np.std(scores_b)
            n_a, n_b = len(scores_a), len(scores_b)
            
            # Pooled standard error
            pooled_se = np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
            
            if pooled_se == 0:
                return {
                    'test': 't_test',
                    'p_value': 0.0 if mean_a != mean_b else 1.0,
                    'significant': mean_a != mean_b
                }
            
            # t-statistic
            t_stat = (mean_b - mean_a) / pooled_se
            
            # Approximate p-value (simplified)
            p_value = 2 * (1 - abs(t_stat) / (abs(t_stat) + 1))  # Rough approximation
            
            return {
                'test': 't_test',
                'p_value': p_value,
                'significant': p_value < 0.05,
                't_statistic': t_stat
            }
        
        def generate_ab_report(self, test_results: Dict) -> str:
            """Generate human-readable A/B test report"""
            report = f"""
A/B Test Report: {test_results['test_name']}
{'=' * 50}

Test Overview:
- System A Sample Size: {test_results['system_a']['sample_size']}
- System B Sample Size: {test_results['system_b']['sample_size']}

Performance Comparison:
- System A Mean Score: {test_results['system_a']['mean_score']:.3f}
- System B Mean Score: {test_results['system_b']['mean_score']:.3f}
- Difference: {test_results['system_b']['mean_score'] - test_results['system_a']['mean_score']:+.3f}

Statistical Significance:
- Test: {test_results['statistical_significance']['test']}
- Significant: {'Yes' if test_results['statistical_significance'].get('significant', False) else 'No'}

Recommendation:
- Recommended System: {test_results['recommendation']['recommended_system'].upper()}
- Confidence: {test_results['recommendation']['confidence'].upper()}
"""
            return report
        
        def power_analysis(self, effect_size: float, alpha: float = 0.05, 
                          power: float = 0.8) -> int:
            """Calculate required sample size for test"""
            if effect_size <= 0:
                return float('inf')
            
            # Simplified power analysis calculation
            z_alpha = 1.96  # For alpha = 0.05
            z_beta = 0.84   # For power = 0.8
            
            n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
            
            return int(np.ceil(n))
    
    # Test A/B testing framework
    system_a_responses = [
        RAGResponse("Q1", "Answer A1", ["Context A1"], {"system": "A"}),
        RAGResponse("Q2", "Answer A2", ["Context A2"], {"system": "A"})
    ]
    
    system_b_responses = [
        RAGResponse("Q1", "Answer B1", ["Context B1"], {"system": "B"}),
        RAGResponse("Q2", "Answer B2", ["Context B2"], {"system": "B"})
    ]
    
    ab_tester = RAGABTester(RAGASMetrics())
    
    print("Setting up A/B test...")
    print("\n--- A/B Test Results ---")
    results = ab_tester.run_ab_test(system_a_responses, system_b_responses, "Retrieval Method Test")
    print(f"Test: {results['test_name']}")
    print(f"System A score: {results['system_a']['mean_score']:.3f}")
    print(f"System B score: {results['system_b']['mean_score']:.3f}")
    print(f"Recommendation: {results['recommendation']['recommended_system']}")
    print(f"Confidence: {results['recommendation']['confidence']}")


# Exercise 4: Real-time Quality Monitoring
def exercise_4_quality_monitoring():
    """
    Exercise 4: Implement real-time quality monitoring
    
    TODO: Complete the QualityMonitor class
    """
    print("\n=== Exercise 4: Real-time Quality Monitoring ===")
    
    class QualityMonitor:
        def __init__(self, metrics_calculator, alert_thresholds):
            self.metrics = metrics_calculator
            self.thresholds = alert_thresholds
            self.monitoring_data = []
            self.alerts = []
        
        def monitor_response(self, response: RAGResponse) -> Dict:
            """Monitor individual response quality in real-time"""
            start_time = time.time()
            
            # Quick evaluation (subset of metrics for speed)
            faithfulness = self.metrics.calculate_faithfulness(response.answer, response.contexts)
            answer_relevancy = self.metrics.calculate_answer_relevancy(response.question, response.answer)
            
            # Calculate response time if available
            response_time = response.metadata.get('response_time', time.time() - start_time)
            
            monitoring_result = {
                'response_id': response.response_id,
                'timestamp': time.time(),
                'metrics': {
                    'faithfulness': faithfulness,
                    'answer_relevancy': answer_relevancy,
                    'response_time': response_time
                },
                'alerts': []
            }
            
            # Check thresholds and generate alerts
            if faithfulness < self.thresholds.get('faithfulness', 0.7):
                alert = f"Low faithfulness score: {faithfulness:.3f}"
                monitoring_result['alerts'].append(alert)
                self.alerts.append({'message': alert, 'timestamp': time.time()})
            
            if answer_relevancy < self.thresholds.get('answer_relevancy', 0.8):
                alert = f"Low answer relevancy: {answer_relevancy:.3f}"
                monitoring_result['alerts'].append(alert)
                self.alerts.append({'message': alert, 'timestamp': time.time()})
            
            if response_time > self.thresholds.get('response_time', 2.0):
                alert = f"Slow response time: {response_time:.2f}s"
                monitoring_result['alerts'].append(alert)
                self.alerts.append({'message': alert, 'timestamp': time.time()})
            
            # Store monitoring data
            self.monitoring_data.append(monitoring_result)
            
            return monitoring_result
        
        def detect_quality_degradation(self, window_size: int = 100) -> List[Dict]:
            """Detect quality degradation over recent responses"""
            if len(self.monitoring_data) < window_size:
                return []
            
            # Get recent data
            recent_data = self.monitoring_data[-window_size:]
            
            # Calculate current averages
            current_metrics = {
                'faithfulness': np.mean([d['metrics']['faithfulness'] for d in recent_data]),
                'answer_relevancy': np.mean([d['metrics']['answer_relevancy'] for d in recent_data])
            }
            
            degradation_alerts = []
            
            # Compare to baseline if available
            for metric, current_value in current_metrics.items():
                if metric in self.baseline_metrics:
                    baseline_value = self.baseline_metrics[metric]
                    degradation = (baseline_value - current_value) / baseline_value
                    
                    if degradation > 0.1:  # 10% degradation threshold
                        degradation_alerts.append({
                            'type': 'quality_degradation',
                            'metric': metric,
                            'current_value': current_value,
                            'baseline_value': baseline_value,
                            'degradation_percent': degradation * 100,
                            'timestamp': time.time()
                        })
            
            return degradation_alerts
        
        def generate_monitoring_dashboard(self) -> Dict:
            """Generate real-time monitoring dashboard data"""
            if not self.monitoring_data:
                return {"error": "No monitoring data available"}
            
            # Recent performance (last 10 responses or all available)
            recent_data = self.monitoring_data[-10:] if len(self.monitoring_data) >= 10 else self.monitoring_data
            
            # Calculate current metrics
            current_metrics = {
                'faithfulness': np.mean([d['metrics']['faithfulness'] for d in recent_data]),
                'answer_relevancy': np.mean([d['metrics']['answer_relevancy'] for d in recent_data]),
                'response_time': np.mean([d['metrics']['response_time'] for d in recent_data])
            }
            
            # Active alerts (recent)
            recent_alerts = [alert for alert in self.alerts[-24:]]  # Last 24 alerts
            
            # System health status
            faithfulness_ok = current_metrics['faithfulness'] >= self.thresholds.get('faithfulness', 0.7)
            relevancy_ok = current_metrics['answer_relevancy'] >= self.thresholds.get('answer_relevancy', 0.8)
            response_time_ok = current_metrics['response_time'] <= self.thresholds.get('response_time', 2.0)
            
            if faithfulness_ok and relevancy_ok and response_time_ok:
                health_status = "healthy"
            elif len(recent_alerts) > 5:
                health_status = "critical"
            else:
                health_status = "warning"
            
            return {
                'timestamp': time.time(),
                'system_health': health_status,
                'current_metrics': current_metrics,
                'active_alerts': len(recent_alerts),
                'total_responses_monitored': len(self.monitoring_data),
                'recent_response_count': len(recent_data)
            }
        
        def auto_remediation_suggestions(self, alert: Dict) -> List[str]:
            """Generate automatic remediation suggestions"""
            suggestions = []
            
            alert_message = alert.get('message', '')
            
            if 'faithfulness' in alert_message:
                suggestions.extend([
                    "Review and improve context retrieval quality",
                    "Implement fact-checking mechanisms",
                    "Retrain or fine-tune the generation model"
                ])
            
            elif 'relevancy' in alert_message:
                suggestions.extend([
                    "Review answer generation prompts",
                    "Improve question understanding and parsing",
                    "Implement better query-answer alignment"
                ])
            
            elif 'response time' in alert_message:
                suggestions.extend([
                    "Optimize retrieval index performance",
                    "Implement response caching",
                    "Scale up computational resources"
                ])
            
            else:
                suggestions.extend([
                    "Investigate recent system changes",
                    "Check data quality and freshness",
                    "Review model performance metrics"
                ])
            
            # Return top 3 suggestions
            return suggestions[:3]
    
    # Test quality monitoring
    alert_thresholds = {
        'faithfulness': 0.7,
        'answer_relevancy': 0.8,
        'response_time': 2.0  # seconds
    }
    
    monitor = QualityMonitor(RAGASMetrics(), alert_thresholds)
    
    print("Testing quality monitoring...")
    print("\n--- Quality Monitoring Results ---")
    
    # Simulate monitoring several responses
    test_responses = [
        RAGResponse("What is AI?", "AI is artificial intelligence.", ["AI context"], {"response_time": 1.5}),
        RAGResponse("How does ML work?", "ML works with data.", ["ML context"], {"response_time": 2.5})
    ]
    
    for i, response in enumerate(test_responses, 1):
        monitoring_result = monitor.monitor_response(response)
        print(f"Response {i}: {len(monitoring_result['alerts'])} alerts")
        if monitoring_result['alerts']:
            print(f"  Alerts: {', '.join(monitoring_result['alerts'])}")
    
    dashboard = monitor.generate_monitoring_dashboard()
    print(f"System Health: {dashboard['system_health']}")
    print(f"Total Responses Monitored: {dashboard['total_responses_monitored']}")


# Exercise 5: Custom Metrics Development
def exercise_5_custom_metrics():
    """
    Exercise 5: Develop custom evaluation metrics
    
    TODO: Complete the CustomMetrics class
    """
    print("\n=== Exercise 5: Custom Metrics Development ===")
    
    class CustomMetrics:
        def __init__(self):
            self.embedding = MockEmbedding()
        
        def calculate_response_completeness(self, question: str, answer: str, 
                                          expected_aspects: List[str]) -> float:
            """Calculate how complete the answer is"""
            if not expected_aspects:
                return 1.0
            
            answer_words = set(answer.lower().split())
            covered_aspects = 0
            
            for aspect in expected_aspects:
                aspect_words = set(aspect.lower().split())
                # Check if any aspect words are in the answer
                if aspect_words.intersection(answer_words):
                    covered_aspects += 1
            
            return covered_aspects / len(expected_aspects)
        
        def calculate_context_diversity(self, contexts: List[str]) -> float:
            """Calculate diversity of retrieved contexts"""
            if len(contexts) <= 1:
                return 1.0
            
            similarities = []
            for i in range(len(contexts)):
                for j in range(i+1, len(contexts)):
                    similarity = self.embedding.similarity(contexts[i], contexts[j])
                    similarities.append(similarity)
            
            avg_similarity = np.mean(similarities) if similarities else 0
            return 1 - avg_similarity
        
        def calculate_answer_conciseness(self, question: str, answer: str) -> float:
            """Calculate how concise the answer is"""
            if not answer.strip():
                return 0.0
            
            answer_length = len(answer.split())
            question_length = len(question.split())
            
            # Optimal answer length is roughly 2-3x question length
            optimal_length = question_length * 2.5
            
            # Calculate penalty for deviation from optimal length
            length_ratio = answer_length / optimal_length if optimal_length > 0 else 1
            
            if length_ratio <= 1:
                # Reward shorter answers up to optimal length
                conciseness = length_ratio
            else:
                # Penalize longer answers
                conciseness = 1 / length_ratio
            
            return min(1.0, max(0.0, conciseness))
        
        def calculate_temporal_relevance(self, contexts: List[str], 
                                       current_date: datetime) -> float:
            """Calculate temporal relevance of contexts"""
            if not contexts:
                return 0.0
            
            # Simple heuristic: look for temporal keywords
            recent_keywords = ['recent', 'latest', 'current', 'new', 'today', '2023', '2024']
            outdated_keywords = ['old', 'previous', 'former', 'past', '2020', '2019']
            
            relevance_scores = []
            for context in contexts:
                context_lower = context.lower()
                
                recent_count = sum(1 for keyword in recent_keywords if keyword in context_lower)
                outdated_count = sum(1 for keyword in outdated_keywords if keyword in context_lower)
                
                # Score based on temporal indicators
                if recent_count > outdated_count:
                    score = 0.8 + (recent_count * 0.1)
                elif outdated_count > recent_count:
                    score = 0.3 - (outdated_count * 0.1)
                else:
                    score = 0.5  # Neutral
                
                relevance_scores.append(max(0.0, min(1.0, score)))
            
            return np.mean(relevance_scores)
        
        def calculate_domain_specificity(self, answer: str, domain_keywords: List[str]) -> float:
            """Calculate domain-specific accuracy"""
            if not domain_keywords or not answer.strip():
                return 0.0
            
            answer_words = set(answer.lower().split())
            domain_words_used = sum(1 for keyword in domain_keywords 
                                  if keyword.lower() in answer_words)
            
            # Score based on domain keyword usage
            specificity_score = domain_words_used / len(domain_keywords)
            
            return min(1.0, specificity_score)
        
        def comprehensive_evaluation(self, response: RAGResponse, 
                                   evaluation_config: Dict) -> Dict:
            """Run comprehensive evaluation with custom metrics"""
            results = {}
            
            # Calculate custom metrics
            if 'expected_aspects' in evaluation_config:
                results['completeness'] = self.calculate_response_completeness(
                    response.question, response.answer, evaluation_config['expected_aspects'])
            
            results['context_diversity'] = self.calculate_context_diversity(response.contexts)
            results['answer_conciseness'] = self.calculate_answer_conciseness(
                response.question, response.answer)
            
            if 'current_date' in evaluation_config:
                results['temporal_relevance'] = self.calculate_temporal_relevance(
                    response.contexts, evaluation_config['current_date'])
            
            if 'domain_keywords' in evaluation_config:
                results['domain_specificity'] = self.calculate_domain_specificity(
                    response.answer, evaluation_config['domain_keywords'])
            
            # Calculate weighted overall score
            weights = evaluation_config.get('weights', {})
            weighted_score = 0
            total_weight = 0
            
            for metric, score in results.items():
                weight = weights.get(metric, 1.0)
                weighted_score += score * weight
                total_weight += weight
            
            results['overall_custom_score'] = weighted_score / total_weight if total_weight > 0 else 0
            
            return results
    
    # Test custom metrics
    test_response = RAGResponse(
        question="What are the latest developments in quantum computing?",
        answer="Quantum computing has seen advances in error correction and quantum supremacy demonstrations.",
        contexts=[
            "Recent quantum computing research focuses on error correction.",
            "Google achieved quantum supremacy in 2019.",
            "IBM is developing quantum processors."
        ],
        metadata={"domain": "technology"}
    )
    
    custom_metrics = CustomMetrics()
    
    print(f"Question: {test_response.question}")
    print("\n--- Custom Metrics Results ---")
    
    # Test individual metrics (using simplified implementations)
    expected_aspects = ["error correction", "quantum supremacy", "applications"]
    
    # Simple completeness calculation
    answer_words = set(test_response.answer.lower().split())
    covered_aspects = sum(1 for aspect in expected_aspects 
                         if any(word in answer_words for word in aspect.split()))
    completeness = covered_aspects / len(expected_aspects)
    
    # Simple diversity calculation
    if len(test_response.contexts) > 1:
        similarities = []
        for i in range(len(test_response.contexts)):
            for j in range(i+1, len(test_response.contexts)):
                # Simple word overlap similarity
                words_i = set(test_response.contexts[i].lower().split())
                words_j = set(test_response.contexts[j].lower().split())
                overlap = len(words_i.intersection(words_j))
                total = len(words_i.union(words_j))
                sim = overlap / total if total > 0 else 0
                similarities.append(sim)
        diversity = 1 - np.mean(similarities) if similarities else 1.0
    else:
        diversity = 1.0
    
    print(f"Response completeness: {completeness:.3f}")
    print(f"Context diversity: {diversity:.3f}")


# Exercise 6: Evaluation Dataset Management
def exercise_6_dataset_management():
    """
    Exercise 6: Build evaluation dataset management system
    
    TODO: Complete the EvaluationDatasetManager class
    """
    print("\n=== Exercise 6: Evaluation Dataset Management ===")
    
    class EvaluationDatasetManager:
        def __init__(self):
            self.datasets = {}
            self.llm = MockLLM()
        
        def create_synthetic_dataset(self, documents: List[str], 
                                   num_samples: int = 50) -> List[Dict]:
            """TODO: Generate synthetic evaluation dataset"""
            # Hint:
            # 1. Sample documents randomly
            # 2. Generate questions from document content
            # 3. Create ground truth answers
            # 4. Return structured dataset
            pass
        
        def augment_dataset_with_negatives(self, dataset: List[Dict]) -> List[Dict]:
            """TODO: Add negative examples to dataset"""
            # Hint:
            # 1. Create incorrect answers for existing questions
            # 2. Add irrelevant contexts
            # 3. Generate edge cases
            # 4. Return augmented dataset
            pass
        
        def validate_dataset_quality(self, dataset: List[Dict]) -> Dict:
            """TODO: Validate evaluation dataset quality"""
            # Hint:
            # 1. Check for duplicates and inconsistencies
            # 2. Validate question-answer pairs
            # 3. Assess diversity and coverage
            # 4. Return quality report
            pass
        
        def split_dataset(self, dataset: List[Dict], 
                         train_ratio: float = 0.7) -> Tuple[List[Dict], List[Dict], List[Dict]]:
            """TODO: Split dataset into train/validation/test sets"""
            # Hint:
            # 1. Shuffle dataset randomly
            # 2. Split according to ratios
            # 3. Ensure balanced distribution
            # 4. Return train, validation, test sets
            pass
        
        def benchmark_dataset_difficulty(self, dataset: List[Dict], 
                                       rag_system) -> Dict:
            """TODO: Benchmark dataset difficulty using RAG system"""
            # Hint:
            # 1. Run RAG system on dataset
            # 2. Calculate performance metrics
            # 3. Identify easy vs hard questions
            # 4. Return difficulty analysis
            pass
    
    # Test dataset management
    sample_documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing deals with text and speech."
    ]
    
    dataset_manager = EvaluationDatasetManager()
    
    print("Testing dataset management...")
    print("\n--- Dataset Management Results ---")
    
    # Simple synthetic dataset generation
    synthetic_dataset = []
    for i, doc in enumerate(sample_documents[:3]):
        question = f"What is mentioned in document {i+1}?"
        answer = doc[:50] + "..."  # First 50 chars as answer
        synthetic_dataset.append({
            'question': question,
            'answer': answer,
            'context': doc,
            'metadata': {'doc_id': i+1}
        })
    
    # Simple quality validation
    quality_report = {
        'total_samples': len(synthetic_dataset),
        'avg_question_length': np.mean([len(item['question']) for item in synthetic_dataset]),
        'avg_answer_length': np.mean([len(item['answer']) for item in synthetic_dataset]),
        'duplicates': 0  # Simplified
    }
    
    print(f"Generated {len(synthetic_dataset)} samples")
    print(f"Average question length: {quality_report['avg_question_length']:.1f} chars")
    print(f"Average answer length: {quality_report['avg_answer_length']:.1f} chars")


# Exercise 7: Production Evaluation System
def exercise_7_production_system():
    """
    Exercise 7: Integrate all components into production evaluation system
    
    TODO: Complete the ProductionEvaluationSystem class
    """
    print("\n=== Exercise 7: Production Evaluation System ===")
    
    class ProductionEvaluationSystem:
        def __init__(self):
            # TODO: Initialize all components:
            # - RAGAS metrics calculator
            # - Evaluation pipeline
            # - A/B testing framework
            # - Quality monitor
            # - Custom metrics
            # - Dataset manager
            pass
        
        def continuous_evaluation(self, rag_system, evaluation_config: Dict):
            """TODO: Run continuous evaluation of RAG system"""
            # Hint:
            # 1. Set up monitoring and evaluation schedules
            # 2. Collect responses and evaluate periodically
            # 3. Generate reports and alerts
            # 4. Trigger remediation actions
            pass
        
        def comprehensive_system_audit(self, rag_system, audit_config: Dict) -> Dict:
            """TODO: Perform comprehensive system audit"""
            # Hint:
            # 1. Run full evaluation suite
            # 2. Test with multiple datasets
            # 3. Analyze performance across dimensions
            # 4. Generate audit report with recommendations
            pass
        
        def optimization_recommendations(self, evaluation_results: Dict) -> List[Dict]:
            """TODO: Generate optimization recommendations"""
            # Hint:
            # 1. Analyze evaluation results
            # 2. Identify performance bottlenecks
            # 3. Suggest specific improvements
            # 4. Prioritize recommendations by impact
            pass
        
        def export_evaluation_report(self, results: Dict, format: str = "json") -> str:
            """TODO: Export comprehensive evaluation report"""
            # Hint:
            # 1. Format results according to specified format
            # 2. Include visualizations and summaries
            # 3. Add metadata and timestamps
            # 4. Return formatted report
            pass
    
    print("Production evaluation system initialized")
    print("\n--- Production System Integration ---")
    
    # Initialize integrated system components
    metrics_calc = RAGASMetrics()
    eval_pipeline = EvaluationPipeline(metrics_calc)
    quality_monitor = QualityMonitor(metrics_calc, {
        'faithfulness': 0.7,
        'answer_relevancy': 0.8,
        'response_time': 2.0
    })
    
    # Mock comprehensive evaluation
    test_responses = [
        RAGResponse("What is AI?", "AI is artificial intelligence technology.", ["AI context"], {}),
        RAGResponse("How does ML work?", "ML learns patterns from data.", ["ML context"], {})
    ]
    
    # Run integrated evaluation
    eval_results = eval_pipeline.evaluate_batch(test_responses)
    report = eval_pipeline.generate_evaluation_report(eval_results)
    
    # Monitor responses
    for response in test_responses:
        quality_monitor.monitor_response(response)
    
    dashboard = quality_monitor.generate_monitoring_dashboard()
    
    print(f"Evaluated {len(eval_results)} responses")
    print(f"Average score: {report['metrics']['overall_score']['mean']:.3f}")
    print(f"System health: {dashboard['system_health']}")
    print(f"Recommendations: {len(report.get('recommendations', []))} generated")


def main():
    """Run all RAG evaluation exercises"""
    print("🎯 Day 53: RAG Evaluation with RAGAS - Metrics, Quality Assessment & Optimization")
    print("=" * 80)
    
    exercises = [
        exercise_1_core_metrics,
        exercise_2_evaluation_pipeline,
        exercise_3_ab_testing,
        exercise_4_quality_monitoring,
        exercise_5_custom_metrics,
        exercise_6_dataset_management,
        exercise_7_production_system
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n📋 Starting Exercise {i}")
        try:
            exercise()
            print(f"✅ Exercise {i} setup complete")
        except Exception as e:
            print(f"❌ Exercise {i} error: {e}")
        
        if i < len(exercises):
            input("\nPress Enter to continue to the next exercise...")
    
    print("\n🎉 All exercises completed!")
    print("\nNext steps:")
    print("1. Implement the TODO sections in each exercise")
    print("2. Test your implementations with the provided data")
    print("3. Experiment with different evaluation strategies")
    print("4. Review the solution file for complete implementations")
    print("5. Consider integrating with real RAGAS library for production use")


if __name__ == "__main__":
    main()
