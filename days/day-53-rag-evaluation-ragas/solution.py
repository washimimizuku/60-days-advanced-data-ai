"""
Day 53: RAG Evaluation with RAGAS - Complete Solutions
Production-ready implementations of RAG evaluation metrics and monitoring systems
"""

import numpy as np
import pandas as pd
import time
import json
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import random
from collections import defaultdict
import asyncio
import statistics
from scipy import stats
import hashlib
import logging


@dataclass
class RAGResponse:
    """RAG system response structure"""
    question: str
    answer: str
    contexts: List[str]
    metadata: Dict
    timestamp: Optional[datetime] = None
    response_id: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.response_id is None:
            self.response_id = hashlib.md5(
                f"{self.question}{self.answer}{self.timestamp}".encode()
            ).hexdigest()[:8]


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
    timestamp: datetime
    metadata: Dict


class MockLLM:
    """Production-quality mock LLM for evaluation"""
    
    def __init__(self):
        self.question_templates = [
            "What is {topic}?",
            "How does {topic} work?",
            "Why is {topic} important?",
            "What are the benefits of {topic}?",
            "How can {topic} be implemented?",
            "What are the challenges with {topic}?",
            "When should you use {topic}?",
            "What are examples of {topic}?"
        ]
    
    def generate_questions_from_answer(self, answer: str, num_questions: int = 3) -> List[str]:
        """Generate questions that the answer could address"""
        words = answer.split()
        if not words:
            return ["What is this about?"] * num_questions
        
        # Extract key topics (nouns and important terms)
        key_terms = []
        for word in words:
            if len(word) > 3 and word.lower() not in ['this', 'that', 'with', 'from', 'they', 'have', 'been']:
                key_terms.append(word.lower())
        
        if not key_terms:
            key_terms = [words[0].lower()]
        
        questions = []
        for i in range(num_questions):
            template = self.question_templates[i % len(self.question_templates)]
            topic = key_terms[i % len(key_terms)]
            question = template.format(topic=topic)
            questions.append(question)
        
        return questions
    
    def extract_claims(self, text: str) -> List[str]:
        """Extract factual claims from text"""
        # Split by sentence-ending punctuation
        import re
        sentences = re.split(r'[.!?]+', text)
        claims = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Filter out very short fragments
                claims.append(sentence)
        
        return claims
    
    def check_claim_support(self, claim: str, contexts: List[str]) -> bool:
        """Check if claim is supported by contexts"""
        claim_words = set(claim.lower().split())
        claim_words = {w for w in claim_words if len(w) > 2}  # Filter short words
        
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(claim_words.intersection(context_words))
            
            # Support if significant overlap (at least 40% of claim words)
            if claim_words and overlap >= len(claim_words) * 0.4:
                return True
        
        return False
    
    def generate_question_from_text(self, text: str) -> str:
        """Generate a question from given text"""
        words = text.split()
        if len(words) < 5:
            return "What is this about?"
        
        # Simple question generation based on text content
        key_word = words[min(3, len(words)-1)]
        templates = [
            f"What is {key_word}?",
            f"How does {key_word} work?",
            f"Why is {key_word} important?"
        ]
        
        return random.choice(templates)


class MockEmbedding:
    """Production-quality mock embedding model"""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate consistent mock embeddings"""
        embeddings = []
        
        for text in texts:
            # Create deterministic embedding based on text content
            text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(text_hash % (2**32))
            
            embedding = np.random.rand(self.dimension)
            
            # Add semantic features based on text characteristics
            words = text.lower().split()
            unique_words = set(words)
            
            # Length features
            embedding[0] = min(len(text) / 1000, 1.0)
            embedding[1] = min(len(words) / 100, 1.0)
            
            # Diversity features
            embedding[2] = len(unique_words) / len(words) if words else 0
            
            # Domain features (simple keyword matching)
            tech_words = {'machine', 'learning', 'ai', 'algorithm', 'data', 'model'}
            embedding[3] = len(tech_words.intersection(unique_words)) / len(tech_words)
            
            # Question vs statement features
            embedding[4] = 1.0 if text.strip().endswith('?') else 0.0
            
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        emb1 = self.encode([text1])[0]
        emb2 = self.encode([text2])[0]
        
        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm_product = np.linalg.norm(emb1) * np.linalg.norm(emb2)
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product


# Solution 1: Core RAGAS Metrics Implementation
class RAGASMetrics:
    """Complete implementation of RAGAS metrics"""
    
    def __init__(self):
        self.llm = MockLLM()
        self.embedding = MockEmbedding()
        self.similarity_threshold = 0.7
    
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
            if question_context_sim >= self.similarity_threshold or answer_context_sim >= self.similarity_threshold:
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
                if similarity >= self.similarity_threshold:
                    covered_contexts += 1
                    break
        
        return covered_contexts / len(ground_truth_contexts)
    
    def calculate_overall_score(self, faithfulness: float, answer_relevancy: float,
                              context_precision: float, context_recall: float) -> float:
        """Calculate weighted overall score"""
        weights = {
            'faithfulness': 0.3,
            'answer_relevancy': 0.3,
            'context_precision': 0.2,
            'context_recall': 0.2
        }
        
        overall = (
            weights['faithfulness'] * faithfulness +
            weights['answer_relevancy'] * answer_relevancy +
            weights['context_precision'] * context_precision +
            weights['context_recall'] * context_recall
        )
        
        return overall
    
    def generate_alerts(self, metrics: Dict[str, float], thresholds: Dict[str, float]) -> List[str]:
        """Generate alerts for low-quality metrics"""
        alerts = []
        
        for metric, value in metrics.items():
            if metric in thresholds and value < thresholds[metric]:
                alerts.append(f"Low {metric}: {value:.3f} < {thresholds[metric]:.3f}")
        
        return alerts
    
    def evaluate_response(self, response: RAGResponse, 
                        ground_truth_contexts: Optional[List[str]] = None,
                        thresholds: Optional[Dict[str, float]] = None) -> EvaluationResult:
        """Complete evaluation of a single response"""
        if thresholds is None:
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
        
        # Calculate overall score
        overall_score = self.calculate_overall_score(
            faithfulness, answer_relevancy, context_precision, context_recall)
        
        # Generate alerts
        metrics_dict = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'overall_score': overall_score
        }
        alerts = self.generate_alerts(metrics_dict, thresholds)
        
        return EvaluationResult(
            response_id=response.response_id,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            overall_score=overall_score,
            alerts=alerts,
            timestamp=datetime.now(),
            metadata=response.metadata
        )


# Solution 2: Automated Evaluation Pipeline
class EvaluationPipeline:
    """Production-ready evaluation pipeline"""
    
    def __init__(self, metrics_calculator: RAGASMetrics):
        self.metrics = metrics_calculator
        self.evaluation_history = []
        self.thresholds = {
            'faithfulness': 0.7,
            'answer_relevancy': 0.8,
            'context_precision': 0.6,
            'overall_score': 0.7
        }
    
    def evaluate_batch(self, responses: List[RAGResponse], 
                      ground_truth_contexts: Optional[Dict[str, List[str]]] = None) -> List[EvaluationResult]:
        """Evaluate a batch of responses"""
        results = []
        
        for response in responses:
            # Get ground truth for this response if available
            gt_contexts = None
            if ground_truth_contexts and response.response_id in ground_truth_contexts:
                gt_contexts = ground_truth_contexts[response.response_id]
            
            # Evaluate response
            result = self.metrics.evaluate_response(response, gt_contexts, self.thresholds)
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
        recall_scores = [r.context_recall for r in results]
        overall_scores = [r.overall_score for r in results]
        
        # Calculate statistics
        def calc_stats(scores):
            return {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'median': np.median(scores),
                'p25': np.percentile(scores, 25),
                'p75': np.percentile(scores, 75)
            }
        
        # Count alerts
        total_alerts = sum(len(r.alerts) for r in results)
        responses_with_alerts = sum(1 for r in results if r.alerts)
        
        # Identify common alert types
        alert_types = defaultdict(int)
        for result in results:
            for alert in result.alerts:
                alert_type = alert.split(':')[0]  # Extract metric name
                alert_types[alert_type] += 1
        
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
                'context_recall': calc_stats(recall_scores),
                'overall_score': calc_stats(overall_scores)
            },
            'alerts': {
                'common_types': dict(alert_types),
                'threshold_violations': self._analyze_threshold_violations(results)
            },
            'recommendations': self._generate_recommendations(results)
        }
        
        return report
    
    def _analyze_threshold_violations(self, results: List[EvaluationResult]) -> Dict:
        """Analyze threshold violations"""
        violations = defaultdict(int)
        
        for result in results:
            if result.faithfulness < self.thresholds['faithfulness']:
                violations['faithfulness'] += 1
            if result.answer_relevancy < self.thresholds['answer_relevancy']:
                violations['answer_relevancy'] += 1
            if result.context_precision < self.thresholds['context_precision']:
                violations['context_precision'] += 1
            if result.overall_score < self.thresholds['overall_score']:
                violations['overall_score'] += 1
        
        return dict(violations)
    
    def _generate_recommendations(self, results: List[EvaluationResult]) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Analyze common issues
        faithfulness_scores = [r.faithfulness for r in results]
        relevancy_scores = [r.answer_relevancy for r in results]
        precision_scores = [r.context_precision for r in results]
        
        avg_faithfulness = np.mean(faithfulness_scores)
        avg_relevancy = np.mean(relevancy_scores)
        avg_precision = np.mean(precision_scores)
        
        if avg_faithfulness < 0.7:
            recommendations.append(
                "Low faithfulness detected. Consider improving context quality or "
                "implementing better fact-checking mechanisms."
            )
        
        if avg_relevancy < 0.8:
            recommendations.append(
                "Low answer relevancy detected. Review answer generation prompts "
                "and ensure responses directly address the questions."
            )
        
        if avg_precision < 0.6:
            recommendations.append(
                "Low context precision detected. Improve retrieval system to "
                "return more relevant contexts."
            )
        
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
        
        # Filter by time window
        cutoff_date = datetime.now() - timedelta(days=window_days)
        recent_results = [
            r for r in self.evaluation_history 
            if r.timestamp >= cutoff_date
        ]
        
        if not recent_results:
            return {"error": f"No results in the last {window_days} days"}
        
        # Group by day
        daily_metrics = defaultdict(list)
        for result in recent_results:
            day = result.timestamp.date()
            daily_metrics[day].append(result)
        
        # Calculate daily averages
        daily_averages = {}
        for day, day_results in daily_metrics.items():
            daily_averages[day.isoformat()] = {
                'faithfulness': np.mean([r.faithfulness for r in day_results]),
                'answer_relevancy': np.mean([r.answer_relevancy for r in day_results]),
                'context_precision': np.mean([r.context_precision for r in day_results]),
                'overall_score': np.mean([r.overall_score for r in day_results]),
                'count': len(day_results)
            }
        
        # Calculate trends
        dates = sorted(daily_averages.keys())
        if len(dates) >= 2:
            first_day = daily_averages[dates[0]]
            last_day = daily_averages[dates[-1]]
            
            trends = {
                'faithfulness': last_day['faithfulness'] - first_day['faithfulness'],
                'answer_relevancy': last_day['answer_relevancy'] - first_day['answer_relevancy'],
                'context_precision': last_day['context_precision'] - first_day['context_precision'],
                'overall_score': last_day['overall_score'] - first_day['overall_score']
            }
        else:
            trends = {}
        
        return {
            'window_days': window_days,
            'total_results': len(recent_results),
            'daily_averages': daily_averages,
            'trends': trends
        }


# Solution 3: A/B Testing Framework
class RAGABTester:
    """Production A/B testing framework for RAG systems"""
    
    def __init__(self, evaluator: RAGASMetrics):
        self.evaluator = evaluator
        self.test_results = []
    
    def run_ab_test(self, system_a_responses: List[RAGResponse], 
                   system_b_responses: List[RAGResponse],
                   test_name: str,
                   ground_truth_contexts: Optional[Dict[str, List[str]]] = None) -> Dict:
        """Run comprehensive A/B test between two RAG systems"""
        
        # Evaluate both systems
        results_a = []
        results_b = []
        
        for response in system_a_responses:
            gt_contexts = ground_truth_contexts.get(response.response_id) if ground_truth_contexts else None
            result = self.evaluator.evaluate_response(response, gt_contexts)
            results_a.append(result)
        
        for response in system_b_responses:
            gt_contexts = ground_truth_contexts.get(response.response_id) if ground_truth_contexts else None
            result = self.evaluator.evaluate_response(response, gt_contexts)
            results_b.append(result)
        
        # Extract metric scores
        metrics_a = self._extract_metric_scores(results_a)
        metrics_b = self._extract_metric_scores(results_b)
        
        # Calculate statistical significance
        significance_results = {}
        for metric in metrics_a.keys():
            significance_results[metric] = self.calculate_statistical_significance(
                metrics_a[metric], metrics_b[metric]
            )
        
        # Generate recommendations
        recommendation = self._make_recommendation(metrics_a, metrics_b, significance_results)
        
        test_result = {
            'test_name': test_name,
            'timestamp': datetime.now(),
            'system_a': {
                'sample_size': len(results_a),
                'metrics': self._calculate_summary_stats(metrics_a),
                'alert_rate': sum(1 for r in results_a if r.alerts) / len(results_a)
            },
            'system_b': {
                'sample_size': len(results_b),
                'metrics': self._calculate_summary_stats(metrics_b),
                'alert_rate': sum(1 for r in results_b if r.alerts) / len(results_b)
            },
            'statistical_significance': significance_results,
            'recommendation': recommendation,
            'effect_sizes': self._calculate_effect_sizes(metrics_a, metrics_b)
        }
        
        self.test_results.append(test_result)
        return test_result
    
    def _extract_metric_scores(self, results: List[EvaluationResult]) -> Dict[str, List[float]]:
        """Extract metric scores from evaluation results"""
        return {
            'faithfulness': [r.faithfulness for r in results],
            'answer_relevancy': [r.answer_relevancy for r in results],
            'context_precision': [r.context_precision for r in results],
            'context_recall': [r.context_recall for r in results],
            'overall_score': [r.overall_score for r in results]
        }
    
    def _calculate_summary_stats(self, metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
        """Calculate summary statistics for metrics"""
        summary = {}
        for metric, scores in metrics.items():
            summary[metric] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'median': np.median(scores),
                'min': np.min(scores),
                'max': np.max(scores)
            }
        return summary
    
    def calculate_statistical_significance(self, scores_a: List[float], 
                                        scores_b: List[float]) -> Dict:
        """Calculate statistical significance of differences"""
        if len(scores_a) < 2 or len(scores_b) < 2:
            return {
                'test': 'insufficient_data',
                'p_value': None,
                'significant': False,
                'confidence_interval': None
            }
        
        # Perform t-test
        try:
            t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
            
            # Calculate confidence interval for difference in means
            mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
            std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)
            n_a, n_b = len(scores_a), len(scores_b)
            
            # Pooled standard error
            pooled_se = np.sqrt((std_a**2 / n_a) + (std_b**2 / n_b))
            
            # 95% confidence interval
            diff = mean_b - mean_a
            margin_error = 1.96 * pooled_se  # Approximate for large samples
            ci_lower = diff - margin_error
            ci_upper = diff + margin_error
            
            return {
                'test': 't_test',
                'p_value': p_value,
                'significant': p_value < 0.05,
                'confidence_interval': (ci_lower, ci_upper),
                'mean_difference': diff,
                't_statistic': t_stat
            }
        
        except Exception as e:
            return {
                'test': 'error',
                'error': str(e),
                'significant': False
            }
    
    def _calculate_effect_sizes(self, metrics_a: Dict[str, List[float]], 
                              metrics_b: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate Cohen's d effect sizes"""
        effect_sizes = {}
        
        for metric in metrics_a.keys():
            scores_a = metrics_a[metric]
            scores_b = metrics_b[metric]
            
            if len(scores_a) < 2 or len(scores_b) < 2:
                effect_sizes[metric] = 0.0
                continue
            
            mean_a, mean_b = np.mean(scores_a), np.mean(scores_b)
            std_a, std_b = np.std(scores_a, ddof=1), np.std(scores_b, ddof=1)
            
            # Pooled standard deviation
            n_a, n_b = len(scores_a), len(scores_b)
            pooled_std = np.sqrt(((n_a - 1) * std_a**2 + (n_b - 1) * std_b**2) / (n_a + n_b - 2))
            
            # Cohen's d
            if pooled_std > 0:
                cohens_d = (mean_b - mean_a) / pooled_std
            else:
                cohens_d = 0.0
            
            effect_sizes[metric] = cohens_d
        
        return effect_sizes
    
    def _make_recommendation(self, metrics_a: Dict[str, List[float]], 
                           metrics_b: Dict[str, List[float]], 
                           significance_results: Dict) -> Dict:
        """Generate recommendation based on test results"""
        
        # Count significant improvements
        significant_improvements = 0
        significant_degradations = 0
        
        for metric, sig_result in significance_results.items():
            if sig_result.get('significant', False):
                mean_diff = sig_result.get('mean_difference', 0)
                if mean_diff > 0:
                    significant_improvements += 1
                else:
                    significant_degradations += 1
        
        # Calculate overall performance
        overall_a = np.mean(metrics_a['overall_score'])
        overall_b = np.mean(metrics_b['overall_score'])
        
        if significant_improvements > significant_degradations and overall_b > overall_a:
            recommendation = "system_b"
            confidence = "high" if significant_improvements >= 3 else "medium"
            reason = f"System B shows {significant_improvements} significant improvements"
        elif significant_degradations > significant_improvements and overall_a > overall_b:
            recommendation = "system_a"
            confidence = "high" if significant_degradations >= 3 else "medium"
            reason = f"System A performs better with {significant_degradations} fewer degradations"
        else:
            recommendation = "inconclusive"
            confidence = "low"
            reason = "No clear winner - consider longer test or different metrics"
        
        return {
            'recommended_system': recommendation,
            'confidence': confidence,
            'reason': reason,
            'significant_improvements': significant_improvements,
            'significant_degradations': significant_degradations
        }
    
    def generate_ab_report(self, test_results: Dict) -> str:
        """Generate human-readable A/B test report"""
        report = f"""
A/B Test Report: {test_results['test_name']}
{'=' * 50}

Test Overview:
- System A Sample Size: {test_results['system_a']['sample_size']}
- System B Sample Size: {test_results['system_b']['sample_size']}
- Test Date: {test_results['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}

Performance Comparison:
"""
        
        # Add metric comparisons
        for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'overall_score']:
            mean_a = test_results['system_a']['metrics'][metric]['mean']
            mean_b = test_results['system_b']['metrics'][metric]['mean']
            sig_result = test_results['statistical_significance'][metric]
            
            report += f"\n{metric.replace('_', ' ').title()}:\n"
            report += f"  System A: {mean_a:.3f}\n"
            report += f"  System B: {mean_b:.3f}\n"
            report += f"  Difference: {mean_b - mean_a:+.3f}\n"
            report += f"  Significant: {'Yes' if sig_result.get('significant', False) else 'No'}\n"
        
        # Add recommendation
        rec = test_results['recommendation']
        report += f"\nRecommendation:\n"
        report += f"  Recommended System: {rec['recommended_system'].upper()}\n"
        report += f"  Confidence: {rec['confidence'].upper()}\n"
        report += f"  Reason: {rec['reason']}\n"
        
        return report
    
    def power_analysis(self, effect_size: float, alpha: float = 0.05, 
                      power: float = 0.8) -> int:
        """Calculate required sample size for test"""
        # Simplified power analysis calculation
        # In production, use statsmodels or similar library
        
        if effect_size <= 0:
            return float('inf')
        
        # Approximate formula for two-sample t-test
        z_alpha = 1.96  # For alpha = 0.05
        z_beta = 0.84   # For power = 0.8
        
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        
        return int(np.ceil(n))


# Solution 4: Real-time Quality Monitoring
class QualityMonitor:
    """Production-ready real-time quality monitoring"""
    
    def __init__(self, metrics_calculator: RAGASMetrics, alert_thresholds: Dict[str, float]):
        self.metrics = metrics_calculator
        self.thresholds = alert_thresholds
        self.monitoring_data = []
        self.alerts = []
        self.baseline_metrics = {}
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
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
            'timestamp': datetime.now(),
            'metrics': {
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy,
                'response_time': response_time
            },
            'alerts': []
        }
        
        # Check thresholds and generate alerts
        if faithfulness < self.thresholds.get('faithfulness', 0.7):
            alert = {
                'type': 'low_faithfulness',
                'severity': 'high' if faithfulness < 0.5 else 'medium',
                'message': f"Low faithfulness score: {faithfulness:.3f}",
                'response_id': response.response_id,
                'timestamp': datetime.now()
            }
            monitoring_result['alerts'].append(alert)
            self.alerts.append(alert)
        
        if answer_relevancy < self.thresholds.get('answer_relevancy', 0.8):
            alert = {
                'type': 'low_relevancy',
                'severity': 'medium',
                'message': f"Low answer relevancy: {answer_relevancy:.3f}",
                'response_id': response.response_id,
                'timestamp': datetime.now()
            }
            monitoring_result['alerts'].append(alert)
            self.alerts.append(alert)
        
        if response_time > self.thresholds.get('response_time', 2.0):
            alert = {
                'type': 'slow_response',
                'severity': 'low',
                'message': f"Slow response time: {response_time:.2f}s",
                'response_id': response.response_id,
                'timestamp': datetime.now()
            }
            monitoring_result['alerts'].append(alert)
            self.alerts.append(alert)
        
        # Store monitoring data
        self.monitoring_data.append(monitoring_result)
        
        # Log alerts
        for alert in monitoring_result['alerts']:
            self.logger.warning(f"Quality Alert: {alert['message']}")
        
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
            'answer_relevancy': np.mean([d['metrics']['answer_relevancy'] for d in recent_data]),
            'response_time': np.mean([d['metrics']['response_time'] for d in recent_data])
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
                        'severity': 'high' if degradation > 0.2 else 'medium',
                        'timestamp': datetime.now()
                    })
        
        return degradation_alerts
    
    def set_baseline_metrics(self, baseline_data: List[Dict]):
        """Set baseline metrics for comparison"""
        if not baseline_data:
            return
        
        self.baseline_metrics = {
            'faithfulness': np.mean([d['metrics']['faithfulness'] for d in baseline_data]),
            'answer_relevancy': np.mean([d['metrics']['answer_relevancy'] for d in baseline_data]),
            'response_time': np.mean([d['metrics']['response_time'] for d in baseline_data])
        }
        
        self.logger.info(f"Baseline metrics set: {self.baseline_metrics}")
    
    def generate_monitoring_dashboard(self) -> Dict:
        """Generate real-time monitoring dashboard data"""
        if not self.monitoring_data:
            return {"error": "No monitoring data available"}
        
        # Recent performance (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_data = [
            d for d in self.monitoring_data 
            if d['timestamp'] >= one_hour_ago
        ]
        
        if not recent_data:
            recent_data = self.monitoring_data[-10:]  # Last 10 responses
        
        # Calculate current metrics
        current_metrics = {
            'faithfulness': {
                'current': np.mean([d['metrics']['faithfulness'] for d in recent_data]),
                'trend': self._calculate_trend([d['metrics']['faithfulness'] for d in recent_data])
            },
            'answer_relevancy': {
                'current': np.mean([d['metrics']['answer_relevancy'] for d in recent_data]),
                'trend': self._calculate_trend([d['metrics']['answer_relevancy'] for d in recent_data])
            },
            'response_time': {
                'current': np.mean([d['metrics']['response_time'] for d in recent_data]),
                'trend': self._calculate_trend([d['metrics']['response_time'] for d in recent_data])
            }
        }
        
        # Active alerts (last 24 hours)
        twenty_four_hours_ago = datetime.now() - timedelta(hours=24)
        active_alerts = [
            alert for alert in self.alerts 
            if alert['timestamp'] >= twenty_four_hours_ago
        ]
        
        # System health status
        health_status = self._calculate_health_status(current_metrics, active_alerts)
        
        return {
            'timestamp': datetime.now(),
            'system_health': health_status,
            'current_metrics': current_metrics,
            'active_alerts': len(active_alerts),
            'alert_breakdown': self._categorize_alerts(active_alerts),
            'total_responses_monitored': len(self.monitoring_data),
            'recent_response_count': len(recent_data)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "degrading"
        else:
            return "stable"
    
    def _calculate_health_status(self, metrics: Dict, alerts: List[Dict]) -> str:
        """Calculate overall system health status"""
        # Count high severity alerts
        high_severity_alerts = sum(1 for alert in alerts if alert.get('severity') == 'high')
        
        # Check if metrics are above thresholds
        faithfulness_ok = metrics['faithfulness']['current'] >= self.thresholds.get('faithfulness', 0.7)
        relevancy_ok = metrics['answer_relevancy']['current'] >= self.thresholds.get('answer_relevancy', 0.8)
        response_time_ok = metrics['response_time']['current'] <= self.thresholds.get('response_time', 2.0)
        
        if high_severity_alerts > 0:
            return "critical"
        elif not (faithfulness_ok and relevancy_ok and response_time_ok):
            return "warning"
        else:
            return "healthy"
    
    def _categorize_alerts(self, alerts: List[Dict]) -> Dict[str, int]:
        """Categorize alerts by type"""
        categories = defaultdict(int)
        for alert in alerts:
            categories[alert['type']] += 1
        return dict(categories)
    
    def auto_remediation_suggestions(self, alert: Dict) -> List[str]:
        """Generate automatic remediation suggestions"""
        suggestions = []
        
        alert_type = alert['type']
        
        if alert_type == 'low_faithfulness':
            suggestions.extend([
                "Review and improve context retrieval quality",
                "Implement fact-checking mechanisms",
                "Retrain or fine-tune the generation model",
                "Add source attribution to responses",
                "Implement confidence scoring for answers"
            ])
        
        elif alert_type == 'low_relevancy':
            suggestions.extend([
                "Review answer generation prompts",
                "Improve question understanding and parsing",
                "Implement better query-answer alignment",
                "Add relevancy scoring to generation pipeline",
                "Consider using different generation strategies"
            ])
        
        elif alert_type == 'slow_response':
            suggestions.extend([
                "Optimize retrieval index performance",
                "Implement response caching",
                "Scale up computational resources",
                "Optimize model inference speed",
                "Consider using smaller, faster models"
            ])
        
        elif alert_type == 'quality_degradation':
            suggestions.extend([
                "Investigate recent system changes",
                "Check data quality and freshness",
                "Review model performance metrics",
                "Consider retraining with recent data",
                "Implement gradual rollback procedures"
            ])
        
        # Prioritize suggestions by impact
        return suggestions[:3]  # Return top 3 suggestions


def demonstrate_complete_evaluation_system():
    """Demonstrate the complete RAG evaluation system"""
    print("🎯 Complete RAG Evaluation System Demonstration")
    print("=" * 60)
    
    # Initialize components
    metrics_calculator = RAGASMetrics()
    evaluation_pipeline = EvaluationPipeline(metrics_calculator)
    ab_tester = RAGABTester(metrics_calculator)
    
    # Quality monitoring setup
    alert_thresholds = {
        'faithfulness': 0.7,
        'answer_relevancy': 0.8,
        'response_time': 2.0
    }
    quality_monitor = QualityMonitor(metrics_calculator, alert_thresholds)
    
    # Sample RAG responses
    sample_responses = [
        RAGResponse(
            question="What is machine learning?",
            answer="Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
            contexts=[
                "Machine learning is a method of data analysis that automates analytical model building.",
                "It is a branch of artificial intelligence based on the idea that systems can learn from data."
            ],
            metadata={"system": "A", "response_time": 1.2}
        ),
        RAGResponse(
            question="How does deep learning work?",
            answer="Deep learning uses artificial neural networks with multiple layers to model and understand complex patterns in data.",
            contexts=[
                "Deep learning is part of machine learning methods based on artificial neural networks.",
                "Neural networks with multiple layers can learn complex representations of data."
            ],
            metadata={"system": "A", "response_time": 1.8}
        ),
        RAGResponse(
            question="What are the applications of AI?",
            answer="AI applications include natural language processing, computer vision, robotics, and autonomous vehicles.",
            contexts=[
                "AI is used in various fields including healthcare, finance, and transportation.",
                "Common applications include image recognition, speech processing, and decision making."
            ],
            metadata={"system": "B", "response_time": 1.5}
        )
    ]
    
    print("\n1. Individual Response Evaluation")
    print("-" * 40)
    for i, response in enumerate(sample_responses[:2], 1):
        result = metrics_calculator.evaluate_response(response)
        print(f"\nResponse {i}:")
        print(f"  Question: {response.question}")
        print(f"  Faithfulness: {result.faithfulness:.3f}")
        print(f"  Answer Relevancy: {result.answer_relevancy:.3f}")
        print(f"  Context Precision: {result.context_precision:.3f}")
        print(f"  Overall Score: {result.overall_score:.3f}")
        if result.alerts:
            print(f"  Alerts: {', '.join(result.alerts)}")
    
    print("\n2. Batch Evaluation Pipeline")
    print("-" * 40)
    batch_results = evaluation_pipeline.evaluate_batch(sample_responses)
    report = evaluation_pipeline.generate_evaluation_report(batch_results)
    
    print(f"Evaluated {report['summary']['total_responses']} responses")
    print(f"Alert rate: {report['summary']['alert_rate']:.2%}")
    print(f"Average overall score: {report['metrics']['overall_score']['mean']:.3f}")
    
    if report['recommendations']:
        print("Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
    
    print("\n3. A/B Testing")
    print("-" * 40)
    # Split responses for A/B test
    system_a_responses = [r for r in sample_responses if r.metadata.get('system') == 'A']
    system_b_responses = [r for r in sample_responses if r.metadata.get('system') == 'B']
    
    if system_a_responses and system_b_responses:
        ab_result = ab_tester.run_ab_test(
            system_a_responses, 
            system_b_responses, 
            "System Comparison Test"
        )
        
        print(f"A/B Test: {ab_result['test_name']}")
        print(f"System A avg score: {ab_result['system_a']['metrics']['overall_score']['mean']:.3f}")
        print(f"System B avg score: {ab_result['system_b']['metrics']['overall_score']['mean']:.3f}")
        print(f"Recommendation: {ab_result['recommendation']['recommended_system']}")
        print(f"Confidence: {ab_result['recommendation']['confidence']}")
    
    print("\n4. Real-time Quality Monitoring")
    print("-" * 40)
    
    # Monitor responses in real-time
    for response in sample_responses:
        monitoring_result = quality_monitor.monitor_response(response)
        if monitoring_result['alerts']:
            print(f"Alert for {response.response_id}: {len(monitoring_result['alerts'])} issues")
    
    # Generate dashboard
    dashboard = quality_monitor.generate_monitoring_dashboard()
    print(f"System Health: {dashboard['system_health']}")
    print(f"Active Alerts: {dashboard['active_alerts']}")
    print(f"Responses Monitored: {dashboard['total_responses_monitored']}")
    
    print("\n5. Performance Trends")
    print("-" * 40)
    trends = evaluation_pipeline.track_performance_over_time(window_days=1)
    if 'daily_averages' in trends:
        print(f"Tracking performance over {trends['window_days']} day(s)")
        print(f"Total results analyzed: {trends['total_results']}")
        
        if trends.get('trends'):
            for metric, trend in trends['trends'].items():
                direction = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                print(f"  {metric}: {direction} {trend:+.3f}")


def main():
    """Run complete RAG evaluation system demonstration"""
    print("🚀 Day 53: RAG Evaluation with RAGAS - Complete Solutions")
    print("=" * 70)
    
    # Run comprehensive demonstration
    demonstrate_complete_evaluation_system()
    
    print("\n✅ All demonstrations completed successfully!")
    print("\nKey Features Demonstrated:")
    print("• Core RAGAS metrics (faithfulness, relevancy, precision, recall)")
    print("• Automated evaluation pipelines with reporting")
    print("• A/B testing framework with statistical significance")
    print("• Real-time quality monitoring and alerting")
    print("• Custom metrics development")
    print("• Performance trend analysis")
    print("• Production-ready evaluation system integration")
    
    print("\nProduction Considerations:")
    print("• Integrate with real RAGAS library for enhanced functionality")
    print("• Implement persistent storage for evaluation history")
    print("• Add visualization dashboards (Grafana, Streamlit)")
    print("• Set up automated alerting (Slack, email, PagerDuty)")
    print("• Configure continuous evaluation schedules")
    print("• Implement evaluation result APIs for system integration")


if __name__ == "__main__":
    main()
