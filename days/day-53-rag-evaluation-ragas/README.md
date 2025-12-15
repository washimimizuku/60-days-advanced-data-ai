# Day 53: RAG Evaluation with RAGAS - Metrics, Quality Assessment & Optimization

## Learning Objectives
By the end of this session, you will be able to:
- Implement comprehensive RAG evaluation using the RAGAS framework
- Understand and apply key RAG metrics: faithfulness, answer relevancy, context precision, and context recall
- Build automated evaluation pipelines for continuous RAG system improvement
- Design A/B testing frameworks for RAG system optimization
- Create production monitoring dashboards for RAG quality assessment

## Theory (15 minutes)

### What is RAGAS?

RAGAS (Retrieval Augmented Generation Assessment) is a comprehensive framework for evaluating RAG systems. It provides reference-free evaluation metrics that assess both retrieval quality and generation quality without requiring ground truth answers.

### Core RAGAS Metrics

#### 1. Faithfulness
Measures how factually accurate the generated answer is based on the retrieved context.

```python
# Faithfulness calculation
def calculate_faithfulness(answer, contexts):
    """
    Faithfulness = Number of claims in answer supported by context / Total claims in answer
    """
    # Extract claims from answer
    claims = extract_claims(answer)
    
    # Check each claim against contexts
    supported_claims = 0
    for claim in claims:
        if is_supported_by_context(claim, contexts):
            supported_claims += 1
    
    return supported_claims / len(claims) if claims else 0
```

#### 2. Answer Relevancy
Evaluates how relevant the generated answer is to the original question.

```python
# Answer Relevancy using question generation
def calculate_answer_relevancy(question, answer, llm):
    """
    Generate questions from answer and measure similarity to original question
    """
    # Generate questions from the answer
    generated_questions = llm.generate_questions_from_answer(answer)
    
    # Calculate semantic similarity
    similarities = []
    for gen_q in generated_questions:
        similarity = semantic_similarity(question, gen_q)
        similarities.append(similarity)
    
    return np.mean(similarities)
```

#### 3. Context Precision
Measures how relevant the retrieved contexts are to the question.

```python
# Context Precision calculation
def calculate_context_precision(question, contexts, answer):
    """
    Precision@K = Relevant contexts in top K / K
    """
    relevant_contexts = 0
    for i, context in enumerate(contexts):
        if is_context_relevant(question, context, answer):
            relevant_contexts += 1
    
    return relevant_contexts / len(contexts) if contexts else 0
```

#### 4. Context Recall
Evaluates whether all relevant information needed to answer the question was retrieved.

```python
# Context Recall calculation
def calculate_context_recall(ground_truth_contexts, retrieved_contexts):
    """
    Recall = Retrieved relevant contexts / Total relevant contexts
    """
    relevant_retrieved = 0
    for gt_context in ground_truth_contexts:
        for ret_context in retrieved_contexts:
            if contexts_overlap(gt_context, ret_context):
                relevant_retrieved += 1
                break
    
    return relevant_retrieved / len(ground_truth_contexts) if ground_truth_contexts else 0
```

### Advanced RAGAS Implementation

#### Complete Evaluation Pipeline

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
)

class RAGASEvaluator:
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_relevancy
        ]
    
    def evaluate_rag_system(self, dataset):
        """Comprehensive RAG evaluation"""
        results = evaluate(
            dataset=dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings
        )
        return results
```

#### Custom Metric Development

```python
class CustomRAGMetric:
    def __init__(self, name, description):
        self.name = name
        self.description = description
    
    def calculate_response_completeness(self, question, answer, contexts):
        """Custom metric: How complete is the answer?"""
        # Extract key aspects from question
        question_aspects = self.extract_aspects(question)
        
        # Check coverage in answer
        covered_aspects = 0
        for aspect in question_aspects:
            if self.aspect_covered_in_answer(aspect, answer):
                covered_aspects += 1
        
        return covered_aspects / len(question_aspects) if question_aspects else 0
    
    def calculate_context_diversity(self, contexts):
        """Custom metric: How diverse are the retrieved contexts?"""
        if len(contexts) <= 1:
            return 0
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(contexts)):
            for j in range(i+1, len(contexts)):
                sim = semantic_similarity(contexts[i], contexts[j])
                similarities.append(sim)
        
        # Diversity = 1 - average similarity
        return 1 - np.mean(similarities)
```

### Production Evaluation Strategies

#### 1. Automated Evaluation Pipeline

```python
class ProductionRAGEvaluator:
    def __init__(self, rag_system, evaluator):
        self.rag_system = rag_system
        self.evaluator = evaluator
        self.evaluation_history = []
    
    def continuous_evaluation(self, test_queries, interval_hours=24):
        """Run continuous evaluation"""
        while True:
            # Generate responses
            responses = []
            for query in test_queries:
                response = self.rag_system.query(query)
                responses.append({
                    'question': query,
                    'answer': response['answer'],
                    'contexts': response['contexts']
                })
            
            # Evaluate
            results = self.evaluator.evaluate(responses)
            
            # Store results
            self.evaluation_history.append({
                'timestamp': datetime.now(),
                'results': results,
                'system_version': self.rag_system.version
            })
            
            # Alert if quality drops
            self.check_quality_alerts(results)
            
            time.sleep(interval_hours * 3600)
```

#### 2. A/B Testing Framework

```python
class RAGABTester:
    def __init__(self, system_a, system_b, evaluator):
        self.system_a = system_a
        self.system_b = system_b
        self.evaluator = evaluator
    
    def run_ab_test(self, test_queries, traffic_split=0.5):
        """Run A/B test between two RAG systems"""
        results_a = []
        results_b = []
        
        for query in test_queries:
            if random.random() < traffic_split:
                # Test system A
                response = self.system_a.query(query)
                results_a.append(response)
            else:
                # Test system B
                response = self.system_b.query(query)
                results_b.append(response)
        
        # Evaluate both systems
        metrics_a = self.evaluator.evaluate(results_a)
        metrics_b = self.evaluator.evaluate(results_b)
        
        # Statistical significance testing
        significance = self.calculate_significance(metrics_a, metrics_b)
        
        return {
            'system_a_metrics': metrics_a,
            'system_b_metrics': metrics_b,
            'statistical_significance': significance,
            'recommendation': self.make_recommendation(metrics_a, metrics_b, significance)
        }
```

#### 3. Real-time Quality Monitoring

```python
class RAGQualityMonitor:
    def __init__(self, evaluator, thresholds):
        self.evaluator = evaluator
        self.thresholds = thresholds
        self.alerts = []
    
    def monitor_response(self, question, answer, contexts):
        """Monitor individual response quality"""
        # Quick evaluation
        faithfulness_score = self.evaluator.calculate_faithfulness(answer, contexts)
        relevancy_score = self.evaluator.calculate_answer_relevancy(question, answer)
        
        # Check thresholds
        alerts = []
        if faithfulness_score < self.thresholds['faithfulness']:
            alerts.append(f"Low faithfulness: {faithfulness_score:.2f}")
        
        if relevancy_score < self.thresholds['relevancy']:
            alerts.append(f"Low relevancy: {relevancy_score:.2f}")
        
        # Log and alert if needed
        if alerts:
            self.log_quality_issue(question, answer, alerts)
        
        return {
            'faithfulness': faithfulness_score,
            'relevancy': relevancy_score,
            'alerts': alerts
        }
```

### Evaluation Dataset Creation

#### Synthetic Dataset Generation

```python
class RAGDatasetGenerator:
    def __init__(self, llm, document_corpus):
        self.llm = llm
        self.document_corpus = document_corpus
    
    def generate_evaluation_dataset(self, num_samples=100):
        """Generate synthetic evaluation dataset"""
        dataset = []
        
        for _ in range(num_samples):
            # Sample random document
            doc = random.choice(self.document_corpus)
            
            # Generate question from document
            question = self.llm.generate_question_from_text(doc.content)
            
            # Generate ground truth answer
            ground_truth = self.llm.generate_answer_from_text(question, doc.content)
            
            # Create evaluation sample
            sample = {
                'question': question,
                'ground_truth': ground_truth,
                'contexts': [doc.content],
                'metadata': {
                    'source_doc_id': doc.id,
                    'generation_method': 'synthetic'
                }
            }
            dataset.append(sample)
        
        return dataset
```

### Advanced Evaluation Techniques

#### 1. Multi-dimensional Evaluation

```python
class MultiDimensionalEvaluator:
    def __init__(self):
        self.dimensions = {
            'accuracy': ['faithfulness', 'factual_consistency'],
            'relevance': ['answer_relevancy', 'context_precision'],
            'completeness': ['context_recall', 'answer_completeness'],
            'efficiency': ['response_time', 'token_usage'],
            'safety': ['toxicity_score', 'bias_detection']
        }
    
    def comprehensive_evaluation(self, rag_responses):
        """Multi-dimensional RAG evaluation"""
        results = {}
        
        for dimension, metrics in self.dimensions.items():
            dimension_scores = []
            for metric in metrics:
                scores = self.calculate_metric(metric, rag_responses)
                dimension_scores.extend(scores)
            
            results[dimension] = {
                'mean': np.mean(dimension_scores),
                'std': np.std(dimension_scores),
                'min': np.min(dimension_scores),
                'max': np.max(dimension_scores)
            }
        
        return results
```

#### 2. Human-in-the-Loop Evaluation

```python
class HumanEvaluationInterface:
    def __init__(self, evaluation_ui):
        self.ui = evaluation_ui
        self.human_feedback = []
    
    def collect_human_feedback(self, rag_responses, evaluators):
        """Collect human evaluation feedback"""
        for response in rag_responses:
            feedback = self.ui.present_for_evaluation(
                question=response['question'],
                answer=response['answer'],
                contexts=response['contexts'],
                evaluators=evaluators
            )
            
            self.human_feedback.append({
                'response_id': response['id'],
                'feedback': feedback,
                'timestamp': datetime.now()
            })
        
        return self.analyze_human_feedback()
    
    def analyze_human_feedback(self):
        """Analyze collected human feedback"""
        # Calculate inter-annotator agreement
        agreement = self.calculate_inter_annotator_agreement()
        
        # Identify patterns in feedback
        patterns = self.identify_feedback_patterns()
        
        return {
            'agreement_score': agreement,
            'feedback_patterns': patterns,
            'recommendations': self.generate_recommendations()
        }
```

### Why RAG Evaluation with RAGAS Matters

1. **Quality Assurance**: Systematic evaluation ensures consistent RAG system performance
2. **Continuous Improvement**: Metrics-driven optimization leads to better user experiences
3. **Production Confidence**: Automated evaluation provides confidence in system deployments
4. **Cost Optimization**: Identify inefficiencies and optimize resource usage
5. **Compliance**: Meet quality standards and regulatory requirements

### Real-world Applications

- **Enterprise Search**: Evaluate internal knowledge base accuracy
- **Customer Support**: Monitor chatbot response quality
- **Research Assistance**: Assess academic information retrieval
- **Legal Discovery**: Ensure factual accuracy in legal document analysis
- **Medical Information**: Validate clinical decision support systems

## Exercise (40 minutes)
Complete the hands-on exercises in `exercise.py` to build comprehensive RAG evaluation systems using RAGAS metrics, automated pipelines, and quality monitoring.

## Resources
- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub Repository](https://github.com/explodinggradients/ragas)
- [RAG Evaluation Best Practices](https://arxiv.org/abs/2401.15884)
- [LangChain RAG Evaluation](https://python.langchain.com/docs/guides/evaluation/string/rag)
- [Weights & Biases RAG Evaluation](https://wandb.ai/wandb_fc/rag-evaluation)

## Next Steps
- Complete the RAGAS evaluation exercises
- Review automated evaluation patterns
- Take the quiz to test your understanding
- Move to Day 54: Project - Production RAG System
