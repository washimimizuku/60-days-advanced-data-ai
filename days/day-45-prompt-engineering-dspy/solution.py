"""
Day 45: Prompt Engineering with DSPy - Complete Solutions

This file contains complete implementations for all exercises.
These solutions demonstrate production-ready DSPy patterns and best practices.
"""

import dspy
from typing import List, Dict, Any, Optional, Tuple
import hashlib
import time
import logging
import json
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock LM for demonstration (replace with actual LM in practice)
class MockLM:
    def __init__(self):
        self.call_count = 0
        self.responses = {
            "capital france": "Paris is the capital of France.",
            "photosynthesis": "Photosynthesis is the process by which plants convert light energy into chemical energy using chlorophyll.",
            "2+2": "2 + 2 = 4",
            "shakespeare": "William Shakespeare wrote Romeo and Juliet, Hamlet, and many other famous plays.",
            "earth orbit": "Earth takes approximately 365.25 days to complete one orbit around the Sun.",
            "python creator": "Python was created by Guido van Rossum and first released in 1991.",
            "water boil": "Water boils at 100 degrees Celsius (212 degrees Fahrenheit) at sea level.",
            "positive sentiment": "This text expresses positive sentiment with high confidence.",
            "negative sentiment": "This text expresses negative sentiment with high confidence.",
            "neutral sentiment": "This text expresses neutral sentiment with moderate confidence."
        }
    
    def __call__(self, prompt, **kwargs):
        self.call_count += 1
        prompt_lower = prompt.lower()
        
        for key, response in self.responses.items():
            if key in prompt_lower:
                return response
        
        return "I need more specific information to provide an accurate answer."

# Set up mock language model
dspy.settings.configure(lm=MockLM())

# Solution 1: Basic DSPy Signature and Module
class BasicQA(dspy.Signature):
    """Answer questions based on the given context."""
    context = dspy.InputField(desc="Background information")
    question = dspy.InputField(desc="Question to answer")
    answer = dspy.OutputField(desc="Answer to the question")

class SimpleQAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(BasicQA)
    
    def forward(self, context, question):
        return self.generate_answer(context=context, question=question)

def solution_1_basic_signature():
    """Solution 1: Basic DSPy signature and module implementation"""
    print("=== Solution 1: Basic DSPy Signature and Module ===")
    
    qa_module = SimpleQAModule()
    
    context = "France is a country in Western Europe. Its capital city is Paris, which is also its largest city."
    question = "What is the capital of France?"
    
    result = qa_module(context=context, question=question)
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {result.answer}")
    print()

# Solution 2: Multi-Field Signature
class SentimentAnalysis(dspy.Signature):
    """Analyze the sentiment of text and provide reasoning."""
    text = dspy.InputField(desc="Text to analyze")
    sentiment = dspy.OutputField(desc="Sentiment: positive, negative, or neutral")
    confidence = dspy.OutputField(desc="Confidence score from 0 to 1")
    reasoning = dspy.OutputField(desc="Explanation for the sentiment classification")

class SentimentAnalyzer(dspy.Module):
    def __init__(self):
        super().__init__()
        self.analyze = dspy.ChainOfThought(SentimentAnalysis)
    
    def forward(self, text):
        return self.analyze(text=text)

def solution_2_sentiment_analysis():
    """Solution 2: Multi-field signature for sentiment analysis"""
    print("=== Solution 2: Multi-Field Signature (Sentiment Analysis) ===")
    
    analyzer = SentimentAnalyzer()
    
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible and disappointing.",
        "The weather is okay today."
    ]
    
    for text in test_texts:
        result = analyzer(text=text)
        print(f"Text: {text}")
        print(f"Sentiment: {result.sentiment}")
        print(f"Confidence: {result.confidence}")
        print(f"Reasoning: {result.reasoning}")
        print("-" * 40)
    print()

# Solution 3: Multi-Stage Reasoning
class FactExtraction(dspy.Signature):
    """Extract key facts from the given context."""
    context = dspy.InputField()
    facts = dspy.OutputField(desc="List of key facts")

class QuestionAnalysis(dspy.Signature):
    """Analyze what the question is asking for."""
    question = dspy.InputField()
    analysis = dspy.OutputField(desc="Analysis of what the question seeks")

class ReasoningStep(dspy.Signature):
    """Reason about the answer using facts and question analysis."""
    facts = dspy.InputField()
    question_analysis = dspy.InputField()
    reasoning = dspy.OutputField(desc="Step-by-step reasoning")

class FinalAnswer(dspy.Signature):
    """Generate final answer based on reasoning."""
    reasoning = dspy.InputField()
    answer = dspy.OutputField(desc="Final answer")

class MultiStageQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_facts = dspy.ChainOfThought(FactExtraction)
        self.analyze_question = dspy.ChainOfThought(QuestionAnalysis)
        self.reason = dspy.ChainOfThought(ReasoningStep)
        self.generate_answer = dspy.ChainOfThought(FinalAnswer)
    
    def forward(self, context, question):
        # Stage 1: Extract facts
        facts_result = self.extract_facts(context=context)
        
        # Stage 2: Analyze question
        analysis_result = self.analyze_question(question=question)
        
        # Stage 3: Reason through the problem
        reasoning_result = self.reason(
            facts=facts_result.facts,
            question_analysis=analysis_result.analysis
        )
        
        # Stage 4: Generate final answer
        answer_result = self.generate_answer(reasoning=reasoning_result.reasoning)
        
        return dspy.Prediction(
            facts=facts_result.facts,
            analysis=analysis_result.analysis,
            reasoning=reasoning_result.reasoning,
            answer=answer_result.answer
        )

def solution_3_multistage_reasoning():
    """Solution 3: Multi-stage reasoning system"""
    print("=== Solution 3: Multi-Stage Reasoning ===")
    
    multistage_qa = MultiStageQA()
    
    context = """
    The solar system consists of the Sun and the objects that orbit it. 
    There are eight planets: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.
    Mercury is the closest planet to the Sun, while Neptune is the farthest.
    Earth is the third planet from the Sun and the only known planet with life.
    """
    
    question = "Which planet is closest to the Sun and what makes Earth special?"
    
    result = multistage_qa(context=context, question=question)
    
    print(f"Context: {context.strip()}")
    print(f"Question: {question}")
    print(f"Facts: {result.facts}")
    print(f"Analysis: {result.analysis}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Answer: {result.answer}")
    print()

# Solution 4: Self-Correcting System
class AnswerValidation(dspy.Signature):
    """Validate if an answer is correct and complete."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.InputField()
    is_correct = dspy.OutputField(desc="yes or no")
    feedback = dspy.OutputField(desc="Feedback on the answer")

class AnswerCorrection(dspy.Signature):
    """Correct an incorrect answer based on feedback."""
    context = dspy.InputField()
    question = dspy.InputField()
    wrong_answer = dspy.InputField()
    feedback = dspy.InputField()
    corrected_answer = dspy.OutputField(desc="Corrected answer")

class SelfCorrectingQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(BasicQA)
        self.validate = dspy.ChainOfThought(AnswerValidation)
        self.correct = dspy.ChainOfThought(AnswerCorrection)
    
    def forward(self, context, question, max_attempts=3):
        for attempt in range(max_attempts):
            # Generate answer
            answer_result = self.generate(context=context, question=question)
            
            # Validate answer
            validation_result = self.validate(
                context=context,
                question=question,
                answer=answer_result.answer
            )
            
            print(f"Attempt {attempt + 1}: {answer_result.answer}")
            print(f"Validation: {validation_result.is_correct} - {validation_result.feedback}")
            
            if validation_result.is_correct.lower() == "yes":
                return dspy.Prediction(
                    answer=answer_result.answer,
                    attempts=attempt + 1,
                    final_validation=validation_result.feedback
                )
            
            # If not the last attempt, try to correct
            if attempt < max_attempts - 1:
                correction_result = self.correct(
                    context=context,
                    question=question,
                    wrong_answer=answer_result.answer,
                    feedback=validation_result.feedback
                )
                # Use corrected answer for next iteration
                answer_result = dspy.Prediction(answer=correction_result.corrected_answer)
        
        return dspy.Prediction(
            answer=answer_result.answer,
            attempts=max_attempts,
            final_validation="Max attempts reached"
        )

def solution_4_self_correction():
    """Solution 4: Self-correcting QA system"""
    print("=== Solution 4: Self-Correcting QA System ===")
    
    self_correcting_qa = SelfCorrectingQA()
    
    context = "Water boils at 100 degrees Celsius at sea level atmospheric pressure."
    question = "At what temperature does water boil?"
    
    result = self_correcting_qa(context=context, question=question)
    
    print(f"Final Answer: {result.answer}")
    print(f"Attempts Used: {result.attempts}")
    print(f"Final Validation: {result.final_validation}")
    print()

# Solution 5: Ensemble System
class AnswerAggregation(dspy.Signature):
    """Aggregate multiple answers into a final answer."""
    answer1 = dspy.InputField(desc="First answer")
    answer2 = dspy.InputField(desc="Second answer")
    answer3 = dspy.InputField(desc="Third answer")
    final_answer = dspy.OutputField(desc="Best aggregated answer")
    reasoning = dspy.OutputField(desc="Reasoning for selection")

class EnsembleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.direct_qa = dspy.Predict(BasicQA)
        self.cot_qa = dspy.ChainOfThought(BasicQA)
        self.multistage_qa = MultiStageQA()
        self.aggregate = dspy.ChainOfThought(AnswerAggregation)
    
    def forward(self, context, question):
        # Get answers from different approaches
        direct_result = self.direct_qa(context=context, question=question)
        cot_result = self.cot_qa(context=context, question=question)
        multistage_result = self.multistage_qa(context=context, question=question)
        
        # Aggregate results
        final_result = self.aggregate(
            answer1=direct_result.answer,
            answer2=cot_result.answer,
            answer3=multistage_result.answer
        )
        
        return dspy.Prediction(
            direct_answer=direct_result.answer,
            cot_answer=cot_result.answer,
            multistage_answer=multistage_result.answer,
            final_answer=final_result.final_answer,
            reasoning=final_result.reasoning
        )

def solution_5_ensemble():
    """Solution 5: Ensemble QA system"""
    print("=== Solution 5: Ensemble QA System ===")
    
    ensemble_qa = EnsembleQA()
    
    context = "Python is a high-level programming language created by Guido van Rossum in 1991."
    question = "Who created Python and when?"
    
    result = ensemble_qa(context=context, question=question)
    
    print(f"Direct Answer: {result.direct_answer}")
    print(f"CoT Answer: {result.cot_answer}")
    print(f"Multistage Answer: {result.multistage_answer}")
    print(f"Final Answer: {result.final_answer}")
    print(f"Reasoning: {result.reasoning}")
    print()

# Solution 6: Production-Ready System
class ProductionQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.primary_qa = dspy.ChainOfThought(BasicQA)
        self.fallback_qa = dspy.Predict(BasicQA)
        self.cache = {}
        self.error_count = 0
        self.success_count = 0
        self.total_time = 0.0
    
    def _cache_key(self, context: str, question: str) -> str:
        """Generate cache key for context and question"""
        combined = f"{context}|{question}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _log_metrics(self, success: bool, execution_time: float):
        """Log performance metrics"""
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.total_time += execution_time
        
        logger.info(f"Operation {'succeeded' if success else 'failed'} in {execution_time:.3f}s")
    
    def forward(self, context, question):
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._cache_key(context, question)
            if cache_key in self.cache:
                execution_time = time.time() - start_time
                self._log_metrics(True, execution_time)
                logger.info("Cache hit")
                return self.cache[cache_key]
            
            # Try primary method
            try:
                result = self.primary_qa(context=context, question=question)
                
                # Validate result
                if hasattr(result, 'answer') and len(result.answer.strip()) > 0:
                    self.cache[cache_key] = result
                    execution_time = time.time() - start_time
                    self._log_metrics(True, execution_time)
                    return result
                else:
                    raise ValueError("Empty or invalid answer from primary method")
                    
            except Exception as e:
                logger.warning(f"Primary method failed: {e}, trying fallback")
                
                # Try fallback method
                result = self.fallback_qa(context=context, question=question)
                
                if hasattr(result, 'answer') and len(result.answer.strip()) > 0:
                    execution_time = time.time() - start_time
                    self._log_metrics(True, execution_time)
                    return result
                else:
                    raise ValueError("Fallback method also failed")
        
        except Exception as e:
            execution_time = time.time() - start_time
            self._log_metrics(False, execution_time)
            logger.error(f"All methods failed: {e}")
            
            return dspy.Prediction(
                answer="I apologize, but I'm unable to process your question at this time. Please try again later.",
                error=str(e)
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_requests = self.success_count + self.error_count
        success_rate = self.success_count / total_requests if total_requests > 0 else 0
        avg_time = self.total_time / total_requests if total_requests > 0 else 0
        
        return {
            'total_requests': total_requests,
            'successful_requests': self.success_count,
            'failed_requests': self.error_count,
            'success_rate': success_rate,
            'cache_size': len(self.cache),
            'average_response_time': avg_time
        }

def solution_6_production_system():
    """Solution 6: Production-ready QA system"""
    print("=== Solution 6: Production-Ready System ===")
    
    production_qa = ProductionQA()
    
    test_cases = [
        ("The Earth orbits the Sun once every 365.25 days.", "How long does Earth take to orbit the Sun?"),
        ("Shakespeare wrote many plays including Hamlet and Macbeth.", "Name two plays by Shakespeare."),
        ("", "What is the meaning of life?"),  # Test error handling
        ("The Earth orbits the Sun once every 365.25 days.", "How long does Earth take to orbit the Sun?"),  # Test caching
    ]
    
    for i, (context, question) in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"Context: {context}")
        print(f"Question: {question}")
        
        result = production_qa(context=context, question=question)
        print(f"Answer: {result.answer}")
        
        if hasattr(result, 'error'):
            print(f"Error: {result.error}")
        
        print("-" * 40)
    
    # Print system statistics
    stats = production_qa.get_stats()
    print("System Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print()

# Solution 7: Custom Metrics
def accuracy_metric(example, prediction, trace=None):
    """Custom accuracy metric for evaluation"""
    expected = example.get('expected', '').lower().strip()
    predicted = prediction.answer.lower().strip() if hasattr(prediction, 'answer') else str(prediction).lower().strip()
    
    # Exact match
    if expected == predicted:
        return 1.0
    
    # Partial match (if expected answer is contained in prediction)
    if expected in predicted or predicted in expected:
        return 0.7
    
    # Keyword overlap
    expected_words = set(expected.split())
    predicted_words = set(predicted.split())
    
    if expected_words and predicted_words:
        overlap = len(expected_words.intersection(predicted_words))
        union = len(expected_words.union(predicted_words))
        jaccard_similarity = overlap / union if union > 0 else 0
        return jaccard_similarity * 0.5
    
    return 0.0

def semantic_similarity_metric(example, prediction, trace=None):
    """Semantic similarity metric using simple keyword matching"""
    expected = example.get('expected', '').lower()
    predicted = prediction.answer.lower() if hasattr(prediction, 'answer') else str(prediction).lower()
    
    # Simple keyword-based similarity
    expected_words = set(expected.split())
    predicted_words = set(predicted.split())
    
    if not expected_words or not predicted_words:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = expected_words.intersection(predicted_words)
    union = expected_words.union(predicted_words)
    
    return len(intersection) / len(union) if union else 0.0

def comprehensive_metric(example, prediction, trace=None):
    """Comprehensive metric combining multiple factors"""
    accuracy_score = accuracy_metric(example, prediction, trace)
    semantic_score = semantic_similarity_metric(example, prediction, trace)
    
    # Length appropriateness (penalize very short or very long answers)
    predicted_text = prediction.answer if hasattr(prediction, 'answer') else str(prediction)
    length_score = 1.0
    
    if len(predicted_text) < 5:
        length_score = 0.3  # Too short
    elif len(predicted_text) > 500:
        length_score = 0.7  # Too long
    
    # Weighted combination
    final_score = (
        0.4 * accuracy_score +
        0.4 * semantic_score +
        0.2 * length_score
    )
    
    return final_score

def solution_7_custom_metrics():
    """Solution 7: Custom evaluation metrics"""
    print("=== Solution 7: Custom Evaluation Metrics ===")
    
    # Sample data for testing metrics
    examples = [
        {"question": "What is 2+2?", "expected": "4"},
        {"question": "Capital of France?", "expected": "Paris"},
        {"question": "Who created Python?", "expected": "Guido van Rossum"},
    ]
    
    predictions = [
        dspy.Prediction(answer="The answer is 4"),
        dspy.Prediction(answer="Paris is the capital of France"),
        dspy.Prediction(answer="Python was created by Guido van Rossum in 1991"),
    ]
    
    print("Metric Evaluation Results:")
    print("-" * 50)
    
    for i, (example, prediction) in enumerate(zip(examples, predictions)):
        print(f"Example {i+1}: {example['question']}")
        print(f"Expected: {example['expected']}")
        print(f"Predicted: {prediction.answer}")
        
        acc_score = accuracy_metric(example, prediction)
        sem_score = semantic_similarity_metric(example, prediction)
        comp_score = comprehensive_metric(example, prediction)
        
        print(f"Accuracy Score: {acc_score:.3f}")
        print(f"Semantic Score: {sem_score:.3f}")
        print(f"Comprehensive Score: {comp_score:.3f}")
        print("-" * 30)
    
    print()

# Bonus: Optimization Example
class OptimizableQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(BasicQA)
    
    def forward(self, context, question):
        return self.qa(context=context, question=question)

def solution_bonus_optimization():
    """Bonus: DSPy optimization example"""
    print("=== Bonus: DSPy Optimization Example ===")
    
    # Create sample training data
    trainset = [
        dspy.Example(
            context="France is in Europe. Paris is its capital.",
            question="What is the capital of France?",
            answer="Paris"
        ).with_inputs('context', 'question'),
        dspy.Example(
            context="Python was created by Guido van Rossum in 1991.",
            question="Who created Python?",
            answer="Guido van Rossum"
        ).with_inputs('context', 'question'),
    ]
    
    # Create QA module
    qa_module = OptimizableQA()
    
    # Test before optimization
    print("Before optimization:")
    for example in trainset:
        result = qa_module(context=example.context, question=example.question)
        print(f"Q: {example.question}")
        print(f"A: {result.answer}")
        print()
    
    print("Note: In a real scenario, you would use DSPy optimizers like:")
    print("- BootstrapFewShot for automatic few-shot example generation")
    print("- MIPRO for instruction optimization")
    print("- SignatureOptimizer for signature improvement")
    print("- Custom optimizers for specific use cases")
    print()

def main():
    """Run all solutions"""
    print("Day 45: Prompt Engineering with DSPy - Complete Solutions")
    print("=" * 70)
    print()
    
    # Run all solutions
    solution_1_basic_signature()
    solution_2_sentiment_analysis()
    solution_3_multistage_reasoning()
    solution_4_self_correction()
    solution_5_ensemble()
    solution_6_production_system()
    solution_7_custom_metrics()
    solution_bonus_optimization()
    
    print("=" * 70)
    print("All solutions completed successfully!")
    print()
    print("Key takeaways:")
    print("1. DSPy enables systematic prompt engineering through programming")
    print("2. Modular design allows for complex, composable prompt systems")
    print("3. Automatic optimization reduces manual prompt tuning")
    print("4. Production systems need error handling, caching, and monitoring")
    print("5. Custom metrics enable domain-specific evaluation")
    print("6. Ensemble methods can improve robustness and accuracy")
    print()
    print("Next steps:")
    print("- Experiment with real language models (OpenAI, Anthropic, etc.)")
    print("- Try different DSPy optimizers on your data")
    print("- Build domain-specific DSPy applications")
    print("- Implement A/B testing for prompt strategies")

if __name__ == "__main__":
    main()
