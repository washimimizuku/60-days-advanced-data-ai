"""
Day 45: Prompt Engineering with DSPy - Exercises

Complete the following exercises to practice DSPy implementation:
1. Basic DSPy signature and module creation
2. Chain of Thought implementation
3. Multi-stage reasoning system
4. Self-correcting QA system
5. Ensemble approach with multiple strategies
6. Production-ready system with error handling

Run each exercise and observe the outputs.
"""

import dspy
from typing import List, Dict, Any, Optional
import random
import time

# Mock LM for exercises (replace with actual LM in practice)
class MockLM:
    def __init__(self):
        self.responses = {
            "What is the capital of France?": "Paris is the capital of France.",
            "Explain photosynthesis": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "What is 2+2?": "2+2 equals 4.",
            "Who wrote Romeo and Juliet?": "William Shakespeare wrote Romeo and Juliet."
        }
    
    def __call__(self, prompt, **kwargs):
        # Simple mock response based on question
        for question, answer in self.responses.items():
            if question.lower() in prompt.lower():
                return answer
        return "I need more information to answer that question."

# Set up mock language model
dspy.settings.configure(lm=MockLM())

# Exercise 1: Basic DSPy Signature and Module
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

def exercise_1_basic_signature():
    """Exercise 1: Create and test a basic DSPy signature and module"""
    print("=== Exercise 1: Basic DSPy Signature and Module ===")
    
    qa_module = SimpleQAModule()
    
    # Test data
    context = "France is a country in Western Europe. Its capital city is Paris, which is also its largest city."
    question = "What is the capital of France?"
    
    result = qa_module(context=context, question=question)
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Answer: {result.answer}")
    print()

# Exercise 2: Multi-Field Signature
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

def exercise_2_sentiment_analysis():
    """Exercise 2: Multi-field signature for sentiment analysis"""
    print("=== Exercise 2: Multi-Field Signature (Sentiment Analysis) ===")
    
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

# Exercise 3: Multi-Stage Reasoning
class MultiStageQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # Stage 1: Extract key facts from context
        self.extract_facts = dspy.ChainOfThought("context -> facts")
        # Stage 2: Analyze the question
        self.analyze_question = dspy.ChainOfThought("question -> analysis")
        # Stage 3: Reason about the answer
        self.reason = dspy.ChainOfThought("facts, analysis -> reasoning")
        # Stage 4: Generate final answer
        self.generate_answer = dspy.ChainOfThought("reasoning -> answer")
    
    def forward(self, context, question):
        # 1. Extract facts from context
        facts_result = self.extract_facts(context=context)
        # 2. Analyze what the question is asking
        analysis_result = self.analyze_question(question=question)
        # 3. Reason through the problem
        reasoning_result = self.reason(facts=facts_result.facts, analysis=analysis_result.analysis)
        # 4. Generate final answer
        answer_result = self.generate_answer(reasoning=reasoning_result.reasoning)
        
        return dspy.Prediction(
            facts=facts_result.facts,
            analysis=analysis_result.analysis,
            reasoning=reasoning_result.reasoning,
            answer=answer_result.answer
        )

def exercise_3_multistage_reasoning():
    """Exercise 3: Multi-stage reasoning system"""
    print("=== Exercise 3: Multi-Stage Reasoning ===")
    
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

# Exercise 4: Self-Correcting System
class SelfCorrectingQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # 1. Initial answer generation
        self.generate = dspy.ChainOfThought(BasicQA)
        # 2. Answer validation
        self.validate = dspy.ChainOfThought("context, question, answer -> is_correct")
        # 3. Correction if needed
        self.correct = dspy.ChainOfThought("context, question, wrong_answer -> corrected_answer")
    
    def forward(self, context, question, max_attempts=3):
        for attempt in range(max_attempts):
            # 1. Generate initial answer
            answer_result = self.generate(context=context, question=question)
            
            # 2. Validate the answer
            validation_result = self.validate(
                context=context,
                question=question,
                answer=answer_result.answer
            )
            
            print(f"Attempt {attempt + 1}: {answer_result.answer}")
            print(f"Validation: {validation_result.is_correct}")
            
            if validation_result.is_correct.lower() == "yes":
                return dspy.Prediction(
                    answer=answer_result.answer,
                    attempts=attempt + 1,
                    validation="Correct"
                )
            
            # 3. If invalid, generate correction (if not last attempt)
            if attempt < max_attempts - 1:
                correction_result = self.correct(
                    context=context,
                    question=question,
                    wrong_answer=answer_result.answer
                )
                answer_result = dspy.Prediction(answer=correction_result.corrected_answer)
        
        # 4. Return final attempt if max_attempts reached
        return dspy.Prediction(
            answer=answer_result.answer,
            attempts=max_attempts,
            validation="Max attempts reached"
        )

def exercise_4_self_correction():
    """Exercise 4: Self-correcting QA system"""
    print("=== Exercise 4: Self-Correcting QA System ===")
    
    self_correcting_qa = SelfCorrectingQA()
    
    context = "Water boils at 100 degrees Celsius at sea level atmospheric pressure."
    question = "At what temperature does water boil?"
    
    result = self_correcting_qa(context=context, question=question)
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Final Answer: {result.answer}")
    print(f"Attempts Used: {result.attempts}")
    print(f"Validation: {result.validation}")
    print()

# Exercise 5: Ensemble System
class EnsembleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # 1. Direct prediction
        self.direct_qa = dspy.Predict(BasicQA)
        # 2. Chain of thought
        self.cot_qa = dspy.ChainOfThought(BasicQA)
        # 3. Multi-stage reasoning
        self.multistage_qa = MultiStageQA()
        # 4. Aggregation module
        self.aggregate = dspy.ChainOfThought("answer1, answer2, answer3 -> final_answer")
    
    def forward(self, context, question):
        # 1. Get answers from multiple methods
        direct_result = self.direct_qa(context=context, question=question)
        cot_result = self.cot_qa(context=context, question=question)
        multistage_result = self.multistage_qa(context=context, question=question)
        
        # 2. Aggregate the results
        final_result = self.aggregate(
            answer1=direct_result.answer,
            answer2=cot_result.answer,
            answer3=multistage_result.answer
        )
        
        # 3. Return best answer
        return dspy.Prediction(
            direct_answer=direct_result.answer,
            cot_answer=cot_result.answer,
            multistage_answer=multistage_result.answer,
            final_answer=final_result.final_answer
        )

def exercise_5_ensemble():
    """Exercise 5: Ensemble QA system"""
    print("=== Exercise 5: Ensemble QA System ===")
    
    ensemble_qa = EnsembleQA()
    
    context = "Python is a high-level programming language created by Guido van Rossum in 1991."
    question = "Who created Python and when?"
    
    result = ensemble_qa(context=context, question=question)
    print(f"Context: {context}")
    print(f"Question: {question}")
    print(f"Direct Answer: {result.direct_answer}")
    print(f"CoT Answer: {result.cot_answer}")
    print(f"Multistage Answer: {result.multistage_answer}")
    print(f"Final Answer: {result.final_answer}")
    print()

# Exercise 6: Production-Ready System
class ProductionQA(dspy.Module):
    def __init__(self):
        super().__init__()
        # 1. Error handling & 4. Fallback mechanisms
        self.primary_qa = dspy.ChainOfThought(BasicQA)
        self.fallback_qa = dspy.Predict(BasicQA)
        # 2. Caching
        self.cache = {}
        # 3. Monitoring
        self.error_count = 0
        self.success_count = 0
        self.total_time = 0.0
    
    def _cache_key(self, context: str, question: str) -> str:
        """Generate cache key for context and question"""
        import hashlib
        combined = f"{context}|{question}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _log_metrics(self, success: bool, execution_time: float):
        """Log performance metrics"""
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        self.total_time += execution_time
    
    def forward(self, context, question):
        start_time = time.time()
        
        try:
            # 1. Cache checking
            cache_key = self._cache_key(context, question)
            if cache_key in self.cache:
                execution_time = time.time() - start_time
                self._log_metrics(True, execution_time)
                return self.cache[cache_key]
            
            # 2. Error handling - Try primary method
            try:
                result = self.primary_qa(context=context, question=question)
                if hasattr(result, 'answer') and len(result.answer.strip()) > 0:
                    self.cache[cache_key] = result
                    execution_time = time.time() - start_time
                    self._log_metrics(True, execution_time)
                    return result
                else:
                    raise ValueError("Empty answer from primary method")
            
            except Exception as e:
                print(f"Primary method failed: {e}, trying fallback")
                # 4. Fallback strategies
                result = self.fallback_qa(context=context, question=question)
                execution_time = time.time() - start_time
                self._log_metrics(True, execution_time)
                return result
        
        except Exception as e:
            execution_time = time.time() - start_time
            self._log_metrics(False, execution_time)
            return dspy.Prediction(
                answer="I apologize, but I'm unable to process your question at this time.",
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

def exercise_6_production_system():
    """Exercise 6: Production-ready QA system"""
    print("=== Exercise 6: Production-Ready System ===")
    
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

# Exercise 7: Custom Metric Implementation
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
    """Semantic similarity metric"""
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

def exercise_7_custom_metrics():
    """Exercise 7: Custom evaluation metrics"""
    print("=== Exercise 7: Custom Evaluation Metrics ===")
    
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
        
        print(f"Accuracy Score: {acc_score:.3f}")
        print(f"Semantic Score: {sem_score:.3f}")
        print("-" * 30)
    
    print()

def main():
    """Run all exercises"""
    print("Day 45: Prompt Engineering with DSPy - Exercises")
    print("=" * 60)
    print()
    
    # Run all exercises
    exercise_1_basic_signature()
    exercise_2_sentiment_analysis()
    exercise_3_multistage_reasoning()
    exercise_4_self_correction()
    exercise_5_ensemble()
    exercise_6_production_system()
    exercise_7_custom_metrics()
    
    print("=" * 60)
    print("Exercises completed! Check the solution.py file for complete implementations.")
    print()
    print("Next steps:")
    print("1. Implement the TODO sections in each exercise")
    print("2. Test with real language models (OpenAI, Anthropic, etc.)")
    print("3. Experiment with different optimization strategies")
    print("4. Build your own DSPy applications")

if __name__ == "__main__":
    main()
