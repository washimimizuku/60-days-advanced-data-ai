# Day 45: Prompt Engineering with DSPy

## Learning Objectives
By the end of this session, you will be able to:
- Understand the DSPy framework and its core concepts
- Implement systematic prompt optimization using DSPy
- Build modular, composable prompt programs
- Apply automatic prompt tuning and optimization techniques
- Design production-ready prompt engineering systems
- Evaluate and improve prompt performance systematically

## Theory (30 minutes)

### What is DSPy?

DSPy (Declarative Self-improving Python) is a framework that treats prompting as programming rather than manual prompt crafting. Instead of writing prompts manually, you write programs that generate and optimize prompts automatically.

**Key Principles:**
- **Declarative**: Specify what you want, not how to prompt for it
- **Modular**: Compose complex behaviors from simple components
- **Optimizable**: Automatically tune prompts based on data and metrics
- **Systematic**: Replace trial-and-error with principled optimization

### Core DSPy Components

#### 1. Signatures
Signatures define the input-output behavior of a module without specifying the prompt:

```python
import dspy

# Simple signature
class BasicQA(dspy.Signature):
    """Answer questions based on context."""
    context = dspy.InputField()
    question = dspy.InputField()
    answer = dspy.OutputField()

# Inline signature
generate_answer = dspy.ChainOfThought("context, question -> answer")
```

#### 2. Modules
Modules are the building blocks that implement signatures:

```python
# Chain of Thought module
class CoTQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(BasicQA)
    
    def forward(self, context, question):
        return self.generate_answer(context=context, question=question)

# Predict module (direct prediction)
class DirectQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.Predict(BasicQA)
    
    def forward(self, context, question):
        return self.generate_answer(context=context, question=question)
```

#### 3. Optimizers
Optimizers automatically improve prompts based on training data:

```python
from dspy.teleprompt import BootstrapFewShot

# Bootstrap few-shot optimizer
optimizer = BootstrapFewShot(metric=validate_answer)
optimized_qa = optimizer.compile(CoTQA(), trainset=train_examples)
```

### Advanced DSPy Patterns

#### 1. Multi-Stage Reasoning
Break complex tasks into multiple reasoning stages:

```python
class MultiStageQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_facts = dspy.ChainOfThought("context -> facts")
        self.reason = dspy.ChainOfThought("facts, question -> reasoning")
        self.answer = dspy.ChainOfThought("reasoning -> answer")
    
    def forward(self, context, question):
        facts = self.extract_facts(context=context)
        reasoning = self.reason(facts=facts.facts, question=question)
        answer = self.answer(reasoning=reasoning.reasoning)
        return answer
```

#### 2. Self-Correction and Validation
Implement self-correction mechanisms:

```python
class SelfCorrectingQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought("context, question -> answer")
        self.validate = dspy.ChainOfThought("context, question, answer -> is_correct")
        self.correct = dspy.ChainOfThought("context, question, wrong_answer -> corrected_answer")
    
    def forward(self, context, question, max_attempts=3):
        for attempt in range(max_attempts):
            answer = self.generate(context=context, question=question)
            validation = self.validate(context=context, question=question, answer=answer.answer)
            
            if validation.is_correct.lower() == "yes":
                return answer
            
            if attempt < max_attempts - 1:
                answer = self.correct(
                    context=context, 
                    question=question, 
                    wrong_answer=answer.answer
                )
        
        return answer
```

#### 3. Ensemble Methods
Combine multiple approaches for robust results:

```python
class EnsembleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot_qa = dspy.ChainOfThought(BasicQA)
        self.direct_qa = dspy.Predict(BasicQA)
        self.multistage_qa = MultiStageQA()
        self.aggregate = dspy.ChainOfThought("answer1, answer2, answer3 -> final_answer")
    
    def forward(self, context, question):
        answer1 = self.cot_qa(context=context, question=question)
        answer2 = self.direct_qa(context=context, question=question)
        answer3 = self.multistage_qa(context=context, question=question)
        
        final = self.aggregate(
            answer1=answer1.answer,
            answer2=answer2.answer,
            answer3=answer3.answer
        )
        return final
```

### Optimization Strategies

#### 1. Bootstrap Few-Shot Learning
Automatically generate few-shot examples:

```python
def accuracy_metric(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Bootstrap optimizer
bootstrap = BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=8,
    max_labeled_demos=16
)

optimized_model = bootstrap.compile(CoTQA(), trainset=train_data)
```

#### 2. Random Search Optimization
Explore different prompt variations:

```python
from dspy.teleprompt import MIPRO

mipro = MIPRO(
    metric=accuracy_metric,
    num_candidates=20,
    init_temperature=1.0
)

optimized_model = mipro.compile(
    CoTQA(),
    trainset=train_data,
    valset=val_data
)
```

#### 3. Signature Optimization
Automatically improve signature descriptions:

```python
from dspy.teleprompt import SignatureOptimizer

sig_optimizer = SignatureOptimizer(
    metric=accuracy_metric,
    breadth=10,
    depth=3
)

optimized_signature = sig_optimizer.compile(
    BasicQA,
    trainset=train_data
)
```

### Production Considerations

#### 1. Caching and Performance
Implement caching for expensive operations:

```python
import functools
from typing import Dict, Any

class CachedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(BasicQA)
        self.cache: Dict[str, Any] = {}
    
    def _cache_key(self, context: str, question: str) -> str:
        return f"{hash(context)}_{hash(question)}"
    
    def forward(self, context, question):
        cache_key = self._cache_key(context, question)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.qa(context=context, question=question)
        self.cache[cache_key] = result
        return result
```

#### 2. Error Handling and Fallbacks
Implement robust error handling:

```python
class RobustQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.primary = dspy.ChainOfThought(BasicQA)
        self.fallback = dspy.Predict(BasicQA)
    
    def forward(self, context, question):
        try:
            result = self.primary(context=context, question=question)
            if self._validate_result(result):
                return result
        except Exception as e:
            print(f"Primary method failed: {e}")
        
        try:
            return self.fallback(context=context, question=question)
        except Exception as e:
            return dspy.Prediction(answer="I'm sorry, I couldn't process your question.")
    
    def _validate_result(self, result):
        return hasattr(result, 'answer') and len(result.answer.strip()) > 0
```

#### 3. Monitoring and Logging
Track performance and usage:

```python
import logging
import time
from typing import Optional

class MonitoredQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(BasicQA)
        self.logger = logging.getLogger(__name__)
    
    def forward(self, context, question):
        start_time = time.time()
        
        try:
            result = self.qa(context=context, question=question)
            
            # Log successful execution
            execution_time = time.time() - start_time
            self.logger.info(f"QA completed in {execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"QA failed after {execution_time:.2f}s: {e}")
            raise
```

### Evaluation and Metrics

#### 1. Custom Metrics
Define domain-specific evaluation metrics:

```python
def comprehensive_metric(example, pred, trace=None):
    """Multi-faceted evaluation metric"""
    
    # Exact match
    exact_match = example.answer.lower().strip() == pred.answer.lower().strip()
    
    # Semantic similarity (using embeddings)
    semantic_score = calculate_semantic_similarity(example.answer, pred.answer)
    
    # Length appropriateness
    length_score = 1.0 if 10 <= len(pred.answer) <= 200 else 0.5
    
    # Combine scores
    final_score = (
        0.4 * exact_match + 
        0.4 * semantic_score + 
        0.2 * length_score
    )
    
    return final_score

def calculate_semantic_similarity(text1: str, text2: str) -> float:
    """Calculate semantic similarity between two texts"""
    # Implementation would use sentence transformers or similar
    # Placeholder implementation
    return 0.8 if text1.lower() in text2.lower() or text2.lower() in text1.lower() else 0.3
```

#### 2. A/B Testing Framework
Compare different prompt strategies:

```python
class ABTestFramework:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.results_a = []
        self.results_b = []
    
    def evaluate(self, test_data, metric_fn):
        import random
        
        for example in test_data:
            if random.random() < self.split_ratio:
                pred = self.model_a(context=example.context, question=example.question)
                score = metric_fn(example, pred)
                self.results_a.append(score)
            else:
                pred = self.model_b(context=example.context, question=example.question)
                score = metric_fn(example, pred)
                self.results_b.append(score)
    
    def get_results(self):
        avg_a = sum(self.results_a) / len(self.results_a) if self.results_a else 0
        avg_b = sum(self.results_b) / len(self.results_b) if self.results_b else 0
        
        return {
            'model_a_avg': avg_a,
            'model_b_avg': avg_b,
            'model_a_count': len(self.results_a),
            'model_b_count': len(self.results_b),
            'winner': 'A' if avg_a > avg_b else 'B'
        }
```

### Best Practices

1. **Start Simple**: Begin with basic signatures and modules, then add complexity
2. **Iterate on Data**: Use real examples to guide optimization
3. **Measure Everything**: Implement comprehensive metrics and monitoring
4. **Modular Design**: Build composable components for reusability
5. **Version Control**: Track prompt versions and performance over time
6. **Fallback Strategies**: Always have backup approaches for critical systems
7. **Cost Optimization**: Monitor API usage and implement caching
8. **Human-in-the-Loop**: Include mechanisms for human feedback and correction

### Why DSPy Matters

DSPy transforms prompt engineering from an art to a science by:

- **Reducing Manual Work**: Automates prompt creation and optimization
- **Improving Consistency**: Systematic approach reduces variability
- **Enabling Scalability**: Modular design supports complex applications
- **Facilitating Maintenance**: Code-based prompts are easier to version and update
- **Enhancing Performance**: Data-driven optimization improves results
- **Supporting Collaboration**: Shared modules and signatures improve team productivity

## Exercise (25 minutes)
Complete the hands-on exercises in `exercise.py` to practice DSPy implementation.

## Resources
- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [DSPy Paper: "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"](https://arxiv.org/abs/2310.03714)
- [Stanford CS224U: DSPy Tutorial](https://web.stanford.edu/class/cs224u/)

## Next Steps
- Complete the exercises to practice DSPy implementation
- Take the quiz to test your understanding
- Explore advanced DSPy patterns in your own projects
- Move to Day 46: Prompt Security
