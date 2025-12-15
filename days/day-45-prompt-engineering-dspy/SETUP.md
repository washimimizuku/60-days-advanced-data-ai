# Day 45: Prompt Engineering with DSPy - Setup Guide

## Overview

This guide covers setting up DSPy (Declarative Self-improving Python) for advanced prompt engineering, including framework installation, language model configuration, and production deployment patterns.

## Prerequisites

- Python 3.8+
- Basic understanding of prompt engineering
- API keys for language models (OpenAI, Anthropic, etc.)
- Familiarity with machine learning concepts

## Installation

### 1. Install Dependencies

```bash
# Install from requirements.txt
pip install -r requirements.txt

# Or install core packages individually
pip install dspy-ai openai anthropic
```

### 2. Environment Setup

Create a `.env` file:

```bash
# Language Model APIs
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
COHERE_API_KEY=your_cohere_key_here

# Optional: Weights & Biases for experiment tracking
WANDB_API_KEY=your_wandb_key_here

# Optional: Redis for caching
REDIS_URL=redis://localhost:6379

# Configuration
DEFAULT_MODEL=gpt-3.5-turbo
MAX_TOKENS=1000
TEMPERATURE=0.7
```

### 3. DSPy Configuration

```python
import dspy
import os
from dotenv import load_dotenv

load_dotenv()

# Configure OpenAI
lm = dspy.OpenAI(
    model=os.getenv('DEFAULT_MODEL', 'gpt-3.5-turbo'),
    api_key=os.getenv('OPENAI_API_KEY'),
    max_tokens=int(os.getenv('MAX_TOKENS', 1000)),
    temperature=float(os.getenv('TEMPERATURE', 0.7))
)

# Configure DSPy settings
dspy.settings.configure(lm=lm)
```

## DSPy Fundamentals

### Core Concepts

#### 1. Signatures
Signatures define input-output behavior without specifying prompts:

```python
import dspy

class BasicQA(dspy.Signature):
    """Answer questions based on context."""
    context = dspy.InputField(desc="Background information")
    question = dspy.InputField(desc="Question to answer")
    answer = dspy.OutputField(desc="Answer to the question")

# Inline signature (shorthand)
generate_answer = dspy.ChainOfThought("context, question -> answer")
```

#### 2. Modules
Modules implement signatures with specific prompting strategies:

```python
class QAModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(BasicQA)
    
    def forward(self, context, question):
        return self.generate_answer(context=context, question=question)
```

#### 3. Optimizers
Optimizers automatically improve prompts:

```python
from dspy.teleprompt import BootstrapFewShot

# Define evaluation metric
def accuracy_metric(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Create optimizer
optimizer = BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=8
)

# Optimize module
optimized_qa = optimizer.compile(QAModule(), trainset=train_data)
```

### Module Types

#### 1. dspy.Predict
Direct prediction without reasoning:

```python
class DirectQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(BasicQA)
    
    def forward(self, context, question):
        return self.predict(context=context, question=question)
```

#### 2. dspy.ChainOfThought
Step-by-step reasoning:

```python
class CoTQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.cot = dspy.ChainOfThought(BasicQA)
    
    def forward(self, context, question):
        return self.cot(context=context, question=question)
```

#### 3. dspy.ReAct
Reasoning and acting pattern:

```python
class ReActQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.react = dspy.ReAct(BasicQA)
    
    def forward(self, context, question):
        return self.react(context=context, question=question)
```

## Advanced Patterns

### 1. Multi-Stage Reasoning

```python
class MultiStageQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract_facts = dspy.ChainOfThought("context -> facts")
        self.analyze_question = dspy.ChainOfThought("question -> analysis")
        self.reason = dspy.ChainOfThought("facts, analysis -> reasoning")
        self.answer = dspy.ChainOfThought("reasoning -> answer")
    
    def forward(self, context, question):
        facts = self.extract_facts(context=context)
        analysis = self.analyze_question(question=question)
        reasoning = self.reason(facts=facts.facts, analysis=analysis.analysis)
        answer = self.answer(reasoning=reasoning.reasoning)
        return answer
```

### 2. Self-Correction

```python
class SelfCorrectingQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(BasicQA)
        self.validate = dspy.ChainOfThought("context, question, answer -> is_correct")
        self.correct = dspy.ChainOfThought("context, question, wrong_answer -> corrected_answer")
    
    def forward(self, context, question, max_attempts=3):
        for attempt in range(max_attempts):
            answer = self.generate(context=context, question=question)
            validation = self.validate(
                context=context, 
                question=question, 
                answer=answer.answer
            )
            
            if validation.is_correct.lower() == "yes":
                return answer
            
            if attempt < max_attempts - 1:
                correction = self.correct(
                    context=context,
                    question=question,
                    wrong_answer=answer.answer
                )
                answer = dspy.Prediction(answer=correction.corrected_answer)
        
        return answer
```

### 3. Ensemble Methods

```python
class EnsembleQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.direct_qa = dspy.Predict(BasicQA)
        self.cot_qa = dspy.ChainOfThought(BasicQA)
        self.multistage_qa = MultiStageQA()
        self.aggregate = dspy.ChainOfThought("answer1, answer2, answer3 -> final_answer")
    
    def forward(self, context, question):
        answer1 = self.direct_qa(context=context, question=question)
        answer2 = self.cot_qa(context=context, question=question)
        answer3 = self.multistage_qa(context=context, question=question)
        
        final = self.aggregate(
            answer1=answer1.answer,
            answer2=answer2.answer,
            answer3=answer3.answer
        )
        return final
```

## Optimization Strategies

### 1. BootstrapFewShot

Automatically generates few-shot examples:

```python
from dspy.teleprompt import BootstrapFewShot

optimizer = BootstrapFewShot(
    metric=accuracy_metric,
    max_bootstrapped_demos=8,
    max_labeled_demos=16,
    teacher_settings=dict(temperature=0.7),
    student_settings=dict(temperature=0.1)
)

optimized_model = optimizer.compile(QAModule(), trainset=train_data)
```

### 2. MIPRO (Multi-prompt Instruction Proposal and Refinement Optimization)

```python
from dspy.teleprompt import MIPRO

mipro = MIPRO(
    metric=accuracy_metric,
    num_candidates=20,
    init_temperature=1.0,
    verbose=True
)

optimized_model = mipro.compile(
    QAModule(),
    trainset=train_data,
    valset=val_data,
    num_trials=50
)
```

### 3. Signature Optimization

```python
from dspy.teleprompt import SignatureOptimizer

sig_optimizer = SignatureOptimizer(
    metric=accuracy_metric,
    breadth=10,
    depth=3,
    verbose=True
)

optimized_signature = sig_optimizer.compile(
    BasicQA,
    trainset=train_data,
    valset=val_data
)
```

## Production Considerations

### 1. Caching Implementation

```python
import hashlib
import json
from typing import Dict, Any

class CachedQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(BasicQA)
        self.cache: Dict[str, Any] = {}
    
    def _cache_key(self, context: str, question: str) -> str:
        combined = json.dumps({"context": context, "question": question}, sort_keys=True)
        return hashlib.md5(combined.encode()).hexdigest()
    
    def forward(self, context, question):
        cache_key = self._cache_key(context, question)
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        result = self.qa(context=context, question=question)
        self.cache[cache_key] = result
        return result
```

### 2. Error Handling

```python
import logging
from typing import Optional

logger = logging.getLogger(__name__)

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
            else:
                logger.warning("Primary method returned invalid result, trying fallback")
                return self.fallback(context=context, question=question)
        
        except Exception as e:
            logger.error(f"Primary method failed: {e}, trying fallback")
            try:
                return self.fallback(context=context, question=question)
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {fallback_error}")
                return dspy.Prediction(
                    answer="I'm sorry, I couldn't process your question at this time.",
                    error=str(e)
                )
    
    def _validate_result(self, result) -> bool:
        return (
            hasattr(result, 'answer') and 
            isinstance(result.answer, str) and 
            len(result.answer.strip()) > 0
        )
```

### 3. Monitoring and Metrics

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class RequestMetrics:
    timestamp: float
    success: bool
    response_time: float
    input_tokens: int
    output_tokens: int

class MonitoredQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(BasicQA)
        self.metrics: List[RequestMetrics] = []
    
    def forward(self, context, question):
        start_time = time.time()
        
        try:
            result = self.qa(context=context, question=question)
            
            # Log metrics
            self.metrics.append(RequestMetrics(
                timestamp=start_time,
                success=True,
                response_time=time.time() - start_time,
                input_tokens=len(f"{context} {question}".split()),
                output_tokens=len(result.answer.split()) if hasattr(result, 'answer') else 0
            ))
            
            return result
            
        except Exception as e:
            self.metrics.append(RequestMetrics(
                timestamp=start_time,
                success=False,
                response_time=time.time() - start_time,
                input_tokens=len(f"{context} {question}".split()),
                output_tokens=0
            ))
            raise
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        if not self.metrics:
            return {}
        
        successful = [m for m in self.metrics if m.success]
        
        return {
            'total_requests': len(self.metrics),
            'successful_requests': len(successful),
            'success_rate': len(successful) / len(self.metrics),
            'avg_response_time': sum(m.response_time for m in successful) / len(successful) if successful else 0,
            'total_input_tokens': sum(m.input_tokens for m in self.metrics),
            'total_output_tokens': sum(m.output_tokens for m in self.metrics)
        }
```

## Evaluation Framework

### Custom Metrics

```python
def comprehensive_metric(example, prediction, trace=None):
    """Multi-faceted evaluation metric"""
    
    # Exact match
    exact_match = example.answer.lower().strip() == prediction.answer.lower().strip()
    
    # Semantic similarity (simplified)
    def jaccard_similarity(text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        return len(intersection) / len(union) if union else 0
    
    semantic_score = jaccard_similarity(example.answer, prediction.answer)
    
    # Length appropriateness
    length_score = 1.0
    if len(prediction.answer) < 10:
        length_score = 0.5
    elif len(prediction.answer) > 500:
        length_score = 0.7
    
    # Combine scores
    final_score = (
        0.4 * exact_match + 
        0.4 * semantic_score + 
        0.2 * length_score
    )
    
    return final_score
```

### A/B Testing Framework

```python
import random
from typing import List, Callable

class ABTestFramework:
    def __init__(self, model_a, model_b, split_ratio=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.split_ratio = split_ratio
        self.results_a = []
        self.results_b = []
    
    def evaluate(self, test_data: List, metric_fn: Callable):
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
            'winner': 'A' if avg_a > avg_b else 'B',
            'confidence': abs(avg_a - avg_b)
        }
```

## Testing

Run the test suite:

```bash
# Run all tests
python -m pytest test_dspy.py -v

# Run specific test class
python -m pytest test_dspy.py::TestProductionQA -v

# Run with coverage
pip install pytest-cov
python -m pytest test_dspy.py --cov=solution --cov-report=html
```

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```python
   # Verify API key is set
   import os
   print(os.getenv('OPENAI_API_KEY'))  # Should not be None
   ```

2. **Module Import Errors**
   ```bash
   # Reinstall DSPy
   pip uninstall dspy-ai
   pip install dspy-ai>=2.4.0
   ```

3. **Rate Limiting**
   ```python
   # Add retry logic
   import time
   from functools import wraps
   
   def retry_on_rate_limit(max_retries=3, delay=1):
       def decorator(func):
           @wraps(func)
           def wrapper(*args, **kwargs):
               for attempt in range(max_retries):
                   try:
                       return func(*args, **kwargs)
                   except Exception as e:
                       if "rate limit" in str(e).lower() and attempt < max_retries - 1:
                           time.sleep(delay * (2 ** attempt))  # Exponential backoff
                           continue
                       raise
               return wrapper
           return decorator
   ```

4. **Memory Issues with Large Contexts**
   ```python
   # Implement context truncation
   def truncate_context(context: str, max_tokens: int = 2000) -> str:
       words = context.split()
       if len(words) <= max_tokens:
           return context
       return ' '.join(words[:max_tokens]) + "..."
   ```

## Best Practices

1. **Start Simple**: Begin with basic signatures and modules
2. **Iterate on Data**: Use real examples to guide optimization
3. **Measure Everything**: Implement comprehensive metrics and monitoring
4. **Modular Design**: Build composable components for reusability
5. **Version Control**: Track prompt versions and performance over time
6. **Fallback Strategies**: Always have backup approaches for critical systems
7. **Cost Optimization**: Monitor API usage and implement caching
8. **Human-in-the-Loop**: Include mechanisms for human feedback and correction

## Next Steps

1. Complete the exercises in `exercise.py`
2. Run the test suite to verify implementations
3. Experiment with different optimizers on your data
4. Build domain-specific DSPy applications
5. Implement production monitoring and alerting
6. Explore advanced patterns like retrieval-augmented generation (RAG) with DSPy

## Resources

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy)
- [DSPy Paper](https://arxiv.org/abs/2310.03714)
- [Stanford CS224U DSPy Tutorial](https://web.stanford.edu/class/cs224u/)