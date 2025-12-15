# Day 53: RAG Evaluation with RAGAS Setup Guide

## Overview
This guide helps you set up the environment for comprehensive RAG evaluation using the RAGAS framework, including metrics calculation, automated pipelines, A/B testing, and quality monitoring.

## Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB recommended for large-scale evaluation)
- Basic understanding of RAG systems and evaluation metrics
- Completed Day 52 (Advanced RAG Systems)

## Quick Setup

### 1. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install RAGAS from source for latest features
pip install git+https://github.com/explodinggradients/ragas.git
```

### 2. Verify Installation
```bash
python -c "import ragas, numpy, pandas, scipy; print('✅ Core libraries installed')"
```

### 3. Run Basic Test
```bash
python exercise.py
```

## Detailed Setup

### Environment Configuration

Create `.env` file:
```bash
# LLM Configuration for RAGAS
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-ada-002

# Alternative LLM Providers
ANTHROPIC_API_KEY=your_anthropic_key
COHERE_API_KEY=your_cohere_key
HUGGINGFACE_API_KEY=your_hf_key

# Evaluation Configuration
EVALUATION_BATCH_SIZE=10
MAX_CONCURRENT_EVALUATIONS=5
EVALUATION_TIMEOUT_SECONDS=30

# Quality Monitoring Thresholds
FAITHFULNESS_THRESHOLD=0.7
ANSWER_RELEVANCY_THRESHOLD=0.8
CONTEXT_PRECISION_THRESHOLD=0.6
CONTEXT_RECALL_THRESHOLD=0.7
RESPONSE_TIME_THRESHOLD=2.0

# Statistical Testing
AB_TEST_ALPHA=0.05
AB_TEST_POWER=0.8
MIN_SAMPLE_SIZE=30

# Database Configuration (for storing evaluation results)
DATABASE_URL=sqlite:///rag_evaluation.db
REDIS_URL=redis://localhost:6379/0

# Monitoring and Alerting
ENABLE_MONITORING=true
ALERT_WEBHOOK_URL=your_slack_webhook_url
DASHBOARD_PORT=8501

# Logging Configuration
LOG_LEVEL=INFO
LOG_FORMAT=json
ENABLE_STRUCTURED_LOGGING=true
```

### RAGAS Framework Setup

#### Option 1: Using OpenAI (Recommended)
```python
from ragas.llms import openai
from ragas.embeddings import openai as openai_embeddings

# Configure RAGAS with OpenAI
llm = openai.OpenAILLM(model="gpt-3.5-turbo")
embeddings = openai_embeddings.OpenAIEmbeddings(model="text-embedding-ada-002")
```

#### Option 2: Using Local Models
```python
from ragas.llms import LangchainLLM
from langchain.llms import HuggingFacePipeline
from sentence_transformers import SentenceTransformer

# Local LLM setup
local_llm = HuggingFacePipeline.from_model_id(
    model_id="microsoft/DialoGPT-medium",
    task="text-generation"
)
llm = LangchainLLM(llm=local_llm)

# Local embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
```

#### Option 3: Using Azure OpenAI
```python
from ragas.llms import AzureOpenAILLM
from ragas.embeddings import AzureOpenAIEmbeddings

llm = AzureOpenAILLM(
    azure_endpoint="your-endpoint",
    api_key="your-key",
    api_version="2023-05-15",
    deployment_name="your-deployment"
)
```

### Database Setup (Optional)

#### SQLite (Default)
```python
from sqlalchemy import create_engine
import pandas as pd

# SQLite setup (no additional configuration needed)
engine = create_engine('sqlite:///rag_evaluation.db')
```

#### PostgreSQL (Production)
```bash
# Install PostgreSQL driver
pip install psycopg2-binary

# Create database
createdb rag_evaluation
```

```python
# PostgreSQL connection
engine = create_engine('postgresql://user:password@localhost/rag_evaluation')
```

#### Redis (for Caching)
```bash
# Install and start Redis
# macOS
brew install redis
redis-server

# Ubuntu/Debian
sudo apt-get install redis-server
sudo systemctl start redis

# Docker
docker run -d -p 6379:6379 redis:alpine
```

## Core Concepts

### 1. RAGAS Metrics Overview

**Faithfulness**: Measures factual accuracy of generated answers
```python
# Claims in answer that are supported by context / Total claims
faithfulness_score = supported_claims / total_claims
```

**Answer Relevancy**: Evaluates how relevant the answer is to the question
```python
# Generate questions from answer, measure similarity to original
generated_questions = llm.generate_questions(answer)
relevancy_score = mean_similarity(original_question, generated_questions)
```

**Context Precision**: Measures relevance of retrieved contexts
```python
# Relevant contexts in top-k / k
precision_score = relevant_contexts / total_contexts
```

**Context Recall**: Evaluates coverage of relevant information
```python
# Retrieved relevant contexts / All relevant contexts
recall_score = retrieved_relevant / all_relevant
```

### 2. Evaluation Pipeline Architecture

```python
# Complete evaluation workflow
class EvaluationWorkflow:
    def __init__(self):
        self.metrics_calculator = RAGASMetrics()
        self.pipeline = EvaluationPipeline()
        self.monitor = QualityMonitor()
        self.ab_tester = RAGABTester()
    
    def evaluate_system(self, rag_responses):
        # 1. Calculate individual metrics
        results = self.pipeline.evaluate_batch(rag_responses)
        
        # 2. Generate comprehensive report
        report = self.pipeline.generate_evaluation_report(results)
        
        # 3. Monitor quality in real-time
        for response in rag_responses:
            self.monitor.monitor_response(response)
        
        # 4. Track performance trends
        trends = self.pipeline.track_performance_over_time()
        
        return {
            'individual_results': results,
            'aggregate_report': report,
            'performance_trends': trends
        }
```

### 3. A/B Testing Framework

```python
# Statistical A/B testing for RAG systems
class RAGABTestFramework:
    def design_test(self, effect_size=0.2, power=0.8, alpha=0.05):
        # Calculate required sample size
        sample_size = self.power_analysis(effect_size, power, alpha)
        
        return {
            'sample_size_per_group': sample_size,
            'total_sample_size': sample_size * 2,
            'expected_duration': self.estimate_duration(sample_size)
        }
    
    def run_test(self, system_a_responses, system_b_responses):
        # Evaluate both systems
        results_a = self.evaluate_system(system_a_responses)
        results_b = self.evaluate_system(system_b_responses)
        
        # Statistical significance testing
        significance = self.statistical_test(results_a, results_b)
        
        # Generate recommendation
        recommendation = self.make_recommendation(results_a, results_b, significance)
        
        return {
            'system_a_performance': results_a,
            'system_b_performance': results_b,
            'statistical_significance': significance,
            'recommendation': recommendation
        }
```

## Testing Your Setup

### 1. Run Unit Tests
```bash
# Run all tests
pytest test_rag_evaluation.py -v

# Run specific test categories
pytest test_rag_evaluation.py::TestRAGASMetrics -v
pytest test_rag_evaluation.py::TestEvaluationPipeline -v
pytest test_rag_evaluation.py::TestQualityMonitor -v

# Run integration tests
pytest test_rag_evaluation.py::TestIntegration -v
```

### 2. Validate RAGAS Integration
```bash
python -c "
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# Test RAGAS integration
sample_data = {
    'question': ['What is AI?'],
    'answer': ['AI is artificial intelligence.'],
    'contexts': [['AI stands for artificial intelligence.']],
    'ground_truths': [['AI is artificial intelligence.']]
}

dataset = Dataset.from_dict(sample_data)
result = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
print('✅ RAGAS integration working')
print(f'Results: {result}')
"
```

### 3. Performance Benchmarks
```bash
# Benchmark evaluation speed
python -c "
import time
from solution import RAGASMetrics, RAGResponse

metrics = RAGASMetrics()
responses = [
    RAGResponse(f'Question {i}', f'Answer {i}', [f'Context {i}'], {})
    for i in range(100)
]

start_time = time.time()
for response in responses:
    metrics.evaluate_response(response)
end_time = time.time()

avg_time = (end_time - start_time) / len(responses)
print(f'Average evaluation time: {avg_time*1000:.2f}ms per response')
"
```

### 4. Memory Usage Check
```bash
python -c "
import psutil
import os
from solution import EvaluationPipeline, RAGASMetrics

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

# Create evaluation system
metrics = RAGASMetrics()
pipeline = EvaluationPipeline(metrics)

mem_after = process.memory_info().rss / 1024 / 1024
print(f'Memory usage: {mem_after - mem_before:.1f} MB')
"
```

## Common Issues & Solutions

### Issue 1: RAGAS API Rate Limits
**Problem:** OpenAI API rate limits during evaluation
**Solutions:**
- Implement exponential backoff and retry logic
- Use batch processing with delays
- Consider using local models for development
- Implement caching for repeated evaluations

```python
import time
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def evaluate_with_retry(dataset, metrics):
    return evaluate(dataset, metrics=metrics)
```

### Issue 2: Slow Evaluation Performance
**Problem:** Evaluation takes too long for large datasets
**Solutions:**
- Use smaller, faster models for development
- Implement parallel processing
- Cache evaluation results
- Use sampling for quick assessments

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

async def parallel_evaluation(responses, max_workers=5):
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(executor, evaluate_response, response)
            for response in responses
        ]
        results = await asyncio.gather(*tasks)
    return results
```

### Issue 3: Inconsistent Metric Scores
**Problem:** Evaluation metrics vary significantly between runs
**Solutions:**
- Set random seeds for reproducibility
- Use multiple evaluation runs and average results
- Implement confidence intervals
- Validate with human evaluation

```python
import numpy as np

def stable_evaluation(responses, num_runs=3):
    all_scores = []
    for run in range(num_runs):
        np.random.seed(42 + run)  # Different seed each run
        scores = evaluate_batch(responses)
        all_scores.append(scores)
    
    # Calculate mean and confidence intervals
    mean_scores = np.mean(all_scores, axis=0)
    std_scores = np.std(all_scores, axis=0)
    
    return {
        'mean': mean_scores,
        'std': std_scores,
        'confidence_interval': 1.96 * std_scores / np.sqrt(num_runs)
    }
```

### Issue 4: Memory Issues with Large Datasets
**Problem:** Out of memory errors with large evaluation datasets
**Solutions:**
- Process data in batches
- Use generators instead of loading all data
- Implement data streaming
- Clear intermediate results

```python
def batch_evaluation(responses, batch_size=50):
    results = []
    for i in range(0, len(responses), batch_size):
        batch = responses[i:i + batch_size]
        batch_results = evaluate_batch(batch)
        results.extend(batch_results)
        
        # Clear memory
        del batch_results
        import gc
        gc.collect()
    
    return results
```

## Production Deployment

### 1. Monitoring Dashboard
```python
import streamlit as st
import plotly.express as px

def create_monitoring_dashboard():
    st.title("RAG Evaluation Dashboard")
    
    # Real-time metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Faithfulness", "0.82", "↑ 0.05")
    with col2:
        st.metric("Relevancy", "0.89", "↓ 0.02")
    with col3:
        st.metric("Precision", "0.76", "→ 0.00")
    with col4:
        st.metric("Overall Score", "0.84", "↑ 0.03")
    
    # Performance trends
    st.subheader("Performance Trends")
    # Add trend charts here
    
    # Active alerts
    st.subheader("Active Alerts")
    # Add alerts table here

# Run dashboard
if __name__ == "__main__":
    create_monitoring_dashboard()
```

### 2. API Service
```python
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel

app = FastAPI(title="RAG Evaluation API")

class EvaluationRequest(BaseModel):
    responses: List[dict]
    metrics: List[str] = ["faithfulness", "answer_relevancy"]

@app.post("/evaluate")
async def evaluate_responses(request: EvaluationRequest):
    # Convert to RAGResponse objects
    rag_responses = [RAGResponse(**r) for r in request.responses]
    
    # Evaluate
    results = evaluation_pipeline.evaluate_batch(rag_responses)
    
    return {
        "evaluation_id": generate_evaluation_id(),
        "results": [asdict(r) for r in results],
        "summary": generate_summary(results)
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now()}
```

### 3. Automated Evaluation Pipeline
```python
import schedule
import time

def automated_evaluation_job():
    """Run automated evaluation on production data"""
    # Fetch recent RAG responses
    recent_responses = fetch_recent_responses()
    
    # Evaluate
    results = evaluation_pipeline.evaluate_batch(recent_responses)
    
    # Store results
    store_evaluation_results(results)
    
    # Check for alerts
    check_quality_alerts(results)
    
    # Generate report
    generate_daily_report(results)

# Schedule evaluations
schedule.every().hour.do(automated_evaluation_job)
schedule.every().day.at("09:00").do(generate_daily_report)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Next Steps

1. **Complete the exercises** in `exercise.py`
2. **Run the test suite** to verify your implementation
3. **Experiment with different evaluation strategies**:
   - Try different RAGAS metrics combinations
   - Test various statistical significance methods
   - Implement custom domain-specific metrics
4. **Build a monitoring dashboard** using Streamlit or Dash
5. **Set up automated evaluation** for production systems
6. **Move to Day 54**: Project - Production RAG System

## Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub Repository](https://github.com/explodinggradients/ragas)
- [Statistical Testing in Python](https://docs.scipy.org/doc/scipy/reference/stats.html)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [RAG Evaluation Best Practices](https://arxiv.org/abs/2401.15884)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the test cases for examples
3. Consult the solution file for complete implementations
4. Check RAGAS documentation for API changes
5. Verify your API keys and rate limits