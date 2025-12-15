# Day 52: Advanced RAG Setup Guide

## Overview
This guide helps you set up the environment for advanced Retrieval-Augmented Generation (RAG) systems with hybrid search, re-ranking, and production optimization.

## Prerequisites
- Python 3.8+
- 8GB+ RAM (16GB recommended for large embeddings)
- Basic understanding of vector databases and embeddings
- Completed Day 51 (LLM Serving Optimization)

## Quick Setup

### 1. Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Optional: Install specific vector databases
pip install faiss-gpu  # If you have CUDA GPU
pip install chromadb[server]  # For ChromaDB server mode
```

### 2. Verify Installation
```bash
python -c "import numpy, sentence_transformers, faiss; print('✅ Core libraries installed')"
```

### 3. Run Basic Test
```bash
python exercise.py
```

## Detailed Setup

### Environment Configuration

Create `.env` file:
```bash
# Vector Database Configuration
CHROMA_HOST=localhost
CHROMA_PORT=8000
PINECONE_API_KEY=your_pinecone_key
PINECONE_ENVIRONMENT=your_environment

# Embedding Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Cache Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
CACHE_TTL=3600

# Performance Settings
MAX_CONCURRENT_REQUESTS=10
BATCH_SIZE=32
EMBEDDING_CACHE_SIZE=1000
QUERY_CACHE_SIZE=500

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
```

### Vector Database Setup

#### Option 1: ChromaDB (Recommended for Development)
```bash
# Install ChromaDB
pip install chromadb

# Start ChromaDB server (optional)
chroma run --host localhost --port 8000
```

#### Option 2: Faiss (High Performance)
```bash
# CPU version
pip install faiss-cpu

# GPU version (if CUDA available)
pip install faiss-gpu
```

#### Option 3: Pinecone (Cloud)
```bash
pip install pinecone-client

# Set up Pinecone
export PINECONE_API_KEY="your-api-key"
export PINECONE_ENVIRONMENT="your-environment"
```

### Redis Setup (Optional - for Caching)
```bash
# Install Redis
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

### 1. Hybrid Search Architecture
Advanced RAG combines multiple retrieval methods:

```python
# Dense retrieval (semantic similarity)
query_embedding = embedding_model.encode(query)
dense_results = vector_db.similarity_search(query_embedding, k=20)

# Sparse retrieval (keyword matching)
sparse_results = bm25_index.search(query, k=20)

# Hybrid fusion using Reciprocal Rank Fusion (RRF)
combined_results = reciprocal_rank_fusion(dense_results, sparse_results)
```

### 2. Re-ranking Pipeline
Improve relevance with cross-encoder models:

```python
# Initial retrieval (broad recall)
candidates = hybrid_retriever.search(query, k=100)

# Re-ranking (high precision)
reranked = cross_encoder.rerank(query, candidates, k=10)

# Final selection
final_results = reranked[:5]
```

### 3. Production Optimizations

**Multi-level Caching:**
- Query cache: Store complete search results
- Embedding cache: Cache computed embeddings
- Result cache: Cache intermediate results

**Async Processing:**
- Parallel retrieval methods
- Concurrent re-ranking
- Batch processing

**Monitoring:**
- Latency tracking
- Cache hit rates
- Quality metrics (MRR, Recall@K)

## Testing Your Setup

### 1. Run Unit Tests
```bash
# Run all tests
pytest test_advanced_rag.py -v

# Run specific test categories
pytest test_advanced_rag.py::TestHybridRetriever -v
pytest test_advanced_rag.py::TestCrossEncoderReranker -v
pytest test_advanced_rag.py::TestProductionRAGSystem -v

# Run async tests
pytest test_advanced_rag.py -k "async" -v
```

### 2. Performance Benchmarks
```bash
# Run performance tests
python -c "
from solution import ProductionRAGSystem, Document
import time

# Create test system
docs = [Document(str(i), f'Document {i} content', {}) for i in range(100)]
system = ProductionRAGSystem(docs)

# Benchmark search
start = time.time()
for i in range(10):
    system.search(f'query {i}', top_k=5)
end = time.time()

print(f'Average search time: {(end-start)/10*1000:.2f}ms')
"
```

### 3. Memory Usage Check
```bash
python -c "
import psutil
import os
from solution import ProductionRAGSystem, Document

# Monitor memory before
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 / 1024

# Create system with many documents
docs = [Document(str(i), f'Document {i} with content' * 10, {}) for i in range(1000)]
system = ProductionRAGSystem(docs)

# Monitor memory after
mem_after = process.memory_info().rss / 1024 / 1024
print(f'Memory usage: {mem_after - mem_before:.1f} MB for 1000 documents')
"
```

## Common Issues & Solutions

### Issue 1: Slow Embedding Computation
**Problem:** Embedding generation is too slow
**Solutions:**
- Use smaller embedding models (e.g., all-MiniLM-L6-v2)
- Enable GPU acceleration if available
- Implement embedding caching
- Use batch processing for multiple texts

```python
# Optimize embedding computation
model = SentenceTransformer('all-MiniLM-L6-v2')
model = model.to('cuda')  # Use GPU if available

# Batch processing
embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
```

### Issue 2: High Memory Usage
**Problem:** System uses too much memory
**Solutions:**
- Use memory-efficient vector databases (Faiss with IVF)
- Implement LRU caching with size limits
- Use quantized embeddings
- Stream processing for large document sets

```python
# Memory-efficient setup
import faiss

# Use IVF index for large datasets
index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
index.train(training_embeddings)
```

### Issue 3: Poor Retrieval Quality
**Problem:** Retrieved documents are not relevant
**Solutions:**
- Tune hybrid search weights
- Improve query expansion
- Use domain-specific embedding models
- Implement better re-ranking

```python
# Tune RRF parameters
def reciprocal_rank_fusion(dense_results, sparse_results, k=60, alpha=0.6):
    # Adjust alpha to weight dense vs sparse results
    for rank, (doc, score) in enumerate(dense_results):
        scores[doc.id] += alpha * (1 / (k + rank + 1))
    for rank, (doc, score) in enumerate(sparse_results):
        scores[doc.id] += (1 - alpha) * (1 / (k + rank + 1))
```

### Issue 4: Slow Search Performance
**Problem:** Search queries take too long
**Solutions:**
- Implement proper caching
- Use async processing
- Optimize vector database configuration
- Reduce candidate set size

```python
# Performance optimization
async def optimized_search(query, top_k=5):
    # Parallel retrieval
    dense_task = asyncio.create_task(dense_search(query))
    sparse_task = asyncio.create_task(sparse_search(query))
    
    dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
    
    # Quick fusion and return
    return fusion_and_rerank(dense_results, sparse_results, top_k)
```

## Production Deployment

### 1. Docker Setup
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. API Service
```python
from fastapi import FastAPI
from solution import ProductionRAGSystem

app = FastAPI()
rag_system = ProductionRAGSystem(documents)

@app.post("/search")
async def search(query: str, top_k: int = 5):
    return await rag_system.async_search(query, top_k)

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/metrics")
async def metrics():
    return rag_system.get_system_stats()
```

### 3. Monitoring Setup
```python
# Prometheus metrics
from prometheus_client import Counter, Histogram, generate_latest

search_counter = Counter('rag_searches_total', 'Total searches')
search_duration = Histogram('rag_search_duration_seconds', 'Search duration')

@app.middleware("http")
async def add_metrics(request, call_next):
    if request.url.path == "/search":
        search_counter.inc()
        with search_duration.time():
            response = await call_next(request)
    else:
        response = await call_next(request)
    return response
```

## Next Steps

1. **Complete the exercises** in `exercise.py`
2. **Run the test suite** to verify your implementation
3. **Experiment with different configurations**:
   - Try different embedding models
   - Tune RRF parameters
   - Test various caching strategies
4. **Build a production API** using FastAPI
5. **Set up monitoring** with Prometheus/Grafana
6. **Move to Day 53**: RAG Evaluation with RAGAS

## Resources

- [Sentence Transformers Documentation](https://www.sbert.net/)
- [Faiss Documentation](https://faiss.ai/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Pinecone Documentation](https://docs.pinecone.io/)
- [Advanced RAG Techniques Paper](https://arxiv.org/abs/2312.10997)
- [Hybrid Search Best Practices](https://docs.pinecone.io/docs/hybrid-search)

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the test cases for examples
3. Consult the solution file for complete implementations
4. Check system requirements and dependencies