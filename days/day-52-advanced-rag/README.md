# Day 52: Advanced RAG - Retrieval-Augmented Generation Systems

## Learning Objectives
By the end of this session, you will be able to:
- Implement hybrid search combining dense and sparse retrieval methods
- Build advanced re-ranking systems for improved relevance
- Design production-ready RAG architectures with proper scaling
- Apply advanced retrieval techniques like query expansion and multi-hop reasoning
- Optimize RAG systems for latency, accuracy, and cost in production environments

## Theory (15 minutes)

### What is Advanced RAG?

Retrieval-Augmented Generation (RAG) has evolved beyond basic vector similarity search. Advanced RAG systems combine multiple retrieval strategies, sophisticated re-ranking mechanisms, and production-optimized architectures to deliver highly relevant and contextually accurate responses.

### Core Components of Advanced RAG

#### 1. Hybrid Search Architecture

**Dense Retrieval (Vector Search)**
```python
# Semantic similarity using embeddings
query_embedding = embedding_model.encode(query)
similar_docs = vector_db.similarity_search(query_embedding, k=20)
```

**Sparse Retrieval (Keyword Search)**
```python
# Traditional keyword matching (BM25, TF-IDF)
keyword_results = bm25_index.search(query, k=20)
```

**Hybrid Fusion**
```python
# Reciprocal Rank Fusion (RRF)
def reciprocal_rank_fusion(dense_results, sparse_results, k=60):
    scores = {}
    for rank, doc in enumerate(dense_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank + 1)
    for rank, doc in enumerate(sparse_results):
        scores[doc.id] = scores.get(doc.id, 0) + 1/(k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

#### 2. Advanced Re-ranking Systems

**Cross-Encoder Re-ranking**
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query, documents, top_k=5):
    pairs = [(query, doc.content) for doc in documents]
    scores = reranker.predict(pairs)
    
    # Sort by relevance score
    ranked_docs = sorted(zip(documents, scores), 
                        key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked_docs[:top_k]]
```

**Multi-stage Retrieval Pipeline**
```python
class AdvancedRAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = CrossEncoderReranker()
        self.generator = LLMGenerator()
    
    def retrieve_and_generate(self, query):
        # Stage 1: Hybrid retrieval (broad recall)
        candidates = self.retriever.search(query, k=100)
        
        # Stage 2: Re-ranking (precision)
        relevant_docs = self.reranker.rerank(query, candidates, k=5)
        
        # Stage 3: Generation with context
        response = self.generator.generate(query, relevant_docs)
        return response
```

#### 3. Query Enhancement Techniques

**Query Expansion**
```python
def expand_query(original_query, expansion_model):
    # Generate related terms and synonyms
    expanded_terms = expansion_model.generate_expansions(original_query)
    
    # Combine with original query
    enhanced_query = f"{original_query} {' '.join(expanded_terms)}"
    return enhanced_query
```

**Multi-Query Generation**
```python
def generate_multiple_queries(original_query, llm):
    prompt = f"""
    Generate 3 different ways to ask this question: {original_query}
    Focus on different aspects and phrasings.
    """
    alternative_queries = llm.generate(prompt).split('\n')
    return [original_query] + alternative_queries
```

#### 4. Advanced Retrieval Strategies

**Hierarchical Retrieval**
```python
class HierarchicalRetriever:
    def __init__(self):
        self.document_retriever = DocumentRetriever()
        self.chunk_retriever = ChunkRetriever()
    
    def retrieve(self, query):
        # First, find relevant documents
        relevant_docs = self.document_retriever.search(query, k=10)
        
        # Then, find specific chunks within those documents
        relevant_chunks = []
        for doc in relevant_docs:
            chunks = self.chunk_retriever.search_within_doc(query, doc, k=3)
            relevant_chunks.extend(chunks)
        
        return relevant_chunks
```

**Multi-hop Reasoning**
```python
def multi_hop_retrieval(query, max_hops=3):
    current_query = query
    all_context = []
    
    for hop in range(max_hops):
        # Retrieve documents for current query
        docs = retriever.search(current_query, k=5)
        all_context.extend(docs)
        
        # Generate follow-up query based on retrieved content
        follow_up = generate_follow_up_query(current_query, docs)
        if not follow_up or follow_up == current_query:
            break
        current_query = follow_up
    
    return all_context
```

### Production Optimization Strategies

#### 1. Caching and Performance

**Multi-level Caching**
```python
class CachedRAGSystem:
    def __init__(self):
        self.query_cache = LRUCache(maxsize=1000)
        self.embedding_cache = RedisCache()
        self.result_cache = DatabaseCache()
    
    def search(self, query):
        # Check query cache first
        if query in self.query_cache:
            return self.query_cache[query]
        
        # Check embedding cache
        query_hash = hash(query)
        if self.embedding_cache.exists(query_hash):
            embedding = self.embedding_cache.get(query_hash)
        else:
            embedding = self.embed_query(query)
            self.embedding_cache.set(query_hash, embedding)
        
        # Perform search and cache result
        results = self.vector_search(embedding)
        self.query_cache[query] = results
        return results
```

#### 2. Async Processing

**Concurrent Retrieval**
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncRAGSystem:
    async def parallel_search(self, query):
        loop = asyncio.get_event_loop()
        
        # Run different retrieval methods concurrently
        with ThreadPoolExecutor() as executor:
            dense_task = loop.run_in_executor(executor, self.dense_search, query)
            sparse_task = loop.run_in_executor(executor, self.sparse_search, query)
            
            dense_results, sparse_results = await asyncio.gather(
                dense_task, sparse_task
            )
        
        # Combine results
        return self.fusion_strategy(dense_results, sparse_results)
```

#### 3. Monitoring and Evaluation

**RAG Metrics Tracking**
```python
class RAGMetrics:
    def __init__(self):
        self.metrics = {
            'retrieval_latency': [],
            'generation_latency': [],
            'relevance_scores': [],
            'user_feedback': []
        }
    
    def track_retrieval(self, query, results, latency):
        self.metrics['retrieval_latency'].append(latency)
        
        # Calculate retrieval quality metrics
        relevance = self.calculate_relevance(query, results)
        self.metrics['relevance_scores'].append(relevance)
    
    def calculate_mrr(self, queries_and_results):
        """Mean Reciprocal Rank"""
        reciprocal_ranks = []
        for query, results, ground_truth in queries_and_results:
            for rank, result in enumerate(results, 1):
                if result.id in ground_truth:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)
        return sum(reciprocal_ranks) / len(reciprocal_ranks)
```

### Why Advanced RAG Matters

1. **Improved Accuracy**: Hybrid search and re-ranking significantly improve retrieval precision
2. **Better User Experience**: Faster responses and more relevant results
3. **Production Scalability**: Optimized architectures handle enterprise-scale workloads
4. **Cost Efficiency**: Smart caching and optimization reduce computational costs
5. **Measurable Quality**: Advanced metrics enable continuous improvement

### Real-world Applications

- **Enterprise Search**: Internal knowledge bases with millions of documents
- **Customer Support**: Intelligent chatbots with accurate information retrieval
- **Research Assistance**: Academic and scientific literature search
- **Legal Discovery**: Case law and document analysis
- **Medical Information**: Clinical decision support systems

## Exercise (40 minutes)
Complete the hands-on exercises in `exercise.py` to build production-ready RAG systems with hybrid search, re-ranking, and optimization techniques.

## Resources
- [LangChain RAG Documentation](https://python.langchain.com/docs/use_cases/question_answering)
- [Pinecone Hybrid Search Guide](https://docs.pinecone.io/docs/hybrid-search)
- [Sentence Transformers Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html)
- [RAG Evaluation with RAGAS](https://docs.ragas.io/)
- [Advanced RAG Techniques Paper](https://arxiv.org/abs/2312.10997)

## Next Steps
- Complete the advanced RAG exercises
- Review production optimization patterns
- Take the quiz to test your understanding
- Move to Day 53: RAG Evaluation with RAGAS
