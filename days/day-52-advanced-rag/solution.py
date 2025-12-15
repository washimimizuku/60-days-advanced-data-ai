"""
Day 52: Advanced RAG - Complete Solutions
Production-ready implementations of hybrid search, re-ranking, and optimization
"""

import numpy as np
import time
import asyncio
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib


@dataclass
class Document:
    """Document representation for RAG system"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class MockEmbeddingModel:
    """Mock embedding model for demonstrations"""
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings with realistic properties"""
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            # Create deterministic embedding based on text content
            text_hash = int(hashlib.md5(text.encode()).hexdigest(), 16)
            np.random.seed(text_hash % (2**32))
            
            embedding = np.random.rand(384)
            
            # Add semantic features
            words = text.lower().split()
            embedding[0] = len(text) / 1000  # Length feature
            embedding[1] = len(words) / 100  # Word count feature
            embedding[2] = len(set(words)) / len(words) if words else 0  # Diversity
            
            # Add domain-specific features
            ml_words = ['machine', 'learning', 'algorithm', 'model', 'data']
            embedding[3] = sum(1 for word in ml_words if word in text.lower()) / len(ml_words)
            
            embeddings.append(embedding)
        
        return np.array(embeddings)


class BM25Retriever:
    """Production-ready BM25 implementation"""
    
    def __init__(self, documents: List[Document], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = {}
        self.avgdl = 0
        self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().replace('.', '').replace(',', '').split()
    
    def _build_index(self):
        """Build BM25 index with proper preprocessing"""
        total_len = 0
        word_doc_count = defaultdict(int)
        
        for doc in self.documents:
            words = self._tokenize(doc.content)
            self.doc_len[doc.id] = len(words)
            total_len += len(words)
            
            # Count unique words per document
            unique_words = set(words)
            for word in unique_words:
                word_doc_count[word] += 1
        
        self.avgdl = total_len / len(self.documents) if self.documents else 0
        
        # Calculate IDF for each word
        N = len(self.documents)
        for word, doc_count in word_doc_count.items():
            self.idf[word] = np.log((N - doc_count + 0.5) / (doc_count + 0.5))
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """BM25 search with proper scoring"""
        query_words = self._tokenize(query)
        scores = {}
        
        for doc in self.documents:
            doc_words = self._tokenize(doc.content)
            word_counts = defaultdict(int)
            for word in doc_words:
                word_counts[word] += 1
            
            score = 0
            for word in query_words:
                if word in word_counts and word in self.idf:
                    tf = word_counts[word]
                    idf = self.idf[word]
                    
                    # BM25 formula
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[doc.id] / self.avgdl)
                    score += idf * (numerator / denominator)
            
            if score > 0:
                scores[doc.id] = score
        
        # Sort by score and return top k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(next(d for d in self.documents if d.id == doc_id), score) 
                for doc_id, score in sorted_docs]


# Solution 1: Hybrid Search with RRF
class HybridRetriever:
    """Complete hybrid retrieval system with RRF"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.embedding_model = MockEmbeddingModel()
        self.bm25_retriever = BM25Retriever(documents)
        
        # Pre-compute embeddings
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(contents)
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def dense_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Dense (vector) search using cosine similarity"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        similarities = []
        for doc in self.documents:
            if doc.embedding is not None:
                similarity = self.cosine_similarity(query_embedding, doc.embedding)
                similarities.append((doc, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]
    
    def sparse_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Sparse search using BM25"""
        return self.bm25_retriever.search(query, k)
    
    def reciprocal_rank_fusion(self, dense_results: List[Tuple[Document, float]], 
                             sparse_results: List[Tuple[Document, float]], 
                             k: int = 60) -> List[Tuple[Document, float]]:
        """Reciprocal Rank Fusion implementation"""
        scores = defaultdict(float)
        
        # Add scores from dense results
        for rank, (doc, _) in enumerate(dense_results):
            scores[doc.id] += 1 / (k + rank + 1)
        
        # Add scores from sparse results
        for rank, (doc, _) in enumerate(sparse_results):
            scores[doc.id] += 1 / (k + rank + 1)
        
        # Sort by combined scores
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return documents with their RRF scores
        result = []
        for doc_id, score in sorted_scores:
            doc = next(d for d in self.documents if d.id == doc_id)
            result.append((doc, score))
        
        return result
    
    def hybrid_search(self, query: str, k: int = 5) -> List[Tuple[Document, float]]:
        """Complete hybrid search with RRF"""
        # Get results from both methods
        dense_results = self.dense_search(query, k=20)
        sparse_results = self.sparse_search(query, k=20)
        
        # Apply reciprocal rank fusion
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
        
        return fused_results[:k]


# Solution 2: Cross-Encoder Re-ranking
class CrossEncoderReranker:
    """Cross-encoder re-ranking implementation"""
    
    def __init__(self):
        # In production, this would load a real cross-encoder model
        pass
    
    def _calculate_overlap_score(self, query: str, document: str) -> float:
        """Calculate word overlap score as a proxy for relevance"""
        query_words = set(query.lower().split())
        doc_words = set(document.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(doc_words))
        return overlap / len(query_words)
    
    def _calculate_length_penalty(self, document: str) -> float:
        """Apply length penalty - prefer moderate length documents"""
        doc_len = len(document.split())
        optimal_length = 50  # Assume 50 words is optimal
        penalty = 1.0 - abs(doc_len - optimal_length) / (optimal_length * 2)
        return max(0.1, penalty)  # Minimum penalty of 0.1
    
    def score_pairs(self, query: str, documents: List[Document]) -> List[float]:
        """Score query-document pairs using mock cross-encoder"""
        scores = []
        
        for doc in documents:
            # Combine multiple scoring factors
            overlap_score = self._calculate_overlap_score(query, doc.content)
            length_penalty = self._calculate_length_penalty(doc.content)
            
            # Add some randomness for realism (in production, this would be model prediction)
            text_hash = hash(query + doc.content) % 1000
            randomness = (text_hash / 1000) * 0.2  # 20% randomness
            
            final_score = (overlap_score * 0.7 + randomness * 0.3) * length_penalty
            scores.append(final_score)
        
        return scores
    
    def rerank(self, query: str, documents: List[Document], top_k: int = 5) -> List[Tuple[Document, float]]:
        """Re-rank documents using cross-encoder scores"""
        if not documents:
            return []
        
        scores = self.score_pairs(query, documents)
        
        # Combine documents with scores and sort
        doc_score_pairs = list(zip(documents, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        return doc_score_pairs[:top_k]


# Solution 3: Query Expansion
class QueryExpander:
    """Advanced query expansion implementation"""
    
    def __init__(self):
        # Comprehensive synonym dictionary
        self.synonyms = {
            "machine": ["algorithm", "automated", "computer", "artificial", "system"],
            "learning": ["training", "education", "study", "acquisition", "development"],
            "data": ["information", "dataset", "records", "facts", "statistics"],
            "analysis": ["examination", "evaluation", "study", "investigation", "assessment"],
            "model": ["framework", "system", "structure", "representation", "architecture"],
            "neural": ["network", "artificial", "brain", "cognitive", "synaptic"],
            "deep": ["profound", "advanced", "complex", "multilayer", "hierarchical"],
            "natural": ["human", "linguistic", "conversational", "textual", "semantic"],
            "language": ["text", "linguistic", "verbal", "communication", "speech"],
            "processing": ["analysis", "computation", "handling", "manipulation", "treatment"]
        }
    
    def expand_with_synonyms(self, query: str, max_synonyms: int = 2) -> str:
        """Expand query with relevant synonyms"""
        words = query.lower().split()
        expanded_terms = []
        
        for word in words:
            expanded_terms.append(word)
            if word in self.synonyms:
                # Add up to max_synonyms for each word
                synonyms = self.synonyms[word][:max_synonyms]
                expanded_terms.extend(synonyms)
        
        return " ".join(expanded_terms)
    
    def generate_alternative_queries(self, query: str, num_alternatives: int = 2) -> List[str]:
        """Generate alternative query formulations"""
        words = query.split()
        alternatives = []
        
        # Alternative 1: Add question words
        question_starters = ["what is", "how does", "explain", "describe"]
        if num_alternatives >= 1:
            starter = question_starters[hash(query) % len(question_starters)]
            alternatives.append(f"{starter} {query}")
        
        # Alternative 2: Reorder words (for multi-word queries)
        if len(words) > 1 and num_alternatives >= 2:
            reversed_query = " ".join(reversed(words))
            alternatives.append(reversed_query)
        
        # Alternative 3: Add context words
        if num_alternatives >= 3:
            context_words = ["advanced", "introduction to", "basics of"]
            context = context_words[hash(query) % len(context_words)]
            alternatives.append(f"{context} {query}")
        
        return alternatives[:num_alternatives]
    
    def multi_query_search(self, original_query: str, retriever, k: int = 5) -> List[Tuple[Document, float]]:
        """Search using multiple query variations and combine results"""
        # Generate query variations
        expanded_query = self.expand_with_synonyms(original_query)
        alternative_queries = self.generate_alternative_queries(original_query, 2)
        
        all_queries = [original_query, expanded_query] + alternative_queries
        
        # Search with each query variation
        all_results = {}
        for query_variant in all_queries:
            results = retriever.hybrid_search(query_variant, k=k*2)
            for doc, score in results:
                if doc.id in all_results:
                    # Combine scores (average)
                    all_results[doc.id] = (all_results[doc.id][0], 
                                         (all_results[doc.id][1] + score) / 2)
                else:
                    all_results[doc.id] = (doc, score)
        
        # Sort by combined scores
        sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
        return sorted_results[:k]


# Solution 4: Multi-level Caching System
class LRUCache:
    """Efficient LRU Cache implementation"""
    
    def __init__(self, maxsize: int = 100):
        self.maxsize = maxsize
        self.cache = {}
        self.access_order = []
    
    def get(self, key):
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, key, value):
        if key in self.cache:
            # Update existing key
            self.access_order.remove(key)
        elif len(self.cache) >= self.maxsize:
            # Remove least recently used
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.access_order.append(key)
    
    def size(self):
        return len(self.cache)


class CachedRAGSystem:
    """Production-ready caching system for RAG"""
    
    def __init__(self):
        self.query_cache = LRUCache(maxsize=100)
        self.embedding_cache = LRUCache(maxsize=500)
        self.result_cache = LRUCache(maxsize=200)
        self.embedding_model = MockEmbeddingModel()
        
        # Metrics
        self.search_count = 0
        self.cache_hits = 0
        self.embedding_cache_hits = 0
        self.embedding_computations = 0
    
    def _hash_query(self, query: str) -> str:
        """Create hash for query caching"""
        return hashlib.md5(query.encode()).hexdigest()
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        text_hash = self._hash_query(text)
        
        # Check cache first
        cached_embedding = self.embedding_cache.get(text_hash)
        if cached_embedding is not None:
            self.embedding_cache_hits += 1
            return cached_embedding
        
        # Compute and cache embedding
        embedding = self.embedding_model.encode([text])[0]
        self.embedding_cache.set(text_hash, embedding)
        self.embedding_computations += 1
        
        return embedding
    
    def cached_search(self, query: str, documents: List[Document], 
                     search_func, **kwargs) -> List[Tuple[Document, float]]:
        """Perform cached search with any search function"""
        self.search_count += 1
        query_hash = self._hash_query(f"{query}_{str(kwargs)}")
        
        # Check query cache
        cached_result = self.query_cache.get(query_hash)
        if cached_result is not None:
            self.cache_hits += 1
            return cached_result
        
        # Perform search
        results = search_func(query, **kwargs)
        
        # Cache results
        self.query_cache.set(query_hash, results)
        
        return results
    
    def get_cache_stats(self) -> Dict:
        """Return comprehensive cache statistics"""
        query_hit_rate = self.cache_hits / self.search_count if self.search_count > 0 else 0
        embedding_hit_rate = (self.embedding_cache_hits / 
                             (self.embedding_cache_hits + self.embedding_computations) 
                             if (self.embedding_cache_hits + self.embedding_computations) > 0 else 0)
        
        return {
            "total_searches": self.search_count,
            "query_cache_hits": self.cache_hits,
            "query_hit_rate": query_hit_rate,
            "embedding_cache_hits": self.embedding_cache_hits,
            "embedding_computations": self.embedding_computations,
            "embedding_hit_rate": embedding_hit_rate,
            "query_cache_size": self.query_cache.size(),
            "embedding_cache_size": self.embedding_cache.size()
        }


# Solution 5: Asynchronous RAG Pipeline
class AsyncRAGPipeline:
    """High-performance async RAG pipeline"""
    
    def __init__(self):
        self.embedding_model = MockEmbeddingModel()
        self.cached_system = CachedRAGSystem()
    
    async def async_dense_search(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Async dense search simulation"""
        # Simulate network/computation delay
        await asyncio.sleep(0.1)
        
        # Perform dense search
        retriever = HybridRetriever(documents)
        return retriever.dense_search(query, k=10)
    
    async def async_sparse_search(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Async sparse search simulation"""
        # Simulate network/computation delay
        await asyncio.sleep(0.08)
        
        # Perform sparse search
        bm25_retriever = BM25Retriever(documents)
        return bm25_retriever.search(query, k=10)
    
    async def async_rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Async re-ranking simulation"""
        # Simulate model inference delay
        await asyncio.sleep(0.05)
        
        # Perform re-ranking
        reranker = CrossEncoderReranker()
        return reranker.rerank(query, documents, top_k=len(documents))
    
    async def parallel_search_and_rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Run search methods in parallel and combine results"""
        # Run dense and sparse search concurrently
        dense_task = self.async_dense_search(query, documents)
        sparse_task = self.async_sparse_search(query, documents)
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        # Combine using RRF
        retriever = HybridRetriever(documents)
        combined_results = retriever.reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Get top candidates for re-ranking
        top_candidates = [doc for doc, _ in combined_results[:20]]
        
        # Apply async re-ranking
        final_results = await self.async_rerank(query, top_candidates)
        
        return final_results[:5]


# Solution 6: RAG Metrics and Evaluation
class RAGEvaluator:
    """Comprehensive RAG evaluation system"""
    
    def __init__(self):
        self.metrics_history = []
        self.latency_history = defaultdict(list)
    
    def calculate_mrr(self, queries_results_relevance: List[Tuple[str, List[Document], List[str]]]) -> float:
        """Calculate Mean Reciprocal Rank"""
        reciprocal_ranks = []
        
        for query, retrieved_docs, relevant_doc_ids in queries_results_relevance:
            # Find rank of first relevant document
            for rank, doc in enumerate(retrieved_docs, 1):
                if doc.id in relevant_doc_ids:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                # No relevant document found
                reciprocal_ranks.append(0.0)
        
        return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def calculate_recall_at_k(self, retrieved_docs: List[Document], 
                            relevant_docs: List[str], k: int) -> float:
        """Calculate Recall@K"""
        if not relevant_docs:
            return 0.0
        
        retrieved_ids = [doc.id for doc in retrieved_docs[:k]]
        relevant_found = len(set(retrieved_ids).intersection(set(relevant_docs)))
        
        return relevant_found / len(relevant_docs)
    
    def calculate_precision_at_k(self, retrieved_docs: List[Document], 
                               relevant_docs: List[str], k: int) -> float:
        """Calculate Precision@K"""
        if k == 0:
            return 0.0
        
        retrieved_ids = [doc.id for doc in retrieved_docs[:k]]
        relevant_found = len(set(retrieved_ids).intersection(set(relevant_docs)))
        
        return relevant_found / min(k, len(retrieved_docs))
    
    def calculate_f1_at_k(self, retrieved_docs: List[Document], 
                         relevant_docs: List[str], k: int) -> float:
        """Calculate F1@K score"""
        precision = self.calculate_precision_at_k(retrieved_docs, relevant_docs, k)
        recall = self.calculate_recall_at_k(retrieved_docs, relevant_docs, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[Document], 
                                 relevant_doc_ids: List[str]) -> Dict[str, float]:
        """Comprehensive retrieval evaluation"""
        metrics = {}
        
        # Calculate metrics for different k values
        for k in [1, 3, 5, 10]:
            if k <= len(retrieved_docs):
                metrics[f"precision_at_{k}"] = self.calculate_precision_at_k(
                    retrieved_docs, relevant_doc_ids, k)
                metrics[f"recall_at_{k}"] = self.calculate_recall_at_k(
                    retrieved_docs, relevant_doc_ids, k)
                metrics[f"f1_at_{k}"] = self.calculate_f1_at_k(
                    retrieved_docs, relevant_doc_ids, k)
        
        # Calculate MRR for this single query
        mrr = self.calculate_mrr([(query, retrieved_docs, relevant_doc_ids)])
        metrics["mrr"] = mrr
        
        return metrics
    
    def track_latency(self, operation: str, latency: float):
        """Track operation latencies"""
        self.latency_history[operation].append(latency)
    
    def get_performance_report(self) -> Dict:
        """Generate comprehensive performance report"""
        report = {
            "total_evaluations": len(self.metrics_history),
            "latency_stats": {}
        }
        
        # Calculate latency statistics
        for operation, latencies in self.latency_history.items():
            if latencies:
                report["latency_stats"][operation] = {
                    "mean": np.mean(latencies),
                    "median": np.median(latencies),
                    "p95": np.percentile(latencies, 95),
                    "p99": np.percentile(latencies, 99),
                    "count": len(latencies)
                }
        
        # Calculate average metrics if we have history
        if self.metrics_history:
            avg_metrics = {}
            for key in self.metrics_history[0].keys():
                values = [m[key] for m in self.metrics_history if key in m]
                if values:
                    avg_metrics[f"avg_{key}"] = np.mean(values)
            report["average_metrics"] = avg_metrics
        
        return report


# Solution 7: Production RAG System Integration
class ProductionRAGSystem:
    """Complete production-ready RAG system"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        
        # Initialize all components
        self.hybrid_retriever = HybridRetriever(documents)
        self.reranker = CrossEncoderReranker()
        self.query_expander = QueryExpander()
        self.cached_system = CachedRAGSystem()
        self.evaluator = RAGEvaluator()
        self.async_pipeline = AsyncRAGPipeline()
        
        # System statistics
        self.total_queries = 0
        self.start_time = time.time()
    
    def search(self, query: str, top_k: int = 5, use_expansion: bool = True, 
              use_reranking: bool = True, use_cache: bool = True) -> Dict:
        """Complete end-to-end RAG search with all optimizations"""
        start_time = time.time()
        self.total_queries += 1
        
        try:
            # Step 1: Query expansion (if enabled)
            if use_expansion:
                expanded_query = self.query_expander.expand_with_synonyms(query)
                search_query = expanded_query
            else:
                search_query = query
            
            # Step 2: Hybrid search with caching
            if use_cache:
                search_func = lambda q, **kwargs: self.hybrid_retriever.hybrid_search(q, **kwargs)
                results = self.cached_system.cached_search(
                    search_query, self.documents, search_func, k=top_k*2)
            else:
                results = self.hybrid_retriever.hybrid_search(search_query, k=top_k*2)
            
            # Step 3: Re-ranking (if enabled)
            if use_reranking and results:
                candidate_docs = [doc for doc, _ in results]
                reranked_results = self.reranker.rerank(query, candidate_docs, top_k=top_k)
            else:
                reranked_results = results[:top_k]
            
            # Step 4: Track metrics
            search_latency = time.time() - start_time
            self.evaluator.track_latency("search", search_latency)
            
            # Prepare response
            response = {
                "query": query,
                "expanded_query": search_query if use_expansion else None,
                "documents": [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    }
                    for doc, score in reranked_results
                ],
                "search_metadata": {
                    "latency_ms": search_latency * 1000,
                    "total_candidates": len(results),
                    "final_results": len(reranked_results),
                    "used_expansion": use_expansion,
                    "used_reranking": use_reranking,
                    "used_cache": use_cache
                }
            }
            
            return response
            
        except Exception as e:
            # Error handling
            error_latency = time.time() - start_time
            self.evaluator.track_latency("error", error_latency)
            
            return {
                "query": query,
                "documents": [],
                "error": str(e),
                "search_metadata": {
                    "latency_ms": error_latency * 1000,
                    "status": "error"
                }
            }
    
    async def async_search(self, query: str, top_k: int = 5) -> Dict:
        """Async version of search for high-throughput scenarios"""
        start_time = time.time()
        
        try:
            results = await self.async_pipeline.parallel_search_and_rerank(
                query, self.documents)
            
            search_latency = time.time() - start_time
            self.evaluator.track_latency("async_search", search_latency)
            
            return {
                "query": query,
                "documents": [
                    {
                        "id": doc.id,
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "score": float(score)
                    }
                    for doc, score in results[:top_k]
                ],
                "search_metadata": {
                    "latency_ms": search_latency * 1000,
                    "method": "async_parallel"
                }
            }
        except Exception as e:
            return {
                "query": query,
                "documents": [],
                "error": str(e)
            }
    
    def batch_search(self, queries: List[str], **search_kwargs) -> List[Dict]:
        """Process multiple queries efficiently"""
        results = []
        batch_start = time.time()
        
        for query in queries:
            result = self.search(query, **search_kwargs)
            results.append(result)
        
        batch_latency = time.time() - batch_start
        self.evaluator.track_latency("batch_search", batch_latency)
        
        return results
    
    async def async_batch_search(self, queries: List[str], **search_kwargs) -> List[Dict]:
        """Process multiple queries asynchronously"""
        batch_start = time.time()
        
        # Create tasks for all queries
        tasks = [self.async_search(query, **search_kwargs) for query in queries]
        
        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        batch_latency = time.time() - batch_start
        self.evaluator.track_latency("async_batch_search", batch_latency)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append({
                    "query": queries[i],
                    "documents": [],
                    "error": str(result)
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def evaluate_system(self, test_queries: List[Tuple[str, List[str]]]) -> Dict:
        """Evaluate system performance with ground truth"""
        evaluation_results = []
        
        for query, relevant_doc_ids in test_queries:
            # Perform search
            search_result = self.search(query, top_k=10, use_cache=False)
            retrieved_docs = [
                Document(d["id"], d["content"], d["metadata"]) 
                for d in search_result["documents"]
            ]
            
            # Evaluate quality
            metrics = self.evaluator.evaluate_retrieval_quality(
                query, retrieved_docs, relevant_doc_ids)
            evaluation_results.append(metrics)
        
        # Calculate average metrics
        if evaluation_results:
            avg_metrics = {}
            for key in evaluation_results[0].keys():
                values = [r[key] for r in evaluation_results if key in r]
                avg_metrics[key] = np.mean(values) if values else 0.0
            
            return {
                "individual_results": evaluation_results,
                "average_metrics": avg_metrics,
                "total_queries": len(test_queries)
            }
        
        return {"error": "No evaluation results"}
    
    def get_system_stats(self) -> Dict:
        """Return comprehensive system statistics"""
        uptime = time.time() - self.start_time
        
        stats = {
            "system_info": {
                "uptime_seconds": uptime,
                "total_queries": self.total_queries,
                "queries_per_second": self.total_queries / uptime if uptime > 0 else 0,
                "total_documents": len(self.documents)
            },
            "cache_stats": self.cached_system.get_cache_stats(),
            "performance_stats": self.evaluator.get_performance_report()
        }
        
        return stats


def demonstrate_complete_rag_system():
    """Demonstrate the complete RAG system with all features"""
    print("🚀 Production RAG System Demonstration")
    print("=" * 50)
    
    # Create sample documents
    documents = [
        Document("1", "Advanced machine learning algorithms for predictive analytics and data mining", 
                {"category": "ML", "difficulty": "advanced"}),
        Document("2", "Deep learning neural network architectures including CNNs and RNNs", 
                {"category": "DL", "difficulty": "intermediate"}),
        Document("3", "Natural language processing with transformer models and attention mechanisms", 
                {"category": "NLP", "difficulty": "advanced"}),
        Document("4", "Computer vision techniques for image classification and object detection", 
                {"category": "CV", "difficulty": "intermediate"}),
        Document("5", "Reinforcement learning algorithms for autonomous decision making systems", 
                {"category": "RL", "difficulty": "advanced"}),
        Document("6", "Introduction to machine learning concepts and basic algorithms", 
                {"category": "ML", "difficulty": "beginner"}),
        Document("7", "Data preprocessing and feature engineering for machine learning models", 
                {"category": "ML", "difficulty": "intermediate"}),
        Document("8", "Ensemble methods and model optimization techniques in machine learning", 
                {"category": "ML", "difficulty": "advanced"}),
    ]
    
    # Initialize production system
    rag_system = ProductionRAGSystem(documents)
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "neural networks deep learning",
        "natural language processing transformers",
        "computer vision image classification"
    ]
    
    print("\n1. Basic Search Demonstration")
    print("-" * 30)
    for query in test_queries[:2]:
        result = rag_system.search(query, top_k=3)
        print(f"\nQuery: {query}")
        print(f"Latency: {result['search_metadata']['latency_ms']:.2f}ms")
        for i, doc in enumerate(result['documents'], 1):
            print(f"  {i}. [{doc['score']:.3f}] {doc['content'][:60]}...")
    
    print("\n2. Advanced Search with All Features")
    print("-" * 40)
    query = "machine learning algorithms"
    result = rag_system.search(
        query, 
        top_k=3, 
        use_expansion=True, 
        use_reranking=True, 
        use_cache=True
    )
    print(f"Query: {query}")
    print(f"Expanded: {result['expanded_query']}")
    print(f"Results: {len(result['documents'])}")
    print(f"Metadata: {result['search_metadata']}")
    
    print("\n3. Batch Search Performance")
    print("-" * 30)
    batch_results = rag_system.batch_search(test_queries)
    print(f"Processed {len(batch_results)} queries")
    avg_latency = np.mean([r['search_metadata']['latency_ms'] for r in batch_results])
    print(f"Average latency: {avg_latency:.2f}ms")
    
    print("\n4. System Statistics")
    print("-" * 20)
    stats = rag_system.get_system_stats()
    print(f"Total queries: {stats['system_info']['total_queries']}")
    print(f"Cache hit rate: {stats['cache_stats']['query_hit_rate']:.2%}")
    print(f"Average search latency: {np.mean(stats['performance_stats']['latency_stats'].get('search', {}).get('mean', 0)):.2f}ms")
    
    print("\n5. System Evaluation")
    print("-" * 20)
    # Define ground truth for evaluation
    evaluation_queries = [
        ("machine learning algorithms", ["1", "6", "7", "8"]),  # ML-related docs
        ("neural networks", ["2", "3"]),  # DL and NLP docs
        ("computer vision", ["4"]),  # CV doc
    ]
    
    eval_results = rag_system.evaluate_system(evaluation_queries)
    print("Evaluation Results:")
    for metric, value in eval_results['average_metrics'].items():
        print(f"  {metric}: {value:.3f}")


async def demonstrate_async_rag():
    """Demonstrate async RAG capabilities"""
    print("\n🔄 Async RAG Demonstration")
    print("=" * 30)
    
    documents = [
        Document("1", "Async programming with Python asyncio", {}),
        Document("2", "Concurrent processing in distributed systems", {}),
        Document("3", "High-performance computing with parallel algorithms", {}),
    ]
    
    rag_system = ProductionRAGSystem(documents)
    
    # Test async search
    queries = ["async programming", "concurrent processing", "parallel computing"]
    
    start_time = time.time()
    results = await rag_system.async_batch_search(queries)
    end_time = time.time()
    
    print(f"Processed {len(queries)} queries asynchronously")
    print(f"Total time: {(end_time - start_time) * 1000:.2f}ms")
    
    for result in results:
        print(f"Query: {result['query']} - Results: {len(result['documents'])}")


def main():
    """Run complete RAG system demonstrations"""
    print("🎯 Day 52: Advanced RAG - Complete Solutions")
    print("=" * 60)
    
    # Run synchronous demonstrations
    demonstrate_complete_rag_system()
    
    # Run async demonstration
    print("\n" + "=" * 60)
    asyncio.run(demonstrate_async_rag())
    
    print("\n✅ All demonstrations completed successfully!")
    print("\nKey Features Demonstrated:")
    print("• Hybrid search with dense + sparse retrieval")
    print("• Reciprocal Rank Fusion (RRF)")
    print("• Cross-encoder re-ranking")
    print("• Query expansion with synonyms")
    print("• Multi-level caching system")
    print("• Asynchronous processing")
    print("• Comprehensive evaluation metrics")
    print("• Production-ready system integration")


if __name__ == "__main__":
    main()
