"""
Day 52: Advanced RAG - Retrieval-Augmented Generation Systems
Exercises for hybrid search, re-ranking, and production optimization
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import asyncio
from concurrent.futures import ThreadPoolExecutor


@dataclass
class Document:
    """Document representation for RAG system"""
    id: str
    content: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None


class MockEmbeddingModel:
    """Mock embedding model for exercises"""
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings"""
        if isinstance(texts, str):
            texts = [texts]
        # Simple mock: use text length and hash for embedding
        embeddings = []
        for text in texts:
            # Create a simple embedding based on text characteristics
            embedding = np.random.rand(384)  # 384-dimensional embedding
            embedding[0] = len(text) / 100  # Length feature
            embedding[1] = hash(text) % 100 / 100  # Hash feature
            embeddings.append(embedding)
        return np.array(embeddings)


class BM25Retriever:
    """Simple BM25 implementation for sparse retrieval"""
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.doc_freqs = {}
        self.idf = {}
        self.doc_len = {}
        self.avgdl = 0
        self._build_index()
    
    def _build_index(self):
        """Build BM25 index"""
        total_len = 0
        for doc in self.documents:
            words = doc.content.lower().split()
            self.doc_len[doc.id] = len(words)
            total_len += len(words)
            
            for word in set(words):
                if word not in self.doc_freqs:
                    self.doc_freqs[word] = 0
                self.doc_freqs[word] += 1
        
        self.avgdl = total_len / len(self.documents)
        
        # Calculate IDF
        N = len(self.documents)
        for word, freq in self.doc_freqs.items():
            self.idf[word] = np.log((N - freq + 0.5) / (freq + 0.5))
    
    def search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
        """Search using BM25 scoring"""
        query_words = query.lower().split()
        scores = defaultdict(float)
        
        k1, b = 1.5, 0.75  # BM25 parameters
        
        for doc in self.documents:
            doc_words = doc.content.lower().split()
            word_counts = defaultdict(int)
            for word in doc_words:
                word_counts[word] += 1
            
            score = 0
            for word in query_words:
                if word in word_counts:
                    tf = word_counts[word]
                    idf = self.idf.get(word, 0)
                    
                    # BM25 formula
                    numerator = tf * (k1 + 1)
                    denominator = tf + k1 * (1 - b + b * self.doc_len[doc.id] / self.avgdl)
                    score += idf * (numerator / denominator)
            
            scores[doc.id] = score
        
        # Sort by score and return top k
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [(next(d for d in self.documents if d.id == doc_id), score) 
                for doc_id, score in sorted_docs if score > 0]


# Exercise 1: Implement Hybrid Search with RRF
def exercise_1_hybrid_search():
    """
    Exercise 1: Implement hybrid search combining dense and sparse retrieval
    
    TODO: Complete the HybridRetriever class
    """
    print("=== Exercise 1: Hybrid Search with Reciprocal Rank Fusion ===")
    
    # Sample documents
    documents = [
        Document("1", "Machine learning algorithms for data analysis", {"category": "ML"}),
        Document("2", "Deep learning neural networks and applications", {"category": "DL"}),
        Document("3", "Natural language processing with transformers", {"category": "NLP"}),
        Document("4", "Computer vision and image recognition systems", {"category": "CV"}),
        Document("5", "Reinforcement learning for autonomous systems", {"category": "RL"}),
    ]
    
    class HybridRetriever:
        def __init__(self, documents: List[Document]):
            self.documents = documents
            self.embedding_model = MockEmbeddingModel()
            self.bm25_retriever = BM25Retriever(documents)
            
            # Pre-compute embeddings
            contents = [doc.content for doc in documents]
            embeddings = self.embedding_model.encode(contents)
            for doc, embedding in zip(documents, embeddings):
                doc.embedding = embedding
        
        def dense_search(self, query: str, k: int = 10) -> List[Tuple[Document, float]]:
            """Dense (vector) search using cosine similarity"""
            query_embedding = self.embedding_model.encode([query])[0]
            
            similarities = []
            for doc in self.documents:
                if doc.embedding is not None:
                    # Calculate cosine similarity
                    dot_product = np.dot(query_embedding, doc.embedding)
                    norm_query = np.linalg.norm(query_embedding)
                    norm_doc = np.linalg.norm(doc.embedding)
                    similarity = dot_product / (norm_query * norm_doc)
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
    
    # Test the hybrid retriever
    retriever = HybridRetriever(documents)
    query = "machine learning algorithms"
    
    print(f"Query: {query}")
    print("\n--- Your implementation should return hybrid search results ---")
    # results = retriever.hybrid_search(query, k=3)
    # for doc, score in results:
    #     print(f"Score: {score:.4f} - {doc.content}")


# Exercise 2: Implement Cross-Encoder Re-ranking
def exercise_2_reranking():
    """
    Exercise 2: Implement cross-encoder re-ranking for improved relevance
    
    TODO: Complete the CrossEncoderReranker class
    """
    print("\n=== Exercise 2: Cross-Encoder Re-ranking ===")
    
    class CrossEncoderReranker:
        def __init__(self):
            # Mock cross-encoder model
            pass
        
        def score_pairs(self, query: str, documents: List[Document]) -> List[float]:
            """Mock cross-encoder scoring with realistic factors"""
            scores = []
            query_words = set(query.lower().split())
            
            for doc in documents:
                doc_words = set(doc.content.lower().split())
                
                # Calculate word overlap
                overlap = len(query_words.intersection(doc_words))
                overlap_score = overlap / len(query_words) if query_words else 0
                
                # Length penalty (prefer moderate length)
                doc_len = len(doc.content.split())
                optimal_length = 50
                length_penalty = 1.0 - abs(doc_len - optimal_length) / (optimal_length * 2)
                length_penalty = max(0.1, length_penalty)
                
                # Add some deterministic "randomness" based on content
                content_hash = hash(query + doc.content) % 1000
                randomness = (content_hash / 1000) * 0.2
                
                # Combine factors
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
    
    # Test data
    documents = [
        Document("1", "Python programming language basics", {}),
        Document("2", "Advanced Python data structures", {}),
        Document("3", "Java programming fundamentals", {}),
        Document("4", "Python machine learning libraries", {}),
        Document("5", "Web development with Python Flask", {}),
    ]
    
    reranker = CrossEncoderReranker()
    query = "Python programming"
    
    print(f"Query: {query}")
    print("\n--- Your implementation should return re-ranked results ---")
    # results = reranker.rerank(query, documents, top_k=3)
    # for doc, score in results:
    #     print(f"Score: {score:.4f} - {doc.content}")


# Exercise 3: Implement Query Expansion
def exercise_3_query_expansion():
    """
    Exercise 3: Implement query expansion for improved retrieval coverage
    
    TODO: Complete the QueryExpander class
    """
    print("\n=== Exercise 3: Query Expansion ===")
    
    class QueryExpander:
        def __init__(self):
            # Simple synonym dictionary for demonstration
            self.synonyms = {
                "machine": ["algorithm", "automated", "computer"],
                "learning": ["training", "education", "study"],
                "data": ["information", "dataset", "records"],
                "analysis": ["examination", "evaluation", "study"],
                "model": ["framework", "system", "structure"]
            }
        
        def expand_with_synonyms(self, query: str) -> str:
            """Expand query with synonyms"""
            words = query.lower().split()
            expanded_terms = []
            
            for word in words:
                expanded_terms.append(word)
                if word in self.synonyms:
                    # Add up to 2 synonyms for each word
                    synonyms = self.synonyms[word][:2]
                    expanded_terms.extend(synonyms)
            
            return " ".join(expanded_terms)
        
        def generate_alternative_queries(self, query: str, num_alternatives: int = 2) -> List[str]:
            """Generate alternative query formulations"""
            words = query.split()
            alternatives = []
            
            # Alternative 1: Add question words
            if num_alternatives >= 1:
                question_starters = ["what is", "how does", "explain", "describe"]
                starter = question_starters[hash(query) % len(question_starters)]
                alternatives.append(f"{starter} {query}")
            
            # Alternative 2: Reorder words (for multi-word queries)
            if len(words) > 1 and num_alternatives >= 2:
                reversed_query = " ".join(reversed(words))
                alternatives.append(reversed_query)
            
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
                try:
                    results = retriever.hybrid_search(query_variant, k=k*2)
                    for doc, score in results:
                        if doc.id in all_results:
                            # Combine scores (average)
                            all_results[doc.id] = (all_results[doc.id][0], 
                                                 (all_results[doc.id][1] + score) / 2)
                        else:
                            all_results[doc.id] = (doc, score)
                except:
                    # Skip if retriever doesn't have hybrid_search method
                    continue
            
            # Sort by combined scores
            sorted_results = sorted(all_results.values(), key=lambda x: x[1], reverse=True)
            return sorted_results[:k]
    
    # Test the query expander
    expander = QueryExpander()
    query = "machine learning data analysis"
    
    print(f"Original query: {query}")
    print("\n--- Your implementation should show query expansion ---")
    # expanded = expander.expand_with_synonyms(query)
    # alternatives = expander.generate_alternative_queries(query)
    # print(f"Expanded: {expanded}")
    # print(f"Alternatives: {alternatives}")


# Exercise 4: Implement Caching System
def exercise_4_caching():
    """
    Exercise 4: Implement multi-level caching for production RAG
    
    TODO: Complete the CachedRAGSystem class
    """
    print("\n=== Exercise 4: Multi-level Caching System ===")
    
    class LRUCache:
        """Simple LRU Cache implementation"""
        def __init__(self, maxsize: int = 100):
            self.maxsize = maxsize
            self.cache = {}
            self.access_order = []
        
        def get(self, key):
            if key in self.cache:
                self.access_order.remove(key)
                self.access_order.append(key)
                return self.cache[key]
            return None
        
        def set(self, key, value):
            if key in self.cache:
                self.access_order.remove(key)
            elif len(self.cache) >= self.maxsize:
                oldest = self.access_order.pop(0)
                del self.cache[oldest]
            
            self.cache[key] = value
            self.access_order.append(key)
    
    class CachedRAGSystem:
        def __init__(self):
            self.query_cache = LRUCache(maxsize=50)
            self.embedding_cache = LRUCache(maxsize=200)
            self.embedding_model = MockEmbeddingModel()
            self.search_count = 0
            self.cache_hits = 0
        
        def get_embedding(self, text: str) -> np.ndarray:
            """Get embedding with caching"""
            text_hash = str(hash(text))
            
            # Check cache first
            cached_embedding = self.embedding_cache.get(text_hash)
            if cached_embedding is not None:
                self.cache_hits += 1
                return cached_embedding
            
            # Compute and cache embedding
            embedding = self.embedding_model.encode([text])[0]
            self.embedding_cache.set(text_hash, embedding)
            
            return embedding
        
        def cached_search(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
            """Perform cached search with performance tracking"""
            self.search_count += 1
            query_hash = str(hash(query))
            
            # Check query cache
            cached_result = self.query_cache.get(query_hash)
            if cached_result is not None:
                self.cache_hits += 1
                return cached_result
            
            # Perform simple search (mock implementation)
            query_embedding = self.get_embedding(query)
            results = []
            
            for doc in documents:
                doc_embedding = self.get_embedding(doc.content)
                # Simple cosine similarity
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                results.append((doc, float(similarity)))
            
            # Sort and cache
            results.sort(key=lambda x: x[1], reverse=True)
            self.query_cache.set(query_hash, results)
            
            return results
        
        def get_cache_stats(self) -> Dict:
            """Return cache performance statistics"""
            hit_rate = self.cache_hits / self.search_count if self.search_count > 0 else 0
            
            return {
                "total_searches": self.search_count,
                "cache_hits": self.cache_hits,
                "hit_rate": hit_rate,
                "query_cache_size": self.query_cache.size(),
                "embedding_cache_size": self.embedding_cache.size()
            }
    
    # Test the caching system
    documents = [
        Document("1", "Caching strategies for web applications", {}),
        Document("2", "Database query optimization techniques", {}),
        Document("3", "Memory management in distributed systems", {}),
    ]
    
    cached_system = CachedRAGSystem()
    
    print("Testing caching system...")
    print("\n--- Your implementation should show cache performance ---")
    # Test with repeated queries to see caching benefits
    # queries = ["caching strategies", "database optimization", "caching strategies"]
    # for query in queries:
    #     results = cached_system.cached_search(query, documents)
    # stats = cached_system.get_cache_stats()
    # print(f"Cache statistics: {stats}")


# Exercise 5: Implement Async RAG Pipeline
def exercise_5_async_rag():
    """
    Exercise 5: Implement asynchronous RAG pipeline for better performance
    
    TODO: Complete the AsyncRAGPipeline class
    """
    print("\n=== Exercise 5: Asynchronous RAG Pipeline ===")
    
    class AsyncRAGPipeline:
        def __init__(self):
            self.embedding_model = MockEmbeddingModel()
        
        async def async_dense_search(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
            """Async dense search simulation"""
            # Simulate async operation
            await asyncio.sleep(0.1)
            
            # Perform dense search
            query_embedding = self.embedding_model.encode([query])[0]
            results = []
            
            for doc in documents:
                doc_embedding = self.embedding_model.encode([doc.content])[0]
                similarity = np.dot(query_embedding, doc_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
                )
                results.append((doc, float(similarity)))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:10]
        
        async def async_sparse_search(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
            """Async sparse search simulation"""
            # Simulate async operation
            await asyncio.sleep(0.08)
            
            # Perform BM25 search
            bm25_retriever = BM25Retriever(documents)
            return bm25_retriever.search(query, k=10)
        
        async def async_rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
            """Async re-ranking simulation"""
            # Simulate async operation
            await asyncio.sleep(0.05)
            
            # Simple re-ranking based on query-document overlap
            query_words = set(query.lower().split())
            results = []
            
            for doc in documents:
                doc_words = set(doc.content.lower().split())
                overlap = len(query_words.intersection(doc_words))
                score = overlap / len(query_words) if query_words else 0
                results.append((doc, score))
            
            results.sort(key=lambda x: x[1], reverse=True)
            return results
        
        async def parallel_search_and_rerank(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
            """Run search methods in parallel and combine results"""
            # Run dense and sparse search concurrently
            dense_task = self.async_dense_search(query, documents)
            sparse_task = self.async_sparse_search(query, documents)
            
            dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
            
            # Simple combination (could use RRF here)
            all_docs = {}
            for doc, score in dense_results:
                all_docs[doc.id] = (doc, score)
            for doc, score in sparse_results:
                if doc.id in all_docs:
                    # Average the scores
                    existing_score = all_docs[doc.id][1]
                    all_docs[doc.id] = (doc, (existing_score + score) / 2)
                else:
                    all_docs[doc.id] = (doc, score)
            
            # Get top candidates for re-ranking
            combined_results = list(all_docs.values())
            combined_results.sort(key=lambda x: x[1], reverse=True)
            top_candidates = [doc for doc, _ in combined_results[:10]]
            
            # Apply async re-ranking
            final_results = await self.async_rerank(query, top_candidates)
            
            return final_results[:5]
    
    # Test async pipeline
    async def test_async_pipeline():
        documents = [
            Document("1", "Asynchronous programming in Python", {}),
            Document("2", "Concurrent processing with asyncio", {}),
            Document("3", "Parallel computing frameworks", {}),
        ]
        
        pipeline = AsyncRAGPipeline()
        query = "async programming"
        
        print(f"Query: {query}")
        print("\n--- Your implementation should show async processing ---")
        
        start_time = time.time()
        results = await pipeline.parallel_search_and_rerank(query, documents)
        end_time = time.time()
        
        print(f"Async processing completed in {end_time - start_time:.2f} seconds")
        for doc, score in results:
            print(f"Score: {score:.4f} - {doc.content}")
    
    # Run async test
    # asyncio.run(test_async_pipeline())
    print("Async pipeline test setup complete (uncomment to run)")


# Exercise 6: Implement RAG Metrics and Evaluation
def exercise_6_rag_metrics():
    """
    Exercise 6: Implement metrics for evaluating RAG system performance
    
    TODO: Complete the RAGEvaluator class
    """
    print("\n=== Exercise 6: RAG Metrics and Evaluation ===")
    
    class RAGEvaluator:
        def __init__(self):
            self.metrics_history = []
        
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
        
        def evaluate_retrieval_quality(self, query: str, retrieved_docs: List[Document], 
                                     relevant_doc_ids: List[str]) -> Dict[str, float]:
            """Comprehensive retrieval evaluation"""
            metrics = {}
            
            # Calculate metrics for different k values
            for k in [1, 3, 5]:
                if k <= len(retrieved_docs):
                    metrics[f"precision_at_{k}"] = self.calculate_precision_at_k(
                        retrieved_docs, relevant_doc_ids, k)
                    metrics[f"recall_at_{k}"] = self.calculate_recall_at_k(
                        retrieved_docs, relevant_doc_ids, k)
            
            # Calculate MRR for this single query
            mrr = self.calculate_mrr([(query, retrieved_docs, relevant_doc_ids)])
            metrics["mrr"] = mrr
            
            return metrics
        
        def track_latency(self, operation: str, latency: float):
            """Track operation latencies"""
            self.metrics_history.append({
                "operation": operation,
                "latency": latency,
                "timestamp": time.time()
            })
        
        def get_performance_report(self) -> Dict:
            """Generate comprehensive performance report"""
            if not self.metrics_history:
                return {"total_operations": 0}
            
            # Group by operation type
            operations = defaultdict(list)
            for metric in self.metrics_history:
                operations[metric["operation"]].append(metric["latency"])
            
            report = {
                "total_operations": len(self.metrics_history),
                "operation_stats": {}
            }
            
            for op, latencies in operations.items():
                report["operation_stats"][op] = {
                    "count": len(latencies),
                    "mean_latency": sum(latencies) / len(latencies),
                    "min_latency": min(latencies),
                    "max_latency": max(latencies)
                }
            
            return report
    
    # Test evaluation system
    evaluator = RAGEvaluator()
    
    # Sample data for testing
    retrieved_docs = [
        Document("1", "Python programming", {}),
        Document("2", "Java development", {}),
        Document("3", "Python data science", {}),
    ]
    relevant_doc_ids = ["1", "3"]  # Documents 1 and 3 are relevant
    
    print("Testing RAG evaluation metrics...")
    print("\n--- Your implementation should show evaluation metrics ---")
    # metrics = evaluator.evaluate_retrieval_quality("Python programming", retrieved_docs, relevant_doc_ids)
    # print(f"Evaluation metrics: {metrics}")


# Exercise 7: Production RAG System Integration
def exercise_7_production_rag():
    """
    Exercise 7: Integrate all components into a production-ready RAG system
    
    TODO: Complete the ProductionRAGSystem class
    """
    print("\n=== Exercise 7: Production RAG System ===")
    
    class ProductionRAGSystem:
        def __init__(self, documents: List[Document]):
            self.documents = documents
            
            # Initialize all components
            self.hybrid_retriever = HybridRetriever(documents)
            self.reranker = CrossEncoderReranker()
            self.query_expander = QueryExpander()
            self.cached_system = CachedRAGSystem()
            self.evaluator = RAGEvaluator()
            
            # System statistics
            self.total_queries = 0
            self.start_time = time.time()
        
        def search(self, query: str, top_k: int = 5, use_expansion: bool = True, 
                  use_reranking: bool = True) -> Dict:
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
                
                # Step 2: Hybrid search
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
                        "used_reranking": use_reranking
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
        
        def batch_search(self, queries: List[str]) -> List[Dict]:
            """Process multiple queries efficiently"""
            results = []
            batch_start = time.time()
            
            for query in queries:
                result = self.search(query)
                results.append(result)
            
            batch_latency = time.time() - batch_start
            self.evaluator.track_latency("batch_search", batch_latency)
            
            return results
        
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
    
    # Test production system
    documents = [
        Document("1", "Advanced machine learning algorithms and techniques", {"category": "ML"}),
        Document("2", "Deep learning neural network architectures", {"category": "DL"}),
        Document("3", "Natural language processing with transformer models", {"category": "NLP"}),
        Document("4", "Computer vision and image classification systems", {"category": "CV"}),
        Document("5", "Reinforcement learning for decision making", {"category": "RL"}),
    ]
    
    # production_system = ProductionRAGSystem(documents)
    
    print("Production RAG system initialized")
    print("\n--- Your implementation should provide full RAG functionality ---")
    
    # Test queries
    test_queries = [
        "machine learning algorithms",
        "neural networks deep learning",
        "natural language processing"
    ]
    
    # for query in test_queries:
    #     result = production_system.search(query, top_k=3)
    #     print(f"\nQuery: {query}")
    #     print(f"Results: {len(result.get('documents', []))} documents found")
    
    # stats = production_system.get_system_stats()
    # print(f"\nSystem Statistics: {stats}")


def main():
    """Run all RAG exercises"""
    print("🚀 Day 52: Advanced RAG - Retrieval-Augmented Generation Systems")
    print("=" * 70)
    
    exercises = [
        exercise_1_hybrid_search,
        exercise_2_reranking,
        exercise_3_query_expansion,
        exercise_4_caching,
        exercise_5_async_rag,
        exercise_6_rag_metrics,
        exercise_7_production_rag
    ]
    
    for i, exercise in enumerate(exercises, 1):
        print(f"\n📋 Starting Exercise {i}")
        try:
            exercise()
            print(f"✅ Exercise {i} setup complete")
        except Exception as e:
            print(f"❌ Exercise {i} error: {e}")
        
        if i < len(exercises):
            input("\nPress Enter to continue to the next exercise...")
    
    print("\n🎉 All exercises completed!")
    print("\nNext steps:")
    print("1. Implement the TODO sections in each exercise")
    print("2. Test your implementations with the provided data")
    print("3. Experiment with different parameters and configurations")
    print("4. Review the solution file for complete implementations")


if __name__ == "__main__":
    main()
