"""
Day 52: Advanced RAG - Comprehensive Test Suite
Tests for hybrid search, re-ranking, caching, and production features
"""

import pytest
import numpy as np
import asyncio
import time
from unittest.mock import Mock, patch
from typing import List, Dict, Tuple

# Import classes from solution (in real scenario, these would be from the main module)
import sys
import os
sys.path.append(os.path.dirname(__file__))

from solution import (
    Document, MockEmbeddingModel, BM25Retriever, HybridRetriever,
    CrossEncoderReranker, QueryExpander, LRUCache, CachedRAGSystem,
    AsyncRAGPipeline, RAGEvaluator, ProductionRAGSystem
)


class TestDocument:
    """Test Document dataclass"""
    
    def test_document_creation(self):
        doc = Document("1", "Test content", {"category": "test"})
        assert doc.id == "1"
        assert doc.content == "Test content"
        assert doc.metadata == {"category": "test"}
        assert doc.embedding is None
    
    def test_document_with_embedding(self):
        embedding = np.array([0.1, 0.2, 0.3])
        doc = Document("1", "Test", {}, embedding)
        assert np.array_equal(doc.embedding, embedding)


class TestMockEmbeddingModel:
    """Test MockEmbeddingModel functionality"""
    
    def test_single_text_encoding(self):
        model = MockEmbeddingModel()
        embedding = model.encode("test text")
        assert embedding.shape == (1, 384)
        assert isinstance(embedding, np.ndarray)
    
    def test_multiple_text_encoding(self):
        model = MockEmbeddingModel()
        texts = ["text1", "text2", "text3"]
        embeddings = model.encode(texts)
        assert embeddings.shape == (3, 384)
    
    def test_deterministic_encoding(self):
        model = MockEmbeddingModel()
        text = "consistent text"
        embedding1 = model.encode(text)
        embedding2 = model.encode(text)
        np.testing.assert_array_equal(embedding1, embedding2)
    
    def test_embedding_features(self):
        model = MockEmbeddingModel()
        text = "machine learning algorithm"
        embedding = model.encode(text)[0]
        
        # Check that length feature is set
        assert embedding[0] > 0  # Length feature
        assert embedding[1] >= 0  # Hash feature
        assert embedding[2] >= 0  # Diversity feature
        assert embedding[3] > 0   # ML words feature (should be > 0 for ML text)


class TestBM25Retriever:
    """Test BM25 retrieval functionality"""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            Document("1", "machine learning algorithms", {"category": "ML"}),
            Document("2", "deep learning neural networks", {"category": "DL"}),
            Document("3", "natural language processing", {"category": "NLP"}),
            Document("4", "computer vision systems", {"category": "CV"}),
        ]
    
    def test_bm25_initialization(self, sample_documents):
        retriever = BM25Retriever(sample_documents)
        assert len(retriever.documents) == 4
        assert retriever.avgdl > 0
        assert len(retriever.idf) > 0
    
    def test_bm25_search(self, sample_documents):
        retriever = BM25Retriever(sample_documents)
        results = retriever.search("machine learning", k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        
        # Check that results are sorted by score
        if len(results) > 1:
            assert results[0][1] >= results[1][1]
    
    def test_bm25_no_results(self, sample_documents):
        retriever = BM25Retriever(sample_documents)
        results = retriever.search("nonexistent query", k=5)
        assert len(results) == 0
    
    def test_bm25_parameters(self, sample_documents):
        retriever = BM25Retriever(sample_documents, k1=2.0, b=0.5)
        assert retriever.k1 == 2.0
        assert retriever.b == 0.5


class TestHybridRetriever:
    """Test hybrid search functionality"""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            Document("1", "machine learning algorithms for data analysis", {}),
            Document("2", "deep learning neural network architectures", {}),
            Document("3", "natural language processing with transformers", {}),
        ]
    
    @pytest.fixture
    def hybrid_retriever(self, sample_documents):
        return HybridRetriever(sample_documents)
    
    def test_hybrid_initialization(self, hybrid_retriever):
        assert len(hybrid_retriever.documents) == 3
        assert hybrid_retriever.embedding_model is not None
        assert hybrid_retriever.bm25_retriever is not None
        
        # Check that embeddings are computed
        for doc in hybrid_retriever.documents:
            assert doc.embedding is not None
            assert doc.embedding.shape == (384,)
    
    def test_dense_search(self, hybrid_retriever):
        results = hybrid_retriever.dense_search("machine learning", k=2)
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)
    
    def test_sparse_search(self, hybrid_retriever):
        results = hybrid_retriever.sparse_search("machine learning", k=2)
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc, _ in results)
    
    def test_cosine_similarity(self, hybrid_retriever):
        a = np.array([1, 0, 0])
        b = np.array([1, 0, 0])
        c = np.array([0, 1, 0])
        
        assert hybrid_retriever.cosine_similarity(a, b) == pytest.approx(1.0)
        assert hybrid_retriever.cosine_similarity(a, c) == pytest.approx(0.0)
    
    def test_reciprocal_rank_fusion(self, hybrid_retriever, sample_documents):
        dense_results = [(sample_documents[0], 0.9), (sample_documents[1], 0.8)]
        sparse_results = [(sample_documents[1], 0.7), (sample_documents[2], 0.6)]
        
        fused = hybrid_retriever.reciprocal_rank_fusion(dense_results, sparse_results)
        
        assert len(fused) == 3  # All unique documents
        assert all(isinstance(doc, Document) for doc, _ in fused)
        assert all(isinstance(score, float) for _, score in fused)
        
        # Check that scores are sorted
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)
    
    def test_hybrid_search(self, hybrid_retriever):
        results = hybrid_retriever.hybrid_search("machine learning", k=2)
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc, _ in results)


class TestCrossEncoderReranker:
    """Test cross-encoder re-ranking functionality"""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            Document("1", "Python programming language", {}),
            Document("2", "Java development framework", {}),
            Document("3", "Python data science libraries", {}),
        ]
    
    def test_reranker_initialization(self):
        reranker = CrossEncoderReranker()
        assert reranker is not None
    
    def test_score_pairs(self, sample_documents):
        reranker = CrossEncoderReranker()
        scores = reranker.score_pairs("Python programming", sample_documents)
        
        assert len(scores) == 3
        assert all(isinstance(score, float) for score in scores)
        assert all(0 <= score <= 1 for score in scores)
    
    def test_rerank(self, sample_documents):
        reranker = CrossEncoderReranker()
        results = reranker.rerank("Python programming", sample_documents, top_k=2)
        
        assert len(results) <= 2
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)
        
        # Check that results are sorted by score
        if len(results) > 1:
            assert results[0][1] >= results[1][1]
    
    def test_overlap_score_calculation(self):
        reranker = CrossEncoderReranker()
        
        # Perfect overlap
        score1 = reranker._calculate_overlap_score("python", "python programming")
        assert score1 == 1.0
        
        # Partial overlap
        score2 = reranker._calculate_overlap_score("python java", "python programming")
        assert score2 == 0.5
        
        # No overlap
        score3 = reranker._calculate_overlap_score("python", "java programming")
        assert score3 == 0.0
    
    def test_length_penalty(self):
        reranker = CrossEncoderReranker()
        
        # Test different document lengths
        short_doc = "short"
        optimal_doc = " ".join(["word"] * 50)  # 50 words
        long_doc = " ".join(["word"] * 200)   # 200 words
        
        penalty_short = reranker._calculate_length_penalty(short_doc)
        penalty_optimal = reranker._calculate_length_penalty(optimal_doc)
        penalty_long = reranker._calculate_length_penalty(long_doc)
        
        assert penalty_optimal >= penalty_short
        assert penalty_optimal >= penalty_long
        assert all(p >= 0.1 for p in [penalty_short, penalty_optimal, penalty_long])


class TestQueryExpander:
    """Test query expansion functionality"""
    
    def test_expander_initialization(self):
        expander = QueryExpander()
        assert len(expander.synonyms) > 0
        assert "machine" in expander.synonyms
    
    def test_expand_with_synonyms(self):
        expander = QueryExpander()
        expanded = expander.expand_with_synonyms("machine learning")
        
        assert "machine" in expanded
        assert "learning" in expanded
        assert len(expanded.split()) > 2  # Should have added synonyms
    
    def test_generate_alternative_queries(self):
        expander = QueryExpander()
        alternatives = expander.generate_alternative_queries("machine learning", 2)
        
        assert len(alternatives) == 2
        assert all(isinstance(alt, str) for alt in alternatives)
        assert all("machine" in alt or "learning" in alt for alt in alternatives)
    
    def test_multi_query_search(self):
        # Mock retriever for testing
        mock_retriever = Mock()
        mock_retriever.hybrid_search.return_value = [
            (Document("1", "test", {}), 0.9)
        ]
        
        expander = QueryExpander()
        results = expander.multi_query_search("test query", mock_retriever, k=5)
        
        assert len(results) >= 0
        assert mock_retriever.hybrid_search.called


class TestLRUCache:
    """Test LRU Cache implementation"""
    
    def test_cache_initialization(self):
        cache = LRUCache(maxsize=3)
        assert cache.maxsize == 3
        assert cache.size() == 0
    
    def test_cache_set_get(self):
        cache = LRUCache(maxsize=3)
        cache.set("key1", "value1")
        
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
    
    def test_cache_eviction(self):
        cache = LRUCache(maxsize=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
    
    def test_cache_lru_order(self):
        cache = LRUCache(maxsize=2)
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        # Access key1 to make it most recent
        cache.get("key1")
        
        # Add key3, should evict key2 (least recent)
        cache.set("key3", "value3")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"


class TestCachedRAGSystem:
    """Test caching system for RAG"""
    
    def test_cached_system_initialization(self):
        system = CachedRAGSystem()
        assert system.query_cache is not None
        assert system.embedding_cache is not None
        assert system.search_count == 0
        assert system.cache_hits == 0
    
    def test_get_embedding_caching(self):
        system = CachedRAGSystem()
        
        # First call should compute embedding
        embedding1 = system.get_embedding("test text")
        assert system.embedding_computations == 1
        assert system.embedding_cache_hits == 0
        
        # Second call should use cache
        embedding2 = system.get_embedding("test text")
        assert system.embedding_computations == 1
        assert system.embedding_cache_hits == 1
        
        np.testing.assert_array_equal(embedding1, embedding2)
    
    def test_cached_search(self):
        system = CachedRAGSystem()
        documents = [Document("1", "test", {})]
        
        # Mock search function
        def mock_search(query, **kwargs):
            return [(documents[0], 0.9)]
        
        # First search
        results1 = system.cached_search("test query", documents, mock_search)
        assert system.search_count == 1
        assert system.cache_hits == 0
        
        # Second search (should hit cache)
        results2 = system.cached_search("test query", documents, mock_search)
        assert system.search_count == 2
        assert system.cache_hits == 1
        
        assert results1 == results2
    
    def test_cache_stats(self):
        system = CachedRAGSystem()
        system.search_count = 10
        system.cache_hits = 3
        system.embedding_cache_hits = 5
        system.embedding_computations = 2
        
        stats = system.get_cache_stats()
        
        assert stats["total_searches"] == 10
        assert stats["query_cache_hits"] == 3
        assert stats["query_hit_rate"] == 0.3
        assert stats["embedding_cache_hits"] == 5
        assert stats["embedding_computations"] == 2


class TestAsyncRAGPipeline:
    """Test asynchronous RAG pipeline"""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            Document("1", "async programming", {}),
            Document("2", "concurrent processing", {}),
        ]
    
    @pytest.mark.asyncio
    async def test_async_dense_search(self, sample_documents):
        pipeline = AsyncRAGPipeline()
        results = await pipeline.async_dense_search("async", sample_documents)
        
        assert isinstance(results, list)
        assert all(isinstance(doc, Document) for doc, _ in results)
    
    @pytest.mark.asyncio
    async def test_async_sparse_search(self, sample_documents):
        pipeline = AsyncRAGPipeline()
        results = await pipeline.async_sparse_search("async", sample_documents)
        
        assert isinstance(results, list)
    
    @pytest.mark.asyncio
    async def test_async_rerank(self, sample_documents):
        pipeline = AsyncRAGPipeline()
        results = await pipeline.async_rerank("async", sample_documents)
        
        assert isinstance(results, list)
        assert all(isinstance(doc, Document) for doc, _ in results)
    
    @pytest.mark.asyncio
    async def test_parallel_search_and_rerank(self, sample_documents):
        pipeline = AsyncRAGPipeline()
        
        start_time = time.time()
        results = await pipeline.parallel_search_and_rerank("async", sample_documents)
        end_time = time.time()
        
        assert isinstance(results, list)
        assert len(results) <= 5
        # Should be faster than sequential execution due to parallelism
        assert end_time - start_time < 1.0  # Should complete quickly


class TestRAGEvaluator:
    """Test RAG evaluation metrics"""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            Document("1", "relevant doc 1", {}),
            Document("2", "irrelevant doc", {}),
            Document("3", "relevant doc 2", {}),
        ]
    
    def test_evaluator_initialization(self):
        evaluator = RAGEvaluator()
        assert evaluator.metrics_history == []
        assert len(evaluator.latency_history) == 0
    
    def test_calculate_mrr(self, sample_documents):
        evaluator = RAGEvaluator()
        
        # Test data: query, retrieved docs, relevant doc IDs
        test_data = [
            ("query1", sample_documents, ["1", "3"]),  # First doc is relevant
            ("query2", sample_documents, ["3"]),       # Third doc is relevant
        ]
        
        mrr = evaluator.calculate_mrr(test_data)
        
        # First query: relevant doc at rank 1 -> RR = 1.0
        # Second query: relevant doc at rank 3 -> RR = 1/3
        # MRR = (1.0 + 1/3) / 2 = 2/3
        expected_mrr = (1.0 + 1/3) / 2
        assert mrr == pytest.approx(expected_mrr)
    
    def test_calculate_recall_at_k(self, sample_documents):
        evaluator = RAGEvaluator()
        relevant_docs = ["1", "3"]
        
        # All relevant docs in top 3
        recall_3 = evaluator.calculate_recall_at_k(sample_documents, relevant_docs, 3)
        assert recall_3 == 1.0
        
        # Only first relevant doc in top 1
        recall_1 = evaluator.calculate_recall_at_k(sample_documents, relevant_docs, 1)
        assert recall_1 == 0.5
    
    def test_calculate_precision_at_k(self, sample_documents):
        evaluator = RAGEvaluator()
        relevant_docs = ["1", "3"]
        
        # 2 relevant out of 3 retrieved
        precision_3 = evaluator.calculate_precision_at_k(sample_documents, relevant_docs, 3)
        assert precision_3 == pytest.approx(2/3)
        
        # 1 relevant out of 1 retrieved
        precision_1 = evaluator.calculate_precision_at_k(sample_documents, relevant_docs, 1)
        assert precision_1 == 1.0
    
    def test_calculate_f1_at_k(self, sample_documents):
        evaluator = RAGEvaluator()
        relevant_docs = ["1", "3"]
        
        f1_3 = evaluator.calculate_f1_at_k(sample_documents, relevant_docs, 3)
        
        # Precision@3 = 2/3, Recall@3 = 1.0
        # F1 = 2 * (2/3 * 1.0) / (2/3 + 1.0) = 2 * (2/3) / (5/3) = 4/5
        expected_f1 = 2 * (2/3 * 1.0) / (2/3 + 1.0)
        assert f1_3 == pytest.approx(expected_f1)
    
    def test_evaluate_retrieval_quality(self, sample_documents):
        evaluator = RAGEvaluator()
        relevant_docs = ["1", "3"]
        
        metrics = evaluator.evaluate_retrieval_quality(
            "test query", sample_documents, relevant_docs)
        
        assert "precision_at_1" in metrics
        assert "recall_at_1" in metrics
        assert "f1_at_1" in metrics
        assert "mrr" in metrics
        
        assert all(isinstance(v, float) for v in metrics.values())
    
    def test_track_latency(self):
        evaluator = RAGEvaluator()
        evaluator.track_latency("search", 0.1)
        evaluator.track_latency("search", 0.2)
        
        assert len(evaluator.latency_history["search"]) == 2
        assert evaluator.latency_history["search"] == [0.1, 0.2]
    
    def test_get_performance_report(self):
        evaluator = RAGEvaluator()
        evaluator.track_latency("search", 0.1)
        evaluator.track_latency("search", 0.2)
        evaluator.track_latency("rerank", 0.05)
        
        report = evaluator.get_performance_report()
        
        assert "total_evaluations" in report
        assert "latency_stats" in report
        assert "search" in report["latency_stats"]
        assert "rerank" in report["latency_stats"]
        
        search_stats = report["latency_stats"]["search"]
        assert "mean" in search_stats
        assert "median" in search_stats
        assert "p95" in search_stats
        assert "count" in search_stats


class TestProductionRAGSystem:
    """Test production RAG system integration"""
    
    @pytest.fixture
    def sample_documents(self):
        return [
            Document("1", "machine learning algorithms", {"category": "ML"}),
            Document("2", "deep learning networks", {"category": "DL"}),
            Document("3", "natural language processing", {"category": "NLP"}),
        ]
    
    @pytest.fixture
    def production_system(self, sample_documents):
        return ProductionRAGSystem(sample_documents)
    
    def test_system_initialization(self, production_system):
        assert production_system.documents is not None
        assert production_system.hybrid_retriever is not None
        assert production_system.reranker is not None
        assert production_system.query_expander is not None
        assert production_system.cached_system is not None
        assert production_system.evaluator is not None
        assert production_system.total_queries == 0
    
    def test_basic_search(self, production_system):
        result = production_system.search("machine learning", top_k=2)
        
        assert "query" in result
        assert "documents" in result
        assert "search_metadata" in result
        
        assert result["query"] == "machine learning"
        assert len(result["documents"]) <= 2
        assert "latency_ms" in result["search_metadata"]
        
        # Check document structure
        for doc in result["documents"]:
            assert "id" in doc
            assert "content" in doc
            assert "metadata" in doc
            assert "score" in doc
    
    def test_search_with_options(self, production_system):
        result = production_system.search(
            "machine learning",
            top_k=1,
            use_expansion=True,
            use_reranking=True,
            use_cache=True
        )
        
        assert result["expanded_query"] is not None
        assert result["search_metadata"]["used_expansion"] is True
        assert result["search_metadata"]["used_reranking"] is True
        assert result["search_metadata"]["used_cache"] is True
    
    def test_search_without_options(self, production_system):
        result = production_system.search(
            "machine learning",
            use_expansion=False,
            use_reranking=False,
            use_cache=False
        )
        
        assert result["expanded_query"] is None
        assert result["search_metadata"]["used_expansion"] is False
        assert result["search_metadata"]["used_reranking"] is False
        assert result["search_metadata"]["used_cache"] is False
    
    def test_batch_search(self, production_system):
        queries = ["machine learning", "deep learning"]
        results = production_system.batch_search(queries, top_k=1)
        
        assert len(results) == 2
        assert all("query" in result for result in results)
        assert all("documents" in result for result in results)
    
    @pytest.mark.asyncio
    async def test_async_search(self, production_system):
        result = await production_system.async_search("machine learning", top_k=2)
        
        assert "query" in result
        assert "documents" in result
        assert "search_metadata" in result
        assert result["search_metadata"]["method"] == "async_parallel"
    
    @pytest.mark.asyncio
    async def test_async_batch_search(self, production_system):
        queries = ["machine learning", "deep learning"]
        results = await production_system.async_batch_search(queries, top_k=1)
        
        assert len(results) == 2
        assert all("query" in result for result in results)
    
    def test_evaluate_system(self, production_system):
        test_queries = [
            ("machine learning", ["1"]),  # Doc 1 is relevant
            ("deep learning", ["2"]),     # Doc 2 is relevant
        ]
        
        eval_results = production_system.evaluate_system(test_queries)
        
        assert "individual_results" in eval_results
        assert "average_metrics" in eval_results
        assert "total_queries" in eval_results
        
        assert len(eval_results["individual_results"]) == 2
        assert eval_results["total_queries"] == 2
    
    def test_get_system_stats(self, production_system):
        # Perform some operations to generate stats
        production_system.search("test query")
        
        stats = production_system.get_system_stats()
        
        assert "system_info" in stats
        assert "cache_stats" in stats
        assert "performance_stats" in stats
        
        system_info = stats["system_info"]
        assert "uptime_seconds" in system_info
        assert "total_queries" in system_info
        assert "queries_per_second" in system_info
        assert "total_documents" in system_info
        
        assert system_info["total_queries"] >= 1
        assert system_info["total_documents"] == 3


class TestIntegration:
    """Integration tests for complete RAG workflow"""
    
    @pytest.fixture
    def complete_system(self):
        documents = [
            Document("1", "Advanced machine learning algorithms for predictive analytics", 
                    {"category": "ML", "difficulty": "advanced"}),
            Document("2", "Introduction to deep learning and neural networks", 
                    {"category": "DL", "difficulty": "beginner"}),
            Document("3", "Natural language processing with transformer architectures", 
                    {"category": "NLP", "difficulty": "intermediate"}),
            Document("4", "Computer vision techniques for image classification", 
                    {"category": "CV", "difficulty": "intermediate"}),
        ]
        return ProductionRAGSystem(documents)
    
    def test_end_to_end_workflow(self, complete_system):
        """Test complete RAG workflow from query to response"""
        
        # Test basic search
        result = complete_system.search("machine learning algorithms", top_k=2)
        assert len(result["documents"]) <= 2
        assert result["search_metadata"]["latency_ms"] > 0
        
        # Test with all features enabled
        advanced_result = complete_system.search(
            "deep learning neural networks",
            top_k=3,
            use_expansion=True,
            use_reranking=True,
            use_cache=True
        )
        assert advanced_result["expanded_query"] is not None
        assert len(advanced_result["documents"]) <= 3
        
        # Test caching (second identical query should be faster)
        start_time = time.time()
        cached_result = complete_system.search(
            "deep learning neural networks",
            top_k=3,
            use_expansion=True,
            use_reranking=True,
            use_cache=True
        )
        cached_time = time.time() - start_time
        
        # Results should be identical
        assert cached_result["query"] == advanced_result["query"]
        
        # Get system statistics
        stats = complete_system.get_system_stats()
        assert stats["system_info"]["total_queries"] >= 3
        assert stats["cache_stats"]["query_hit_rate"] > 0  # Should have cache hits
    
    @pytest.mark.asyncio
    async def test_async_workflow(self, complete_system):
        """Test asynchronous RAG workflow"""
        
        queries = [
            "machine learning algorithms",
            "deep learning networks", 
            "natural language processing"
        ]
        
        # Test async batch processing
        start_time = time.time()
        results = await complete_system.async_batch_search(queries, top_k=2)
        async_time = time.time() - start_time
        
        assert len(results) == 3
        assert all("documents" in result for result in results)
        assert all(len(result["documents"]) <= 2 for result in results)
        
        # Async should be reasonably fast
        assert async_time < 2.0  # Should complete within 2 seconds
    
    def test_evaluation_workflow(self, complete_system):
        """Test evaluation and metrics workflow"""
        
        # Define test queries with ground truth
        test_queries = [
            ("machine learning algorithms", ["1"]),  # ML document
            ("deep learning neural networks", ["2"]), # DL document  
            ("natural language processing", ["3"]),   # NLP document
            ("computer vision", ["4"]),               # CV document
        ]
        
        # Evaluate system
        eval_results = complete_system.evaluate_system(test_queries)
        
        assert "average_metrics" in eval_results
        assert "individual_results" in eval_results
        
        # Check that we have reasonable metrics
        avg_metrics = eval_results["average_metrics"]
        assert "precision_at_1" in avg_metrics
        assert "recall_at_1" in avg_metrics
        assert "mrr" in avg_metrics
        
        # Precision@1 should be reasonable (> 0.5) for this simple test
        assert avg_metrics["precision_at_1"] > 0.5
    
    def test_performance_monitoring(self, complete_system):
        """Test performance monitoring and metrics collection"""
        
        # Perform multiple searches to generate metrics
        queries = ["test query 1", "test query 2", "test query 3"]
        
        for query in queries:
            complete_system.search(query, top_k=2)
        
        # Get performance report
        stats = complete_system.get_system_stats()
        perf_stats = stats["performance_stats"]
        
        assert "latency_stats" in perf_stats
        if "search" in perf_stats["latency_stats"]:
            search_stats = perf_stats["latency_stats"]["search"]
            assert "mean" in search_stats
            assert "count" in search_stats
            assert search_stats["count"] >= 3


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])