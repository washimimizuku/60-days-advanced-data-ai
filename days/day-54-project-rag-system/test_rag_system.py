"""
Day 54: Production RAG System - Comprehensive Test Suite
Tests for all components of the production RAG system
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock
from fastapi.testclient import TestClient
import tempfile
import os
from pathlib import Path

# Import the RAG system components
from solution import (
    ProductionRAGSystem, DocumentProcessor, HybridRetriever,
    GenerationService, EvaluationService, app,
    QueryRequest, QueryOptions, Document, DocumentMetadata
)


class TestDocumentProcessor:
    """Test document processing functionality"""
    
    @pytest.fixture
    def processor(self):
        return DocumentProcessor()
    
    @pytest.fixture
    def sample_document_content(self):
        return """
        # API Documentation
        
        ## Authentication
        Use JWT tokens for authentication.
        
        ## Endpoints
        - GET /users - List all users
        - POST /users - Create a new user
        
        ## Rate Limits
        1000 requests per hour for authenticated users.
        """
    
    def test_processor_initialization(self, processor):
        assert processor.text_splitter is not None
        assert processor.embeddings is not None
        assert isinstance(processor.processed_docs, dict)
    
    def test_create_chunks(self, processor):
        # Create a test document
        metadata = DocumentMetadata(source="test", category="api")
        document = Document(
            id="test_doc",
            title="Test Document",
            content="This is a test document with multiple sentences. It should be split into chunks properly.",
            metadata=metadata
        )
        
        chunks = processor._create_chunks(document)
        
        assert len(chunks) > 0
        assert all(chunk.content for chunk in chunks)
        assert all(chunk.id.startswith("test_doc_chunk_") for chunk in chunks)
    
    def test_infer_category(self, processor):
        # Test different path patterns
        api_path = Path("/docs/api/authentication.md")
        policy_path = Path("/company/policy/security.pdf")
        guide_path = Path("/guides/deployment-guide.txt")
        
        assert processor._infer_category(api_path) == "api_documentation"
        assert processor._infer_category(policy_path) == "policy"
        assert processor._infer_category(guide_path) == "guide"
    
    def test_extract_tags(self, processor):
        content = "This document covers Python API development with database integration and security best practices."
        tags = processor._extract_tags(content)
        
        assert "python" in tags
        assert "api" in tags
        assert "database" in tags
        assert "security" in tags
        assert len(tags) <= 5
    
    def test_load_document_with_temp_file(self, processor, sample_document_content):
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write(sample_document_content)
            temp_path = f.name
        
        try:
            document = processor.load_document(temp_path)
            
            assert document.id is not None
            assert document.title == Path(temp_path).stem
            assert len(document.content) > 0
            assert len(document.chunks) > 0
            assert document.metadata.source == temp_path
        finally:
            os.unlink(temp_path)


class TestHybridRetriever:
    """Test hybrid retrieval functionality"""
    
    @pytest.fixture
    def sample_documents(self):
        docs = []
        for i in range(3):
            metadata = DocumentMetadata(
                source=f"doc_{i}",
                category="test",
                tags=["test", "sample"]
            )
            doc = Document(
                id=f"doc_{i}",
                title=f"Document {i}",
                content=f"This is test document {i} about machine learning and AI technology.",
                metadata=metadata
            )
            # Add mock chunks
            doc.chunks = [
                type('Chunk', (), {
                    'id': f"doc_{i}_chunk_0",
                    'content': doc.content,
                    'start_char': 0,
                    'end_char': len(doc.content),
                    'metadata': {'document_id': doc.id, 'chunk_index': 0}
                })()
            ]
            docs.append(doc)
        return docs
    
    @pytest.fixture
    def retriever(self, sample_documents):
        with patch.object(HybridRetriever, '_build_indexes'):
            retriever = HybridRetriever(sample_documents)
            # Mock the indexes
            retriever.vector_store = Mock()
            retriever.bm25_retriever = Mock()
            return retriever
    
    def test_retriever_initialization(self, sample_documents):
        with patch.object(HybridRetriever, '_build_indexes'):
            retriever = HybridRetriever(sample_documents)
            assert len(retriever.documents) == 3
            assert retriever.embeddings is not None
    
    def test_expand_query(self, retriever):
        query = "api error config"
        expanded = retriever.expand_query(query)
        
        assert "api" in expanded
        assert "error" in expanded
        assert "config" in expanded
        # Should contain additional terms
        assert len(expanded.split()) > len(query.split())
    
    def test_reciprocal_rank_fusion(self, retriever):
        dense_results = [("doc1", 0.9), ("doc2", 0.8)]
        sparse_results = ["doc2", "doc3"]
        
        fused = retriever.reciprocal_rank_fusion(dense_results, sparse_results)
        
        assert len(fused) == 3  # All unique documents
        assert all(isinstance(item, tuple) for item in fused)
        assert all(len(item) == 2 for item in fused)
        
        # Results should be sorted by score
        scores = [score for _, score in fused]
        assert scores == sorted(scores, reverse=True)
    
    def test_dense_search_mock(self, retriever):
        # Mock vector store response
        mock_results = [
            (Mock(page_content="test content 1"), 0.9),
            (Mock(page_content="test content 2"), 0.8)
        ]
        retriever.vector_store.similarity_search_with_score.return_value = mock_results
        
        results = retriever.dense_search("test query", k=2)
        
        assert len(results) == 2
        assert results[0][0] == "test content 1"
        assert results[0][1] == 0.9
    
    def test_sparse_search_mock(self, retriever):
        # Mock BM25 retriever response
        mock_docs = [
            Mock(page_content="test content 1"),
            Mock(page_content="test content 2")
        ]
        retriever.bm25_retriever.get_relevant_documents.return_value = mock_docs
        
        results = retriever.sparse_search("test query", k=2)
        
        assert len(results) == 2
        assert results[0] == "test content 1"
        assert results[1] == "test content 2"


class TestGenerationService:
    """Test response generation functionality"""
    
    @pytest.fixture
    def generator(self):
        return GenerationService()
    
    @pytest.fixture
    def sample_contexts(self):
        return [
            {
                'content': 'API authentication requires JWT tokens.',
                'document_id': 'doc1',
                'document_title': 'API Guide',
                'score': 0.9
            },
            {
                'content': 'Rate limits are 1000 requests per hour.',
                'document_id': 'doc2', 
                'document_title': 'Rate Limiting',
                'score': 0.8
            }
        ]
    
    def test_generator_initialization(self, generator):
        assert generator.llm is not None
        assert generator.max_context_length > 0
        assert len(generator.response_templates) > 0
    
    def test_prepare_context(self, generator, sample_contexts):
        context_text = generator._prepare_context(sample_contexts)
        
        assert "API authentication requires JWT tokens" in context_text
        assert "Rate limits are 1000 requests per hour" in context_text
        assert "[Source 1: API Guide]" in context_text
        assert "[Source 2: Rate Limiting]" in context_text
    
    def test_validate_response(self, generator, sample_contexts):
        # Test normal response
        response = "JWT tokens are required for API authentication."
        validated, alerts = generator._validate_response(response, sample_contexts)
        
        assert validated == response
        assert len(alerts) == 0
        
        # Test short response
        short_response = "Yes."
        validated, alerts = generator._validate_response(short_response, sample_contexts)
        
        assert "Response too short" in alerts
        
        # Test long response
        long_response = "A" * 2500
        validated, alerts = generator._validate_response(long_response, sample_contexts)
        
        assert "Response too long" in alerts
        assert len(validated) <= 2003  # 2000 + "..."
    
    def test_generate_response(self, generator, sample_contexts):
        query = "How do I authenticate with the API?"
        
        response, alerts = generator.generate_response(query, sample_contexts)
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert isinstance(alerts, list)


class TestEvaluationService:
    """Test evaluation functionality"""
    
    @pytest.fixture
    def evaluator(self):
        return EvaluationService()
    
    @pytest.fixture
    def sample_contexts(self):
        return [
            {'content': 'JWT tokens are used for API authentication.'},
            {'content': 'Rate limits apply to all API endpoints.'}
        ]
    
    def test_evaluator_initialization(self, evaluator):
        assert evaluator.evaluation_history == []
        assert 'faithfulness' in evaluator.thresholds
        assert 'relevancy' in evaluator.thresholds
        assert 'precision' in evaluator.thresholds
    
    def test_calculate_faithfulness(self, evaluator):
        answer = "JWT tokens are required for authentication"
        contexts = ["JWT tokens are used for API authentication", "Authentication is required"]
        
        score = evaluator._calculate_faithfulness(answer, contexts)
        
        assert 0 <= score <= 1
        assert score > 0.5  # Should have good overlap
    
    def test_calculate_relevancy(self, evaluator):
        query = "How to authenticate API"
        answer = "Use JWT tokens for API authentication"
        
        score = evaluator._calculate_relevancy(query, answer)
        
        assert 0 <= score <= 1
        assert score > 0.3  # Should have some overlap
    
    def test_calculate_precision(self, evaluator):
        query = "API authentication"
        contexts = [
            "JWT tokens are used for API authentication",  # Relevant
            "The weather is sunny today",                   # Irrelevant
            "API endpoints require authentication"          # Relevant
        ]
        
        score = evaluator._calculate_precision(query, contexts)
        
        assert 0 <= score <= 1
        assert score > 0.5  # 2 out of 3 contexts should be relevant
    
    def test_evaluate_response(self, evaluator, sample_contexts):
        query = "How to authenticate with API?"
        answer = "Use JWT tokens for authentication"
        query_id = "test_query_123"
        
        result = evaluator.evaluate_response(query, answer, sample_contexts, query_id)
        
        assert result.query_id == query_id
        assert 0 <= result.faithfulness <= 1
        assert 0 <= result.relevancy <= 1
        assert 0 <= result.context_precision <= 1
        assert 0 <= result.overall_score <= 1
        assert isinstance(result.alerts, list)
        assert result.timestamp is not None
    
    def test_get_evaluation_summary(self, evaluator):
        # Add some mock evaluation results
        for i in range(5):
            result = type('EvaluationResult', (), {
                'faithfulness': 0.8,
                'relevancy': 0.9,
                'context_precision': 0.7,
                'overall_score': 0.8,
                'alerts': [],
                'timestamp': datetime.now()
            })()
            evaluator.evaluation_history.append(result)
        
        summary = evaluator.get_evaluation_summary(hours=24)
        
        assert 'total_evaluations' in summary
        assert 'average_scores' in summary
        assert summary['total_evaluations'] == 5
        assert 'faithfulness' in summary['average_scores']


class TestProductionRAGSystem:
    """Test the complete RAG system"""
    
    @pytest.fixture
    def rag_system(self):
        system = ProductionRAGSystem()
        # Initialize with sample documents to avoid file system dependencies
        system.documents = system._create_sample_documents()
        system.documents = {doc.id: doc for doc in system.documents}
        
        # Mock the retriever to avoid building real indexes
        with patch.object(HybridRetriever, '_build_indexes'):
            system.retriever = HybridRetriever(list(system.documents.values()))
            system.retriever.vector_store = Mock()
            system.retriever.bm25_retriever = Mock()
        
        return system
    
    def test_system_initialization(self, rag_system):
        assert rag_system.document_processor is not None
        assert rag_system.generator is not None
        assert rag_system.evaluator is not None
        assert len(rag_system.documents) > 0
        assert 'start_time' in rag_system.system_stats
    
    def test_create_sample_documents(self, rag_system):
        docs = rag_system._create_sample_documents()
        
        assert len(docs) >= 3
        assert all(isinstance(doc, Document) for doc in docs)
        assert all(len(doc.chunks) > 0 for doc in docs)
        assert all(doc.metadata.category is not None for doc in docs)
    
    @pytest.mark.asyncio
    async def test_process_query_mock(self, rag_system):
        # Mock the retriever to return predictable results
        mock_contexts = [
            {
                'content': 'JWT tokens are used for API authentication.',
                'score': 0.9,
                'document_id': 'doc1',
                'document_title': 'API Guide'
            }
        ]
        rag_system.retriever.retrieve = Mock(return_value=mock_contexts)
        
        # Create test request
        request = QueryRequest(
            query="How do I authenticate with the API?",
            options=QueryOptions(max_results=3)
        )
        
        response = await rag_system.process_query(request)
        
        assert response.answer is not None
        assert len(response.sources) > 0
        assert response.metadata.query_id is not None
        assert response.metadata.processing_time > 0
        assert 'faithfulness' in response.metadata.evaluation_scores
    
    def test_get_system_health(self, rag_system):
        health = rag_system.get_system_health()
        
        assert 'status' in health
        assert 'uptime_seconds' in health
        assert 'total_queries' in health
        assert 'total_documents' in health
        assert health['status'] in ['healthy', 'warning', 'critical']


class TestFastAPIEndpoints:
    """Test FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture(autouse=True)
    def setup_rag_system(self):
        # Mock the RAG system initialization to avoid file system dependencies
        with patch.object(ProductionRAGSystem, 'initialize_system'):
            yield
    
    def test_health_endpoint(self, client):
        with patch('solution.rag_system.get_system_health') as mock_health:
            mock_health.return_value = {
                'status': 'healthy',
                'uptime_seconds': 100,
                'total_queries': 5,
                'total_documents': 10
            }
            
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'healthy'
    
    def test_query_endpoint(self, client):
        with patch('solution.rag_system.process_query') as mock_query:
            # Mock the response
            mock_response = {
                'answer': 'Test answer',
                'sources': [],
                'metadata': {
                    'query_id': 'test123',
                    'processing_time': 1.0,
                    'model_used': 'gpt-3.5-turbo',
                    'evaluation_scores': {
                        'faithfulness': 0.8,
                        'relevancy': 0.9,
                        'context_precision': 0.7,
                        'overall_score': 0.8
                    },
                    'timestamp': datetime.now().isoformat()
                }
            }
            mock_query.return_value = type('RAGResponse', (), mock_response)()
            
            response = client.post("/query", json={
                "query": "Test question",
                "options": {"max_results": 3}
            })
            
            assert response.status_code == 200
            data = response.json()
            assert 'answer' in data
            assert 'sources' in data
            assert 'metadata' in data
    
    def test_documents_endpoint(self, client):
        with patch('solution.rag_system.documents') as mock_docs:
            # Mock documents
            mock_doc = type('Document', (), {
                'id': 'doc1',
                'title': 'Test Doc',
                'metadata': type('Metadata', (), {
                    'category': 'test',
                    'tags': ['test'],
                    'created_at': datetime.now()
                })(),
                'chunks': [1, 2, 3]  # Mock chunks
            })()
            mock_docs.__iter__ = Mock(return_value=iter(['doc1']))
            mock_docs.__getitem__ = Mock(return_value=mock_doc)
            mock_docs.items = Mock(return_value=[('doc1', mock_doc)])
            
            response = client.get("/documents")
            
            assert response.status_code == 200
            data = response.json()
            assert 'total_documents' in data
            assert 'documents' in data
    
    def test_evaluation_summary_endpoint(self, client):
        with patch('solution.rag_system.evaluator.get_evaluation_summary') as mock_summary:
            mock_summary.return_value = {
                'total_evaluations': 10,
                'average_scores': {
                    'faithfulness': 0.8,
                    'relevancy': 0.9,
                    'precision': 0.7
                }
            }
            
            response = client.get("/evaluation/summary?hours=24")
            
            assert response.status_code == 200
            data = response.json()
            assert 'total_evaluations' in data
            assert 'average_scores' in data
    
    def test_metrics_endpoint(self, client):
        response = client.get("/metrics")
        
        # Should return Prometheus metrics format
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; version=0.0.4; charset=utf-8"


class TestIntegration:
    """Integration tests for the complete system"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self):
        """Test complete workflow from document processing to response generation"""
        
        # Initialize system
        system = ProductionRAGSystem()
        
        # Create sample documents
        sample_docs = system._create_sample_documents()
        system.documents = {doc.id: doc for doc in sample_docs}
        
        # Mock retriever to avoid building real indexes
        with patch.object(HybridRetriever, '_build_indexes'):
            system.retriever = HybridRetriever(sample_docs)
            
            # Mock retrieval results
            mock_contexts = [
                {
                    'content': 'JWT tokens are used for API authentication.',
                    'score': 0.9,
                    'document_id': sample_docs[0].id,
                    'document_title': sample_docs[0].title
                }
            ]
            system.retriever.retrieve = Mock(return_value=mock_contexts)
        
        # Test query processing
        request = QueryRequest(
            query="How do I authenticate with the API?",
            options=QueryOptions(max_results=3, use_reranking=True)
        )
        
        response = await system.process_query(request)
        
        # Verify response structure
        assert response.answer is not None
        assert len(response.sources) > 0
        assert response.metadata.query_id is not None
        assert response.metadata.processing_time > 0
        
        # Verify evaluation scores
        eval_scores = response.metadata.evaluation_scores
        assert 'faithfulness' in eval_scores
        assert 'relevancy' in eval_scores
        assert 'context_precision' in eval_scores
        assert 'overall_score' in eval_scores
        
        # Verify all scores are in valid range
        for score in eval_scores.values():
            assert 0 <= score <= 1
        
        # Test system health
        health = system.get_system_health()
        assert health['status'] in ['healthy', 'warning', 'critical']
        assert health['total_queries'] > 0
    
    def test_performance_benchmarks(self):
        """Test system performance meets requirements"""
        
        system = ProductionRAGSystem()
        sample_docs = system._create_sample_documents()
        system.documents = {doc.id: doc for doc in sample_docs}
        
        # Mock retriever
        with patch.object(HybridRetriever, '_build_indexes'):
            system.retriever = HybridRetriever(sample_docs)
            system.retriever.retrieve = Mock(return_value=[{
                'content': 'Test content',
                'score': 0.9,
                'document_id': 'doc1',
                'document_title': 'Test Doc'
            }])
        
        # Test response time
        async def test_response_time():
            request = QueryRequest(query="Test query")
            start_time = time.time()
            await system.process_query(request)
            end_time = time.time()
            
            response_time = end_time - start_time
            assert response_time < 5.0  # Should be under 5 seconds for mock
        
        asyncio.run(test_response_time())
        
        # Test concurrent requests (simplified)
        async def test_concurrent_requests():
            requests = [QueryRequest(query=f"Test query {i}") for i in range(5)]
            
            start_time = time.time()
            tasks = [system.process_query(req) for req in requests]
            await asyncio.gather(*tasks)
            end_time = time.time()
            
            total_time = end_time - start_time
            assert total_time < 10.0  # Should handle 5 concurrent requests quickly
        
        asyncio.run(test_concurrent_requests())


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "--cov=solution", "--cov-report=html"])