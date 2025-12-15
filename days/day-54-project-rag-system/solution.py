"""
Day 54: Project - Production RAG System
Complete implementation of an enterprise-ready RAG system with advanced retrieval,
evaluation, monitoring, and deployment capabilities.
"""

import os
import json
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import uuid

# Core dependencies
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import httpx

# Document processing
from langchain.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
from langchain.retrievers import BM25Retriever
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Evaluation and monitoring
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
import prometheus_client
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Utilities
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")


# Configuration Management
class Config:
    """Centralized configuration management"""
    
    def __init__(self):
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
        self.VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "./data/vectordb")
        self.DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "./data/documents")
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.MAX_CHUNK_SIZE = int(os.getenv("MAX_CHUNK_SIZE", "1000"))
        self.CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
        self.MAX_RETRIEVAL_DOCS = int(os.getenv("MAX_RETRIEVAL_DOCS", "20"))
        self.RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
        self.RESPONSE_TIMEOUT = int(os.getenv("RESPONSE_TIMEOUT", "30"))
        
        # Quality thresholds
        self.FAITHFULNESS_THRESHOLD = float(os.getenv("FAITHFULNESS_THRESHOLD", "0.7"))
        self.RELEVANCY_THRESHOLD = float(os.getenv("RELEVANCY_THRESHOLD", "0.8"))
        self.PRECISION_THRESHOLD = float(os.getenv("PRECISION_THRESHOLD", "0.6"))


config = Config()

# Logging setup
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('rag_requests_total', 'Total RAG requests', ['endpoint', 'status'])
REQUEST_DURATION = Histogram('rag_request_duration_seconds', 'RAG request duration')
EVALUATION_SCORES = Histogram('rag_evaluation_scores', 'RAG evaluation scores', ['metric'])
ACTIVE_CONNECTIONS = Gauge('rag_active_connections', 'Active connections')
DOCUMENT_COUNT = Gauge('rag_documents_total', 'Total documents indexed')


# Data Models
class DocumentMetadata(BaseModel):
    """Document metadata schema"""
    source: str
    author: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    tags: List[str] = []
    category: Optional[str] = None
    version: Optional[str] = "1.0"


class DocumentChunk(BaseModel):
    """Document chunk schema"""
    id: str
    content: str
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = {}


class Document(BaseModel):
    """Complete document schema"""
    id: str
    title: str
    content: str
    metadata: DocumentMetadata
    chunks: List[DocumentChunk] = []


class QueryFilters(BaseModel):
    """Query filtering options"""
    category: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    date_range: Optional[Dict[str, datetime]] = None


class QueryOptions(BaseModel):
    """Query processing options"""
    max_results: int = Field(default=5, ge=1, le=20)
    use_reranking: bool = True
    expand_query: bool = True
    include_sources: bool = True
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class QueryRequest(BaseModel):
    """RAG query request schema"""
    query: str = Field(..., min_length=1, max_length=1000)
    filters: Optional[QueryFilters] = None
    options: Optional[QueryOptions] = QueryOptions()
    
    @validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class SourceReference(BaseModel):
    """Source reference in response"""
    document_id: str
    title: str
    relevance_score: float
    excerpt: str
    url: Optional[str] = None


class ResponseMetadata(BaseModel):
    """Response metadata"""
    query_id: str
    processing_time: float
    model_used: str
    evaluation_scores: Dict[str, float]
    timestamp: datetime


class RAGResponse(BaseModel):
    """RAG system response schema"""
    answer: str
    sources: List[SourceReference]
    metadata: ResponseMetadata


class EvaluationResult(BaseModel):
    """Evaluation result schema"""
    query_id: str
    faithfulness: float
    relevancy: float
    context_precision: float
    context_recall: Optional[float] = None
    overall_score: float
    alerts: List[str] = []
    timestamp: datetime


# Document Processing Pipeline
class DocumentProcessor:
    """Handles document ingestion, processing, and indexing"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.MAX_CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        self.embeddings = self._initialize_embeddings()
        self.processed_docs = {}
        
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if config.OPENAI_API_KEY:
            return OpenAIEmbeddings(
                openai_api_key=config.OPENAI_API_KEY,
                model=config.EMBEDDING_MODEL
            )
        else:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def load_document(self, file_path: str) -> Document:
        """Load and process a single document"""
        try:
            file_path = Path(file_path)
            
            # Choose appropriate loader based on file extension
            if file_path.suffix.lower() == '.pdf':
                loader = PyPDFLoader(str(file_path))
            elif file_path.suffix.lower() == '.md':
                loader = UnstructuredMarkdownLoader(str(file_path))
            else:
                loader = TextLoader(str(file_path))
            
            # Load document
            docs = loader.load()
            content = "\n".join([doc.page_content for doc in docs])
            
            # Create document metadata
            metadata = DocumentMetadata(
                source=str(file_path),
                created_at=datetime.fromtimestamp(file_path.stat().st_ctime),
                updated_at=datetime.fromtimestamp(file_path.stat().st_mtime),
                category=self._infer_category(file_path),
                tags=self._extract_tags(content)
            )
            
            # Create document
            doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
            document = Document(
                id=doc_id,
                title=file_path.stem,
                content=content,
                metadata=metadata
            )
            
            # Process chunks
            document.chunks = self._create_chunks(document)
            
            logger.info(f"Processed document: {file_path} ({len(document.chunks)} chunks)")
            return document
            
        except Exception as e:
            logger.error(f"Error processing document {file_path}: {e}")
            raise
    
    def _create_chunks(self, document: Document) -> List[DocumentChunk]:
        """Create chunks from document content"""
        texts = self.text_splitter.split_text(document.content)
        chunks = []
        
        current_pos = 0
        for i, text in enumerate(texts):
            # Find the position of this chunk in the original content
            start_pos = document.content.find(text, current_pos)
            if start_pos == -1:
                start_pos = current_pos
            
            end_pos = start_pos + len(text)
            current_pos = end_pos
            
            chunk = DocumentChunk(
                id=f"{document.id}_chunk_{i}",
                content=text,
                start_char=start_pos,
                end_char=end_pos,
                metadata={
                    "document_id": document.id,
                    "chunk_index": i,
                    "document_title": document.title
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _infer_category(self, file_path: Path) -> str:
        """Infer document category from path and content"""
        path_parts = file_path.parts
        
        if 'api' in str(file_path).lower():
            return 'api_documentation'
        elif 'policy' in str(file_path).lower():
            return 'policy'
        elif 'guide' in str(file_path).lower():
            return 'guide'
        elif 'readme' in file_path.name.lower():
            return 'readme'
        else:
            return 'general'
    
    def _extract_tags(self, content: str) -> List[str]:
        """Extract tags from document content"""
        # Simple tag extraction based on common technical terms
        tech_terms = [
            'python', 'javascript', 'api', 'database', 'security',
            'authentication', 'deployment', 'testing', 'monitoring',
            'performance', 'scalability', 'architecture'
        ]
        
        content_lower = content.lower()
        tags = [term for term in tech_terms if term in content_lower]
        return tags[:5]  # Limit to 5 tags
    
    def process_directory(self, directory_path: str) -> List[Document]:
        """Process all documents in a directory"""
        directory = Path(directory_path)
        documents = []
        
        supported_extensions = {'.txt', '.md', '.pdf'}
        
        for file_path in directory.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                try:
                    document = self.load_document(file_path)
                    documents.append(document)
                    self.processed_docs[document.id] = document
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
        
        logger.info(f"Processed {len(documents)} documents from {directory_path}")
        return documents


# Hybrid Retrieval Engine
class HybridRetriever:
    """Advanced hybrid retrieval system combining dense and sparse methods"""
    
    def __init__(self, documents: List[Document]):
        self.documents = {doc.id: doc for doc in documents}
        self.embeddings = self._initialize_embeddings()
        self.vector_store = None
        self.bm25_retriever = None
        self.cross_encoder = None
        self._build_indexes()
    
    def _initialize_embeddings(self):
        """Initialize embedding model"""
        if config.OPENAI_API_KEY:
            return OpenAIEmbeddings(
                openai_api_key=config.OPENAI_API_KEY,
                model=config.EMBEDDING_MODEL
            )
        else:
            return HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    
    def _build_indexes(self):
        """Build vector and keyword indexes"""
        if not self.documents:
            logger.warning("No documents provided for indexing")
            return
        
        # Prepare texts and metadata for indexing
        texts = []
        metadatas = []
        
        for document in self.documents.values():
            for chunk in document.chunks:
                texts.append(chunk.content)
                metadatas.append({
                    **chunk.metadata,
                    'document_title': document.title,
                    'document_category': document.metadata.category,
                    'document_tags': document.metadata.tags
                })
        
        # Build vector store
        try:
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            logger.info(f"Built vector index with {len(texts)} chunks")
        except Exception as e:
            logger.error(f"Failed to build vector index: {e}")
            raise
        
        # Build BM25 index
        try:
            self.bm25_retriever = BM25Retriever.from_texts(texts)
            self.bm25_retriever.k = config.MAX_RETRIEVAL_DOCS
            logger.info(f"Built BM25 index with {len(texts)} chunks")
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
            raise
    
    def dense_search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """Perform dense (vector) search"""
        if not self.vector_store:
            return []
        
        try:
            results = self.vector_store.similarity_search_with_score(query, k=k)
            return [(doc.page_content, score) for doc, score in results]
        except Exception as e:
            logger.error(f"Dense search failed: {e}")
            return []
    
    def sparse_search(self, query: str, k: int = 10) -> List[str]:
        """Perform sparse (BM25) search"""
        if not self.bm25_retriever:
            return []
        
        try:
            results = self.bm25_retriever.get_relevant_documents(query)
            return [doc.page_content for doc in results[:k]]
        except Exception as e:
            logger.error(f"Sparse search failed: {e}")
            return []
    
    def reciprocal_rank_fusion(self, dense_results: List[Tuple[str, float]], 
                             sparse_results: List[str], k: int = 60) -> List[Tuple[str, float]]:
        """Combine results using Reciprocal Rank Fusion"""
        scores = defaultdict(float)
        
        # Add scores from dense results
        for rank, (content, _) in enumerate(dense_results):
            scores[content] += 1 / (k + rank + 1)
        
        # Add scores from sparse results
        for rank, content in enumerate(sparse_results):
            scores[content] += 1 / (k + rank + 1)
        
        # Sort by combined scores
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results
    
    def expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        # Simple query expansion - in production, use more sophisticated methods
        expansion_terms = {
            'api': ['endpoint', 'service', 'interface'],
            'error': ['exception', 'failure', 'issue'],
            'deploy': ['deployment', 'release', 'publish'],
            'test': ['testing', 'validation', 'verification'],
            'config': ['configuration', 'settings', 'setup']
        }
        
        query_words = query.lower().split()
        expanded_terms = []
        
        for word in query_words:
            expanded_terms.append(word)
            if word in expansion_terms:
                expanded_terms.extend(expansion_terms[word][:2])  # Add up to 2 synonyms
        
        return ' '.join(expanded_terms)
    
    def retrieve(self, query: str, filters: Optional[QueryFilters] = None, 
                options: Optional[QueryOptions] = None) -> List[Dict[str, Any]]:
        """Main retrieval method combining all techniques"""
        if options is None:
            options = QueryOptions()
        
        # Expand query if requested
        search_query = self.expand_query(query) if options.expand_query else query
        
        # Perform hybrid search
        dense_results = self.dense_search(search_query, k=config.MAX_RETRIEVAL_DOCS)
        sparse_results = self.sparse_search(search_query, k=config.MAX_RETRIEVAL_DOCS)
        
        # Combine using RRF
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Convert to structured format
        retrieved_contexts = []
        for content, score in fused_results[:options.max_results * 2]:  # Get more for re-ranking
            # Find source document
            source_doc = None
            for doc in self.documents.values():
                for chunk in doc.chunks:
                    if chunk.content == content:
                        source_doc = doc
                        break
                if source_doc:
                    break
            
            if source_doc:
                retrieved_contexts.append({
                    'content': content,
                    'score': score,
                    'document_id': source_doc.id,
                    'document_title': source_doc.title,
                    'metadata': source_doc.metadata.dict()
                })
        
        # Apply filters if provided
        if filters:
            retrieved_contexts = self._apply_filters(retrieved_contexts, filters)
        
        # Re-ranking (simplified - in production use cross-encoder)
        if options.use_reranking:
            retrieved_contexts = self._rerank_contexts(query, retrieved_contexts)
        
        return retrieved_contexts[:options.max_results]
    
    def _apply_filters(self, contexts: List[Dict], filters: QueryFilters) -> List[Dict]:
        """Apply filters to retrieved contexts"""
        filtered_contexts = []
        
        for context in contexts:
            metadata = context['metadata']
            
            # Category filter
            if filters.category and metadata.get('category') not in filters.category:
                continue
            
            # Tags filter
            if filters.tags:
                doc_tags = metadata.get('tags', [])
                if not any(tag in doc_tags for tag in filters.tags):
                    continue
            
            # Date range filter
            if filters.date_range:
                doc_date = metadata.get('created_at')
                if doc_date:
                    if isinstance(doc_date, str):
                        doc_date = datetime.fromisoformat(doc_date)
                    
                    start_date = filters.date_range.get('start')
                    end_date = filters.date_range.get('end')
                    
                    if start_date and doc_date < start_date:
                        continue
                    if end_date and doc_date > end_date:
                        continue
            
            filtered_contexts.append(context)
        
        return filtered_contexts
    
    def _rerank_contexts(self, query: str, contexts: List[Dict]) -> List[Dict]:
        """Re-rank contexts using cross-encoder (simplified implementation)"""
        # Simplified re-ranking based on query-context similarity
        # In production, use a proper cross-encoder model
        
        def calculate_relevance(query: str, context: str) -> float:
            query_words = set(query.lower().split())
            context_words = set(context.lower().split())
            
            if not query_words:
                return 0.0
            
            overlap = len(query_words.intersection(context_words))
            return overlap / len(query_words)
        
        # Calculate relevance scores
        for context in contexts:
            relevance = calculate_relevance(query, context['content'])
            # Combine with original retrieval score
            context['score'] = 0.7 * context['score'] + 0.3 * relevance
        
        # Sort by updated scores
        contexts.sort(key=lambda x: x['score'], reverse=True)
        return contexts


# Generation Service
class GenerationService:
    """Handles response generation using LLMs"""
    
    def __init__(self):
        self.llm = self._initialize_llm()
        self.max_context_length = 4000  # Tokens
        self.response_templates = {
            'default': """Based on the provided context, please answer the following question accurately and concisely.

Context:
{context}

Question: {question}

Answer:""",
            'with_sources': """Based on the provided context, please answer the following question accurately and concisely. Include relevant source references.

Context:
{context}

Question: {question}

Please provide a comprehensive answer and mention which sources support your response.

Answer:"""
        }
    
    def _initialize_llm(self):
        """Initialize language model"""
        if config.OPENAI_API_KEY:
            return ChatOpenAI(
                openai_api_key=config.OPENAI_API_KEY,
                model_name=config.LLM_MODEL,
                temperature=0.7,
                max_tokens=1000
            )
        else:
            # Fallback to a mock implementation for demo
            return MockLLM()
    
    def _prepare_context(self, contexts: List[Dict]) -> str:
        """Prepare context string from retrieved documents"""
        context_parts = []
        
        for i, ctx in enumerate(contexts, 1):
            source_info = f"[Source {i}: {ctx['document_title']}]"
            content = ctx['content']
            context_parts.append(f"{source_info}\n{content}\n")
        
        return "\n".join(context_parts)
    
    def _validate_response(self, response: str, contexts: List[Dict]) -> Tuple[str, List[str]]:
        """Validate and clean response"""
        alerts = []
        
        # Check response length
        if len(response) < 10:
            alerts.append("Response too short")
        elif len(response) > 2000:
            alerts.append("Response too long")
            response = response[:2000] + "..."
        
        # Check for hallucination indicators
        if "I don't know" in response or "not provided" in response:
            alerts.append("Potential knowledge gap")
        
        # Remove any unwanted prefixes/suffixes
        response = response.strip()
        if response.startswith("Answer:"):
            response = response[7:].strip()
        
        return response, alerts
    
    def generate_response(self, query: str, contexts: List[Dict], 
                         options: Optional[QueryOptions] = None) -> Tuple[str, List[str]]:
        """Generate response using LLM"""
        if options is None:
            options = QueryOptions()
        
        try:
            # Prepare context
            context_text = self._prepare_context(contexts)
            
            # Choose template
            template_key = 'with_sources' if options.include_sources else 'default'
            template = self.response_templates[template_key]
            
            # Format prompt
            prompt = template.format(context=context_text, question=query)
            
            # Generate response
            if hasattr(self.llm, 'predict'):
                response = self.llm.predict(prompt)
            else:
                response = self.llm.generate(prompt)
            
            # Validate response
            validated_response, alerts = self._validate_response(response, contexts)
            
            return validated_response, alerts
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"I apologize, but I encountered an error while generating a response: {str(e)}", ["generation_error"]


class MockLLM:
    """Mock LLM for demonstration purposes"""
    
    def predict(self, prompt: str) -> str:
        """Generate a mock response"""
        if "api" in prompt.lower():
            return "Based on the provided documentation, the API endpoint allows you to perform the requested operation. Please refer to the authentication requirements and rate limits mentioned in the documentation."
        elif "error" in prompt.lower():
            return "The error you're encountering is typically caused by configuration issues. Please check the logs and verify your setup according to the troubleshooting guide."
        else:
            return "Based on the available information, here's what I can tell you about your question. The documentation provides relevant details that should help address your inquiry."
    
    def generate(self, prompt: str) -> str:
        return self.predict(prompt)


# Evaluation Service
class EvaluationService:
    """Handles response evaluation using RAGAS and custom metrics"""
    
    def __init__(self):
        self.evaluation_history = []
        self.thresholds = {
            'faithfulness': config.FAITHFULNESS_THRESHOLD,
            'relevancy': config.RELEVANCY_THRESHOLD,
            'precision': config.PRECISION_THRESHOLD
        }
    
    def evaluate_response(self, query: str, answer: str, contexts: List[Dict], 
                         query_id: str) -> EvaluationResult:
        """Evaluate a single response using RAGAS metrics"""
        try:
            # Prepare data for RAGAS
            context_texts = [ctx['content'] for ctx in contexts]
            
            # Calculate metrics (simplified implementation)
            faithfulness_score = self._calculate_faithfulness(answer, context_texts)
            relevancy_score = self._calculate_relevancy(query, answer)
            precision_score = self._calculate_precision(query, context_texts)
            
            # Calculate overall score
            overall_score = (
                0.4 * faithfulness_score +
                0.3 * relevancy_score +
                0.3 * precision_score
            )
            
            # Generate alerts
            alerts = []
            if faithfulness_score < self.thresholds['faithfulness']:
                alerts.append(f"Low faithfulness: {faithfulness_score:.3f}")
            if relevancy_score < self.thresholds['relevancy']:
                alerts.append(f"Low relevancy: {relevancy_score:.3f}")
            if precision_score < self.thresholds['precision']:
                alerts.append(f"Low precision: {precision_score:.3f}")
            
            result = EvaluationResult(
                query_id=query_id,
                faithfulness=faithfulness_score,
                relevancy=relevancy_score,
                context_precision=precision_score,
                overall_score=overall_score,
                alerts=alerts,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.evaluation_history.append(result)
            
            # Update Prometheus metrics
            EVALUATION_SCORES.labels(metric='faithfulness').observe(faithfulness_score)
            EVALUATION_SCORES.labels(metric='relevancy').observe(relevancy_score)
            EVALUATION_SCORES.labels(metric='precision').observe(precision_score)
            EVALUATION_SCORES.labels(metric='overall').observe(overall_score)
            
            return result
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return EvaluationResult(
                query_id=query_id,
                faithfulness=0.0,
                relevancy=0.0,
                context_precision=0.0,
                overall_score=0.0,
                alerts=[f"Evaluation error: {str(e)}"],
                timestamp=datetime.now()
            )
    
    def _calculate_faithfulness(self, answer: str, contexts: List[str]) -> float:
        """Calculate faithfulness score (simplified)"""
        if not answer or not contexts:
            return 0.0
        
        # Simple implementation: check if answer content appears in contexts
        answer_words = set(answer.lower().split())
        context_words = set()
        for context in contexts:
            context_words.update(context.lower().split())
        
        if not answer_words:
            return 1.0
        
        supported_words = answer_words.intersection(context_words)
        return len(supported_words) / len(answer_words)
    
    def _calculate_relevancy(self, query: str, answer: str) -> float:
        """Calculate answer relevancy score (simplified)"""
        if not query or not answer:
            return 0.0
        
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        
        if not query_words:
            return 1.0
        
        overlap = query_words.intersection(answer_words)
        return len(overlap) / len(query_words)
    
    def _calculate_precision(self, query: str, contexts: List[str]) -> float:
        """Calculate context precision score (simplified)"""
        if not query or not contexts:
            return 0.0
        
        query_words = set(query.lower().split())
        relevant_contexts = 0
        
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = query_words.intersection(context_words)
            
            # Consider context relevant if it has significant overlap
            if len(overlap) >= len(query_words) * 0.3:
                relevant_contexts += 1
        
        return relevant_contexts / len(contexts)
    
    def get_evaluation_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get evaluation summary for the last N hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_evaluations = [
            eval_result for eval_result in self.evaluation_history
            if eval_result.timestamp >= cutoff_time
        ]
        
        if not recent_evaluations:
            return {"message": f"No evaluations in the last {hours} hours"}
        
        # Calculate statistics
        faithfulness_scores = [e.faithfulness for e in recent_evaluations]
        relevancy_scores = [e.relevancy for e in recent_evaluations]
        precision_scores = [e.context_precision for e in recent_evaluations]
        overall_scores = [e.overall_score for e in recent_evaluations]
        
        total_alerts = sum(len(e.alerts) for e in recent_evaluations)
        
        return {
            "period_hours": hours,
            "total_evaluations": len(recent_evaluations),
            "average_scores": {
                "faithfulness": np.mean(faithfulness_scores),
                "relevancy": np.mean(relevancy_scores),
                "precision": np.mean(precision_scores),
                "overall": np.mean(overall_scores)
            },
            "score_ranges": {
                "faithfulness": {"min": np.min(faithfulness_scores), "max": np.max(faithfulness_scores)},
                "relevancy": {"min": np.min(relevancy_scores), "max": np.max(relevancy_scores)},
                "precision": {"min": np.min(precision_scores), "max": np.max(precision_scores)},
                "overall": {"min": np.min(overall_scores), "max": np.max(overall_scores)}
            },
            "total_alerts": total_alerts,
            "alert_rate": total_alerts / len(recent_evaluations)
        }


# Production RAG System
class ProductionRAGSystem:
    """Complete production RAG system integrating all components"""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.retriever = None
        self.generator = GenerationService()
        self.evaluator = EvaluationService()
        self.documents = {}
        self.system_stats = {
            'start_time': datetime.now(),
            'total_queries': 0,
            'total_documents': 0,
            'average_response_time': 0.0
        }
    
    def initialize_system(self, documents_path: str = None):
        """Initialize the RAG system with documents"""
        try:
            if documents_path is None:
                documents_path = config.DOCUMENTS_PATH
            
            logger.info(f"Initializing RAG system with documents from: {documents_path}")
            
            # Process documents
            if os.path.exists(documents_path):
                documents = self.document_processor.process_directory(documents_path)
            else:
                logger.warning(f"Documents path {documents_path} not found, using sample documents")
                documents = self._create_sample_documents()
            
            # Store documents
            self.documents = {doc.id: doc for doc in documents}
            self.system_stats['total_documents'] = len(documents)
            
            # Initialize retriever
            self.retriever = HybridRetriever(documents)
            
            # Update metrics
            DOCUMENT_COUNT.set(len(documents))
            
            logger.info(f"RAG system initialized with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _create_sample_documents(self) -> List[Document]:
        """Create sample documents for demonstration"""
        sample_docs = [
            {
                "title": "API Authentication Guide",
                "content": """
                # API Authentication Guide
                
                Our API uses JWT tokens for authentication. To authenticate:
                
                1. Send a POST request to /auth/login with your credentials
                2. Include the returned JWT token in the Authorization header
                3. Format: Authorization: Bearer <token>
                
                Tokens expire after 24 hours and must be refreshed.
                
                ## Rate Limits
                - 1000 requests per hour for authenticated users
                - 100 requests per hour for unauthenticated users
                
                ## Error Codes
                - 401: Invalid or expired token
                - 429: Rate limit exceeded
                """,
                "category": "api_documentation",
                "tags": ["authentication", "api", "security"]
            },
            {
                "title": "Deployment Best Practices",
                "content": """
                # Deployment Best Practices
                
                ## Environment Setup
                - Use environment variables for configuration
                - Separate staging and production environments
                - Implement proper logging and monitoring
                
                ## Security Considerations
                - Enable HTTPS in production
                - Use secrets management for sensitive data
                - Implement proper access controls
                
                ## Performance Optimization
                - Use caching where appropriate
                - Implement connection pooling
                - Monitor resource usage
                
                ## Rollback Strategy
                - Maintain previous version for quick rollback
                - Test rollback procedures regularly
                - Document rollback steps
                """,
                "category": "guide",
                "tags": ["deployment", "security", "performance"]
            },
            {
                "title": "Troubleshooting Common Issues",
                "content": """
                # Troubleshooting Common Issues
                
                ## Database Connection Errors
                - Check connection string format
                - Verify database server is running
                - Check firewall settings
                - Validate credentials
                
                ## Performance Issues
                - Monitor CPU and memory usage
                - Check for slow queries
                - Review caching configuration
                - Analyze network latency
                
                ## Authentication Failures
                - Verify token format and expiration
                - Check user permissions
                - Review authentication logs
                - Validate API key configuration
                """,
                "category": "troubleshooting",
                "tags": ["troubleshooting", "database", "performance", "authentication"]
            }
        ]
        
        documents = []
        for i, doc_data in enumerate(sample_docs):
            doc_id = f"sample_doc_{i}"
            metadata = DocumentMetadata(
                source="sample",
                category=doc_data["category"],
                tags=doc_data["tags"],
                created_at=datetime.now()
            )
            
            document = Document(
                id=doc_id,
                title=doc_data["title"],
                content=doc_data["content"],
                metadata=metadata
            )
            
            # Create chunks
            document.chunks = self.document_processor._create_chunks(document)
            documents.append(document)
        
        return documents
    
    async def process_query(self, request: QueryRequest) -> RAGResponse:
        """Process a complete RAG query"""
        start_time = time.time()
        query_id = str(uuid.uuid4())
        
        try:
            # Update metrics
            self.system_stats['total_queries'] += 1
            ACTIVE_CONNECTIONS.inc()
            
            logger.info(f"Processing query {query_id}: {request.query}")
            
            # Retrieve relevant contexts
            contexts = self.retriever.retrieve(
                query=request.query,
                filters=request.filters,
                options=request.options
            )
            
            if not contexts:
                raise HTTPException(
                    status_code=404,
                    detail="No relevant information found for your query"
                )
            
            # Generate response
            answer, generation_alerts = self.generator.generate_response(
                query=request.query,
                contexts=contexts,
                options=request.options
            )
            
            # Evaluate response
            evaluation_result = self.evaluator.evaluate_response(
                query=request.query,
                answer=answer,
                contexts=contexts,
                query_id=query_id
            )
            
            # Prepare source references
            sources = []
            for ctx in contexts:
                source = SourceReference(
                    document_id=ctx['document_id'],
                    title=ctx['document_title'],
                    relevance_score=ctx['score'],
                    excerpt=ctx['content'][:200] + "..." if len(ctx['content']) > 200 else ctx['content']
                )
                sources.append(source)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update average response time
            total_queries = self.system_stats['total_queries']
            current_avg = self.system_stats['average_response_time']
            self.system_stats['average_response_time'] = (
                (current_avg * (total_queries - 1) + processing_time) / total_queries
            )
            
            # Prepare response metadata
            metadata = ResponseMetadata(
                query_id=query_id,
                processing_time=processing_time,
                model_used=config.LLM_MODEL,
                evaluation_scores={
                    'faithfulness': evaluation_result.faithfulness,
                    'relevancy': evaluation_result.relevancy,
                    'context_precision': evaluation_result.context_precision,
                    'overall_score': evaluation_result.overall_score
                },
                timestamp=datetime.now()
            )
            
            # Create response
            response = RAGResponse(
                answer=answer,
                sources=sources,
                metadata=metadata
            )
            
            # Update Prometheus metrics
            REQUEST_DURATION.observe(processing_time)
            REQUEST_COUNT.labels(endpoint='query', status='success').inc()
            
            logger.info(f"Query {query_id} processed successfully in {processing_time:.2f}s")
            
            return response
            
        except Exception as e:
            processing_time = time.time() - start_time
            REQUEST_COUNT.labels(endpoint='query', status='error').inc()
            logger.error(f"Query {query_id} failed: {e}")
            raise
        
        finally:
            ACTIVE_CONNECTIONS.dec()
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health status"""
        uptime = datetime.now() - self.system_stats['start_time']
        
        # Get recent evaluation summary
        eval_summary = self.evaluator.get_evaluation_summary(hours=1)
        
        # Determine health status
        health_status = "healthy"
        if eval_summary.get('alert_rate', 0) > 0.1:  # More than 10% alerts
            health_status = "warning"
        if eval_summary.get('average_scores', {}).get('overall', 1.0) < 0.5:
            health_status = "critical"
        
        return {
            "status": health_status,
            "uptime_seconds": uptime.total_seconds(),
            "total_queries": self.system_stats['total_queries'],
            "total_documents": self.system_stats['total_documents'],
            "average_response_time": self.system_stats['average_response_time'],
            "evaluation_summary": eval_summary,
            "timestamp": datetime.now()
        }


# FastAPI Application
app = FastAPI(
    title="Production RAG System",
    description="Enterprise-ready Retrieval-Augmented Generation system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize RAG system
rag_system = ProductionRAGSystem()

@app.on_event("startup")
async def startup_event():
    """Initialize the RAG system on startup"""
    try:
        rag_system.initialize_system()
        logger.info("RAG system startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to start RAG system: {e}")
        raise


# API Endpoints
@app.post("/query", response_model=RAGResponse)
async def query_endpoint(request: QueryRequest):
    """Main query endpoint for RAG system"""
    try:
        response = await rag_system.process_query(request)
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Query endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        health_status = rag_system.get_system_health()
        status_code = 200
        
        if health_status["status"] == "warning":
            status_code = 200  # Still operational
        elif health_status["status"] == "critical":
            status_code = 503  # Service unavailable
        
        return JSONResponse(
            status_code=status_code,
            content=health_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={"status": "error", "message": str(e)}
        )


@app.get("/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.get("/documents")
async def list_documents():
    """List all indexed documents"""
    try:
        documents_info = []
        for doc_id, document in rag_system.documents.items():
            doc_info = {
                "id": doc_id,
                "title": document.title,
                "category": document.metadata.category,
                "tags": document.metadata.tags,
                "chunk_count": len(document.chunks),
                "created_at": document.metadata.created_at
            }
            documents_info.append(doc_info)
        
        return {
            "total_documents": len(documents_info),
            "documents": documents_info
        }
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@app.get("/evaluation/summary")
async def evaluation_summary(hours: int = 24):
    """Get evaluation summary for the specified time period"""
    try:
        summary = rag_system.evaluator.get_evaluation_summary(hours=hours)
        return summary
    except Exception as e:
        logger.error(f"Evaluation summary error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get evaluation summary: {str(e)}"
        )


@app.post("/documents/upload")
async def upload_document(background_tasks: BackgroundTasks):
    """Upload and process new documents (placeholder)"""
    # This would handle file uploads in a real implementation
    return {
        "message": "Document upload endpoint - implementation depends on specific requirements",
        "status": "placeholder"
    }


# Main execution
def main():
    """Main function to run the RAG system"""
    print("🚀 Production RAG System - Day 54 Project")
    print("=" * 60)
    
    # Check if running as API server
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "server":
        print("Starting FastAPI server...")
        uvicorn.run(
            "solution:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
        return
    
    # Run demonstration
    print("\n📋 Initializing Production RAG System...")
    
    try:
        # Initialize system
        rag_system.initialize_system()
        
        print(f"✅ System initialized with {len(rag_system.documents)} documents")
        
        # Test queries
        test_queries = [
            "How do I authenticate with the API?",
            "What are the deployment best practices?",
            "How do I troubleshoot database connection errors?",
            "What are the rate limits for the API?"
        ]
        
        print("\n🔍 Testing RAG System with Sample Queries...")
        print("-" * 50)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            
            # Create request
            request = QueryRequest(
                query=query,
                options=QueryOptions(max_results=3, use_reranking=True)
            )
            
            # Process query
            start_time = time.time()
            response = asyncio.run(rag_system.process_query(request))
            processing_time = time.time() - start_time
            
            print(f"   Processing Time: {processing_time:.2f}s")
            print(f"   Answer: {response.answer[:150]}...")
            print(f"   Sources: {len(response.sources)} documents")
            print(f"   Evaluation Scores:")
            print(f"     - Faithfulness: {response.metadata.evaluation_scores['faithfulness']:.3f}")
            print(f"     - Relevancy: {response.metadata.evaluation_scores['relevancy']:.3f}")
            print(f"     - Overall: {response.metadata.evaluation_scores['overall_score']:.3f}")
        
        # System health check
        print("\n📊 System Health Status...")
        print("-" * 30)
        health = rag_system.get_system_health()
        print(f"Status: {health['status'].upper()}")
        print(f"Total Queries: {health['total_queries']}")
        print(f"Average Response Time: {health['average_response_time']:.2f}s")
        print(f"Documents Indexed: {health['total_documents']}")
        
        # Evaluation summary
        eval_summary = health.get('evaluation_summary', {})
        if 'average_scores' in eval_summary:
            print(f"\nEvaluation Summary (last hour):")
            avg_scores = eval_summary['average_scores']
            print(f"  - Average Faithfulness: {avg_scores['faithfulness']:.3f}")
            print(f"  - Average Relevancy: {avg_scores['relevancy']:.3f}")
            print(f"  - Average Precision: {avg_scores['precision']:.3f}")
            print(f"  - Alert Rate: {eval_summary.get('alert_rate', 0):.1%}")
        
        print("\n🎉 Production RAG System demonstration completed successfully!")
        
        print("\n📚 Next Steps:")
        print("1. Run 'python solution.py server' to start the API server")
        print("2. Visit http://localhost:8000/docs for interactive API documentation")
        print("3. Use /query endpoint to test the RAG system")
        print("4. Monitor system health at /health endpoint")
        print("5. View metrics at /metrics endpoint")
        
        print("\n🔧 Production Deployment:")
        print("1. Configure environment variables for production")
        print("2. Set up proper document storage and processing")
        print("3. Implement authentication and rate limiting")
        print("4. Configure monitoring and alerting")
        print("5. Set up CI/CD pipeline for automated deployment")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        logger.error(f"Demonstration failed: {e}")


if __name__ == "__main__":
    main()