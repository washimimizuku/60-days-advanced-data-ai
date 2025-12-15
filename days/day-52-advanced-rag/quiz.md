# Day 52 Quiz: Advanced RAG - Retrieval-Augmented Generation Systems

## Instructions
Answer all questions. Each question has one correct answer.

---

1. **What is the primary advantage of hybrid search in RAG systems?**
   - A) It only uses vector embeddings for better semantic understanding
   - B) It combines dense (semantic) and sparse (keyword) retrieval for improved recall and precision
   - C) It reduces computational costs by using fewer embedding models
   - D) It eliminates the need for re-ranking mechanisms

2. **In Reciprocal Rank Fusion (RRF), what does the parameter 'k' control?**
   - A) The number of documents to retrieve from each method
   - B) The weight given to higher-ranked documents in the fusion process
   - C) The embedding dimension size
   - D) The number of retrieval methods to combine

3. **What is the main purpose of cross-encoder re-ranking in advanced RAG?**
   - A) To generate embeddings faster than bi-encoders
   - B) To reduce the number of documents before generation
   - C) To provide more accurate relevance scoring by considering query-document interactions
   - D) To cache frequently accessed documents

4. **Which caching strategy is most effective for production RAG systems?**
   - A) Only cache the final generated responses
   - B) Cache embeddings, intermediate results, and final responses at multiple levels
   - C) Cache only the original documents to save storage
   - D) Avoid caching to ensure real-time accuracy

5. **What is query expansion in the context of advanced RAG?**
   - A) Making queries longer by adding random words
   - B) Converting queries to multiple languages
   - C) Enhancing queries with related terms and synonyms to improve retrieval coverage
   - D) Splitting long queries into smaller parts

6. **In multi-hop reasoning for RAG, what happens at each "hop"?**
   - A) The system retrieves documents and generates follow-up queries based on the results
   - B) The system increases the embedding dimension
   - C) The system reduces the number of candidate documents
   - D) The system switches between different language models

7. **What is the primary benefit of hierarchical retrieval in RAG systems?**
   - A) It reduces storage requirements
   - B) It first identifies relevant documents, then finds specific chunks within them
   - C) It eliminates the need for embeddings
   - D) It works only with structured data

8. **Which metric is most commonly used to evaluate retrieval quality in RAG systems?**
   - A) BLEU score
   - B) Mean Reciprocal Rank (MRR) and Recall@K
   - C) Perplexity
   - D) F1 score only

9. **What is the main advantage of async processing in production RAG systems?**
   - A) It reduces memory usage
   - B) It allows concurrent execution of different retrieval methods, reducing overall latency
   - C) It improves embedding quality
   - D) It eliminates the need for caching

10. **Which approach is most effective for handling enterprise-scale RAG deployments?**
    - A) Use only the largest available language model
    - B) Implement multi-level caching, load balancing, and monitoring with proper scaling strategies
    - C) Store all documents in a single vector database
    - D) Avoid using re-ranking to reduce complexity

---

## Answer Key

**1. B** - Hybrid search combines dense (semantic) and sparse (keyword) retrieval for improved recall and precision. Dense retrieval captures semantic similarity while sparse retrieval ensures exact keyword matches aren't missed, providing the best of both approaches.

**2. B** - In RRF, the parameter 'k' controls the weight given to higher-ranked documents in the fusion process. The formula 1/(k + rank + 1) means higher-ranked documents get more weight, and 'k' adjusts how much emphasis is placed on ranking position.

**3. C** - Cross-encoder re-ranking provides more accurate relevance scoring by considering query-document interactions. Unlike bi-encoders that encode queries and documents separately, cross-encoders process them together, enabling better understanding of their relationship.

**4. B** - Multi-level caching (embeddings, intermediate results, and final responses) is most effective for production RAG systems. This approach reduces computational overhead at different stages while maintaining system responsiveness and reducing costs.

**5. C** - Query expansion enhances queries with related terms and synonyms to improve retrieval coverage. This technique helps capture relevant documents that might use different terminology than the original query, improving recall without sacrificing precision.

**6. A** - In multi-hop reasoning, at each "hop" the system retrieves documents based on the current query and generates follow-up queries based on the retrieved content. This allows for complex reasoning chains and gathering comprehensive information.

**7. B** - Hierarchical retrieval first identifies relevant documents at a high level, then finds specific chunks within those documents. This approach is more efficient than searching all chunks globally and provides better context by maintaining document-level relationships.

**8. B** - Mean Reciprocal Rank (MRR) and Recall@K are most commonly used to evaluate retrieval quality in RAG systems. MRR measures how well the system ranks relevant documents, while Recall@K measures the proportion of relevant documents found in the top K results.

**9. B** - Async processing allows concurrent execution of different retrieval methods (dense, sparse, re-ranking), reducing overall latency. Instead of running these sequentially, they can be executed in parallel, significantly improving system responsiveness.

**10. B** - Enterprise-scale RAG deployments require multi-level caching, load balancing, and monitoring with proper scaling strategies. This comprehensive approach ensures high availability, performance, and reliability while managing costs and maintaining quality at scale.
