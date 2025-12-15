# Day 33: Model Serving at Scale - Quiz

## Questions

### 1. Which serving pattern is most appropriate for processing 10 million transactions daily for compliance reporting?
- a) Real-time REST API serving
- b) Batch processing with scheduled jobs
- c) Streaming processing with Kafka
- d) Synchronous database queries

**Answer: b) Batch processing with scheduled jobs**

**Explanation:** Batch processing is ideal for large-scale, non-time-sensitive workloads like compliance reporting. It's cost-effective, can handle massive datasets efficiently, and doesn't require real-time responses. REST APIs would be too expensive for 10M requests, streaming is overkill for daily reporting, and synchronous database queries would be too slow and resource-intensive.

### 2. What is the primary benefit of implementing multi-level caching in model serving?
- a) Reducing model training time
- b) Improving data quality
- c) Minimizing prediction latency and computational costs
- d) Increasing model accuracy

**Answer: c) Minimizing prediction latency and computational costs**

**Explanation:** Multi-level caching (L1 memory cache + L2 Redis cache) significantly reduces prediction latency by serving frequently requested predictions from cache instead of recomputing them. This also reduces computational costs and server load. Caching doesn't affect training time, data quality, or model accuracy - it's purely a performance optimization for serving.

### 3. In a Kubernetes deployment, which metric is most important for configuring Horizontal Pod Autoscaler (HPA) for ML model serving?
- a) Model accuracy
- b) CPU utilization and response time
- c) Training data size
- d) Number of features

**Answer: b) CPU utilization and response time**

**Explanation:** HPA should scale based on resource utilization (CPU/memory) and performance metrics (response time) to maintain service quality under varying load. Model accuracy, training data size, and number of features are static characteristics that don't change with request volume and aren't suitable for auto-scaling decisions.

### 4. What is the main advantage of using ONNX format for model serving?
- a) Better model accuracy
- b) Cross-platform optimization and faster inference
- c) Easier model training
- d) Automatic feature engineering

**Answer: b) Cross-platform optimization and faster inference**

**Explanation:** ONNX (Open Neural Network Exchange) provides optimized inference engines that can run models faster across different platforms and hardware. It includes optimizations like operator fusion, memory layout optimization, and hardware-specific acceleration. ONNX doesn't improve accuracy, training ease, or provide feature engineering - it's specifically for inference optimization.

### 5. Which pattern best handles model serving failures and maintains system reliability?
- a) Retry all failed requests immediately
- b) Circuit breaker pattern with graceful degradation
- c) Ignore failed requests
- d) Restart the entire system

**Answer: b) Circuit breaker pattern with graceful degradation**

**Explanation:** Circuit breaker pattern prevents cascading failures by temporarily stopping requests to failing services and providing fallback responses. Graceful degradation maintains partial functionality when components fail. Immediate retries can worsen failures, ignoring requests loses data, and restarting the system causes unnecessary downtime.

### 6. For real-time fraud detection requiring <50ms latency, which caching strategy is most effective?
- a) Database-only caching
- b) File-based caching
- c) In-memory cache with Redis fallback
- d) No caching to ensure fresh predictions

**Answer: c) In-memory cache with Redis fallback**

**Explanation:** Multi-level caching with L1 (in-memory) and L2 (Redis) provides the fastest access times. In-memory cache offers microsecond access for hot data, while Redis provides fast distributed caching for broader data. Database caching is too slow, file-based caching has high I/O overhead, and no caching would require recomputing every prediction, making <50ms latency impossible.

### 7. What is the most critical consideration when implementing streaming model serving with Kafka?
- a) Message ordering and exactly-once processing
- b) Model training frequency
- c) Database schema design
- d) User interface design

**Answer: a) Message ordering and exactly-once processing**

**Explanation:** In streaming systems, maintaining message ordering and ensuring exactly-once processing is crucial for data consistency and preventing duplicate predictions. This is especially important for financial transactions where order matters and duplicates can cause serious issues. Model training frequency, database schema, and UI design are important but not critical for streaming reliability.

### 8. Which monitoring metric is most important for detecting model serving performance degradation?
- a) Number of features used
- b) P95 response time and error rate
- c) Model file size
- d) Training dataset size

**Answer: b) P95 response time and error rate**

**Explanation:** P95 response time shows how the system performs for most users (95th percentile captures tail latency), and error rate indicates system reliability. These metrics directly reflect user experience and system health. Feature count, model file size, and training dataset size are static characteristics that don't indicate runtime performance issues.

### 9. What is the primary benefit of using async/await patterns in FastAPI for model serving?
- a) Improved model accuracy
- b) Better handling of concurrent requests without blocking
- c) Faster model training
- d) Automatic data validation

**Answer: b) Better handling of concurrent requests without blocking**

**Explanation:** Async/await allows the server to handle multiple requests concurrently without blocking threads, significantly improving throughput and resource utilization. While one request waits for I/O (database, cache, external API), the server can process other requests. This doesn't affect model accuracy, training speed, or provide data validation - it's purely for concurrency performance.

### 10. For batch processing 100GB of transaction data, which approach provides the best performance?
- a) Single-threaded sequential processing
- b) Distributed processing with data partitioning
- c) Loading all data into memory at once
- d) Processing one record at a time

**Answer: b) Distributed processing with data partitioning**

**Explanation:** Distributed processing (like Apache Spark) with data partitioning allows parallel processing across multiple nodes, dramatically reducing processing time for large datasets. It handles data that doesn't fit in memory and provides fault tolerance. Single-threaded processing is too slow, loading 100GB into memory is often impossible, and processing one record at a time doesn't leverage parallelism.