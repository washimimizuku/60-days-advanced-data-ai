# Day 25 Quiz: Feature Stores - Feast & Tecton

## Questions

### 1. What is the primary problem that feature stores solve in production ML systems?
- a) Reducing model training time by caching preprocessed data
- b) Eliminating training-serving skew by providing consistent feature computation
- c) Automatically selecting the best features for model performance
- d) Compressing feature data to reduce storage costs

### 2. In Feast, what is the purpose of the TTL (Time To Live) setting in a FeatureView?
- a) How long features are cached in memory during training
- b) How long feature data remains valid in the online store before expiring
- c) The maximum time allowed for feature computation
- d) How long to wait for feature materialization to complete

### 3. What is "point-in-time correctness" in the context of feature stores?
- a) Ensuring features are computed at exactly the same time across all models
- b) Guaranteeing that historical features reflect the data available at prediction time
- c) Synchronizing feature updates across distributed systems
- d) Validating that feature timestamps match the current system time

### 4. Which Feast component is responsible for serving features with sub-millisecond latency for real-time inference?
- a) Offline Store (S3/BigQuery)
- b) Feature Registry
- c) Online Store (Redis/DynamoDB)
- d) Batch Engine (Spark/Dask)

### 5. In a StreamFeatureView with aggregations, what does the time_window parameter control?
- a) How often the aggregation is computed
- b) The time range of data used to calculate the aggregated feature
- c) How long the aggregated result is cached
- d) The maximum delay allowed for streaming data

### 6. What is the main advantage of using on-demand feature views in Feast?
- a) They automatically materialize features to the online store
- b) They allow real-time feature transformations without pre-computation
- c) They provide better performance than batch feature views
- d) They eliminate the need for feature versioning

### 7. When implementing feature monitoring in production, what is the most effective approach for detecting feature drift?
- a) Monitor only the mean and standard deviation of features
- b) Use statistical tests like Kolmogorov-Smirnov to compare distributions
- c) Check if feature values exceed predefined thresholds
- d) Monitor only the number of null values in features

### 8. What is the recommended approach for handling feature versioning in a production feature store?
- a) Always overwrite existing features with new versions
- b) Use semantic versioning with backward compatibility and deprecation periods
- c) Create completely new feature names for each version
- d) Store all versions in the same feature view with version columns

### 9. In a ride-sharing scenario, which feature engineering pattern would be most appropriate for calculating "driver_performance_score"?
- a) Simple average of all driver metrics
- b) Weighted combination of acceptance rate, rating, and reliability metrics
- c) Maximum value among all performance indicators
- d) Random selection from available performance metrics

### 10. What is the most critical consideration when designing partition keys for streaming feature aggregations?
- a) Use sequential numbers to ensure ordered processing
- b) Always use random UUIDs to maximize parallelism
- c) Balance even distribution with logical grouping of related events
- d) Use only timestamp-based partitioning for consistency

---

## Answers

### 1. What is the primary problem that feature stores solve in production ML systems?
**Answer: b) Eliminating training-serving skew by providing consistent feature computation**

**Explanation:** The primary problem feature stores solve is training-serving skew, which occurs when features are computed differently during model training versus real-time inference. This leads to model performance degradation in production. Feature stores provide a single source of truth for feature definitions and computation, ensuring that the same feature engineering logic is used for both training (historical features) and serving (online features). While feature stores may improve training time and provide other benefits, their core value proposition is eliminating the inconsistency between training and serving environments that causes models to perform poorly in production despite good offline metrics.

---

### 2. In Feast, what is the purpose of the TTL (Time To Live) setting in a FeatureView?
**Answer: b) How long feature data remains valid in the online store before expiring**

**Explanation:** TTL (Time To Live) in Feast determines how long feature values remain valid in the online store before they are considered stale and need to be refreshed. This is crucial for maintaining data freshness in real-time serving scenarios. For example, a driver's location-based features might have a TTL of 5 minutes, while user preference features might have a TTL of 30 days. The TTL setting helps balance between serving fresh data and avoiding unnecessary recomputation. When features exceed their TTL, Feast will either refresh them from the offline store or return null values, depending on the configuration.

---

### 3. What is "point-in-time correctness" in the context of feature stores?
**Answer: b) Guaranteeing that historical features reflect the data available at prediction time**

**Explanation:** Point-in-time correctness ensures that when retrieving historical features for training, you only use data that was actually available at the time of the prediction event. This prevents data leakage where future information accidentally influences model training. For example, when training a model to predict ride demand at 2 PM on Monday, you should only use features computed from data available before 2 PM on Monday, not data from later that day. Feast implements this by using event timestamps and created timestamps to ensure temporal consistency, which is essential for building reliable ML models that will perform well in production.

---

### 4. Which Feast component is responsible for serving features with sub-millisecond latency for real-time inference?
**Answer: c) Online Store (Redis/DynamoDB)**

**Explanation:** The Online Store is specifically designed for low-latency feature serving in real-time inference scenarios. It uses fast key-value stores like Redis or DynamoDB that can serve features in sub-millisecond to single-digit millisecond latency. The Offline Store (S3/BigQuery) is optimized for batch processing and historical feature retrieval for training, which can take seconds or minutes. The Feature Registry stores metadata about features, and the Batch Engine handles large-scale feature computation. Only the Online Store is architected and optimized for the ultra-low latency requirements of real-time ML inference.

---

### 5. In a StreamFeatureView with aggregations, what does the time_window parameter control?
**Answer: b) The time range of data used to calculate the aggregated feature**

**Explanation:** The time_window parameter in StreamFeatureView aggregations defines the sliding window of time over which the aggregation is computed. For example, if you set time_window=timedelta(hours=1) for a "count" aggregation, it will count events in the last 1 hour from the current time. This creates features like "rides_completed_last_1h" or "avg_rating_last_4h". The time window slides continuously as new events arrive, providing real-time aggregated features. This is different from how often the computation runs (which is determined by the streaming engine) or caching duration (which is controlled by TTL).

---

### 6. What is the main advantage of using on-demand feature views in Feast?
**Answer: b) They allow real-time feature transformations without pre-computation**

**Explanation:** On-demand feature views enable real-time feature transformations that are computed at request time rather than being pre-materialized. This is valuable for features that depend on multiple inputs, require complex business logic, or need to be computed with the most current data. For example, calculating a dynamic surge multiplier based on current demand and supply, or computing a risk score that combines multiple feature views. While this adds some latency compared to pre-computed features, it provides flexibility for complex transformations and ensures the most up-to-date calculations without requiring constant re-materialization of all possible feature combinations.

---

### 7. When implementing feature monitoring in production, what is the most effective approach for detecting feature drift?
**Answer: b) Use statistical tests like Kolmogorov-Smirnov to compare distributions**

**Explanation:** Statistical tests like Kolmogorov-Smirnov (KS test), Chi-square, or Jensen-Shannon divergence are the most effective methods for detecting feature drift because they compare entire distributions rather than just summary statistics. Feature drift often manifests as changes in the shape, spread, or modality of distributions that wouldn't be caught by monitoring only mean and standard deviation. The KS test, for example, can detect shifts in distribution location, scale, or shape. Simple threshold monitoring might miss subtle but important distributional changes, while statistical tests provide a principled approach with p-values that can be used to set appropriate alerting thresholds.

---

### 8. What is the recommended approach for handling feature versioning in a production feature store?
**Answer: b) Use semantic versioning with backward compatibility and deprecation periods**

**Explanation:** Proper feature versioning requires semantic versioning (v1.0, v1.1, v2.0) with careful attention to backward compatibility and planned deprecation periods. This allows multiple model versions to coexist during deployment transitions and gives teams time to migrate to new feature versions. Breaking changes should increment major versions, while backward-compatible improvements use minor versions. Deprecation periods (e.g., 6 months) give downstream consumers time to migrate. This approach prevents production outages when features evolve and enables safe, gradual rollouts of feature improvements across multiple ML models and teams.

---

### 9. In a ride-sharing scenario, which feature engineering pattern would be most appropriate for calculating "driver_performance_score"?
**Answer: b) Weighted combination of acceptance rate, rating, and reliability metrics**

**Explanation:** A weighted combination approach allows you to create a composite score that reflects the relative importance of different performance aspects. For example: (acceptance_rate × 0.3) + (avg_rating/5 × 0.4) + ((1-cancellation_rate) × 0.3). This approach is interpretable, allows for business logic to be encoded in the weights, and can be tuned based on business priorities. Simple averages don't account for the different scales and importance of metrics, while maximum values or random selection don't provide meaningful composite scores. The weighted approach also allows for easy adjustment as business priorities change and provides a single, actionable score for decision-making.

---

### 10. What is the most critical consideration when designing partition keys for streaming feature aggregations?
**Answer: c) Balance even distribution with logical grouping of related events**

**Explanation:** Effective partition key design for streaming aggregations requires balancing two competing needs: even distribution across partitions to maximize parallelism and throughput, while also ensuring that related events that need to be aggregated together end up in the same partition. For example, using a hash of driver_id ensures all events for a driver go to the same partition (enabling proper aggregation) while distributing drivers evenly across partitions. Pure sequential keys create hot partitions, pure random keys prevent proper grouping, and timestamp-only partitioning can create temporal hotspots. The key is finding attributes that provide both logical grouping and good distribution characteristics.