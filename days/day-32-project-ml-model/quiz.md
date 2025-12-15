# Day 32 Quiz: Project - ML Model with Feature Store

## Knowledge Check: Production ML Platform

### 1. **What is the primary purpose of a feature store in production ML systems?**
   - A) Store raw data for analysis
   - B) Provide consistent, reusable features for training and serving
   - C) Cache model predictions
   - D) Store model artifacts

### 2. **Which ensemble method typically provides the best performance for credit risk modeling?**
   - A) Simple averaging
   - B) Weighted averaging based on individual model performance
   - C) Random selection
   - D) Majority voting

### 3. **What is the key advantage of using SHAP for model explainability in financial services?**
   - A) Faster predictions
   - B) Better model accuracy
   - C) Regulatory compliance through interpretable explanations
   - D) Reduced memory usage

### 4. **In fraud detection systems, what is the primary challenge with false positives?**
   - A) They increase computational costs
   - B) They block legitimate transactions, causing customer friction
   - C) They reduce model accuracy
   - D) They require more storage

### 5. **What is the recommended approach for handling model drift in production?**
   - A) Ignore it until accuracy drops significantly
   - B) Retrain models monthly regardless of performance
   - C) Monitor feature distributions and model performance continuously
   - D) Only retrain when customers complain

### 6. **Which metric is most important for evaluating recommendation systems?**
   - A) Accuracy
   - B) Precision@K (e.g., Precision@10)
   - C) F1-score
   - D) Mean Squared Error

### 7. **What is the primary benefit of using Docker containers for ML model deployment?**
   - A) Faster model training
   - B) Better model accuracy
   - C) Consistent environment across development and production
   - D) Reduced storage requirements

### 8. **In time series forecasting, what does MAPE measure?**
   - A) Model training time
   - B) Mean Absolute Percentage Error - forecast accuracy
   - C) Memory usage
   - D) Model complexity

### 9. **What is the key consideration when implementing real-time ML inference APIs?**
   - A) Model accuracy only
   - B) Latency, throughput, and reliability
   - C) Storage capacity
   - D) Training data size

### 10. **Which component is essential for regulatory compliance in financial ML systems?**
   - A) Faster GPUs
   - B) Larger datasets
   - C) Explainable AI and audit trails
   - D) More complex models

---

## Practical Application Questions

### 11. **If your credit risk model shows AUC = 0.87, what does this indicate?**
   - A) Poor performance, needs improvement
   - B) Good performance, suitable for production
   - C) Perfect performance
   - D) Model is overfitted

### 12. **What should you do if your fraud detection system has 95% recall but only 60% precision?**
   - A) Deploy immediately
   - B) Adjust threshold to reduce false positives
   - C) Increase recall further
   - D) Ignore precision

### 13. **In a feature store, what is the difference between online and offline stores?**
   - A) Online stores are faster, offline stores are for batch processing
   - B) Online stores are more accurate
   - C) Offline stores are more secure
   - D) No significant difference

### 14. **What is the recommended API response time for real-time fraud detection?**
   - A) < 5 seconds
   - B) < 1 second
   - C) < 100 milliseconds
   - D) < 10 milliseconds

### 15. **How should you handle missing features in production inference?**
   - A) Fail the request
   - B) Use default values or imputation strategies
   - C) Skip the prediction
   - D) Return random results

---

**Answers:**
1. B - Feature stores provide consistent, reusable features
2. B - Weighted averaging based on performance
3. C - Regulatory compliance through interpretability
4. B - Customer friction from blocked legitimate transactions
5. C - Continuous monitoring of distributions and performance
6. B - Precision@K measures recommendation quality
7. C - Consistent environments across deployments
8. B - Mean Absolute Percentage Error for forecast accuracy
9. B - Latency, throughput, and reliability are key
10. C - Explainable AI and audit trails for compliance
11. B - AUC 0.87 indicates good production-ready performance
12. B - Adjust threshold to balance precision and recall
13. A - Online for real-time, offline for batch processing
14. C - < 100ms for real-time fraud detection
15. B - Use robust imputation strategies
