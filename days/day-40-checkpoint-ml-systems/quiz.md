# Day 40 Quiz: ML Systems Review & Assessment

## Instructions
This comprehensive quiz covers all concepts from Phase 3: Advanced ML & MLOps (Days 25-39). Take your time and think through each question carefully. This assessment will help you identify areas for review before moving to Phase 4.

---

### 1. **What is the primary advantage of using a feature store like Feast in production ML systems?**
   - a) It reduces model training time by 50%
   - b) It provides centralized feature management with online/offline serving capabilities
   - c) It automatically generates new features using AutoML
   - d) It eliminates the need for data preprocessing

### 2. **In ensemble methods, what is the main difference between bagging and boosting?**
   - a) Bagging trains models sequentially, boosting trains them in parallel
   - b) Bagging uses different algorithms, boosting uses the same algorithm
   - c) Bagging trains models in parallel and combines predictions, boosting trains models sequentially to correct previous errors
   - d) Bagging is only for regression, boosting is only for classification

### 3. **What is the most critical metric to monitor for detecting data drift in production ML systems?**
   - a) Model accuracy on training data
   - b) Statistical distance between reference and current data distributions
   - c) Number of predictions made per day
   - d) API response time

### 4. **In A/B testing for ML models, what is the minimum sample size consideration primarily based on?**
   - a) The number of features in the model
   - b) The complexity of the model architecture
   - c) Statistical power analysis to detect meaningful differences
   - d) The cost of running the experiment

### 5. **What is the primary purpose of SHAP (SHapley Additive exPlanations) in production ML systems?**
   - a) To improve model accuracy through feature selection
   - b) To provide model-agnostic explanations for individual predictions
   - c) To automatically retrain models when performance degrades
   - d) To compress models for faster inference

### 6. **In DVC (Data Version Control), what is the main advantage of using remote storage for artifacts?**
   - a) It makes models train faster
   - b) It enables collaboration and reproducibility across team members
   - c) It automatically optimizes hyperparameters
   - d) It provides built-in model monitoring

### 7. **What is the recommended approach for handling model performance degradation in production?**
   - a) Immediately retrain the model with all available data
   - b) Investigate the root cause, check for data drift, then decide on retraining
   - c) Switch to a simpler model architecture
   - d) Increase the prediction confidence threshold

### 8. **In time series forecasting, what is the primary challenge that cross-validation addresses?**
   - a) Handling missing values in the time series
   - b) Preventing data leakage from future information
   - c) Reducing computational complexity
   - d) Improving model interpretability

### 9. **What is the key difference between collaborative filtering and content-based recommendation systems?**
   - a) Collaborative filtering uses user behavior patterns, content-based uses item features
   - b) Collaborative filtering is faster, content-based is more accurate
   - c) Collaborative filtering works only for new users, content-based for existing users
   - d) Collaborative filtering requires more computational resources

### 10. **In anomaly detection, what is the main advantage of using Isolation Forest over statistical methods?**
   - a) It requires less training data
   - b) It can detect anomalies in high-dimensional data without assuming data distribution
   - c) It provides better interpretability of results
   - d) It has lower computational complexity

---

## Answer Key

### 1. **What is the primary advantage of using a feature store like Feast in production ML systems?**
**Answer: b) It provides centralized feature management with online/offline serving capabilities**

**Explanation:** Feature stores like Feast solve the critical problem of feature consistency between training and serving environments. They provide centralized feature definitions, versioning, and both online (real-time) and offline (batch) serving capabilities. This ensures that the same features used during training are available during inference, eliminating training-serving skew. While feature stores may improve efficiency, their primary value is in providing consistent, reliable feature access across the ML lifecycle.

### 2. **In ensemble methods, what is the main difference between bagging and boosting?**
**Answer: c) Bagging trains models in parallel and combines predictions, boosting trains models sequentially to correct previous errors**

**Explanation:** Bagging (Bootstrap Aggregating) trains multiple models independently in parallel using different subsets of the training data, then combines their predictions (usually by averaging or voting). Examples include Random Forest. Boosting trains models sequentially, where each new model focuses on correcting the errors made by previous models. Examples include AdaBoost and Gradient Boosting. This sequential nature allows boosting to potentially achieve higher accuracy but makes it more prone to overfitting.

### 3. **What is the most critical metric to monitor for detecting data drift in production ML systems?**
**Answer: b) Statistical distance between reference and current data distributions**

**Explanation:** Data drift detection relies on comparing the statistical properties of current production data with reference data (typically training data). Metrics like Kolmogorov-Smirnov test, Population Stability Index (PSI), or Jensen-Shannon divergence measure the statistical distance between distributions. While model accuracy is important, it's a lagging indicator and may not immediately reflect data drift. Statistical distance metrics can detect drift before it significantly impacts model performance.

### 4. **In A/B testing for ML models, what is the minimum sample size consideration primarily based on?**
**Answer: c) Statistical power analysis to detect meaningful differences**

**Explanation:** Sample size in A/B testing is determined by statistical power analysis, which considers the minimum effect size you want to detect, the desired statistical power (typically 80%), and the significance level (typically 5%). This ensures you have enough data to reliably detect meaningful differences between models while controlling for Type I and Type II errors. The number of features or model complexity doesn't directly determine sample size requirements.

### 5. **What is the primary purpose of SHAP (SHapley Additive exPlanations) in production ML systems?**
**Answer: b) To provide model-agnostic explanations for individual predictions**

**Explanation:** SHAP provides a unified framework for explaining individual predictions by calculating the contribution of each feature to the prediction. It's model-agnostic, meaning it works with any ML model, and provides both local (individual prediction) and global (model behavior) explanations. This is crucial for regulatory compliance, debugging model behavior, and building trust with stakeholders. SHAP doesn't improve accuracy or provide monitoring capabilities directly.

### 6. **In DVC (Data Version Control), what is the main advantage of using remote storage for artifacts?**
**Answer: b) It enables collaboration and reproducibility across team members**

**Explanation:** DVC's remote storage capability allows teams to share large datasets and model artifacts without storing them in Git repositories. This enables collaboration by allowing team members to access the same versioned data and models, ensures reproducibility by maintaining exact artifact versions, and provides a centralized location for all ML artifacts. While it may have performance benefits, the primary advantage is enabling effective team collaboration and experiment reproducibility.

### 7. **What is the recommended approach for handling model performance degradation in production?**
**Answer: b) Investigate the root cause, check for data drift, then decide on retraining**

**Explanation:** When model performance degrades, the first step should be root cause analysis. This includes checking for data drift, data quality issues, changes in business processes, or external factors. Simply retraining without understanding the cause may not solve the problem and could waste resources. Once the root cause is identified, appropriate actions can be taken, which may include retraining, feature engineering, or addressing data quality issues.

### 8. **In time series forecasting, what is the primary challenge that cross-validation addresses?**
**Answer: b) Preventing data leakage from future information**

**Explanation:** Time series data has temporal dependencies, so traditional random cross-validation can cause data leakage by using future information to predict past events. Time series cross-validation (like time series split or walk-forward validation) ensures that models are only trained on past data and validated on future data, mimicking the real-world scenario where you predict future values based on historical data.

### 9. **What is the key difference between collaborative filtering and content-based recommendation systems?**
**Answer: a) Collaborative filtering uses user behavior patterns, content-based uses item features**

**Explanation:** Collaborative filtering makes recommendations based on user behavior patterns and similarities between users or items (e.g., "users who liked this also liked that"). Content-based filtering uses item features and user preferences to recommend similar items (e.g., "you liked action movies, here are more action movies"). Collaborative filtering can discover unexpected patterns but suffers from cold start problems, while content-based systems work well for new items but may lack diversity.

### 10. **In anomaly detection, what is the main advantage of using Isolation Forest over statistical methods?**
**Answer: b) It can detect anomalies in high-dimensional data without assuming data distribution**

**Explanation:** Isolation Forest is a tree-based algorithm that isolates anomalies by randomly selecting features and split values. It doesn't make assumptions about the underlying data distribution, making it effective for high-dimensional data where statistical methods may struggle. Statistical methods often assume specific distributions (like Gaussian) and may not work well when these assumptions are violated or in high-dimensional spaces where the curse of dimensionality affects performance.

---

## Scoring Guide

- **9-10 correct**: Excellent mastery of Phase 3 concepts - Ready for Phase 4
- **7-8 correct**: Good understanding - Review specific topics before Phase 4
- **5-6 correct**: Solid foundation - Recommend reviewing Days 35-39 before proceeding
- **3-4 correct**: Basic knowledge - Significant review of Phase 3 recommended
- **0-2 correct**: Comprehensive review of Phase 3 required before proceeding

## Areas for Review Based on Incorrect Answers

- **Questions 1, 6**: Review Day 25 (Feature Stores) and Day 35 (Model Versioning)
- **Questions 2, 10**: Review Day 30 (Ensemble Methods) and Day 28 (Anomaly Detection)
- **Questions 3, 7**: Review Day 37 (Feature Monitoring & Drift)
- **Questions 4, 8**: Review Day 34 (A/B Testing) and Day 27 (Time Series Forecasting)
- **Questions 5, 9**: Review Day 31 (Model Explainability) and Day 29 (Recommendation Systems)

Take time to review any areas where you scored incorrectly before proceeding to Phase 4. Strong foundational knowledge from Phase 3 will be essential for success with GenAI and LLM concepts.
