# Day 37 Quiz: Feature Monitoring & Drift - Data Quality, Model Performance

## Questions

### 1. What is the Population Stability Index (PSI) primarily used for in ML monitoring?
- a) Measuring model accuracy over time
- b) Detecting distribution changes in feature values between reference and current data
- c) Optimizing hyperparameters automatically
- d) Calculating feature importance scores

### 2. Which PSI score range typically indicates "major change" that requires investigation?
- a) 0.0 - 0.1
- b) 0.1 - 0.2
- c) 0.2 - 0.5
- d) Above 0.5

### 3. What is the main advantage of using the Kolmogorov-Smirnov test for drift detection?
- a) It works only with categorical features
- b) It provides a statistical p-value to determine significance of distribution differences
- c) It requires no reference data
- d) It automatically triggers model retraining

### 4. In ML-based drift detection using domain classifiers, what does a high AUC score indicate?
- a) The model is performing well on the original task
- b) There is significant drift between reference and current data
- c) The features are highly correlated
- d) No drift has been detected

### 5. What is the recommended approach for real-time drift monitoring in production systems?
- a) Check for drift once per day using batch processing
- b) Use sliding windows with streaming data and configurable alert thresholds
- c) Only monitor drift when model performance drops
- d) Manually inspect data distributions weekly

### 6. Which retraining strategy is most appropriate when severe drift is detected (PSI > 0.5)?
- a) No action needed
- b) Incremental model update
- c) Feature selection retraining
- d) Full model retraining

### 7. What is concept drift in machine learning?
- a) Changes in feature distributions while target relationships remain stable
- b) Changes in the relationship between features and target variable over time
- c) Hardware performance degradation
- d) Database schema modifications

### 8. Which statistical test is most appropriate for detecting drift in categorical features?
- a) Kolmogorov-Smirnov test
- b) Chi-square test
- c) T-test
- d) ANOVA

### 9. What is the primary benefit of automated retraining pipelines in production ML systems?
- a) Reduced computational costs
- b) Elimination of all manual intervention
- c) Rapid response to drift with appropriate retraining strategies based on severity
- d) Guaranteed model performance improvement

### 10. When implementing feature monitoring dashboards, which metrics are most critical to display?
- a) Only model accuracy trends
- b) Drift scores, distribution comparisons, alert history, and feature statistics
- c) Hardware utilization metrics only
- d) Database query performance

## Answers

### 1. What is the Population Stability Index (PSI) primarily used for in ML monitoring?
**Answer: b) Detecting distribution changes in feature values between reference and current data**

**Explanation:** PSI is specifically designed to measure the stability of feature distributions by comparing current data against a reference baseline. It quantifies how much the distribution of a feature has shifted over time by binning the data and comparing proportions between reference and current periods. PSI values help identify when features are experiencing drift that might affect model performance. Options a, c, and d describe other ML monitoring concepts but not PSI's primary purpose.

### 2. Which PSI score range typically indicates "major change" that requires investigation?
**Answer: c) 0.2 - 0.5**

**Explanation:** PSI interpretation follows standard thresholds: 0.0-0.1 indicates no significant change, 0.1-0.2 indicates minor change, 0.2-0.5 indicates major change requiring investigation, and above 0.5 indicates severe change requiring immediate action. The 0.2-0.5 range represents substantial distribution shifts that could impact model performance and warrant detailed analysis and potential retraining. These thresholds are industry-standard guidelines for PSI interpretation.

### 3. What is the main advantage of using the Kolmogorov-Smirnov test for drift detection?
**Answer: b) It provides a statistical p-value to determine significance of distribution differences**

**Explanation:** The KS test is a non-parametric statistical test that compares two distributions and provides a p-value indicating the statistical significance of observed differences. This allows for objective, statistically-grounded decisions about whether drift has occurred based on chosen significance levels (e.g., p < 0.05). Unlike heuristic measures, the KS test provides rigorous statistical inference. Option a is incorrect as KS test works with continuous/numeric features, options c and d describe features not specific to KS testing.

### 4. In ML-based drift detection using domain classifiers, what does a high AUC score indicate?
**Answer: b) There is significant drift between reference and current data**

**Explanation:** Domain classifiers are trained to distinguish between reference data (label 0) and current data (label 1). A high AUC score (close to 1.0) means the classifier can easily distinguish between the two datasets, indicating significant differences in their distributions - i.e., drift has occurred. An AUC around 0.5 would indicate the datasets are indistinguishable (no drift), while high AUC indicates clear distributional differences. This approach treats drift detection as a binary classification problem.

### 5. What is the recommended approach for real-time drift monitoring in production systems?
**Answer: b) Use sliding windows with streaming data and configurable alert thresholds**

**Explanation:** Real-time drift monitoring requires continuous processing of incoming data using sliding windows to maintain recent data samples for comparison against reference distributions. Configurable alert thresholds allow for different sensitivity levels based on business requirements and feature criticality. This approach enables rapid detection and response to drift as it occurs. Options a and d involve too much delay, while option c is reactive rather than proactive.

### 6. Which retraining strategy is most appropriate when severe drift is detected (PSI > 0.5)?
**Answer: d) Full model retraining**

**Explanation:** Severe drift (PSI > 0.5) indicates fundamental changes in data distributions that likely require complete model retraining with new data to maintain performance. Incremental updates or feature selection may be insufficient for such significant changes. Full retraining ensures the model learns the new data patterns completely. The severity of drift should guide retraining strategy: minor drift might need incremental updates, moderate drift might need feature selection, but severe drift typically requires full retraining.

### 7. What is concept drift in machine learning?
**Answer: b) Changes in the relationship between features and target variable over time**

**Explanation:** Concept drift occurs when the underlying relationship between input features and the target variable changes over time, even if feature distributions remain stable. This is different from data drift (option a), which involves changes in feature distributions while relationships remain constant. Concept drift is often harder to detect but more critical as it directly affects model predictions. Options c and d describe technical issues unrelated to ML concept drift.

### 8. Which statistical test is most appropriate for detecting drift in categorical features?
**Answer: b) Chi-square test**

**Explanation:** The chi-square test is designed for categorical data and tests whether the distribution of categories has changed significantly between reference and current data. It compares observed vs. expected frequencies across categories. The KS test (option a) is for continuous data, while t-test and ANOVA (options c and d) are for comparing means of continuous variables, not categorical distributions.

### 9. What is the primary benefit of automated retraining pipelines in production ML systems?
**Answer: c) Rapid response to drift with appropriate retraining strategies based on severity**

**Explanation:** Automated retraining pipelines enable immediate, intelligent responses to detected drift by selecting appropriate retraining strategies (full retrain, incremental update, feature selection) based on drift severity and type. This maintains model performance without manual intervention delays. While cost reduction (option a) may occur, the primary benefit is maintaining model effectiveness. Option b is unrealistic as some manual oversight is always needed, and option d cannot be guaranteed.

### 10. When implementing feature monitoring dashboards, which metrics are most critical to display?
**Answer: b) Drift scores, distribution comparisons, alert history, and feature statistics**

**Explanation:** Effective feature monitoring dashboards must show drift-specific metrics: drift scores (PSI, KS statistics) to quantify changes, distribution comparisons to visualize shifts, alert history to track drift events, and feature statistics to understand the nature of changes. These metrics directly relate to data quality and model health. Options a, c, and d focus on single aspects (model performance, hardware, database) but miss the comprehensive view needed for feature drift monitoring.