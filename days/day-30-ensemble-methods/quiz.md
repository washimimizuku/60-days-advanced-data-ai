# Day 30 Quiz: Ensemble Methods - Bagging, Boosting, Stacking

Test your understanding of ensemble methods and their production applications.

---

## Questions

### 1. What is the main principle behind ensemble methods?
- a) Using the most complex model available
- b) Combining multiple models to create a stronger predictor than any individual model
- c) Always using deep learning approaches
- d) Focusing only on linear models for interpretability

### 2. In Random Forest, what does the "random" refer to?
- a) Random selection of training samples only
- b) Random initialization of model weights
- c) Random selection of both training samples (bootstrap) and features for each split
- d) Random hyperparameter selection

### 3. What is the key difference between bagging and boosting?
- a) Bagging trains models sequentially, boosting trains in parallel
- b) Bagging trains models in parallel on different data subsets, boosting trains sequentially where each model learns from previous errors
- c) Bagging only works with tree models, boosting works with any model
- d) There is no significant difference between them

### 4. In stacking ensemble, what is the role of the meta-model?
- a) To replace all base models
- b) To learn how to best combine predictions from base models
- c) To generate new features for base models
- d) To validate the performance of base models

### 5. Which voting method typically performs better when base models output reliable probability estimates?
- a) Hard voting (majority vote on predicted classes)
- b) Soft voting (averaging predicted probabilities)
- c) Random voting
- d) Weighted voting based on model complexity

### 6. What is a key advantage of using diverse base models in an ensemble?
- a) Faster training time
- b) Lower memory usage
- c) Different models make different types of errors, so combining them reduces overall error
- d) Simpler hyperparameter tuning

### 7. In production ensemble systems, what is prediction drift?
- a) Models becoming slower over time
- b) Changes in the distribution of model predictions compared to baseline
- c) Hardware performance degradation
- d) Increasing model complexity

### 8. What is the main challenge with ensemble methods in real-time serving?
- a) Memory usage
- b) Model interpretability
- c) Increased latency due to multiple model predictions
- d) Data preprocessing complexity

### 9. Which ensemble method is most likely to overfit when base models are already complex?
- a) Simple voting
- b) Bagging with high diversity
- c) Stacking with many layers and complex meta-models
- d) Random Forest with few trees

### 10. What is the best practice for cross-validation in stacking ensembles?
- a) Use the same folds for all base models
- b) Use different random folds for each base model
- c) Use stratified k-fold to generate meta-features and avoid data leakage
- d) Skip cross-validation and use holdout validation

---

## Answers

### 1. What is the main principle behind ensemble methods?
**Answer: b) Combining multiple models to create a stronger predictor than any individual model**

**Explanation:** Ensemble methods are based on the "wisdom of crowds" principle - by combining predictions from multiple diverse models, we can achieve better performance than any single model. This works because different models make different types of errors, and when combined properly, these errors can cancel out, leading to more robust and accurate predictions.

### 2. In Random Forest, what does the "random" refer to?
**Answer: c) Random selection of both training samples (bootstrap) and features for each split**

**Explanation:** Random Forest introduces randomness in two ways: (1) Bootstrap sampling - each tree is trained on a random subset of training samples with replacement, and (2) Feature randomness - at each split in each tree, only a random subset of features is considered. This dual randomness creates diverse trees that, when combined, form a robust ensemble.

### 3. What is the key difference between bagging and boosting?
**Answer: b) Bagging trains models in parallel on different data subsets, boosting trains sequentially where each model learns from previous errors**

**Explanation:** Bagging (Bootstrap Aggregating) trains multiple models independently in parallel, each on a different bootstrap sample of the data, then combines their predictions. Boosting trains models sequentially, where each new model focuses on correcting the errors made by previous models. This fundamental difference affects their bias-variance tradeoff and computational requirements.

### 4. In stacking ensemble, what is the role of the meta-model?
**Answer: b) To learn how to best combine predictions from base models**

**Explanation:** The meta-model (or blender) in stacking learns the optimal way to combine predictions from base models. It's trained on the out-of-fold predictions from base models (generated through cross-validation) and learns patterns about when each base model performs well, enabling it to make intelligent decisions about how to weight and combine their predictions.

### 5. Which voting method typically performs better when base models output reliable probability estimates?
**Answer: b) Soft voting (averaging predicted probabilities)**

**Explanation:** Soft voting averages the predicted probabilities from base models before making the final prediction, which preserves more information than hard voting (which only uses the final class predictions). When base models provide well-calibrated probability estimates, soft voting can capture the confidence levels of predictions and typically leads to better performance.

### 6. What is a key advantage of using diverse base models in an ensemble?
**Answer: c) Different models make different types of errors, so combining them reduces overall error**

**Explanation:** Model diversity is crucial for ensemble success. When base models are diverse (different algorithms, different feature subsets, different hyperparameters), they tend to make errors on different examples. By combining diverse models, the ensemble can correct individual model errors and achieve better generalization than any single model.

### 7. In production ensemble systems, what is prediction drift?
**Answer: b) Changes in the distribution of model predictions compared to baseline**

**Explanation:** Prediction drift occurs when the distribution of model outputs changes significantly from the baseline established during training/validation. This can indicate data drift, model degradation, or changes in the underlying patterns. Monitoring prediction drift is crucial for maintaining model reliability in production environments.

### 8. What is the main challenge with ensemble methods in real-time serving?
**Answer: c) Increased latency due to multiple model predictions**

**Explanation:** Ensemble methods require predictions from multiple base models, which increases computational overhead and latency. In real-time applications where response time is critical, this can be a significant challenge. Solutions include model optimization, parallel prediction, caching, and careful selection of ensemble complexity based on latency requirements.

### 9. Which ensemble method is most likely to overfit when base models are already complex?
**Answer: c) Stacking with many layers and complex meta-models**

**Explanation:** Stacking with multiple layers and complex meta-models can easily overfit, especially when base models are already complex. The meta-model might learn to memorize the training data patterns rather than generalize. Simple meta-models (like logistic regression) and proper cross-validation are essential to prevent overfitting in stacking ensembles.

### 10. What is the best practice for cross-validation in stacking ensembles?
**Answer: c) Use stratified k-fold to generate meta-features and avoid data leakage**

**Explanation:** Proper cross-validation in stacking is crucial to avoid data leakage. Stratified k-fold ensures that each fold maintains the same class distribution as the original dataset. The meta-features should be generated using out-of-fold predictions from base models trained on the remaining folds, ensuring that the meta-model never sees predictions from models trained on the same data it will be evaluated on.
