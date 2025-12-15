# Day 28 Quiz: Anomaly Detection - Statistical & ML-based Methods

## Questions

### 1. What is the main advantage of using Modified Z-score (with MAD) over standard Z-score for anomaly detection?
- a) Modified Z-score is computationally faster than standard Z-score
- b) Modified Z-score is more robust to outliers because it uses median and MAD instead of mean and standard deviation
- c) Modified Z-score works better with normally distributed data
- d) Modified Z-score can detect seasonal anomalies automatically

### 2. In Isolation Forest, why are anomalies easier to isolate than normal points?
- a) Anomalies are always located at the edges of the data distribution
- b) Anomalies are few and different, requiring fewer random splits to separate them from the majority
- c) Anomalies have higher feature values than normal points
- d) Anomalies follow a different probability distribution

### 3. What is the primary purpose of using autoencoders for anomaly detection?
- a) To classify data points into predefined anomaly categories
- b) To learn a compressed representation and detect anomalies based on high reconstruction error
- c) To generate synthetic anomalous data for training
- d) To reduce the dimensionality of the input data

### 4. In time series anomaly detection, what does Statistical Process Control (SPC) monitor?
- a) The correlation between different time series
- b) Whether data points fall outside dynamically calculated control limits based on recent statistics
- c) The frequency domain characteristics of the signal
- d) The seasonal patterns in the data

### 5. When using ensemble anomaly detection, what is the advantage of weighted voting over simple majority voting?
- a) Weighted voting is computationally more efficient
- b) Weighted voting allows better-performing detectors to have more influence on the final decision
- c) Weighted voting eliminates the need for individual detector validation
- d) Weighted voting automatically handles concept drift

### 6. What is concept drift in the context of real-time anomaly detection?
- a) The gradual degradation of model performance over time
- b) Changes in the underlying data distribution that make existing models less effective
- c) The tendency of anomaly detection systems to generate more false positives
- d) The computational drift caused by processing streaming data

### 7. Why is the contamination parameter important in Isolation Forest?
- a) It determines the number of trees in the forest
- b) It sets the expected proportion of anomalies in the training data, affecting the decision threshold
- c) It controls the maximum depth of each tree
- d) It specifies the number of features to use for each split

### 8. In One-Class SVM, what does the nu parameter control?
- a) The kernel bandwidth for RBF kernels
- b) The upper bound on the fraction of training errors and lower bound on the fraction of support vectors
- c) The regularization strength of the model
- d) The number of support vectors to use

### 9. What is the main challenge when evaluating anomaly detection systems in production?
- a) Anomaly detection algorithms are too slow for real-time evaluation
- b) True anomaly labels are often unavailable or delayed, making immediate performance assessment difficult
- c) Anomaly detection systems cannot handle high-dimensional data
- d) Statistical methods always outperform machine learning methods

### 10. When implementing anomaly detection for fraud detection in financial systems, what is the most critical consideration?
- a) Using the most complex machine learning algorithm available
- b) Balancing false positive rates (blocking legitimate transactions) with false negative rates (missing fraud)
- c) Processing transactions as quickly as possible regardless of accuracy
- d) Using only statistical methods for interpretability

---

## Answers

### 1. What is the main advantage of using Modified Z-score (with MAD) over standard Z-score for anomaly detection?
**Answer: b) Modified Z-score is more robust to outliers because it uses median and MAD instead of mean and standard deviation**

**Explanation:** The Modified Z-score uses the median and Median Absolute Deviation (MAD) instead of the mean and standard deviation used in the standard Z-score. This makes it much more robust to outliers because the median and MAD are less sensitive to extreme values than the mean and standard deviation. In standard Z-score, a few extreme outliers can significantly shift the mean and inflate the standard deviation, making it harder to detect other anomalies. The Modified Z-score maintains stable statistics even in the presence of outliers, making it more reliable for anomaly detection in real-world data that often contains existing anomalies.

---

### 2. In Isolation Forest, why are anomalies easier to isolate than normal points?
**Answer: b) Anomalies are few and different, requiring fewer random splits to separate them from the majority**

**Explanation:** Isolation Forest works on the principle that anomalies are "few and different." Because anomalies are rare and have different characteristics from the majority of the data, they can be separated from the rest of the data with fewer random splits in the decision tree. Normal points, being similar to each other and forming dense clusters, require more splits to isolate individual points. The algorithm measures the path length (number of splits) needed to isolate each point - shorter paths indicate anomalies because they were easier to separate. This fundamental insight makes Isolation Forest effective without requiring assumptions about data distribution or density.

---

### 3. What is the primary purpose of using autoencoders for anomaly detection?
**Answer: b) To learn a compressed representation and detect anomalies based on high reconstruction error**

**Explanation:** Autoencoders for anomaly detection work by learning to compress (encode) and then reconstruct (decode) normal data. During training, the autoencoder learns the patterns and structure of normal data. When presented with new data, normal points can be reconstructed accurately (low reconstruction error), while anomalous points, which differ from the learned normal patterns, will have high reconstruction error. The reconstruction error serves as an anomaly score - points with reconstruction error above a threshold are classified as anomalies. This approach is particularly effective for high-dimensional data where traditional statistical methods may struggle.

---

### 4. In time series anomaly detection, what does Statistical Process Control (SPC) monitor?
**Answer: b) Whether data points fall outside dynamically calculated control limits based on recent statistics**

**Explanation:** Statistical Process Control (SPC) monitors whether individual data points fall outside control limits that are dynamically calculated based on recent historical statistics. Typically, SPC calculates rolling mean and standard deviation over a sliding window, then sets upper and lower control limits (usually mean ± 3 standard deviations). Points that fall outside these limits are flagged as anomalies. The "control" aspect refers to monitoring whether the process remains in statistical control - i.e., behaving according to its normal statistical patterns. This method is particularly useful for detecting when a time series process has gone out of its normal operating range.

---

### 5. When using ensemble anomaly detection, what is the advantage of weighted voting over simple majority voting?
**Answer: b) Weighted voting allows better-performing detectors to have more influence on the final decision**

**Explanation:** Weighted voting in ensemble anomaly detection allows detectors with better historical performance to have more influence on the final decision. While simple majority voting treats all detectors equally (each gets one vote), weighted voting assigns different weights based on factors like accuracy, precision, recall, or other performance metrics. This means a highly accurate detector might have a weight of 0.4 while a less reliable detector has a weight of 0.1. The final decision is based on the weighted sum of votes rather than simple counting. This approach typically leads to better overall ensemble performance because it leverages the strengths of the best detectors while minimizing the impact of weaker ones.

---

### 6. What is concept drift in the context of real-time anomaly detection?
**Answer: b) Changes in the underlying data distribution that make existing models less effective**

**Explanation:** Concept drift refers to changes in the underlying statistical properties of the data over time, which can make existing anomaly detection models less effective or even obsolete. For example, in fraud detection, fraudsters continuously evolve their tactics, changing the patterns of fraudulent behavior. In system monitoring, normal system behavior might change due to software updates, hardware changes, or usage pattern shifts. When concept drift occurs, what was previously considered normal might now be anomalous, and vice versa. This requires anomaly detection systems to adapt by retraining models, updating thresholds, or implementing adaptive algorithms that can detect and respond to these distributional changes.

---

### 7. Why is the contamination parameter important in Isolation Forest?
**Answer: b) It sets the expected proportion of anomalies in the training data, affecting the decision threshold**

**Explanation:** The contamination parameter in Isolation Forest specifies the expected proportion of anomalies in the training data, which directly affects the decision threshold for classifying points as anomalies. For example, if contamination=0.1, the algorithm expects 10% of the data to be anomalous and will set the threshold such that approximately 10% of points with the shortest average path lengths are classified as anomalies. This parameter is crucial because it determines how "strict" the anomaly detection will be. Setting it too low might miss real anomalies, while setting it too high might flag too many normal points as anomalies. The parameter should be set based on domain knowledge about the expected anomaly rate in the specific application.

---

### 8. In One-Class SVM, what does the nu parameter control?
**Answer: b) The upper bound on the fraction of training errors and lower bound on the fraction of support vectors**

**Explanation:** The nu parameter in One-Class SVM serves a dual purpose: it provides an upper bound on the fraction of training errors (outliers) and a lower bound on the fraction of support vectors. Essentially, nu controls the trade-off between the volume of the region that contains most of the data and the number of data points outside this region. A smaller nu value creates a tighter boundary around the normal data (fewer training errors allowed but potentially more support vectors), while a larger nu value creates a looser boundary (more training errors allowed). Typically, nu is set between 0 and 1, with values like 0.05 or 0.1 being common, meaning you expect about 5% or 10% of the training data to be outliers.

---

### 9. What is the main challenge when evaluating anomaly detection systems in production?
**Answer: b) True anomaly labels are often unavailable or delayed, making immediate performance assessment difficult**

**Explanation:** The primary challenge in evaluating production anomaly detection systems is the lack of immediate ground truth labels. In many real-world scenarios, determining whether a flagged point is truly anomalous requires human investigation, domain expert analysis, or waiting for consequences to manifest. For example, in fraud detection, confirming whether a transaction is fraudulent might require customer contact or investigation. In system monitoring, determining if an alert represents a real issue might require detailed analysis. This delay in obtaining true labels makes it difficult to immediately assess system performance, tune parameters, or detect when the system is degrading. This is why unsupervised evaluation metrics and business impact metrics often become more important than traditional supervised learning metrics.

---

### 10. When implementing anomaly detection for fraud detection in financial systems, what is the most critical consideration?
**Answer: b) Balancing false positive rates (blocking legitimate transactions) with false negative rates (missing fraud)**

**Explanation:** In financial fraud detection, the most critical consideration is balancing false positives (legitimate transactions incorrectly flagged as fraud) with false negatives (fraudulent transactions that go undetected). False positives create customer friction, potentially blocking legitimate purchases and causing customer dissatisfaction, while false negatives result in financial losses and regulatory issues. The business impact of each type of error must be carefully considered - blocking a customer's legitimate $50 purchase has different consequences than missing a $10,000 fraudulent transaction. This balance is typically achieved through careful threshold tuning, cost-sensitive learning, or multi-tier alert systems where high-confidence anomalies are automatically blocked while medium-confidence ones are flagged for review. The optimal balance depends on the specific business context, customer tolerance, and regulatory requirements.