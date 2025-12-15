# Day 36 Quiz: CI/CD for ML - Automated ML Pipelines, Testing, Infrastructure as Code

## Questions

### 1. What is the primary benefit of implementing CI/CD for ML pipelines?
- a) Reduced model accuracy requirements
- b) Automated testing, deployment, and rollback capabilities
- c) Elimination of data validation needs
- d) Simplified model architecture requirements

### 2. In a blue-green deployment strategy for ML models, what happens during the traffic switch phase?
- a) Both blue and green services run simultaneously with 50/50 traffic
- b) The green service is immediately shut down after deployment
- c) Traffic is gradually moved from blue to green after health checks pass
- d) The blue service handles all traffic while green is being tested

### 3. Which testing strategy is most critical for ML model validation in CI/CD pipelines?
- a) Only unit tests for code functionality
- b) Comprehensive testing including model performance, robustness, and data validation
- c) Manual testing by data scientists only
- d) Load testing without model-specific validation

### 4. What is the main advantage of using Infrastructure as Code (IaC) with Terraform for ML deployments?
- a) Faster model training times
- b) Improved model accuracy
- c) Reproducible, version-controlled infrastructure provisioning
- d) Reduced data storage requirements

### 5. In ML CI/CD pipelines, when should automated model retraining be triggered?
- a) Only on a fixed schedule regardless of performance
- b) When data drift is detected or model performance degrades below thresholds
- c) Never, models should only be retrained manually
- d) Only when new code is committed to the repository

### 6. What is the purpose of canary deployments in ML model serving?
- a) To test models only in development environments
- b) To gradually roll out new models to a small percentage of traffic for validation
- c) To completely replace old models immediately
- d) To run multiple models simultaneously for all traffic

### 7. Which CloudWatch metrics are most important for monitoring ML model performance in production?
- a) Only CPU and memory usage
- b) Model accuracy, prediction latency, error rates, and data drift scores
- c) Only network throughput metrics
- d) Database connection counts only

### 8. What should happen when automated health checks fail during a blue-green deployment?
- a) Continue with the deployment anyway
- b) Automatically rollback to the previous version (blue service)
- c) Wait indefinitely for manual intervention
- d) Deploy to a different environment instead

### 9. In ML CI/CD pipelines, what is the recommended approach for handling model artifacts and data versioning?
- a) Store everything in the Git repository
- b) Use specialized tools like DVC for data versioning and MLflow for model registry
- c) Only version the code, not the data or models
- d) Use only cloud storage without versioning

### 10. What is the most effective strategy for ensuring zero-downtime deployments of ML models?
- a) Deploy during off-peak hours only
- b) Use blue-green or canary deployment strategies with automated health checks and rollback
- c) Always deploy to a single instance and restart it
- d) Deploy manually with extensive downtime for testing

## Answers

### 1. What is the primary benefit of implementing CI/CD for ML pipelines?
**Answer: b) Automated testing, deployment, and rollback capabilities**

**Explanation:** CI/CD for ML provides automated validation of data quality, model performance, and system functionality, along with safe deployment strategies and automatic rollback capabilities when issues are detected. This reduces manual errors, ensures consistent deployments, and enables rapid response to problems. Options a, c, and d are incorrect because CI/CD actually maintains quality standards rather than reducing them, enhances rather than eliminates validation, and works with any model architecture.

### 2. In a blue-green deployment strategy for ML models, what happens during the traffic switch phase?
**Answer: c) Traffic is gradually moved from blue to green after health checks pass**

**Explanation:** In blue-green deployment, the new version (green) is deployed alongside the current version (blue). After comprehensive health checks validate the green service, traffic is switched from blue to green, typically all at once after validation. The blue service remains available for immediate rollback if issues arise. Option a describes canary deployment, option b is incorrect timing, and option d reverses the roles.

### 3. Which testing strategy is most critical for ML model validation in CI/CD pipelines?
**Answer: b) Comprehensive testing including model performance, robustness, and data validation**

**Explanation:** ML models require specialized testing beyond traditional software testing. This includes validating model performance metrics (accuracy, precision, recall), testing robustness to input variations, validating data quality and schema compliance, and ensuring prediction functionality works correctly. Options a and c are insufficient, while option d misses model-specific validation requirements.

### 4. What is the main advantage of using Infrastructure as Code (IaC) with Terraform for ML deployments?
**Answer: c) Reproducible, version-controlled infrastructure provisioning**

**Explanation:** IaC with Terraform allows infrastructure to be defined in code, version-controlled, and deployed consistently across environments. This ensures reproducible deployments, enables infrastructure changes to be reviewed and tested, and provides rollback capabilities for infrastructure changes. Options a, b, and d relate to model performance or storage, not infrastructure management benefits.

### 5. In ML CI/CD pipelines, when should automated model retraining be triggered?
**Answer: b) When data drift is detected or model performance degrades below thresholds**

**Explanation:** Automated retraining should be triggered by performance-based conditions such as accuracy dropping below thresholds, data drift detection indicating the model is no longer suitable for current data patterns, or significant changes in data distribution. This ensures models remain effective over time. Options a and c are too rigid, while option d doesn't consider model performance degradation.

### 6. What is the purpose of canary deployments in ML model serving?
**Answer: b) To gradually roll out new models to a small percentage of traffic for validation**

**Explanation:** Canary deployments allow new model versions to be tested with a small percentage of production traffic (e.g., 5-10%) while the majority of traffic continues using the stable version. This enables real-world validation with limited risk exposure. If issues are detected, traffic can be quickly redirected back to the stable version. Options a, c, and d don't describe the gradual, risk-limited nature of canary deployments.

### 7. Which CloudWatch metrics are most important for monitoring ML model performance in production?
**Answer: b) Model accuracy, prediction latency, error rates, and data drift scores**

**Explanation:** ML-specific metrics are crucial for monitoring model health: accuracy/performance metrics track model effectiveness, prediction latency ensures acceptable response times, error rates indicate system reliability, and data drift scores detect when input data patterns change. While system metrics (CPU, memory) are also important, ML-specific metrics are most critical for model performance monitoring.

### 8. What should happen when automated health checks fail during a blue-green deployment?
**Answer: b) Automatically rollback to the previous version (blue service)**

**Explanation:** When health checks fail during blue-green deployment, the system should automatically rollback to the previous stable version (blue service) to maintain service availability and prevent degraded performance from reaching users. This is a key safety mechanism in automated deployments. Options a and c risk service degradation, while option d doesn't address the immediate need to maintain service quality.

### 9. In ML CI/CD pipelines, what is the recommended approach for handling model artifacts and data versioning?
**Answer: b) Use specialized tools like DVC for data versioning and MLflow for model registry**

**Explanation:** ML workflows require specialized versioning tools: DVC (Data Version Control) handles large datasets and data pipelines efficiently, while MLflow provides model registry capabilities with metadata tracking, stage transitions, and artifact management. Git alone is insufficient for large binary files, and proper versioning is essential for reproducibility and rollback capabilities.

### 10. What is the most effective strategy for ensuring zero-downtime deployments of ML models?
**Answer: b) Use blue-green or canary deployment strategies with automated health checks and rollback**

**Explanation:** Zero-downtime deployments require strategies that maintain service availability during updates. Blue-green and canary deployments achieve this by running new versions alongside existing ones, validating functionality before switching traffic, and providing immediate rollback capabilities if issues arise. Options a and c involve downtime, while option d introduces manual processes that increase risk and deployment time.