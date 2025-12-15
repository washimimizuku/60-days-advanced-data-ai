# Day 35: Model Versioning with DVC - Quiz

## Questions

**Instructions:** Choose the best answer for each question. This quiz tests your understanding of model versioning, DVC pipelines, MLflow model registry, deployment strategies, and production ML monitoring.

### 1. What is the primary advantage of using DVC (Data Version Control) over traditional Git for ML projects?
- a) Better code collaboration features
- b) Efficient handling of large datasets and binary files
- c) Faster commit and push operations
- d) Built-in code review capabilities

**Answer: b) Efficient handling of large datasets and binary files**

**Explanation:** DVC is specifically designed to handle large datasets, models, and other binary artifacts that are common in ML projects but problematic for Git. DVC uses content-based addressing and can store large files in remote storage while keeping lightweight metadata in Git. Traditional Git struggles with large binary files, leading to repository bloat and slow operations.

### 2. In MLflow Model Registry, what is the purpose of model stages (Staging, Production, Archived)?
- a) To organize models by training algorithm
- b) To manage model lifecycle and deployment workflow
- c) To categorize models by performance metrics
- d) To separate models by data source

**Answer: b) To manage model lifecycle and deployment workflow**

**Explanation:** Model stages in MLflow represent different phases of the model deployment lifecycle. Models typically progress from "Staging" (for testing and validation) to "Production" (for live serving) and eventually to "Archived" (for retired models). This provides a structured workflow for model promotion, rollback capabilities, and clear separation between development and production environments.

### 3. What does "content-based addressing" mean in the context of data versioning?
- a) Organizing files by their creation date
- b) Using file names to determine storage location
- c) Using file content hash as the unique identifier
- d) Storing files based on their size

**Answer: c) Using file content hash as the unique identifier**

**Explanation:** Content-based addressing uses a hash (like MD5 or SHA256) of the file's content as its unique identifier. This means identical files have the same hash regardless of their name or location, enabling efficient deduplication and ensuring data integrity. If file content changes, the hash changes, providing automatic versioning and change detection.

### 4. Which deployment strategy provides zero-downtime updates and easy rollback capabilities?
- a) Rolling deployment
- b) Blue-green deployment
- c) Canary deployment
- d) Recreate deployment

**Answer: b) Blue-green deployment**

**Explanation:** Blue-green deployment maintains two identical production environments (blue and green). While one serves live traffic, the other is updated with the new version. Traffic is then switched instantly to the updated environment, providing zero downtime. If issues arise, traffic can be immediately switched back to the previous environment, enabling instant rollback.

### 5. What is data lineage in ML systems?
- a) The chronological order of data collection
- b) The complete history of data transformations and dependencies
- c) The geographical origin of data sources
- d) The organizational hierarchy of data ownership

**Answer: b) The complete history of data transformations and dependencies**

**Explanation:** Data lineage tracks the complete journey of data through an ML pipeline, including all transformations, processing steps, and dependencies. It shows how raw data becomes features, which models use which datasets, and how outputs are generated. This is crucial for debugging, compliance, reproducibility, and understanding the impact of data changes on model performance.

### 6. In DVC pipelines, what is the purpose of the `dvc.lock` file?
- a) To prevent concurrent pipeline execution
- b) To store pipeline execution state and checksums
- c) To lock dependencies during installation
- d) To secure sensitive pipeline parameters

**Answer: b) To store pipeline execution state and checksums**

**Explanation:** The `dvc.lock` file stores the exact state of a pipeline execution, including checksums of all inputs, outputs, and dependencies. This enables DVC to determine which stages need to be re-executed when something changes and ensures reproducible pipeline runs. It's similar to a lockfile in package managers but for ML pipelines.

### 7. What is the main benefit of using MLflow's experiment tracking?
- a) Automatic hyperparameter optimization
- b) Systematic comparison of model runs and reproducibility
- c) Real-time model performance monitoring
- d) Automated model deployment

**Answer: b) Systematic comparison of model runs and reproducibility**

**Explanation:** MLflow experiment tracking allows data scientists to log parameters, metrics, artifacts, and code versions for each model training run. This enables systematic comparison of different experiments, understanding what changes led to performance improvements, and reproducing successful experiments. While MLflow has other capabilities, experiment tracking is primarily about organizing and comparing ML experiments.

### 8. Why is model rollback capability important in production ML systems?
- a) To save storage space by removing old models
- b) To quickly revert to a working model when issues are detected
- c) To comply with data retention policies
- d) To reduce computational costs

**Answer: b) To quickly revert to a working model when issues are detected**

**Explanation:** Model rollback allows teams to quickly return to a previously working model version when the current model exhibits problems like performance degradation, unexpected behavior, or system errors. This minimizes downtime and business impact while teams investigate and fix issues. Rollback is a critical safety mechanism in production ML systems.

### 9. What type of drift detection is most appropriate for categorical features?
- a) Kolmogorov-Smirnov test
- b) Chi-square test
- c) T-test
- d) Mann-Whitney U test

**Answer: b) Chi-square test**

**Explanation:** The chi-square test is designed to detect differences in categorical distributions by comparing observed vs. expected frequencies across categories. For categorical features, you want to detect if the distribution of categories has changed between training and production data. The Kolmogorov-Smirnov test is for continuous distributions, while t-test and Mann-Whitney U test are for comparing means/medians of continuous variables.

### 10. In healthcare AI applications, why is complete audit trail documentation particularly important?
- a) To improve model accuracy
- b) To reduce computational costs
- c) To meet regulatory compliance requirements (FDA, HIPAA)
- d) To enable faster model training

**Answer: c) To meet regulatory compliance requirements (FDA, HIPAA)**

**Explanation:** Healthcare AI applications are subject to strict regulatory requirements from agencies like the FDA (for medical devices) and HIPAA (for patient data privacy). Complete audit trails documenting data sources, transformations, model decisions, and system changes are required for regulatory approval, compliance audits, and ensuring patient safety. This documentation is essential for demonstrating that AI systems are safe, effective, and traceable.

### 11. What is the primary benefit of using DVC's content-based addressing for large ML datasets?
- a) Faster data loading times
- b) Automatic data deduplication and efficient storage
- c) Better data compression ratios
- d) Improved data security

**Answer: b) Automatic data deduplication and efficient storage**

**Explanation:** Content-based addressing uses file content hashes as identifiers, which means identical files (regardless of name or location) have the same hash. This enables automatic deduplication - if the same dataset is used in multiple experiments or by multiple team members, DVC stores it only once. This significantly reduces storage costs and improves efficiency in ML workflows with large datasets.

### 12. In MLflow Model Registry, what happens when you transition a model from "Staging" to "Production"?
- a) The model is automatically deployed to production servers
- b) The model stage metadata is updated, enabling production workflows
- c) The model files are moved to a different storage location
- d) The model is retrained with production data

**Answer: b) The model stage metadata is updated, enabling production workflows**

**Explanation:** Stage transitions in MLflow are metadata operations that update the model's stage label. This enables downstream systems and workflows to identify which model version should be used for production serving, but MLflow itself doesn't automatically deploy models. The stage information is used by deployment systems to determine which model to load and serve.

### 13. Which statistical test is most appropriate for detecting drift in continuous numerical features?
- a) Chi-square test
- b) Kolmogorov-Smirnov test
- c) Fisher's exact test
- d) McNemar's test

**Answer: b) Kolmogorov-Smirnov test**

**Explanation:** The Kolmogorov-Smirnov (KS) test is specifically designed to compare two continuous distributions and detect differences between them. It's non-parametric and doesn't assume any specific distribution shape, making it ideal for detecting distribution drift in numerical features. The chi-square test is for categorical data, while Fisher's exact and McNemar's tests are for specific categorical scenarios.

### 14. What is the main advantage of blue-green deployment over rolling deployment for ML models?
- a) Lower resource requirements
- b) Faster deployment times
- c) Instant rollback capability with zero downtime
- d) Better load balancing

**Answer: c) Instant rollback capability with zero downtime**

**Explanation:** Blue-green deployment maintains two identical environments where traffic can be instantly switched between them. If issues are detected with the new model, traffic can be immediately redirected back to the previous version without any downtime. Rolling deployments gradually replace instances, making rollback slower and potentially causing brief service interruptions.

### 15. In DVC pipelines, what triggers the re-execution of a pipeline stage?
- a) Changes in the stage command only
- b) Changes in dependencies, outputs, or parameters
- c) Manual force execution only
- d) Scheduled time intervals

**Answer: b) Changes in dependencies, outputs, or parameters**

**Explanation:** DVC tracks checksums of all dependencies (input files, code files), parameters, and outputs. When any of these change, DVC detects that the stage needs to be re-executed to maintain consistency. This ensures reproducibility and prevents stale results from being used when upstream changes occur.