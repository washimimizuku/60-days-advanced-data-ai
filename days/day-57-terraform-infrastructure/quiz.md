# Day 57 Quiz: Terraform & Infrastructure as Code - Multi-cloud & Best Practices

## Instructions
Answer all questions. Each question has one correct answer.

---

1. **What is the primary benefit of using Terraform for ML infrastructure deployment?**
   - A) It only works with AWS services
   - B) It provides declarative, version-controlled, and repeatable infrastructure provisioning across multiple cloud providers
   - C) It eliminates the need for cloud providers
   - D) It only manages compute resources

2. **Which Terraform feature is most important for managing different environments (dev, staging, prod)?**
   - A) Variables only
   - B) Workspaces combined with environment-specific configurations and remote state management
   - C) Modules only
   - D) Providers only

3. **What is the recommended approach for managing Terraform state in production environments?**
   - A) Store state files locally on developer machines
   - B) Use remote state backends (like S3) with state locking (DynamoDB) and encryption
   - C) Commit state files to Git repositories
   - D) Use multiple local state files

4. **How should sensitive values like database passwords be handled in Terraform configurations?**
   - A) Hard-code them in the configuration files
   - B) Use Terraform variables with sensitive = true, random providers, and external secret management systems
   - C) Store them in plain text files
   - D) Include them in version control

5. **What is the best practice for organizing Terraform code for large ML infrastructure projects?**
   - A) Put everything in a single main.tf file
   - B) Use modules for reusable components, separate environments, and logical resource grouping
   - C) Create separate files for each resource type only
   - D) Use only variables and outputs

6. **Which approach is most effective for cost optimization in Terraform-managed ML infrastructure?**
   - A) Always use the largest instance types
   - B) Implement spot instances, auto-scaling, lifecycle policies, and resource scheduling based on usage patterns
   - C) Never use auto-scaling
   - D) Keep all resources running 24/7

7. **How should multi-cloud deployments be structured in Terraform?**
   - A) Use separate Terraform projects for each cloud
   - B) Use multiple providers with aliases, shared modules, and consistent resource naming and tagging
   - C) Avoid multi-cloud deployments entirely
   - D) Use only one cloud provider at a time

8. **What is the recommended approach for implementing security in Terraform ML infrastructure?**
   - A) Use default security settings for all resources
   - B) Implement least privilege IAM, network segmentation, encryption at rest and in transit, and security groups
   - C) Disable all security features for simplicity
   - D) Use only public subnets for all resources

9. **How should Terraform modules be designed for maximum reusability in ML projects?**
   - A) Hard-code all values within modules
   - B) Use parameterized inputs, clear outputs, comprehensive documentation, and logical abstraction levels
   - C) Create modules without any variables
   - D) Make modules cloud-provider specific only

10. **What is the best practice for handling Terraform upgrades and provider version management?**
    - A) Always use the latest versions without testing
    - B) Pin provider versions, test upgrades in non-production environments, and use version constraints
    - C) Never update Terraform or providers
    - D) Use different versions in each environment

---

## Answer Key

**1. B** - Terraform provides declarative, version-controlled, and repeatable infrastructure provisioning across multiple cloud providers. This enables consistent infrastructure deployment, collaboration through code, and the ability to recreate identical environments, which is crucial for ML workloads that require consistent compute and storage resources.

**2. B** - Workspaces combined with environment-specific configurations and remote state management is the most effective approach. Workspaces allow you to maintain separate state for each environment while using the same configuration code, and environment-specific configurations enable different resource sizing and settings per environment.

**3. B** - Remote state backends (like S3) with state locking (DynamoDB) and encryption is the production standard. This prevents state corruption from concurrent modifications, provides backup and versioning, enables team collaboration, and secures sensitive infrastructure information.

**4. B** - Sensitive values should use Terraform variables with sensitive = true, random providers for generating secrets, and external secret management systems. This prevents secrets from appearing in logs or state files while maintaining security and enabling automated secret rotation.

**5. B** - Using modules for reusable components, separate environments, and logical resource grouping is the best practice. This promotes code reuse, maintainability, testing, and clear separation of concerns, which is essential for complex ML infrastructure with multiple components.

**6. B** - Cost optimization should implement spot instances for training workloads, auto-scaling based on demand, lifecycle policies for data storage, and resource scheduling. This can reduce costs by 50-80% while maintaining performance for ML workloads with variable resource needs.

**7. B** - Multi-cloud deployments should use multiple providers with aliases, shared modules, and consistent resource naming and tagging. This enables leveraging the best services from each cloud while maintaining consistency and avoiding vendor lock-in.

**8. B** - Security implementation should include least privilege IAM policies, network segmentation with private subnets, encryption at rest and in transit, and properly configured security groups. This creates defense in depth for ML infrastructure handling sensitive data and models.

**9. B** - Modules should use parameterized inputs for flexibility, clear outputs for integration, comprehensive documentation for usability, and logical abstraction levels. This makes modules reusable across different projects and environments while maintaining clarity and functionality.

**10. B** - Version management should pin provider versions in configuration, test all upgrades in non-production environments first, and use version constraints to prevent unexpected changes. This ensures stability and predictability in production ML infrastructure deployments.