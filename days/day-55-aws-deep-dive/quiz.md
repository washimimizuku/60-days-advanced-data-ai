# Day 55 Quiz: AWS Deep Dive - Cloud Infrastructure for Data & AI Systems

## Instructions
Answer all questions. Each question has one correct answer.

---

1. **What is the primary advantage of using Amazon SageMaker for machine learning workflows?**
   - A) It only supports Python programming language
   - B) It provides a fully managed platform for the entire ML lifecycle from data preparation to model deployment
   - C) It requires manual infrastructure management
   - D) It only works with AWS proprietary algorithms

2. **Which AWS service is best suited for running containerized applications with automatic scaling?**
   - A) EC2 instances with manual configuration
   - B) Amazon ECS with Fargate launch type
   - C) AWS Lambda for all container workloads
   - D) Amazon RDS for container orchestration

3. **What is the main benefit of using AWS Lambda for event-driven processing in a data pipeline?**
   - A) It requires pre-provisioned servers
   - B) It automatically scales to zero when not in use and charges only for actual execution time
   - C) It only supports synchronous processing
   - D) It requires manual capacity planning

4. **Which combination of AWS services would be most appropriate for building a scalable data lake?**
   - A) Only Amazon RDS and EC2
   - B) Amazon S3 for storage, AWS Glue for ETL, and Amazon Athena for querying
   - C) Only DynamoDB and Lambda
   - D) ECS and CloudWatch only

5. **What is the purpose of AWS IAM roles in a production ML system?**
   - A) To store application data
   - B) To provide secure, temporary credentials and fine-grained access control to AWS resources
   - C) To monitor system performance
   - D) To handle load balancing

6. **Which AWS service provides distributed tracing for microservices architectures?**
   - A) CloudWatch Logs only
   - B) AWS X-Ray
   - C) Amazon S3
   - D) AWS Config

7. **What is the most cost-effective approach for training ML models that can tolerate interruptions?**
   - A) Always use On-Demand instances
   - B) Use SageMaker training jobs with Spot instances
   - C) Only use the largest available instance types
   - D) Avoid using managed services

8. **Which AWS service is best for storing and querying time-series data from IoT devices?**
   - A) Amazon RDS with standard tables
   - B) Amazon Timestream
   - C) Amazon S3 with manual indexing
   - D) DynamoDB without any optimization

9. **What is the recommended approach for handling secrets and sensitive configuration in AWS?**
   - A) Store them in application code
   - B) Use AWS Systems Manager Parameter Store or AWS Secrets Manager
   - C) Put them in S3 buckets with public access
   - D) Include them in Docker images

10. **Which AWS service combination provides the best solution for real-time model inference with high availability?**
    - A) Single EC2 instance with manual deployment
    - B) SageMaker endpoints with auto-scaling behind an Application Load Balancer
    - C) Lambda functions for all inference workloads regardless of latency requirements
    - D) RDS database for storing model predictions

---

## Answer Key

**1. B** - Amazon SageMaker provides a fully managed platform for the entire ML lifecycle from data preparation to model deployment. It includes integrated development environments, managed training infrastructure, model hosting, and MLOps capabilities, eliminating the need for manual infrastructure management.

**2. B** - Amazon ECS with Fargate launch type is best suited for running containerized applications with automatic scaling. Fargate is serverless, automatically manages the underlying infrastructure, and provides built-in auto-scaling capabilities based on CPU, memory, or custom metrics.

**3. B** - AWS Lambda automatically scales to zero when not in use and charges only for actual execution time. This makes it ideal for event-driven processing where workloads are sporadic or unpredictable, as you don't pay for idle time and don't need to manage server capacity.

**4. B** - Amazon S3 for storage, AWS Glue for ETL, and Amazon Athena for querying form the core of a scalable data lake architecture. S3 provides virtually unlimited storage, Glue handles data transformation and cataloging, and Athena enables SQL queries without managing infrastructure.

**5. B** - AWS IAM roles provide secure, temporary credentials and fine-grained access control to AWS resources. They follow the principle of least privilege, allowing services and applications to access only the resources they need without storing long-term credentials.

**6. B** - AWS X-Ray provides distributed tracing for microservices architectures. It tracks requests as they travel through multiple services, helping identify performance bottlenecks, errors, and dependencies in complex distributed systems.

**7. B** - Using SageMaker training jobs with Spot instances is the most cost-effective approach for training ML models that can tolerate interruptions. Spot instances can provide up to 90% cost savings compared to On-Demand instances, and SageMaker handles the complexity of Spot instance management.

**8. B** - Amazon Timestream is purpose-built for storing and querying time-series data from IoT devices. It provides automatic data lifecycle management, built-in time-series analytics functions, and can scale to trillions of events per day while being more cost-effective than general-purpose databases.

**9. B** - AWS Systems Manager Parameter Store or AWS Secrets Manager are the recommended approaches for handling secrets and sensitive configuration. They provide encryption at rest and in transit, access logging, automatic rotation capabilities, and integration with IAM for access control.

**10. B** - SageMaker endpoints with auto-scaling behind an Application Load Balancer provide the best solution for real-time model inference with high availability. This setup offers automatic scaling based on traffic, health checks, multi-AZ deployment, and managed infrastructure for consistent low-latency inference.
