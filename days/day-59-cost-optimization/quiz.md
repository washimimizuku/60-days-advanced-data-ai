# Day 59 Quiz: Cost Optimization - Resource Management & FinOps

## Instructions
Answer all questions. Each question has one correct answer.

---

1. **What is the most effective approach to cost allocation in ML infrastructure?**
   - A) Track costs only at the service level
   - B) Implement comprehensive tagging strategy with business unit, team, model, and workload type tags
   - C) Monitor costs manually once per month
   - D) Use only default AWS cost categories

2. **Which strategy provides the highest cost savings for ML training workloads?**
   - A) Always use the largest instances available
   - B) Combine spot instances (60-90% savings), right-sizing, and intelligent scheduling
   - C) Only use on-demand instances for reliability
   - D) Run training jobs continuously without optimization

3. **What is the recommended approach for right-sizing ML instances?**
   - A) Use the same instance type for all workloads
   - B) Analyze utilization metrics (CPU, memory, GPU) over 14+ days and match instance types to actual usage patterns
   - C) Always choose the cheapest instance type
   - D) Upgrade to larger instances whenever possible

4. **How should spot instances be used effectively for ML workloads?**
   - A) Use spot instances for all production serving workloads
   - B) Implement diversified spot fleets with fault-tolerant training jobs and checkpointing for interruption handling
   - C) Avoid spot instances entirely due to interruptions
   - D) Use spot instances only for development environments

5. **What is the key principle of FinOps for ML teams?**
   - A) Minimize costs at all costs, regardless of performance
   - B) Create cost awareness, accountability, and optimization culture with shared responsibility between engineering and finance
   - C) Let only the finance team handle cost optimization
   - D) Focus only on infrastructure costs, ignoring operational costs

6. **Which automated scheduling strategy provides the most cost savings?**
   - A) Keep all resources running 24/7 for availability
   - B) Implement intelligent scheduling based on usage patterns, business hours, and workload requirements
   - C) Manually start and stop resources daily
   - D) Use random scheduling to distribute costs

7. **What is the most important metric for measuring ML cost efficiency?**
   - A) Total infrastructure spend only
   - B) Cost per model prediction, training job, or business outcome with ROI analysis
   - C) Number of instances running
   - D) Storage costs alone

8. **How should development and staging environments be optimized for cost?**
   - A) Use the same configuration as production
   - B) Implement automated scheduling, smaller instance types, and shared resources with 60-80% cost reduction
   - C) Keep them running continuously like production
   - D) Use only the most expensive instances for testing

9. **What is the recommended approach for storage cost optimization in ML systems?**
   - A) Store all data in the most expensive storage tier
   - B) Implement intelligent tiering with lifecycle policies, compression, and archival strategies based on access patterns
   - C) Delete data frequently to save costs
   - D) Use only local storage to avoid cloud costs

10. **Which tool combination provides the most comprehensive cost optimization?**
    - A) Manual spreadsheet tracking only
    - B) AWS Cost Explorer, custom tagging, automated rightsizing tools, and FinOps dashboards with ML-specific metrics
    - C) Basic cloud billing reports
    - D) Only infrastructure monitoring tools

---

## Answer Key

**1. B** - Comprehensive tagging strategy with business unit, team, model, and workload type tags enables accurate cost allocation and accountability. This allows teams to understand their spending patterns, optimize resources effectively, and make data-driven decisions about ML infrastructure investments.

**2. B** - Combining spot instances (60-90% savings), right-sizing, and intelligent scheduling provides the highest cost savings. Spot instances offer significant discounts for fault-tolerant workloads, right-sizing eliminates over-provisioning waste, and scheduling reduces costs during non-business hours.

**3. B** - Analyzing utilization metrics (CPU, memory, GPU) over 14+ days and matching instance types to actual usage patterns ensures optimal resource allocation. This data-driven approach prevents over-provisioning while maintaining performance requirements and can reduce costs by 20-50%.

**4. B** - Implementing diversified spot fleets with fault-tolerant training jobs and checkpointing for interruption handling maximizes savings while maintaining reliability. This approach can reduce training costs by 60-90% while handling the inherent interruption risk of spot instances.

**5. B** - FinOps creates cost awareness, accountability, and optimization culture with shared responsibility between engineering and finance. This collaborative approach ensures that cost optimization is built into development processes rather than being an afterthought, leading to sustainable cost management.

**6. B** - Intelligent scheduling based on usage patterns, business hours, and workload requirements can reduce costs by 60-80% for non-production environments. This includes shutting down development resources outside business hours and scaling based on actual demand patterns.

**7. B** - Cost per model prediction, training job, or business outcome with ROI analysis provides meaningful cost efficiency metrics. This business-focused approach helps teams understand the value generated per dollar spent and optimize for business impact rather than just infrastructure costs.

**8. B** - Implementing automated scheduling, smaller instance types, and shared resources can achieve 60-80% cost reduction for development and staging environments. These environments don't need production-level performance or availability, making them ideal candidates for aggressive cost optimization.

**9. B** - Intelligent tiering with lifecycle policies, compression, and archival strategies based on access patterns optimizes storage costs while maintaining data availability. This can reduce storage costs by 50-80% by automatically moving infrequently accessed data to cheaper storage tiers.

**10. B** - AWS Cost Explorer, custom tagging, automated rightsizing tools, and FinOps dashboards with ML-specific metrics provide comprehensive cost optimization capabilities. This combination enables visibility, analysis, automation, and continuous optimization of ML infrastructure costs.