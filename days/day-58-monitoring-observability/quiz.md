# Day 58 Quiz: Monitoring & Observability - Production Systems

## Instructions
Answer all questions. Each question has one correct answer.

---

1. **What are the three pillars of observability in production ML systems?**
   - A) Monitoring, alerting, and dashboards
   - B) Metrics, logs, and traces for comprehensive system understanding
   - C) CPU, memory, and disk usage
   - D) Prometheus, Grafana, and Jaeger only

2. **Which metric type is most appropriate for tracking model inference latency?**
   - A) Counter for total requests
   - B) Histogram to capture latency distribution and percentiles
   - C) Gauge for current memory usage
   - D) Summary for basic statistics only

3. **What is the primary purpose of Service Level Indicators (SLIs) in ML systems?**
   - A) To replace all other monitoring metrics
   - B) To provide quantitative measurements of service performance that align with user experience
   - C) To monitor only infrastructure components
   - D) To track developer productivity

4. **How should structured logging be implemented for ML model predictions?**
   - A) Use plain text logs with timestamps only
   - B) Include JSON format with model metadata, prediction context, and performance metrics
   - C) Log only errors and failures
   - D) Store logs in CSV format

5. **What is the main benefit of distributed tracing in ML pipelines?**
   - A) It replaces the need for metrics and logs
   - B) It provides end-to-end visibility of request flow across microservices and helps identify bottlenecks
   - C) It only monitors database queries
   - D) It reduces system performance overhead

6. **Which Prometheus query would best measure model error rate?**
   - A) ml_model_predictions_total
   - B) rate(ml_model_predictions_total{status="error"}[5m]) / rate(ml_model_predictions_total[5m])
   - C) ml_model_accuracy
   - D) histogram_quantile(0.95, ml_model_latency_bucket)

7. **What is an error budget in the context of SLOs?**
   - A) The financial cost of system failures
   - B) The acceptable amount of unreliability within the SLO target, used to balance reliability and feature velocity
   - C) The number of bugs allowed in code
   - D) The time allocated for debugging

8. **How should data drift be monitored in production ML systems?**
   - A) Only check manually once per month
   - B) Implement continuous monitoring with statistical tests and alert when drift scores exceed thresholds
   - C) Monitor only during model training
   - D) Use only accuracy metrics

9. **What is the recommended approach for alerting in ML systems?**
   - A) Alert on every metric change
   - B) Create actionable alerts based on SLIs with clear runbooks and escalation procedures
   - C) Only alert during business hours
   - D) Send all alerts to everyone

10. **Which tool combination provides the most comprehensive observability stack?**
    - A) Only Prometheus for all observability needs
    - B) Prometheus for metrics, ELK stack for logs, Jaeger for tracing, and Grafana for visualization
    - C) Just application logs and basic monitoring
    - D) Only cloud provider monitoring tools

---

## Answer Key

**1. B** - Metrics, logs, and traces are the three pillars of observability. Metrics provide quantitative measurements, logs offer contextual information, and traces show request flow across distributed systems. Together, they enable comprehensive understanding of system behavior and performance.

**2. B** - Histogram is the appropriate metric type for latency because it captures the distribution of response times and enables calculation of percentiles (P50, P95, P99). This provides better insights than simple averages and helps identify performance outliers.

**3. B** - SLIs provide quantitative measurements of service performance that align with user experience. They form the foundation for SLOs and help teams focus on metrics that matter to users, such as availability, latency, and accuracy rather than just infrastructure metrics.

**4. B** - Structured logging should use JSON format with comprehensive context including model metadata (name, version), prediction details (input features, output, confidence), performance metrics (latency), and event classification. This enables efficient querying and analysis.

**5. B** - Distributed tracing provides end-to-end visibility of request flow across microservices, helping identify bottlenecks, understand dependencies, and debug performance issues in complex ML pipelines. It shows how requests traverse multiple services and where time is spent.

**6. B** - The correct query calculates error rate as the ratio of error requests to total requests over a time window. This provides the percentage of failed predictions, which is a key reliability metric for ML systems.

**7. B** - Error budget is the acceptable amount of unreliability within the SLO target (e.g., if SLO is 99.9% availability, error budget is 0.1%). It helps balance reliability investments with feature development velocity and provides a framework for making trade-off decisions.

**8. B** - Data drift should be monitored continuously using statistical tests (KS test, PSI, etc.) with automated alerts when drift scores exceed predefined thresholds. This enables proactive model retraining and maintains prediction quality over time.

**9. B** - Effective alerting should be actionable (someone can and should respond), based on SLIs that matter to users, include clear runbooks for response procedures, and have proper escalation paths. This reduces alert fatigue and ensures appropriate response to real issues.

**10. B** - A comprehensive observability stack combines Prometheus for metrics collection and alerting, ELK stack (Elasticsearch, Logstash, Kibana) for log aggregation and analysis, Jaeger for distributed tracing, and Grafana for unified visualization and dashboards. This provides complete coverage of the three pillars of observability.