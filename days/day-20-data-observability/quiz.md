# Day 20 Quiz: Data Observability - Metrics, Alerting, Dashboards

## Questions

### 1. What are the five pillars of data observability?
- a) Freshness, Volume, Schema, Distribution, Lineage
- b) Quality, Performance, Security, Availability, Reliability
- c) Monitoring, Alerting, Logging, Tracing, Metrics
- d) Completeness, Accuracy, Consistency, Timeliness, Validity

### 2. Which anomaly detection method is most suitable for seasonal data patterns?
- a) Z-score detection
- b) IQR (Interquartile Range) detection
- c) Seasonal decomposition with residual analysis
- d) Simple threshold-based detection

### 3. What is the primary purpose of intelligent alerting in data observability?
- a) To send as many alerts as possible
- b) To reduce alert fatigue while ensuring critical issues are escalated
- c) To replace human monitoring entirely
- d) To generate detailed reports

### 4. In data lineage tracking, what does "impact score" typically measure?
- a) The size of the dataset
- b) The number of downstream consumers and their business criticality
- c) The frequency of data updates
- d) The complexity of data transformations

### 5. Which statistical method is most robust against outliers when detecting anomalies?
- a) Mean and standard deviation (Z-score)
- b) Median and MAD (Median Absolute Deviation)
- c) Linear regression
- d) Simple thresholding

### 6. What is the main advantage of consensus anomaly detection?
- a) It's faster than single-method detection
- b) It reduces false positives by requiring multiple methods to agree
- c) It uses less computational resources
- d) It works only with structured data

### 7. In schema monitoring, which type of change is typically considered "breaking"?
- a) Adding a new nullable column
- b) Removing an existing column
- c) Changing column order
- d) Adding an index

### 8. What should be the primary consideration when setting alert escalation policies?
- a) The time of day
- b) The severity of the issue and business impact
- c) The number of people available
- d) The cost of notifications

### 9. Which metric is most important for measuring data freshness?
- a) Total number of records
- b) Time elapsed since last data update
- c) Data processing speed
- d) Storage utilization

### 10. What is the purpose of alert correlation in intelligent alerting systems?
- a) To send more alerts
- b) To identify related alerts that might indicate a common root cause
- c) To reduce system performance
- d) To create backup alerts

---

## Answers

### 1. What are the five pillars of data observability?
**Answer: a) Freshness, Volume, Schema, Distribution, Lineage**

**Explanation:** The five pillars of data observability are universally recognized as: Freshness (how up-to-date is the data), Volume (is the expected amount of data present), Schema (has the structure changed), Distribution (are the values within expected ranges), and Lineage (understanding data flow and dependencies). These pillars provide comprehensive coverage of data health monitoring and are the foundation of any robust data observability strategy.

---

### 2. Which anomaly detection method is most suitable for seasonal data patterns?
**Answer: c) Seasonal decomposition with residual analysis**

**Explanation:** Seasonal decomposition breaks down time series data into trend, seasonal, and residual components, then applies anomaly detection to the residuals after removing the seasonal pattern. This approach is specifically designed for data with recurring patterns (daily, weekly, monthly cycles) and prevents normal seasonal variations from being flagged as anomalies. Simple methods like Z-score or IQR would incorrectly identify seasonal peaks and troughs as anomalies.

---

### 3. What is the primary purpose of intelligent alerting in data observability?
**Answer: b) To reduce alert fatigue while ensuring critical issues are escalated**

**Explanation:** Intelligent alerting systems are designed to balance two competing needs: ensuring that critical issues receive immediate attention while preventing alert fatigue that can cause teams to ignore or miss important notifications. This is achieved through context-aware severity calculation, suppression of duplicate or low-priority alerts, correlation analysis, and appropriate escalation policies based on business impact.

---

### 4. In data lineage tracking, what does "impact score" typically measure?
**Answer: b) The number of downstream consumers and their business criticality**

**Explanation:** Impact score in data lineage quantifies the potential business impact if a data source fails or has quality issues. It considers both the number of downstream consumers (dashboards, ML models, APIs, reports) and their business criticality. For example, a table feeding a financial reporting system would have a higher impact score than one feeding an internal analytics dashboard, helping prioritize incident response efforts.

---

### 5. Which statistical method is most robust against outliers when detecting anomalies?
**Answer: b) Median and MAD (Median Absolute Deviation)**

**Explanation:** Median and MAD are robust statistics that are not heavily influenced by outliers, unlike mean and standard deviation which can be skewed by extreme values. When using robust statistics for anomaly detection, the presence of existing outliers doesn't distort the baseline, making the detection more accurate. This is particularly important in real-world data where outliers are common and you want to detect new anomalies without being influenced by historical ones.

---

### 6. What is the main advantage of consensus anomaly detection?
**Answer: b) It reduces false positives by requiring multiple methods to agree**

**Explanation:** Consensus anomaly detection runs multiple detection algorithms (e.g., Z-score, IQR, Isolation Forest) and only flags anomalies that are detected by multiple methods. This approach significantly reduces false positives because it's unlikely that multiple independent methods would all incorrectly identify the same normal data point as anomalous. The trade-off is potentially missing some true anomalies, but the improved precision is usually worth it in production systems.

---

### 7. In schema monitoring, which type of change is typically considered "breaking"?
**Answer: b) Removing an existing column**

**Explanation:** Removing an existing column is a breaking change because downstream consumers (applications, queries, reports) that depend on that column will fail. Adding nullable columns, changing column order, or adding indexes are typically non-breaking changes that don't affect existing functionality. Breaking changes require coordinated updates across all consumers and careful migration planning to avoid system failures.

---

### 8. What should be the primary consideration when setting alert escalation policies?
**Answer: b) The severity of the issue and business impact**

**Explanation:** Alert escalation policies should primarily be based on the severity of the technical issue combined with its business impact. Critical issues affecting revenue-generating systems should escalate immediately to on-call engineers, while low-impact issues might only require team notifications. The escalation timeline, notification channels, and personnel involved should all be determined by this risk assessment rather than arbitrary factors like time of day or availability.

---

### 9. Which metric is most important for measuring data freshness?
**Answer: b) Time elapsed since last data update**

**Explanation:** Data freshness is fundamentally about timeliness - how recently the data was created or updated. This is typically measured as the time elapsed since the last update compared to expected update frequency. While other metrics like record count or processing speed are important for other aspects of data health, freshness specifically measures whether data is current enough for its intended use cases.

---

### 10. What is the purpose of alert correlation in intelligent alerting systems?
**Answer: b) To identify related alerts that might indicate a common root cause**

**Explanation:** Alert correlation analyzes multiple alerts occurring within a time window to identify patterns that might indicate a common root cause. For example, if data freshness, volume, and schema alerts all fire for the same table within minutes of each other, they're likely related to a single upstream issue. This helps incident responders understand the scope of problems, avoid duplicate work, and focus on root cause resolution rather than treating symptoms separately.

---

## Score Interpretation

- **9-10 correct**: Data Observability Expert! You understand comprehensive monitoring and intelligent alerting
- **7-8 correct**: Strong observability knowledge! Review anomaly detection methods and alerting strategies
- **5-6 correct**: Good foundation! Focus on the five pillars and statistical methods
- **3-4 correct**: Basic understanding present! Study observability frameworks and monitoring patterns
- **Below 3**: Review the theory section and practice with observability tools

---

## Key Concepts to Remember

### Five Pillars of Data Observability
1. **Freshness** measures how up-to-date data is relative to expected update frequency
2. **Volume** tracks whether expected amounts of data are present and detects anomalies
3. **Schema** monitors structural changes and their potential downstream impact
4. **Distribution** analyzes statistical properties and detects value anomalies
5. **Lineage** maps data dependencies and assesses downstream impact of issues

### Anomaly Detection Methods
6. **Z-score detection** identifies outliers based on standard deviations from the mean
7. **IQR detection** uses interquartile ranges to identify outliers, more robust than Z-score
8. **Seasonal decomposition** handles time series with recurring patterns
9. **Consensus detection** combines multiple methods to reduce false positives
10. **Robust statistics** (median/MAD) are less influenced by existing outliers

### Intelligent Alerting
11. **Context-aware severity** considers both technical deviation and business impact
12. **Alert suppression** prevents duplicate and low-value notifications
13. **Correlation analysis** identifies related alerts indicating common root causes
14. **Escalation policies** route alerts based on severity and business criticality
15. **Alert enrichment** provides context, suggested actions, and historical information

### Monitoring and Dashboards
16. **Real-time monitoring** enables proactive issue detection and faster response
17. **Executive dashboards** provide high-level data health visibility for leadership
18. **Operational dashboards** give detailed metrics for day-to-day monitoring
19. **Impact visualization** shows data lineage and dependency relationships
20. **Trend analysis** identifies gradual degradation and seasonal patterns

### Production Best Practices
- **Comprehensive coverage** across all five pillars prevents blind spots
- **Balanced alerting** reduces noise while ensuring critical issues are escalated
- **Business context** drives severity assessment and escalation decisions
- **Automated response** reduces manual effort and improves response times
- **Continuous improvement** based on incident analysis and feedback
- **Stakeholder alignment** ensures monitoring meets business needs

### Common Anti-Patterns to Avoid
- **Alert fatigue** from too many low-priority or duplicate notifications
- **Single-method detection** leading to high false positive rates
- **Ignoring seasonality** in time series data causing false anomalies
- **Lack of context** making alerts difficult to prioritize and respond to
- **Manual monitoring** that doesn't scale with data growth
- **Reactive approach** waiting for users to report issues instead of proactive detection

Ready to move on to Day 21! 🚀
