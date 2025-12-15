# Day 10 Quiz: Data Privacy - GDPR, PII Handling

## Questions

### 1. What is the maximum fine under GDPR for serious violations?
- a) €10 million or 2% of global annual revenue
- b) €20 million or 4% of global annual revenue
- c) €50 million or 6% of global annual revenue
- d) €100 million or 10% of global annual revenue

### 2. Which of the following is NOT considered a direct identifier under GDPR?
- a) Email address
- b) Social Security Number
- c) Age combined with zip code
- d) Passport number

### 3. What is the primary difference between anonymization and pseudonymization?
- a) Anonymization is reversible, pseudonymization is not
- b) Pseudonymization is reversible with additional information, anonymization is not
- c) They are the same technique with different names
- d) Anonymization only applies to numeric data

### 4. Under GDPR, how long does an organization have to respond to a data subject access request?
- a) 15 days
- b) 30 days (1 month)
- c) 60 days (2 months)
- d) 90 days (3 months)

### 5. What is k-anonymity designed to protect against?
- a) Data breaches
- b) Re-identification attacks using quasi-identifiers
- c) Unauthorized access to databases
- d) Data corruption

### 6. Which GDPR principle requires collecting only the data necessary for the specified purpose?
- a) Lawfulness, fairness, and transparency
- b) Purpose limitation
- c) Data minimization
- d) Storage limitation

### 7. What is the main challenge with implementing differential privacy?
- a) It's too expensive to implement
- b) Balancing privacy protection with data utility
- c) It only works with small datasets
- d) It requires special hardware

### 8. Which of the following is considered "special category" personal data under GDPR?
- a) Email addresses
- b) Phone numbers
- c) Health information
- d) Purchase history

### 9. What is the "Right to be Forgotten" also known as under GDPR?
- a) Right of access
- b) Right to rectification
- c) Right to erasure
- d) Right to data portability

### 10. In PII detection systems, what does a high uniqueness ratio (>95%) typically indicate?
- a) The data is well-anonymized
- b) The column likely contains identifiers or quasi-identifiers
- c) The data quality is poor
- d) The dataset is too small

---

## Answers

### 1. What is the maximum fine under GDPR for serious violations?
**Answer: b) €20 million or 4% of global annual revenue**

**Explanation:** GDPR Article 83 sets administrative fines at up to €20 million or 4% of the total worldwide annual turnover of the preceding financial year, whichever is higher. This applies to the most serious violations, including infringements of basic principles for processing (Article 5), legal bases for processing (Article 6), and data subject rights (Chapter III). Lesser violations can result in fines up to €10 million or 2% of global annual revenue.

---

### 2. Which of the following is NOT considered a direct identifier under GDPR?
**Answer: c) Age combined with zip code**

**Explanation:** Age combined with zip code is a quasi-identifier, not a direct identifier. Direct identifiers can uniquely identify an individual on their own (email, SSN, passport number), while quasi-identifiers are attributes that, when combined with other attributes, can potentially identify someone. Research shows that 87% of Americans can be uniquely identified using just age, gender, and zip code, making this combination highly identifying but still technically quasi-identifiers rather than direct identifiers.

---

### 3. What is the primary difference between anonymization and pseudonymization?
**Answer: b) Pseudonymization is reversible with additional information, anonymization is not**

**Explanation:** Pseudonymization replaces identifying information with pseudonyms but maintains the ability to re-identify individuals if you have the key or mapping table. Anonymization removes or modifies identifying information in such a way that individuals cannot be re-identified, even with additional information. GDPR treats these differently - pseudonymized data is still considered personal data and subject to GDPR, while properly anonymized data is not.

---

### 4. Under GDPR, how long does an organization have to respond to a data subject access request?
**Answer: b) 30 days (1 month)**

**Explanation:** GDPR Article 12(3) requires organizations to respond to data subject requests "without undue delay and in any event within one month of receipt of the request." This period can be extended by two additional months for complex requests, but the data subject must be informed of the extension and the reasons within the first month. The clock starts ticking from when the request is received, not when identity is verified.

---

### 5. What is k-anonymity designed to protect against?
**Answer: b) Re-identification attacks using quasi-identifiers**

**Explanation:** K-anonymity ensures that each individual in a dataset is indistinguishable from at least k-1 other individuals with respect to quasi-identifying attributes. This protects against re-identification attacks where an attacker tries to link records in an anonymized dataset back to specific individuals using combinations of quasi-identifiers like age, zip code, and gender. It doesn't protect against data breaches, unauthorized access, or corruption - those require different security measures.

---

### 6. Which GDPR principle requires collecting only the data necessary for the specified purpose?
**Answer: c) Data minimization**

**Explanation:** GDPR Article 5(1)(c) establishes the data minimization principle, stating that personal data shall be "adequate, relevant and limited to what is necessary in relation to the purposes for which they are processed." This means organizations should only collect and process the minimum amount of personal data needed to achieve their stated purpose. Purpose limitation (Article 5(1)(b)) is different - it requires that data be collected for specified, explicit, and legitimate purposes.

---

### 7. What is the main challenge with implementing differential privacy?
**Answer: b) Balancing privacy protection with data utility**

**Explanation:** The fundamental challenge with differential privacy is the privacy-utility tradeoff. Adding more noise provides stronger privacy guarantees but reduces the accuracy and usefulness of query results. The epsilon parameter controls this tradeoff - smaller epsilon means stronger privacy but less accurate results. Organizations must carefully balance their privacy requirements with their need for accurate analytics, which often requires domain expertise and extensive testing.

---

### 8. Which of the following is considered "special category" personal data under GDPR?
**Answer: c) Health information**

**Explanation:** GDPR Article 9 defines special categories of personal data (formerly "sensitive data") as data revealing racial or ethnic origin, political opinions, religious or philosophical beliefs, trade union membership, genetic data, biometric data, health data, or data concerning sex life or sexual orientation. Health information falls under this category and requires explicit consent or other specific legal bases for processing, along with enhanced protection measures.

---

### 9. What is the "Right to be Forgotten" also known as under GDPR?
**Answer: c) Right to erasure**

**Explanation:** The "Right to be Forgotten" is formally called the "Right to erasure" under GDPR Article 17. This right allows individuals to request deletion of their personal data under specific circumstances, such as when the data is no longer necessary for the original purpose, consent is withdrawn, or the data has been unlawfully processed. However, this right is not absolute and has exceptions for freedom of expression, legal compliance, and legitimate interests.

---

### 10. In PII detection systems, what does a high uniqueness ratio (>95%) typically indicate?
**Answer: b) The column likely contains identifiers or quasi-identifiers**

**Explanation:** A high uniqueness ratio means most values in the column are unique or nearly unique, which is characteristic of identifiers (like user IDs, email addresses) or quasi-identifiers (like precise timestamps, detailed addresses). This high uniqueness creates re-identification risk because unique or rare values can be used to link records across datasets or identify specific individuals. Well-anonymized data would have lower uniqueness ratios due to generalization or suppression techniques.

---

## Score Interpretation

- **9-10 correct**: Excellent! You have a strong understanding of data privacy and GDPR requirements
- **7-8 correct**: Good job! Review the questions you missed and focus on technical implementation details
- **5-6 correct**: You're on the right track. Study GDPR principles and privacy techniques more thoroughly
- **Below 5**: Review the theory section and practice with the hands-on exercises

---

## Key Concepts to Remember

1. **GDPR fines can reach €20 million or 4% of global revenue** for serious violations
2. **Direct identifiers uniquely identify individuals** while quasi-identifiers require combination
3. **Pseudonymization is reversible, anonymization is not** - different privacy protections
4. **Data subject requests must be answered within 30 days** under GDPR
5. **K-anonymity protects against re-identification** using quasi-identifiers
6. **Data minimization requires collecting only necessary data** for the stated purpose
7. **Differential privacy balances privacy and utility** through controlled noise addition
8. **Special category data includes health information** and requires explicit consent
9. **Right to erasure is the formal name** for "Right to be Forgotten"
10. **High uniqueness ratios indicate identifier columns** with re-identification risk

---

## Data Privacy Best Practices

### Technical Implementation
- **Implement automated PII detection** across all data systems
- **Use multiple privacy techniques** (anonymization, pseudonymization, differential privacy)
- **Apply privacy by design** principles from system inception
- **Regular privacy impact assessments** for new processing activities

### Organizational Measures
- **Appoint a Data Protection Officer (DPO)** for GDPR compliance
- **Provide regular privacy training** to all staff handling personal data
- **Establish clear data governance** policies and procedures
- **Implement privacy-preserving analytics** techniques

### Legal Compliance
- **Document legal bases** for all personal data processing
- **Implement data subject rights** workflows and response procedures
- **Maintain processing records** as required by GDPR Article 30
- **Establish data breach response** procedures with 72-hour notification

### Risk Management
- **Conduct regular privacy audits** and compliance assessments
- **Monitor for privacy violations** and unauthorized data access
- **Implement data retention policies** with automated deletion
- **Use privacy-enhancing technologies** (PETs) where appropriate

### Common Pitfalls to Avoid
- **Don't assume consent covers everything** - check legal bases carefully
- **Don't ignore quasi-identifiers** - they can be as risky as direct identifiers
- **Don't delay data subject responses** - GDPR timelines are strict
- **Don't forget about data transfers** - international transfers need safeguards
- **Don't neglect vendor management** - third parties must also be GDPR compliant

Ready to move on to Day 11! 🚀
