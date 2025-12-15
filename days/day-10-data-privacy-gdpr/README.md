# Day 10: Data Privacy - GDPR, PII Handling

## 📖 Learning Objectives

By the end of today, you will:
- Understand comprehensive data privacy regulations (GDPR, CCPA, PIPEDA)
- Implement automated PII detection and classification systems
- Build data anonymization and pseudonymization pipelines
- Apply privacy-preserving techniques (differential privacy, k-anonymity)
- Create GDPR-compliant data subject request workflows
- Deploy production-ready privacy governance systems

---

## Theory

### The Data Privacy Landscape

Data privacy has evolved from a compliance checkbox to a fundamental business requirement. Modern organizations handle massive volumes of personal data across complex systems, making privacy protection both critical and challenging.

**Key Privacy Regulations**:
- **GDPR** (General Data Protection Regulation) - EU, affects global companies
- **CCPA** (California Consumer Privacy Act) - California, affects US companies
- **PIPEDA** (Personal Information Protection and Electronic Documents Act) - Canada
- **LGPD** (Lei Geral de Proteção de Dados) - Brazil
- **PDPA** (Personal Data Protection Act) - Singapore, Thailand

### Why Data Privacy Matters

#### The Business Impact

```
Without Privacy Controls:
💰 GDPR fines up to 4% of global revenue
📉 Customer trust erosion and churn
⚖️ Legal liability and litigation costs
🚫 Market access restrictions (EU blocking)
🔍 Regulatory investigations and audits
```

```
With Privacy Controls:
✅ Regulatory compliance and reduced risk
📈 Enhanced customer trust and loyalty
🌍 Global market access and expansion
🛡️ Competitive advantage in privacy-conscious markets
📊 Better data governance and quality
```

#### Real-World Privacy Violations

**Major GDPR Fines**:
- **Amazon**: €746 million (2021) - Improper data processing
- **WhatsApp**: €225 million (2021) - Lack of transparency
- **Google**: €90 million (2019) - Insufficient consent mechanisms
- **British Airways**: €22 million (2020) - Data breach

**Common Violation Patterns**:
- Collecting data without proper consent
- Failing to implement data subject rights
- Inadequate data breach notifications
- Transferring data without proper safeguards
- Retaining data longer than necessary

### Understanding Personal Data and PII

#### What Constitutes Personal Data

**GDPR Definition**: Any information relating to an identified or identifiable natural person.

**Direct Identifiers**:
```python
direct_identifiers = {
    "names": ["first_name", "last_name", "full_name", "maiden_name"],
    "contact": ["email", "phone", "address", "postal_code"],
    "government_ids": ["ssn", "passport", "driver_license", "tax_id"],
    "financial": ["credit_card", "bank_account", "iban"],
    "biometric": ["fingerprint", "face_recognition", "voice_print"],
    "online": ["ip_address", "device_id", "cookie_id", "user_id"]
}
```

**Indirect Identifiers** (Quasi-identifiers):
```python
quasi_identifiers = {
    "demographic": ["age", "gender", "race", "ethnicity"],
    "geographic": ["zip_code", "city", "country", "coordinates"],
    "temporal": ["birth_date", "hire_date", "transaction_date"],
    "behavioral": ["purchase_history", "browsing_patterns", "preferences"],
    "professional": ["job_title", "company", "salary", "department"],
    "health": ["medical_conditions", "medications", "treatments"]
}
```

**Special Categories** (Sensitive Personal Data):
- Racial or ethnic origin
- Political opinions
- Religious or philosophical beliefs
- Trade union membership
- Genetic data
- Biometric data
- Health data
- Sex life or sexual orientation

#### PII Detection Challenges

**Context-Dependent Identification**:
```python
# Same data, different privacy implications
examples = {
    "employee_database": {
        "employee_id": "12345",  # Not PII in isolation
        "department": "Engineering",  # Not PII alone
        "salary": "$120,000",  # Not PII alone
        # But together: potentially identifiable
        "combined_risk": "HIGH - unique combination"
    },
    
    "anonymized_dataset": {
        "age": 34,
        "zip_code": "90210", 
        "profession": "Software Engineer",
        # Research shows 87% of Americans uniquely identifiable
        # with just age, gender, and zip code
        "reidentification_risk": "VERY HIGH"
    }
}
```

**Dynamic PII**:
```python
# Data that becomes PII through combination or context
dynamic_pii_examples = {
    "location_tracking": {
        "single_location": "Not PII",
        "location_pattern": "Reveals home/work - PII",
        "combined_with_time": "Highly identifying - PII"
    },
    
    "transaction_data": {
        "single_purchase": "Not PII",
        "purchase_pattern": "Reveals preferences - PII", 
        "combined_with_demographics": "Highly identifying - PII"
    }
}
```

### GDPR Principles and Requirements

#### Core Principles

**1. Lawfulness, Fairness, and Transparency**
```python
lawful_bases = {
    "consent": "Freely given, specific, informed agreement",
    "contract": "Processing necessary for contract performance", 
    "legal_obligation": "Required by law",
    "vital_interests": "Protecting life or health",
    "public_task": "Official authority or public interest",
    "legitimate_interests": "Balanced against individual rights"
}
```

**2. Purpose Limitation**
```python
purpose_limitation = {
    "original_purpose": "Customer service and support",
    "compatible_uses": [
        "Service improvement based on feedback",
        "Security monitoring for fraud prevention"
    ],
    "incompatible_uses": [
        "Marketing to third parties",  # ❌ Different purpose
        "Employee background checks"   # ❌ Different context
    ]
}
```

**3. Data Minimization**
```python
# Collect only what's necessary
data_minimization_example = {
    "newsletter_signup": {
        "necessary": ["email"],
        "optional": ["name", "preferences"],
        "excessive": ["phone", "address", "age"]  # ❌ Not needed
    },
    
    "e_commerce_order": {
        "necessary": ["shipping_address", "payment_info"],
        "optional": ["phone_for_delivery"],
        "excessive": ["mother_maiden_name", "ssn"]  # ❌ Not needed
    }
}
```

**4. Accuracy**
```python
accuracy_requirements = {
    "keep_updated": "Regular data refresh and validation",
    "correction_process": "Easy way for individuals to update data",
    "deletion_of_inaccurate": "Remove data that cannot be corrected",
    "verification": "Verify data accuracy at collection and processing"
}
```

**5. Storage Limitation**
```python
retention_policies = {
    "customer_data": {
        "active_customers": "Duration of relationship + 7 years",
        "inactive_customers": "3 years after last interaction",
        "marketing_data": "Until consent withdrawn or 2 years"
    },
    
    "employee_data": {
        "current_employees": "Duration of employment + 7 years",
        "applicant_data": "6 months if not hired",
        "payroll_records": "7 years after termination"
    }
}
```

**6. Integrity and Confidentiality (Security)**
```python
security_measures = {
    "encryption": {
        "at_rest": "AES-256 for stored data",
        "in_transit": "TLS 1.3 for data transmission",
        "key_management": "Hardware security modules (HSM)"
    },
    
    "access_controls": {
        "authentication": "Multi-factor authentication required",
        "authorization": "Role-based access control (RBAC)",
        "audit_logging": "All access logged and monitored"
    },
    
    "data_protection": {
        "pseudonymization": "Replace identifiers with pseudonyms",
        "anonymization": "Remove all identifying information",
        "differential_privacy": "Add statistical noise"
    }
}
```

**7. Accountability**
```python
accountability_requirements = {
    "documentation": [
        "Data processing records",
        "Privacy impact assessments",
        "Consent management logs",
        "Data breach incident reports"
    ],
    
    "governance": [
        "Data Protection Officer (DPO) appointment",
        "Privacy by design implementation",
        "Regular compliance audits",
        "Staff training programs"
    ]
}
```

#### Individual Rights

**Right to Information**
```python
transparency_requirements = {
    "privacy_notice_contents": [
        "Identity of data controller",
        "Purposes of processing",
        "Legal basis for processing",
        "Recipients of data",
        "Retention periods",
        "Individual rights",
        "Right to withdraw consent",
        "Right to lodge complaints"
    ]
}
```

**Right of Access (Subject Access Request)**
```python
access_request_response = {
    "timeline": "1 month (extendable to 3 months)",
    "information_provided": [
        "Confirmation of processing",
        "Purposes of processing", 
        "Categories of data",
        "Recipients of data",
        "Retention periods",
        "Copy of personal data",
        "Source of data if not collected directly"
    ],
    "format": "Structured, commonly used, machine-readable"
}
```

**Right to Rectification**
```python
rectification_process = {
    "timeline": "1 month",
    "scope": "Inaccurate or incomplete personal data",
    "notification": "Inform recipients of corrections",
    "implementation": "Update all systems containing the data"
}
```

**Right to Erasure (Right to be Forgotten)**
```python
erasure_grounds = [
    "Data no longer necessary for original purpose",
    "Consent withdrawn and no other legal basis",
    "Data processed unlawfully",
    "Erasure required for legal compliance",
    "Data collected from children without proper consent"
]

erasure_exceptions = [
    "Freedom of expression and information",
    "Legal compliance requirements",
    "Public health interests",
    "Archiving in public interest",
    "Legal claims establishment or defense"
]
```

### Automated PII Detection and Classification

#### Pattern-Based Detection

```python
import re
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class PIIPattern:
    name: str
    pattern: str
    confidence: float
    category: str

class PIIDetector:
    """Production-ready PII detection system"""
    
    def __init__(self):
        self.patterns = self._load_patterns()
        self.ml_model = self._load_ml_model()
    
    def _load_patterns(self) -> Dict[str, PIIPattern]:
        """Load regex patterns for common PII types"""
        return {
            "ssn": PIIPattern(
                name="Social Security Number",
                pattern=r'\b\d{3}-?\d{2}-?\d{4}\b',
                confidence=0.95,
                category="government_id"
            ),
            
            "email": PIIPattern(
                name="Email Address",
                pattern=r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                confidence=0.98,
                category="contact"
            ),
            
            "phone": PIIPattern(
                name="Phone Number",
                pattern=r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                confidence=0.90,
                category="contact"
            ),
            
            "credit_card": PIIPattern(
                name="Credit Card",
                pattern=r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|3[0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b',
                confidence=0.92,
                category="financial"
            ),
            
            "ip_address": PIIPattern(
                name="IP Address",
                pattern=r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                confidence=0.85,
                category="online"
            )
        }
    
    def detect_pii_in_text(self, text: str) -> List[Dict]:
        """Detect PII in unstructured text"""
        findings = []
        
        for pii_type, pattern in self.patterns.items():
            matches = re.finditer(pattern.pattern, text, re.IGNORECASE)
            
            for match in matches:
                findings.append({
                    "type": pii_type,
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": pattern.confidence,
                    "category": pattern.category
                })
        
        return findings
    
    def detect_pii_in_dataframe(self, df) -> Dict[str, List[Dict]]:
        """Detect PII in structured data (DataFrame)"""
        results = {}
        
        for column in df.columns:
            column_findings = []
            
            # Check column name for PII indicators
            column_score = self._score_column_name(column)
            if column_score > 0.7:
                column_findings.append({
                    "detection_type": "column_name",
                    "confidence": column_score,
                    "evidence": f"Column name '{column}' suggests PII"
                })
            
            # Sample data for pattern matching
            sample_data = df[column].dropna().astype(str).head(100)
            
            for idx, value in sample_data.items():
                text_findings = self.detect_pii_in_text(str(value))
                for finding in text_findings:
                    finding["row_index"] = idx
                    column_findings.append(finding)
            
            # ML-based detection for complex patterns
            ml_score = self._ml_detect_pii(sample_data.tolist())
            if ml_score > 0.8:
                column_findings.append({
                    "detection_type": "ml_classification",
                    "confidence": ml_score,
                    "evidence": "ML model detected PII patterns"
                })
            
            if column_findings:
                results[column] = column_findings
        
        return results
    
    def _score_column_name(self, column_name: str) -> float:
        """Score column name for PII likelihood"""
        pii_indicators = {
            "email": 0.95, "mail": 0.8, "e_mail": 0.95,
            "phone": 0.9, "tel": 0.8, "mobile": 0.85,
            "name": 0.9, "first_name": 0.95, "last_name": 0.95,
            "address": 0.9, "addr": 0.8, "street": 0.85,
            "ssn": 0.98, "social": 0.9, "security": 0.7,
            "id": 0.6, "identifier": 0.7, "user_id": 0.8,
            "birth": 0.8, "dob": 0.9, "birthday": 0.85,
            "credit": 0.8, "card": 0.6, "payment": 0.7
        }
        
        column_lower = column_name.lower()
        max_score = 0
        
        for indicator, score in pii_indicators.items():
            if indicator in column_lower:
                max_score = max(max_score, score)
        
        return max_score
    
    def _ml_detect_pii(self, data_sample: List[str]) -> float:
        """Use ML model for PII detection"""
        # Placeholder for ML model integration
        # In production, this would use trained models for:
        # - Named entity recognition (NER)
        # - Pattern classification
        # - Semantic similarity
        
        # For demo, return a mock score
        return 0.5
    
    def _load_ml_model(self):
        """Load pre-trained ML model for PII detection"""
        # In production, load actual ML models
        # - spaCy NER models
        # - Custom trained classifiers
        # - Transformer-based models
        pass

# Example usage
detector = PIIDetector()

# Detect in text
text = "Contact John Doe at john.doe@email.com or 555-123-4567. SSN: 123-45-6789"
pii_findings = detector.detect_pii_in_text(text)

for finding in pii_findings:
    print(f"Found {finding['type']}: {finding['value']} (confidence: {finding['confidence']})")
```

#### Advanced PII Classification

```python
from enum import Enum
from typing import Optional
import pandas as pd

class PIICategory(Enum):
    DIRECT_IDENTIFIER = "direct_identifier"
    QUASI_IDENTIFIER = "quasi_identifier"
    SENSITIVE_ATTRIBUTE = "sensitive_attribute"
    NON_PII = "non_pii"

class PIIRiskLevel(Enum):
    CRITICAL = "critical"      # Direct identifiers
    HIGH = "high"             # Quasi-identifiers with high risk
    MEDIUM = "medium"         # Quasi-identifiers with medium risk
    LOW = "low"              # Low risk attributes
    NONE = "none"            # Non-PII

@dataclass
class PIIClassification:
    column_name: str
    category: PIICategory
    risk_level: PIIRiskLevel
    confidence: float
    evidence: List[str]
    recommendations: List[str]

class AdvancedPIIClassifier:
    """Advanced PII classification with risk assessment"""
    
    def __init__(self):
        self.classification_rules = self._load_classification_rules()
    
    def _load_classification_rules(self) -> Dict:
        """Load comprehensive PII classification rules"""
        return {
            "direct_identifiers": {
                "patterns": ["email", "ssn", "passport", "driver_license"],
                "risk_level": PIIRiskLevel.CRITICAL,
                "recommendations": [
                    "Encrypt at rest and in transit",
                    "Implement strict access controls",
                    "Consider tokenization or pseudonymization",
                    "Regular access audits required"
                ]
            },
            
            "quasi_identifiers": {
                "high_risk": {
                    "patterns": ["zip_code", "birth_date", "ip_address"],
                    "risk_level": PIIRiskLevel.HIGH,
                    "recommendations": [
                        "Generalize or suppress for analytics",
                        "Combine with k-anonymity techniques",
                        "Monitor for re-identification risks"
                    ]
                },
                "medium_risk": {
                    "patterns": ["age", "gender", "city", "job_title"],
                    "risk_level": PIIRiskLevel.MEDIUM,
                    "recommendations": [
                        "Consider generalization for public datasets",
                        "Monitor combination with other attributes"
                    ]
                }
            },
            
            "sensitive_attributes": {
                "patterns": ["health", "medical", "religion", "political", "sexual"],
                "risk_level": PIIRiskLevel.CRITICAL,
                "recommendations": [
                    "Requires explicit consent",
                    "Enhanced security measures",
                    "Special category data handling",
                    "Regular compliance reviews"
                ]
            }
        }
    
    def classify_dataset(self, df: pd.DataFrame) -> Dict[str, PIIClassification]:
        """Classify all columns in a dataset"""
        classifications = {}
        
        for column in df.columns:
            classification = self.classify_column(column, df[column])
            classifications[column] = classification
        
        return classifications
    
    def classify_column(self, column_name: str, column_data: pd.Series) -> PIIClassification:
        """Classify a single column"""
        
        # Initialize classification
        category = PIICategory.NON_PII
        risk_level = PIIRiskLevel.NONE
        confidence = 0.0
        evidence = []
        recommendations = []
        
        # Check column name patterns
        name_score, name_category = self._analyze_column_name(column_name)
        if name_score > 0.7:
            category = name_category
            confidence = name_score
            evidence.append(f"Column name '{column_name}' matches PII patterns")
        
        # Analyze data patterns
        data_score, data_category = self._analyze_column_data(column_data)
        if data_score > confidence:
            category = data_category
            confidence = data_score
            evidence.append("Data patterns match PII signatures")
        
        # Determine risk level and recommendations
        risk_level, recommendations = self._determine_risk_and_recommendations(
            category, column_name, column_data
        )
        
        return PIIClassification(
            column_name=column_name,
            category=category,
            risk_level=risk_level,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _analyze_column_name(self, column_name: str) -> Tuple[float, PIICategory]:
        """Analyze column name for PII indicators"""
        name_lower = column_name.lower()
        
        # Direct identifiers
        direct_patterns = ["email", "ssn", "passport", "license", "credit_card"]
        for pattern in direct_patterns:
            if pattern in name_lower:
                return 0.95, PIICategory.DIRECT_IDENTIFIER
        
        # Quasi-identifiers
        quasi_patterns = ["zip", "postal", "birth", "age", "phone", "address"]
        for pattern in quasi_patterns:
            if pattern in name_lower:
                return 0.85, PIICategory.QUASI_IDENTIFIER
        
        # Sensitive attributes
        sensitive_patterns = ["health", "medical", "religion", "political", "race"]
        for pattern in sensitive_patterns:
            if pattern in name_lower:
                return 0.90, PIICategory.SENSITIVE_ATTRIBUTE
        
        return 0.0, PIICategory.NON_PII
    
    def _analyze_column_data(self, column_data: pd.Series) -> Tuple[float, PIICategory]:
        """Analyze column data for PII patterns"""
        
        # Sample data for analysis
        sample = column_data.dropna().astype(str).head(100)
        
        if len(sample) == 0:
            return 0.0, PIICategory.NON_PII
        
        # Check for email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = sum(1 for val in sample if re.match(email_pattern, val))
        if email_matches / len(sample) > 0.8:
            return 0.95, PIICategory.DIRECT_IDENTIFIER
        
        # Check for phone patterns
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phone_matches = sum(1 for val in sample if re.match(phone_pattern, val))
        if phone_matches / len(sample) > 0.8:
            return 0.90, PIICategory.DIRECT_IDENTIFIER
        
        # Check for numeric patterns that might be IDs
        numeric_pattern = r'^\d+$'
        numeric_matches = sum(1 for val in sample if re.match(numeric_pattern, val))
        if numeric_matches / len(sample) > 0.9:
            # Could be ID, age, zip code, etc.
            return 0.70, PIICategory.QUASI_IDENTIFIER
        
        return 0.0, PIICategory.NON_PII
    
    def _determine_risk_and_recommendations(self, category: PIICategory, 
                                          column_name: str, 
                                          column_data: pd.Series) -> Tuple[PIIRiskLevel, List[str]]:
        """Determine risk level and provide recommendations"""
        
        if category == PIICategory.DIRECT_IDENTIFIER:
            return PIIRiskLevel.CRITICAL, [
                "Encrypt data at rest and in transit",
                "Implement strict access controls",
                "Consider tokenization or pseudonymization",
                "Regular access audits required",
                "Data retention policy enforcement"
            ]
        
        elif category == PIICategory.QUASI_IDENTIFIER:
            # Assess uniqueness for risk level
            uniqueness = column_data.nunique() / len(column_data)
            
            if uniqueness > 0.9:  # High uniqueness = high risk
                return PIIRiskLevel.HIGH, [
                    "High re-identification risk",
                    "Consider generalization or suppression",
                    "Apply k-anonymity techniques",
                    "Monitor for combination attacks"
                ]
            else:
                return PIIRiskLevel.MEDIUM, [
                    "Monitor combination with other attributes",
                    "Consider generalization for public datasets"
                ]
        
        elif category == PIICategory.SENSITIVE_ATTRIBUTE:
            return PIIRiskLevel.CRITICAL, [
                "Special category data - requires explicit consent",
                "Enhanced security measures required",
                "Regular compliance reviews",
                "Strict purpose limitation"
            ]
        
        else:
            return PIIRiskLevel.NONE, ["No special privacy measures required"]

# Example usage
classifier = AdvancedPIIClassifier()

# Create sample dataset
sample_data = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'email': ['john@email.com', 'jane@email.com', 'bob@email.com', 'alice@email.com', 'charlie@email.com'],
    'age': [25, 30, 35, 28, 42],
    'zip_code': ['12345', '67890', '11111', '22222', '33333'],
    'purchase_amount': [100.50, 250.75, 75.25, 300.00, 150.25]
})

# Classify dataset
classifications = classifier.classify_dataset(sample_data)

for column, classification in classifications.items():
    print(f"\nColumn: {column}")
    print(f"Category: {classification.category.value}")
    print(f"Risk Level: {classification.risk_level.value}")
    print(f"Confidence: {classification.confidence:.2f}")
    print(f"Recommendations: {', '.join(classification.recommendations)}")
```

### Data Anonymization and Pseudonymization

#### Anonymization Techniques

**K-Anonymity**:
```python
import pandas as pd
from typing import List, Dict, Any

class KAnonymizer:
    """Implement k-anonymity for dataset protection"""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def anonymize(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
        """Apply k-anonymity to dataset"""
        
        # Group by quasi-identifiers
        grouped = df.groupby(quasi_identifiers)
        
        anonymized_groups = []
        
        for group_key, group_df in grouped:
            if len(group_df) < self.k:
                # Group too small, needs generalization or suppression
                generalized_group = self._generalize_group(group_df, quasi_identifiers)
                anonymized_groups.append(generalized_group)
            else:
                # Group already satisfies k-anonymity
                anonymized_groups.append(group_df)
        
        return pd.concat(anonymized_groups, ignore_index=True)
    
    def _generalize_group(self, group_df: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
        """Generalize small groups to achieve k-anonymity"""
        
        generalized_df = group_df.copy()
        
        for column in quasi_identifiers:
            if column in ['age']:
                # Age generalization: 25 -> "20-30"
                generalized_df[column] = self._generalize_age(generalized_df[column])
            elif column in ['zip_code']:
                # Zip code generalization: 12345 -> "123**"
                generalized_df[column] = self._generalize_zip(generalized_df[column])
            elif column in ['salary']:
                # Salary generalization: 75000 -> "70000-80000"
                generalized_df[column] = self._generalize_salary(generalized_df[column])
        
        return generalized_df
    
    def _generalize_age(self, ages: pd.Series) -> pd.Series:
        """Generalize ages into ranges"""
        def age_range(age):
            if pd.isna(age):
                return age
            age = int(age)
            range_start = (age // 10) * 10
            return f"{range_start}-{range_start + 9}"
        
        return ages.apply(age_range)
    
    def _generalize_zip(self, zip_codes: pd.Series) -> pd.Series:
        """Generalize zip codes by masking last digits"""
        def mask_zip(zip_code):
            if pd.isna(zip_code):
                return zip_code
            zip_str = str(zip_code)
            if len(zip_str) >= 3:
                return zip_str[:3] + "*" * (len(zip_str) - 3)
            return zip_str
        
        return zip_codes.apply(mask_zip)
    
    def _generalize_salary(self, salaries: pd.Series) -> pd.Series:
        """Generalize salaries into ranges"""
        def salary_range(salary):
            if pd.isna(salary):
                return salary
            salary = float(salary)
            range_start = (int(salary) // 10000) * 10000
            return f"{range_start}-{range_start + 9999}"
        
        return salaries.apply(salary_range)

# Example usage
anonymizer = KAnonymizer(k=3)

# Sample dataset
data = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 26, 35, 36, 45],
    'zip_code': ['12345', '12346', '67890', '67891', '11111'],
    'salary': [75000, 76000, 85000, 86000, 95000],
    'disease': ['flu', 'cold', 'diabetes', 'hypertension', 'asthma']
})

# Apply k-anonymity (excluding direct identifiers like 'name')
quasi_identifiers = ['age', 'zip_code', 'salary']
anonymized_data = anonymizer.anonymize(data.drop('name', axis=1), quasi_identifiers)

print("Original data:")
print(data)
print("\nAnonymized data:")
print(anonymized_data)
```

**L-Diversity**:
```python
class LDiversityAnonymizer:
    """Implement l-diversity for sensitive attribute protection"""
    
    def __init__(self, l: int = 2):
        self.l = l
    
    def anonymize(self, df: pd.DataFrame, quasi_identifiers: List[str], 
                 sensitive_attributes: List[str]) -> pd.DataFrame:
        """Apply l-diversity to dataset"""
        
        # Group by quasi-identifiers
        grouped = df.groupby(quasi_identifiers)
        
        diverse_groups = []
        
        for group_key, group_df in grouped:
            if self._check_l_diversity(group_df, sensitive_attributes):
                # Group already satisfies l-diversity
                diverse_groups.append(group_df)
            else:
                # Need to merge with other groups or suppress
                suppressed_group = self._suppress_group(group_df, sensitive_attributes)
                diverse_groups.append(suppressed_group)
        
        return pd.concat(diverse_groups, ignore_index=True)
    
    def _check_l_diversity(self, group_df: pd.DataFrame, sensitive_attributes: List[str]) -> bool:
        """Check if group satisfies l-diversity"""
        
        for sensitive_attr in sensitive_attributes:
            unique_values = group_df[sensitive_attr].nunique()
            if unique_values < self.l:
                return False
        
        return True
    
    def _suppress_group(self, group_df: pd.DataFrame, sensitive_attributes: List[str]) -> pd.DataFrame:
        """Suppress sensitive values that don't meet l-diversity"""
        
        suppressed_df = group_df.copy()
        
        for sensitive_attr in sensitive_attributes:
            # Replace with generic value if diversity is too low
            unique_count = suppressed_df[sensitive_attr].nunique()
            if unique_count < self.l:
                suppressed_df[sensitive_attr] = "*"
        
        return suppressed_df

# Example usage
l_anonymizer = LDiversityAnonymizer(l=2)

# Apply l-diversity
quasi_identifiers = ['age', 'zip_code']
sensitive_attributes = ['disease']

l_diverse_data = l_anonymizer.anonymize(data.drop('name', axis=1), 
                                       quasi_identifiers, 
                                       sensitive_attributes)

print("L-diverse data:")
print(l_diverse_data)
```

**Differential Privacy**:
```python
import numpy as np
from typing import Union

class DifferentialPrivacy:
    """Implement differential privacy mechanisms"""
    
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon  # Privacy budget
    
    def add_laplace_noise(self, value: Union[float, int], sensitivity: float = 1.0) -> float:
        """Add Laplace noise for differential privacy"""
        
        # Laplace mechanism: noise ~ Laplace(0, sensitivity/epsilon)
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale)
        
        return float(value) + noise
    
    def add_gaussian_noise(self, value: Union[float, int], sensitivity: float = 1.0, 
                          delta: float = 1e-5) -> float:
        """Add Gaussian noise for (epsilon, delta)-differential privacy"""
        
        # Gaussian mechanism
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma)
        
        return float(value) + noise
    
    def privatize_count(self, count: int) -> int:
        """Add noise to count queries"""
        noisy_count = self.add_laplace_noise(count, sensitivity=1.0)
        return max(0, int(round(noisy_count)))  # Ensure non-negative integer
    
    def privatize_sum(self, total: float, max_contribution: float) -> float:
        """Add noise to sum queries"""
        return self.add_laplace_noise(total, sensitivity=max_contribution)
    
    def privatize_average(self, average: float, count: int, max_contribution: float) -> float:
        """Add noise to average queries"""
        # For average, sensitivity = max_contribution / count
        sensitivity = max_contribution / count
        return self.add_laplace_noise(average, sensitivity=sensitivity)
    
    def exponential_mechanism(self, candidates: List[Any], 
                            utility_function: callable, 
                            sensitivity: float) -> Any:
        """Select from candidates using exponential mechanism"""
        
        # Calculate utilities
        utilities = [utility_function(candidate) for candidate in candidates]
        
        # Calculate probabilities
        scaled_utilities = [self.epsilon * u / (2 * sensitivity) for u in utilities]
        max_utility = max(scaled_utilities)
        
        # Normalize to prevent overflow
        normalized_utilities = [u - max_utility for u in scaled_utilities]
        probabilities = [np.exp(u) for u in normalized_utilities]
        
        # Normalize probabilities
        total_prob = sum(probabilities)
        probabilities = [p / total_prob for p in probabilities]
        
        # Sample according to probabilities
        return np.random.choice(candidates, p=probabilities)

# Example usage
dp = DifferentialPrivacy(epsilon=1.0)

# Original statistics
original_count = 1000
original_sum = 75000
original_avg = 75.0

# Add differential privacy
private_count = dp.privatize_count(original_count)
private_sum = dp.privatize_sum(original_sum, max_contribution=200)
private_avg = dp.privatize_average(original_avg, original_count, max_contribution=200)

print(f"Original count: {original_count}, Private count: {private_count}")
print(f"Original sum: {original_sum}, Private sum: {private_sum:.2f}")
print(f"Original average: {original_avg}, Private average: {private_avg:.2f}")
```

#### Pseudonymization Techniques

```python
import hashlib
import hmac
import secrets
from cryptography.fernet import Fernet
from typing import Dict, Optional

class Pseudonymizer:
    """Production-ready pseudonymization system"""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.cipher_suite = Fernet(Fernet.generate_key())
        self.pseudonym_mapping = {}  # For reversible pseudonymization
    
    def hash_pseudonymize(self, identifier: str, salt: Optional[str] = None) -> str:
        """Create irreversible pseudonym using hash"""
        
        if salt is None:
            salt = "default_salt"  # In production, use unique salt per dataset
        
        # Use HMAC for keyed hashing
        pseudonym = hmac.new(
            self.secret_key,
            f"{identifier}{salt}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"pseudo_{pseudonym[:16]}"  # Truncate for readability
    
    def encrypt_pseudonymize(self, identifier: str) -> str:
        """Create reversible pseudonym using encryption"""
        
        encrypted = self.cipher_suite.encrypt(identifier.encode())
        pseudonym_id = hashlib.sha256(encrypted).hexdigest()[:16]
        
        # Store mapping for reversal
        self.pseudonym_mapping[f"pseudo_{pseudonym_id}"] = encrypted
        
        return f"pseudo_{pseudonym_id}"
    
    def reverse_pseudonym(self, pseudonym: str) -> Optional[str]:
        """Reverse pseudonym to original identifier (if reversible)"""
        
        if pseudonym in self.pseudonym_mapping:
            encrypted = self.pseudonym_mapping[pseudonym]
            return self.cipher_suite.decrypt(encrypted).decode()
        
        return None  # Cannot reverse hash-based pseudonyms
    
    def format_preserving_pseudonymize(self, identifier: str, format_type: str) -> str:
        """Create pseudonym that preserves original format"""
        
        if format_type == "email":
            # Preserve email format: user@domain.com -> pseudo123@domain.com
            local, domain = identifier.split("@")
            pseudo_local = self.hash_pseudonymize(local)[:8]
            return f"{pseudo_local}@{domain}"
        
        elif format_type == "phone":
            # Preserve phone format: (555) 123-4567 -> (555) 987-6543
            digits = ''.join(filter(str.isdigit, identifier))
            pseudo_digits = self.hash_pseudonymize(digits)
            # Take numeric characters from hash
            pseudo_numbers = ''.join(filter(str.isdigit, pseudo_digits))[:len(digits)]
            
            # Reconstruct format
            if len(digits) == 10:
                return f"({pseudo_numbers[:3]}) {pseudo_numbers[3:6]}-{pseudo_numbers[6:]}"
        
        elif format_type == "ssn":
            # Preserve SSN format: 123-45-6789 -> 987-65-4321
            digits = ''.join(filter(str.isdigit, identifier))
            pseudo_digits = self.hash_pseudonymize(digits)
            pseudo_numbers = ''.join(filter(str.isdigit, pseudo_digits))[:9]
            return f"{pseudo_numbers[:3]}-{pseudo_numbers[3:5]}-{pseudo_numbers[5:]}"
        
        return self.hash_pseudonymize(identifier)
    
    def consistent_pseudonymize(self, identifier: str, context: str = "default") -> str:
        """Ensure same identifier gets same pseudonym across datasets"""
        
        # Use context to allow different pseudonyms in different contexts
        contextual_identifier = f"{context}:{identifier}"
        return self.hash_pseudonymize(contextual_identifier)

# Example usage
pseudonymizer = Pseudonymizer()

# Original identifiers
email = "john.doe@company.com"
phone = "(555) 123-4567"
ssn = "123-45-6789"
user_id = "user_12345"

# Different pseudonymization techniques
hash_pseudo = pseudonymizer.hash_pseudonymize(user_id)
encrypt_pseudo = pseudonymizer.encrypt_pseudonymize(user_id)
email_pseudo = pseudonymizer.format_preserving_pseudonymize(email, "email")
phone_pseudo = pseudonymizer.format_preserving_pseudonymize(phone, "phone")
ssn_pseudo = pseudonymizer.format_preserving_pseudonymize(ssn, "ssn")

print(f"Original user ID: {user_id}")
print(f"Hash pseudonym: {hash_pseudo}")
print(f"Encrypt pseudonym: {encrypt_pseudo}")
print(f"Reversed pseudonym: {pseudonymizer.reverse_pseudonym(encrypt_pseudo)}")

print(f"\nOriginal email: {email}")
print(f"Email pseudonym: {email_pseudo}")

print(f"\nOriginal phone: {phone}")
print(f"Phone pseudonym: {phone_pseudo}")

print(f"\nOriginal SSN: {ssn}")
print(f"SSN pseudonym: {ssn_pseudo}")

# Consistency check
pseudo1 = pseudonymizer.consistent_pseudonymize(user_id, "dataset1")
pseudo2 = pseudonymizer.consistent_pseudonymize(user_id, "dataset1")
pseudo3 = pseudonymizer.consistent_pseudonymize(user_id, "dataset2")

print(f"\nConsistency check:")
print(f"Same context: {pseudo1} == {pseudo2} -> {pseudo1 == pseudo2}")
print(f"Different context: {pseudo1} == {pseudo3} -> {pseudo1 == pseudo3}")
```

### GDPR Compliance Implementation

#### Data Subject Request Workflow

```python
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Optional
import json

class RequestType(Enum):
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    PORTABILITY = "portability"
    RESTRICTION = "restriction"
    OBJECTION = "objection"

class RequestStatus(Enum):
    RECEIVED = "received"
    VERIFIED = "verified"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    EXTENDED = "extended"

@dataclass
class DataSubjectRequest:
    request_id: str
    request_type: RequestType
    subject_email: str
    subject_identifiers: Dict[str, str]  # Additional identifiers
    request_details: str
    received_date: datetime
    due_date: datetime
    status: RequestStatus
    assigned_to: str
    notes: List[str]
    attachments: List[str]

class GDPRRequestHandler:
    """Handle GDPR data subject requests"""
    
    def __init__(self, data_systems: Dict[str, Any]):
        self.data_systems = data_systems
        self.requests = {}  # In production, use database
        self.lineage_system = None  # Integration with lineage tracking
    
    def submit_request(self, request_type: RequestType, subject_email: str,
                      subject_identifiers: Dict[str, str], 
                      request_details: str) -> str:
        """Submit new data subject request"""
        
        request_id = self._generate_request_id()
        received_date = datetime.now()
        due_date = received_date + timedelta(days=30)  # GDPR 1-month requirement
        
        request = DataSubjectRequest(
            request_id=request_id,
            request_type=request_type,
            subject_email=subject_email,
            subject_identifiers=subject_identifiers,
            request_details=request_details,
            received_date=received_date,
            due_date=due_date,
            status=RequestStatus.RECEIVED,
            assigned_to="",
            notes=[],
            attachments=[]
        )
        
        self.requests[request_id] = request
        
        # Send acknowledgment
        self._send_acknowledgment(request)
        
        # Auto-assign based on request type
        self._auto_assign_request(request)
        
        return request_id
    
    def verify_identity(self, request_id: str, verification_data: Dict) -> bool:
        """Verify data subject identity"""
        
        request = self.requests.get(request_id)
        if not request:
            return False
        
        # In production, implement robust identity verification
        # - Document verification
        # - Multi-factor authentication
        # - Third-party identity services
        
        # For demo, simple email verification
        if verification_data.get("email") == request.subject_email:
            request.status = RequestStatus.VERIFIED
            request.notes.append(f"Identity verified on {datetime.now()}")
            return True
        
        return False
    
    def process_access_request(self, request_id: str) -> Dict:
        """Process right of access request"""
        
        request = self.requests.get(request_id)
        if not request or request.request_type != RequestType.ACCESS:
            return {"error": "Invalid request"}
        
        request.status = RequestStatus.IN_PROGRESS
        
        # Find all personal data for the subject
        personal_data = self._find_all_personal_data(request.subject_identifiers)
        
        # Compile access report
        access_report = {
            "request_id": request_id,
            "subject_email": request.subject_email,
            "generated_date": datetime.now().isoformat(),
            "data_sources": [],
            "processing_purposes": [],
            "recipients": [],
            "retention_periods": [],
            "rights_information": self._get_rights_information()
        }
        
        for system_name, data in personal_data.items():
            if data:
                access_report["data_sources"].append({
                    "system": system_name,
                    "data_categories": list(data.keys()),
                    "record_count": sum(len(records) for records in data.values()),
                    "last_updated": self._get_last_updated(system_name, request.subject_identifiers)
                })
        
        # Include processing purposes and legal bases
        access_report["processing_purposes"] = self._get_processing_purposes(request.subject_identifiers)
        
        # Include data recipients
        access_report["recipients"] = self._get_data_recipients(request.subject_identifiers)
        
        # Include retention periods
        access_report["retention_periods"] = self._get_retention_periods()
        
        request.status = RequestStatus.COMPLETED
        request.notes.append(f"Access request completed on {datetime.now()}")
        
        return access_report
    
    def process_erasure_request(self, request_id: str) -> Dict:
        """Process right to erasure (right to be forgotten) request"""
        
        request = self.requests.get(request_id)
        if not request or request.request_type != RequestType.ERASURE:
            return {"error": "Invalid request"}
        
        # Check if erasure is legally required
        erasure_assessment = self._assess_erasure_request(request)
        
        if not erasure_assessment["can_erase"]:
            request.status = RequestStatus.REJECTED
            request.notes.append(f"Erasure rejected: {erasure_assessment['reason']}")
            return {
                "status": "rejected",
                "reason": erasure_assessment["reason"],
                "legal_basis": erasure_assessment["legal_basis"]
            }
        
        request.status = RequestStatus.IN_PROGRESS
        
        # Find all systems containing personal data
        systems_with_data = self._find_systems_with_personal_data(request.subject_identifiers)
        
        deletion_plan = {
            "request_id": request_id,
            "systems_to_update": [],
            "deletion_order": [],
            "verification_steps": [],
            "completion_date": None
        }
        
        # Use lineage system to determine deletion order
        if self.lineage_system:
            deletion_order = self.lineage_system.get_deletion_order(systems_with_data)
        else:
            deletion_order = systems_with_data  # Simple approach
        
        # Execute deletions
        for system in deletion_order:
            deletion_result = self._delete_from_system(system, request.subject_identifiers)
            
            deletion_plan["systems_to_update"].append({
                "system": system,
                "deletion_result": deletion_result,
                "timestamp": datetime.now().isoformat()
            })
            
            deletion_plan["verification_steps"].append(
                f"Verify {system} no longer contains data for subject"
            )
        
        # Verify complete deletion
        verification_result = self._verify_complete_deletion(request.subject_identifiers)
        
        if verification_result["complete"]:
            request.status = RequestStatus.COMPLETED
            deletion_plan["completion_date"] = datetime.now().isoformat()
            request.notes.append(f"Erasure completed on {datetime.now()}")
        else:
            request.notes.append(f"Erasure incomplete: {verification_result['remaining_systems']}")
        
        return deletion_plan
    
    def process_portability_request(self, request_id: str) -> Dict:
        """Process data portability request"""
        
        request = self.requests.get(request_id)
        if not request or request.request_type != RequestType.PORTABILITY:
            return {"error": "Invalid request"}
        
        request.status = RequestStatus.IN_PROGRESS
        
        # Find portable data (data provided by subject or generated through use)
        portable_data = self._find_portable_data(request.subject_identifiers)
        
        # Format in structured, machine-readable format
        portability_package = {
            "request_id": request_id,
            "subject_email": request.subject_email,
            "export_date": datetime.now().isoformat(),
            "format": "JSON",
            "data": portable_data,
            "metadata": {
                "total_records": sum(len(data) for data in portable_data.values()),
                "data_categories": list(portable_data.keys()),
                "export_method": "automated"
            }
        }
        
        request.status = RequestStatus.COMPLETED
        request.notes.append(f"Portability request completed on {datetime.now()}")
        
        return portability_package
    
    def _find_all_personal_data(self, subject_identifiers: Dict[str, str]) -> Dict:
        """Find all personal data across systems"""
        
        all_data = {}
        
        for system_name, system_config in self.data_systems.items():
            try:
                system_data = self._query_system_for_personal_data(
                    system_config, subject_identifiers
                )
                all_data[system_name] = system_data
            except Exception as e:
                all_data[system_name] = {"error": str(e)}
        
        return all_data
    
    def _query_system_for_personal_data(self, system_config: Dict, 
                                       subject_identifiers: Dict[str, str]) -> Dict:
        """Query specific system for personal data"""
        
        # This would integrate with actual data systems
        # For demo, return mock data
        
        return {
            "profile_data": [
                {"field": "email", "value": subject_identifiers.get("email", "")},
                {"field": "name", "value": "John Doe"},
                {"field": "phone", "value": "+1-555-123-4567"}
            ],
            "transaction_data": [
                {"date": "2024-01-15", "amount": 100.50, "description": "Purchase"},
                {"date": "2024-01-20", "amount": 75.25, "description": "Refund"}
            ],
            "behavioral_data": [
                {"event": "login", "timestamp": "2024-01-15T10:30:00Z"},
                {"event": "page_view", "timestamp": "2024-01-15T10:31:00Z"}
            ]
        }
    
    def _assess_erasure_request(self, request: DataSubjectRequest) -> Dict:
        """Assess whether erasure request can be fulfilled"""
        
        # Check for legal obligations to retain data
        retention_requirements = self._check_retention_requirements(request.subject_identifiers)
        
        if retention_requirements["must_retain"]:
            return {
                "can_erase": False,
                "reason": retention_requirements["reason"],
                "legal_basis": retention_requirements["legal_basis"]
            }
        
        # Check for legitimate interests
        legitimate_interests = self._check_legitimate_interests(request.subject_identifiers)
        
        if legitimate_interests["override_erasure"]:
            return {
                "can_erase": False,
                "reason": legitimate_interests["reason"],
                "legal_basis": "Legitimate interests override"
            }
        
        return {
            "can_erase": True,
            "reason": "No legal barriers to erasure",
            "legal_basis": "Right to erasure applies"
        }
    
    def _check_retention_requirements(self, subject_identifiers: Dict[str, str]) -> Dict:
        """Check legal requirements to retain data"""
        
        # In production, this would check:
        # - Tax law requirements (7 years for financial records)
        # - Employment law requirements
        # - Industry-specific regulations
        # - Ongoing legal proceedings
        
        return {
            "must_retain": False,
            "reason": "",
            "legal_basis": ""
        }
    
    def _check_legitimate_interests(self, subject_identifiers: Dict[str, str]) -> Dict:
        """Check legitimate interests that might override erasure"""
        
        # In production, this would check:
        # - Fraud prevention
        # - Security monitoring
        # - Legal claims
        # - Freedom of expression
        
        return {
            "override_erasure": False,
            "reason": ""
        }
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        import uuid
        return f"DSR-{uuid.uuid4().hex[:8].upper()}"
    
    def _send_acknowledgment(self, request: DataSubjectRequest):
        """Send acknowledgment to data subject"""
        # In production, integrate with email system
        print(f"Acknowledgment sent to {request.subject_email} for request {request.request_id}")
    
    def _auto_assign_request(self, request: DataSubjectRequest):
        """Auto-assign request based on type and workload"""
        # In production, implement intelligent assignment
        request.assigned_to = "privacy-team@company.com"

# Example usage
data_systems = {
    "customer_db": {"type": "postgresql", "connection": "..."},
    "analytics_db": {"type": "bigquery", "connection": "..."},
    "crm_system": {"type": "salesforce", "connection": "..."}
}

gdpr_handler = GDPRRequestHandler(data_systems)

# Submit access request
request_id = gdpr_handler.submit_request(
    request_type=RequestType.ACCESS,
    subject_email="john.doe@email.com",
    subject_identifiers={"email": "john.doe@email.com", "user_id": "12345"},
    request_details="I would like to see all personal data you have about me"
)

print(f"Request submitted: {request_id}")

# Verify identity
verified = gdpr_handler.verify_identity(request_id, {"email": "john.doe@email.com"})
print(f"Identity verified: {verified}")

# Process access request
if verified:
    access_report = gdpr_handler.process_access_request(request_id)
    print(f"Access report generated with {len(access_report['data_sources'])} data sources")
```

---

## 💻 Hands-On Exercise

See `exercise.py` for hands-on practice with data privacy implementation.

**What you'll build**:
1. Implement automated PII detection system
2. Create data classification and risk assessment
3. Build anonymization pipeline with k-anonymity
4. Implement pseudonymization with format preservation
5. Create GDPR data subject request workflow
6. Build privacy impact assessment system

**Expected time**: 45 minutes

---

## 📚 Resources

- [GDPR Official Text](https://gdpr-info.eu/)
- [CCPA Official Guide](https://oag.ca.gov/privacy/ccpa)
- [Privacy by Design Principles](https://www.ipc.on.ca/wp-content/uploads/resources/7foundationalprinciples.pdf)
- [Differential Privacy Book](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf)
- [K-Anonymity Research](https://dataprivacylab.org/dataprivacy/projects/kanonymity/paper3.pdf)
- [NIST Privacy Framework](https://www.nist.gov/privacy-framework)

---

## 🎯 Key Takeaways

- **Data privacy is a fundamental business requirement**, not just compliance
- **PII detection requires both pattern matching and context analysis** for accuracy
- **Anonymization techniques must balance privacy protection with data utility**
- **Pseudonymization enables privacy-preserving analytics** while maintaining data relationships
- **GDPR compliance requires systematic processes** for data subject rights
- **Privacy by design** should be integrated into all data systems from the start
- **Regular privacy impact assessments** help identify and mitigate risks
- **Automated privacy controls** are essential for scale and consistency

---

## 🔄 What's Next?

Tomorrow we'll explore **Access Control - RBAC, Row-Level Security** where you'll learn to implement fine-grained access controls and security policies for data systems. This builds on today's privacy concepts by adding the security layer that protects personal data from unauthorized access.

**Preview of Day 11**:
- Role-based access control (RBAC) implementation
- Row-level security policies
- Attribute-based access control (ABAC)
- Dynamic access control with context
- Integration with identity providers
- Audit logging and compliance monitoring
