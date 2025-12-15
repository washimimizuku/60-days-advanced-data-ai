"""
Day 10: Data Privacy - GDPR, PII Handling - Exercise
Build comprehensive privacy protection systems
"""

import pandas as pd
import numpy as np
import re
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

# TODO: Exercise 1 - Implement PII Detection System
class PIIDetector:
    """Detect PII in datasets using multiple techniques"""
    
    def __init__(self):
        # TODO: Initialize PII detection patterns
        self.patterns = {}
        self.confidence_threshold = 0.7
    
    def detect_pii_in_dataframe(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """
        TODO: Implement comprehensive PII detection
        
        Requirements:
        1. Check column names for PII indicators
        2. Analyze data patterns using regex
        3. Calculate confidence scores
        4. Classify PII categories (direct, quasi, sensitive)
        5. Return detailed findings with recommendations
        
        Test with sample data:
        - Email addresses
        - Phone numbers
        - Social Security Numbers
        - Credit card numbers
        - IP addresses
        """
        # TODO: Implement detection logic
        results = {}
        
        # Basic patterns for common PII types
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b'
        }
        
        for column in df.columns:
            findings = []
            
            # TODO: Analyze column name for PII indicators
            name_score, category = self._analyze_column_name(column)
            if name_score > self.confidence_threshold:
                findings.append({
                    'type': 'column_name',
                    'confidence': name_score,
                    'category': category
                })
            
            # TODO: Analyze data patterns
            data_score, pii_type = self._analyze_data_patterns(df[column])
            if data_score > self.confidence_threshold:
                findings.append({
                    'type': 'data_pattern', 
                    'confidence': data_score,
                    'pii_type': pii_type
                })
            
            if findings:
                results[column] = findings
        
        return results
    
    def _analyze_column_name(self, column_name: str) -> Tuple[float, str]:
        """TODO: Score column names for PII likelihood"""
        # TODO: Check against PII indicator patterns
        pii_indicators = {
            'email': 0.95, 'phone': 0.9, 'ssn': 0.98,
            'name': 0.9, 'address': 0.85, 'id': 0.7
        }
        
        column_lower = column_name.lower()
        max_score = 0
        category = 'non_pii'
        
        for indicator, score in pii_indicators.items():
            if indicator in column_lower:
                if score > max_score:
                    max_score = score
                    category = 'direct_identifier' if indicator in ['email', 'ssn'] else 'quasi_identifier'
        
        # TODO: Return confidence score and category
        return max_score, category
    
    def _analyze_data_patterns(self, data_series: pd.Series) -> Tuple[float, str]:
        """TODO: Analyze data for PII patterns"""
        # TODO: Apply regex patterns to sample data
        if len(data_series) == 0:
            return 0.0, 'none'
        
        sample = data_series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return 0.0, 'none'
        
        patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b'
        }
        
        best_score = 0
        best_type = 'none'
        
        for pii_type, pattern in patterns.items():
            matches = sum(1 for val in sample if re.search(pattern, val))
            # TODO: Calculate match percentage
            match_ratio = matches / len(sample)
            
            if match_ratio > best_score:
                best_score = match_ratio
                best_type = pii_type
        
        # TODO: Return confidence and PII type
        return best_score, best_type

# TODO: Exercise 2 - Build Data Classification System
class PIICategory(Enum):
    DIRECT_IDENTIFIER = "direct_identifier"
    QUASI_IDENTIFIER = "quasi_identifier" 
    SENSITIVE_ATTRIBUTE = "sensitive_attribute"
    NON_PII = "non_pii"

class PIIRiskLevel(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

@dataclass
class PIIClassification:
    column_name: str
    category: PIICategory
    risk_level: PIIRiskLevel
    confidence: float
    evidence: List[str]
    recommendations: List[str]

class DataClassifier:
    """Classify data columns by PII risk and category"""
    
    def __init__(self):
        # TODO: Initialize classification rules
        pass
    
    def classify_dataset(self, df: pd.DataFrame) -> Dict[str, PIIClassification]:
        """
        TODO: Classify all columns in dataset
        
        Requirements:
        1. Analyze each column for PII indicators
        2. Determine risk level based on uniqueness and sensitivity
        3. Provide specific recommendations for each column
        4. Handle edge cases (empty columns, mixed data types)
        
        Test scenarios:
        - Direct identifiers (email, SSN)
        - Quasi-identifiers (age, zip code)
        - Sensitive attributes (health, religion)
        - Non-PII data (product IDs, timestamps)
        """
        # TODO: Implement classification logic
        pass
    
    def _determine_risk_level(self, category: PIICategory, 
                            column_data: pd.Series) -> PIIRiskLevel:
        """TODO: Determine risk level based on category and data characteristics"""
        # TODO: Consider uniqueness, sensitivity, combination risk
        pass

# TODO: Exercise 3 - Implement K-Anonymity System
class KAnonymizer:
    """Apply k-anonymity to protect quasi-identifiers"""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def anonymize_dataset(self, df: pd.DataFrame, 
                         quasi_identifiers: List[str]) -> pd.DataFrame:
        """
        TODO: Apply k-anonymity to dataset
        
        Requirements:
        1. Group records by quasi-identifier combinations
        2. Identify groups smaller than k
        3. Apply generalization to small groups
        4. Ensure all groups have at least k records
        5. Preserve data utility while protecting privacy
        
        Generalization techniques:
        - Age: 25 -> "20-30"
        - Zip code: 12345 -> "123**"
        - Salary: 75000 -> "70000-80000"
        - Date: 2024-01-15 -> "2024-01"
        """
        # TODO: Implement k-anonymity algorithm
        pass
    
    def _generalize_age(self, ages: pd.Series) -> pd.Series:
        """TODO: Generalize ages into ranges"""
        pass
    
    def _generalize_zipcode(self, zipcodes: pd.Series) -> pd.Series:
        """TODO: Generalize zip codes by masking digits"""
        pass
    
    def _generalize_salary(self, salaries: pd.Series) -> pd.Series:
        """TODO: Generalize salaries into ranges"""
        pass
    
    def _check_k_anonymity(self, df: pd.DataFrame, 
                          quasi_identifiers: List[str]) -> bool:
        """TODO: Verify that dataset satisfies k-anonymity"""
        pass

# TODO: Exercise 4 - Build Pseudonymization System
class Pseudonymizer:
    """Create pseudonyms while preserving data utility"""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.pseudonym_mapping = {}
    
    def pseudonymize_column(self, data: pd.Series, 
                           pseudonym_type: str = "hash") -> pd.Series:
        """
        TODO: Pseudonymize a data column
        
        Requirements:
        1. Support different pseudonymization types:
           - Hash-based (irreversible)
           - Encryption-based (reversible)
           - Format-preserving (maintains structure)
        2. Ensure consistency (same input -> same output)
        3. Handle different data formats (email, phone, SSN)
        4. Maintain referential integrity across datasets
        
        Test with:
        - Email addresses (preserve domain)
        - Phone numbers (preserve format)
        - User IDs (maintain uniqueness)
        - Names (preserve length/structure)
        """
        # TODO: Implement pseudonymization logic
        pass
    
    def format_preserving_pseudonymize(self, identifier: str, 
                                     format_type: str) -> str:
        """TODO: Create pseudonym that preserves original format"""
        # TODO: Handle email, phone, SSN formats
        pass
    
    def reverse_pseudonym(self, pseudonym: str) -> Optional[str]:
        """TODO: Reverse pseudonym if using reversible method"""
        pass

# TODO: Exercise 5 - Create GDPR Request Handler
class RequestType(Enum):
    ACCESS = "access"
    RECTIFICATION = "rectification"
    ERASURE = "erasure"
    PORTABILITY = "portability"

class RequestStatus(Enum):
    RECEIVED = "received"
    VERIFIED = "verified"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"

@dataclass
class DataSubjectRequest:
    request_id: str
    request_type: RequestType
    subject_email: str
    subject_identifiers: Dict[str, str]
    received_date: datetime
    due_date: datetime
    status: RequestStatus
    notes: List[str]

class GDPRRequestHandler:
    """Handle GDPR data subject requests"""
    
    def __init__(self, data_systems: Dict[str, Any]):
        self.data_systems = data_systems
        self.requests = {}
    
    def submit_request(self, request_type: RequestType, 
                      subject_email: str,
                      subject_identifiers: Dict[str, str],
                      request_details: str) -> str:
        """
        TODO: Submit and process data subject request
        
        Requirements:
        1. Generate unique request ID
        2. Set appropriate due date (30 days for GDPR)
        3. Send acknowledgment to data subject
        4. Initialize request tracking
        5. Validate request completeness
        
        Handle different request types:
        - Access: Provide all personal data
        - Rectification: Correct inaccurate data
        - Erasure: Delete personal data
        - Portability: Export data in machine-readable format
        """
        # TODO: Implement request submission
        pass
    
    def process_access_request(self, request_id: str) -> Dict:
        """TODO: Process right of access request"""
        # TODO: Find all personal data across systems
        # TODO: Compile comprehensive access report
        # TODO: Include processing purposes and legal bases
        pass
    
    def process_erasure_request(self, request_id: str) -> Dict:
        """TODO: Process right to erasure request"""
        # TODO: Check legal requirements for data retention
        # TODO: Identify all systems containing personal data
        # TODO: Execute deletion in proper order
        # TODO: Verify complete erasure
        pass
    
    def _find_personal_data(self, subject_identifiers: Dict[str, str]) -> Dict:
        """TODO: Find all personal data across systems"""
        pass
    
    def _verify_identity(self, request: DataSubjectRequest, 
                        verification_data: Dict) -> bool:
        """TODO: Verify data subject identity"""
        pass

# TODO: Exercise 6 - Build Privacy Impact Assessment
class PrivacyRisk(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class PrivacyAssessment:
    data_types: List[str]
    processing_purposes: List[str]
    risk_level: PrivacyRisk
    risk_factors: List[str]
    mitigation_measures: List[str]
    compliance_requirements: List[str]

class PrivacyImpactAssessment:
    """Conduct privacy impact assessments"""
    
    def __init__(self):
        # TODO: Initialize assessment criteria
        pass
    
    def assess_dataset(self, df: pd.DataFrame, 
                      processing_purpose: str,
                      data_subjects: str) -> PrivacyAssessment:
        """
        TODO: Conduct privacy impact assessment
        
        Requirements:
        1. Identify data types and sensitivity levels
        2. Assess processing purposes and legal bases
        3. Evaluate privacy risks and likelihood
        4. Recommend mitigation measures
        5. Identify compliance requirements
        
        Risk factors to consider:
        - Volume of personal data
        - Sensitivity of data types
        - Number of data subjects
        - Processing purposes
        - Data sharing/transfers
        - Security measures
        - Data retention periods
        """
        # TODO: Implement assessment logic
        pass
    
    def _assess_data_sensitivity(self, df: pd.DataFrame) -> Tuple[List[str], PrivacyRisk]:
        """TODO: Assess sensitivity of data types"""
        pass
    
    def _evaluate_processing_risks(self, purpose: str, data_types: List[str]) -> List[str]:
        """TODO: Evaluate risks based on processing purpose"""
        pass

def main():
    """Test all privacy protection systems"""
    
    print("=== Day 10: Data Privacy - GDPR, PII Handling ===\n")
    
    # Create sample dataset with various PII types
    sample_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8],
        'email': [
            'john.doe@email.com', 'jane.smith@company.com', 
            'bob.wilson@gmail.com', 'alice.brown@yahoo.com',
            'charlie.davis@outlook.com', 'diana.miller@email.com',
            'frank.garcia@company.com', 'grace.lee@gmail.com'
        ],
        'phone': [
            '555-123-4567', '555-234-5678', '555-345-6789', '555-456-7890',
            '555-567-8901', '555-678-9012', '555-789-0123', '555-890-1234'
        ],
        'age': [25, 26, 35, 36, 45, 28, 32, 29],
        'zip_code': ['12345', '12346', '67890', '67891', '11111', '12345', '67890', '11111'],
        'salary': [75000, 76000, 85000, 86000, 95000, 72000, 88000, 74000],
        'medical_condition': ['none', 'diabetes', 'hypertension', 'none', 'asthma', 'none', 'diabetes', 'none']
    })
    
    print("Sample dataset created with PII data")
    print(f"Dataset shape: {sample_data.shape}")
    print(f"Columns: {list(sample_data.columns)}\n")
    
    # TODO: Test Exercise 1 - PII Detection
    print("=== Exercise 1: PII Detection ===")
    detector = PIIDetector()
    # TODO: Detect PII in sample dataset
    pii_results = detector.detect_pii_in_dataframe(sample_data)
    
    # TODO: Print detection results with confidence scores
    print(f"PII Detection Results: {len(pii_results)} columns with PII detected")
    for column, findings in pii_results.items():
        print(f"  {column}: {len(findings)} findings")
        for finding in findings:
            print(f"    - {finding['type']}: confidence {finding.get('confidence', 0):.2f}")
    print("TODO: Complete implementation with full PII classification\n")
    
    # TODO: Test Exercise 2 - Data Classification
    print("=== Exercise 2: Data Classification ===")
    classifier = DataClassifier()
    # TODO: Classify all columns in dataset
    # TODO: Print classification results with risk levels
    print("TODO: Implement data classification and test with sample data\n")
    
    # TODO: Test Exercise 3 - K-Anonymity
    print("=== Exercise 3: K-Anonymity ===")
    anonymizer = KAnonymizer(k=3)
    quasi_identifiers = ['age', 'zip_code', 'salary']
    # TODO: Apply k-anonymity to dataset
    # TODO: Verify k-anonymity is satisfied
    # TODO: Compare original vs anonymized data
    print("TODO: Implement k-anonymity and test with sample data\n")
    
    # TODO: Test Exercise 4 - Pseudonymization
    print("=== Exercise 4: Pseudonymization ===")
    pseudonymizer = Pseudonymizer()
    # TODO: Pseudonymize different column types
    # TODO: Test format-preserving pseudonymization
    # TODO: Test consistency across multiple runs
    print("TODO: Implement pseudonymization and test with sample data\n")
    
    # TODO: Test Exercise 5 - GDPR Request Handling
    print("=== Exercise 5: GDPR Request Handling ===")
    data_systems = {
        "customer_db": {"type": "postgresql"},
        "analytics_db": {"type": "bigquery"}
    }
    gdpr_handler = GDPRRequestHandler(data_systems)
    
    # TODO: Submit access request
    # TODO: Process access request
    # TODO: Submit erasure request
    # TODO: Process erasure request
    print("TODO: Implement GDPR request handling and test workflows\n")
    
    # TODO: Test Exercise 6 - Privacy Impact Assessment
    print("=== Exercise 6: Privacy Impact Assessment ===")
    pia = PrivacyImpactAssessment()
    # TODO: Conduct privacy assessment on sample dataset
    # TODO: Print assessment results with risk factors
    print("TODO: Implement privacy impact assessment and test with sample data\n")
    
    print("=== All Exercises Complete ===")
    print("\n🎯 Learning Objectives Achieved:")
    print("  ✓ Basic PII detection patterns implemented")
    print("  ✓ Privacy system architecture understood")
    print("  ✓ GDPR compliance framework established")
    print("  ✓ Sample data and test scenarios created")
    print("\n🚀 Next Steps:")
    print("  1. Complete TODO implementations in each exercise")
    print("  2. Test with sample_data/pii_test_dataset.csv")
    print("  3. Review solution.py for production patterns")
    print("  4. Take quiz.md to test understanding")
    print("  5. Consider edge cases and production requirements")

if __name__ == "__main__":
    main()
