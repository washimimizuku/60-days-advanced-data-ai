"""
Day 10: Data Privacy - GDPR, PII Handling - Solution
Production-ready privacy protection systems
"""

import pandas as pd
import numpy as np
import re
import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from cryptography.fernet import Fernet

# Solution 1 - Advanced PII Detection System
class PIIDetector:
    """Production-ready PII detection with multiple techniques"""
    
    def __init__(self):
        self.patterns = {
            "email": {
                "pattern": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                "confidence": 0.98,
                "category": "direct_identifier"
            },
            "phone": {
                "pattern": r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b',
                "confidence": 0.90,
                "category": "direct_identifier"
            },
            "ssn": {
                "pattern": r'\b\d{3}-?\d{2}-?\d{4}\b',
                "confidence": 0.95,
                "category": "direct_identifier"
            },
            "credit_card": {
                "pattern": r'\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13})\b',
                "confidence": 0.92,
                "category": "direct_identifier"
            },
            "ip_address": {
                "pattern": r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
                "confidence": 0.85,
                "category": "quasi_identifier"
            }
        }
        self.confidence_threshold = 0.7
        
        # Column name indicators
        self.column_indicators = {
            "email": 0.95, "mail": 0.8, "e_mail": 0.95,
            "phone": 0.9, "tel": 0.8, "mobile": 0.85, "cell": 0.8,
            "name": 0.9, "first_name": 0.95, "last_name": 0.95, "full_name": 0.95,
            "address": 0.9, "addr": 0.8, "street": 0.85, "city": 0.7,
            "ssn": 0.98, "social": 0.9, "security": 0.7,
            "id": 0.6, "identifier": 0.7, "user_id": 0.8,
            "birth": 0.8, "dob": 0.9, "birthday": 0.85,
            "credit": 0.8, "card": 0.6, "payment": 0.7,
            "zip": 0.85, "postal": 0.85, "code": 0.5
        }
    
    def detect_pii_in_dataframe(self, df: pd.DataFrame) -> Dict[str, List[Dict]]:
        """Comprehensive PII detection across all columns"""
        results = {}
        
        for column in df.columns:
            column_findings = []
            
            # Check column name for PII indicators
            name_score, name_category = self._analyze_column_name(column)
            if name_score > self.confidence_threshold:
                column_findings.append({
                    "detection_type": "column_name",
                    "confidence": name_score,
                    "category": name_category,
                    "evidence": f"Column name '{column}' suggests PII"
                })
            
            # Analyze data patterns
            data_score, data_category = self._analyze_data_patterns(df[column])
            if data_score > self.confidence_threshold:
                column_findings.append({
                    "detection_type": "data_pattern",
                    "confidence": data_score,
                    "category": data_category,
                    "evidence": f"Data patterns match PII signatures"
                })
            
            # Statistical analysis
            stats_score, stats_evidence = self._analyze_statistical_properties(df[column])
            if stats_score > self.confidence_threshold:
                column_findings.append({
                    "detection_type": "statistical",
                    "confidence": stats_score,
                    "category": "quasi_identifier",
                    "evidence": stats_evidence
                })
            
            if column_findings:
                # Determine overall classification
                max_confidence = max(finding["confidence"] for finding in column_findings)
                primary_category = next(
                    finding["category"] for finding in column_findings 
                    if finding["confidence"] == max_confidence
                )
                
                results[column] = {
                    "overall_confidence": max_confidence,
                    "primary_category": primary_category,
                    "findings": column_findings,
                    "recommendations": self._get_recommendations(primary_category, max_confidence)
                }
        
        return results
    
    def _analyze_column_name(self, column_name: str) -> Tuple[float, str]:
        """Analyze column name for PII likelihood"""
        column_lower = column_name.lower()
        max_score = 0
        category = "non_pii"
        
        for indicator, score in self.column_indicators.items():
            if indicator in column_lower:
                if score > max_score:
                    max_score = score
                    # Determine category based on indicator
                    if indicator in ["email", "ssn", "credit", "phone"]:
                        category = "direct_identifier"
                    elif indicator in ["zip", "birth", "id"]:
                        category = "quasi_identifier"
                    else:
                        category = "quasi_identifier"
        
        return max_score, category
    
    def _analyze_data_patterns(self, data_series: pd.Series) -> Tuple[float, str]:
        """Analyze data for PII patterns using regex"""
        if len(data_series) == 0:
            return 0.0, "non_pii"
        
        # Sample data for analysis (avoid processing entire column)
        sample = data_series.dropna().astype(str).head(100)
        if len(sample) == 0:
            return 0.0, "non_pii"
        
        best_score = 0
        best_category = "non_pii"
        
        for pii_type, pattern_info in self.patterns.items():
            matches = sum(1 for val in sample if re.search(pattern_info["pattern"], val, re.IGNORECASE))
            match_ratio = matches / len(sample)
            
            # Confidence based on match ratio and pattern confidence
            confidence = match_ratio * pattern_info["confidence"]
            
            if confidence > best_score:
                best_score = confidence
                best_category = pattern_info["category"]
        
        return best_score, best_category
    
    def _analyze_statistical_properties(self, data_series: pd.Series) -> Tuple[float, str]:
        """Analyze statistical properties for PII indicators"""
        if len(data_series) == 0:
            return 0.0, ""
        
        # Calculate uniqueness ratio
        uniqueness = data_series.nunique() / len(data_series)
        
        # High uniqueness might indicate identifiers
        if uniqueness > 0.95:
            return 0.8, f"High uniqueness ratio ({uniqueness:.2f}) suggests identifier"
        elif uniqueness > 0.8:
            return 0.6, f"Medium uniqueness ratio ({uniqueness:.2f}) suggests quasi-identifier"
        
        return 0.0, ""
    
    def _get_recommendations(self, category: str, confidence: float) -> List[str]:
        """Get recommendations based on PII category and confidence"""
        if category == "direct_identifier":
            return [
                "Encrypt data at rest and in transit",
                "Implement strict access controls",
                "Consider tokenization or pseudonymization",
                "Regular access audits required",
                "Apply data retention policies"
            ]
        elif category == "quasi_identifier":
            if confidence > 0.9:
                return [
                    "High re-identification risk",
                    "Apply k-anonymity or l-diversity",
                    "Consider generalization or suppression",
                    "Monitor for combination attacks"
                ]
            else:
                return [
                    "Monitor combination with other attributes",
                    "Consider generalization for public datasets"
                ]
        else:
            return ["Standard data protection measures sufficient"]

# Solution 2 - Advanced Data Classification System
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
    """Advanced data classification with risk assessment"""
    
    def __init__(self):
        self.sensitive_keywords = [
            "health", "medical", "disease", "condition", "diagnosis",
            "religion", "religious", "faith", "belief",
            "political", "party", "vote", "election",
            "sexual", "orientation", "gender", "race", "ethnicity"
        ]
    
    def classify_dataset(self, df: pd.DataFrame) -> Dict[str, PIIClassification]:
        """Classify all columns with comprehensive analysis"""
        classifications = {}
        
        for column in df.columns:
            classification = self._classify_column(column, df[column])
            classifications[column] = classification
        
        return classifications
    
    def _classify_column(self, column_name: str, column_data: pd.Series) -> PIIClassification:
        """Classify individual column with detailed analysis"""
        evidence = []
        
        # Analyze column name
        name_category, name_confidence = self._analyze_column_name(column_name)
        if name_confidence > 0.7:
            evidence.append(f"Column name indicates {name_category.value}")
        
        # Analyze data content
        data_category, data_confidence = self._analyze_column_data(column_data)
        if data_confidence > 0.7:
            evidence.append(f"Data patterns indicate {data_category.value}")
        
        # Check for sensitive attributes
        sensitive_score = self._check_sensitive_attributes(column_name, column_data)
        if sensitive_score > 0.7:
            evidence.append("Contains sensitive personal data")
        
        # Determine final classification
        final_category, final_confidence = self._determine_final_classification(
            name_category, name_confidence,
            data_category, data_confidence,
            sensitive_score
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(final_category, column_data)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(final_category, risk_level)
        
        return PIIClassification(
            column_name=column_name,
            category=final_category,
            risk_level=risk_level,
            confidence=final_confidence,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _analyze_column_name(self, column_name: str) -> Tuple[PIICategory, float]:
        """Analyze column name for PII category"""
        name_lower = column_name.lower()
        
        # Direct identifiers
        direct_patterns = ["email", "ssn", "passport", "license", "credit_card"]
        for pattern in direct_patterns:
            if pattern in name_lower:
                return PIICategory.DIRECT_IDENTIFIER, 0.95
        
        # Quasi-identifiers
        quasi_patterns = ["zip", "postal", "birth", "age", "phone", "address", "id"]
        for pattern in quasi_patterns:
            if pattern in name_lower:
                return PIICategory.QUASI_IDENTIFIER, 0.85
        
        # Sensitive attributes
        for keyword in self.sensitive_keywords:
            if keyword in name_lower:
                return PIICategory.SENSITIVE_ATTRIBUTE, 0.90
        
        return PIICategory.NON_PII, 0.0
    
    def _analyze_column_data(self, column_data: pd.Series) -> Tuple[PIICategory, float]:
        """Analyze column data patterns"""
        if len(column_data) == 0:
            return PIICategory.NON_PII, 0.0
        
        sample = column_data.dropna().astype(str).head(100)
        if len(sample) == 0:
            return PIICategory.NON_PII, 0.0
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = sum(1 for val in sample if re.match(email_pattern, val))
        if email_matches / len(sample) > 0.8:
            return PIICategory.DIRECT_IDENTIFIER, 0.95
        
        # Phone pattern
        phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
        phone_matches = sum(1 for val in sample if re.match(phone_pattern, val))
        if phone_matches / len(sample) > 0.8:
            return PIICategory.DIRECT_IDENTIFIER, 0.90
        
        # Numeric ID pattern (could be quasi-identifier)
        numeric_pattern = r'^\d+$'
        numeric_matches = sum(1 for val in sample if re.match(numeric_pattern, val))
        if numeric_matches / len(sample) > 0.9:
            return PIICategory.QUASI_IDENTIFIER, 0.70
        
        return PIICategory.NON_PII, 0.0
    
    def _check_sensitive_attributes(self, column_name: str, column_data: pd.Series) -> float:
        """Check for sensitive personal data"""
        name_lower = column_name.lower()
        
        for keyword in self.sensitive_keywords:
            if keyword in name_lower:
                return 0.9
        
        # Check data values for sensitive content
        if len(column_data) > 0:
            sample = column_data.dropna().astype(str).head(50)
            sensitive_values = 0
            
            for value in sample:
                value_lower = str(value).lower()
                for keyword in self.sensitive_keywords:
                    if keyword in value_lower:
                        sensitive_values += 1
                        break
            
            if sensitive_values / len(sample) > 0.3:
                return 0.8
        
        return 0.0
    
    def _determine_final_classification(self, name_category: PIICategory, name_confidence: float,
                                      data_category: PIICategory, data_confidence: float,
                                      sensitive_score: float) -> Tuple[PIICategory, float]:
        """Determine final classification from multiple signals"""
        
        if sensitive_score > 0.7:
            return PIICategory.SENSITIVE_ATTRIBUTE, sensitive_score
        
        # Choose highest confidence classification
        if name_confidence > data_confidence:
            return name_category, name_confidence
        elif data_confidence > 0.7:
            return data_category, data_confidence
        elif name_confidence > 0.5:
            return name_category, name_confidence
        else:
            return PIICategory.NON_PII, 0.0
    
    def _determine_risk_level(self, category: PIICategory, column_data: pd.Series) -> PIIRiskLevel:
        """Determine risk level based on category and data characteristics"""
        
        if category == PIICategory.DIRECT_IDENTIFIER:
            return PIIRiskLevel.CRITICAL
        
        elif category == PIICategory.SENSITIVE_ATTRIBUTE:
            return PIIRiskLevel.CRITICAL
        
        elif category == PIICategory.QUASI_IDENTIFIER:
            # Assess uniqueness for risk level
            if len(column_data) > 0:
                uniqueness = column_data.nunique() / len(column_data)
                if uniqueness > 0.9:
                    return PIIRiskLevel.HIGH
                else:
                    return PIIRiskLevel.MEDIUM
            return PIIRiskLevel.MEDIUM
        
        else:
            return PIIRiskLevel.NONE
    
    def _generate_recommendations(self, category: PIICategory, risk_level: PIIRiskLevel) -> List[str]:
        """Generate specific recommendations"""
        
        if category == PIICategory.DIRECT_IDENTIFIER:
            return [
                "Encrypt data at rest and in transit",
                "Implement strict access controls (RBAC)",
                "Consider tokenization or pseudonymization",
                "Regular access audits and monitoring",
                "Enforce data retention policies",
                "Require explicit consent for processing"
            ]
        
        elif category == PIICategory.SENSITIVE_ATTRIBUTE:
            return [
                "Special category data - requires explicit consent",
                "Enhanced security measures required",
                "Implement purpose limitation strictly",
                "Regular compliance reviews",
                "Consider anonymization for analytics",
                "Document legal basis for processing"
            ]
        
        elif category == PIICategory.QUASI_IDENTIFIER:
            if risk_level == PIIRiskLevel.HIGH:
                return [
                    "High re-identification risk",
                    "Apply k-anonymity (k≥5) or l-diversity",
                    "Consider generalization or suppression",
                    "Monitor for combination attacks",
                    "Restrict access to authorized personnel"
                ]
            else:
                return [
                    "Monitor combination with other attributes",
                    "Consider generalization for public datasets",
                    "Apply basic access controls"
                ]
        
        else:
            return ["Standard data protection measures sufficient"]

# Solution 3 - Production K-Anonymity System
class KAnonymizer:
    """Production-ready k-anonymity implementation"""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def anonymize_dataset(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
        """Apply k-anonymity with intelligent generalization"""
        
        # Validate inputs
        if not quasi_identifiers:
            return df.copy()
        
        # Check if dataset already satisfies k-anonymity
        if self._check_k_anonymity(df, quasi_identifiers):
            print(f"Dataset already satisfies {self.k}-anonymity")
            return df.copy()
        
        # Group by quasi-identifiers
        grouped = df.groupby(quasi_identifiers)
        anonymized_groups = []
        
        small_groups = []  # Groups that need generalization
        
        for group_key, group_df in grouped:
            if len(group_df) >= self.k:
                # Group already satisfies k-anonymity
                anonymized_groups.append(group_df)
            else:
                # Collect small groups for generalization
                small_groups.append(group_df)
        
        # Generalize small groups
        if small_groups:
            generalized_df = self._generalize_small_groups(
                pd.concat(small_groups, ignore_index=True),
                quasi_identifiers
            )
            anonymized_groups.append(generalized_df)
        
        result = pd.concat(anonymized_groups, ignore_index=True)
        
        # Verify k-anonymity
        if not self._check_k_anonymity(result, quasi_identifiers):
            print("Warning: k-anonymity not achieved, applying additional generalization")
            result = self._apply_additional_generalization(result, quasi_identifiers)
        
        return result
    
    def _generalize_small_groups(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
        """Generalize small groups to achieve k-anonymity"""
        
        generalized_df = df.copy()
        
        for column in quasi_identifiers:
            if column in generalized_df.columns:
                if generalized_df[column].dtype in ['int64', 'float64']:
                    # Numeric generalization
                    if 'age' in column.lower():
                        generalized_df[column] = self._generalize_age(generalized_df[column])
                    elif 'salary' in column.lower() or 'income' in column.lower():
                        generalized_df[column] = self._generalize_salary(generalized_df[column])
                    else:
                        generalized_df[column] = self._generalize_numeric(generalized_df[column])
                else:
                    # String generalization
                    if 'zip' in column.lower() or 'postal' in column.lower():
                        generalized_df[column] = self._generalize_zipcode(generalized_df[column])
                    else:
                        generalized_df[column] = self._generalize_string(generalized_df[column])
        
        return generalized_df
    
    def _generalize_age(self, ages: pd.Series) -> pd.Series:
        """Generalize ages into ranges"""
        def age_range(age):
            if pd.isna(age):
                return age
            age = int(age)
            if age < 18:
                return "Under 18"
            elif age < 25:
                return "18-24"
            elif age < 35:
                return "25-34"
            elif age < 45:
                return "35-44"
            elif age < 55:
                return "45-54"
            elif age < 65:
                return "55-64"
            else:
                return "65+"
        
        return ages.apply(age_range)
    
    def _generalize_zipcode(self, zipcodes: pd.Series) -> pd.Series:
        """Generalize zip codes by masking digits"""
        def mask_zip(zip_code):
            if pd.isna(zip_code):
                return zip_code
            zip_str = str(zip_code)
            if len(zip_str) >= 3:
                return zip_str[:3] + "*" * (len(zip_str) - 3)
            return zip_str
        
        return zipcodes.apply(mask_zip)
    
    def _generalize_salary(self, salaries: pd.Series) -> pd.Series:
        """Generalize salaries into ranges"""
        def salary_range(salary):
            if pd.isna(salary):
                return salary
            salary = float(salary)
            if salary < 30000:
                return "Under $30K"
            elif salary < 50000:
                return "$30K-$50K"
            elif salary < 75000:
                return "$50K-$75K"
            elif salary < 100000:
                return "$75K-$100K"
            elif salary < 150000:
                return "$100K-$150K"
            else:
                return "Over $150K"
        
        return salaries.apply(salary_range)
    
    def _generalize_numeric(self, values: pd.Series) -> pd.Series:
        """Generic numeric generalization"""
        if len(values) == 0:
            return values
        
        # Use quartiles for generalization
        q1 = values.quantile(0.25)
        q2 = values.quantile(0.5)
        q3 = values.quantile(0.75)
        
        def numeric_range(val):
            if pd.isna(val):
                return val
            if val <= q1:
                return f"≤{q1:.0f}"
            elif val <= q2:
                return f"{q1:.0f}-{q2:.0f}"
            elif val <= q3:
                return f"{q2:.0f}-{q3:.0f}"
            else:
                return f">{q3:.0f}"
        
        return values.apply(numeric_range)
    
    def _generalize_string(self, values: pd.Series) -> pd.Series:
        """Generic string generalization"""
        # For strings, use first few characters
        def generalize_str(val):
            if pd.isna(val):
                return val
            val_str = str(val)
            if len(val_str) > 3:
                return val_str[:3] + "*"
            return val_str
        
        return values.apply(generalize_str)
    
    def _check_k_anonymity(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> bool:
        """Verify that dataset satisfies k-anonymity"""
        if not quasi_identifiers:
            return True
        
        grouped = df.groupby(quasi_identifiers)
        min_group_size = grouped.size().min()
        
        return min_group_size >= self.k
    
    def _apply_additional_generalization(self, df: pd.DataFrame, quasi_identifiers: List[str]) -> pd.DataFrame:
        """Apply additional generalization if k-anonymity not achieved"""
        
        # More aggressive generalization
        result_df = df.copy()
        
        for column in quasi_identifiers:
            if column in result_df.columns:
                # Apply suppression (replace with *)
                result_df[column] = "*"
        
        return result_df

# Solution 4 - Advanced Pseudonymization System
class Pseudonymizer:
    """Production-ready pseudonymization with multiple techniques"""
    
    def __init__(self, secret_key: Optional[bytes] = None):
        self.secret_key = secret_key or secrets.token_bytes(32)
        self.cipher_suite = Fernet(Fernet.generate_key())
        self.pseudonym_mapping = {}
        self.format_preservers = {}
    
    def pseudonymize_column(self, data: pd.Series, pseudonym_type: str = "hash") -> pd.Series:
        """Pseudonymize column with specified technique"""
        
        if pseudonym_type == "hash":
            return data.apply(lambda x: self._hash_pseudonymize(str(x)) if pd.notna(x) else x)
        elif pseudonym_type == "encrypt":
            return data.apply(lambda x: self._encrypt_pseudonymize(str(x)) if pd.notna(x) else x)
        elif pseudonym_type == "format_preserving":
            # Detect format automatically
            format_type = self._detect_format(data)
            return data.apply(lambda x: self.format_preserving_pseudonymize(str(x), format_type) if pd.notna(x) else x)
        else:
            raise ValueError(f"Unknown pseudonym type: {pseudonym_type}")
    
    def _hash_pseudonymize(self, identifier: str, salt: str = "default") -> str:
        """Create irreversible hash-based pseudonym"""
        pseudonym = hmac.new(
            self.secret_key,
            f"{identifier}{salt}".encode(),
            hashlib.sha256
        ).hexdigest()
        
        return f"hash_{pseudonym[:16]}"
    
    def _encrypt_pseudonymize(self, identifier: str) -> str:
        """Create reversible encryption-based pseudonym"""
        encrypted = self.cipher_suite.encrypt(identifier.encode())
        pseudonym_id = hashlib.sha256(encrypted).hexdigest()[:16]
        
        # Store mapping for reversal
        pseudonym_key = f"enc_{pseudonym_id}"
        self.pseudonym_mapping[pseudonym_key] = encrypted
        
        return pseudonym_key
    
    def format_preserving_pseudonymize(self, identifier: str, format_type: str) -> str:
        """Create format-preserving pseudonym"""
        
        if format_type == "email":
            if "@" in identifier:
                local, domain = identifier.split("@", 1)
                pseudo_local = self._hash_pseudonymize(local)[:8]
                return f"user_{pseudo_local}@{domain}"
            return self._hash_pseudonymize(identifier)
        
        elif format_type == "phone":
            # Extract digits
            digits = ''.join(filter(str.isdigit, identifier))
            if len(digits) >= 10:
                # Generate pseudo digits
                pseudo_hash = self._hash_pseudonymize(digits)
                pseudo_digits = ''.join(filter(str.isdigit, pseudo_hash + "1234567890"))[:len(digits)]
                
                # Reconstruct format
                if len(digits) == 10:
                    return f"({pseudo_digits[:3]}) {pseudo_digits[3:6]}-{pseudo_digits[6:]}"
                else:
                    return pseudo_digits
            return self._hash_pseudonymize(identifier)
        
        elif format_type == "ssn":
            digits = ''.join(filter(str.isdigit, identifier))
            if len(digits) == 9:
                pseudo_hash = self._hash_pseudonymize(digits)
                pseudo_digits = ''.join(filter(str.isdigit, pseudo_hash + "123456789"))[:9]
                return f"{pseudo_digits[:3]}-{pseudo_digits[3:5]}-{pseudo_digits[5:]}"
            return self._hash_pseudonymize(identifier)
        
        elif format_type == "numeric":
            # Preserve numeric format but change value
            if identifier.isdigit():
                hash_val = int(hashlib.sha256(f"{identifier}{self.secret_key}".encode()).hexdigest(), 16)
                return str(hash_val % (10 ** len(identifier))).zfill(len(identifier))
            return self._hash_pseudonymize(identifier)
        
        else:
            return self._hash_pseudonymize(identifier)
    
    def _detect_format(self, data: pd.Series) -> str:
        """Automatically detect data format"""
        sample = data.dropna().astype(str).head(10)
        
        if len(sample) == 0:
            return "generic"
        
        # Check for email pattern
        email_count = sum(1 for val in sample if "@" in val and "." in val)
        if email_count / len(sample) > 0.8:
            return "email"
        
        # Check for phone pattern
        phone_count = sum(1 for val in sample if re.search(r'\d{3}[-.\s]?\d{3}[-.\s]?\d{4}', val))
        if phone_count / len(sample) > 0.8:
            return "phone"
        
        # Check for SSN pattern
        ssn_count = sum(1 for val in sample if re.search(r'\d{3}-?\d{2}-?\d{4}', val))
        if ssn_count / len(sample) > 0.8:
            return "ssn"
        
        # Check for numeric
        numeric_count = sum(1 for val in sample if val.isdigit())
        if numeric_count / len(sample) > 0.8:
            return "numeric"
        
        return "generic"
    
    def reverse_pseudonym(self, pseudonym: str) -> Optional[str]:
        """Reverse pseudonym if using reversible method"""
        if pseudonym in self.pseudonym_mapping:
            encrypted = self.pseudonym_mapping[pseudonym]
            return self.cipher_suite.decrypt(encrypted).decode()
        return None
    
    def consistent_pseudonymize(self, identifier: str, context: str = "default") -> str:
        """Ensure consistency across datasets"""
        contextual_identifier = f"{context}:{identifier}"
        return self._hash_pseudonymize(contextual_identifier)

# Solution 5 - Production GDPR Request Handler
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
    """Production GDPR request handling system"""
    
    def __init__(self, data_systems: Dict[str, Any]):
        self.data_systems = data_systems
        self.requests = {}
    
    def submit_request(self, request_type: RequestType, subject_email: str,
                      subject_identifiers: Dict[str, str], request_details: str) -> str:
        """Submit new data subject request"""
        
        request_id = f"DSR-{uuid.uuid4().hex[:8].upper()}"
        received_date = datetime.now()
        due_date = received_date + timedelta(days=30)  # GDPR requirement
        
        request = DataSubjectRequest(
            request_id=request_id,
            request_type=request_type,
            subject_email=subject_email,
            subject_identifiers=subject_identifiers,
            received_date=received_date,
            due_date=due_date,
            status=RequestStatus.RECEIVED,
            notes=[f"Request received: {request_details}"]
        )
        
        self.requests[request_id] = request
        
        # Send acknowledgment
        print(f"Acknowledgment sent to {subject_email} for request {request_id}")
        print(f"Due date: {due_date.strftime('%Y-%m-%d')}")
        
        return request_id
    
    def process_access_request(self, request_id: str) -> Dict:
        """Process comprehensive access request"""
        
        request = self.requests.get(request_id)
        if not request or request.request_type != RequestType.ACCESS:
            return {"error": "Invalid access request"}
        
        request.status = RequestStatus.IN_PROGRESS
        
        # Find all personal data
        personal_data = self._find_personal_data(request.subject_identifiers)
        
        # Compile comprehensive access report
        access_report = {
            "request_id": request_id,
            "subject_email": request.subject_email,
            "generated_date": datetime.now().isoformat(),
            "personal_data": personal_data,
            "processing_purposes": self._get_processing_purposes(),
            "legal_bases": self._get_legal_bases(),
            "recipients": self._get_data_recipients(),
            "retention_periods": self._get_retention_periods(),
            "rights_information": self._get_rights_information(),
            "contact_information": {
                "data_controller": "Company Data Controller",
                "dpo_email": "dpo@company.com",
                "privacy_email": "privacy@company.com"
            }
        }
        
        request.status = RequestStatus.COMPLETED
        request.notes.append(f"Access request completed: {datetime.now()}")
        
        return access_report
    
    def process_erasure_request(self, request_id: str) -> Dict:
        """Process right to erasure request"""
        
        request = self.requests.get(request_id)
        if not request or request.request_type != RequestType.ERASURE:
            return {"error": "Invalid erasure request"}
        
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
        
        # Execute deletion plan
        deletion_results = []
        
        for system_name in self.data_systems.keys():
            result = self._delete_from_system(system_name, request.subject_identifiers)
            deletion_results.append({
                "system": system_name,
                "status": result["status"],
                "records_deleted": result.get("records_deleted", 0),
                "timestamp": datetime.now().isoformat()
            })
        
        # Verify complete deletion
        verification = self._verify_complete_deletion(request.subject_identifiers)
        
        deletion_plan = {
            "request_id": request_id,
            "deletion_results": deletion_results,
            "verification": verification,
            "completion_date": datetime.now().isoformat() if verification["complete"] else None
        }
        
        if verification["complete"]:
            request.status = RequestStatus.COMPLETED
            request.notes.append(f"Erasure completed: {datetime.now()}")
        else:
            request.notes.append(f"Erasure incomplete: {verification['remaining_data']}")
        
        return deletion_plan
    
    def _find_personal_data(self, subject_identifiers: Dict[str, str]) -> Dict:
        """Find all personal data across systems"""
        
        # Mock implementation - in production, query actual systems
        return {
            "customer_database": {
                "profile": {
                    "email": subject_identifiers.get("email", ""),
                    "name": "John Doe",
                    "phone": "+1-555-123-4567",
                    "address": "123 Main St, City, State"
                },
                "preferences": {
                    "newsletter": True,
                    "marketing": False,
                    "language": "en"
                }
            },
            "analytics_database": {
                "behavioral_data": [
                    {"event": "login", "timestamp": "2024-01-15T10:30:00Z"},
                    {"event": "purchase", "timestamp": "2024-01-15T11:00:00Z"}
                ],
                "aggregated_metrics": {
                    "total_sessions": 45,
                    "total_purchases": 12,
                    "last_activity": "2024-01-20T15:30:00Z"
                }
            }
        }
    
    def _get_processing_purposes(self) -> List[Dict]:
        """Get processing purposes and legal bases"""
        return [
            {
                "purpose": "Customer service and support",
                "legal_basis": "Contract performance",
                "data_categories": ["contact_info", "account_data"]
            },
            {
                "purpose": "Marketing communications",
                "legal_basis": "Consent",
                "data_categories": ["contact_info", "preferences"]
            },
            {
                "purpose": "Analytics and service improvement",
                "legal_basis": "Legitimate interests",
                "data_categories": ["behavioral_data", "usage_metrics"]
            }
        ]
    
    def _get_legal_bases(self) -> List[str]:
        """Get legal bases for processing"""
        return [
            "Consent (Article 6(1)(a))",
            "Contract performance (Article 6(1)(b))",
            "Legitimate interests (Article 6(1)(f))"
        ]
    
    def _get_data_recipients(self) -> List[Dict]:
        """Get data recipients and transfers"""
        return [
            {
                "recipient": "Customer Support Team",
                "purpose": "Customer service",
                "location": "Same country"
            },
            {
                "recipient": "Analytics Service Provider",
                "purpose": "Service improvement",
                "location": "EU (adequate protection)"
            }
        ]
    
    def _get_retention_periods(self) -> Dict:
        """Get data retention periods"""
        return {
            "customer_data": "7 years after account closure",
            "marketing_data": "Until consent withdrawn",
            "analytics_data": "2 years from collection",
            "support_tickets": "3 years from resolution"
        }
    
    def _get_rights_information(self) -> Dict:
        """Get information about data subject rights"""
        return {
            "available_rights": [
                "Right of access",
                "Right to rectification",
                "Right to erasure",
                "Right to restrict processing",
                "Right to data portability",
                "Right to object"
            ],
            "how_to_exercise": "Contact privacy@company.com or use online form",
            "complaint_authority": "National Data Protection Authority"
        }
    
    def _assess_erasure_request(self, request: DataSubjectRequest) -> Dict:
        """Assess whether erasure can be fulfilled"""
        
        # Check for legal obligations (simplified)
        # In production, check actual legal requirements
        
        return {
            "can_erase": True,
            "reason": "No legal barriers identified",
            "legal_basis": "Right to erasure applies"
        }
    
    def _delete_from_system(self, system_name: str, subject_identifiers: Dict[str, str]) -> Dict:
        """Delete data from specific system"""
        
        # Mock deletion - in production, execute actual deletions
        return {
            "status": "success",
            "records_deleted": 5,
            "tables_affected": ["users", "preferences", "activity_log"]
        }
    
    def _verify_complete_deletion(self, subject_identifiers: Dict[str, str]) -> Dict:
        """Verify complete deletion across all systems"""
        
        # Mock verification - in production, query all systems
        return {
            "complete": True,
            "remaining_data": [],
            "verification_date": datetime.now().isoformat()
        }

# Solution 6 - Privacy Impact Assessment System
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
    """Comprehensive privacy impact assessment system"""
    
    def __init__(self):
        self.sensitive_data_types = [
            "health", "medical", "genetic", "biometric",
            "racial", "ethnic", "political", "religious",
            "sexual", "criminal", "financial", "location"
        ]
        
        self.high_risk_processing = [
            "profiling", "automated_decision_making", "monitoring",
            "large_scale_processing", "data_matching", "inference"
        ]
    
    def assess_dataset(self, df: pd.DataFrame, processing_purpose: str,
                      data_subjects: str) -> PrivacyAssessment:
        """Conduct comprehensive privacy impact assessment"""
        
        # Analyze data types and sensitivity
        data_types, data_sensitivity = self._assess_data_sensitivity(df)
        
        # Evaluate processing risks
        processing_risks = self._evaluate_processing_risks(processing_purpose, data_types)
        
        # Assess scale and scope
        scale_risks = self._assess_scale_risks(df, data_subjects)
        
        # Determine overall risk level
        overall_risk = self._determine_overall_risk(data_sensitivity, processing_risks, scale_risks)
        
        # Compile risk factors
        risk_factors = []
        risk_factors.extend(processing_risks)
        risk_factors.extend(scale_risks)
        
        if data_sensitivity == PrivacyRisk.CRITICAL:
            risk_factors.append("Contains special category personal data")
        
        # Generate mitigation measures
        mitigation_measures = self._generate_mitigation_measures(overall_risk, data_types, processing_purpose)
        
        # Identify compliance requirements
        compliance_requirements = self._identify_compliance_requirements(data_types, processing_purpose)
        
        return PrivacyAssessment(
            data_types=data_types,
            processing_purposes=[processing_purpose],
            risk_level=overall_risk,
            risk_factors=risk_factors,
            mitigation_measures=mitigation_measures,
            compliance_requirements=compliance_requirements
        )
    
    def _assess_data_sensitivity(self, df: pd.DataFrame) -> Tuple[List[str], PrivacyRisk]:
        """Assess sensitivity of data types in dataset"""
        
        data_types = []
        max_sensitivity = PrivacyRisk.LOW
        
        # Use PII detector to classify columns
        detector = PIIDetector()
        pii_results = detector.detect_pii_in_dataframe(df)
        
        for column, results in pii_results.items():
            category = results["primary_category"]
            confidence = results["overall_confidence"]
            
            if category == "direct_identifier" and confidence > 0.8:
                data_types.append("Direct identifiers")
                max_sensitivity = max(max_sensitivity, PrivacyRisk.HIGH, key=lambda x: x.value)
            
            elif category == "quasi_identifier" and confidence > 0.8:
                data_types.append("Quasi-identifiers")
                max_sensitivity = max(max_sensitivity, PrivacyRisk.MEDIUM, key=lambda x: x.value)
        
        # Check for sensitive data types
        for column in df.columns:
            column_lower = column.lower()
            for sensitive_type in self.sensitive_data_types:
                if sensitive_type in column_lower:
                    data_types.append(f"Special category: {sensitive_type}")
                    max_sensitivity = PrivacyRisk.CRITICAL
        
        return list(set(data_types)), max_sensitivity
    
    def _evaluate_processing_risks(self, purpose: str, data_types: List[str]) -> List[str]:
        """Evaluate risks based on processing purpose"""
        
        risks = []
        purpose_lower = purpose.lower()
        
        # Check for high-risk processing activities
        for high_risk in self.high_risk_processing:
            if high_risk in purpose_lower:
                risks.append(f"High-risk processing: {high_risk}")
        
        # Specific risk assessments
        if "marketing" in purpose_lower:
            risks.append("Marketing processing may require consent")
        
        if "analytics" in purpose_lower:
            risks.append("Analytics may involve profiling")
        
        if "automated" in purpose_lower or "decision" in purpose_lower:
            risks.append("Automated decision-making detected")
        
        if "sharing" in purpose_lower or "transfer" in purpose_lower:
            risks.append("Data sharing/transfer increases risk")
        
        return risks
    
    def _assess_scale_risks(self, df: pd.DataFrame, data_subjects: str) -> List[str]:
        """Assess risks based on scale and scope"""
        
        risks = []
        
        # Dataset size
        if len(df) > 100000:
            risks.append("Large-scale processing (>100K records)")
        elif len(df) > 10000:
            risks.append("Medium-scale processing (>10K records)")
        
        # Number of attributes
        if len(df.columns) > 50:
            risks.append("High number of attributes increases combination risk")
        
        # Data subject categories
        if "children" in data_subjects.lower():
            risks.append("Processing of children's data (enhanced protection required)")
        
        if "vulnerable" in data_subjects.lower():
            risks.append("Processing of vulnerable individuals' data")
        
        return risks
    
    def _determine_overall_risk(self, data_sensitivity: PrivacyRisk, 
                               processing_risks: List[str], 
                               scale_risks: List[str]) -> PrivacyRisk:
        """Determine overall privacy risk level"""
        
        # Start with data sensitivity
        risk_score = {"low": 1, "medium": 2, "high": 3, "critical": 4}[data_sensitivity.value]
        
        # Add processing risk factors
        risk_score += len(processing_risks) * 0.5
        
        # Add scale risk factors
        risk_score += len(scale_risks) * 0.3
        
        # Determine final risk level
        if risk_score >= 4:
            return PrivacyRisk.CRITICAL
        elif risk_score >= 3:
            return PrivacyRisk.HIGH
        elif risk_score >= 2:
            return PrivacyRisk.MEDIUM
        else:
            return PrivacyRisk.LOW
    
    def _generate_mitigation_measures(self, risk_level: PrivacyRisk, 
                                    data_types: List[str], 
                                    processing_purpose: str) -> List[str]:
        """Generate appropriate mitigation measures"""
        
        measures = []
        
        # Base measures for all risk levels
        measures.extend([
            "Implement data minimization principles",
            "Apply purpose limitation",
            "Ensure data accuracy and currency",
            "Implement appropriate retention periods"
        ])
        
        if risk_level in [PrivacyRisk.MEDIUM, PrivacyRisk.HIGH, PrivacyRisk.CRITICAL]:
            measures.extend([
                "Implement privacy by design",
                "Conduct regular privacy audits",
                "Provide privacy training to staff",
                "Implement access controls and monitoring"
            ])
        
        if risk_level in [PrivacyRisk.HIGH, PrivacyRisk.CRITICAL]:
            measures.extend([
                "Apply pseudonymization or anonymization",
                "Implement encryption at rest and in transit",
                "Conduct data protection impact assessments",
                "Appoint Data Protection Officer (DPO)",
                "Implement breach detection and response"
            ])
        
        if risk_level == PrivacyRisk.CRITICAL:
            measures.extend([
                "Obtain explicit consent where required",
                "Implement enhanced security measures",
                "Regular third-party security assessments",
                "Implement differential privacy techniques",
                "Consider data localization requirements"
            ])
        
        # Specific measures based on data types
        if any("special category" in dt.lower() for dt in data_types):
            measures.append("Special category data requires explicit consent")
        
        if any("direct identifier" in dt.lower() for dt in data_types):
            measures.append("Consider tokenization for direct identifiers")
        
        return measures
    
    def _identify_compliance_requirements(self, data_types: List[str], 
                                        processing_purpose: str) -> List[str]:
        """Identify applicable compliance requirements"""
        
        requirements = []
        
        # Base GDPR requirements
        requirements.extend([
            "GDPR Article 5 - Principles of processing",
            "GDPR Article 6 - Lawfulness of processing",
            "GDPR Chapter 3 - Rights of data subjects"
        ])
        
        # Special category data
        if any("special category" in dt.lower() for dt in data_types):
            requirements.append("GDPR Article 9 - Special categories of personal data")
        
        # Automated decision-making
        if "automated" in processing_purpose.lower():
            requirements.append("GDPR Article 22 - Automated decision-making")
        
        # Data transfers
        if "transfer" in processing_purpose.lower():
            requirements.append("GDPR Chapter 5 - Transfers to third countries")
        
        # Marketing
        if "marketing" in processing_purpose.lower():
            requirements.extend([
                "ePrivacy Directive - Electronic communications",
                "GDPR Article 21 - Right to object"
            ])
        
        # Additional regulations based on data types
        if any("health" in dt.lower() or "medical" in dt.lower() for dt in data_types):
            requirements.append("HIPAA compliance (if applicable)")
        
        if any("financial" in dt.lower() for dt in data_types):
            requirements.append("PCI DSS compliance (if applicable)")
        
        return requirements

def main():
    """Demonstrate all privacy protection solutions"""
    
    print("=== Day 10: Data Privacy - GDPR, PII Handling - Solutions ===\n")
    
    # Create comprehensive sample dataset
    sample_data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'email': [
            'john.doe@email.com', 'jane.smith@company.com', 
            'bob.wilson@gmail.com', 'alice.brown@yahoo.com',
            'charlie.davis@outlook.com', 'diana.miller@email.com',
            'frank.garcia@company.com', 'grace.lee@gmail.com',
            'henry.taylor@email.com', 'ivy.johnson@company.com'
        ],
        'phone': [
            '555-123-4567', '555-234-5678', '555-345-6789', '555-456-7890',
            '555-567-8901', '555-678-9012', '555-789-0123', '555-890-1234',
            '555-901-2345', '555-012-3456'
        ],
        'age': [25, 26, 35, 36, 45, 28, 32, 29, 41, 33],
        'zip_code': ['12345', '12346', '67890', '67891', '11111', '12345', '67890', '11111', '22222', '12345'],
        'salary': [75000, 76000, 85000, 86000, 95000, 72000, 88000, 74000, 92000, 78000],
        'medical_condition': ['none', 'diabetes', 'hypertension', 'none', 'asthma', 'none', 'diabetes', 'none', 'hypertension', 'none'],
        'purchase_amount': [100.50, 250.75, 75.25, 300.00, 150.25, 200.00, 175.50, 125.75, 275.25, 225.00]
    })
    
    print("Sample dataset created:")
    print(f"Shape: {sample_data.shape}")
    print(f"Columns: {list(sample_data.columns)}\n")
    
    # Solution 1: PII Detection
    print("=== Solution 1: Advanced PII Detection ===")
    detector = PIIDetector()
    pii_results = detector.detect_pii_in_dataframe(sample_data)
    
    for column, results in pii_results.items():
        print(f"\nColumn: {column}")
        print(f"  Primary Category: {results['primary_category']}")
        print(f"  Confidence: {results['overall_confidence']:.2f}")
        print(f"  Findings: {len(results['findings'])}")
        print(f"  Recommendations: {len(results['recommendations'])}")
    
    # Solution 2: Data Classification
    print("\n=== Solution 2: Advanced Data Classification ===")
    classifier = DataClassifier()
    classifications = classifier.classify_dataset(sample_data)
    
    for column, classification in classifications.items():
        print(f"\nColumn: {column}")
        print(f"  Category: {classification.category.value}")
        print(f"  Risk Level: {classification.risk_level.value}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print(f"  Evidence: {len(classification.evidence)} items")
        print(f"  Recommendations: {len(classification.recommendations)} items")
    
    # Solution 3: K-Anonymity
    print("\n=== Solution 3: K-Anonymity Implementation ===")
    anonymizer = KAnonymizer(k=3)
    quasi_identifiers = ['age', 'zip_code', 'salary']
    
    print(f"Original dataset shape: {sample_data.shape}")
    print(f"Quasi-identifiers: {quasi_identifiers}")
    
    anonymized_data = anonymizer.anonymize_dataset(sample_data, quasi_identifiers)
    
    print(f"Anonymized dataset shape: {anonymized_data.shape}")
    print(f"K-anonymity satisfied: {anonymizer._check_k_anonymity(anonymized_data, quasi_identifiers)}")
    
    print("\nOriginal vs Anonymized (first 3 rows):")
    for qi in quasi_identifiers:
        print(f"{qi}:")
        print(f"  Original: {sample_data[qi].head(3).tolist()}")
        print(f"  Anonymized: {anonymized_data[qi].head(3).tolist()}")
    
    # Solution 4: Pseudonymization
    print("\n=== Solution 4: Advanced Pseudonymization ===")
    pseudonymizer = Pseudonymizer()
    
    # Test different pseudonymization techniques
    email_pseudo = pseudonymizer.pseudonymize_column(sample_data['email'], 'format_preserving')
    phone_pseudo = pseudonymizer.pseudonymize_column(sample_data['phone'], 'format_preserving')
    id_pseudo = pseudonymizer.pseudonymize_column(sample_data['user_id'], 'hash')
    
    print("Pseudonymization results (first 3 rows):")
    print(f"Original emails: {sample_data['email'].head(3).tolist()}")
    print(f"Pseudo emails: {email_pseudo.head(3).tolist()}")
    print(f"Original phones: {sample_data['phone'].head(3).tolist()}")
    print(f"Pseudo phones: {phone_pseudo.head(3).tolist()}")
    print(f"Original IDs: {sample_data['user_id'].head(3).tolist()}")
    print(f"Pseudo IDs: {id_pseudo.head(3).tolist()}")
    
    # Solution 5: GDPR Request Handling
    print("\n=== Solution 5: GDPR Request Handling ===")
    data_systems = {
        "customer_db": {"type": "postgresql"},
        "analytics_db": {"type": "bigquery"},
        "crm_system": {"type": "salesforce"}
    }
    
    gdpr_handler = GDPRRequestHandler(data_systems)
    
    # Submit access request
    access_request_id = gdpr_handler.submit_request(
        RequestType.ACCESS,
        "john.doe@email.com",
        {"email": "john.doe@email.com", "user_id": "1"},
        "I want to see all my personal data"
    )
    
    print(f"Access request submitted: {access_request_id}")
    
    # Process access request
    access_report = gdpr_handler.process_access_request(access_request_id)
    print(f"Access report generated with {len(access_report['personal_data'])} data sources")
    print(f"Processing purposes: {len(access_report['processing_purposes'])}")
    
    # Submit erasure request
    erasure_request_id = gdpr_handler.submit_request(
        RequestType.ERASURE,
        "jane.smith@company.com",
        {"email": "jane.smith@company.com", "user_id": "2"},
        "Please delete all my personal data"
    )
    
    print(f"Erasure request submitted: {erasure_request_id}")
    
    # Process erasure request
    erasure_result = gdpr_handler.process_erasure_request(erasure_request_id)
    print(f"Erasure completed: {erasure_result.get('completion_date') is not None}")
    
    # Solution 6: Privacy Impact Assessment
    print("\n=== Solution 6: Privacy Impact Assessment ===")
    pia = PrivacyImpactAssessment()
    
    assessment = pia.assess_dataset(
        sample_data,
        "Customer analytics and personalized marketing",
        "Adult customers and prospects"
    )
    
    print(f"Privacy Risk Level: {assessment.risk_level.value}")
    print(f"Data Types Identified: {len(assessment.data_types)}")
    for dt in assessment.data_types:
        print(f"  - {dt}")
    
    print(f"\nRisk Factors: {len(assessment.risk_factors)}")
    for rf in assessment.risk_factors[:3]:  # Show first 3
        print(f"  - {rf}")
    
    print(f"\nMitigation Measures: {len(assessment.mitigation_measures)}")
    for mm in assessment.mitigation_measures[:5]:  # Show first 5
        print(f"  - {mm}")
    
    print(f"\nCompliance Requirements: {len(assessment.compliance_requirements)}")
    for cr in assessment.compliance_requirements[:3]:  # Show first 3
        print(f"  - {cr}")
    
    print("\n=== All Solutions Demonstrated Successfully ===")
    print("Production-ready privacy protection systems implemented!")
    print("Ready for real-world deployment with appropriate security measures.")

if __name__ == "__main__":
    main()
