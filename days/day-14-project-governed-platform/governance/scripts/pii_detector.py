#!/usr/bin/env python3
"""
PII Detection and Protection System
Automated detection and protection of personally identifiable information
"""

import re
import hashlib
import logging
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class PIIField:
    """Represents a field containing PII"""
    table: str
    column: str
    pii_type: str
    confidence: float
    protection_method: str

class PIIDetector:
    """Advanced PII detection and protection system"""
    
    def __init__(self):
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
            'phone': r'\b\d{3}-\d{3}-\d{4}\b',
            'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'ip_address': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'
        }
        
    def scan_for_pii(self, data: Dict[str, Any]) -> List[PIIField]:
        """Scan data for PII patterns"""
        detected_pii = []
        
        for table, columns in data.items():
            for column, values in columns.items():
                pii_type = self._detect_pii_type(column, values)
                if pii_type:
                    detected_pii.append(PIIField(
                        table=table,
                        column=column,
                        pii_type=pii_type,
                        confidence=0.95,
                        protection_method=self._get_protection_method(pii_type)
                    ))
        
        return detected_pii
    
    def _detect_pii_type(self, column_name: str, sample_values: List[str]) -> str:
        """Detect PII type based on column name and sample values"""
        column_lower = column_name.lower()
        
        # Check column name patterns
        if 'email' in column_lower:
            return 'email'
        elif 'phone' in column_lower or 'mobile' in column_lower:
            return 'phone'
        elif 'ssn' in column_lower or 'social' in column_lower:
            return 'ssn'
        
        # Check sample values against patterns
        for value in sample_values[:10]:  # Check first 10 values
            if isinstance(value, str):
                for pii_type, pattern in self.pii_patterns.items():
                    if re.search(pattern, value):
                        return pii_type
        
        return None
    
    def _get_protection_method(self, pii_type: str) -> str:
        """Get appropriate protection method for PII type"""
        protection_map = {
            'email': 'hash',
            'phone': 'mask',
            'ssn': 'encrypt',
            'credit_card': 'encrypt',
            'ip_address': 'hash'
        }
        return protection_map.get(pii_type, 'mask')
    
    def protect_pii(self, value: str, method: str, salt: str = "datacorp_salt") -> str:
        """Apply protection to PII value"""
        if method == 'hash':
            return hashlib.sha256((value + salt).encode()).hexdigest()
        elif method == 'mask':
            if '@' in value:  # Email
                parts = value.split('@')
                return f"{parts[0][:2]}***@{parts[1]}"
            elif '-' in value and len(value) == 12:  # Phone
                return f"{value[:3]}-XXX-{value[-4:]}"
            else:
                return value[:2] + '*' * (len(value) - 4) + value[-2:]
        elif method == 'encrypt':
            # Simplified encryption (use proper encryption in production)
            return f"ENCRYPTED_{hashlib.md5(value.encode()).hexdigest()}"
        
        return value

if __name__ == "__main__":
    # Test the PII detector
    detector = PIIDetector()
    
    sample_data = {
        'customers': {
            'email': ['john.doe@example.com', 'jane.smith@test.com'],
            'phone': ['555-123-4567', '555-987-6543'],
            'name': ['John Doe', 'Jane Smith']
        }
    }
    
    detected = detector.scan_for_pii(sample_data)
    print(f"Detected PII fields: {len(detected)}")
    
    for field in detected:
        print(f"- {field.table}.{field.column}: {field.pii_type} ({field.protection_method})")
        
        # Test protection
        sample_value = sample_data[field.table][field.column][0]
        protected = detector.protect_pii(sample_value, field.protection_method)
        print(f"  Original: {sample_value} -> Protected: {protected}")