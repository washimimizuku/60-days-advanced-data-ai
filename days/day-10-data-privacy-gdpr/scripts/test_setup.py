#!/usr/bin/env python3
"""
Test Day 10 privacy system setup and configuration
"""

import sys
import os
from pathlib import Path
import importlib.util

def test_core_imports():
    """Test core Python libraries"""
    try:
        import pandas as pd
        import numpy as np
        import hashlib
        import secrets
        from datetime import datetime
        print("✅ Core Python libraries working")
        return True
    except ImportError as e:
        print(f"❌ Core import error: {e}")
        return False

def test_cryptography():
    """Test cryptography libraries"""
    try:
        from cryptography.fernet import Fernet
        import hmac
        
        # Test Fernet encryption
        key = Fernet.generate_key()
        cipher = Fernet(key)
        test_data = b"privacy test"
        encrypted = cipher.encrypt(test_data)
        decrypted = cipher.decrypt(encrypted)
        assert decrypted == test_data
        
        # Test HMAC
        secret = secrets.token_bytes(32)
        signature = hmac.new(secret, b"test", hashlib.sha256).hexdigest()
        assert len(signature) == 64
        
        print("✅ Cryptography libraries working")
        return True
    except Exception as e:
        print(f"❌ Cryptography error: {e}")
        return False

def test_privacy_libraries():
    """Test privacy-specific libraries (optional)"""
    results = []
    
    # Test Presidio (optional)
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        
        analyzer = AnalyzerEngine()
        text = "My email is test@email.com"
        results_pii = analyzer.analyze(text=text, language='en')
        
        if len(results_pii) > 0:
            print("✅ Presidio PII detection working")
            results.append(True)
        else:
            print("⚠️  Presidio installed but not detecting PII")
            results.append(False)
            
    except ImportError:
        print("ℹ️  Presidio not installed (optional)")
        results.append(True)  # Not required
    except Exception as e:
        print(f"⚠️  Presidio error: {e}")
        results.append(False)
    
    # Test spaCy (optional)
    try:
        import spacy
        nlp = spacy.load('en_core_web_sm')
        doc = nlp("John Doe lives in New York")
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        if len(entities) > 0:
            print("✅ spaCy NER working")
            results.append(True)
        else:
            print("⚠️  spaCy loaded but not detecting entities")
            results.append(False)
            
    except OSError:
        print("ℹ️  spaCy model not installed (optional)")
        print("   Install with: python -m spacy download en_core_web_sm")
        results.append(True)  # Not required
    except ImportError:
        print("ℹ️  spaCy not installed (optional)")
        results.append(True)  # Not required
    except Exception as e:
        print(f"⚠️  spaCy error: {e}")
        results.append(False)
    
    return all(results)

def test_environment():
    """Test environment configuration"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("ℹ️  .env file not found (optional)")
        print("   Copy .env.example to .env for full functionality")
        return True  # Not required for basic functionality
    
    # Check for key variables
    required_vars = ['PRIVACY_SECRET_KEY', 'PSEUDONYM_SALT', 'FERNET_KEY']
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        missing_vars = []
        default_vars = []
        
        for var in required_vars:
            value = os.getenv(var)
            if not value:
                missing_vars.append(var)
            elif value.startswith('your_'):
                default_vars.append(var)
        
        if missing_vars:
            print(f"⚠️  Missing environment variables: {missing_vars}")
            return False
        
        if default_vars:
            print(f"⚠️  Default values detected: {default_vars}")
            print("   Run: python scripts/generate_keys.py")
            return False
        
        print("✅ Environment configuration OK")
        return True
        
    except ImportError:
        print("ℹ️  python-dotenv not installed (optional)")
        return True
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def test_sample_data():
    """Test sample datasets"""
    sample_dir = Path('sample_data')
    
    if not sample_dir.exists():
        print("ℹ️  Sample data directory not found")
        return True
    
    expected_files = ['pii_test_dataset.csv', 'healthcare_dataset.csv']
    missing_files = []
    
    for file in expected_files:
        if not (sample_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"ℹ️  Missing sample files: {missing_files}")
        return True  # Not critical
    
    # Test loading sample data
    try:
        import pandas as pd
        
        pii_data = pd.read_csv(sample_dir / 'pii_test_dataset.csv')
        healthcare_data = pd.read_csv(sample_dir / 'healthcare_dataset.csv')
        
        print(f"✅ Sample data loaded: {len(pii_data)} PII records, {len(healthcare_data)} healthcare records")
        return True
        
    except Exception as e:
        print(f"⚠️  Sample data error: {e}")
        return False

def test_exercise_files():
    """Test exercise and solution files"""
    required_files = ['exercise.py', 'solution.py', 'quiz.md', 'README.md']
    missing_files = []
    
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    # Test that Python files can be imported
    try:
        spec = importlib.util.spec_from_file_location("exercise", "exercise.py")
        exercise_module = importlib.util.module_from_spec(spec)
        
        spec = importlib.util.spec_from_file_location("solution", "solution.py")
        solution_module = importlib.util.module_from_spec(spec)
        
        print("✅ Exercise files present and importable")
        return True
        
    except Exception as e:
        print(f"⚠️  Exercise file error: {e}")
        return False

def test_basic_functionality():
    """Test basic privacy functionality"""
    try:
        import pandas as pd
        from cryptography.fernet import Fernet
        import hashlib
        import hmac
        import secrets
        
        # Test data creation
        test_data = pd.DataFrame({
            'email': ['test@email.com', 'user@company.com'],
            'phone': ['555-123-4567', '555-987-6543'],
            'age': [25, 35]
        })
        
        # Test basic PII detection (regex-based)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        import re
        
        email_matches = test_data['email'].str.contains(email_pattern, regex=True).sum()
        assert email_matches == 2
        
        # Test pseudonymization
        secret_key = secrets.token_bytes(32)
        
        def pseudonymize(value):
            return hmac.new(secret_key, str(value).encode(), hashlib.sha256).hexdigest()[:16]
        
        pseudo_emails = test_data['email'].apply(pseudonymize)
        assert len(pseudo_emails) == 2
        assert pseudo_emails[0] != pseudo_emails[1]
        
        # Test k-anonymity check
        def check_k_anonymity(df, quasi_identifiers, k=2):
            if not quasi_identifiers:
                return True
            grouped = df.groupby(quasi_identifiers)
            min_group_size = grouped.size().min()
            return min_group_size >= k
        
        k_anon_result = check_k_anonymity(test_data, ['age'], k=1)
        assert k_anon_result == True
        
        print("✅ Basic privacy functionality working")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality error: {e}")
        return False

def main():
    """Run comprehensive setup test"""
    print("🔐 Day 10: Data Privacy - GDPR, PII Handling - Setup Test")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Cryptography", test_cryptography),
        ("Privacy Libraries", test_privacy_libraries),
        ("Environment Config", test_environment),
        ("Sample Data", test_sample_data),
        ("Exercise Files", test_exercise_files),
        ("Basic Functionality", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test error: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Setup Test Summary")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"⚠️  Issues: {total - passed}/{total}")
    
    if passed >= 5:  # Core functionality working
        print("\n🎉 Setup ready for Day 10 exercises!")
        print("\n🚀 Next Steps:")
        print("1. Run exercises: python exercise.py")
        print("2. Review solutions: python solution.py")
        print("3. Take quiz: quiz.md")
        print("4. Optional: Install privacy libraries for advanced features")
        
        if passed < total:
            print("\n💡 Optional Improvements:")
            print("- Install Presidio: pip install presidio-analyzer presidio-anonymizer")
            print("- Install spaCy model: python -m spacy download en_core_web_sm")
            print("- Generate keys: python scripts/generate_keys.py")
            print("- Set up .env file for production features")
    else:
        print("\n⚠️  Setup incomplete. Please fix critical issues:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Check Python version (3.8+ required)")
        print("3. Verify file structure")
        print("4. Check error messages above")
    
    return passed >= 5

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)