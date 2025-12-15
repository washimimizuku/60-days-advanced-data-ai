# Day 10: Data Privacy - GDPR, PII Handling - Setup Guide

## Quick Start (10 minutes)

### 1. Install Dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Install spaCy model for PII detection
python -m spacy download en_core_web_sm
```

### 2. Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Generate secure keys
python scripts/generate_keys.py

# Edit .env with your configuration
nano .env
```

### 3. Run Exercises
```bash
# Test setup
python scripts/test_setup.py

# Run exercises
python exercise.py

# Run complete solution
python solution.py
```

---

## Detailed Setup

### Prerequisites
- Python 3.8+
- 4GB+ RAM (for ML models)
- 2GB+ disk space
- OpenSSL (for cryptography)

### Core Dependencies Installation

#### Privacy Libraries
```bash
# Install core privacy frameworks
pip install opendp diffprivlib anonymizedf

# Install PII detection tools
pip install presidio-analyzer presidio-anonymizer
pip install spacy transformers

# Download language models
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_lg  # Optional: better accuracy
```

#### Cryptography Setup
```bash
# Install cryptography libraries
pip install cryptography pycryptodome

# Verify installation
python -c "from cryptography.fernet import Fernet; print('Cryptography OK')"
```

### Database Setup (Optional)

#### PostgreSQL for Metadata
```bash
# Install PostgreSQL
# Ubuntu/Debian: sudo apt-get install postgresql
# macOS: brew install postgresql

# Create database
sudo -u postgres psql
CREATE DATABASE privacy_db;
CREATE USER privacy_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE privacy_db TO privacy_user;
```

#### Redis for Caching
```bash
# Install Redis
# Ubuntu/Debian: sudo apt-get install redis-server
# macOS: brew install redis

# Start Redis
redis-server

# Test connection
redis-cli ping
```

### Security Configuration

#### Generate Encryption Keys
```bash
# Create key generation script
cat > scripts/generate_keys.py << 'EOF'
#!/usr/bin/env python3
"""Generate secure keys for privacy system"""

import secrets
import base64
from cryptography.fernet import Fernet

def generate_keys():
    """Generate all required keys"""
    
    # Generate secret key (32 bytes)
    secret_key = secrets.token_hex(32)
    
    # Generate salt for pseudonymization
    salt = secrets.token_hex(16)
    
    # Generate Fernet key
    fernet_key = Fernet.generate_key().decode()
    
    print("Generated secure keys:")
    print(f"PRIVACY_SECRET_KEY={secret_key}")
    print(f"PSEUDONYM_SALT={salt}")
    print(f"FERNET_KEY={fernet_key}")
    print()
    print("Add these to your .env file")

if __name__ == "__main__":
    generate_keys()
EOF

# Make executable and run
chmod +x scripts/generate_keys.py
python scripts/generate_keys.py
```

#### Update Environment File
```bash
# Update .env with generated keys
# Replace placeholder values with generated keys
```

### PII Detection Setup

#### Configure Presidio
```bash
# Test Presidio installation
python -c "
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

text = 'My email is john.doe@email.com'
results = analyzer.analyze(text=text, language='en')
print('Presidio working:', len(results) > 0)
"
```

#### Configure spaCy
```bash
# Verify spaCy model
python -c "
import spacy
nlp = spacy.load('en_core_web_sm')
doc = nlp('John Doe lives in New York')
print('spaCy entities:', [(ent.text, ent.label_) for ent in doc.ents])
"
```

### Testing Setup

#### Create Test Script
```bash
# Create comprehensive test script
cat > scripts/test_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test privacy system setup"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all required imports"""
    try:
        import pandas as pd
        import numpy as np
        from cryptography.fernet import Fernet
        import secrets
        import hashlib
        print("✅ Core libraries imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_privacy_libraries():
    """Test privacy-specific libraries"""
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_anonymizer import AnonymizerEngine
        import spacy
        
        # Test spaCy model
        nlp = spacy.load('en_core_web_sm')
        
        print("✅ Privacy libraries working")
        return True
    except Exception as e:
        print(f"❌ Privacy library error: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found")
        return False
    
    # Check for required variables
    required_vars = [
        'PRIVACY_SECRET_KEY',
        'PSEUDONYM_SALT', 
        'FERNET_KEY'
    ]
    
    with open(env_file) as f:
        content = f.read()
        
    missing_vars = []
    for var in required_vars:
        if var not in content or f"{var}=your_" in content:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing or default environment variables: {missing_vars}")
        return False
    
    print("✅ Environment configuration OK")
    return True

def test_pii_detection():
    """Test PII detection functionality"""
    try:
        from presidio_analyzer import AnalyzerEngine
        
        analyzer = AnalyzerEngine()
        text = "My email is john.doe@email.com and phone is 555-123-4567"
        results = analyzer.analyze(text=text, language='en')
        
        if len(results) >= 2:  # Should detect email and phone
            print("✅ PII detection working")
            return True
        else:
            print("❌ PII detection not finding expected entities")
            return False
            
    except Exception as e:
        print(f"❌ PII detection error: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Day 10 Privacy System Setup")
    print("=" * 50)
    
    tests = [
        ("Core Imports", test_imports),
        ("Privacy Libraries", test_privacy_libraries),
        ("Environment Config", test_environment),
        ("PII Detection", test_pii_detection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 50)
    print("📊 Test Summary")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Ready for Day 10 exercises.")
        print("\nNext steps:")
        print("1. Run: python exercise.py")
        print("2. Complete all 6 exercises")
        print("3. Review: python solution.py")
        print("4. Take quiz: quiz.md")
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please fix issues before proceeding.")
        print("\nTroubleshooting:")
        print("1. Check requirements.txt installation")
        print("2. Verify .env configuration")
        print("3. Download spaCy models")
        print("4. Generate encryption keys")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
EOF

# Make executable
chmod +x scripts/test_setup.py
```

### Performance Optimization

#### Configure Memory Settings
```bash
# For large datasets, increase memory limits
export PYTHONHASHSEED=0
export OMP_NUM_THREADS=4

# Configure pandas memory usage
python -c "
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.memory_usage', 'deep')
"
```

#### Optimize PII Detection
```bash
# Use faster spaCy model for development
python -m spacy download en_core_web_sm

# Use more accurate model for production
python -m spacy download en_core_web_lg
```

### Docker Setup (Optional)

#### Create Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 privacy && chown -R privacy:privacy /app
USER privacy

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "solution.py"]
```

#### Create Docker Compose
```yaml
version: '3.8'

services:
  privacy-system:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PRIVACY_SECRET_KEY=${PRIVACY_SECRET_KEY}
      - PSEUDONYM_SALT=${PSEUDONYM_SALT}
      - FERNET_KEY=${FERNET_KEY}
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    command: redis-server --requirepass ${REDIS_PASSWORD}

  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=${POSTGRES_DB}
      - POSTGRES_USER=${POSTGRES_USER}
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### Verification

#### Run Complete Test
```bash
# Test all components
python scripts/test_setup.py

# Test with sample data
python -c "
import pandas as pd
from solution import PIIDetector, DataClassifier

# Create test data
data = pd.DataFrame({
    'email': ['test@email.com'],
    'phone': ['555-123-4567'],
    'age': [25]
})

# Test PII detection
detector = PIIDetector()
results = detector.detect_pii_in_dataframe(data)
print('PII Detection Results:', len(results))

# Test classification
classifier = DataClassifier()
classifications = classifier.classify_dataset(data)
print('Classification Results:', len(classifications))

print('✅ System working correctly')
"
```

### Troubleshooting

#### Common Issues

**Import Errors:**
```bash
# Reinstall with specific versions
pip install --force-reinstall -r requirements.txt

# Check Python version
python --version  # Should be 3.8+
```

**spaCy Model Issues:**
```bash
# Download model manually
python -m spacy download en_core_web_sm --force

# Verify model
python -c "import spacy; spacy.load('en_core_web_sm')"
```

**Cryptography Issues:**
```bash
# Install build tools
# Ubuntu: sudo apt-get install build-essential libssl-dev libffi-dev
# macOS: xcode-select --install

# Reinstall cryptography
pip uninstall cryptography
pip install cryptography
```

**Memory Issues:**
```bash
# Increase memory limits
export PYTHONMAXMEMORY=4G

# Use smaller datasets for testing
head -n 1000 large_dataset.csv > test_dataset.csv
```

### Production Deployment

#### Security Checklist
- [ ] Generate unique encryption keys
- [ ] Enable HTTPS/TLS encryption
- [ ] Configure proper access controls
- [ ] Set up audit logging
- [ ] Implement rate limiting
- [ ] Configure backup procedures
- [ ] Set up monitoring and alerting

#### Performance Checklist
- [ ] Optimize database queries
- [ ] Configure caching strategies
- [ ] Set up load balancing
- [ ] Monitor memory usage
- [ ] Implement batch processing
- [ ] Configure auto-scaling

### Getting Help

1. **Check logs:**
   ```bash
   tail -f logs/privacy.log
   ```

2. **Run diagnostics:**
   ```bash
   python scripts/test_setup.py
   ```

3. **Verify configuration:**
   ```bash
   python -c "from dotenv import load_dotenv; load_dotenv(); import os; print('Keys loaded:', bool(os.getenv('PRIVACY_SECRET_KEY')))"
   ```

Ready to protect privacy with GDPR compliance! 🔒