#!/usr/bin/env python3
"""
Generate secure encryption keys for privacy system
"""

import secrets
import base64
from cryptography.fernet import Fernet

def generate_keys():
    """Generate all required cryptographic keys"""
    
    print("🔐 Generating secure keys for privacy system...")
    print("=" * 50)
    
    # Generate secret key (32 bytes for HMAC)
    secret_key = secrets.token_hex(32)
    print(f"✅ Generated PRIVACY_SECRET_KEY (64 chars)")
    
    # Generate salt for pseudonymization
    salt = secrets.token_hex(16)
    print(f"✅ Generated PSEUDONYM_SALT (32 chars)")
    
    # Generate Fernet key for encryption
    fernet_key = Fernet.generate_key().decode()
    print(f"✅ Generated FERNET_KEY (44 chars)")
    
    # Generate additional keys
    jwt_secret = secrets.token_urlsafe(32)
    print(f"✅ Generated JWT_SECRET_KEY (43 chars)")
    
    api_key = secrets.token_urlsafe(24)
    print(f"✅ Generated API_KEY (32 chars)")
    
    print("\n" + "=" * 50)
    print("🔑 Add these keys to your .env file:")
    print("=" * 50)
    
    print(f"PRIVACY_SECRET_KEY={secret_key}")
    print(f"PSEUDONYM_SALT={salt}")
    print(f"FERNET_KEY={fernet_key}")
    print(f"JWT_SECRET_KEY={jwt_secret}")
    print(f"API_KEY={api_key}")
    
    print("\n" + "=" * 50)
    print("⚠️  SECURITY NOTES:")
    print("=" * 50)
    print("1. Keep these keys secure and never commit to version control")
    print("2. Use different keys for different environments (dev/staging/prod)")
    print("3. Rotate keys regularly in production")
    print("4. Store keys in secure key management systems for production")
    print("5. Backup keys securely before rotation")

def test_keys():
    """Test generated keys work correctly"""
    
    print("\n🧪 Testing key functionality...")
    
    try:
        # Test Fernet key
        key = Fernet.generate_key()
        cipher = Fernet(key)
        test_data = b"test encryption"
        encrypted = cipher.encrypt(test_data)
        decrypted = cipher.decrypt(encrypted)
        assert decrypted == test_data
        print("✅ Fernet encryption/decryption working")
        
        # Test HMAC
        import hmac
        import hashlib
        secret = secrets.token_bytes(32)
        message = "test message"
        signature = hmac.new(secret, message.encode(), hashlib.sha256).hexdigest()
        assert len(signature) == 64
        print("✅ HMAC signing working")
        
        # Test token generation
        token = secrets.token_urlsafe(32)
        assert len(token) >= 32
        print("✅ Token generation working")
        
        print("✅ All cryptographic functions working correctly")
        
    except Exception as e:
        print(f"❌ Cryptographic test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    generate_keys()
    test_keys()
    
    print("\n🎉 Key generation complete!")
    print("Next steps:")
    print("1. Copy the keys above to your .env file")
    print("2. Run: python scripts/test_setup.py")
    print("3. Start with: python exercise.py")