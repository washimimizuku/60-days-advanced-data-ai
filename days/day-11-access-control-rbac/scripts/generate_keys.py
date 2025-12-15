#!/usr/bin/env python3
"""
Generate secure keys for access control system
"""

import secrets
import base64
from cryptography.fernet import Fernet

def generate_secret_key(length=32):
    """Generate a secure secret key"""
    return secrets.token_hex(length)

def generate_jwt_key(length=32):
    """Generate a JWT secret key"""
    return secrets.token_urlsafe(length)

def generate_encryption_key():
    """Generate a Fernet encryption key"""
    return Fernet.generate_key().decode()

def main():
    print("=== Access Control Security Keys ===\n")
    
    print("SECRET_KEY=" + generate_secret_key())
    print("JWT_SECRET_KEY=" + generate_jwt_key())
    print("ENCRYPTION_KEY=" + generate_encryption_key())
    
    print("\n=== Copy these to your .env file ===")
    print("Replace the placeholder values with the keys above")

if __name__ == "__main__":
    main()