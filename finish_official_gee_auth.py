#!/usr/bin/env python3
"""
Complete authentication with authorization code from official GEE URL
Usage: python3 finish_official_gee_auth.py YOUR_AUTH_CODE
"""

import ee
import sys

print("=" * 70)
print("COMPLETING GOOGLE EARTH ENGINE AUTHENTICATION")
print("=" * 70)
print()

if len(sys.argv) < 2:
    print("✗ Error: No authorization code provided")
    print()
    print("Usage: python3 finish_official_gee_auth.py YOUR_AUTH_CODE")
    print()
    print("Example: python3 finish_official_gee_auth.py 4/1AY0e-g5...")
    sys.exit(1)

auth_code = sys.argv[1].strip()

# Load the saved code_verifier
try:
    with open('/tmp/gee_official_verifier.txt', 'r') as f:
        code_verifier = f.read().strip()
    print("✓ Loaded code verifier")
except FileNotFoundError:
    print("✗ Error: Code verifier not found")
    print("Please run generate_official_gee_url.py first")
    sys.exit(1)

print(f"✓ Using authorization code: {auth_code[:20]}...")
print()
print("Completing authentication...")

try:
    ee.Authenticate(
        authorization_code=auth_code,
        code_verifier=code_verifier,
        quiet=False
    )

    print()
    print("=" * 70)
    print("✓ AUTHENTICATION SUCCESSFUL!")
    print("=" * 70)
    print()

    # Test connection
    print("Testing connection to Google Earth Engine...")
    ee.Initialize()

    print("✓ Successfully connected to Google Earth Engine!")
    print()
    print("=" * 70)
    print("YOU'RE ALL SET!")
    print("=" * 70)
    print()
    print("Now I can run the gold detection analysis!")
    print()

except Exception as e:
    print()
    print("=" * 70)
    print("✗ AUTHENTICATION FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    print("Please try:")
    print("1. Make sure you copied the complete code")
    print("2. Run generate_official_gee_url.py again for a fresh URL")
    print()
