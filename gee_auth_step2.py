#!/usr/bin/env python3
"""
Step 2: Complete authentication with verification code
Usage: python3 gee_auth_step2.py YOUR_VERIFICATION_CODE
"""

import ee
import sys

print("=" * 70)
print("GOOGLE EARTH ENGINE AUTHENTICATION - STEP 2")
print("=" * 70)
print()

if len(sys.argv) < 2:
    print("✗ Error: No verification code provided")
    print()
    print("Usage: python3 gee_auth_step2.py YOUR_VERIFICATION_CODE")
    print()
    print("Example: python3 gee_auth_step2.py 4/1AY0e-g5...")
    sys.exit(1)

verification_code = sys.argv[1]

# Load the code verifier from step 1
try:
    with open('/tmp/gee_code_verifier.txt', 'r') as f:
        code_verifier = f.read().strip()
except FileNotFoundError:
    print("✗ Error: Code verifier not found. Please run gee_auth_step1.py first.")
    sys.exit(1)

print("Completing authentication with your verification code...")
print()

try:
    # Complete the authentication
    ee.Authenticate(
        authorization_code=verification_code,
        code_verifier=code_verifier,
        quiet=False
    )

    print()
    print("=" * 70)
    print("✓ AUTHENTICATION SUCCESSFUL!")
    print("=" * 70)
    print()
    print("Testing connection to Earth Engine...")

    # Test initialization
    ee.Initialize()
    print("✓ Successfully connected to Google Earth Engine!")
    print()
    print("You're all set! You can now run:")
    print("  python3 gold_deposit_detection.py")
    print()
    print("=" * 70)

except Exception as e:
    print()
    print("=" * 70)
    print("✗ AUTHENTICATION FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    print("Please make sure you copied the complete verification code.")
    print("Try running gee_auth_step1.py again to get a new URL.")
    print()
