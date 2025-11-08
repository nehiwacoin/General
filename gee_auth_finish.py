#!/usr/bin/env python3
"""
Finish Google Earth Engine authentication with authorization code
Usage: python3 gee_auth_finish.py YOUR_AUTHORIZATION_CODE
"""

import ee
from ee import oauth
import sys
import os

print("=" * 70)
print("COMPLETING GOOGLE EARTH ENGINE AUTHENTICATION")
print("=" * 70)
print()

if len(sys.argv) < 2:
    print("✗ Error: No authorization code provided")
    print()
    print("Usage: python3 gee_auth_finish.py YOUR_AUTHORIZATION_CODE")
    print()
    print("Example: python3 gee_auth_finish.py 4/1AY0e-g5...")
    sys.exit(1)

auth_code = sys.argv[1]

# Load the saved code_verifier
verifier_file = '/tmp/gee_code_verifier_saved.txt'
try:
    with open(verifier_file, 'r') as f:
        code_verifier = f.read().strip()
    print(f"✓ Loaded code verifier from {verifier_file}")
except FileNotFoundError:
    print(f"✗ Error: Code verifier file not found at {verifier_file}")
    print("Please run gee_auth_complete.py first to generate the authentication URL")
    sys.exit(1)

print(f"✓ Using authorization code: {auth_code[:20]}...")
print()
print("Completing authentication...")
print()

try:
    # Complete authentication with the code_verifier
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

    # Test the connection
    print("Testing connection to Google Earth Engine...")
    ee.Initialize()

    print("✓ Successfully connected to Google Earth Engine!")
    print()
    print("=" * 70)
    print("YOU'RE ALL SET!")
    print("=" * 70)
    print()
    print("You can now run the gold detection analysis:")
    print("  python3 gold_deposit_detection.py")
    print()

except Exception as e:
    print()
    print("=" * 70)
    print("✗ AUTHENTICATION FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    print("Please try the following:")
    print("1. Make sure you copied the complete authorization code")
    print("2. Run gee_auth_complete.py again to get a fresh URL")
    print("3. Use the new authorization code from that URL")
    print()
