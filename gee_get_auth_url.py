#!/usr/bin/env python3
"""
Get Google Earth Engine authentication URL
This script displays the URL you need to visit to authenticate.
"""

import ee
import sys
from io import StringIO
import threading

print("=" * 70)
print("GOOGLE EARTH ENGINE AUTHENTICATION")
print("=" * 70)
print()

try:
    # Try to initialize with existing credentials
    ee.Initialize()
    print("✓ Already authenticated! You're ready to go.")
    print()
    print("You can now run: python3 gold_deposit_detection.py")
    sys.exit(0)
except:
    pass

print("Setting up authentication...")
print()
print("=" * 70)
print("FOLLOW THESE STEPS:")
print("=" * 70)
print()
print("I'll display the authentication URL below.")
print("The script will wait for you to provide the verification code.")
print()
print("1. Copy the URL that appears")
print("2. Open it in your browser")
print("3. Sign in with your Google account")
print("4. Click 'Generate Token'")
print("5. Copy the verification code")
print("6. Paste the code below when prompted")
print()
print("=" * 70)
print()
print("Starting authentication process...")
print()
print("=" * 70)

# This will print the URL and wait for input
try:
    ee.Authenticate(auth_mode='notebook', quiet=False)
    print()
    print("=" * 70)
    print("✓ AUTHENTICATION SUCCESSFUL!")
    print("=" * 70)
    print()

    # Test it
    ee.Initialize()
    print("✓ Successfully connected to Google Earth Engine!")
    print()
    print("You can now run: python3 gold_deposit_detection.py")
    print()

except KeyboardInterrupt:
    print()
    print("Authentication cancelled.")
    sys.exit(1)
except Exception as e:
    print()
    print("=" * 70)
    print("✗ AUTHENTICATION FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print()
