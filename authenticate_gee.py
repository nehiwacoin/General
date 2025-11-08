#!/usr/bin/env python3
"""
Google Earth Engine Authentication Helper
==========================================
This script helps you authenticate with Google Earth Engine.
"""

import ee

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
    print("=" * 70)
except:
    print("Starting authentication process...")
    print()
    print("=" * 70)
    print("INSTRUCTIONS:")
    print("=" * 70)
    print()
    print("1. A URL will appear below")
    print("2. Open it in your web browser")
    print("3. Sign in with your Google account")
    print("4. Click 'Generate Token'")
    print("5. Copy the verification code")
    print("6. Paste it back here when prompted")
    print()
    print("=" * 70)
    print()

    try:
        ee.Authenticate(auth_mode='notebook')
        print()
        print("=" * 70)
        print("✓ AUTHENTICATION SUCCESSFUL!")
        print("=" * 70)
        print()
        print("You can now run: python3 gold_deposit_detection.py")
        print()
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ AUTHENTICATION FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()
        print("Please try again or visit:")
        print("https://developers.google.com/earth-engine/guides/auth")
