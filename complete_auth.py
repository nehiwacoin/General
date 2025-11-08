#!/usr/bin/env python3
"""
Complete Google Earth Engine authentication with authorization code
"""

import ee
import sys

auth_code = "4/1Ab32j93Mwfllti-cZXBKFk1VZANQ9pr1Mi-3f_1VKnf0hnWZxPGO_tdYiao"

print("=" * 70)
print("COMPLETING GOOGLE EARTH ENGINE AUTHENTICATION")
print("=" * 70)
print()

print("Using authorization code to complete authentication...")
print()

try:
    # Authenticate with the provided code
    ee.Authenticate(authorization_code=auth_code, quiet=False)

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
    print("READY TO RUN ANALYSIS!")
    print("=" * 70)
    print()

except Exception as e:
    print()
    print("=" * 70)
    print("✗ AUTHENTICATION FAILED")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
