#!/usr/bin/env python3
"""
Try to use the existing authorization code directly
"""

import ee
import sys

auth_code = "4/1Ab32j93Mwfllti-cZXBKFk1VZANQ9pr1Mi-3f_1VKnf0hnWZxPGO_tdYiao"

print("=" * 70)
print("ATTEMPTING AUTHENTICATION WITH YOUR CODE")
print("=" * 70)
print()

print("Trying different authentication methods with your code...")
print()

# Method 1: Try without code_verifier
print("Method 1: Direct authentication (no code_verifier)...")
try:
    ee.Authenticate(authorization_code=auth_code, quiet=False)
    print("✓ SUCCESS!")
    ee.Initialize()
    print("✓ Connected to Earth Engine!")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed: {e}")
    print()

# Method 2: Try with empty code_verifier
print("Method 2: With empty code_verifier...")
try:
    ee.Authenticate(authorization_code=auth_code, code_verifier="", quiet=False)
    print("✓ SUCCESS!")
    ee.Initialize()
    print("✓ Connected to Earth Engine!")
    sys.exit(0)
except Exception as e:
    print(f"✗ Failed: {e}")
    print()

# Method 3: Try to use it to manually write credentials
print("Method 3: Manual credential creation...")
try:
    from ee import oauth
    import json

    # Try to get token directly
    token_info = oauth.request_token(
        auth_code,
        code_verifier=None
    )

    print(f"✓ Got token info!")

    # Write credentials
    credentials_path = oauth.get_credentials_path()
    with open(credentials_path, 'w') as f:
        json.dump(token_info, f)

    print(f"✓ Wrote credentials to {credentials_path}")

    # Test
    ee.Initialize()
    print("✓ Connected to Earth Engine!")
    sys.exit(0)

except Exception as e:
    print(f"✗ Failed: {e}")
    print()

print("=" * 70)
print("All methods failed.")
print("=" * 70)
print()
print("The issue is that the authorization code needs the matching")
print("code_verifier from when it was generated.")
print()
print("We need to generate a fresh authentication URL and use the")
print("code from that specific URL.")
