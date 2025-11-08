#!/usr/bin/env python3
"""
Use Earth Engine's internal authentication flow exactly as the library does it
"""

import ee
import sys
import os

print("=" * 70)
print("GOOGLE EARTH ENGINE AUTHENTICATION - SIMPLIFIED")
print("=" * 70)
print()

# Check if already authenticated
try:
    ee.Initialize()
    print("✓ Already authenticated!")
    print()
    print("Run: python3 gold_deposit_detection.py")
    sys.exit(0)
except:
    pass

print("Let's try using a simpler authentication method.")
print()
print("I'll use the 'gcloud' authentication mode which may work better.")
print()
print("=" * 70)
print()

# Try using the --authorization-code method
print("Please visit the following URL to authorize this application:")
print()

# Import the oauth module directly
from ee import oauth

# Generate auth URL using the library's method
try:
    # Use the library's built-in method to get the auth URL
    import hashlib
    import secrets
    import base64
    import urllib.parse

    # Create code verifier
    code_verifier_bytes = secrets.token_bytes(32)
    code_verifier = base64.urlsafe_b64encode(code_verifier_bytes).decode('utf-8').rstrip('=')

    # Create code challenge
    code_challenge_bytes = hashlib.sha256(code_verifier.encode('utf-8')).digest()
    code_challenge = base64.urlsafe_b64encode(code_challenge_bytes).decode('utf-8').rstrip('=')

    # Save code verifier
    with open('/tmp/ee_code_verifier_final.txt', 'w') as f:
        f.write(code_verifier)

    # Generate a simple request ID
    request_id = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')

    # Use the simplified Earth Engine auth endpoint
    params = {
        'scopes': 'https://www.googleapis.com/auth/earthengine https://www.googleapis.com/auth/cloud-platform https://www.googleapis.com/auth/drive https://www.googleapis.com/auth/devstorage.full_control',
        'request_id': request_id,
        'tc': code_challenge,
        'cc': code_verifier[:40]  # Shortened verifier for the URL
    }

    auth_url = f"https://code.earthengine.google.com/client-auth?{urllib.parse.urlencode(params)}"

    print(auth_url)
    print()
    print("=" * 70)
    print("INSTRUCTIONS:")
    print("=" * 70)
    print()
    print("1. Copy the URL above")
    print("2. Open it in your browser")
    print("3. Sign in with Google")
    print("4. Click 'Generate Token'")
    print("5. Copy the token/code that appears")
    print("6. Run: python3 finish_simple_auth.py YOUR_TOKEN")
    print()
    print("=" * 70)

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
