#!/usr/bin/env python3
"""
Complete two-part Google Earth Engine authentication
This script generates a URL, saves the code_verifier, then waits for the auth code
"""

import ee
from ee import oauth
import sys
import os

print("=" * 70)
print("GOOGLE EARTH ENGINE AUTHENTICATION (COMPLETE SOLUTION)")
print("=" * 70)
print()

# Check if already authenticated
try:
    ee.Initialize()
    print("✓ Already authenticated! You're ready to go.")
    print()
    print("Run: python3 gold_deposit_detection.py")
    sys.exit(0)
except:
    pass

# Generate authentication URL with saved code_verifier
print("Generating new authentication URL...")
print()

# Create code verifier
import secrets
import hashlib
import base64

# Generate code_verifier (OAuth2 PKCE)
code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')

# Generate code_challenge
code_challenge = base64.urlsafe_b64encode(
    hashlib.sha256(code_verifier.encode('utf-8')).digest()
).decode('utf-8').rstrip('=')

# Save code_verifier for later
verifier_file = '/tmp/gee_code_verifier_saved.txt'
with open(verifier_file, 'w') as f:
    f.write(code_verifier)

print(f"✓ Code verifier saved to {verifier_file}")
print()

# Build the authorization URL
scopes = [
    'https://www.googleapis.com/auth/earthengine',
    'https://www.googleapis.com/auth/devstorage.full_control'
]

import urllib.parse
params = {
    'scope': ' '.join(scopes),
    'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob',
    'response_type': 'code',
    'client_id': '517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359.apps.googleusercontent.com',
    'code_challenge': code_challenge,
    'code_challenge_method': 'S256'
}

auth_url = 'https://accounts.google.com/o/oauth2/auth?' + urllib.parse.urlencode(params)

print("=" * 70)
print("AUTHENTICATION URL:")
print("=" * 70)
print()
print(auth_url)
print()
print("=" * 70)
print("INSTRUCTIONS:")
print("=" * 70)
print()
print("1. Copy the URL above")
print("2. Open it in your web browser")
print("3. Sign in with your Google account")
print("4. Click 'Allow' to grant permissions")
print("5. Copy the authorization code that appears")
print("6. Run: python3 gee_auth_finish.py YOUR_CODE_HERE")
print()
print("=" * 70)
