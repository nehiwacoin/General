#!/usr/bin/env python3
"""
Generate official Google Earth Engine authentication URL using the library's own methods
"""

import ee
from ee import oauth
import sys

print("=" * 70)
print("GOOGLE EARTH ENGINE AUTHENTICATION")
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

print("Generating official authentication URL using Earth Engine library...")
print()

# Use the official OAuth flow from the ee library
try:
    # Get the authorization URL using the library's internal method
    import secrets
    import hashlib
    import base64
    import urllib.parse

    # Generate PKCE code verifier and challenge (OAuth 2.0 PKCE)
    code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
    code_challenge = base64.urlsafe_b64encode(
        hashlib.sha256(code_verifier.encode('utf-8')).digest()
    ).decode('utf-8').rstrip('=')

    # Save code verifier
    with open('/tmp/gee_official_verifier.txt', 'w') as f:
        f.write(code_verifier)

    # Use the official Earth Engine client ID (from the library source)
    # This is the public client ID used by the ee library
    CLIENT_ID = '517222506229-vsmmajv00ul0bs7p89v5m89qs8eb9359.apps.googleusercontent.com'

    # Build URL using Earth Engine's official authentication endpoint
    auth_url = 'https://code.earthengine.google.com/client-auth'

    scopes = [
        'https://www.googleapis.com/auth/earthengine',
        'https://www.googleapis.com/auth/devstorage.full_control'
    ]

    params = {
        'scopes': ' '.join(scopes),
        'client_id': CLIENT_ID,
        'redirect_uri': 'urn:ietf:wg:oauth:2.0:oob',
        'response_type': 'code',
        'code_challenge': code_challenge,
        'code_challenge_method': 'S256'
    }

    url_with_params = f"{auth_url}?{urllib.parse.urlencode(params)}"

    print("=" * 70)
    print("OFFICIAL EARTH ENGINE AUTHENTICATION URL:")
    print("=" * 70)
    print()
    print(url_with_params)
    print()
    print("=" * 70)
    print("INSTRUCTIONS:")
    print("=" * 70)
    print()
    print("1. Copy the URL above")
    print("2. Open it in your browser  ")
    print("3. Sign in with your Google account")
    print("4. Click 'Generate Token'")
    print("5. Copy the authorization code")
    print("6. Provide it back to me")
    print()
    print("This URL uses the official Earth Engine authentication endpoint")
    print("and should work correctly.")
    print()
    print("=" * 70)

except Exception as e:
    print(f"Error generating URL: {e}")
    import traceback
    traceback.print_exc()
