#!/usr/bin/env python3
"""
Step 1: Get the authentication URL
This script generates the URL you need to visit to get your verification code.
"""

import ee

print("=" * 70)
print("GOOGLE EARTH ENGINE AUTHENTICATION - STEP 1")
print("=" * 70)
print()

try:
    # Try to initialize with existing credentials
    ee.Initialize()
    print("✓ Already authenticated! You're ready to go.")
    print()
    print("You can now run the gold detection analysis!")
except:
    print("Generating authentication URL...")
    print()

    # Generate the auth URL without trying to read input
    from ee import oauth

    # Get the authorization URL
    code_verifier = oauth.CodeVerifier()
    auth_url = oauth.get_authorization_url(code_verifier.value)

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
    print("4. Click 'Generate Token'")
    print("5. Copy the verification code that appears")
    print("6. Provide the code back to me")
    print()
    print("=" * 70)
    print()
    print("After you get the code, tell me what it is, and I'll complete")
    print("the authentication for you!")
    print()

    # Save the code verifier for step 2
    with open('/tmp/gee_code_verifier.txt', 'w') as f:
        f.write(code_verifier.value)

    print("✓ Code verifier saved for step 2")
