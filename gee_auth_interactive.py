#!/usr/bin/env python3
"""
Interactive Google Earth Engine authentication using actual ee.Authenticate()
This captures the real authentication URL and allows you to provide the code back.
"""

import ee
import sys
import subprocess
import time

print("=" * 70)
print("GOOGLE EARTH ENGINE AUTHENTICATION")
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

print("Starting authentication process...")
print()
print("I'll launch the authentication and capture the URL for you.")
print("This may take a moment...")
print()
print("=" * 70)
print()

# Use subprocess to run authentication and capture output
try:
    # Run earthengine authenticate command if available
    result = subprocess.run(
        ['earthengine', 'authenticate', '--quiet'],
        capture_output=True,
        text=True,
        timeout=5
    )

    if 'https://' in result.stdout or 'https://' in result.stderr:
        output = result.stdout + result.stderr
        print("Authentication URL found!")
        print()
        print(output)

except (subprocess.TimeoutExpired, FileNotFoundError):
    # earthengine command not available, use Python API
    print("Using Python API for authentication...")
    print()
    print("I'll show you the authentication instructions.")
    print("You'll need to manually complete the process.")
    print()
    print("=" * 70)
    print()

    # Start a subprocess that will run ee.Authenticate()
    # and we'll kill it after getting the URL
    proc = subprocess.Popen(
        [sys.executable, '-c',
         'import ee; ee.Authenticate(auth_mode="notebook", quiet=False)'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    # Wait a bit for the URL to be generated
    time.sleep(2)

    # Terminate the process
    proc.terminate()

    # Get the output
    stdout, stderr = proc.communicate()
    output = stdout + stderr

    if 'https://' in output:
        print("=" * 70)
        print("AUTHENTICATION URL:")
        print("=" * 70)
        print()

        # Extract and display the URL
        for line in output.split('\n'):
            if 'https://' in line:
                print(line.strip())

        print()
        print("=" * 70)
        print("NEXT STEPS:")
        print("=" * 70)
        print()
        print("1. Copy the URL above")
        print("2. Open it in your browser")
        print("3. Sign in and authorize")
        print("4. Copy the authorization code")
        print("5. Tell me: 'Here is the code: [paste it]'")
        print()
    else:
        print("Could not extract authentication URL.")
        print()
        print("Please run this command manually in a terminal:")
        print("  python3 -c \"import ee; ee.Authenticate()\"")
        print()

print("=" * 70)
