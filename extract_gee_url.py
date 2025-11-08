#!/usr/bin/env python3
"""
Extract Google Earth Engine authentication URL without blocking on input
"""

import ee
import sys
import builtins

# Check if already authenticated
try:
    ee.Initialize()
    print("✓ Already authenticated! You're ready to go.")
    print()
    print("Run: python3 gold_deposit_detection.py")
    sys.exit(0)
except:
    pass

print("=" * 70)
print("GOOGLE EARTH ENGINE AUTHENTICATION")
print("=" * 70)
print()

# Monkey-patch input() to capture the URL and then exit
url_captured = []

original_input = builtins.input
def mock_input(prompt=''):
    # Print the prompt (which contains the URL)
    print(prompt)
    # Extract URL from the output that was just printed
    sys.exit(0)  # Exit cleanly after showing the URL

builtins.input = mock_input

try:
    ee.Authenticate(auth_mode='notebook', quiet=False)
except SystemExit:
    pass
except Exception as e:
    print(f"Error during authentication setup: {e}")
finally:
    builtins.input = original_input
