#!/usr/bin/env python3
import subprocess
import sys
import time
import signal

print("=" * 70)
print("CAPTURING GOOGLE EARTH ENGINE AUTHENTICATION URL")
print("=" * 70)
print()

# Start the authentication process
proc = subprocess.Popen(
    [sys.executable, '-c', 'import ee; ee.Authenticate(auth_mode="notebook")'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)

# Give it time to print the URL
time.sleep(3)

# Terminate the process
proc.send_signal(signal.SIGTERM)

# Get all output
stdout, stderr = proc.communicate(timeout=2)

full_output = stdout + stderr

print("Output captured:")
print("=" * 70)
print(full_output)
print("=" * 70)
print()

# Try to extract just the URL
if 'https://' in full_output:
    print("Authentication URL found:")
    print()
    for line in full_output.split('\n'):
        if 'https://code.earthengine' in line:
            url = line.strip()
            print(url)
            print()
else:
    print("No URL found in output")
