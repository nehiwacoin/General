"""
Download script for gold_deposit_detection.py
Run this to get the script from the repository
"""

import urllib.request
import os

print("Downloading gold_deposit_detection.py...")

# GitHub raw URL (update this with your actual repo URL)
url = "https://raw.githubusercontent.com/YOUR_USERNAME/YOUR_REPO/main/gold_deposit_detection.py"

try:
    urllib.request.urlretrieve(url, "gold_deposit_detection.py")
    print("✓ Downloaded successfully!")
    print(f"✓ Saved to: {os.path.abspath('gold_deposit_detection.py')}")
except Exception as e:
    print(f"✗ Download failed: {e}")
    print("\nAlternative: Copy the script manually from the repository")
    print("Or ask me to paste the complete script content here")
