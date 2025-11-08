#!/bin/bash
# Capture Google Earth Engine authentication URL

echo "Generating Google Earth Engine authentication URL..."
echo ""

# Run Python script in background, capture output, and kill after URL is shown
(python3 << 'PYEOF'
import ee
import sys

try:
    ee.Initialize()
    print("Already authenticated!")
    sys.exit(0)
except:
    pass

try:
    ee.Authenticate(auth_mode='notebook', quiet=False)
except:
    pass
PYEOF
) 2>&1 &

PID=$!
sleep 2
kill $PID 2>/dev/null
wait $PID 2>/dev/null

echo ""
echo "If you see a URL above, use that for authentication."
echo "If not, run: python3 -c 'import ee; ee.Authenticate()'"
