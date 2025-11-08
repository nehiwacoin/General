# Fixing "No module named earthengine" Error on Windows

## The Issue
The `earthengine-api` package didn't install with conda. Let's install it with pip instead.

## Solution

### Step 1: Make sure you're in the gee_gold environment
You should see `(gee_gold)` at the start of your prompt. If not:
```powershell
conda activate gee_gold
```

### Step 2: Install earthengine-api using pip
```powershell
python -m pip install earthengine-api
```

### Step 3: Verify it's installed
```powershell
python -c "import ee; print('Earth Engine version:', ee.__version__)"
```

You should see something like: `Earth Engine version: 0.1.xxx`

### Step 4: Now authenticate
```powershell
python -m earthengine authenticate
```

OR if that still doesn't work:
```powershell
python -c "import ee; ee.Authenticate()"
```

This will:
1. Open your browser
2. Ask you to sign in with Google
3. Generate credentials automatically

### Step 5: Test the authentication
```powershell
python -c "import ee; ee.Initialize(); print('Successfully authenticated!')"
```

### Step 6: Run the gold detection script
```powershell
python gold_deposit_detection.py
```

## If You Still Get Errors

Try installing a specific version:
```powershell
python -m pip install earthengine-api==0.1.411
```

Or reinstall:
```powershell
python -m pip uninstall earthengine-api -y
python -m pip install earthengine-api
```

## Quick Command Summary (Copy & Paste)

```powershell
# Install earthengine-api
python -m pip install earthengine-api

# Authenticate
python -c "import ee; ee.Authenticate()"

# Test
python -c "import ee; ee.Initialize(); print('Success!')"

# Run the script
python gold_deposit_detection.py
```
