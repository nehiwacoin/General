# Windows Setup Guide for Gold Detection Script

## You're Using: Windows + Anaconda ✓

Since you have Anaconda/Miniconda installed (I can see "(base)" in your prompt), let's use conda which avoids the permission issues.

## Step 1: Create a New Conda Environment

Open **Anaconda Prompt** (or PowerShell) as **Administrator** and run:

```bash
conda create -n gee_gold python=3.11 -y
conda activate gee_gold
```

## Step 2: Install Required Packages

With the environment activated, install the packages:

```bash
# Install from conda-forge (better for geospatial packages)
conda install -c conda-forge earthengine-api geopandas rasterio folium -y
```

**OR** if conda install is slow, use pip after activating the environment:

```bash
python -m pip install earthengine-api geopandas rasterio folium
```

**Note:** Using `python -m pip` instead of just `pip` often fixes permission issues.

## Step 3: Download the Script

The script is in the git repository. You have two options:

### Option A: Download from GitHub
If the repo is pushed to GitHub, clone or download it.

### Option B: I'll create it here
Tell me and I'll provide you with the complete script content to copy.

## Step 4: Authenticate with Google Earth Engine

Once packages are installed:

```bash
python -m earthengine authenticate
```

This will:
1. Open a browser window
2. Ask you to sign in with Google
3. Generate credentials automatically
4. Save them to your system

## Step 5: Run the Script

```bash
python gold_deposit_detection.py
```

## Troubleshooting

### If you get "Access Denied" with pip:
- Run PowerShell/Anaconda Prompt as **Administrator**
- OR use: `python -m pip install` instead of `pip install`
- OR use conda: `conda install -c conda-forge package_name`

### If earthengine command not found:
- Make sure you've activated the conda environment: `conda activate gee_gold`
- The earthengine command is installed with the `earthengine-api` package

### If python3 command not found on Windows:
- Use `python` instead of `python3` on Windows
- Windows Python is typically just called `python`

## Quick Start Commands (Copy & Paste)

```powershell
# 1. Create environment
conda create -n gee_gold python=3.11 -y

# 2. Activate it
conda activate gee_gold

# 3. Install packages
conda install -c conda-forge earthengine-api geopandas rasterio folium -y

# 4. Authenticate
python -m earthengine authenticate

# 5. Run the script
python gold_deposit_detection.py
```

## Need the Script File?

The `gold_deposit_detection.py` file is in the repository. If you don't have it locally yet, let me know and I'll provide the complete script for you to save.
