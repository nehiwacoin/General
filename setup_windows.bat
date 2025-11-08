@echo off
REM Windows Setup Script for Gold Detection with Google Earth Engine
REM Run this from Anaconda Prompt as Administrator

echo ======================================================================
echo GOOGLE EARTH ENGINE - GOLD DETECTION SETUP (WINDOWS)
echo ======================================================================
echo.

echo Step 1: Creating conda environment 'gee_gold'...
call conda create -n gee_gold python=3.11 -y
if %errorlevel% neq 0 (
    echo Error creating environment. Please run this from Anaconda Prompt as Administrator.
    pause
    exit /b 1
)

echo.
echo Step 2: Activating environment...
call conda activate gee_gold

echo.
echo Step 3: Installing packages...
echo This may take a few minutes...
call conda install -c conda-forge earthengine-api geopandas rasterio folium -y
if %errorlevel% neq 0 (
    echo Conda install failed, trying pip...
    python -m pip install earthengine-api geopandas rasterio folium
)

echo.
echo ======================================================================
echo SETUP COMPLETE!
echo ======================================================================
echo.
echo Next steps:
echo 1. Run: python -m earthengine authenticate
echo 2. Sign in with Google when browser opens
echo 3. Run: python gold_deposit_detection.py
echo.
echo To activate this environment in the future, run:
echo    conda activate gee_gold
echo.
pause
