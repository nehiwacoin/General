# Data Acquisition Guide

This guide explains how to download and prepare external datasets for the Mat-Su gold prospectivity system.

## Required External Data

### 1. USGS Mineral Resources Data System (MRDS)

**What it is**: Database of known mineral occurrences and mines worldwide.

**Why we need it**: To get training data for machine learning and validate our model.

**How to download**:

#### Option A: Manual Download (Recommended)

1. Visit: https://mrdata.usgs.gov/mrds/
2. Click "Download Data"
3. Select Alaska region or download full dataset
4. Filter for:
   - Commodity: Gold
   - Deposit Type: Lode (exclude placer)
   - Region: Alaska
5. Save as CSV to: `data/usgs_mrds_alaska_gold.csv`

#### Option B: Direct URL Download

```bash
# Download Alaska mineral occurrences
wget -O data/usgs_mrds_alaska.zip https://mrdata.usgs.gov/mrds/mrds-csv.zip

# Unzip
unzip data/usgs_mrds_alaska.zip -d data/

# Filter for Alaska gold (you'll need to do this manually or with Python)
```

#### Option C: Python Script

Create and run this script:

```python
import pandas as pd
import urllib.request
from pathlib import Path

# Create data directory
Path("data").mkdir(exist_ok=True)

# Download MRDS data
print("Downloading USGS MRDS data...")
url = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
urllib.request.urlretrieve(url, "data/mrds.zip")

# Unzip
import zipfile
with zipfile.ZipFile("data/mrds.zip", 'r') as zip_ref:
    zip_ref.extractall("data/")

# Load and filter
df = pd.read_csv("data/mrds.csv", low_memory=False)

# Filter for Alaska lode gold
alaska_gold = df[
    (df['state'] == 'AK') &
    (df['commod1'].str.contains('Au', na=False) |
     df['commod2'].str.contains('Au', na=False) |
     df['commod3'].str.contains('Au', na=False)) &
    (~df['dep_type'].str.contains('placer', case=False, na=False))
]

# Save
alaska_gold.to_csv("data/usgs_mrds_alaska_gold.csv", index=False)
print(f"✓ Saved {len(alaska_gold)} Alaska lode gold occurrences")
```

**Expected columns** in the CSV:
- `dep_name` - Deposit name
- `latitude` - Latitude (decimal degrees)
- `longitude` - Longitude (decimal degrees)
- `commod1`, `commod2`, `commod3` - Commodities
- `dep_type` - Deposit type
- `dev_stat` - Development status

### 2. User Mine Validation Dataset

**What it is**: Your personal dataset of known successful mine locations in Mat-Su.

**Format needed**: CSV file with the following columns:

```csv
name,latitude,longitude,type,notes
Independence Mine,61.7817,-149.2819,historic_producer,Major producer
Lucky Shot Mine,61.7750,-149.2889,historic_producer,High grade
Fern Mine,61.7700,-149.2750,historic_producer,
Gold Cord Mine,61.7850,-149.2950,prospect,
```

**Required columns**:
- `name` - Mine name
- `latitude` - Latitude in decimal degrees
- `longitude` - Longitude in decimal degrees
- `type` - One of: `historic_producer`, `prospect`, `occurrence`

**Optional columns**:
- `notes` - Any additional information
- `production` - Historical production if known
- `grade` - Grade information

**Save to**: `validation/user_mines_matsu.csv`

### 3. Alaska Geological Data (Optional but Recommended)

**Alaska Division of Geological & Geophysical Surveys (DGGS)**

Website: https://dggs.alaska.gov/

Recommended datasets:
- Geologic maps for Willow Creek district
- Aeromagnetic surveys (if available)
- Known fault/structural data

**Note**: These are optional but can improve accuracy. The system will work without them.

## Data Directory Structure

After downloading all data, your directory should look like:

```
General/
├── data/
│   ├── usgs_mrds_alaska_gold.csv        # MRDS data
│   └── alaska_geology/                   # Optional geological data
├── validation/
│   └── user_mines_matsu.csv             # YOUR mine locations
├── outputs/
│   ├── maps/                             # Generated HTML maps
│   └── exports/                          # CSV/KML exports
└── [Python scripts...]
```

## Verification

After downloading data, verify with:

```python
import pandas as pd

# Check MRDS data
mrds = pd.read_csv("data/usgs_mrds_alaska_gold.csv")
print(f"MRDS records: {len(mrds)}")
print(f"Columns: {list(mrds.columns)}")

# Check user mines
mines = pd.read_csv("validation/user_mines_matsu.csv")
print(f"User mines: {len(mines)}")
print(f"Columns: {list(mines.columns)}")
```

Expected output:
- MRDS: 200-500 records (Alaska lode gold occurrences)
- User mines: 5-20 records (your known mines)

## Next Steps

Once data is acquired:

1. Run `python config.py` to verify configuration
2. Run the main analysis script: `python matsu_gold_detection_v2.py`
3. The system will automatically use your validation data

## Troubleshooting

**Problem**: USGS MRDS download link doesn't work

**Solution**:
- Try the manual download option from the website
- Or use the legacy download: https://mrdata.usgs.gov/metadata/mrds.faq.html

**Problem**: Don't have user mine CSV yet

**Solution**:
- The system has default known mines in `config.py`
- You can start with those and add your data later
- To use defaults, don't provide a CSV path to the validation framework

**Problem**: Columns don't match expected format

**Solution**:
```python
import pandas as pd

# Load your data
df = pd.read_csv("your_data.csv")

# Rename columns to match expected format
df.rename(columns={
    'LAT': 'latitude',
    'LON': 'longitude',
    'NAME': 'name',
    'TYPE': 'type'
}, inplace=True)

# Save
df.to_csv("validation/user_mines_matsu.csv", index=False)
```

## Data Privacy Note

Your user mine validation data (`validation/user_mines_matsu.csv`) is:
- Stored locally only
- NOT uploaded to Earth Engine
- NOT shared externally
- Used only for model validation

The USGS MRDS data is public domain.
