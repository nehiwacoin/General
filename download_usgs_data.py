#!/usr/bin/env python3
"""
Download and prepare USGS MRDS data for Mat-Su gold prospectivity analysis.
"""

import pandas as pd
import urllib.request
import zipfile
from pathlib import Path
import sys

def download_usgs_mrds():
    """Download and filter USGS MRDS data for Alaska lode gold."""

    print("=" * 70)
    print("USGS MRDS DATA DOWNLOAD")
    print("=" * 70)

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # URLs - USGS MRDS
    # Note: These URLs may change. If download fails, visit https://mrdata.usgs.gov/mrds/
    urls = {
        'mrds_csv': 'https://mrdata.usgs.gov/mrds/mrds-csv.zip',
        'mrds_kml': 'https://mrdata.usgs.gov/services/mrds?request=getcapabilities&service=WMS&version=1.1.1&',
    }

    print("\nStep 1: Downloading USGS MRDS database...")
    print("This may take several minutes (large file)...")

    zip_path = data_dir / "mrds.zip"

    try:
        urllib.request.urlretrieve(urls['mrds_csv'], zip_path)
        print(f"✓ Downloaded to: {zip_path}")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://mrdata.usgs.gov/mrds/")
        print("2. Click 'Download Data'")
        print("3. Save as: data/mrds.zip")
        print("4. Re-run this script")
        return False

    # Unzip
    print("\nStep 2: Extracting data...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("✓ Extracted")
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return False

    # Find the CSV file (name may vary)
    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        print("✗ No CSV file found in extracted data")
        return False

    mrds_csv = csv_files[0]
    print(f"✓ Found: {mrds_csv}")

    # Load data
    print("\nStep 3: Loading and filtering data...")
    try:
        df = pd.read_csv(mrds_csv, low_memory=False, encoding='latin1')
        print(f"✓ Loaded {len(df):,} total records")
    except Exception as e:
        print(f"✗ Failed to load CSV: {e}")
        return False

    # Display columns
    print(f"\nAvailable columns: {list(df.columns)[:10]}...")

    # Filter for Alaska
    print("\nStep 4: Filtering for Alaska...")

    # Find state column (name varies)
    state_col = None
    for col in ['state', 'State', 'STATE', 'st', 'ST']:
        if col in df.columns:
            state_col = col
            break

    if state_col is None:
        print("⚠ Warning: Could not find state column")
        alaska_df = df
    else:
        alaska_df = df[df[state_col] == 'AK']
        print(f"✓ Alaska records: {len(alaska_df):,}")

    # Filter for gold
    print("\nStep 5: Filtering for gold deposits...")

    # Find commodity columns
    commod_cols = [col for col in alaska_df.columns if 'commod' in col.lower()]

    if commod_cols:
        gold_mask = pd.Series([False] * len(alaska_df))
        for col in commod_cols:
            gold_mask |= alaska_df[col].astype(str).str.contains('Au', case=False, na=False)

        alaska_gold = alaska_df[gold_mask]
        print(f"✓ Gold deposits: {len(alaska_gold):,}")
    else:
        print("⚠ Warning: Could not find commodity columns, keeping all records")
        alaska_gold = alaska_df

    # Filter out placer deposits (we want lode only)
    print("\nStep 6: Filtering out placer deposits (keeping lode only)...")

    dep_type_col = None
    for col in ['dep_type', 'Dep_Type', 'DEP_TYPE', 'deposit_type']:
        if col in alaska_gold.columns:
            dep_type_col = col
            break

    if dep_type_col:
        # Exclude placer
        lode_mask = ~alaska_gold[dep_type_col].astype(str).str.contains('placer', case=False, na=False)
        alaska_lode_gold = alaska_gold[lode_mask]
        print(f"✓ Lode gold deposits: {len(alaska_lode_gold):,}")
    else:
        print("⚠ Warning: Could not find deposit type column, keeping all")
        alaska_lode_gold = alaska_gold

    # Filter for Mat-Su area (approximately)
    print("\nStep 7: Filtering for Mat-Su region...")

    # Find lat/lon columns
    lat_col, lon_col = None, None
    for col in alaska_lode_gold.columns:
        col_lower = col.lower()
        if 'lat' in col_lower and lat_col is None:
            lat_col = col
        if 'lon' in col_lower and lon_col is None:
            lon_col = col

    if lat_col and lon_col:
        # Mat-Su bounding box
        matsu_mask = (
            (alaska_lode_gold[lat_col] >= 61.5) &
            (alaska_lode_gold[lat_col] <= 62.5) &
            (alaska_lode_gold[lon_col] >= -150.5) &
            (alaska_lode_gold[lon_col] <= -148.0)
        )
        matsu_gold = alaska_lode_gold[matsu_mask]
        print(f"✓ Mat-Su area lode gold: {len(matsu_gold):,}")

        # Save Mat-Su specific dataset
        matsu_output = data_dir / "usgs_mrds_matsu_gold.csv"
        matsu_gold.to_csv(matsu_output, index=False)
        print(f"✓ Saved Mat-Su data: {matsu_output}")
    else:
        print("⚠ Warning: Could not find lat/lon columns for regional filtering")
        matsu_gold = alaska_lode_gold

    # Save full Alaska lode gold dataset
    output_path = data_dir / "usgs_mrds_alaska_gold.csv"
    alaska_lode_gold.to_csv(output_path, index=False)
    print(f"\n✓ Saved Alaska lode gold data: {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"\nFiles created:")
    print(f"  1. {output_path}")
    print(f"     - {len(alaska_lode_gold):,} Alaska lode gold deposits")
    if lat_col and lon_col:
        print(f"  2. {data_dir / 'usgs_mrds_matsu_gold.csv'}")
        print(f"     - {len(matsu_gold):,} Mat-Su area deposits")

    print("\nData preview:")
    print(alaska_lode_gold.head())

    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Review the data files in the 'data/' directory")
    print("2. Prepare your user mine validation CSV (see DATA_ACQUISITION_GUIDE.md)")
    print("3. Run: python matsu_gold_detection_v2.py")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = download_usgs_mrds()
    sys.exit(0 if success else 1)
