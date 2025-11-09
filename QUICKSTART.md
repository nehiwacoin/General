# Quick Start Guide - Mat-Su Gold Prospectivity v2.0

## What This System Does

**Identifies potential lode gold deposits in Alaska's Mat-Su Valley using:**
- Google Earth Engine satellite imagery (Landsat 8/9)
- Spectral analysis for alteration minerals (iron oxide, carbonate, sericite)
- Structural analysis (fault detection, lineament mapping)
- Validation against known producing mines
- Automated prospectivity scoring and ranking

**Key Improvement from v1.0**: Fixes "everything is red" problem with proper normalization.

---

## Installation (5 minutes)

### Step 1: Set Up Python Environment

**Windows (Anaconda Prompt)**:
```powershell
conda create -n matsu_gold python=3.11 -y
conda activate matsu_gold
pip install earthengine-api pandas geopandas folium numpy
```

**Linux/Mac**:
```bash
conda create -n matsu_gold python=3.11 -y
conda activate matsu_gold
pip install earthengine-api pandas geopandas folium numpy
```

### Step 2: Authenticate Google Earth Engine

```bash
python -c "import ee; ee.Authenticate()"
```

This will:
1. Open your browser
2. Ask you to sign in with Google
3. Generate credentials automatically

---

## Running the Analysis (2 commands)

### Option A: Quick Test (Uses Default Mines)

```bash
python download_usgs_data.py        # Download USGS data (once)
python matsu_gold_detection_v2.py   # Run analysis
```

### Option B: With Your Own Mine Data (Recommended)

1. Create `validation/user_mines_matsu.csv`:
```csv
name,latitude,longitude,type
My Mine 1,61.7817,-149.2819,historic_producer
My Mine 2,61.7750,-149.2889,prospect
```

2. Run:
```bash
python download_usgs_data.py
python matsu_gold_detection_v2.py validation/user_mines_matsu.csv
```

---

## Viewing Results

### 1. Interactive Map (Primary Output)

```bash
# Open in browser
outputs/maps/matsu_willow_prospectivity_map_v2.html
```

**What you'll see**:
- 7 layers (toggle on/off):
  1. True Color - Visual reference
  2. Iron Oxide - Oxidized zones
  3. **Carbonate** - DIAGNOSTIC for Willow gold
  4. Sericite - Phyllic alteration
  5. Fault Proximity - Structural control
  6. Lineaments - Detected faults
  7. **GOLD PROSPECTIVITY** - Final model
- Red stars = Known mines
- Red/orange areas = High prospectivity

### 2. Target List (CSV)

```bash
outputs/exports/matsu_willow_targets.csv
```

Top 100 high-priority locations, ranked by score.

### 3. Google Earth (KML)

```bash
outputs/exports/matsu_willow_targets.kml
```

Open in Google Earth to view targets in 3D.

### 4. Validation Report

```bash
outputs/matsu_willow_report.txt
validation/matsu_willow_validation_results.csv
```

Shows how many known mines were detected in "HIGH" zones.

---

## Understanding the Output

### Color Scheme (Prospectivity Map)

| Color | Score | Meaning | Action |
|-------|-------|---------|--------|
| Red | >0.70 | Very High | **Top priority** - field investigate |
| Orange | 0.50-0.70 | High | Plan detailed survey |
| Yellow | 0.30-0.50 | Medium | Investigate if accessible |
| Green/Blue | <0.30 | Low | Low priority |

### Validation Results

**Target**: ≥80% of known mines should be in "HIGH" zones

**Example Output**:
```
Detection Rate: 85.7%
Required Rate:  80.0%

STATUS: PASSED ✓
```

If it says **FAILED ✗**, see "Troubleshooting" section in README_v2.md.

---

## File Structure (What Was Created)

```
General/
├── config.py                    # Settings (edit to tune model)
├── utils.py                     # Core functions
├── validation.py                # Validation framework
├── matsu_gold_detection_v2.py  # Main script ← RUN THIS
├── download_usgs_data.py        # Data download helper
│
├── data/
│   └── usgs_mrds_alaska_gold.csv      # Downloaded USGS data
│
├── validation/
│   └── user_mines_matsu.csv           # YOUR mines (create this)
│
└── outputs/
    ├── maps/
    │   └── matsu_willow_prospectivity_map_v2.html  ← OPEN THIS
    ├── exports/
    │   ├── matsu_willow_targets.csv
    │   └── matsu_willow_targets.kml
    └── matsu_willow_report.txt
```

---

## Tuning the Model (If Validation Fails)

Edit `config.py`:

### Adjust Spectral Weights

If mines have strong carbonate alteration, increase carbonate weight:

```python
WILLOW_INDEX_WEIGHTS = {
    'iron_oxide': 0.25,
    'carbonate': 0.45,    # Increased from 0.40
    'sericite': 0.20,
    'chlorite': 0.10,
}
```

### Adjust Prospectivity Thresholds

If detection rate is 75% (close to 80%), lower the HIGH threshold:

```python
PROSPECTIVITY_THRESHOLDS = {
    'high': 0.65,      # Lowered from 0.70
    'medium': 0.50,
    'low': 0.30,
}
```

### Adjust Model Weights

If deposits are highly fault-controlled, increase structural weight:

```python
PROSPECTIVITY_WEIGHTS = {
    'spectral_alteration': 0.25,
    'structural': 0.50,          # Increased from 0.45
    'ml_probability': 0.25,
}
```

After changes, re-run: `python matsu_gold_detection_v2.py`

---

## Common Issues

### 1. Authentication Error

**Problem**: "Failed to initialize Earth Engine"

**Fix**:
```bash
python -c "import ee; ee.Authenticate()"
python -c "import ee; ee.Initialize(project='ee-markjamesp'); print('Success!')"
```

### 2. "No module named earthengine"

**Fix**:
```bash
conda activate matsu_gold
pip install earthengine-api
```

### 3. Script Runs Forever

**Normal**: First run takes 5-15 minutes (processing satellite imagery)

**Check progress**: Look for console output showing phases 1-5

**If truly stuck**: Ctrl+C and re-run (Earth Engine has server-side caching)

### 4. No Targets Generated

**Possible causes**:
- Thresholds too strict → Lower `PROSPECTIVITY_THRESHOLDS['high']` to 0.60
- Wrong study area → Check `config.DEFAULT_STUDY_AREA`
- Poor imagery → Try different date range in `config.PRIMARY_DATE_RANGE`

### 5. Validation Shows 0% Detection

**Check**:
1. CSV format: `name,latitude,longitude,type` (header must match exactly)
2. Coordinates: Decimal degrees (e.g., 61.7817, not 61°46'54")
3. Study area: Mines must be within bounds (-149.50 to -149.00°W, 61.70 to 62.00°N)

---

## Next Steps

1. ✅ **Run the analysis** (use commands above)
2. ✅ **Open the HTML map** in browser
3. ✅ **Review top 10-20 targets** in CSV
4. ✅ **Cross-reference** with geological maps
5. ✅ **Plan field work** for most promising areas

### Advanced Usage

See `README_v2.md` for:
- Detailed explanation of all components
- Layer-by-layer interpretation guide
- Advanced configuration options
- Troubleshooting guide

See `OPTIMIZATION_PLAN.md` for:
- Full 21-day development roadmap
- Machine learning integration
- Advanced structural analysis
- Multi-element targeting

---

## Key Files Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `matsu_gold_detection_v2.py` | **Main script** | Run this to do analysis |
| `config.py` | **Settings** | Edit to tune model |
| `validation.py` | Validation framework | Auto-used by main script |
| `utils.py` | Helper functions | Auto-used by main script |
| `download_usgs_data.py` | Data download | Run once before analysis |
| `README_v2.md` | Full documentation | Read for details |
| `QUICKSTART.md` | This file | Quick reference |
| `DATA_ACQUISITION_GUIDE.md` | Data download help | If download fails |

---

## Support

**Before asking for help**:
1. Check console output for specific error messages
2. Read `README_v2.md` troubleshooting section
3. Review `config.py` to ensure settings are correct
4. Check that your CSV format matches exactly

**Most common mistake**: Wrong CSV column names
- Must be: `name,latitude,longitude,type` (all lowercase)
- Not: `Name,Latitude,Longitude,Type`
- Not: `mine_name,lat,lon,mine_type`

---

## What Makes This v2.0 Better?

| v1.0 (Basic Script) | v2.0 (Production System) |
|---------------------|--------------------------|
| Everything shows as red | Only top 5-10% is high priority |
| Generic indices | Willow-specific (carbonate emphasis) |
| No structural analysis | Multi-directional fault detection |
| No validation | Automated validation framework |
| Single monolithic script | Modular architecture |
| Hard-coded values | Configurable via config.py |
| Basic map | Professional 7-layer interactive map |
| No target ranking | Top 100 targets in CSV/KML |

---

**Time Investment**:
- Setup: 5 minutes
- Data download: 2 minutes
- Analysis: 10-15 minutes
- **Total: ~20 minutes to get results**

**Start now**: `python matsu_gold_detection_v2.py`
