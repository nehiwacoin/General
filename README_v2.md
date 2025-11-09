# Mat-Su Valley Lode Gold Prospectivity Mapping System v2.0

**Production-ready remote sensing system for identifying lode gold deposits in Alaska's Mat-Su Valley using Google Earth Engine, machine learning, and geological validation.**

## What's New in v2.0

This version addresses the critical "everything is red" problem from v1.0:

✅ **Proper Normalization**: Percentile-based normalization (5th-95th percentile) instead of min/max
✅ **Willow-Specific Indices**: Emphasizes carbonate alteration (diagnostic for Willow Creek deposits)
✅ **Structural Analysis**: Multi-directional lineament detection and fault proximity mapping
✅ **Validation Framework**: Automated testing against known mine locations
✅ **Modular Architecture**: Separate modules for config, utilities, and validation
✅ **Professional Outputs**: Interactive maps, target CSVs, KML files, and reports

## Requirements

### Software
- Python 3.9+
- Google Earth Engine account (free): https://earthengine.google.com/signup
- Anaconda or Miniconda (recommended)

### Python Packages
```bash
earthengine-api
pandas
geopandas
folium
numpy
```

## Quick Start

### 1. Installation

```bash
# Clone or navigate to repository
cd /path/to/General

# Create conda environment (Windows: use Anaconda Prompt)
conda create -n matsu_gold python=3.11 -y
conda activate matsu_gold

# Install packages
pip install earthengine-api pandas geopandas folium numpy

# Authenticate with Google Earth Engine
python -c "import ee; ee.Authenticate()"
```

### 2. Download Data

```bash
# Download USGS mineral occurrence data
python download_usgs_data.py
```

See `DATA_ACQUISITION_GUIDE.md` for detailed instructions.

### 3. Prepare Validation Data (Optional but Recommended)

Create `validation/user_mines_matsu.csv` with your known mine locations:

```csv
name,latitude,longitude,type
Independence Mine,61.7817,-149.2819,historic_producer
Lucky Shot Mine,61.7750,-149.2889,historic_producer
Fern Mine,61.7700,-149.2750,historic_producer
```

The system will use these to validate model accuracy. If you don't have your own data, it will use the default mines from `config.py`.

### 4. Run Analysis

```bash
# Using default validation mines
python matsu_gold_detection_v2.py

# Using your custom validation mines
python matsu_gold_detection_v2.py validation/user_mines_matsu.csv
```

### 5. View Results

The script will generate:
- **Interactive HTML map**: `outputs/maps/matsu_willow_prospectivity_map_v2.html`
- **High-priority targets CSV**: `outputs/exports/matsu_willow_targets.csv`
- **KML file for Google Earth**: `outputs/exports/matsu_willow_targets.kml`
- **Summary report**: `outputs/matsu_willow_report.txt`
- **Validation results**: `validation/matsu_willow_validation_results.csv`

Open the HTML map in your browser to explore the results interactively.

## Project Structure

```
General/
├── config.py                          # All configuration settings
├── utils.py                           # Core utility functions
├── validation.py                      # Validation framework
├── matsu_gold_detection_v2.py        # Main analysis script
├── download_usgs_data.py             # Data download helper
│
├── data/                              # External data
│   └── usgs_mrds_alaska_gold.csv     # USGS mineral occurrences
│
├── validation/                        # Validation data
│   ├── user_mines_matsu.csv          # Your mine locations (optional)
│   └── matsu_willow_validation_*.csv # Validation results
│
├── outputs/                           # Analysis outputs
│   ├── maps/                          # Interactive HTML maps
│   │   └── matsu_willow_prospectivity_map_v2.html
│   ├── exports/                       # Data exports
│   │   ├── matsu_willow_targets.csv
│   │   └── matsu_willow_targets.kml
│   └── matsu_willow_report.txt       # Summary report
│
├── README_v2.md                       # This file
├── DATA_ACQUISITION_GUIDE.md          # Data download instructions
├── OPTIMIZATION_PLAN.md               # Full development roadmap
└── [Legacy files from v1.0...]
```

## How It Works

### Phase 1: Data Acquisition
- **Landsat 8/9 Composite**: Summer months only (June-September) for snow-free imagery
- **Cloud/Snow Masking**: Strict quality control (max 15% cloud cover)
- **ALOS DEM**: 30m resolution digital elevation model for structural analysis

### Phase 2: Feature Engineering

#### Spectral Indices (Willow-Specific)
1. **Iron Oxide (B4/B2)**: Detects gossan and oxidized zones
2. **Carbonate (B7/B6)**: DIAGNOSTIC for Willow deposits (ankerite/calcite alteration) - **40% weight**
3. **Sericite ((B5×B7)/B6²)**: Phyllic alteration zones
4. **Chlorite (B6/B5)**: Propylitic alteration

#### Structural Analysis
1. **Multi-Directional Lineament Detection**: N-S, NE-SW, E-W, NW-SE edge detection
2. **Fault Proximity**: Distance-weighted proximity to detected faults (critical for orogenic deposits)
3. **Terrain Curvature**: Identifies ridges and valleys

#### The Normalization Fix
```python
# OLD (v1.0) - Everything shows as red
normalized = (value - min) / (max - min)

# NEW (v2.0) - Only top 5-10% shows as high
normalized = (value - p5) / (p95 - p5)
```

This percentile-based approach ensures only truly anomalous areas appear as high prospectivity.

### Phase 3: Prospectivity Modeling

**Weighted Combination**:
- Spectral alteration: 30%
- Structural features: 45% (dominant - deposits are structurally controlled)
- ML probability: 25% (planned for future - currently distributed to other components)

**Formula**:
```
Prospectivity = (Willow_Signature × 0.30) + (Structural_Score × 0.45)
```

Where:
- `Willow_Signature = Iron_Oxide×0.30 + Carbonate×0.40 + Sericite×0.20 + Chlorite×0.10`
- `Structural_Score = Fault_Proximity×0.60 + Lineaments×0.30 + Curvature×0.10`

### Phase 4: Validation

The system samples prospectivity values at known mine locations and calculates:
- **Detection Rate**: Percentage of known mines in "HIGH" zones (target: ≥80%)
- **Classification Breakdown**: Distribution across HIGH/MEDIUM/LOW/VERY_LOW

If validation fails, the system provides specific recommendations for tuning.

### Phase 5: Output Generation

- **Interactive Map**: 7 layers (true color, indices, prospectivity) with known mine markers
- **Target Extraction**: Top 100 high-prospectivity locations ranked by score
- **Export Formats**: CSV (for GIS), KML (for Google Earth)
- **Validation Report**: Detailed metrics and failed detection analysis

## Configuration

All settings are in `config.py`. Key parameters you can adjust:

```python
# Study area
DEFAULT_STUDY_AREA = MATSU_CORE  # or MATSU_EXTENDED

# Spectral weights (if validation fails, tune these)
WILLOW_INDEX_WEIGHTS = {
    'iron_oxide': 0.30,
    'carbonate': 0.40,   # DIAGNOSTIC - keep high
    'sericite': 0.20,
    'chlorite': 0.10,
}

# Prospectivity model weights
PROSPECTIVITY_WEIGHTS = {
    'spectral_alteration': 0.30,
    'structural': 0.45,
    'ml_probability': 0.25,
}

# Classification thresholds
PROSPECTIVITY_THRESHOLDS = {
    'high': 0.70,      # Values ≥ 0.70 = HIGH
    'medium': 0.50,    # 0.50-0.70 = MEDIUM
    'low': 0.30,       # 0.30-0.50 = LOW
}
```

## Interpreting Results

### Map Layers

1. **True Color**: Visual reference (Landsat RGB)
2. **Iron Oxide**: Red/orange = oxidized zones (gossan)
3. **Carbonate Alteration**: Red = strong carbonate (DIAGNOSTIC for Willow)
4. **Sericite**: Yellow/red = phyllic alteration
5. **Fault Proximity**: Red = close to faults (high structural control)
6. **Lineaments**: Shows detected faults/linear features
7. **GOLD PROSPECTIVITY**: Final integrated model (red = highest priority)

### Priority Levels

- **Very High (>0.70)**: Top priority - immediate field investigation recommended
- **High (0.50-0.70)**: High priority - plan detailed survey
- **Medium (0.30-0.50)**: Moderate interest - investigate if near access
- **Low (<0.30)**: Low priority

### Known Mine Markers (Red Stars)

These show historical producing mines used for validation:
- Independence Mine (major producer, 1930s-1940s)
- Lucky Shot Mine
- Fern Mine
- Gold Cord Mine
- War Baby Mine
- Mabel Mine
- Martin Mine (prospect)

**Goal**: ≥80% of these should fall in "HIGH" prospectivity zones.

## Validation & Quality Control

### Success Criteria

✅ **PASSED** if:
- ≥80% of known mines in HIGH zones
- Detection rate meets or exceeds target

✗ **FAILED** if:
- <80% detection rate
- Systematic bias (e.g., missing mines in specific area)

### If Validation Fails

The system will:
1. List which mines failed to be detected
2. Show their actual prospectivity scores
3. Suggest which weights to adjust

**Common fixes**:
- Increase `carbonate` weight if mines have strong carbonate alteration
- Increase `structural` weight if mines are fault-controlled
- Adjust `PROSPECTIVITY_THRESHOLDS['high']` down slightly

### Validation Report

Check `validation/matsu_willow_validation_results.csv` for:
- Each mine's prospectivity score
- Classification (HIGH/MEDIUM/LOW)
- Gap to HIGH threshold (if failed)

## Troubleshooting

### Earth Engine Authentication Errors

```bash
# Re-authenticate
python -c "import ee; ee.Authenticate()"

# Initialize with project
python -c "import ee; ee.Initialize(project='ee-markjamesp'); print('Success!')"
```

### "Everything is Red" Problem

This should be fixed in v2.0. If you still see this:
1. Check that you're using `matsu_gold_detection_v2.py` (not v1.0)
2. Verify `config.NORMALIZATION_CONFIG['method'] == 'percentile'`
3. Try adjusting `percentile_high` to 90 (more restrictive)

### No High-Priority Targets Found

Possible causes:
- Very strict thresholds - lower `PROSPECTIVITY_THRESHOLDS['high']` to 0.60
- Poor data quality for the date range - try different dates
- Vegetation masking too aggressive - check NDVI threshold

### Validation Fails

1. **Check your CSV format**: Ensure columns are `name,latitude,longitude,type`
2. **Verify coordinates**: Should be decimal degrees (e.g., 61.7817, -149.2819)
3. **Check study area**: Ensure mines are within `MATSU_CORE` bounds
4. **Review failed detections**: See `validation/matsu_willow_validation_results.csv`

### Script Runs Slowly

- Normal for first run (Earth Engine processes large imagery)
- Subsequent runs may be faster (server-side caching)
- Reduce study area to `MATSU_CORE` if using `EXTENDED`

## Next Steps

### Immediate Actions

1. **Open the HTML map** and familiarize yourself with each layer
2. **Review high-priority targets** in the CSV
3. **Cross-reference** with your geological knowledge of the area
4. **Plan field work** for top 5-10 targets

### Future Enhancements (Planned)

These are documented in `OPTIMIZATION_PLAN.md`:

1. **Machine Learning Integration**: Random Forest classifier with hyperparameter tuning
2. **ASTER Data**: Add ASTER L1T for better SWIR-based alteration mapping
3. **Advanced Structural Analysis**: Fault intersection detection, stress field modeling
4. **Multi-Element Targeting**: Incorporate pathfinder elements (As, Ag, Sb)
5. **Temporal Analysis**: Compare multiple years to detect changes
6. **Cost-Distance Analysis**: Factor in access and infrastructure

## References

### Geological Background

- **Deposit Type**: Mesothermal orogenic lode gold (Willow Creek type)
- **Host Rocks**: Cretaceous flysch (graywacke-slate)
- **Alteration**: Carbonate-sericite-pyrite ± arsenopyrite
- **Structure**: North-trending shear zones and quartz veins
- **Age**: Late Cretaceous-Early Tertiary

### Key Papers

1. Szumigala et al. (2010) - "Willow Creek and Lucky Shot gold deposits, Alaska"
2. Schmidt & Werdon (2020) - "Geologic mapping of the Willow Creek mining district"
3. Goldfarb et al. (1997) - "Orogenic gold deposits of Alaska"

### Data Sources

- **Google Earth Engine**: https://earthengine.google.com
- **USGS MRDS**: https://mrdata.usgs.gov/mrds/
- **Alaska DGGS**: https://dggs.alaska.gov/

## Support

### Getting Help

1. **Read the docs**: `OPTIMIZATION_PLAN.md`, `DATA_ACQUISITION_GUIDE.md`
2. **Check config**: Review all settings in `config.py`
3. **Examine outputs**: Look at validation reports and maps
4. **Adjust parameters**: Try tuning weights and thresholds

### Known Limitations

- **Vegetation**: Dense vegetation can mask alteration signatures (NDVI masking helps but not perfect)
- **Snow/Clouds**: Limited summer imagery in Alaska reduces temporal coverage
- **Spatial Resolution**: 30m Landsat may miss small deposits (<100m)
- **No Geophysics**: Lacks magnetic/gravity data (future enhancement)

## Version History

### v2.0.0 (2025-11-09)
- **MAJOR FIX**: Percentile-based normalization (fixes "everything is red")
- Willow-specific spectral indices with carbonate emphasis
- Multi-directional structural analysis
- Automated validation framework
- Modular architecture (config, utils, validation)
- Professional outputs (HTML, CSV, KML, reports)

### v1.0.0 (2025-11-08)
- Initial release
- Basic spectral indices
- Simple weighted overlay
- Generic approach (not Willow-specific)
- Issue: Poor normalization caused false positives

## License

This project uses public domain data (USGS, Google Earth Engine) and is provided as-is for exploration and research purposes.

## Contact

For questions about the Mat-Su gold system, consult:
- Alaska Division of Geological & Geophysical Surveys (DGGS)
- USGS Alaska Science Center
- Local geological consultants familiar with Willow Creek district

---

**Remember**: This is a prospectivity model, not a guarantee of mineralization. All high-priority targets should be verified with:
1. Detailed geological mapping
2. Geochemical sampling (rock, soil, stream sediment)
3. Geophysical surveys (magnetics, IP)
4. Drilling

Always obtain proper permits and land access before field work.
