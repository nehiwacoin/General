"""
Configuration file for Mat-Su Gold Prospectivity Mapping System
================================================================
All project settings, study area definitions, and parameters.
"""

import ee
from pathlib import Path

# ============================================================================
# PROJECT METADATA
# ============================================================================
PROJECT_NAME = "MatSu_Lode_Gold_Prospectivity"
VERSION = "2.0.0"
STUDY_AREA = "Mat-Su Valley (Willow Creek District), Alaska"

# ============================================================================
# EARTH ENGINE CONFIGURATION
# ============================================================================
GEE_PROJECT_ID = "ee-markjamesp"  # User's registered GEE project

# ============================================================================
# STUDY AREA DEFINITION - MAT-SU VALLEY (WILLOW CREEK DISTRICT)
# ============================================================================

# Core Willow Creek District (tight focus around known mines)
MATSU_CORE = {
    'name': 'matsu_willow_core',
    'bounds': [-149.50, 61.70, -149.00, 62.00],  # [west, south, east, north]
    'description': 'Core Willow Creek mining district',
    'geometry': ee.Geometry.Rectangle([-149.50, 61.70, -149.00, 62.00])
}

# Extended Mat-Su Valley (broader exploration area)
MATSU_EXTENDED = {
    'name': 'matsu_valley_extended',
    'bounds': [-150.00, 61.50, -148.50, 62.20],
    'description': 'Extended Mat-Su Valley exploration area',
    'geometry': ee.Geometry.Rectangle([-150.00, 61.50, -148.50, 62.20])
}

# Default study area (use core for production)
DEFAULT_STUDY_AREA = MATSU_CORE

# ============================================================================
# DATE RANGES FOR IMAGERY
# ============================================================================

# Primary date range (recent data)
PRIMARY_DATE_RANGE = {
    'start': '2022-01-01',
    'end': '2024-12-31'
}

# Summer months (snow-free, better for Alaska)
SUMMER_MONTHS = [6, 7, 8, 9]  # June through September

# ============================================================================
# CLOUD/SNOW/QUALITY THRESHOLDS
# ============================================================================

MAX_CLOUD_COVER = 15  # percent (stricter than before)
MAX_SNOW_COVER = 10   # percent (critical for Alaska)
MIN_PIXEL_QA_SCORE = 0.7  # Quality score for pixel acceptance

# ============================================================================
# GEOLOGICAL PARAMETERS - WILLOW CREEK TYPE LODE DEPOSITS
# ============================================================================

DEPOSIT_TYPE = "Orogenic Lode Gold (Willow Creek Type)"

# Known mines for validation (user should provide CSV with actual coordinates)
KNOWN_MINES = {
    'Independence Mine': {'lat': 61.7817, 'lon': -149.2819, 'type': 'historic_producer'},
    'Lucky Shot Mine': {'lat': 61.7750, 'lon': -149.2889, 'type': 'historic_producer'},
    'Fern Mine': {'lat': 61.7700, 'lon': -149.2750, 'type': 'historic_producer'},
    'Gold Cord Mine': {'lat': 61.7850, 'lon': -149.2950, 'type': 'historic_producer'},
    'War Baby Mine': {'lat': 61.7780, 'lon': -149.2700, 'type': 'historic_producer'},
    'Mabel Mine': {'lat': 61.7720, 'lon': -149.2830, 'type': 'historic_producer'},
    'Martin Mine': {'lat': 61.7790, 'lon': -149.2880, 'type': 'prospect'},
}

# Spectral index weights (Willow-specific - emphasizes carbonate alteration)
WILLOW_INDEX_WEIGHTS = {
    'iron_oxide': 0.30,        # Gossan, limonite
    'carbonate': 0.40,         # DIAGNOSTIC for Willow deposits (ankerite/calcite)
    'sericite': 0.20,          # Phyllic alteration
    'chlorite': 0.10,          # Propylitic alteration
}

# Structural analysis parameters
STRUCTURAL_PARAMS = {
    'lineament_threshold': 0.7,        # Threshold for lineament detection
    'fault_buffer_distance': 500,      # meters - proximity buffer for faults
    'fault_intersection_buffer': 200,  # meters - buffer for fault intersections
    'structural_trend': [0, 20],       # North-trending (0-20 degrees) - Willow characteristic
}

# ============================================================================
# MACHINE LEARNING PARAMETERS
# ============================================================================

ML_CONFIG = {
    'classifier': 'RandomForest',
    'n_trees': 100,
    'max_features': None,  # sqrt of n_features
    'min_samples_leaf': 5,
    'training_sample_size': 5000,
    'validation_split': 0.3,
    'random_seed': 42,
}

# Success criteria
VALIDATION_CRITERIA = {
    'min_detection_rate': 0.80,  # 80% of known mines must be in "High" zones
    'max_false_positive_rate': 0.20,
    'target_precision': 0.75,
}

# ============================================================================
# PROSPECTIVITY MODEL WEIGHTS
# ============================================================================

PROSPECTIVITY_WEIGHTS = {
    'spectral_alteration': 0.30,   # Spectral indices (iron, carbonate, sericite, chlorite)
    'structural': 0.45,            # Fault proximity, intersections, lineaments (CRITICAL for Willow)
    'ml_probability': 0.25,        # Random Forest prediction
}

# ============================================================================
# NORMALIZATION PARAMETERS
# ============================================================================

NORMALIZATION_CONFIG = {
    'method': 'percentile',        # 'percentile' or 'z-score'
    'percentile_low': 5,           # 5th percentile
    'percentile_high': 95,         # 95th percentile
    'clip_outliers': True,
}

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Directory structure
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
VALIDATION_DIR = PROJECT_ROOT / "validation"
MAPS_DIR = OUTPUT_DIR / "maps"
EXPORTS_DIR = OUTPUT_DIR / "exports"

# Create directories if they don't exist
for directory in [DATA_DIR, OUTPUT_DIR, VALIDATION_DIR, MAPS_DIR, EXPORTS_DIR]:
    directory.mkdir(exist_ok=True)

# Output file naming
OUTPUT_PREFIX = "matsu_willow"

# Map visualization parameters
MAP_VIS_PARAMS = {
    'true_color': {
        'bands': ['SR_B4', 'SR_B3', 'SR_B2'],
        'min': 0,
        'max': 0.3,
        'gamma': 1.4
    },
    'iron_oxide': {
        'min': 0.0,
        'max': 1.0,
        'palette': ['blue', 'cyan', 'yellow', 'orange', 'red']
    },
    'carbonate': {
        'min': 0.0,
        'max': 1.0,
        'palette': ['#2c7bb6', '#abd9e9', '#ffffbf', '#fdae61', '#d7191c']
    },
    'prospectivity': {
        'min': 0.0,
        'max': 1.0,
        'palette': ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8',
                    '#ffffbf', '#fee090', '#fdae61', '#f46d43', '#d73027', '#a50026']
    },
}

# Prospectivity classification thresholds
PROSPECTIVITY_THRESHOLDS = {
    'high': 0.70,      # High prospectivity (top priority targets)
    'medium': 0.50,    # Medium prospectivity
    'low': 0.30,       # Low prospectivity
    # < 0.30 = Very low
}

# ============================================================================
# DATA SOURCES
# ============================================================================

DATA_SOURCES = {
    'landsat': {
        'collection_8': 'LANDSAT/LC08/C02/T1_L2',
        'collection_9': 'LANDSAT/LC09/C02/T1_L2',
    },
    'dem': {
        'alos': 'JAXA/ALOS/AW3D30/V3_2',  # 30m DEM (preferred for Alaska)
        'srtm': 'USGS/SRTMGL1_003',       # Backup
    },
    'aster': {
        'l1t': 'ASTER/AST_L1T_003',  # ASTER Level 1T (if needed for SWIR)
    },
    'usgs_mrds': {
        'url': 'https://mrdata.usgs.gov/mrds/',
        'local_file': DATA_DIR / 'usgs_mrds_alaska_gold.csv',
    },
}

# ============================================================================
# PROCESSING PARAMETERS
# ============================================================================

PROCESSING_CONFIG = {
    'scale': 30,                    # meters (Landsat resolution)
    'max_pixels': 1e13,             # Maximum pixels for export
    'tile_size': 256,               # Tile size for processing
    'crs': 'EPSG:3338',            # Alaska Albers (NAD83)
    'export_crs': 'EPSG:4326',     # WGS84 for compatibility
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_study_area_geometry(area_type='core'):
    """
    Get study area geometry for GEE operations.

    Args:
        area_type: 'core' or 'extended'

    Returns:
        ee.Geometry
    """
    if area_type == 'core':
        return MATSU_CORE['geometry']
    elif area_type == 'extended':
        return MATSU_EXTENDED['geometry']
    else:
        raise ValueError(f"Unknown area type: {area_type}")

def initialize_earth_engine():
    """Initialize Earth Engine with project credentials."""
    try:
        ee.Initialize(project=GEE_PROJECT_ID)
        print(f"✓ Earth Engine initialized successfully (Project: {GEE_PROJECT_ID})")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize Earth Engine: {e}")
        print("\nPlease authenticate first:")
        print("  python -c \"import ee; ee.Authenticate()\"")
        return False

def print_config_summary():
    """Print configuration summary."""
    print("=" * 70)
    print(f"{PROJECT_NAME} v{VERSION}")
    print("=" * 70)
    print(f"\nStudy Area: {STUDY_AREA}")
    print(f"Deposit Type: {DEPOSIT_TYPE}")
    print(f"\nCore Area Bounds: {MATSU_CORE['bounds']}")
    print(f"Date Range: {PRIMARY_DATE_RANGE['start']} to {PRIMARY_DATE_RANGE['end']}")
    print(f"Known Mines for Validation: {len(KNOWN_MINES)}")
    print(f"\nValidation Criteria:")
    print(f"  - Minimum Detection Rate: {VALIDATION_CRITERIA['min_detection_rate']*100}%")
    print(f"  - Target Precision: {VALIDATION_CRITERIA['target_precision']*100}%")
    print("\n" + "=" * 70)

if __name__ == "__main__":
    print_config_summary()
    initialize_earth_engine()
